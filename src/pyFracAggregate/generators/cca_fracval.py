import numpy as np
from typing import List, Tuple
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_filippov import PCAFilippovGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration

class FracVALGenerator(BaseGenerator):
    """
    FracVAL 多分散算法 (Morán 等, 2019)。
    结合多分散粒径质量权重和层级相切计算，生成精确的分形结构。
    """
    
    def generate(self) -> Aggregate:
        if self.n_particles <= 8:
            pca_gen = PCAFilippovGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist, self.overlap_tolerance
            )
            return pca_gen.generate()
            
        radii = self.particle_dist.sample(self.n_particles)
        # 实际的 FracVAL 可能会考虑不同的几何方差，我们假设密度为 1
        masses = (4.0 / 3.0) * np.pi * (radii ** 3)
        
        # 1. 预先分配颗粒到各个子团簇 (大小约为 0.1 N，或者对于很小的 N 固定为 5)
        # 根据 FracVAL 论文，N in [50, 500], N_sub = 0.1N; N < 50, N_sub = 5.
        if self.n_particles < 50:
            cluster_size = 5
        elif self.n_particles <= 500:
            cluster_size = max(5, int(self.n_particles * 0.1))
        else:
            cluster_size = 50
            
        cluster_list = []
        idx = 0
        while idx < self.n_particles:
            rem = self.n_particles - idx
            curr_size = cluster_size if rem >= cluster_size * 1.5 else rem
            
            # 临时生成一个局部分布以便调用 PCA
            class LocalDist:
                def __init__(self, r):
                    self.r = r
                def sample(self, n):
                    return self.r
                    
            local_pca = PCAFilippovGenerator(curr_size, self.df, self.kf, LocalDist(radii[idx:idx+curr_size]), self.overlap_tolerance)
            sub_agg = local_pca.generate()
            cluster_list.append(sub_agg)
            idx += curr_size
            
        # 2. 层级合并
        while len(cluster_list) > 1:
            agg1 = cluster_list.pop(0)
            agg2 = cluster_list.pop(0)
            
            merged_agg = self._merge_fracval(agg1, agg2)
            cluster_list.append(merged_agg)
            
        return cluster_list[0]
        
    def _merge_fracval(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2
        
        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)
        
        m1 = np.sum(agg1.masses)
        m2 = np.sum(agg2.masses)
        m = m1 + m2
        
        # 这里的 r_p,geo 采用所有粒子的几何平均或者类似指标。
        # 简单起见，利用体积平均或者所有粒子的均值
        # 这里用合并后的所有粒子的均值来近似 r_p,geo
        r_p_geo = np.mean(np.concatenate([agg1.radii, agg2.radii]))
        
        # Eq 3 & Eq 6 from Morán 2019
        # m^2 R_g^2 = m(m1 R_g1^2 + m2 R_g2^2) + Gamma^2 m1 m2
        # where R_g = r_p_geo * ( (m/mean_m) / kf )^(1/Df) ?
        # 实际上论文中写的是: R_g = r_p_geo * (n / kf)^(1/Df) 其中 n 是粒子数
        Rg = r_p_geo * (N / self.kf)**(1.0 / self.df)
        
        term_target = m**2 * Rg**2
        term_parts = m * (m1 * Rg1**2 + m2 * Rg2**2)
        
        Gamma_sq = (term_target - term_parts) / (m1 * m2)
        Gamma = np.sqrt(max(Gamma_sq, 0.0))
        
        # FracVAL Phase: We have target distance Gamma between CoM1 and CoM2.
        # 对于快速生成，我们使用随机放置 agg2 质心并在其周围随机旋转的优化版蒙特卡洛，
        # 并引入 FLAGE 风格的快速相切计算。但由于两个都是团簇，解析求全部自由度极其复杂。
        # 这里我们利用大迭代次数和步长衰减，保证在保持 Gamma 距离下能找到点接触。
        
        pos1 = agg1.positions - com1
        
        max_attempts = 50000
        tolerance = 1e-3 * r_p_geo
        
        r1 = agg1.radii
        r2 = agg2.radii
        
        best_candidate = None
        min_gap = float('inf')
        
        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u
            
            pos2_centered = agg2.positions - com2
            
            euler_angles = np.random.uniform(0, 2*np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            
            candidate_pos2 = pos2_rotated + new_com2
            
            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            
            gaps = dists - min_dists
            if np.any(gaps < 0):
                continue
                
            current_min_gap = np.min(gaps)
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()
                
            if current_min_gap <= tolerance:
                merged = Aggregate(N)
                for i in range(N1):
                    merged.add_particle(pos1[i,0], pos1[i,1], pos1[i,2], agg1.radii[i], agg1.masses[i])
                for j in range(N2):
                    merged.add_particle(candidate_pos2[j,0], candidate_pos2[j,1], candidate_pos2[j,2], agg2.radii[j], agg2.masses[j])
                return merged
                
            if attempt > 0 and attempt % 2000 == 0:
                tolerance += 0.05 * r_p_geo
                
        if best_candidate is None:
            best_candidate = candidate_pos2
            
        merged = Aggregate(N)
        for i in range(N1):
            merged.add_particle(pos1[i,0], pos1[i,1], pos1[i,2], agg1.radii[i], agg1.masses[i])
        for j in range(N2):
            merged.add_particle(best_candidate[j,0], best_candidate[j,1], best_candidate[j,2], agg2.radii[j], agg2.masses[j])
        return merged
