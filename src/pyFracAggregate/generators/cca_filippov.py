import numpy as np
from typing import List, Tuple
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_filippov import PCAFilippovGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration

class CCAFilippovGenerator(BaseGenerator):
    """
    Filippov 等 (2000) 基础的可调簇-簇聚集 (Cluster-Cluster Aggregation, CCA) 算法。
    """
    
    def generate(self) -> Aggregate:
        # 如果粒子数很少，直接使用 PCA 退化处理
        if self.n_particles <= 8:
            pca_gen = PCAFilippovGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist, self.overlap_tolerance
            )
            return pca_gen.generate()
            
        radii = self.particle_dist.sample(self.n_particles)
        masses = (4.0 / 3.0) * np.pi * (radii ** 3)
        
        # 1. 使用 PCA 初始化子团簇列表，每个团簇 5-8 个粒子 (这里固定选 5 个)
        cluster_list = []
        cluster_size = 5
        
        idx = 0
        while idx < self.n_particles:
            rem = self.n_particles - idx
            curr_size = cluster_size if rem >= cluster_size * 1.5 else rem
            
            sub_agg = Aggregate(curr_size)
            # 对于第一阶段，先生成单分散或取局部的几何属性
            pca_gen = PCAFilippovGenerator(
                curr_size, self.df, self.kf, self.particle_dist, self.overlap_tolerance
            )
            
            # 手动注入特定的半径和质量，复用 PCA 逻辑有点麻烦
            # 我们用现有的 PCAFilippovGenerator 生成结构，然后缩放半径使其匹配
            # 简化做法：生成一个标准的子团簇，然后把我们预设的 radius 替换进去 (或者修改 PCA 以接收 pre-sampled 数组)
            
            # 由于当前 PCAFilippov 不支持直接传入 pre-sampled 半径，我们可以动态创建一个伪 distribution
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
        # 为了兼容单分散，Filippov CCA 原文中公式 [14] 假设 a_0 等于 a。
        # 对于多分散，这里采用一种基于等效平均半径的方法。
        # a = radii.mean()
        
        while len(cluster_list) > 1:
            # 每次合并前两个
            agg1 = cluster_list.pop(0)
            agg2 = cluster_list.pop(0)
            
            merged_agg = self._merge_clusters(agg1, agg2)
            cluster_list.append(merged_agg)
            
        return cluster_list[0]
        
    def _merge_clusters(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2
        
        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)
        
        # 将 agg1 平移至原点
        pos1 = agg1.positions - com1
        
        # Filippov 2000 公式 [14] 的修正版 (参考 Skorupski Eq 4)
        # 对于多分散，用平均半径近似。后续如果有 FracVAL 会用更精确的质量加权公式
        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N
        
        term1 = (a**2 * N**2) / (N1 * N2) * (N / self.kf)**(2.0 / self.df)
        term2 = (N / N2) * (Rg1**2)
        term3 = (N / N1) * (Rg2**2)
        
        Gamma_sq = term1 - term2 - term3
        Gamma = np.sqrt(max(Gamma_sq, 0.0))
        
        # 开始尝试合并
        max_attempts = 20000
        tolerance = 1e-3 * a
        
        # 提取数据以加快运算
        r1 = agg1.radii
        r2 = agg2.radii
        
        best_candidate = None
        min_gap = float('inf')
        
        for attempt in range(max_attempts):
            # 随机在 Gamma 半径的球面上选取 agg2 的新质心
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u
            
            # agg2 也先平移至原点
            pos2_centered = agg2.positions - com2
            
            # 随机旋转 agg2
            euler_angles = np.random.uniform(0, 2*np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            
            # 平移到新质心
            candidate_pos2 = pos2_rotated + new_com2
            
            # 碰撞检测：利用广播计算所有 N1 x N2 的距离
            # pos1: (N1, 3), candidate_pos2: (N2, 3)
            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            
            # 最小允许距离矩阵
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            
            gaps = dists - min_dists
            if np.any(gaps < 0):
                # 发生重叠，失败
                continue
                
            current_min_gap = np.min(gaps)
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()
                
            # 检查是否有点接触
            if current_min_gap <= tolerance:
                # 成功合并
                merged = Aggregate(N)
                for i in range(N1):
                    merged.add_particle(pos1[i,0], pos1[i,1], pos1[i,2], agg1.radii[i], agg1.masses[i])
                for j in range(N2):
                    merged.add_particle(candidate_pos2[j,0], candidate_pos2[j,1], candidate_pos2[j,2], agg2.radii[j], agg2.masses[j])
                return merged
                
            if attempt > 0 and attempt % 2000 == 0:
                tolerance += 0.05 * a
                
        # Fallback: 返回不重叠但间隙最小的情况
        if best_candidate is None:
            # 极端情况：连一个不重叠的都没找到，只能直接用最后一次尝试 (理论上只有 Gamma 小于 r1+r2 才会发生)
            best_candidate = candidate_pos2
            
        merged = Aggregate(N)
        for i in range(N1):
            merged.add_particle(pos1[i,0], pos1[i,1], pos1[i,2], agg1.radii[i], agg1.masses[i])
        for j in range(N2):
            merged.add_particle(best_candidate[j,0], best_candidate[j,1], best_candidate[j,2], agg2.radii[j], agg2.masses[j])
        return merged
