import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator

class PCAFilippovGenerator(BaseGenerator):
    """
    Filippov 等 (2000) 基础的 Particle-Cluster Aggregation (PCA) 算法。
    采用逐个添加粒子的方式，并通过缓慢增大的容差来避免 Monte Carlo 的死循环。
    """
    def generate(self) -> Aggregate:
        agg = Aggregate(self.n_particles)
        radii = self.particle_dist.sample(self.n_particles)
        
        # 假设密度为 1，质量与体积成正比
        masses = (4.0 / 3.0) * np.pi * (radii ** 3)
        
        # 添加第一个粒子于原点
        agg.add_particle(0.0, 0.0, 0.0, radii[0], masses[0])
        if self.n_particles == 1:
            return agg
            
        a = radii.mean() # 使用平均半径作为公式中的单分散基准 a
        
        for n in range(2, self.n_particles + 1):
            r_N = radii[n-1]
            m_N = masses[n-1]
            
            # 计算前 N-1 个粒子的几何中心
            geom_center = np.mean(agg.positions, axis=0)
            
            # 根据公式 [10] 计算新粒子所在球面半径 L (即 Γ)
            term1 = (n**2 * a**2) / (n - 1) * (n / self.kf)**(2.0 / self.df)
            term2 = (n * a**2) / (n - 1)
            term3 = n * a**2 * ((n - 1) / self.kf)**(2.0 / self.df)
            L_sq = term1 - term2 - term3
            
            if L_sq < 0:
                L = r_N
            else:
                L = np.sqrt(L_sq)
                
            placed = False
            max_attempts = 10000
            tolerance = 1e-3 * a
            
            for attempt in range(max_attempts):
                # 在球面上随机选取一个方向
                u = np.random.normal(size=3)
                norm_u = np.linalg.norm(u)
                if norm_u < 1e-8:
                    continue
                u /= norm_u
                
                candidate_pos = geom_center + L * u
                
                # 计算到已存在粒子的欧式距离
                dists = np.linalg.norm(agg.positions - candidate_pos, axis=1)
                min_allowed_dists = agg.radii + r_N - self.overlap_tolerance
                
                # 检查是否重叠
                if np.any(dists < min_allowed_dists):
                    continue
                    
                # 检查是否与至少一个粒子接触 (满足容差范围)
                if np.any(dists <= min_allowed_dists + tolerance):
                    agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)
                    placed = True
                    break
                
                # 如果长时间找不到，逐步放宽相切容差以防死循环
                if attempt > 0 and attempt % 1000 == 0:
                    tolerance += 0.05 * a
            
            # 极限 fallback (极小概率发生): 直接随机贴在某一个粒子表面
            if not placed:
                idx = np.random.randint(n - 1)
                ref_pos = agg.positions[idx]
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                candidate_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
                agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)
                
        return agg
