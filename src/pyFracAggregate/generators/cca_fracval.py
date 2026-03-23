import numpy as np
from typing import List, Tuple
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_filippov import PCAFilippovGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration

class FracVALGenerator(BaseGenerator):
    """FracVAL polydisperse algorithm (Moran et al., 2019).

    Generates precise fractal structures by combining polydisperse mass weights 
    and hierarchical tangential calculations.
    """
    
    def generate(self) -> Aggregate:
        if self.n_particles <= 8:
            pca_gen = PCAFilippovGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist, self.overlap_tolerance
            )
            return pca_gen.generate()
            
        radii = self.particle_dist.sample(self.n_particles)
        # Actual FracVAL might consider different geometric variances; 
        # we assume unit density here.
        masses = (4.0 / 3.0) * np.pi * (radii ** 3)
        
        # 1. Pre-allocate particles to sub-clusters (approx. 0.1 N, or fixed at 5 for small N).
        # According to Moran 2019: N in [50, 500], N_sub = 0.1N; N < 50, N_sub = 5.
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
            
            # Temporarily generate a local distribution to call PCA
            class LocalDist:
                def __init__(self, r):
                    self.r = r
                def sample(self, n):
                    return self.r
                    
            local_pca = PCAFilippovGenerator(curr_size, self.df, self.kf, LocalDist(radii[idx:idx+curr_size]), self.overlap_tolerance)
            sub_agg = local_pca.generate()
            cluster_list.append(sub_agg)
            idx += curr_size
            
        # 2. Hierarchical merging
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
        
        # r_p,geo uses the geometric mean of all particles or a similar metric.
        # For simplicity, we use volume averaging or the mean of all particles.
        # Here we approximate r_p,geo using the mean of all merged particles.
        r_p_geo = np.mean(np.concatenate([agg1.radii, agg2.radii]))
        
        # Eq 3 & Eq 6 from Moran 2019
        # m^2 R_g^2 = m(m1 R_g1^2 + m2 R_g2^2) + Gamma^2 m1 m2
        # where R_g = r_p_geo * ( (m/mean_m) / kf )^(1/Df) ?
        # Actually, the paper states: R_g = r_p_geo * (n / kf)^(1/Df) where n is particle count.
        Rg = r_p_geo * (N / self.kf)**(1.0 / self.df)
        
        term_target = m**2 * Rg**2
        term_parts = m * (m1 * Rg1**2 + m2 * Rg2**2)
        
        Gamma_sq = (term_target - term_parts) / (m1 * m2)
        Gamma = np.sqrt(max(Gamma_sq, 0.0))
        
        # FracVAL Phase: We have target distance Gamma between CoM1 and CoM2.
        # For fast generation, we use optimized Monte Carlo with random placement
        # of agg2's CoM and random rotation, introducing FLAGE-style fast touching calculations.
        # Since both are clusters, solving all degrees of freedom analytically is complex.
        # We use high iteration counts and step size decay to ensure point contact at Gamma distance.
        
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
