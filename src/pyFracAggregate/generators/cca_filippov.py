import numpy as np
from typing import List, Tuple
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_filippov import PCAFilippovGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration

class CCAFilippovGenerator(BaseGenerator):
    """Basic tunable Cluster-Cluster Aggregation (CCA) algorithm (Filippov et al., 2000).
    """
    
    def generate(self) -> Aggregate:
        # If particle count is small, fallback to PCA processing
        if self.n_particles <= 8:
            pca_gen = PCAFilippovGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist, self.overlap_tolerance
            )
            return pca_gen.generate()
            
        radii = self.particle_dist.sample(self.n_particles)
        masses = (4.0 / 3.0) * np.pi * (radii ** 3)
        
        # 1. Initialize sub-cluster list using PCA, each with 5-8 particles (fixed at 5 here)
        cluster_list = []
        cluster_size = 5
        
        idx = 0
        while idx < self.n_particles:
            rem = self.n_particles - idx
            curr_size = cluster_size if rem >= cluster_size * 1.5 else rem
            
            sub_agg = Aggregate(curr_size)
            # For the first phase, use monodisperse or local geometric properties
            pca_gen = PCAFilippovGenerator(
                curr_size, self.df, self.kf, self.particle_dist, self.overlap_tolerance
            )
            
            # Manually inject specific radii and masses; reusing PCA logic is slightly complex.
            # We generate structure with PCAFilippovGenerator, then scale radii to match.
            # Simplified: generate standard sub-cluster, then replace with pre-sampled radii.
            
            # Since current PCAFilippov doesn't support direct pre-sampled radii injection,
            # we dynamically create a pseudo-distribution.
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
        # For monodisperse compatibility, Filippov CCA (2000) Eq [14] assumes a_0 = a.
        # For polydisperse, an equivalent mean radius method is used here.
        # a = radii.mean()
        
        while len(cluster_list) > 1:
            # Merge the first two each time
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
        
        # Translate agg1 to origin
        pos1 = agg1.positions - com1
        
        # Modified Filippov 2000 Eq [14] (referencing Skorupski Eq 4)
        # For polydisperse, approximate with mean radius. FracVAL uses more precise 
        # mass-weighted formulas.
        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N
        
        term1 = (a**2 * N**2) / (N1 * N2) * (N / self.kf)**(2.0 / self.df)
        term2 = (N / N2) * (Rg1**2)
        term3 = (N / N1) * (Rg2**2)
        
        Gamma_sq = term1 - term2 - term3
        Gamma = np.sqrt(max(Gamma_sq, 0.0))
        
        # Attempt merging
        max_attempts = 20000
        tolerance = 1e-3 * a
        
        # Extract data for faster computation
        r1 = agg1.radii
        r2 = agg2.radii
        
        best_candidate = None
        min_gap = float('inf')
        
        for attempt in range(max_attempts):
            # Randomly select new CoM for agg2 on the sphere of radius Gamma
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u
            
            # Translate agg2 to origin
            pos2_centered = agg2.positions - com2
            
            # Randomly rotate agg2
            euler_angles = np.random.uniform(0, 2*np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            
            # Translate to new CoM
            candidate_pos2 = pos2_rotated + new_com2
            
            # Collision detection: use broadcasting for all N1 x N2 distances
            # pos1: (N1, 3), candidate_pos2: (N2, 3)
            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            
            # Minimum allowed distance matrix
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            
            gaps = dists - min_dists
            if np.any(gaps < 0):
                # Overlap occurred, failure
                continue
                
            current_min_gap = np.min(gaps)
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()
                
            # Check for point contact within tolerance
            if current_min_gap <= tolerance:
                # Successful merge
                merged = Aggregate(N)
                for i in range(N1):
                    merged.add_particle(pos1[i,0], pos1[i,1], pos1[i,2], agg1.radii[i], agg1.masses[i])
                for j in range(N2):
                    merged.add_particle(candidate_pos2[j,0], candidate_pos2[j,1], candidate_pos2[j,2], agg2.radii[j], agg2.masses[j])
                return merged
                
            if attempt > 0 and attempt % 2000 == 0:
                tolerance += 0.05 * a
                
        # Fallback: return situation with minimal gap without overlap
        if best_candidate is None:
            # Extreme case: no non-overlapping position found, fallback to last attempt 
            # (theoretically only happens if Gamma < r1 + r2)
            best_candidate = candidate_pos2
            
        merged = Aggregate(N)
        for i in range(N1):
            merged.add_particle(pos1[i,0], pos1[i,1], pos1[i,2], agg1.radii[i], agg1.masses[i])
        for j in range(N2):
            merged.add_particle(best_candidate[j,0], best_candidate[j,1], best_candidate[j,2], agg2.radii[j], agg2.masses[j])
        return merged
