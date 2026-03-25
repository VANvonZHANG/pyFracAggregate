import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator

class PCAFilippovGenerator(BaseGenerator):
    """Basic Particle-Cluster Aggregation (PCA) algorithm (Filippov et al., 2000).

    Adds particles one by one and avoids Monte Carlo deadlocks by gradually 
    increasing the tolerance.
    """
    def generate(self) -> Aggregate:
        agg = Aggregate(self.n_particles, self.length_unit, self.mass_unit, self.density)
        radii = self.particle_dist.sample(self.n_particles)
        
        # Calculate mass based on density and volume.
        masses = self.density * (4.0 / 3.0) * np.pi * (radii ** 3)
        
        # Add the first particle at the origin
        agg.add_particle(0.0, 0.0, 0.0, radii[0], masses[0])
        if self.n_particles == 1:
            return agg
            
        a = radii.mean() # Use mean radius as the monodisperse baseline 'a' in formulas
        
        for n in range(2, self.n_particles + 1):
            r_N = radii[n-1]
            m_N = masses[n-1]
            
            # Calculate geometric center of the first N-1 particles
            geom_center = np.mean(agg.positions, axis=0)
            
            # Calculate radius L (i.e., Gamma) of the sphere for the new particle (Eq [10])
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
                # Randomly select a direction on the sphere
                u = np.random.normal(size=3)
                norm_u = np.linalg.norm(u)
                if norm_u < 1e-8:
                    continue
                u /= norm_u
                
                candidate_pos = geom_center + L * u
                
                # Calculate Euclidean distance to existing particles
                dists = np.linalg.norm(agg.positions - candidate_pos, axis=1)
                min_allowed_dists = agg.radii + r_N - self.overlap_tolerance
                
                # Check for overlap
                if np.any(dists < min_allowed_dists):
                    continue
                    
                # Check for contact with at least one particle within tolerance
                if np.any(dists <= min_allowed_dists + tolerance):
                    agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)
                    placed = True
                    break
                
                # If not found for a long time, gradually relax the tolerance to avoid deadlocks
                if attempt > 0 and attempt % 1000 == 0:
                    tolerance += 0.05 * a
            
            # Extreme fallback (very low probability): attach directly to a random particle's surface
            if not placed:
                idx = np.random.randint(n - 1)
                ref_pos = agg.positions[idx]
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                candidate_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
                agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)
                
        return agg
