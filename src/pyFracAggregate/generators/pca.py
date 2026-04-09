import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator


class PCAGenerator(BaseGenerator):
    """Particle-Cluster Aggregation with pluggable placement strategy."""

    def generate(self) -> Aggregate:
        agg = Aggregate(self.n_particles, self.length_unit, self.mass_unit, self.density)
        radii = self.particle_dist.sample(self.n_particles)
        masses = self.density * (4.0 / 3.0) * np.pi * (radii ** 3)

        agg.add_particle(0.0, 0.0, 0.0, radii[0], masses[0])
        if self.n_particles == 1:
            return agg

        a = np.mean(radii)

        for n in range(2, self.n_particles + 1):
            r_N = radii[n - 1]
            m_N = masses[n - 1]

            geom_center = np.mean(agg.positions, axis=0)

            # Filippov Eq [10]
            term1 = (n**2 * a**2) / (n - 1) * (n / self.kf) ** (2.0 / self.df)
            term2 = (n * a**2) / (n - 1)
            term3 = n * a**2 * ((n - 1) / self.kf) ** (2.0 / self.df)
            L_sq = term1 - term2 - term3
            L = np.sqrt(max(L_sq, r_N**2))

            pos = self.placement.place_particle(agg, r_N, m_N, geom_center, L, a)

            if pos is not None:
                agg.add_particle(pos[0], pos[1], pos[2], r_N, m_N)
            else:
                # Extreme fallback
                idx = np.random.randint(n - 1)
                ref_pos = agg.positions[idx]
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                fallback_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
                agg.add_particle(fallback_pos[0], fallback_pos[1], fallback_pos[2], r_N, m_N)

        return agg
