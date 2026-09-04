import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator


class PCAGenerator(BaseGenerator):
    """Particle-Cluster Aggregation with pluggable placement strategy."""

    def generate(self) -> Aggregate:
        agg = Aggregate(self.n_particles, self.length_unit, self.mass_unit, self.density)
        radii = self.particle_dist.sample(self.n_particles, rng=self.rng)
        masses = self.density * (4.0 / 3.0) * np.pi * (radii ** 3)

        agg.add_particle(0.0, 0.0, 0.0, radii[0], masses[0])
        if self.n_particles == 1:
            return agg

        a = np.mean(radii)

        for n in range(2, self.n_particles + 1):
            r_N = radii[n - 1]
            m_N = masses[n - 1]

            center, L = self.scaling.pca_step(agg, r_N, m_N, radii)

            pos = self.placement.place_particle(agg, r_N, m_N, center, L, a)

            if pos is not None:
                agg.add_particle(pos[0], pos[1], pos[2], r_N, m_N)
            else:
                # Extreme fallback
                idx = self.rng.integers(n - 1)
                ref_pos = agg.positions[idx]
                u = self.rng.normal(size=3)
                u /= np.linalg.norm(u)
                fallback_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
                agg.add_particle(fallback_pos[0], fallback_pos[1], fallback_pos[2], r_N, m_N)

        return agg
