import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.optimizer_flage import (
    build_particle_list_pca,
    find_exact_touching_points_pca,
    filter_overlapping_candidates,
)


class PCAFlageGenerator(BaseGenerator):
    """Fast Particle-Cluster Aggregation using FLAGE algebraic placement (Skorupski et al., 2014).

    Instead of random Monte Carlo sampling on the placement sphere, uses analytical
    sphere-sphere intersection to compute exact touching points, then filters for overlaps.
    Falls back to random sampling if algebraic method fails.
    """

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

            placed = False

            # --- FLAGE algebraic path ---
            candidate_list = build_particle_list_pca(agg.positions, agg.radii, L, a)

            if len(candidate_list) > 0:
                np.random.shuffle(candidate_list)
                ref_changes = 0
                while ref_changes < min(5, len(candidate_list)):
                    ref_idx = candidate_list[ref_changes % len(candidate_list)]
                    ref_pos = agg.positions[ref_idx]
                    r_ref = agg.radii[ref_idx]

                    candidates = find_exact_touching_points_pca(
                        geom_center, L, ref_pos, r_N, r_ref, num_points=8
                    )
                    if len(candidates) == 0:
                        ref_changes += 1
                        continue

                    valid = filter_overlapping_candidates(
                        candidates, agg.positions, agg.radii, r_N, self.overlap_tolerance
                    )
                    if len(valid) > 0:
                        pt = valid[np.random.randint(len(valid))]
                        agg.add_particle(pt[0], pt[1], pt[2], r_N, m_N)
                        placed = True
                        break
                    ref_changes += 1

            # --- Fallback: random Monte Carlo ---
            if not placed:
                max_attempts = 10000
                tolerance = 1e-3 * a
                for attempt in range(max_attempts):
                    u = np.random.normal(size=3)
                    norm_u = np.linalg.norm(u)
                    if norm_u < 1e-8:
                        continue
                    u /= norm_u

                    candidate_pos = geom_center + L * u
                    dists = np.linalg.norm(agg.positions - candidate_pos, axis=1)
                    min_allowed = agg.radii + r_N - self.overlap_tolerance

                    if np.any(dists < min_allowed):
                        continue
                    if np.any(dists <= min_allowed + tolerance):
                        agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)
                        placed = True
                        break
                    if attempt > 0 and attempt % 1000 == 0:
                        tolerance += 0.05 * a

            # --- Extreme fallback ---
            if not placed:
                idx = np.random.randint(n - 1)
                ref_pos = agg.positions[idx]
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                candidate_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
                agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)

        return agg
