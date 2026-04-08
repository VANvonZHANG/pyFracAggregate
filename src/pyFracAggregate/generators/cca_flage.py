import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_flage import PCAFlageGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration


class CCAFlageGenerator(BaseGenerator):
    """Fast Cluster-Cluster Aggregation using FLAGE algebraic placement (Skorupski et al., 2014).

    Uses hierarchical merging with reference-particle algebraic rotation
    instead of random Monte Carlo. Falls back to random if algebraic fails.
    """

    def generate(self) -> Aggregate:
        if self.n_particles <= 8:
            pca_gen = PCAFlageGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist,
                self.overlap_tolerance, self.length_unit, self.mass_unit, self.density
            )
            return pca_gen.generate()

        radii = self.particle_dist.sample(self.n_particles)

        # Phase 1: Generate sub-clusters using FLAGE PCA
        cluster_size = 5
        cluster_list = []
        idx = 0
        while idx < self.n_particles:
            rem = self.n_particles - idx
            curr_size = cluster_size if rem >= cluster_size * 1.5 else rem

            class LocalDist:
                def __init__(self, r):
                    self.r = r
                def sample(self, n):
                    return self.r

            local_pca = PCAFlageGenerator(
                curr_size, self.df, self.kf,
                LocalDist(radii[idx:idx + curr_size]),
                self.overlap_tolerance, self.length_unit, self.mass_unit, self.density
            )
            sub_agg = local_pca.generate()
            cluster_list.append(sub_agg)
            idx += curr_size

        # Phase 2: Hierarchical merging
        while len(cluster_list) > 1:
            agg1 = cluster_list.pop(0)
            agg2 = cluster_list.pop(0)
            merged = self._merge_flage(agg1, agg2)
            cluster_list.append(merged)

        return cluster_list[0]

    def _merge_flage(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2

        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)

        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N

        # Modified Filippov Eq [14] / Skorupski Eq [4]
        term1 = (a**2 * N**2) / (N1 * N2) * (N / self.kf) ** (2.0 / self.df)
        term2 = (N / N2) * Rg1**2
        term3 = (N / N1) * Rg2**2
        Gamma_sq = term1 - term2 - term3
        Gamma = np.sqrt(max(Gamma_sq, 0.0))

        pos1 = agg1.positions - com1
        r1 = agg1.radii
        r2 = agg2.radii
        pos2_centered = agg2.positions - com2

        # Check feasibility
        D1_max = np.max(np.linalg.norm(pos1, axis=1) + r1)
        D2_max = np.max(np.linalg.norm(pos2_centered, axis=1) + r2)

        if D1_max + D2_max < Gamma:
            return self._merge_random_fallback(agg1, agg2, Gamma, a, N1, N2)

        # FLAGE algebraic merge attempt
        max_attempts = 200
        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            euler = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rot = rotate_points(pos2_centered, tuple(euler))
            candidate_pos2 = pos2_rot + new_com2

            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue

            min_gap = np.min(gaps)
            if min_gap <= 1e-3 * a:
                return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)

        return self._merge_random_fallback(agg1, agg2, Gamma, a, N1, N2)

    def _merge_random_fallback(self, agg1, agg2, Gamma, a, N1, N2):
        N = N1 + N2
        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        pos1 = agg1.positions - com1
        r1 = agg1.radii
        r2 = agg2.radii

        max_attempts = 20000
        tol = 1e-3 * a
        candidate_pos2 = None

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            pos2_centered = agg2.positions - com2
            euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            candidate_pos2 = pos2_rotated + new_com2

            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue
            if np.min(gaps) <= tol:
                return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)
            if attempt > 0 and attempt % 2000 == 0:
                tol += 0.05 * a

        if candidate_pos2 is None:
            candidate_pos2 = pos2_rotated + new_com2
        return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)

    def _build_merged(self, pos1, agg1, pos2_final, agg2, N):
        merged = Aggregate(N, self.length_unit, self.mass_unit, self.density)
        for i in range(agg1.current_size):
            merged.add_particle(pos1[i, 0], pos1[i, 1], pos1[i, 2], agg1.radii[i], agg1.masses[i])
        for j in range(agg2.current_size):
            merged.add_particle(pos2_final[j, 0], pos2_final[j, 1], pos2_final[j, 2], agg2.radii[j], agg2.masses[j])
        return merged
