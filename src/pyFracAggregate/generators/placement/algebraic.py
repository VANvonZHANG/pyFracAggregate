import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.optimizer_flage import (
    build_particle_list_pca,
    find_exact_touching_points_pca,
    filter_overlapping_candidates,
)
from pyFracAggregate.generators.placement.base import PlacementStrategy


class AlgebraicPlacement(PlacementStrategy):
    """FLAGE algebraic placement (Skorupski et al., 2014).

    Uses sphere-sphere intersection to compute exact touching points,
    with random Monte Carlo as fallback.
    """

    def __init__(self, overlap_tolerance: float = 0.0):
        self.overlap_tolerance = overlap_tolerance

    def place_particle(self, agg, candidate_radius, candidate_mass, geom_center, L, mean_radius):
        """Try algebraic placement, then fall back to random sampling."""
        candidate_list = build_particle_list_pca(agg.positions, agg.radii, L, mean_radius)

        if len(candidate_list) > 0:
            np.random.shuffle(candidate_list)
            max_ref = min(5, len(candidate_list))
            for i in range(max_ref):
                ref_idx = candidate_list[i % len(candidate_list)]
                ref_pos = agg.positions[ref_idx]
                r_ref = agg.radii[ref_idx]

                candidates = find_exact_touching_points_pca(
                    geom_center, L, ref_pos, candidate_radius, r_ref, num_points=8
                )
                if len(candidates) == 0:
                    continue

                valid = filter_overlapping_candidates(
                    candidates, agg.positions, agg.radii, candidate_radius, self.overlap_tolerance
                )
                if len(valid) > 0:
                    pt = valid[np.random.randint(len(valid))]
                    return (pt[0], pt[1], pt[2])

        # Fallback: random Monte Carlo
        return self._random_place_particle(agg, candidate_radius, geom_center, L, mean_radius)

    def _random_place_particle(self, agg, r_N, geom_center, L, a):
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
                return (candidate_pos[0], candidate_pos[1], candidate_pos[2])
            if attempt > 0 and attempt % 1000 == 0:
                tolerance += 0.05 * a

        # Extreme fallback
        idx = np.random.randint(agg.current_size)
        ref_pos = agg.positions[idx]
        u = np.random.normal(size=3)
        u /= np.linalg.norm(u)
        fallback_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
        return (fallback_pos[0], fallback_pos[1], fallback_pos[2])

    def merge_clusters(self, pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius):
        """FLAGE-style merge with surface particle filtering + random fallback."""
        N1 = agg1.current_size
        N2 = agg2.current_size

        D1_max = np.max(np.linalg.norm(pos1, axis=1) + r1)
        D2_max = np.max(np.linalg.norm(pos2_centered, axis=1) + r2)

        if D1_max + D2_max < Gamma:
            return self._random_merge(pos1, r1, pos2_centered, r2, Gamma, mean_radius)

        dists1 = np.linalg.norm(pos1, axis=1)
        la1 = dists1 + r1
        la2 = np.linalg.norm(pos2_centered, axis=1) + r2

        surface1_idx = np.where(la1 >= Gamma * 0.3)[0]
        surface2_idx = np.where(la2 >= Gamma * 0.3)[0]

        if len(surface1_idx) == 0:
            surface1_idx = np.arange(N1)
        if len(surface2_idx) == 0:
            surface2_idx = np.arange(N2)

        np.random.shuffle(surface1_idx)
        np.random.shuffle(surface2_idx)

        max_ref_tries = min(50, N1 * N2)
        ref_try = 0

        for si in surface1_idx:
            for sj in surface2_idx:
                ref_try += 1
                if ref_try > max_ref_tries:
                    break

                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                new_com2 = Gamma * u

                euler = np.random.uniform(0, 2 * np.pi, size=3)
                pos2_rot = rotate_points(pos2_centered, tuple(euler))
                pos2_trial = pos2_rot + new_com2

                dists = np.linalg.norm(
                    pos1[:, np.newaxis, :] - pos2_trial[np.newaxis, :, :], axis=2
                )
                min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
                gaps = dists - min_dists

                if not np.any(gaps < 0):
                    min_gap = np.min(gaps)
                    if min_gap <= 1e-3 * mean_radius:
                        return pos2_trial

            if ref_try > max_ref_tries:
                break

        return self._random_merge(pos1, r1, pos2_centered, r2, Gamma, mean_radius)

    def _random_merge(self, pos1, r1, pos2_centered, r2, Gamma, mean_radius):
        max_attempts = 20000
        tol = 1e-3 * mean_radius
        candidate_pos2 = None

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            candidate_pos2 = pos2_rotated + new_com2

            dists = np.linalg.norm(
                pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2
            )
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue
            if np.min(gaps) <= tol:
                return candidate_pos2
            if attempt > 0 and attempt % 2000 == 0:
                tol += 0.05 * mean_radius

        if candidate_pos2 is None:
            candidate_pos2 = pos2_rotated + new_com2
        return candidate_pos2
