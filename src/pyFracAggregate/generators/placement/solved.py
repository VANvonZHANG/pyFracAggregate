import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.placement.base import PlacementStrategy
from pyFracAggregate.generators.placement.solvers import (
    build_particle_list_pca,
    filter_overlapping_candidates,
    mc_touch_merge,
    mc_touch_place,
    solve_tangency,
)


class SolvedPlacement(PlacementStrategy):
    """Emergent contact via closed-form tangency solving (Skorupski et al.,
    2014, FLAGE), with Monte Carlo fallback."""

    def __init__(self, overlap_tolerance: float = 1e-5, surface_beta: float = 0.3):
        self.overlap_tolerance = overlap_tolerance
        self.surface_beta = surface_beta

    def place_particle(
        self,
        agg: Aggregate,
        candidate_radius: float,
        candidate_mass: float,
        geom_center: np.ndarray,
        L: float,
        mean_radius: float,
    ) -> tuple | None:
        """Try algebraic placement, then fall back to random sampling."""
        candidate_list = build_particle_list_pca(agg.positions, agg.radii, L, mean_radius)

        if len(candidate_list) > 0:
            np.random.shuffle(candidate_list)
            max_ref = min(5, len(candidate_list))
            for i in range(max_ref):
                ref_idx = candidate_list[i % len(candidate_list)]
                ref_pos = agg.positions[ref_idx]
                r_ref = agg.radii[ref_idx]

                candidates = solve_tangency(
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
        return mc_touch_place(
            agg, candidate_radius, geom_center, L, mean_radius, self.overlap_tolerance
        )

    def merge_clusters(
        self,
        pos1: np.ndarray,
        r1: np.ndarray,
        agg1: Aggregate,
        pos2_centered: np.ndarray,
        r2: np.ndarray,
        agg2: Aggregate,
        Gamma: float,
        mean_radius: float,
    ) -> np.ndarray:
        """FLAGE-style merge with surface particle filtering + random fallback."""
        N1 = agg1.current_size
        N2 = agg2.current_size

        D1_max = np.max(np.linalg.norm(pos1, axis=1) + r1)
        D2_max = np.max(np.linalg.norm(pos2_centered, axis=1) + r2)

        if D1_max + D2_max < Gamma:
            return mc_touch_merge(
                pos1, r1, pos2_centered, r2, Gamma, mean_radius, self.overlap_tolerance
            )

        dists1 = np.linalg.norm(pos1, axis=1)
        la1 = dists1 + r1
        la2 = np.linalg.norm(pos2_centered, axis=1) + r2

        surface1_idx = np.where(la1 >= Gamma * self.surface_beta)[0]
        surface2_idx = np.where(la2 >= Gamma * self.surface_beta)[0]

        if len(surface1_idx) == 0:
            surface1_idx = np.arange(N1)
        if len(surface2_idx) == 0:
            surface2_idx = np.arange(N2)

        np.random.shuffle(surface1_idx)
        np.random.shuffle(surface2_idx)

        max_ref_tries = min(50, N1 * N2)
        ref_try = 0

        # Loop variables iterate the surface-particle filter order; the trial
        # body samples fresh orientations (emergent contact), it does not solve
        # for si/sj contact geometry — that is ConstructedPlacement's job.
        for _si in surface1_idx:
            for _sj in surface2_idx:
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

        return mc_touch_merge(
            pos1, r1, pos2_centered, r2, Gamma, mean_radius, self.overlap_tolerance
        )
