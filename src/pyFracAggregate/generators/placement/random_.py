import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.placement.base import PlacementStrategy


class RandomPlacement(PlacementStrategy):
    """Random Monte Carlo placement (Filippov et al., 2000).

    Samples positions on the Gamma sphere with gradual tolerance relaxation.
    """

    def __init__(self, overlap_tolerance: float = 0.0):
        self.overlap_tolerance = overlap_tolerance

    def place_particle(self, agg, candidate_radius, candidate_mass, geom_center, L, mean_radius):
        r_N = candidate_radius
        max_attempts = 10000
        tolerance = 1e-3 * mean_radius

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-8:
                continue
            u /= norm_u

            candidate_pos = geom_center + L * u
            dists = np.linalg.norm(agg.positions - candidate_pos, axis=1)
            min_allowed_dists = agg.radii + r_N - self.overlap_tolerance

            if np.any(dists < min_allowed_dists):
                continue

            if np.any(dists <= min_allowed_dists + tolerance):
                return (candidate_pos[0], candidate_pos[1], candidate_pos[2])

            if attempt > 0 and attempt % 1000 == 0:
                tolerance += 0.05 * mean_radius

        # Extreme fallback
        idx = np.random.randint(agg.current_size)
        ref_pos = agg.positions[idx]
        u = np.random.normal(size=3)
        u /= np.linalg.norm(u)
        fallback_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
        return (fallback_pos[0], fallback_pos[1], fallback_pos[2])

    def merge_clusters(self, pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius):
        N1 = agg1.current_size
        N2 = agg2.current_size
        max_attempts = 20000
        tolerance = 1e-3 * mean_radius

        best_candidate = None
        min_gap = float('inf')
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

            current_min_gap = np.min(gaps)
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()

            if current_min_gap <= tolerance:
                return candidate_pos2

            if attempt > 0 and attempt % 2000 == 0:
                tolerance += 0.05 * mean_radius

        if best_candidate is None:
            best_candidate = candidate_pos2

        return best_candidate
