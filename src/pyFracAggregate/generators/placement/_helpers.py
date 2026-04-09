import numpy as np
from pyFracAggregate.core.aggregate import Aggregate


def random_monte_carlo_place(
    agg: Aggregate,
    r_N: float,
    geom_center: np.ndarray,
    L: float,
    mean_radius: float,
    overlap_tolerance: float,
) -> tuple | None:
    """Random Monte Carlo placement on the Gamma sphere with tolerance relaxation."""
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
        min_allowed_dists = agg.radii + r_N - overlap_tolerance

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
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-8:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u /= norm_u
    fallback_pos = ref_pos + (agg.radii[idx] + r_N - overlap_tolerance) * u
    return (fallback_pos[0], fallback_pos[1], fallback_pos[2])


def random_monte_carlo_merge(
    pos1: np.ndarray,
    r1: np.ndarray,
    pos2_centered: np.ndarray,
    r2: np.ndarray,
    Gamma: float,
    mean_radius: float,
    overlap_tolerance: float,
    track_best: bool = False,
) -> np.ndarray | None:
    """Random rotation + collision detection for cluster merging.

    If track_best=True, returns the candidate with minimum gap (like Filippov CCA).
    If track_best=False, returns immediately on first acceptable result.
    """
    max_attempts = 20000
    tolerance = 1e-3 * mean_radius
    best_candidate = None
    min_gap = float('inf')
    candidate_pos2 = None

    for attempt in range(max_attempts):
        u = np.random.normal(size=3)
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-8:
            continue
        u /= norm_u
        new_com2 = Gamma * u

        euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
        from pyFracAggregate.core.math_utils import rotate_points
        pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
        candidate_pos2 = pos2_rotated + new_com2

        dists = np.linalg.norm(
            pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2
        )
        min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - overlap_tolerance
        gaps = dists - min_dists

        if np.any(gaps < 0):
            continue

        current_min_gap = np.min(gaps)

        if track_best:
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()

        if current_min_gap <= tolerance:
            return candidate_pos2

        if attempt > 0 and attempt % 2000 == 0:
            tolerance += 0.05 * mean_radius

    if track_best and best_candidate is not None:
        return best_candidate

    if candidate_pos2 is None:
        # Should not happen but guard against it
        return pos2_centered

    return candidate_pos2 if not track_best else best_candidate
