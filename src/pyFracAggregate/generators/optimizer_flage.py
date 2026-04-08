import numpy as np
from typing import Tuple, List, Optional

def find_exact_touching_points_pca(
    center: np.ndarray,
    L: float,
    ref_pos: np.ndarray,
    r_new: float,
    r_ref: float,
    num_points: int = 8
) -> np.ndarray:
    """Analytical geometric solver for PCA based on the FLAGE algorithm (Skorupski et al., 2014).
    
    Calculates positions on a sphere centered at `center` with radius `L` where
    a new particle exactly touches a reference particle.
    
    This is equivalent to finding the intersection of two spheres:
    1. Sphere centered at `center` with radius `L`.
    2. Sphere centered at `ref_pos` with radius `r_new + r_ref`.
    
    Args:
        center (np.ndarray): Cluster centroid/geometric center.
        L (float): Distance constraint for the new particle from the center.
        ref_pos (np.ndarray): Coordinates of the reference particle.
        r_new (float): Radius of the new particle.
        r_ref (float): Radius of the reference particle.
        num_points (int): Number of points to sample on the intersection circle.
        
    Returns:
        np.ndarray: Array of valid coordinate points (K, 3), may be empty if spheres don't intersect.
    """
    # Vector C -> B (Center to reference particle)
    CB = ref_pos - center
    dist_CB = np.linalg.norm(CB)
    
    if dist_CB < 1e-8:
        # Reference particle at center, intersection is the entire sphere if L == r_new + r_ref
        # This degenerate case is not handled here
        return np.empty((0, 3))
        
    # Law of Cosines to calculate angle alpha
    # In triangle CAB:
    # |CA| = L
    # |CB| = dist_CB
    # |AB| = r_new + r_ref
    dist_AB = r_new + r_ref
    
    cos_alpha = (L**2 + dist_CB**2 - dist_AB**2) / (2 * L * dist_CB)
    
    if cos_alpha < -1.0 or cos_alpha > 1.0:
        # Spheres do not intersect
        return np.empty((0, 3))
        
    alpha = np.arccos(cos_alpha)
    
    # Build an orthogonal basis where u is parallel to CB
    u = CB / dist_CB
    
    # Find a random vector v orthogonal to u
    temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, temp)
    v /= np.linalg.norm(v)
    
    # w forms a right-handed orthogonal basis with u and v
    w = np.cross(u, v)
    
    # Circle radius and center
    # Projection of new particle A onto CB has length L * cos_alpha
    circle_center = center + u * (L * cos_alpha)
    circle_radius = L * np.sin(alpha)
    
    # Sample num_points on the circle
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    points = np.zeros((num_points, 3))
    for i, theta in enumerate(thetas):
        points[i] = circle_center + circle_radius * (np.cos(theta) * v + np.sin(theta) * w)
        
    return points

def filter_overlapping_candidates(
    candidates: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    r_new: float,
    overlap_tolerance: float = 1e-5
) -> np.ndarray:
    """Filters candidate points that overlap with existing particles.
    """
    if len(candidates) == 0:
        return candidates
        
    valid_candidates = []
    # min_dists broadcast
    min_dists = radii + r_new - overlap_tolerance
    
    for cand in candidates:
        dists = np.linalg.norm(positions - cand, axis=1)
        if not np.any(dists < min_dists):
            valid_candidates.append(cand)

    return np.array(valid_candidates)

def build_particle_list_pca(
    positions: np.ndarray,
    radii: np.ndarray,
    L: float,
    a: float,
) -> List[int]:
    """Build list of particle indices that could intersect with the new sphere.

    From Skorupski 2014: particles within distance [L - 2*a, L + 12*a] of center.
    """
    dists = np.linalg.norm(positions, axis=1)
    lower = max(L - 2.0 * a - a, 0.0)
    upper = L + 12.0 * a + a
    mask = (dists >= lower) & (dists <= upper)
    return list(np.where(mask)[0])

def solve_pca_placement(
    center: np.ndarray,
    L: float,
    ref_pos: np.ndarray,
    r_new: float,
    radii: np.ndarray,
    positions: np.ndarray,
    overlap_tolerance: float = 1e-5,
    ref_idx: Optional[int] = None,
    max_ref_changes: int = 5,
    points_per_ref: int = 8,
) -> Optional[np.ndarray]:
    """Algebraic PCA placement using FLAGE method.

    Picks a reference particle, computes exact touching circle, samples points,
    checks for overlaps. If all points overlap, rotates around reference axis
    (quaternion) before trying a new reference.
    """
    r_ref = radii[ref_idx] if ref_idx is not None else radii[0]

    for _ in range(max_ref_changes):
        candidates = find_exact_touching_points_pca(
            center, L, ref_pos, r_new, r_ref, num_points=points_per_ref
        )
        if len(candidates) == 0:
            return None

        valid = filter_overlapping_candidates(
            candidates, positions, radii, r_new, overlap_tolerance
        )
        if len(valid) > 0:
            return valid[np.random.randint(len(valid))]

    return None
