import numpy as np
from typing import Tuple, List

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
