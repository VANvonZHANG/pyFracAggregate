import numpy as np
from scipy.spatial import cKDTree
from pyFracAggregate.core.aggregate import Aggregate

def pair_correlation_function(
    aggregate: Aggregate, 
    bins: int = 50, 
    r_max: float = None
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the two-point density correlation function C(r).

    Efficient calculation based on scipy.spatial.cKDTree.
    
    C(r) = n(r) / (4 * pi * r^2 * h * N)
    where n(r) is the number of particle pairs between distance r and r+h,
    N is the total number of particles, and h is the step size (bin width).
    
    Args:
        aggregate (Aggregate): Cluster object.
        bins (int): Number of bins for r.
        r_max (float, optional): Maximum distance for calculating the correlation function.
            If None, the maximum distance between particles is used.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (r_centers, C_r) where r_centers are the bin 
            center distances and C_r are the corresponding correlation values.
    """
    if aggregate.current_size < 2:
        return np.array([]), np.array([])
        
    positions = aggregate.positions
    N = aggregate.current_size
    
    # Build KDTree for accelerated queries
    tree = cKDTree(positions)
    
    if r_max is None:
        # Estimate maximum distance (upper bound of the furthest pair)
        # Use 2x the maximum distance from center as a safe upper bound
        com = np.mean(positions, axis=0)
        max_dist_to_center = np.max(np.linalg.norm(positions - com, axis=1))
        r_max = 2.0 * max_dist_to_center
        if r_max == 0:
            return np.array([]), np.array([])
            
    # Calculate distance statistics (upper triangle only to avoid double counting, 
    # though tree.count_neighbors handles pair counting).
    # tree.count_neighbors returns cumulative counts (<= r), so we differentiate.
    r_edges = np.linspace(0, r_max, bins + 1)
    
    # count_neighbors can take multiple radii at once
    # cumulative_counts[i] contains number of pairs with distance <= r_edges[i]
    cumulative_counts = tree.count_neighbors(tree, r_edges)
    
    # Subtract self-matches (N) at r=0 to avoid interference with real pair statistics.
    # (Coincident points are not expected due to collision detection).
    cumulative_counts = np.array(cumulative_counts, dtype=np.float64)
    # Subtract N self-matches for all r >= 0 counts
    cumulative_counts -= N
    # Ensure no negative values (preventing anomalies due to precision)
    cumulative_counts = np.maximum(cumulative_counts, 0)
    
    # Differentiate to get counts in each bin: n(r)
    n_r = np.diff(cumulative_counts)
    
    # r_centers are the interval midpoints
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    h = r_edges[1] - r_edges[0]
    
    # Avoid division by zero at r=0, although r_centers > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        c_r = n_r / (4.0 * np.pi * (r_centers ** 2) * h * N)
        
    # Assign 0 for r=0 or cases where distance is too small (causing div by 0)
    c_r = np.nan_to_num(c_r, posinf=0.0)
    
    return r_centers, c_r
