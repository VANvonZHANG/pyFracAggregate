import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from pyFracAggregate.core.aggregate import Aggregate

def pair_correlation_function(
    aggregate: Aggregate,
    bins: int = 50,
    r_max: "float | None" = None
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

def mass_pair_correlation_function(
    aggregate: Aggregate,
    bins: int = 50,
    r_max: "float | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the mass-weighted pair correlation function C_m(r).

    Each pair (i, j) contributes weight m_i * m_j with m_i = r_i**3
    (volume; with constant material density mass- and volume-weighting are
    identical up to a constant factor, so the slope is unaffected):

        C_m(r) = (sum of m_i*m_j over ordered pairs (i != j) with distance
                 in [r, r+h]) / (4 * pi * r^2 * h * M)

    where M = sum_i m_i and h is the linear bin width.
    Each unordered pair contributes twice (once per endpoint), matching
    ``pair_correlation_function``'s ordered-pair counting so the two curves
    are absolutely comparable (monodisperse: C_m = r^3 * C).

    Statistical caveat: a single-realization C_m(r) is noisy — it is
    dominated by the few largest primaries (participation ratio
    sum(m)^2/sum(m^2) is small for lognormal radii). Prefer
    ``mass_sandbox_dimension`` for a single aggregate's Df,m; use this
    function for ensemble-averaged curves.

    Args:
        aggregate (Aggregate): Cluster object.
        bins (int): Number of linear bins for r.
        r_max (float, optional): Maximum distance; defaults to twice the
            largest centre distance from the centroid.

    Returns:
        tuple[np.ndarray, np.ndarray]: (r_centers, C_m).
    """
    if aggregate.current_size < 2:
        return np.array([]), np.array([])

    positions = aggregate.positions
    n = aggregate.current_size
    w = aggregate.radii ** 3

    if r_max is None:
        com = np.mean(positions, axis=0)
        r_max = 2.0 * np.max(np.linalg.norm(positions - com, axis=1))
        if r_max == 0:
            return np.array([]), np.array([])

    r_edges = np.linspace(0, r_max, bins + 1)
    ii, jj = np.triu_indices(n, k=1)
    d = pdist(positions)
    hist, _ = np.histogram(d, bins=r_edges, weights=w[ii] * w[jj])

    r_centers = (r_edges[:-1] + r_edges[1:]) / 2.0
    h = r_edges[1] - r_edges[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        c_m = 2.0 * hist / (4.0 * np.pi * (r_centers ** 2) * h * w.sum())
    c_m = np.nan_to_num(c_m, posinf=0.0)

    return r_centers, c_m

def estimate_fractal_dimension(
    r_centers: np.ndarray,
    c_r: np.ndarray,
    r_min: "float | None" = None,
    r_max: "float | None" = None
) -> tuple[float, float, dict]:
    """Estimates the fractal dimension Df from the pair correlation function C(r).

    Performs log-log linear regression on the fractal regime (a < r < Rg).
    Df is calculated as: Df = slope + 3.

    Args:
        r_centers (np.ndarray): Bin center distances.
        c_r (np.ndarray): Correlation function values.
        r_min (float, optional): Lower bound for regression.
        r_max (float, optional): Upper bound for regression.

    Returns:
        tuple[float, float, dict]: (Df, R_squared, fit_results)
            - Df: Estimated fractal dimension.
            - R_squared: Coefficient of determination.
            - fit_results: Dictionary containing 'slope', 'intercept', 'x_fit', 'y_fit'.
    """
    # Filter valid data (C(r) > 0 for log)
    mask = c_r > 0
    if r_min is not None:
        mask &= (r_centers >= r_min)
    if r_max is not None:
        mask &= (r_centers <= r_max)
    
    x = r_centers[mask]
    y = c_r[mask]
    
    if len(x) < 2:
        return 0.0, 0.0, {}

    log_x = np.log10(x)
    log_y = np.log10(y)
    
    # Linear regression: log10(C(r)) = slope * log10(r) + intercept
    slope, intercept = np.polyfit(log_x, log_y, 1)
    
    # Calculate R-squared
    y_pred = slope * log_x + intercept
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    df = slope + 3.0
    
    fit_results = {
        'slope': slope,
        'intercept': intercept,
        'r_min': np.min(x),
        'r_max': np.max(x),
        'x_fit': x,
        'y_fit': 10 ** y_pred
    }
    
    return df, r_squared, fit_results

def plot_pair_correlation(
    aggregate: Aggregate,
    bins: int = 50,
    show_fit: bool = True,
    reference_df: "float | None" = None,
    measure: str = "num",
    save_path: "str | None" = None
) -> None:
    """Plots the pair correlation function(s) and optionally the fractal fit.

    Args:
        aggregate (Aggregate): Cluster object.
        bins (int): Number of bins for the PCF.
        show_fit (bool): Whether to show the fractal dimension fit.
        reference_df (float, optional): Reference Df to show in plot.
        measure (str): 'num' (C(r), default), 'mass' (C_m(r)), or 'both'.
            Single-realization 'mass' curves are noisy — see
            mass_pair_correlation_function.
        save_path (str, optional): Path to save the figure.
    """
    if measure not in ("num", "mass", "both"):
        raise ValueError(
            f"measure must be 'num', 'mass', or 'both', got {measure!r}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting. Install it with 'pip install matplotlib'.")
        return

    from pyFracAggregate.analysis.morphology import radius_of_gyration

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if measure in ("num", "both"):
        curves["num"] = pair_correlation_function(aggregate, bins=bins)
    if measure in ("mass", "both"):
        curves["mass"] = mass_pair_correlation_function(aggregate, bins=bins)
    if not curves or all(len(v[0]) == 0 for v in curves.values()):
        print("Warning: Aggregate has too few particles for correlation analysis.")
        return

    plt.figure(figsize=(8, 6))
    colors = {"num": "tab:blue", "mass": "tab:orange"}
    labels = {"num": r"$C(r)$", "mass": r"$C_m(r)$"}
    r_min_fit = float(np.mean(aggregate.radii))
    r_max_fit = radius_of_gyration(aggregate)

    for key, (r_centers, c_r) in curves.items():
        plt.loglog(r_centers, c_r, "o", markersize=4, alpha=0.7,
                   color=colors[key], label=f"{labels[key]} ({key})")
        if show_fit:
            df, r2, fit = estimate_fractal_dimension(
                r_centers, c_r, r_min=r_min_fit, r_max=r_max_fit
            )
            if fit:
                plt.loglog(fit['x_fit'], fit['y_fit'], '-', linewidth=2,
                           color=colors[key],
                           label=f"Fit ({key}): $D_f$={df:.2f}, $R^2$={r2:.3f}")

    if reference_df is not None:
        r_ref, c_ref = next(iter(curves.values()))
        mid_idx = len(r_ref) // 2
        ref_intercept = (np.log10(c_ref[mid_idx])
                         - (reference_df - 3.0) * np.log10(r_ref[mid_idx]))
        plt.loglog(r_ref, 10 ** ((reference_df - 3.0) * np.log10(r_ref) + ref_intercept),
                   "g--", alpha=0.5, label=f'Ref: $D_f$={reference_df}')

    if show_fit:
        plt.axvline(r_min_fit, color="gray", linestyle="--", alpha=0.5,
                    label="Min Fit Bound")
        plt.axvline(r_max_fit, color="gray", linestyle=":", alpha=0.5,
                    label="Max Fit Bound ($R_g$)")

    plt.xlabel(f'Distance $r$ [{aggregate.length_unit}]')
    plt.ylabel('Correlation function')
    plt.title('Pair Correlation Function Analysis')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
