"""Shared metrics for all benchmark experiments (single source of truth).

Conventions follow the spec (docs/superpowers/specs/2026-09-02-paper-benchmarks-design.md):
C(r) pair-count form (Skorupski et al. 2014 Eq. 11), fit window [2*r_mean, Rg],
KS test against lognormal CDF (Moran et al. 2019 Sec. 4.1).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats


def pair_count_cr(positions: np.ndarray, n_bins: int = 50,
                  r_max: float | None = None):
    """Pair-count correlation function C(r) = n(r) / (4*pi*r^2*dr*N).

    Log-spaced bins from the minimum pair distance to r_max
    (default: max pair distance). Returns (r_centers, c_r, dr).
    """
    tree = cKDTree(positions)
    if r_max is None:
        r_max = float(np.max(np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2)))
    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("no pairs within r_max")
    dists = np.linalg.norm(
        positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    d_min = float(dists.min())
    edges = np.geomspace(d_min, r_max, n_bins + 1)
    counts, _ = np.histogram(dists, bins=edges)
    r_centers = np.sqrt(edges[:-1] * edges[1:])
    dr = np.diff(edges)
    n = len(positions)
    shell_vol = 4 * np.pi * r_centers**2 * dr
    c_r = counts / (shell_vol * n)
    keep = c_r > 0
    return r_centers[keep], c_r[keep], dr[keep]


def fit_df(r_centers: np.ndarray, c_r: np.ndarray,
           r_min: float, r_max: float):
    """Log-log OLS of C(r) in [r_min, r_max]; Df = slope + 3.

    Returns (df_est, r_squared, n_points); n_points < 3 -> (nan, nan, n).
    """
    mask = (r_centers >= r_min) & (r_centers <= r_max) & (c_r > 0)
    x, y = np.log(r_centers[mask]), np.log(c_r[mask])
    if len(x) < 3:
        return float("nan"), float("nan"), int(len(x))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope + 3.0), r2, int(len(x))


def ks_lognormal(radii: np.ndarray, r_geo: float, sg: float) -> float:
    """Two-sided KS statistic of radii vs lognormal(mu=ln r_geo, sigma=ln sg)."""
    cdf = stats.lognorm(s=np.log(sg), scale=r_geo).cdf
    return float(stats.kstest(np.asarray(radii, dtype=float), cdf).statistic)


def overlap_worst(positions: np.ndarray, radii: np.ndarray) -> float:
    """Worst overlap ratio max[(ri+rj-d)/min(ri,rj)] over close pairs."""
    tree = cKDTree(positions)
    rmax = float(np.max(radii))
    pairs = tree.query_pairs(r=2 * rmax, output_type="ndarray")
    worst = 0.0
    for i, j in pairs:
        d = float(np.linalg.norm(positions[i] - positions[j]))
        viol = float(radii[i] + radii[j]) - d
        if viol > 0:
            worst = max(worst, viol / float(min(radii[i], radii[j])))
    return worst


def evaluate_run(spec: dict, agg, elapsed: float) -> dict:
    """Compute the full metric row for one generated aggregate."""
    n = len(agg.radii)
    r_mean = float(np.mean(agg.radii))
    rg = float(np.sqrt(np.mean(np.sum(
        (agg.positions - agg.positions.mean(axis=0)) ** 2, axis=1))))
    rg_target = r_mean * (n / spec["kf"]) ** (1.0 / spec["df"])
    rc, cr, dr = pair_count_cr(agg.positions, n_bins=50)
    df_est, r2, npts = fit_df(rc, cr, 2 * r_mean, rg)
    integral = float(np.sum(cr * 4 * np.pi * rc**2 * dr))
    norm_err = abs(integral / ((n - 1) / 2) - 1.0)
    ks_d = (ks_lognormal(agg.radii, 1.0, spec["sg"])
            if spec["sg"] > 1.0 else float("nan"))
    return {
        "df_est": df_est, "df_est_r2": r2, "fit_npts": npts,
        "norm_err": norm_err, "rg": rg, "rg_target": rg_target,
        "ks_d": ks_d, "overlap_worst": overlap_worst(agg.positions, agg.radii),
    }
