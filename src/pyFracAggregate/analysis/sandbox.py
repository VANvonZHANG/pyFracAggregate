"""Sandbox (mass-radius) estimators for the number and mass measures.

Two definitions that differ by one word:

- number Df:  the number of primary centres within radius r of a primary
  scales as ``N(r) ~ r**Df``;
- mass Df:    the summed mass of primaries within radius r of a primary
  scales as ``M(r) ~ r**(Df, m)``.

The material density is constant, so mass weights ``m_i ~ r_i**3`` differ
from volume weights only by a constant factor: mass-based and volume-based
weighting are identical, and the fitted exponent is unaffected.

The sandbox average includes the seed particle itself (standard sandbox
convention); the fit window is ``[mean primary radius, Rg]`` unless
overridden.
"""

import numpy as np
from scipy.spatial import cKDTree

from pyFracAggregate.analysis.correlation import estimate_fractal_dimension
from pyFracAggregate.analysis.morphology import radius_of_gyration
from pyFracAggregate.core.aggregate import Aggregate


def _radius_curve(
    aggregate: Aggregate,
    weights: np.ndarray,
    bins: int = 15,
    r_min: "float | None" = None,
    r_max: "float | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sandbox core: mean cumulative weighted neighbour sum ``<W(r)>``.

    For each primary (seed included) sum ``weights`` over all centres within
    distance r, then average over all seeds. Log-spaced grid of ``bins``
    points spanning ``[r_min, r_max]`` (defaults: mean primary radius to
    radius of gyration).
    """
    if aggregate.current_size < 2:
        return np.array([]), np.array([])

    if r_max is None:
        r_max = radius_of_gyration(aggregate)
    if r_min is None:
        r_min = float(np.mean(aggregate.radii))
    if r_min <= 0 or r_max <= r_min:
        return np.array([]), np.array([])

    r_centers = np.geomspace(r_min, r_max, bins)
    tree = cKDTree(aggregate.positions)
    curve = np.array([
        np.mean([weights[nb].sum() for nb in tree.query_ball_point(aggregate.positions, r)])
        for r in r_centers
    ])
    return r_centers, curve


def number_radius_function(
    aggregate: Aggregate,
    bins: int = 15,
    r_min: "float | None" = None,
    r_max: "float | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``<N(r)>``: mean number of primary centres within r of a primary.

    Args:
        aggregate (Aggregate): Cluster object.
        bins (int): Number of log-spaced grid points.
        r_min (float, optional): Lower grid bound (default: mean primary radius).
        r_max (float, optional): Upper grid bound (default: radius of gyration).

    Returns:
        tuple[np.ndarray, np.ndarray]: (r_centers, N_r).
    """
    return _radius_curve(aggregate, np.ones(aggregate.current_size), bins, r_min, r_max)


def mass_radius_function(
    aggregate: Aggregate,
    bins: int = 15,
    r_min: "float | None" = None,
    r_max: "float | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``<M(r)>``: mean summed primary mass within r of a primary.

    Weights are ``r_i**3`` (volume); with constant material density mass-
    and volume-weighting differ only by a constant factor and yield
    identical exponents.

    Args:
        aggregate (Aggregate): Cluster object.
        bins (int): Number of log-spaced grid points.
        r_min (float, optional): Lower grid bound (default: mean primary radius).
        r_max (float, optional): Upper grid bound (default: radius of gyration).

    Returns:
        tuple[np.ndarray, np.ndarray]: (r_centers, M_r).
    """
    return _radius_curve(aggregate, aggregate.radii ** 3, bins, r_min, r_max)


def _sandbox_dimension(
    aggregate: Aggregate,
    weights: np.ndarray,
    bins: int = 15,
    r_min: "float | None" = None,
    r_max: "float | None" = None,
) -> tuple[float, float, dict]:
    """One-shot core: curve + power-law fit over the (effective) window."""
    r_centers, curve = _radius_curve(aggregate, weights, bins, r_min, r_max)
    if len(r_centers) < 2:
        return 0.0, 0.0, {}
    # estimate_fractal_dimension returns slope + 3 (the pair-correlation
    # convention C(r) ~ r**(Df-3)); the sandbox cumulative curve obeys
    # <W(r)> ~ r**Df directly, so Df is the raw slope: undo the +3.
    df, r2, fit = estimate_fractal_dimension(
        r_centers, curve, r_min=r_centers[0], r_max=r_centers[-1]
    )
    return df - 3.0, r2, fit


def number_sandbox_dimension(
    aggregate: Aggregate,
    bins: int = 15,
    r_min: "float | None" = None,
    r_max: "float | None" = None,
) -> tuple[float, float, dict]:
    """Estimate the number-based fractal dimension Df,n from ``<N(r)>``.

    Fits ``<N(r)> ~ r**Df`` on a log-log grid (default window: mean primary
    radius to Rg); returns (Df, R_squared, fit_results) like
    ``estimate_fractal_dimension``.
    """
    return _sandbox_dimension(
        aggregate, np.ones(aggregate.current_size), bins, r_min, r_max
    )


def mass_sandbox_dimension(
    aggregate: Aggregate,
    bins: int = 15,
    r_min: "float | None" = None,
    r_max: "float | None" = None,
) -> tuple[float, float, dict]:
    """Estimate the mass-based fractal dimension Df,m from ``<M(r)>``.

    Fits ``<M(r)> ~ r**(Df, m)``; weights are ``r_i**3`` (volume ≡ mass at
    constant density). For monodisperse primaries the result is identical
    to ``number_sandbox_dimension`` (constant weights cancel in the slope).
    """
    return _sandbox_dimension(
        aggregate, aggregate.radii ** 3, bins, r_min, r_max
    )


def plot_sandbox(
    aggregate: Aggregate,
    bins: int = 15,
    show_fit: bool = True,
    reference_df: "float | None" = None,
    measure: str = "both",
    save_path: "str | None" = None,
) -> None:
    """Plot ``<N(r)>`` and/or ``<M(r)>`` on log-log axes with fractal fits.

    The default ``measure="both"`` overlays the number and mass curves with
    their respective fits — the measure-comparison figure.

    Args:
        aggregate (Aggregate): Cluster object.
        bins (int): Number of log-spaced grid points.
        show_fit (bool): Whether to draw the power-law fits.
        reference_df (float, optional): Reference Df slope to show.
        measure (str): 'num', 'mass', or 'both'.
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

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if measure in ("num", "both"):
        curves["num"] = number_radius_function(aggregate, bins=bins)
    if measure in ("mass", "both"):
        curves["mass"] = mass_radius_function(aggregate, bins=bins)
    if not curves or all(len(v[0]) == 0 for v in curves.values()):
        print("Warning: Aggregate has too few particles for sandbox analysis.")
        return

    plt.figure(figsize=(8, 6))
    colors = {"num": "tab:blue", "mass": "tab:orange"}
    names = {"num": r"$\langle N(r) \rangle$", "mass": r"$\langle M(r) \rangle$"}
    for key, (r, curve) in curves.items():
        plt.loglog(r, curve, "o", markersize=4, alpha=0.7, color=colors[key],
                   label=f"{names[key]} ({key})")
        if show_fit:
            df_pcf, r2, fit = estimate_fractal_dimension(
                r, curve, r_min=r[0], r_max=r[-1]
            )
            if fit:
                df = df_pcf - 3.0  # cumulative curve: Df is the raw slope
                plt.loglog(fit["x_fit"], fit["y_fit"], "-", linewidth=2,
                           color=colors[key],
                           label=f"Fit ({key}): $D_f$={df:.2f}, $R^2$={r2:.3f}")

    if reference_df is not None:
        r_ref, c_ref = next(iter(curves.values()))
        mid = len(r_ref) // 2
        # cumulative curve: the reference slope is Df itself (no -3; that is
        # the differenced-PCF convention used in plot_pair_correlation)
        intercept = np.log10(c_ref[mid]) - reference_df * np.log10(r_ref[mid])
        plt.loglog(r_ref, 10 ** (reference_df * np.log10(r_ref) + intercept),
                   "g--", alpha=0.5, label=f"Ref: $D_f$={reference_df}")

    plt.axvline(float(np.mean(aggregate.radii)), color="gray", linestyle="--",
                alpha=0.5, label="Min Fit Bound")
    plt.axvline(radius_of_gyration(aggregate), color="gray", linestyle=":",
                alpha=0.5, label="Max Fit Bound ($R_g$)")

    plt.xlabel(f"Distance $r$ [{aggregate.length_unit}]")
    plt.ylabel(r"$\langle N(r) \rangle$ / $\langle M(r) \rangle$")
    plt.title("Sandbox (Mass-Radius) Analysis")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
