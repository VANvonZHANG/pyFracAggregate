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

from pyFracAggregate.analysis.correlation import estimate_fractal_dimension  # noqa: F401
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
