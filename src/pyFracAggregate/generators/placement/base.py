import warnings
from abc import ABC, abstractmethod

import numpy as np

from pyFracAggregate.core.aggregate import Aggregate


class PlacementStrategy(ABC):
    """Strategy for placing particles during fractal aggregate generation.

    The owning generator sets `rng` to its seeded
    `numpy.random.Generator` so placement draws are reproducible with
    `seed=`.
    """

    rng: np.random.Generator

    @abstractmethod
    def place_particle(
        self,
        agg: Aggregate,
        candidate_radius: float,
        candidate_mass: float,
        geom_center: np.ndarray,
        L: float,
        mean_radius: float,
    ) -> tuple | None:
        """Place a single particle onto the Gamma sphere (PCA stage).

        Args:
            agg: Current aggregate with existing particles.
            candidate_radius: Radius of the new particle.
            candidate_mass: Mass of the new particle.
            geom_center: Geometric center of existing particles.
            L: Required distance from center to new particle.
            mean_radius: Mean particle radius.

        Returns:
            (x, y, z) position tuple, or None if placement failed.
        """

    @abstractmethod
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
    ) -> np.ndarray | None:
        """Merge two sub-clusters (CCA stage).

        Args:
            pos1: Positions of cluster 1 centered at origin.
            r1: Radii of cluster 1.
            agg1: Cluster 1 aggregate.
            pos2_centered: Positions of cluster 2 centered at its COM.
            r2: Radii of cluster 2.
            agg2: Cluster 2 aggregate.
            Gamma: Required COM distance between clusters.
            mean_radius: Mean particle radius.

        Returns:
            pos2_final array (N2, 3), or None if merge failed.
        """


def get_placement(
    name_or_strategy,
    *,
    overlap_tolerance: float = 1e-5,
    surface_beta: float = 0.3,
    rng: "np.random.Generator | None" = None,
) -> PlacementStrategy:
    """Factory for placement strategies.

    Accepts a name ('sampled' | 'solved' | 'constructed') or a
    PlacementStrategy instance (pandas-style coercion). The legacy names
    'algebraic' and 'random' resolve with a DeprecationWarning; they will be
    removed in 1.0.

    Args:
        name_or_strategy: Strategy name or instance.
        overlap_tolerance: Allowed interpenetration between sphere surfaces.
        surface_beta: Surface-particle filter fraction (solved only).
        rng: Random generator threaded from the owning generator.

    Raises:
        ValueError: If the name is not recognized.
    """
    from pyFracAggregate.generators.placement.constructed import ConstructedPlacement
    from pyFracAggregate.generators.placement.sampled import SampledPlacement
    from pyFracAggregate.generators.placement.solved import SolvedPlacement

    if isinstance(name_or_strategy, PlacementStrategy):
        return name_or_strategy

    _DEPRECATED = {
        "algebraic": ("solved", "placement='algebraic' is deprecated; use placement='solved'"),
        "random": ("sampled", "placement='random' is deprecated; use placement='sampled'"),
    }
    name = str(name_or_strategy).lower()
    if name in _DEPRECATED:
        new_name, msg = _DEPRECATED[name]
        warnings.warn(f"{msg}. The alias will be removed in 1.0.",
                      DeprecationWarning, stacklevel=2)
        name = new_name

    if name == "solved":
        return SolvedPlacement(overlap_tolerance=overlap_tolerance,
                               surface_beta=surface_beta, rng=rng)
    if name == "sampled":
        return SampledPlacement(overlap_tolerance=overlap_tolerance, rng=rng)
    if name == "constructed":
        return ConstructedPlacement(overlap_tolerance=overlap_tolerance, rng=rng)
    raise ValueError(
        f"Unknown placement strategy: {name_or_strategy!r}. "
        "Valid values: 'sampled', 'solved', 'constructed'."
    )
