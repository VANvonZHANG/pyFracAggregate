from abc import ABC, abstractmethod

import numpy as np

from pyFracAggregate.core.aggregate import Aggregate


class PlacementStrategy(ABC):
    """Strategy for placing particles during fractal aggregate generation."""

    @abstractmethod
    def place_particle(
        self,
        agg: Aggregate,
        candidate_radius: float,
        candidate_mass: float,
        geom_center: np.ndarray,
        L: float,
        mean_radius: float,
    ) -> tuple:
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
    ) -> np.ndarray:
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


def get_placement(name: str) -> PlacementStrategy:
    """Factory for placement strategies.

    Args:
        name: 'algebraic' (default, FLAGE) or 'random' (Filippov).

    Raises:
        ValueError: If name is not recognized.
    """
    # Will be populated in Task 2 and Task 3
    raise ValueError(f"Unknown placement strategy: {name}")
