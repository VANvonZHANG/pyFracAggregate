import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.placement._helpers import (
    random_monte_carlo_place,
    random_monte_carlo_merge,
)
from pyFracAggregate.generators.placement.base import PlacementStrategy


class RandomPlacement(PlacementStrategy):
    """Random Monte Carlo placement (Filippov et al., 2000).

    Samples positions on the Gamma sphere with gradual tolerance relaxation.
    """

    def __init__(self, overlap_tolerance: float = 0.0):
        self.overlap_tolerance = overlap_tolerance

    def place_particle(
        self,
        agg: Aggregate,
        candidate_radius: float,
        candidate_mass: float,
        geom_center: np.ndarray,
        L: float,
        mean_radius: float,
    ) -> tuple | None:
        return random_monte_carlo_place(
            agg, candidate_radius, geom_center, L, mean_radius, self.overlap_tolerance
        )

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
        return random_monte_carlo_merge(
            pos1, r1, pos2_centered, r2, Gamma, mean_radius,
            self.overlap_tolerance, track_best=True,
        )
