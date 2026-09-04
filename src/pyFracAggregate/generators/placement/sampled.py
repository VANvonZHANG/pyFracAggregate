import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.placement.base import PlacementStrategy
from pyFracAggregate.generators.placement.solvers import mc_touch_merge, mc_touch_place


class SampledPlacement(PlacementStrategy):
    """Emergent contact via Monte Carlo sampling (Filippov et al., 2000)."""

    def __init__(self, overlap_tolerance: float = 1e-5,
                 rng: "np.random.Generator | None" = None):
        self.overlap_tolerance = overlap_tolerance
        self.rng = rng if rng is not None else np.random.default_rng()

    def place_particle(
        self,
        agg: Aggregate,
        candidate_radius: float,
        candidate_mass: float,
        geom_center: np.ndarray,
        L: float,
        mean_radius: float,
    ) -> tuple | None:
        return mc_touch_place(agg, candidate_radius, geom_center, L,
                              mean_radius, self.overlap_tolerance, self.rng)

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
        return mc_touch_merge(pos1, r1, pos2_centered, r2, Gamma, mean_radius,
                              self.overlap_tolerance, self.rng, track_best=True)
