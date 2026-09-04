import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.placement.base import PlacementStrategy


class ConstructedPlacement(PlacementStrategy):
    """Specified contact pair + attitude construction + COM correction
    (Moran et al., 2019, FracVAL sub-steps b-d). Merge-stage only; the merge
    implementation lands in the next commit (plan Task 6)."""

    def __init__(self, overlap_tolerance: float = 1e-5):
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
        raise NotImplementedError(
            "constructed placement applies to cluster merging "
            "(method='cca') only; single particles cannot be 'constructed'."
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
    ) -> np.ndarray:
        raise NotImplementedError("port lands in plan Task 6")
