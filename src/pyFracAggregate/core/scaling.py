"""Parallel-axis scaling laws: the target-distance equations of generation.

Both laws implement the parallel-axis (two-body) theorem. CountScaling uses
particle-count weights (Filippov et al., 2000, Eqs. 10-13); MassScaling uses
mass weights (Moran et al., 2019, Eqs. 3 and 6). For a monodisperse primary
distribution the two are mathematically equivalent.
"""
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from pyFracAggregate.analysis.morphology import radius_of_gyration
from pyFracAggregate.core.aggregate import Aggregate


class ScalingLaw(ABC):
    """Strategy for target distances in PCA steps and CCA merges."""

    name: ClassVar[str]

    def __init__(self, df: float, kf: float) -> None:
        self.df = df
        self.kf = kf

    @abstractmethod
    def weights(self, aggregate: Aggregate) -> np.ndarray:
        """Per-particle weights used for centers of mass."""

    @abstractmethod
    def char_radius(self, radii: np.ndarray) -> float:
        """Characteristic primary radius for the scaling-law target."""

    def target_rg_sq(self, n: int, radii: np.ndarray) -> float:
        a = self.char_radius(radii)
        return a * a * (n / self.kf) ** (2.0 / self.df)

    @abstractmethod
    def pca_step(self, agg: Aggregate, r_new: float, m_new: float,
                 all_radii: np.ndarray) -> tuple[np.ndarray, float]:
        """Reference center and step distance for adding one particle.

        Args:
            agg: Aggregate with n-1 existing particles.
            r_new: Radius of the incoming particle.
            m_new: Mass of the incoming particle.
            all_radii: Radii of all n particles (existing + incoming).

        Returns:
            (center, distance): the placement reference point and the
            required distance of the new particle from it.
        """

    @abstractmethod
    def cca_gamma(self, agg1: Aggregate, agg2: Aggregate) -> float:
        """Required center-of-mass separation for merging two clusters."""


class CountScaling(ScalingLaw):
    """Count-weighted parallel axis law (Filippov et al., 2000)."""

    name = "count"

    def weights(self, aggregate: Aggregate) -> np.ndarray:
        return np.ones(aggregate.current_size)

    def char_radius(self, radii: np.ndarray) -> float:
        return float(np.mean(radii))

    def pca_step(self, agg: Aggregate, r_new: float, m_new: float,
                 all_radii: np.ndarray) -> tuple[np.ndarray, float]:
        n = agg.current_size + 1
        a = np.mean(all_radii)
        # Filippov Eq [10]
        term1 = (n**2 * a**2) / (n - 1) * (n / self.kf) ** (2.0 / self.df)
        term2 = (n * a**2) / (n - 1)
        term3 = n * a**2 * ((n - 1) / self.kf) ** (2.0 / self.df)
        L_sq = term1 - term2 - term3
        L = np.sqrt(max(L_sq, r_new**2))
        center = np.mean(agg.positions, axis=0)
        return center, float(L)

    def cca_gamma(self, agg1: Aggregate, agg2: Aggregate) -> float:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)
        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N
        term1 = (a**2 * N**2) / (N1 * N2) * (N / self.kf) ** (2.0 / self.df)
        term2 = (N / N2) * Rg1**2
        term3 = (N / N1) * Rg2**2
        Gamma_sq = term1 - term2 - term3
        return float(np.sqrt(max(Gamma_sq, 0.0)))


class MassScaling(ScalingLaw):
    """Mass-weighted parallel axis law (Moran et al., 2019, FracVAL)."""

    name = "mass"

    def weights(self, aggregate: Aggregate) -> np.ndarray:
        # Dimensionless weights referenced to the first particle keep the
        # monodisperse case bit-close to CountScaling (x/x == 1.0 exactly).
        return aggregate.masses / aggregate.masses[0]

    def char_radius(self, radii: np.ndarray) -> float:
        return float(np.mean(radii))

    def pca_step(self, agg: Aggregate, r_new: float, m_new: float,
                 all_radii: np.ndarray) -> tuple[np.ndarray, float]:
        # Two-body parallel-axis with mass weights, structured to reduce to
        # CountScaling for monodisperse input: the new particle is the
        # reference weight unit (w2 == 1), matching the count equation's
        # intrinsic a^2 term.
        w1 = float(np.sum(self.weights(agg)))
        w2 = 1.0
        w = w1 + w2
        a = np.mean(all_radii)
        rg_target_sq = a**2 * (w / self.kf) ** (2.0 / self.df)
        rg_old_target_sq = a**2 * (w1 / self.kf) ** (2.0 / self.df)
        # For monodisperse (w1 == n-1, w == n exactly) this reduces to the
        # count Filippov Eq [10] form term1 - term2 - term3 with:
        #   term1 = (w**2 * a**2) / w1 * (w / kf)**(2/df)
        #   term2 = (w * a**2 * w2) / w1          # intrinsic of the new sphere
        #   term3 = w * a**2 * ((w1 / kf)**(2/df))
        Gamma_sq = (w**2 * rg_target_sq - w * (w1 * rg_old_target_sq + w2 * r_new**2)) / (w1 * w2)
        L = np.sqrt(max(Gamma_sq, r_new**2))
        center = np.average(agg.positions, axis=0, weights=self.weights(agg))
        return center, float(L)

    def cca_gamma(self, agg1: Aggregate, agg2: Aggregate) -> float:
        # Moran 2019 Eq 3 & 6 (verbatim port of the pre-refactor fracval path)
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2
        m1 = np.sum(agg1.masses)
        m2 = np.sum(agg2.masses)
        m = m1 + m2
        r_p = np.mean(np.concatenate([agg1.radii, agg2.radii]))
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)
        Rg = r_p * (N / self.kf) ** (1.0 / self.df)
        term_target = m**2 * Rg**2
        term_parts = m * (m1 * Rg1**2 + m2 * Rg2**2)
        Gamma_sq = (term_target - term_parts) / (m1 * m2)
        return float(np.sqrt(max(Gamma_sq, 0.0)))


def get_scaling(name_or_law: "str | ScalingLaw", df: float, kf: float) -> ScalingLaw:
    """Resolve a scaling law by name (pandas-style) or pass an instance through."""
    if isinstance(name_or_law, ScalingLaw):
        return name_or_law
    registry = {"count": CountScaling, "mass": MassScaling}
    if name_or_law not in registry:
        raise ValueError(f"Unknown scaling law: {name_or_law!r}. "
                         f"Valid values: {sorted(registry)}")
    return registry[name_or_law](df, kf)
