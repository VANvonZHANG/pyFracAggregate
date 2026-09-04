import numpy as np
import pytest
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.scaling import (
    CountScaling, MassScaling, ScalingLaw, get_scaling,
)

MASS = (4.0 / 3.0) * np.pi * 1.0 ** 3  # monodisperse unit-radius particle mass


def _agg(positions, radius=1.0):
    agg = Aggregate(len(positions))
    for p in positions:
        agg.add_particle(p[0], p[1], p[2], radius, MASS)
    return agg


class TestCountScaling:
    def test_weights_are_ones(self):
        agg = _agg([[0, 0, 0], [2, 0, 0]])
        assert np.array_equal(CountScaling(1.8, 1.3).weights(agg), np.ones(2))

    def test_char_radius_is_arithmetic_mean(self):
        assert CountScaling(1.8, 1.3).char_radius(np.array([1.0, 3.0])) == pytest.approx(2.0)

    def test_pca_step_matches_filippov_eq10(self):
        # Hand-check against the pre-refactor pca.py expression for n=3.
        agg = _agg([[0, 0, 0], [2, 0, 0]])
        radii = np.ones(3)
        n, df, kf, a = 3, 1.8, 1.3, 1.0
        term1 = (n**2 * a**2) / (n - 1) * (n / kf) ** (2.0 / df)
        term2 = (n * a**2) / (n - 1)
        term3 = n * a**2 * ((n - 1) / kf) ** (2.0 / df)
        expected_L = np.sqrt(max(term1 - term2 - term3, 1.0))
        center, L = CountScaling(df, kf).pca_step(agg, 1.0, MASS, radii)
        assert L == pytest.approx(expected_L, rel=1e-15)
        assert np.array_equal(center, np.mean(agg.positions, axis=0))

    def test_cca_gamma_matches_old_cca_expression(self):
        agg1 = _agg([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
        agg2 = _agg([[0, 0, 0], [2, 0, 0]])
        df, kf = 1.8, 1.3
        from pyFracAggregate.analysis.morphology import radius_of_gyration
        N1, N2 = 3, 2
        N = N1 + N2
        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N
        t1 = (a**2 * N**2) / (N1 * N2) * (N / kf) ** (2.0 / df)
        t2 = (N / N2) * radius_of_gyration(agg1) ** 2
        t3 = (N / N1) * radius_of_gyration(agg2) ** 2
        expected = np.sqrt(max(t1 - t2 - t3, 0.0))
        assert CountScaling(df, kf).cca_gamma(agg1, agg2) == pytest.approx(expected, rel=1e-15)


class TestMassScaling:
    def test_weights_are_dimensionless_and_exact_for_monodisperse(self):
        agg = _agg([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
        w = MassScaling(1.8, 1.3).weights(agg)
        assert np.array_equal(w, np.ones(3))  # bit-exact by IEEE x/x == 1.0

    def test_pca_step_reduces_to_count_for_monodisperse(self):
        # Mathematically exact reduction (deviation N2); assert machine precision.
        agg = _agg([[0, 0, 0], [2, 0, 0]])
        radii = np.ones(4)
        count = CountScaling(1.8, 1.3).pca_step(agg, 1.0, MASS, radii)
        mass = MassScaling(1.8, 1.3).pca_step(agg, 1.0, MASS, radii)
        assert mass[1] == pytest.approx(count[1], rel=1e-12)
        assert np.allclose(mass[0], count[0], rtol=1e-12, atol=1e-12)

    def test_cca_gamma_matches_fracval_expression(self):
        agg1 = _agg([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
        agg2 = _agg([[0, 0, 0], [2, 0, 0]])
        df, kf = 1.8, 1.3
        from pyFracAggregate.analysis.morphology import radius_of_gyration
        N1, N2 = 3, 2
        N = N1 + N2
        m1 = np.sum(agg1.masses)
        m2 = np.sum(agg2.masses)
        m = m1 + m2
        r_p = np.mean(np.concatenate([agg1.radii, agg2.radii]))
        Rg = r_p * (N / kf) ** (1.0 / df)
        expected = np.sqrt(max((m**2 * Rg**2 - m * (m1 * radius_of_gyration(agg1)**2
                               + m2 * radius_of_gyration(agg2)**2)) / (m1 * m2), 0.0))
        assert MassScaling(df, kf).cca_gamma(agg1, agg2) == pytest.approx(expected, rel=1e-15)

    def test_cca_gamma_reduces_to_count_for_monodisperse(self):
        agg1 = _agg([[0, 0, 0], [2, 0, 0], [0, 2, 0]])
        agg2 = _agg([[0, 0, 0], [2, 0, 0]])
        c = CountScaling(1.8, 1.3).cca_gamma(agg1, agg2)
        m = MassScaling(1.8, 1.3).cca_gamma(agg1, agg2)
        assert m == pytest.approx(c, rel=1e-12)


class TestRegistry:
    def test_get_scaling_by_name(self):
        assert isinstance(get_scaling("count", 1.8, 1.3), CountScaling)
        assert isinstance(get_scaling("mass", 1.8, 1.3), MassScaling)

    def test_get_scaling_passthrough_instance(self):
        law = CountScaling(1.8, 1.3)
        assert get_scaling(law, 1.8, 1.3) is law

    def test_get_scaling_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scaling law"):
            get_scaling("volumetric", 1.8, 1.3)

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ScalingLaw(1.8, 1.3)
