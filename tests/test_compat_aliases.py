"""T4 alias equivalence + T8 deprecation warnings (spec 6)."""
import numpy as np
import pytest
import pyFracAggregate as pfa


def test_method_fracval_alias_equals_explicit_coordinate():
    dist = pfa.LognormalDistribution(1.0, 1.6)
    with pytest.warns(DeprecationWarning, match="method='fracval'"):
        a = pfa.generate(60, 1.8, 1.3, method="fracval", particle_dist=dist, seed=99)
    b = pfa.generate(60, 1.8, 1.3, method="cca", scaling="mass",
                     placement="constructed", particle_dist=dist, seed=99)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.radii, b.radii)


@pytest.mark.parametrize("old,new", [("algebraic", "solved"), ("random", "sampled")])
def test_placement_old_name_alias(old, new):
    with pytest.warns(DeprecationWarning, match=f"placement='{old}'"):
        a = pfa.generate(40, 1.8, 1.3, method="pca", placement=old, seed=5)
    b = pfa.generate(40, 1.8, 1.3, method="pca", placement=new, seed=5)
    assert np.array_equal(a.positions, b.positions)


def test_removed_classes_are_gone():
    import pyFracAggregate
    assert not hasattr(pyFracAggregate, "FracVALGenerator")
    assert not hasattr(pyFracAggregate, "ThouyJullienGenerator")


def test_tdcca_removed_message():
    with pytest.raises(ValueError, match="removed in v0.4"):
        pfa.generate(16, 1.8, 1.3, method="tdcca")


def test_surface_beta_kwarg_still_reaches_solved():
    a = pfa.generate(50, 1.8, 1.3, method="cca", surface_beta=0.7, seed=3)
    assert a.current_size == 50
