import inspect
import numpy as np
import pytest
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.placement import get_placement, PlacementStrategy
from pyFracAggregate.generators.placement.solved import SolvedPlacement
from pyFracAggregate.generators.placement.sampled import SampledPlacement
from pyFracAggregate.generators.placement.constructed import ConstructedPlacement


def _mass(r=1.0):
    return (4.0 / 3.0) * np.pi * r ** 3


def _agg3():
    agg = Aggregate(3)
    for p in [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]:
        agg.add_particle(*p, 1.0, _mass())
    return agg


@pytest.mark.parametrize("name,cls", [("sampled", SampledPlacement),
                                      ("solved", SolvedPlacement),
                                      ("constructed", ConstructedPlacement)])
def test_get_placement_by_name(name, cls):
    assert isinstance(get_placement(name), cls)


def test_get_placement_instance_passthrough():
    s = SolvedPlacement()
    assert get_placement(s) is s


def test_get_placement_unknown_raises():
    with pytest.raises(ValueError, match="Unknown placement strategy"):
        get_placement("nonexistent")


def test_old_names_resolve_with_deprecation():
    with pytest.warns(DeprecationWarning, match="placement='algebraic'"):
        assert isinstance(get_placement("algebraic"), SolvedPlacement)
    with pytest.warns(DeprecationWarning, match="placement='random'"):
        assert isinstance(get_placement("random"), SampledPlacement)


def test_abc_requires_both_methods():
    assert set(PlacementStrategy.__abstractmethods__) == {"place_particle", "merge_clusters"}


def test_solved_place_particle_basic():
    agg = _agg3()
    result = SolvedPlacement().place_particle(
        agg, 1.0, _mass(), np.mean(agg.positions, axis=0), 3.0, 1.0)
    assert result is None or len(result) == 3


def test_solved_merge_clusters_basic():
    agg = _agg3()
    out = SolvedPlacement().merge_clusters(
        agg.positions, agg.radii, agg, agg.positions, agg.radii, agg, 5.0, 1.0)
    assert out is not None and out.shape == (3, 3)


def test_sampled_place_particle_basic():
    np.random.seed(0)
    agg = _agg3()
    result = SampledPlacement().place_particle(
        agg, 1.0, _mass(), np.mean(agg.positions, axis=0), 4.0, 1.0)
    assert result is not None and len(result) == 3


def test_sampled_merge_clusters_basic():
    np.random.seed(0)
    agg = _agg3()
    out = SampledPlacement().merge_clusters(
        agg.positions, agg.radii, agg, agg.positions, agg.radii, agg, 5.0, 1.0)
    assert out is not None and out.shape == (3, 3)


def test_solved_surface_beta_default():
    assert SolvedPlacement().surface_beta == 0.3


def test_constructed_place_particle_raises():
    agg = _agg3()
    with pytest.raises(NotImplementedError, match="method='cca'"):
        ConstructedPlacement().place_particle(
            agg, 1.0, _mass(), np.zeros(3), 3.0, 1.0)


def test_base_generator_default_placement_solved():
    from pyFracAggregate.generators.base import BaseGenerator
    import pyFracAggregate as pfa

    class Concrete(BaseGenerator):
        def generate(self):
            raise NotImplementedError

    gen = Concrete(n_particles=10, df=1.8, kf=1.3, particle_dist=pfa.Monodisperse(1.0))
    assert isinstance(gen.placement, SolvedPlacement)
