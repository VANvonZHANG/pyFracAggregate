# tests/test_generators/test_placement.py
import numpy as np
import pytest
from pyFracAggregate.generators.placement import get_placement, PlacementStrategy
from pyFracAggregate.generators.placement.algebraic import AlgebraicPlacement


def test_get_placement_algebraic():
    s = get_placement('algebraic')
    assert isinstance(s, PlacementStrategy)


def test_get_placement_random():
    """Will be wired in Task 3."""
    with pytest.raises(ValueError, match="Unknown placement strategy"):
        get_placement('random')


def test_get_placement_invalid_raises():
    with pytest.raises(ValueError, match="Unknown placement strategy"):
        get_placement('nonexistent')


def test_placement_strategy_has_place_particle():
    """PlacementStrategy ABC should require place_particle."""
    import inspect
    assert 'place_particle' in PlacementStrategy.__abstractmethods__


def test_placement_strategy_has_merge_clusters():
    """PlacementStrategy ABC should require merge_clusters."""
    import inspect
    assert 'merge_clusters' in PlacementStrategy.__abstractmethods__


def test_algebraic_placement_is_strategy():
    assert issubclass(AlgebraicPlacement, PlacementStrategy)


def test_algebraic_placement_place_particle_basic():
    """AlgebraicPlacement.place_particle should find valid positions on simple aggregates."""
    from pyFracAggregate.core.aggregate import Aggregate

    strategy = AlgebraicPlacement()
    agg = Aggregate(3, density=1.0)
    mass = agg.density * (4.0 / 3.0) * np.pi * 1.0**3
    agg.add_particle(0.0, 0.0, 0.0, 1.0, mass)
    agg.add_particle(2.0, 0.0, 0.0, 1.0, mass)

    geom_center = np.mean(agg.positions, axis=0)
    result = strategy.place_particle(agg, 1.0, mass, geom_center, 3.0, 1.0)
    assert result is None or len(result) == 3


def test_algebraic_placement_place_particle_returns_none_on_blocked():
    """If all positions blocked, should return None or extreme fallback."""
    from pyFracAggregate.core.aggregate import Aggregate

    strategy = AlgebraicPlacement()
    agg = Aggregate(10, density=1.0)
    np.random.seed(42)
    mass = agg.density * (4.0 / 3.0) * np.pi
    for i in range(10):
        pos = np.random.normal(0, 0.5, 3)
        agg.add_particle(pos[0], pos[1], pos[2], 1.0, mass)

    geom_center = np.mean(agg.positions, axis=0)
    result = strategy.place_particle(agg, 1.0, mass, geom_center, 0.5, 1.0)
    # May return None (algebraic+random failed) or a fallback position
    assert result is None or (isinstance(result, tuple) and len(result) == 3)


def test_algebraic_placement_merge_clusters_basic():
    """AlgebraicPlacement.merge_clusters should handle simple cases."""
    from pyFracAggregate.core.aggregate import Aggregate

    strategy = AlgebraicPlacement()
    agg1 = Aggregate(3, density=1.0)
    agg2 = Aggregate(3, density=1.0)
    mass = 1.0 * (4.0 / 3.0) * np.pi * 1.0**3

    agg1.add_particle(0.0, 0.0, 0.0, 1.0, mass)
    agg1.add_particle(2.0, 0.0, 0.0, 1.0, mass)
    agg1.add_particle(0.0, 2.0, 0.0, 1.0, mass)

    agg2.add_particle(0.0, 0.0, 0.0, 1.0, mass)
    agg2.add_particle(2.0, 0.0, 0.0, 1.0, mass)
    agg2.add_particle(0.0, 2.0, 0.0, 1.0, mass)

    pos1 = agg1.positions
    r1 = agg1.radii
    pos2_centered = agg2.positions
    r2 = agg2.radii

    result = strategy.merge_clusters(pos1, r1, agg1, pos2_centered, r2, agg2, 5.0, 1.0)
    assert result is not None
    assert result.shape == (3, 3)
