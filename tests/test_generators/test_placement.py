# tests/test_generators/test_placement.py
import pytest
from pyFracAggregate.generators.placement import get_placement, PlacementStrategy


def test_get_placement_algebraic():
    s = get_placement('algebraic')
    assert isinstance(s, PlacementStrategy)


def test_get_placement_random():
    s = get_placement('random')
    assert isinstance(s, PlacementStrategy)


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
