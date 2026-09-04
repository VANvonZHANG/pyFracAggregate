"""T1: combination matrix — 10 legal cells reach generation; 1 illegal raises."""
import pytest
import pyFracAggregate as pfa

LEGAL = [
    ("pca", "count", "sampled"), ("pca", "count", "solved"),
    ("pca", "mass", "sampled"), ("pca", "mass", "solved"),
    ("cca", "count", "sampled"), ("cca", "count", "solved"),
    ("cca", "count", "constructed"),
    ("cca", "mass", "sampled"), ("cca", "mass", "solved"),
    ("cca", "mass", "constructed"),
]


@pytest.mark.parametrize("method,scaling,placement", LEGAL)
def test_legal_combination_generates(method, scaling, placement):
    agg = pfa.generate(n_particles=20, df=1.8, kf=1.3, method=method,
                       scaling=scaling, placement=placement, seed=42)
    assert agg.current_size == 20


def test_pca_constructed_raises_with_helpful_message():
    with pytest.raises(ValueError, match=
            r"method='pca' does not support placement='constructed'.*"
            r"Valid placements for 'pca': 'sampled', 'solved'"):
        pfa.generate(n_particles=20, df=1.8, kf=1.3, method="pca",
                     placement="constructed")


def test_bad_scaling_raises():
    with pytest.raises(ValueError, match="Unknown scaling law"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, scaling="volumetric")


def test_bad_placement_raises():
    with pytest.raises(ValueError, match="Unknown placement strategy"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, placement="algebra-ish")


def test_tdcca_removed_raises():
    with pytest.raises(ValueError, match="removed in v0.4"):
        pfa.generate(n_particles=16, df=1.8, kf=1.3, method="tdcca")
