import pytest
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.analysis.correlation import pair_correlation_function

def test_pair_correlation_function_two_particles():
    agg = Aggregate(2)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    agg.add_particle(10.0, 0.0, 0.0, 1.0, 1.0)
    
    # 2 particles, distance is exactly 10
    r_centers, c_r = pair_correlation_function(agg, bins=20, r_max=12.0)
    
    assert len(r_centers) == 20
    assert len(c_r) == 20
    
    # C(r) = n(r) / (4 * pi * r^2 * h * N)
    # n(r): Only 1 pair of particles with distance 10. Due to count_neighbors 
    # being bidirectional, the count should be 2.
    # Find the bin where r_centers is closest to 10
    bin_idx = np.argmin(np.abs(r_centers - 10.0))
    
    h = 12.0 / 20.0
    N = 2
    r_val = r_centers[bin_idx]
    
    expected_c_r = 2 / (4.0 * np.pi * r_val**2 * h * N)
    assert np.isclose(c_r[bin_idx], expected_c_r)
    
    # Other bins should be 0
    for i in range(20):
        if i != bin_idx:
            assert c_r[i] == 0.0

def test_pair_correlation_function_empty_or_single():
    agg0 = Aggregate(10)
    r_centers, c_r = pair_correlation_function(agg0)
    assert len(r_centers) == 0
    assert len(c_r) == 0
    
    agg1 = Aggregate(1)
    agg1.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    r_centers, c_r = pair_correlation_function(agg1)
    assert len(r_centers) == 0
    assert len(c_r) == 0

import pyFracAggregate as pfa
from pyFracAggregate.analysis.correlation import mass_pair_correlation_function


def test_mass_pcf_two_particles_analytic():
    agg = Aggregate(3)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    agg.add_particle(10.0, 0.0, 0.0, 2.0, 1.0)

    r_centers, c_m = mass_pair_correlation_function(agg, bins=20, r_max=12.0)
    bin_idx = int(np.argmin(np.abs(r_centers - 10.0)))
    h = 12.0 / 20.0
    total_w = 1.0 ** 3 + 2.0 ** 3
    expected = 2.0 * (1.0 ** 3 * 2.0 ** 3) / (4.0 * np.pi * r_centers[bin_idx] ** 2 * h * total_w)
    assert np.isclose(c_m[bin_idx], expected)
    for i in range(20):
        if i != bin_idx:
            assert c_m[i] == 0.0


def test_mass_pcf_monodisperse_is_scaled_counting_pcf():
    agg = pfa.generate(80, 1.8, 1.9, method="pca", seed=5)
    r, c = pair_correlation_function(agg)
    r_m, c_m = mass_pair_correlation_function(agg)
    assert np.allclose(r, r_m)
    mask = c > 0
    ratio = c_m[mask] / c[mask]
    assert np.allclose(ratio, ratio[0])       # constant factor -> same slope


def test_mass_pcf_too_few_particles():
    agg = Aggregate(1)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    r, c_m = mass_pair_correlation_function(agg)
    assert len(r) == 0 and len(c_m) == 0


def test_plot_pair_correlation_measures(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    agg = pfa.generate(80, 1.8, 1.9, method="pca", seed=5)
    for measure in ("num", "mass", "both"):
        out = tmp_path / f"pcf_{measure}.png"
        pfa.plot_pair_correlation(agg, measure=measure, save_path=str(out))
        assert out.exists() and out.stat().st_size > 0


def test_plot_pair_correlation_rejects_unknown_measure():
    agg = pfa.generate(20, 1.8, 1.9, method="pca", seed=1)
    with pytest.raises(ValueError, match="measure"):
        pfa.plot_pair_correlation(agg, measure="nmu")
