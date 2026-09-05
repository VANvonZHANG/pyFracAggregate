import numpy as np
import pytest

import pyFracAggregate as pfa
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.analysis.sandbox import (
    number_radius_function,
    mass_radius_function,
    number_sandbox_dimension,
    mass_sandbox_dimension,
)


@pytest.fixture(scope="module")
def agg_mono():
    return pfa.generate(100, 1.8, 1.9, method="pca", seed=3)


def test_curve_shape_grid_and_monotonicity(agg_mono):
    r, curve = number_radius_function(agg_mono)
    assert r.shape == curve.shape == (15,)
    assert np.all(np.diff(r) > 0)          # log-spaced grid, strictly increasing
    assert np.all(np.diff(curve) >= 0)     # cumulative curve is non-decreasing
    assert np.all(curve >= 1.0)            # seed particle counts itself


def test_default_window_matches_convention(agg_mono):
    r, _ = number_radius_function(agg_mono)
    assert np.isclose(r[0], float(np.mean(agg_mono.radii)))
    assert np.isclose(r[-1], pfa.radius_of_gyration(agg_mono))


def test_explicit_window_and_bins(agg_mono):
    r, _ = number_radius_function(agg_mono, bins=7, r_min=2.0, r_max=10.0)
    assert len(r) == 7
    assert np.isclose(r[0], 2.0)
    assert np.isclose(r[-1], 10.0)


def test_mass_curve_is_scaled_number_curve_for_monodisperse(agg_mono):
    r_n, c_n = number_radius_function(agg_mono)
    r_m, c_m = mass_radius_function(agg_mono)
    assert np.array_equal(r_n, r_m)
    ratio = c_m / c_n
    assert np.allclose(ratio, ratio[0])    # constant factor r^3 per particle


def test_too_few_particles_returns_empty():
    for agg in (Aggregate(10), Aggregate(1)):
        if agg.current_size == 1:
            agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
        r, curve = number_radius_function(agg)
        assert len(r) == 0 and len(curve) == 0
        r, curve = mass_radius_function(agg)
        assert len(r) == 0 and len(curve) == 0


@pytest.fixture(scope="module")
def agg_poly():
    return pfa.generate(
        256, 1.8, 1.9,
        method="cca", scaling="mass", placement="constructed",
        particle_dist=pfa.LognormalDistribution(mean=1.0, std=1.6),
        seed=0,
    )


def test_monodisperse_mass_equals_number_exactly(agg_mono):
    # constant weights -> identical curves up to scale -> identical fit
    dfn, r2n, fitn = number_sandbox_dimension(agg_mono)
    dfm, r2m, fitm = mass_sandbox_dimension(agg_mono)
    assert dfn == pytest.approx(dfm, abs=1e-10)
    assert r2n == pytest.approx(r2m, abs=1e-10)
    assert set(fitn) == {"slope", "intercept", "r_min", "r_max", "x_fit", "y_fit"}


def test_poly_seed0_regression_anchors(agg_poly):
    dfn, r2n, _ = number_sandbox_dimension(agg_poly)
    dfm, r2m, _ = mass_sandbox_dimension(agg_poly)
    assert dfn == pytest.approx(1.782, abs=0.01)
    assert dfm == pytest.approx(1.879, abs=0.01)
    assert r2n > 0.95
    assert r2m > 0.95


def test_dimension_too_few_particles():
    agg = Aggregate(1)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    for func in (number_sandbox_dimension, mass_sandbox_dimension):
        df, r2, fit = func(agg)
        assert df == 0.0 and r2 == 0.0 and fit == {}


from pyFracAggregate.analysis.sandbox import plot_sandbox


def test_plot_sandbox_saves_figure(agg_poly, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out = tmp_path / "sandbox.png"
    plot_sandbox(
        agg_poly, show_fit=True, reference_df=1.8,
        measure="both", save_path=str(out),
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_sandbox_single_measure(agg_mono, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for measure in ("num", "mass"):
        out = tmp_path / f"sandbox_{measure}.png"
        plot_sandbox(agg_mono, measure=measure, save_path=str(out))
        assert out.exists()


def test_plot_sandbox_too_few_particles(tmp_path, capsys):
    agg = Aggregate(1)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    plot_sandbox(agg, save_path=str(tmp_path / "x.png"))
    assert "too few particles" in capsys.readouterr().out


def test_plot_sandbox_rejects_unknown_measure(agg_mono):
    with pytest.raises(ValueError, match="measure"):
        plot_sandbox(agg_mono, measure="nmu")


def test_plot_sandbox_reference_slope_is_df_itself(agg_poly, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    plot_sandbox(agg_poly, show_fit=True, reference_df=1.8, measure="both",
                 save_path=str(tmp_path / "s.png"))
    ax = plt.gca()
    ref = [ln for ln in ax.lines if ln.get_label().startswith("Ref")]
    assert len(ref) == 1
    x, y = ref[0].get_xdata(), ref[0].get_ydata()
    slope = np.polyfit(np.log10(x), np.log10(y), 1)[0]
    assert slope == pytest.approx(1.8, abs=1e-6)   # cumulative: no -3
    plt.close("all")
