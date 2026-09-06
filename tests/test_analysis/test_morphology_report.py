import dataclasses

import numpy as np
import pytest

import pyFracAggregate as pfa

FIELDS = {
    "rg", "com", "n", "estimator",
    "df_num_est", "r2_num", "r_num", "num_correlation",
    "df_mass_est", "r2_mass", "r_mass", "mass_correlation",
}


@pytest.fixture(scope="module")
def agg():
    return pfa.generate(50, 1.8, 1.3, method="pca", seed=42)


def test_analyze_default_is_sandbox(agg):
    report = pfa.analyze(agg)
    assert isinstance(report, pfa.MorphologyReport)
    assert report.estimator == "sandbox"
    assert isinstance(report.rg, float) and report.rg > 0
    assert report.com.shape == (3,)
    assert report.n == 50
    assert 1.0 < report.df_num_est < 3.0
    assert 1.0 < report.df_mass_est < 3.0
    assert 0.0 <= report.r2_num <= 1.0
    assert 0.0 <= report.r2_mass <= 1.0
    assert report.r_num.shape == report.num_correlation.shape == (15,)
    assert report.r_mass.shape == report.mass_correlation.shape == (15,)


def test_analyze_pcf_mode(agg):
    report = pfa.analyze(agg, estimator="pcf")
    assert report.estimator == "pcf"
    assert report.r_num.shape == report.num_correlation.shape == (50,)
    assert report.r_mass.shape == report.mass_correlation.shape == (50,)
    sandbox = pfa.analyze(agg)
    assert report.df_num_est != sandbox.df_num_est  # different estimators


def test_analyze_rejects_unknown_estimator(agg):
    with pytest.raises(ValueError, match="estimator"):
        pfa.analyze(agg, estimator="box")


def test_report_field_set(agg):
    d = dataclasses.asdict(pfa.analyze(agg))
    assert set(d) == FIELDS


def test_monodisperse_report_measures_coincide(agg):
    # monodisperse primaries -> mass weights constant -> sandbox estimates equal
    report = pfa.analyze(agg)
    assert report.df_num_est == pytest.approx(report.df_mass_est, abs=1e-10)
    assert report.r2_num == pytest.approx(report.r2_mass, abs=1e-10)


def test_new_functions_exported_at_top_level():
    for name in (
        "number_radius_function", "mass_radius_function",
        "number_sandbox_dimension", "mass_sandbox_dimension",
        "mass_pair_correlation_function", "plot_sandbox",
    ):
        assert callable(getattr(pfa, name))
        assert name in pfa.__all__


def test_version_bumped():
    assert pfa.__version__ == "0.6.1"


def test_legacy_dict_access_pattern_gone(agg):
    report = pfa.analyze(agg)
    with pytest.raises(TypeError):
        report["Rg"]
