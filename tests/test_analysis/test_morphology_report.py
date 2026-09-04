import dataclasses
import numpy as np
import pytest
import pyFracAggregate as pfa


@pytest.fixture(scope="module")
def agg():
    return pfa.generate(50, 1.8, 1.3, method="pca", seed=42)


def test_analyze_returns_report(agg):
    report = pfa.analyze(agg)
    assert isinstance(report, pfa.MorphologyReport)
    assert isinstance(report.rg, float) and report.rg > 0
    assert 1.0 < report.df_est < 3.0
    assert 0.0 <= report.r2 <= 1.0
    assert report.r_centers.shape == report.pair_correlation.shape
    assert report.com.shape == (3,)
    assert report.n == 50


def test_report_is_asdictable(agg):
    d = dataclasses.asdict(pfa.analyze(agg))
    assert set(d) == {"rg", "df_est", "r2", "r_centers",
                      "pair_correlation", "com", "n"}


def test_legacy_dict_access_pattern_gone(agg):
    report = pfa.analyze(agg)
    with pytest.raises(TypeError):
        report["Rg"]
