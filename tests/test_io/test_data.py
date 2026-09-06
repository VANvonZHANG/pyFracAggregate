import os
import yaml
import numpy as np
import pytest
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.io.data import export_yaml


def _make_aggregate(n=3):
    agg = Aggregate(max_particles=n)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    agg.add_particle(2.0, 0.0, 0.0, 1.0, 1.0)
    if n >= 3:
        agg.add_particle(4.0, 0.0, 0.0, 1.0, 1.0)
    return agg


def test_export_yaml_basic(tmp_path):
    agg = _make_aggregate(3)
    path = str(tmp_path / "basic.yaml")
    export_yaml(agg, path)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert "aggregate" in data
    assert data["aggregate"]["n_particles"] == 3
    assert len(data["aggregate"]["positions"]) == 3
    assert isinstance(data["aggregate"]["positions"][0], list)
    assert data["aggregate"]["density"] == 1.0
    assert data["aggregate"]["length_unit"] == "nm"


def test_export_yaml_with_generation_params(tmp_path):
    agg = _make_aggregate(2)
    params = {"method": "pca", "n_particles": 2, "df": 1.8, "kf": 1.2, "placement": "algebraic"}
    path = str(tmp_path / "with_params.yaml")
    export_yaml(agg, path, generation_params=params)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert data["generation"]["method"] == "pca"
    assert data["generation"]["df"] == 1.8
    assert "aggregate" in data
    assert "analysis" not in data


def test_export_yaml_with_analysis_results(tmp_path):
    agg = _make_aggregate(2)
    analysis = {"Rg": 5.0, "center_of_mass": [1.0, 0.0, 0.0], "fractal_dimension": 1.8}
    path = str(tmp_path / "with_analysis.yaml")
    export_yaml(agg, path, analysis_results=analysis)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert data["analysis"]["Rg"] == 5.0
    assert data["analysis"]["center_of_mass"] == [1.0, 0.0, 0.0]
    assert "aggregate" in data
    assert "generation" not in data


def test_export_yaml_all_sections(tmp_path):
    agg = _make_aggregate(2)
    params = {"method": "cca", "n_particles": 2, "df": 2.0, "kf": 1.0}
    analysis = {"Rg": 3.0}
    path = str(tmp_path / "all.yaml")
    export_yaml(agg, path, generation_params=params, analysis_results=analysis)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert "generation" in data
    assert "aggregate" in data
    assert "analysis" in data


def test_export_yaml_numpy_arrays_converted(tmp_path):
    """NumPy types must be serializable (no numpy.float64 in output)."""
    agg = _make_aggregate(2)
    path = str(tmp_path / "numpy_safe.yaml")
    # Should not raise
    export_yaml(agg, path)


def test_export_yaml_accepts_morphology_report_with_v06_keys(tmp_path):
    import pyFracAggregate as pfa
    agg = pfa.generate(30, 1.8, 1.3, method="pca", seed=7)
    report = pfa.analyze(agg)
    path = str(tmp_path / "report.yaml")
    export_yaml(agg, path, analysis_results=report)
    with open(path) as f:
        data = yaml.safe_load(f)
    # v0.6 snapshot keys
    assert data["analysis"]["Rg"] == pytest.approx(report.rg)
    assert data["analysis"]["CoM"] == pytest.approx(report.com.tolist())
    assert data["analysis"]["N"] == 30
    assert data["analysis"]["estimator"] == "sandbox"
    assert data["analysis"]["Df_num_estimated"] == pytest.approx(report.df_num_est)
    assert data["analysis"]["R2_num"] == pytest.approx(report.r2_num)
    assert data["analysis"]["Df_mass_estimated"] == pytest.approx(report.df_mass_est)
    assert data["analysis"]["R2_mass"] == pytest.approx(report.r2_mass)
    assert len(data["analysis"]["r_num"]) == len(report.r_num)
    assert len(data["analysis"]["num_correlation"]) == len(report.num_correlation)
    assert len(data["analysis"]["r_mass"]) == len(report.r_mass)
    assert len(data["analysis"]["mass_correlation"]) == len(report.mass_correlation)
    # pre-0.6 keys are gone (breaking change, spec D2/D7)
    for old_key in ("Df_estimated", "R2", "r_centers", "pair_correlation"):
        assert old_key not in data["analysis"]


def test_export_yaml_accepts_plain_dict_still(tmp_path):
    agg = _make_aggregate(2)
    path = str(tmp_path / "plain.yaml")
    export_yaml(agg, path, analysis_results={"Rg": 3.0})
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["analysis"]["Rg"] == 3.0
