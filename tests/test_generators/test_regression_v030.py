"""Bit-identity regression against recorded v0.3.0 outputs.

These gates hold only while generation still draws from the global legacy
numpy.random stream (deviation N9). The seed sweep in Task 7 switches the
statistical anchors in test_anchors.py to the front.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import pyFracAggregate as pfa

FIX = Path(__file__).resolve().parents[1] / "fixtures"
SNAP = np.load(FIX / "baseline_v030.npz")
STATS = json.loads((FIX / "baseline_v030_stats.json").read_text())
N, DF, KF = 100, 1.8, 1.3


def _regenerate(case_kwargs):
    np.random.seed(0)
    return pfa.generate(n_particles=N, df=DF, kf=KF, **case_kwargs)


def test_pca_default_bit_identical():
    agg = _regenerate(dict(method="pca"))
    assert np.array_equal(agg.positions, SNAP["pca_default"])


def test_pca_random_bit_identical():
    agg = _regenerate(dict(method="pca", placement="random"))
    assert np.array_equal(agg.positions, SNAP["pca_random"])
