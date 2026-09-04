"""Record v0.3.0 generation outputs as regression anchors.

Run from repo root:  python tests/tools/record_baseline.py
Writes tests/fixtures/baseline_v030.npz (positions per case, np.random.seed(0))
and baseline_v030_stats.json (analyze() stats per case, seed 0).

Note: uses Path.resolve().parents[n] (the brief's Path.resolve_parents[n] is
Python 3.14+ only; this environment is Python 3.13.9). Fully equivalent.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pyFracAggregate as pfa  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

CASES = {
    "pca_default":   dict(method="pca"),
    "pca_random":    dict(method="pca", placement="random"),
    "cca_default":   dict(method="cca"),
    "cca_random":    dict(method="cca", placement="random"),
    "cca_poly":      dict(method="cca", particle_dist=pfa.LognormalDistribution(1.0, 1.6)),
    "fracval_mono":  dict(method="fracval"),
    "fracval_poly":  dict(method="fracval", particle_dist=pfa.LognormalDistribution(1.0, 1.6)),
}
N = 100
DF, KF = 1.8, 1.3

positions, stats = {}, {}
for name, kwargs in CASES.items():
    np.random.seed(0)
    agg = pfa.generate(n_particles=N, df=DF, kf=KF, **kwargs)
    assert agg.current_size == N
    positions[name] = agg.positions.copy()
    np.random.seed(0)
    agg2 = pfa.generate(n_particles=N, df=DF, kf=KF, **kwargs)
    assert np.array_equal(agg.positions, agg2.positions), f"{name} not seed-deterministic"
    res = pfa.analyze(agg)
    stats[name] = {"rg": float(res["Rg"]), "df_est": float(res["Df_estimated"]),
                   "r2": float(res["R2"])}

FIXTURES.mkdir(exist_ok=True)
np.savez(FIXTURES / "baseline_v030.npz", **positions)
(FIXTURES / "baseline_v030_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
