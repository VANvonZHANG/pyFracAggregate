"""Runner tests: dry-run listing, idempotent resume, failure rows."""
import csv
import numpy as np
import pytest
from benchmarks.run import execute, TINY_GRID, estimate_seconds


def test_tiny_grid_idempotent(tmp_path):
    csv_path = tmp_path / "runs_smoke.csv"
    c1 = execute(TINY_GRID, str(csv_path), tier="smoke")
    assert c1["done"] == len(TINY_GRID) and c1["skipped"] == 0
    c2 = execute(TINY_GRID, str(csv_path), tier="smoke")
    assert c2["done"] == 0 and c2["skipped"] == len(TINY_GRID)
    with open(csv_path) as f:
        assert len(list(csv.reader(f))) == 1 + len(TINY_GRID)  # header + rows


def test_failure_row_recorded(tmp_path):
    csv_path = tmp_path / "runs_smoke.csv"
    bad = [dict(TINY_GRID[0], sg=-2.0)]  # LognormalDistribution raises ValueError
    c = execute(bad, str(csv_path), tier="smoke")
    assert c["failed"] == 1
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "failed" and rows[0]["err_type"]


def test_estimate_seconds_covers_all_smoke_rows():
    from benchmarks.grids import build_grid
    for r in build_grid("smoke"):
        assert np.isfinite(estimate_seconds(r))


def test_max_minutes_stops_early(tmp_path):
    csv_path = tmp_path / "runs_smoke.csv"
    c = execute(TINY_GRID, str(csv_path), tier="smoke", max_minutes=-1.0)
    assert c["done"] == 0 and c["stopped_early"] is True
