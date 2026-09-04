"""Idempotent benchmark runner with resume-by-key semantics."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

if __package__ in (None, ""):  # executed as a script: put repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import time
import numpy as np

import pyFracAggregate as pfa
from benchmarks.grids import build_grid, sort_runs, make_dist, ROW_FIELDS
from benchmarks.metrics import evaluate_run

# rough per-run seconds for --dry-run estimates: (method, placement) -> {N: s}
_COST = {
    ("pca", "solved"): {50: 0.4, 100: 1.0, 500: 15.0, 1024: 90.0},
    ("pca", "sampled"): {50: 1.0, 100: 2.5, 500: 40.0, 1024: 250.0},
    ("cca", "solved"): {50: 2.0, 100: 7.5, 500: 90.0, 1024: 450.0},
    ("cca", "sampled"): {50: 3.0, 100: 10.0, 500: 120.0, 1024: 600.0},
    ("cca", "constructed"): {16: 0.2, 100: 5.0, 400: 21.0, 500: 25.0, 1024: 150.0},
}

TINY_GRID = [  # test-only grid, ~1 s total
    {"exp": 1, "method": "pca", "placement": "solved", "beta": None,
     "N": 20, "df": 1.8, "kf": 1.3, "sg": 1.0, "seed": 0},
    {"exp": 1, "method": "cca", "placement": "constructed", "beta": None,
     "N": 16, "df": 1.8, "kf": 1.3, "sg": 1.0, "seed": 0},
]


def estimate_seconds(row) -> float:
    d = _COST.get((row["method"], row["placement"]), {})
    return float(d.get(row["N"], 1.0))


def _key(row, tier):
    return (tier, row["exp"], row["method"], row["placement"], row["beta"],
            row["N"], row["df"], row["kf"], row["sg"], row["seed"])


def _existing_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            keys.add((r["tier"], int(r["exp"]), r["method"], r["placement"],
                      None if r["beta"] in ("", "None") else float(r["beta"]),
                      int(r["N"]), float(r["df"]), float(r["kf"]),
                      float(r["sg"]), int(r["seed"])))
    return keys


def execute(rows, csv_path, tier, max_minutes=None):
    """Run all rows not already in csv_path. Returns counter dict."""
    rows = sort_runs(list(rows))
    done_keys = _existing_keys(csv_path)
    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
    if new_file:
        w.writeheader()
    counters = {"done": 0, "skipped": 0, "failed": 0, "stopped_early": False}
    t0 = time.monotonic()
    try:
        for row in rows:
            if max_minutes is not None and \
                    (time.monotonic() - t0) / 60.0 > max_minutes:
                counters["stopped_early"] = True
                break
            k = _key(row, tier)
            if k in done_keys:
                counters["skipped"] += 1
                continue
            out = dict(tier=tier, **{k2: row[k2] for k2 in
                                     ("exp", "method", "placement", "beta",
                                      "N", "df", "kf", "sg", "seed")})
            try:
                kwargs = {"seed": row["seed"]}
                if row["beta"] is not None:
                    kwargs["surface_beta"] = row["beta"]
                if row["placement"] == "constructed":
                    kwargs["scaling"] = "mass"
                t1 = time.perf_counter()
                agg = pfa.generate(
                    n_particles=row["N"], df=row["df"], kf=row["kf"],
                    method=row["method"], particle_dist=make_dist(row["sg"]),
                    placement=row["placement"], **kwargs)
                elapsed = time.perf_counter() - t1
                out.update(evaluate_run(row, agg, elapsed))
                out.update(time_s=elapsed, status="ok", err_type="")
            except Exception as e:  # noqa: BLE001 — record and continue
                out.update(status="failed", err_type=type(e).__name__,
                           time_s="", df_est="", df_est_r2="", fit_npts="",
                           norm_err="", rg="", rg_target="", ks_d="",
                           overlap_worst="")
                counters["failed"] += 1
            else:
                counters["done"] += 1
            w.writerow(out)
            f.flush()
            print(f"RUN|{out['method']}|{out['placement']}|N={row['N']}"
                  f"|df={row['df']}|kf={row['kf']}|sg={row['sg']}"
                  f"|seed={row['seed']}|beta={row['beta']}"
                  f"|{out['status']}|{out['time_s']}", flush=True)
    finally:
        f.close()
    return counters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--exp", choices=["all", "1", "3"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="graceful stop after N minutes (smoke fuse)")
    ap.add_argument("--probe", action="store_true",
                    help="single cca@N=1024 probe run")
    ap.add_argument("--random-full", action="store_true",
                    help="full tier: also random placement at N=1024")
    args = ap.parse_args()
    if args.probe:
        rows = [{"exp": 1, "method": "cca", "placement": "solved",
                 "beta": None, "N": 1024, "df": 1.8, "kf": 1.3,
                 "sg": 1.0, "seed": 0}]
        tier = "probe"
    else:
        rows = build_grid(args.tier, exp=args.exp,
                          random_full=args.random_full)
        tier = args.tier
    total = sum(estimate_seconds(r) for r in rows)
    print(f"tier={tier} rows={len(rows)} est_total={total / 60:.1f} min")
    if args.dry_run:
        for r in sort_runs(rows):
            print(f"  {r['exp']}|{r['method']}|{r['placement']}|N={r['N']}"
                  f"|df={r['df']}|kf={r['kf']}|sg={r['sg']}|seed={r['seed']}"
                  f"|beta={r['beta']}|~{estimate_seconds(r):.1f}s")
        return
    os.makedirs("benchmarks/results", exist_ok=True)
    c = execute(rows, f"benchmarks/results/runs_{tier}.csv", tier,
                max_minutes=args.max_minutes)
    print(f"SUMMARY|done={c['done']}|skipped={c['skipped']}"
          f"|failed={c['failed']}|stopped_early={c['stopped_early']}")


if __name__ == "__main__":
    main()
