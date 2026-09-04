"""Grid declarations for the paper experiments (see spec Sec. 2).

v0.4 coordinate system: method pca|cca, placement solved|sampled|constructed
(the former "fracval" experiment is (cca, mass, constructed); tdcca rows were
removed with the algorithm in v0.4)."""
from pyFracAggregate import Monodisperse, LognormalDistribution

PAIRS = [(1.40, 1.80), (1.79, 1.40), (2.40, 0.80), (1.8, 1.3)]
FULL_N = (50, 100, 500, 1024)
SMOKE_N = (50, 100)
SEEDS_FULL = range(10)
SEEDS_SMOKE = (0, 1)
BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
# poly rows: (df, kf, sg, (N values)) — FracVAL Sec 4.2.2 pairs + patent point
POLY_ROWS = [
    (1.68, 1.4, 2.0, (100, 1024)),
    (1.48, 1.8, 3.0, (100, 1024)),
    (1.8, 1.3, 1.5, (400,)),
]

ROW_FIELDS = ["tier", "exp", "method", "placement", "beta", "N", "df", "kf",
              "sg", "seed", "time_s", "status", "err_type", "df_est",
              "df_est_r2", "fit_npts", "norm_err", "rg", "rg_target",
              "ks_d", "overlap_worst"]


def make_dist(sg: float):
    """Monodisperse for sg == 1.0; else Lognormal(r_geo=1, sg) (geometric)."""
    return Monodisperse(1.0) if sg == 1.0 else LognormalDistribution(1.0, sg)


def _row(exp, method, placement, beta, N, df, kf, sg, seed):
    return {"exp": exp, "method": method, "placement": placement,
            "beta": beta, "N": N, "df": df, "kf": kf, "sg": sg, "seed": seed}


def build_grid(tier: str, exp: str = "all", random_full: bool = False):
    """Build the run list. tier in {'smoke','full'}; exp in {'all','1','3'}."""
    smoke = tier == "smoke"
    seeds = SEEDS_SMOKE if smoke else SEEDS_FULL
    n_mono = SMOKE_N if smoke else FULL_N
    rows = []
    if exp in ("all", "1"):
        for df, kf in PAIRS:
            for N in n_mono:
                for s in seeds:
                    rows.append(_row(1, "pca", "solved", None, N, df, kf, 1.0, s))
                    rows.append(_row(1, "cca", "solved", None, N, df, kf, 1.0, s))
                    rows.append(_row(1, "cca", "constructed", None, N, df, kf, 1.0, s))
                    if N <= 500 or random_full:
                        rows.append(_row(1, "pca", "sampled", None, N, df, kf, 1.0, s))
                        rows.append(_row(1, "cca", "sampled", None, N, df, kf, 1.0, s))
        # poly rows (constructed only)
        for df, kf, sg, nvals in POLY_ROWS:
            for N in nvals:
                use = (N == min(nvals)) if smoke else True
                if not use:
                    continue
                svals = (0,) if smoke else seeds
                for s in svals:
                    rows.append(_row(1, "cca", "constructed", None, N, df, kf, sg, s))
    if exp in ("all", "3"):
        beta_N = (100,) if smoke else (100, 500)
        for b in BETAS:
            for N in beta_N:
                for s in seeds:
                    rows.append(_row(3, "cca", "solved", b, N, 1.8, 1.3, 1.0, s))
    return rows


def _rank(r: dict) -> int:
    """Cheap-first execution order (spec Sec. 3.3)."""
    if r["exp"] == 3:
        return 20
    m, p, N, sg = r["method"], r["placement"], r["N"], r["sg"]
    if m == "pca":
        return (5 if N <= 500 else 30) + (0 if p == "solved" else 1)
    if m == "cca" and p == "constructed":
        if sg > 1.0:
            return 15  # poly rows (incl. N=1024) before beta sweep, per spec order
        return 10 if N <= 500 else 35
    if m == "cca":
        return (25 if N <= 500 else 40) + (0 if p == "solved" else 1)
    raise ValueError(r)


def sort_runs(rows):
    """Stable sort by cheap-first rank."""
    return sorted(rows, key=_rank)
