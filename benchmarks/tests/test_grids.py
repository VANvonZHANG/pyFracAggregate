"""Grid declaration tests: smoke subset of full (v0.4 coordinates)."""
import math
from benchmarks.grids import build_grid, ROW_FIELDS, sort_runs, make_dist


def key(r):
    return (r["exp"], r["method"], r["placement"], r["beta"], r["N"],
            r["df"], r["kf"], r["sg"], r["seed"])


def test_smoke_rowcount():
    # 4 pairs x 2 N x 2 seeds x 5 (pca-solved/pca-sampled/cca-solved/
    # cca-sampled/cca-constructed) = 80
    # + poly 3 rows + beta 7 x 2 seeds = 14
    assert len(build_grid("smoke")) == 80 + 3 + 14


def test_full_rowcount_default():
    # mono: 4 pairs x 4 N x 10 seeds x 3 = 480; sampled(N<=500): 4 x 3 x 10 x 2 = 240
    # poly: (2+2+1) x 10 = 50; beta: 7 x 2 x 10 = 140
    assert len(build_grid("full")) == 480 + 240 + 50 + 140


def test_full_rowcount_random_full():
    assert len(build_grid("full", random_full=True)) == \
        len(build_grid("full")) + 4 * 1 * 10 * 2  # sampled at N=1024 added back


def test_smoke_subset_of_full():
    full_keys = {key(r) for r in build_grid("full")}
    for r in build_grid("smoke"):
        assert key(r) in full_keys, f"smoke row not in full: {r}"



def test_beta_grid():
    rows = build_grid("full", exp="3")
    assert {r["beta"] for r in rows} == {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
    assert all(r["method"] == "cca" and r["placement"] == "solved"
               and r["N"] in (100, 500) for r in rows)


def test_exp_filter():
    assert all(r["exp"] == 3 for r in build_grid("smoke", exp="3"))


def test_row_fields_match_spec_keys():
    for r in build_grid("smoke"):
        for k in r:
            assert k in ROW_FIELDS


def test_make_dist():
    from pyFracAggregate import Monodisperse, LognormalDistribution
    assert isinstance(make_dist(1.0), Monodisperse)
    d = make_dist(2.0)
    assert isinstance(d, LognormalDistribution)
    assert d.std == 2.0  # geometric std semantics


def test_sort_runs_cheap_first():
    rows = build_grid("smoke")
    out = sort_runs(rows)
    coords = [(r["method"], r["placement"]) for r in out]
    # constructed cca (rank 10) runs before plain cca/solved (rank 25)
    assert coords.index(("cca", "constructed")) < coords.index(("cca", "solved"))
    # beta (exp3) after all N<=500 mono pca/constructed:
    i_beta = min(i for i, r in enumerate(out) if r["exp"] == 3)
    i_pca100 = max(i for i, r in enumerate(out)
                   if r["method"] == "pca" and r["N"] <= 100)
    assert i_pca100 < i_beta
