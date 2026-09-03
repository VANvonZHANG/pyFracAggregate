"""Analyze tests on a tiny synthetic CSV (no generation runs)."""
import csv
from benchmarks.analyze import analyze


HEADER = ["tier", "exp", "method", "placement", "beta", "N", "df", "kf", "sg",
          "seed", "time_s", "status", "err_type", "df_est", "df_est_r2",
          "fit_npts", "norm_err", "rg", "rg_target", "ks_d", "overlap_worst"]


def _write_csv(tmp_path):
    p = tmp_path / "runs_smoke.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for seed in (0, 1):
            for (m, N, df) in (("pca", 50, 1.8), ("cca", 50, 1.8),
                               ("tdcca", 128, 1.8), ("fracval", 50, 1.8)):
                w.writerow(["smoke", 1, m, "algebraic", "", N, df, 1.3, 1.0,
                            seed, 1.5 + 0.1 * seed, "ok", "",
                            df - 0.02 + 0.01 * seed, 0.98, 20, 0.01,
                            5.0, 5.1, "", 0.0])
            w.writerow(["smoke", 3, "cca", "algebraic", 0.3, 100, 1.8, 1.3,
                        1.0, seed, 7.0, "ok", "", 1.79, 0.97, 20, 0.01,
                        9.0, 9.1, "", 0.0])
    return p


def test_analyze_outputs(tmp_path):
    csv_path = _write_csv(tmp_path)
    outdir = tmp_path / "out"
    figs = outdir / "figs"
    analyze(str(csv_path), outdir=str(outdir), watermark="SMOKE")
    for stem in ("fig_df_vs_target", "fig_scaling_law", "fig_timing",
                 "fig_beta_tradeoff"):
        assert (figs / f"{stem}.png").exists()
        assert (figs / f"{stem}.pdf").exists()
    for stem in ("tab_multiseed", "tab_timing"):
        assert (outdir / f"{stem}.tex").exists()
        assert (outdir / f"{stem}.csv").exists()
    tex = (outdir / "tab_multiseed.tex").read_text()
    assert "pca" in tex and "cca" in tex and "tdcca" in tex
