"""Aggregate runs CSV -> paper figures (PDF+PNG) and LaTeX tables."""
import os
import sys

if __package__ in (None, ""):  # executed as a script: put repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            for k in ("beta", "N", "df", "kf", "sg", "seed", "time_s",
                      "df_est", "df_est_r2", "rg", "rg_target"):
                r[k] = float(r[k]) if r[k] != "" else float("nan")
            rows.append(r)
    return rows


def _mean_std(vals):
    v = [x for x in vals if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0, len(v)
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var), len(v)


def _wm(fig, watermark):
    if watermark:
        fig.text(0.5, 0.5, watermark, alpha=0.12, fontsize=56,
                 rotation=30, ha="center", va="center")


def _groupby(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def analyze(csv_path, outdir="benchmarks/results", watermark=None):
    figs = os.path.join(outdir, "figs")
    os.makedirs(figs, exist_ok=True)
    rows = _load(csv_path)
    exp1 = [r for r in rows if r["exp"] == "1"]
    exp3 = [r for r in rows if r["exp"] == "3"]

    # ---- Fig 1: Df fidelity (x = target Df per pair, y = Df_est ± std)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ("pca", "cca", "fracval", "tdcca")):
        groups = defaultdict(list)
        for r in exp1:
            if r["method"] == m and r["placement"] == "algebraic":
                groups[(r["df"], r["N"])].append(r["df_est"])
        for N in sorted({k[1] for k in groups}):
            xs, ys, es = [], [], []
            for (df, n), vals in sorted(groups.items()):
                if n != N:
                    continue
                mean, std, _ = _mean_std(vals)
                xs.append(df); ys.append(mean); es.append(std)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                        label=f"N={int(N)}")
        ax.plot([1.3, 2.5], [1.3, 2.5], "k--", lw=0.8, label="1:1")
        ax.set_title(m); ax.set_xlabel("target $D_f$")
    axes[0][0].set_ylabel("$D_{f,est}$")
    axes[0][0].legend(fontsize=8)
    _wm(fig, watermark)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig_df_vs_target.png"), dpi=200)
    fig.savefig(os.path.join(figs, "fig_df_vs_target.pdf"))
    plt.close(fig)

    # ---- Fig 2: scaling law log N vs log(Rg/r_mean)
    fig, ax = plt.subplots(figsize=(6, 5))
    for (df, kf), grp in _groupby(exp1, ("df", "kf")).items():
        pts = defaultdict(list)
        for r in grp:
            pts[r["N"]].append(math.log10(r["rg"] / 1.0))
        xs = sorted(pts)
        ys = [_mean_std(pts[x])[0] for x in xs]
        ax.plot([math.log10(x) for x in xs], ys, "o-", ms=3,
                label=f"Df={df},kf={kf}")
    ax.set_xlabel("log10 N"); ax.set_ylabel("log10 Rg")
    ax.legend(fontsize=7)
    _wm(fig, watermark)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig_scaling_law.png"), dpi=200)
    fig.savefig(os.path.join(figs, "fig_scaling_law.pdf"))
    plt.close(fig)

    # ---- Fig 3: timing vs N per method x placement
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (m, p), grp in _groupby(exp1, ("method", "placement")).items():
        pts = defaultdict(list)
        for r in grp:
            pts[r["N"]].append(r["time_s"])
        xs = sorted(pts)
        ys = [_mean_std(pts[x])[0] for x in xs]
        ax.loglog(xs, ys, "o-", ms=3, label=f"{m}/{p}")
    ax.set_xlabel("N"); ax.set_ylabel("generation time [s]")
    ax.legend(fontsize=7)
    _wm(fig, watermark)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig_timing.png"), dpi=200)
    fig.savefig(os.path.join(figs, "fig_timing.pdf"))
    plt.close(fig)

    # ---- Fig 4: beta trade-off (two panels)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    by_beta = defaultdict(list)
    for r in exp3:
        by_beta[(r["beta"], r["N"])].append(r)
    betas = sorted({b for b, _ in by_beta})
    for N in sorted({n for _, n in by_beta}):
        t = [_mean_std([r["time_s"] for r in by_beta[(b, N)]])[0]
             for b in betas]
        d = [_mean_std([abs(r["df_est"] - r["df"]) for r in by_beta[(b, N)]])[0]
             for b in betas]
        a1.plot(betas, t, "o-", label=f"N={int(N)}")
        a2.plot(betas, d, "o-", label=f"N={int(N)}")
    a1.set_xlabel(r"$\beta$"); a1.set_ylabel("time [s]"); a1.legend(fontsize=8)
    a2.set_xlabel(r"$\beta$")
    a2.set_ylabel(r"$|D_{f,est}-D_f|$"); a2.set_yscale("log")
    _wm(fig, watermark)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig_beta_tradeoff.png"), dpi=200)
    fig.savefig(os.path.join(figs, "fig_beta_tradeoff.pdf"))
    plt.close(fig)

    # ---- Tables
    _write_tables(exp1, outdir)


def _write_tables(exp1, outdir):
    # tab_multiseed: rows method x (df,kf), cols N, cells Df_est±std (R2)
    cells = defaultdict(list)
    for r in exp1:
        cells[(r["method"], r["df"], r["kf"], r["N"])].append(r["df_est"])
    Ns = sorted({k[3] for k in cells})
    lines = ["\\begin{tabular}{ll" + "c" * len(Ns) + "}", "\\toprule",
             "Method & $(D_f,k_f)$ & " + " & ".join(f"N={int(n)}" for n in Ns) + " \\\\",
             "\\midrule"]
    csv_lines = ["method,df,kf," + ",".join(f"N{int(n)}" for n in Ns)]
    for (m, df, kf) in sorted({(k[0], k[1], k[2]) for k in cells}):
        row_tex = [f"{m} & ({df},{kf})"]
        row_csv = [m, str(df), str(kf)]
        for n in Ns:
            mean, std, cnt = _mean_std(cells.get((m, df, kf, n), []))
            if cnt == 0:
                row_tex.append("--"); row_csv.append("")
            else:
                row_tex.append(f"{mean:.3f}$\\pm${std:.3f}")
                row_csv.append(f"{mean:.3f}±{std:.3f}")
        lines.append(" & ".join(row_tex) + " \\\\")
        csv_lines.append(",".join(row_csv))
    lines += ["\\bottomrule", "\\end{tabular}"]
    _dump(os.path.join(outdir, "tab_multiseed.tex"), "\n".join(lines))
    _dump(os.path.join(outdir, "tab_multiseed.csv"), "\n".join(csv_lines))

    # tab_timing: rows method/placement, cols N, cells mean s
    tcells = defaultdict(list)
    for r in exp1:
        tcells[(r["method"], r["placement"], r["N"])].append(r["time_s"])
    lines = ["\\begin{tabular}{ll" + "c" * len(Ns) + "}", "\\toprule",
             "Method & Placement & " + " & ".join(f"N={int(n)}" for n in Ns) + " \\\\",
             "\\midrule"]
    for (m, p) in sorted({(k[0], k[1]) for k in tcells}):
        row = [f"{m} & {p}"]
        for n in Ns:
            mean, _, cnt = _mean_std(tcells.get((m, p, n), []))
            row.append("--" if cnt == 0 else f"{mean:.2f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _dump(os.path.join(outdir, "tab_timing.tex"), "\n".join(lines))
    csv_lines = ["method,placement," + ",".join(f"N{int(n)}" for n in Ns)]
    for (m, p) in sorted({(k[0], k[1]) for k in tcells}):
        row = [m, p]
        for n in Ns:
            mean, _, cnt = _mean_std(tcells.get((m, p, n), []))
            row.append("" if cnt == 0 else f"{mean:.2f}")
        csv_lines.append(",".join(row))
    _dump(os.path.join(outdir, "tab_timing.csv"), "\n".join(csv_lines))


def _dump(path, text):
    with open(path, "w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["smoke", "full", "probe"], default="smoke")
    ap.add_argument("--outdir", default="benchmarks/results")
    ap.add_argument("--watermark", default=None,
                    help="e.g. SMOKE; default none for full tier")
    args = ap.parse_args()
    watermark = args.watermark
    if watermark is None and args.tier == "smoke":
        watermark = "SMOKE"
    analyze(os.path.join(args.outdir, f"runs_{args.tier}.csv"),
            outdir=args.outdir, watermark=watermark)


if __name__ == "__main__":
    main()
