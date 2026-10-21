"""
research/plots.py
-----------------
Publication figures generated ONLY from the aggregated tables written by
``research.aggregate`` (never from hard-coded numbers).

python -m research.plots [--out results/v2] [--checkpoint selected]

Error bars: mean ± one sample standard deviation over seeds; dots: individual
seeds.  n (number of seeds) is printed in every figure.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from research import config as C

# Validated categorical slots (dataviz reference palette, light mode); at most
# three hues share one plot except the 4-line heterogeneity convergence plot.
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"
FAMILY = {"fedavg": BLUE, "centralized": ORANGE, "local": AQUA}
LABELS = {"centralized": "Centralized\n(budget-matched)", "centralized_legacy_es": "Centralized\n(original ES)",
          "fedavg_iid": "FedAvg\nIID window", "local_iid": "Local-only\nIID window",
          "fedavg_engine": "FedAvg\nwhole-engine", "local_engine": "Local-only\nwhole-engine"}
METRIC_LABEL = {"rmse": "Test RMSE (cycles)", "mae": "Test MAE (cycles)", "nasa_score": "NASA score (sum, 100 engines)"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.dpi": 300, "figure.dpi": 110,
                     "legend.frameon": False, "lines.linewidth": 1.6})


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _n_note(ns):
    ns = sorted(set(int(n) for n in ns))
    return f"n = {ns[0]} seeds" if len(ns) == 1 else f"n = {ns[0]}-{ns[-1]} seeds"


def main_comparison(ps, ck, out):
    exps = [e for e in LABELS if e in set(ps.experiment)]
    if not exps:
        return
    sub = ps[(ps.checkpoint == ck) & ps.experiment.isin(exps) & (ps.grid_label == "default")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, m in zip(axes, ("rmse", "mae", "nasa_score")):
        for i, e in enumerate(exps):
            v = sub[sub.experiment == e][m].dropna().values
            if len(v) == 0:
                continue
            col = FAMILY[sub[sub.experiment == e].method.iloc[0]]
            ax.bar(i, v.mean(), width=0.62, color=col, alpha=0.85 if "legacy" not in e else 0.45,
                   yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#222"})
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=9, color="#222", zorder=3)
        ax.set_xticks(range(len(exps)), [LABELS[e] for e in exps], fontsize=7, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel(METRIC_LABEL[m])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()]
    fig.legend(handles, ["FedAvg", "Centralized", "Local-only (mean of K=5 clients)"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06))
    fig.text(0.99, -0.02, f"mean ± SD, dots = seeds; {_n_note(sub.groupby('experiment').size())}; checkpoint = {ck}",
             ha="right", fontsize=7, color=GREY)
    fig.tight_layout()
    _save(fig, out, f"fig1_main_comparison_{ck}")


def convergence(cv, out):
    if cv.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    series = [("fedavg_iid", "default", "FedAvg IID window", BLUE),
              ("fedavg_engine", "default", "FedAvg whole-engine", AQUA),
              ("centralized", "default", "Centralized (per 5 epochs)", ORANGE)]
    ax = axes[0]
    for e, g, lab, col in series:
        s = cv[(cv.experiment == e) & (cv.grid_label == g)].copy()
        if s.empty:
            continue
        if s.step_type.iloc[0] == "epoch":
            s = s[s.step % 5 == 0].assign(step=lambda d: d.step // 5)
        ax.plot(s.step, s["mean"], color=col, label=f"{lab} (n={int(s['count'].max())})")
        ax.fill_between(s.step, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0), color=col, alpha=0.15, lw=0)
    ax.set_xlabel("Communication round (centralized: epoch / 5)")
    ax.set_ylabel("Validation RMSE (cycles)")
    ax.set_title("Same budget: R x E = 100 passes over the data", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    het = [("fedavg_iid", "default", "IID window", BLUE), ("fedavg_dirichlet", "dirichlet_alpha=1.0", "Dirichlet α=1.0", ORANGE),
           ("fedavg_dirichlet", "dirichlet_alpha=0.5", "Dirichlet α=0.5", AQUA), ("fedavg_dirichlet", "dirichlet_alpha=0.1", "Dirichlet α=0.1", YELLOW)]
    for e, g, lab, col in het:
        s = cv[(cv.experiment == e) & (cv.grid_label == g)]
        if s.empty:
            continue
        ax.plot(s.step, s["mean"], color=col, label=f"{lab} (n={int(s['count'].max())})")
        ax.fill_between(s.step, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0), color=col, alpha=0.12, lw=0)
    ax.set_xlabel("Communication round")
    ax.set_title("FedAvg under window-level label skew (K=5)", fontsize=9)
    ax.legend(fontsize=7)
    axes[0].set_ylim(10, 30)
    for a_ in axes:
        a_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.text(0.99, -0.03, "line = mean over seeds, band = ± SD; y-axis clipped at 30 (rounds 1-2 of FedAvg lie above)",
             ha="right", fontsize=7, color=GREY)
    fig.tight_layout()
    _save(fig, out, "fig2_convergence")


def heterogeneity(ps, ck, out):
    sub = ps[ps.checkpoint == ck]
    rows = [("IID\nwindow", sub[(sub.experiment == "fedavg_iid")]),
            *[(f"Dirichlet\nα={a}", sub[(sub.experiment == "fedavg_dirichlet") & (sub.alpha == a)]) for a in (1.0, 0.5, 0.1)],
            ("Whole-\nengine", sub[(sub.experiment == "fedavg_engine")])]
    rows = [(l, d) for l, d in rows if len(d)]
    if len(rows) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    for ax, m in zip(axes, ("rmse", "nasa_score")):
        for i, (lab, d) in enumerate(rows):
            v = d[m].values
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="o", color=BLUE, ms=6, capsize=3)
            ax.scatter(np.full(len(v), i) + 0.18, v, s=8, color=GREY)
        ax.set_xticks(range(len(rows)), [l for l, _ in rows], fontsize=7)
        ax.set_ylabel(METRIC_LABEL[m])
    fig.suptitle(f"FedAvg (K=5, R=20, E=5) by partition protocol; {_n_note([len(d) for _, d in rows])}; checkpoint={ck}", fontsize=9)
    fig.tight_layout()
    _save(fig, out, f"fig3_partitions_{ck}")


def sweep(ps, ck, out, exp, xkey, xlabel, name, extra=None):
    sub = ps[(ps.checkpoint == ck) & (ps.experiment.isin([exp] + ([extra[0]] if extra else [])))]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    groups = [(exp, "IID window", BLUE)] + ([(extra[0], extra[1], AQUA)] if extra else [])
    for ax, (m, lab) in zip(axes, (("rmse", METRIC_LABEL["rmse"]), ("wall_s", "Wall-clock time (s), clean runs only"),
                                   ("comm_total_MB", "Model payload, up + down (MB)"))):
        for e, gl, col in groups:
            d = sub[sub.experiment == e]
            if m == "wall_s" and "timing_clean" in d:
                d = d[d.timing_clean.astype(bool)]   # drop runs that shared the GPU or spanned a suspend
            if d.empty:
                continue
            g = d.groupby(xkey)[m].agg(["mean", "std", "count"]).reset_index().sort_values(xkey)
            ax.errorbar(g[xkey], g["mean"], yerr=g["std"].fillna(0), marker="o", ms=5, capsize=3, color=col,
                        label=f"{gl}" if extra else None)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(lab)
        ax.set_xticks(sorted(sub[xkey].dropna().unique()))
        if extra:
            ax.legend(fontsize=7)
    axes[2].set_title("serialized-payload estimate", fontsize=8, color=GREY)
    fig.text(0.99, -0.03, f"mean ± SD; {_n_note(sub.groupby(['experiment', xkey]).size())}; checkpoint={ck}",
             ha="right", fontsize=7, color=GREY)
    fig.tight_layout()
    _save(fig, out, name)


def comm_tradeoff(ps, ck, out):
    sub = ps[(ps.checkpoint == ck) & (ps.experiment == "sweep_rounds")]
    if sub.empty:
        return
    g = sub.groupby("num_rounds").agg(rmse=("rmse", "mean"), sd=("rmse", "std"), mb=("comm_total_MB", "mean"),
                                      up=("comm_upload_MB", "mean"), down=("comm_download_MB", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.errorbar(g.mb, g.rmse, yerr=g.sd.fillna(0), marker="o", color=BLUE, capsize=3)
    for _, r in g.iterrows():
        ax.annotate(f"R={int(r.num_rounds)}", (r.mb, r.rmse), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Cumulative model payload, upload + download (MB)\n(K=5, fp32, no protocol overhead; estimate)")
    ax.set_ylabel(METRIC_LABEL["rmse"])
    fig.tight_layout()
    _save(fig, out, f"fig7_accuracy_vs_communication_{ck}")


def per_engine(eng, out):
    if eng.empty:
        return
    exps = [e for e in LABELS if e in set(eng.experiment)]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    data = [eng[eng.experiment == e]["abs_err_selected"].values for e in exps]
    bp = ax.boxplot(data, widths=0.55, showfliers=True, patch_artist=True,
                    medianprops={"color": "#111"}, flierprops={"markersize": 2.5, "markerfacecolor": GREY, "markeredgecolor": GREY})
    for patch, e in zip(bp["boxes"], exps):
        fam = "fedavg" if e.startswith("fedavg") else "local" if e.startswith("local") else "centralized"
        patch.set_facecolor(FAMILY[fam]); patch.set_alpha(0.55)
    ax.set_xticks(range(1, len(exps) + 1), [LABELS[e] for e in exps], fontsize=7, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Absolute test error per engine (cycles)")
    ax.set_title("Per-engine error distribution, pooled over seeds (local-only: mean over clients)", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig8_per_engine_errors")


def partitions(parts, out, seed):
    if parts.empty:
        return
    conds = [("fedavg_iid", "default", "IID window"), ("fedavg_dirichlet", "dirichlet_alpha=1.0", "Dirichlet α=1.0"),
             ("fedavg_dirichlet", "dirichlet_alpha=0.5", "Dirichlet α=0.5"),
             ("fedavg_dirichlet", "dirichlet_alpha=0.1", "Dirichlet α=0.1"), ("fedavg_engine", "default", "Whole-engine")]
    conds = [c for c in conds if len(parts[(parts.experiment == c[0]) & (parts.grid_label == c[1]) & (parts.seed == seed)])]
    if not conds:
        return
    fig, axes = plt.subplots(2, len(conds), figsize=(2.3 * len(conds), 4.2), sharey="row")
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, (e, g, lab) in enumerate(conds):
        d = parts[(parts.experiment == e) & (parts.grid_label == g) & (parts.seed == seed)].sort_values("client_id")
        axes[0, j].bar(d.client_id, d.n_windows, color=BLUE, width=0.7)
        axes[0, j].set_title(lab, fontsize=8)
        bp = axes[1, j]
        bp.bar(d.client_id, d.rul_mean, color=AQUA, width=0.7, yerr=d.rul_std, capsize=2, error_kw={"elinewidth": 0.8})
        bp.set_xlabel("Client")
        for _, r in d.iterrows():
            axes[0, j].text(r.client_id, r.n_windows, f"{int(r.n_engines)} eng", ha="center", va="bottom", fontsize=6)
    axes[0, 0].set_ylabel("Training windows")
    axes[1, 0].set_ylabel("RUL label mean ± SD")
    fig.suptitle(f"Client partitions (seed {seed}); 'eng' = engines with ≥1 window on the client", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig9_client_partitions")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.DEFAULT_OUT_DIR)
    ap.add_argument("--checkpoint", default="selected", choices=["selected", "final", "both"])
    a = ap.parse_args(argv)
    t = os.path.join(a.out, "tables")
    fdir = os.path.join(a.out, "figures")
    ps = pd.read_csv(os.path.join(t, "per_seed.csv"))
    rd = lambda n: pd.read_csv(os.path.join(t, n)) if os.path.exists(os.path.join(t, n)) else pd.DataFrame()
    cv, eng, parts = rd("convergence.csv"), rd("per_engine_errors.csv"), rd("partition_stats.csv")
    for ck in (["selected", "final"] if a.checkpoint == "both" else [a.checkpoint]):
        main_comparison(ps, ck, fdir)
        heterogeneity(ps, ck, fdir)
        sweep(ps, ck, fdir, "sweep_rounds", "num_rounds", "Communication rounds R (K=5, E=5)", f"fig4_sweep_rounds_{ck}")
        sweep(ps, ck, fdir, "sweep_local_epochs", "local_epochs", "Local epochs E (K=5, R=20)", f"fig5_sweep_local_epochs_{ck}")
        sweep(ps, ck, fdir, "sweep_clients", "num_clients", "Clients K (R=20, E=5, total data fixed)",
              f"fig6_sweep_clients_{ck}", extra=("sweep_clients_engine", "Whole-engine"))
        comm_tradeoff(ps, ck, fdir)
    convergence(cv, fdir)
    per_engine(eng, fdir)
    if not parts.empty:
        partitions(parts, fdir, int(parts.seed.min()) if 42 not in set(parts.seed) else 42)
    return 0


if __name__ == "__main__":
    sys.exit(main())
