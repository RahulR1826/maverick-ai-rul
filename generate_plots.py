"""
generate_plots.py
-----------------
Generates all 11 publication-ready figures for the Federated RUL paper.

Reads from results/ CSV files produced by run_experiments.py.

Usage
-----
    python generate_plots.py

Output: results/plots/  (11 PNG files @ 300 DPI)
"""

import os
import sys
import glob
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from config import RESULTS_DIR

RES      = RESULTS_DIR
PLOT_DIR = os.path.join(RES, "plots")
DPI      = 300
SENTINEL = "NOT_RUN"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("generate_plots")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.7,
    "figure.dpi":        100,
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
})

COLORS = {
    "blue":   "#2563EB",
    "orange": "#EA580C",
    "green":  "#16A34A",
    "red":    "#DC2626",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
    "teal":   "#0891B2",
}


def mkdirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _read_csv(path: str, required_cols=None) -> pd.DataFrame:
    if not os.path.exists(path):
        log.warning("File not found: %s", path); return None
    df = pd.read_csv(path)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log.warning("Missing columns %s in %s", missing, path); return None
    return df


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _save(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved -> %s", path)


# ===========================================================================
# Figure 1 — Centralized vs FL-IID bar chart
# ===========================================================================

def plot_centralized_vs_fl():
    path = os.path.join(RES, "comparisons", "centralized_vs_fl.csv")
    df   = _read_csv(path, ["configuration", "RMSE_mean", "MAE_mean"])
    if df is None: return

    df["RMSE_mean"] = _to_num(df["RMSE_mean"])
    df["MAE_mean"]  = _to_num(df["MAE_mean"])
    df["RMSE_std"]  = _to_num(df.get("RMSE_std", pd.Series([0]*len(df))))
    df["MAE_std"]   = _to_num(df.get("MAE_std",  pd.Series([0]*len(df))))
    df = df.dropna(subset=["RMSE_mean"])

    if df.empty: return

    configs = df["configuration"].tolist()
    rmse    = df["RMSE_mean"].values
    mae     = df["MAE_mean"].values
    rmse_e  = df["RMSE_std"].fillna(0).values
    mae_e   = df["MAE_std"].fillna(0).values

    x  = np.arange(len(configs))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))

    bars1 = ax.bar(x - w/2, rmse, w, yerr=rmse_e, capsize=4,
                   label="RMSE (cycles)", color=COLORS["blue"], alpha=0.9)
    bars2 = ax.bar(x + w/2, mae,  w, yerr=mae_e,  capsize=4,
                   label="MAE (cycles)",  color=COLORS["orange"], alpha=0.9)

    ax.set_xlabel("Training Configuration")
    ax.set_ylabel("Error (cycles)")
    ax.set_title("Centralized vs Federated IID: Prediction Accuracy")
    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.legend(); ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    _save(fig, "centralized_vs_fl.png")


# ===========================================================================
# Figure 2 — FL convergence (per-round RMSE + val_loss)
# ===========================================================================

def plot_fl_convergence():
    path = os.path.join(RES, "fl_iid", "seed_42_rounds.csv")
    df   = _read_csv(path, ["round", "rmse"])
    if df is None: return

    df["round"]    = _to_num(df["round"])
    df["rmse"]     = _to_num(df["rmse"])
    df["val_loss"] = _to_num(df.get("val_loss", pd.Series([float("nan")]*len(df))))
    df = df.dropna(subset=["round", "rmse"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

    ax1.plot(df["round"], df["rmse"], color=COLORS["blue"], lw=2, marker="o",
             markersize=3, label="RMSE")
    ax1.set_xlabel("Communication Round"); ax1.set_ylabel("RMSE (cycles)")
    ax1.set_title("FL-IID Convergence: RMSE"); ax1.legend()

    if df["val_loss"].notna().any():
        ax2.plot(df["round"], df["val_loss"], color=COLORS["orange"], lw=2,
                 marker="s", markersize=3, label="Val Loss")
        ax2.set_xlabel("Communication Round"); ax2.set_ylabel("Validation Loss (MSE)")
        ax2.set_title("FL-IID Convergence: Validation Loss"); ax2.legend()

    fig.tight_layout()
    _save(fig, "fl_convergence.png")


# ===========================================================================
# Figure 3 — Rounds sweep: RMSE vs R
# ===========================================================================

def plot_rounds_vs_rmse():
    path = os.path.join(RES, "rounds", "summary.csv")
    df   = _read_csv(path, ["rounds", "rmse"])
    if df is None: return

    df["rounds"] = _to_num(df["rounds"]); df["rmse"] = _to_num(df["rmse"])
    df = df.dropna(subset=["rounds", "rmse"]).sort_values("rounds")
    if df.empty: return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["rounds"], df["rmse"], color=COLORS["blue"], lw=2,
            marker="o", markersize=6, label="RMSE")
    ax.set_xlabel("Number of Communication Rounds (R)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Communication Rounds vs Prediction Accuracy")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend()
    _save(fig, "rounds_vs_rmse.png")


# ===========================================================================
# Figure 4 — Communication vs RMSE (rounds sweep)
# ===========================================================================

def plot_communication_vs_rmse():
    path = os.path.join(RES, "rounds", "summary.csv")
    df   = _read_csv(path, ["total_comm_MB", "rmse"])
    if df is None: return

    df["total_comm_MB"] = _to_num(df["total_comm_MB"])
    df["rmse"]          = _to_num(df["rmse"])
    df["rounds"]        = _to_num(df.get("rounds", pd.Series([float("nan")]*len(df))))
    df = df.dropna(subset=["total_comm_MB", "rmse"]).sort_values("total_comm_MB")
    if df.empty: return

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["total_comm_MB"], df["rmse"],
                    c=df.get("rounds", df.index), cmap="Blues_r",
                    s=80, zorder=3)
    ax.plot(df["total_comm_MB"], df["rmse"], color=COLORS["gray"], lw=1,
            linestyle="--", alpha=0.5)

    if "rounds" in df.columns:
        for _, row in df.iterrows():
            ax.annotate(f"R={int(row['rounds'])}",
                        xy=(row["total_comm_MB"], row["rmse"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Total Communication (MB) [ESTIMATE]")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Communication Cost vs Accuracy Trade-off")
    plt.colorbar(sc, ax=ax, label="Rounds")
    _save(fig, "communication_vs_rmse.png")


# ===========================================================================
# Figure 5 — Local epochs vs RMSE
# ===========================================================================

def plot_local_epochs_vs_rmse():
    path = os.path.join(RES, "local_epochs", "summary.csv")
    df   = _read_csv(path, ["local_epochs", "rmse"])
    if df is None: return

    df["local_epochs"] = _to_num(df["local_epochs"])
    df["rmse"]         = _to_num(df["rmse"])
    df = df.dropna(subset=["local_epochs", "rmse"]).sort_values("local_epochs")
    if df.empty: return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["local_epochs"].astype(str), df["rmse"],
           color=COLORS["green"], alpha=0.85, width=0.5)
    ax.set_xlabel("Local Epochs per Round (E)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Local Epoch Sweep: Accuracy")
    for i, (e, v) in enumerate(zip(df["local_epochs"], df["rmse"])):
        ax.text(i, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "local_epochs_vs_rmse.png")


# ===========================================================================
# Figure 6 — Local epochs vs training time
# ===========================================================================

def plot_local_epochs_vs_time():
    path = os.path.join(RES, "local_epochs", "summary.csv")
    df   = _read_csv(path, ["local_epochs"])
    if df is None: return

    df["local_epochs"]    = _to_num(df["local_epochs"])
    df["total_training_s"] = _to_num(df.get("total_training_s",
                                            pd.Series([float("nan")]*len(df))))
    df = df.dropna(subset=["local_epochs", "total_training_s"]).sort_values("local_epochs")
    if df.empty:
        log.warning("No training time data in local_epochs summary."); return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["local_epochs"], df["total_training_s"], color=COLORS["purple"],
            lw=2, marker="^", markersize=7)
    ax.set_xlabel("Local Epochs per Round (E)")
    ax.set_ylabel("Total Training Time (s)")
    ax.set_title("Local Epoch Sweep: Training Time")
    _save(fig, "local_epochs_vs_training_time.png")


# ===========================================================================
# Figure 7 — Clients vs RMSE
# ===========================================================================

def plot_clients_vs_rmse():
    path = os.path.join(RES, "clients", "summary.csv")
    df   = _read_csv(path, ["num_clients", "rmse"])
    if df is None: return

    df["num_clients"] = _to_num(df["num_clients"])
    df["rmse"]        = _to_num(df["rmse"])
    df = df.dropna(subset=["num_clients", "rmse"]).sort_values("num_clients")
    if df.empty: return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["num_clients"].astype(str), df["rmse"],
           color=COLORS["teal"], alpha=0.85, width=0.5)
    ax.set_xlabel("Number of Clients (K)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Client Scalability: Accuracy")
    for i, (k, v) in enumerate(zip(df["num_clients"], df["rmse"])):
        ax.text(i, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "clients_vs_rmse.png")


# ===========================================================================
# Figure 8 — Clients vs communication
# ===========================================================================

def plot_clients_vs_communication():
    path = os.path.join(RES, "clients", "summary.csv")
    df   = _read_csv(path, ["num_clients", "total_comm_MB"])
    if df is None: return

    df["num_clients"]   = _to_num(df["num_clients"])
    df["total_comm_MB"] = _to_num(df["total_comm_MB"])
    df = df.dropna(subset=["num_clients", "total_comm_MB"]).sort_values("num_clients")
    if df.empty: return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["num_clients"], df["total_comm_MB"], color=COLORS["red"],
            lw=2, marker="D", markersize=7)
    ax.set_xlabel("Number of Clients (K)")
    ax.set_ylabel("Total Communication (MB) [ESTIMATE]")
    ax.set_title("Client Scalability: Communication Cost")
    _save(fig, "clients_vs_communication.png")


# ===========================================================================
# Figure 9 — IID vs Non-IID convergence curves
# ===========================================================================

def plot_iid_vs_noniid_convergence():
    out_dir = os.path.join(RES, "non_iid")
    configs = [
        ("IID",       "IID_seed42_rounds.csv",        COLORS["blue"],   "IID"),
        ("alpha_1.0", "alpha_1.0_seed42_rounds.csv",  COLORS["green"],  "Non-IID alpha=1.0"),
        ("alpha_0.5", "alpha_0.5_seed42_rounds.csv",  COLORS["orange"], "Non-IID alpha=0.5"),
        ("alpha_0.1", "alpha_0.1_seed42_rounds.csv",  COLORS["red"],    "Non-IID alpha=0.1"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False

    for key, fname, color, label in configs:
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            log.warning("Non-IID rounds file not found: %s", path); continue
        df = _read_csv(path, ["round", "rmse"])
        if df is None: continue
        df["round"] = _to_num(df["round"])
        df["rmse"]  = _to_num(df["rmse"])
        df = df.dropna(subset=["round", "rmse"])
        if df.empty: continue
        ax.plot(df["round"], df["rmse"], lw=2, color=color, label=label)
        any_plotted = True

    if not any_plotted:
        log.warning("No Non-IID convergence data found — skipping figure 9.")
        plt.close(fig); return

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("IID vs Non-IID Convergence")
    ax.legend()
    _save(fig, "iid_vs_non_iid_convergence.png")


# ===========================================================================
# Figure 10 — Heterogeneity vs RMSE bar chart
# ===========================================================================

def plot_heterogeneity_vs_rmse():
    path = os.path.join(RES, "non_iid", "summary.csv")
    df   = _read_csv(path, ["alpha", "rmse"])
    if df is None: return

    df["rmse"]  = _to_num(df["rmse"])
    # Use primary seed only
    if "seed" in df.columns:
        df["seed"] = _to_num(df["seed"])
        df = df[df["seed"] == 42]
    df = df.dropna(subset=["rmse"])

    # Sort: IID first, then by alpha descending
    order_map = {"IID": -1, "1.0": 1.0, "0.5": 0.5, "0.1": 0.1}
    df["_order"] = df["alpha"].map(lambda a: order_map.get(str(a), float("nan")))
    df = df.sort_values("_order")
    if df.empty: return

    labels = df["alpha"].astype(str).tolist()
    rmse   = df["rmse"].values
    colors = [COLORS["blue"] if l == "IID" else COLORS["orange"] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, rmse, color=colors, alpha=0.85, width=0.5)
    ax.set_xlabel("Partition Strategy (alpha)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Data Heterogeneity vs Prediction Accuracy")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    _save(fig, "heterogeneity_vs_rmse.png")


# ===========================================================================
# Figure 11 — Engine-level vs FL-IID comparison
# ===========================================================================

def plot_engine_level_comparison():
    # Load results
    el_path = os.path.join(RES, "engine_level", "summary.csv")
    fl_path = os.path.join(RES, "fl_iid", "all_seeds.csv")

    el_df = _read_csv(el_path)
    fl_df = _read_csv(fl_path)

    if el_df is None or fl_df is None:
        log.warning("Engine-level or FL-IID data file not found."); return

    el_df["seed"] = _to_num(el_df.get("seed", pd.Series([42]*len(el_df))))
    fl_df["seed"] = _to_num(fl_df.get("seed", pd.Series([42]*len(fl_df))))

    el_row = el_df[el_df["seed"] == 42]
    fl_row = fl_df[fl_df["seed"] == 42]

    if el_row.empty or fl_row.empty:
        log.warning("Could not find seed=42 rows for engine-level comparison."); return

    def _get_metric(df, cols):
        for c in cols:
            if c in df.columns:
                v = _to_num(df[c].iloc[0])
                if not np.isnan(v): return float(v)
        return float("nan")

    el_rmse = _get_metric(el_row, ["test_rmse","rmse"])
    el_mae  = _get_metric(el_row, ["test_mae","mae"])
    fl_rmse = _get_metric(fl_row, ["test_rmse","rmse"])
    fl_mae  = _get_metric(fl_row, ["test_mae","mae"])

    if np.isnan(el_rmse) or np.isnan(fl_rmse):
        log.warning("Engine-level or FL-IID RMSE not available — skipping figure 11."); return

    configs = ["FL-IID", "Engine-Level"]
    rmses   = [fl_rmse, el_rmse]
    maes    = [fl_mae,  el_mae]
    x = np.arange(len(configs)); w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - w/2, rmses, w, label="RMSE (cycles)", color=COLORS["blue"], alpha=0.9)
    bars2 = ax.bar(x + w/2, maes,  w, label="MAE (cycles)",  color=COLORS["orange"], alpha=0.9)

    ax.set_xlabel("Partitioning Strategy")
    ax.set_ylabel("Error (cycles)")
    ax.set_title("IID vs Engine-Level Partitioning")
    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.legend()
    for bar in list(bars1) + list(bars2):
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    _save(fig, "engine_level_comparison.png")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    mkdirs(PLOT_DIR)

    log.info("=" * 60)
    log.info("  Generating Publication-Ready Plots")
    log.info("=" * 60)

    plots = [
        ("Figure 1 — Centralized vs FL-IID",     plot_centralized_vs_fl),
        ("Figure 2 — FL Convergence",             plot_fl_convergence),
        ("Figure 3 — Rounds vs RMSE",             plot_rounds_vs_rmse),
        ("Figure 4 — Communication vs RMSE",      plot_communication_vs_rmse),
        ("Figure 5 — Local Epochs vs RMSE",       plot_local_epochs_vs_rmse),
        ("Figure 6 — Local Epochs vs Time",       plot_local_epochs_vs_time),
        ("Figure 7 — Clients vs RMSE",            plot_clients_vs_rmse),
        ("Figure 8 — Clients vs Communication",   plot_clients_vs_communication),
        ("Figure 9 — IID vs Non-IID Convergence", plot_iid_vs_noniid_convergence),
        ("Figure 10 — Heterogeneity vs RMSE",     plot_heterogeneity_vs_rmse),
        ("Figure 11 — Engine-Level Comparison",   plot_engine_level_comparison),
    ]

    n_ok = 0
    for name, fn in plots:
        log.info("  %s ...", name)
        try:
            fn()
            n_ok += 1
        except Exception as e:
            import traceback
            log.error("  FAILED: %s\n%s", name, traceback.format_exc())

    # Verify outputs
    pngs = glob.glob(os.path.join(PLOT_DIR, "*.png"))
    log.info("=" * 60)
    log.info("  Plots generated: %d / %d", n_ok, len(plots))
    log.info("  PNG files in %s: %d", PLOT_DIR, len(pngs))
    for p in sorted(pngs):
        log.info("    %s", os.path.basename(p))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
