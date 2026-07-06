"""
visualization.py
----------------
Generates publication-quality plots for the federated RUL pipeline.

Plots produced
--------------
1. Centralized training/validation loss curves
2. Centralized RMSE / MAE curves
3. Federated global evaluation metrics per round (loss, RMSE, MAE, NASA score)
4. Federated convergence (RMSE over rounds)
5. Side-by-side comparison: centralized vs federated RMSE / MAE
6. Prediction scatter plot (true vs predicted RUL)

All figures are saved as high-resolution PNGs in ``results_dir``.
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # non-interactive backend (safe for servers / CI)
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

from config import CFG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(filename: str) -> Optional[pd.DataFrame]:
    """Load a CSV log file or return ``None`` if it doesn't exist."""
    path = os.path.join(CFG.log_dir, filename)
    if not os.path.exists(path):
        logger.warning("CSV log not found: %s", path)
        return None
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> str:
    """Save a Matplotlib figure and close it.

    Parameters
    ----------
    fig:
        Figure to save.
    name:
        Filename stem (``results_dir/name.png``).

    Returns
    -------
    str
        Full path to the saved PNG.
    """
    os.makedirs(CFG.results_dir, exist_ok=True)
    path = os.path.join(CFG.results_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Centralized training curves
# ---------------------------------------------------------------------------

def plot_centralized_curves(
    csv_filename: str = "centralized_metrics.csv",
    dataset_id: Optional[str] = None,
) -> List[str]:
    """Plot training/validation loss and RMSE/MAE curves for centralized run.

    Parameters
    ----------
    csv_filename:
        CSV log file produced by ``train_model``.
    dataset_id:
        Dataset label for plot titles.

    Returns
    -------
    List[str]
        Paths to saved PNG files.
    """
    df = _load_csv(csv_filename)
    if df is None:
        return []

    dataset_id = dataset_id or CFG.data.dataset_id
    saved_paths = []

    # ── Loss curves ─────────────────────────────────────────────────────── #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2, color="#2196F3")
    ax.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2,
            color="#FF5722", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Centralized Training – Loss ({dataset_id})")
    ax.legend()
    saved_paths.append(_save(fig, f"centralized_loss_{dataset_id}"))

    # ── Metric curves ────────────────────────────────────────────────────── #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["epoch"], df["val_rmse"], color="#4CAF50", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE (cycles)")
    axes[0].set_title(f"Validation RMSE ({dataset_id})")

    axes[1].plot(df["epoch"], df["val_mae"], color="#9C27B0", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (cycles)")
    axes[1].set_title(f"Validation MAE ({dataset_id})")

    fig.suptitle("Centralized Baseline – Validation Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    saved_paths.append(_save(fig, f"centralized_metrics_{dataset_id}"))

    return saved_paths


# ---------------------------------------------------------------------------
# 2. Federated convergence
# ---------------------------------------------------------------------------

def plot_federated_convergence(
    csv_filename: str = "federated_server_metrics.csv",
    dataset_id: Optional[str] = None,
) -> List[str]:
    """Plot federated round-wise evaluation metrics.

    Parameters
    ----------
    csv_filename:
        Server-side metric CSV produced by ``FedAvgRUL``.
    dataset_id:
        Dataset label for titles.

    Returns
    -------
    List[str]
        Paths to saved PNG files.
    """
    df = _load_csv(csv_filename)
    if df is None:
        return []

    df = df.dropna(subset=["eval_rmse"])   # only rounds that have eval metrics
    dataset_id = dataset_id or CFG.data.dataset_id
    saved_paths = []

    # ── 4-panel convergence figure ───────────────────────────────────────── #
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Federated Learning Convergence – {dataset_id}\n"
        f"({CFG.fed.num_clients} clients, {'IID' if CFG.fed.iid else 'Non-IID'})",
        fontsize=14, fontweight="bold",
    )

    panels = [
        ("eval_loss",       "Eval Loss",             "#2196F3", axes[0, 0]),
        ("eval_rmse",       "RMSE (cycles)",          "#4CAF50", axes[0, 1]),
        ("eval_mae",        "MAE (cycles)",            "#9C27B0", axes[1, 0]),
        ("eval_nasa_score", "NASA Score",             "#FF5722", axes[1, 1]),
    ]

    for col, label, color, ax in panels:
        if col in df.columns:
            ax.plot(df["round"], df[col], marker="o", markersize=4,
                    linewidth=2, color=color, label=label)
            ax.set_xlabel("Round")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend(fontsize=9)

    fig.tight_layout()
    saved_paths.append(_save(fig, f"federated_convergence_{dataset_id}"))

    return saved_paths


# ---------------------------------------------------------------------------
# 3. Centralized vs Federated comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    central_csv: str = "centralized_metrics.csv",
    fed_csv: str = "federated_server_metrics.csv",
    dataset_id: Optional[str] = None,
) -> List[str]:
    """Overlay centralized and federated RMSE/MAE on the same axes.

    Parameters
    ----------
    central_csv:
        Centralized training metrics CSV.
    fed_csv:
        Federated server metrics CSV.
    dataset_id:
        Dataset label.

    Returns
    -------
    List[str]
        Paths to saved figures.
    """
    c_df = _load_csv(central_csv)
    f_df = _load_csv(fed_csv)
    dataset_id = dataset_id or CFG.data.dataset_id
    saved_paths = []

    if c_df is None or f_df is None:
        logger.warning("Cannot generate comparison plot: missing log files.")
        return saved_paths

    f_df = f_df.dropna(subset=["eval_rmse"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Centralized vs Federated Learning – {dataset_id}",
        fontsize=14, fontweight="bold",
    )

    # Normalise x-axis to «proportion of total training steps»
    c_epochs = np.linspace(0, 1, len(c_df))
    f_rounds = np.linspace(0, 1, len(f_df))

    # RMSE
    axes[0].plot(c_epochs, c_df["val_rmse"], label="Centralized", color="#2196F3", linewidth=2)
    axes[0].plot(f_rounds, f_df["eval_rmse"], label="Federated", color="#FF5722",
                 linewidth=2, linestyle="--")
    axes[0].set_xlabel("Normalised training progress")
    axes[0].set_ylabel("RMSE (cycles)")
    axes[0].set_title("RMSE Comparison")
    axes[0].legend()

    # MAE
    axes[1].plot(c_epochs, c_df["val_mae"], label="Centralized", color="#2196F3", linewidth=2)
    axes[1].plot(f_rounds, f_df["eval_mae"], label="Federated", color="#FF5722",
                 linewidth=2, linestyle="--")
    axes[1].set_xlabel("Normalised training progress")
    axes[1].set_ylabel("MAE (cycles)")
    axes[1].set_title("MAE Comparison")
    axes[1].legend()

    fig.tight_layout()
    saved_paths.append(_save(fig, f"comparison_{dataset_id}"))

    return saved_paths


# ---------------------------------------------------------------------------
# 4. Prediction scatter plot
# ---------------------------------------------------------------------------

def plot_predictions(
    pred_csv: str,
    run_name: str = "centralized",
    dataset_id: Optional[str] = None,
) -> str:
    """Plot true vs predicted RUL (scatter + identity line).

    Parameters
    ----------
    pred_csv:
        CSV file with columns ``y_true`` and ``y_pred``.
    run_name:
        Label for the plot title.
    dataset_id:
        Dataset label.

    Returns
    -------
    str
        Path to saved PNG.
    """
    if not os.path.exists(pred_csv):
        logger.warning("Prediction CSV not found: %s", pred_csv)
        return ""

    df = pd.read_csv(pred_csv)
    dataset_id = dataset_id or CFG.data.dataset_id

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["y_true"], df["y_pred"], alpha=0.4, s=10, color="#2196F3")

    # Identity line
    lim = max(df["y_true"].max(), df["y_pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{run_name.capitalize()} – True vs Predicted RUL ({dataset_id})")
    ax.legend()

    return _save(fig, f"scatter_{run_name}_{dataset_id}")


# ---------------------------------------------------------------------------
# 5. Client data distribution (Non-IID visualisation)
# ---------------------------------------------------------------------------

def plot_client_distribution(
    client_labels: Dict[int, np.ndarray],
    dataset_id: Optional[str] = None,
) -> str:
    """Bar chart showing RUL distribution across clients.

    Parameters
    ----------
    client_labels:
        Dict ``{client_id: y_array}`` of per-client RUL arrays.
    dataset_id:
        Dataset label.

    Returns
    -------
    str
        Path to saved PNG.
    """
    dataset_id = dataset_id or CFG.data.dataset_id
    n_clients = len(client_labels)
    bins = np.linspace(0, CFG.data.rul_cap, 11)

    fig, axes = plt.subplots(1, n_clients, figsize=(3 * n_clients, 4), sharey=True)
    if n_clients == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_clients))  # type: ignore[attr-defined]
    for cid, ax in enumerate(axes):
        y = client_labels.get(cid, np.array([]))
        ax.hist(y, bins=bins, color=colors[cid], edgecolor="white", alpha=0.85)
        ax.set_title(f"Client {cid}", fontsize=9)
        ax.set_xlabel("RUL" if cid == n_clients // 2 else "")
        if cid == 0:
            ax.set_ylabel("Count")

    fig.suptitle(
        f"Per-Client RUL Distribution – {dataset_id}\n"
        f"({'IID' if CFG.fed.iid else f'Non-IID α={CFG.fed.dirichlet_alpha}'})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, f"client_dist_{dataset_id}")
