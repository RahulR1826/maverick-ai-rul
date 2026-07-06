"""
main.py
-------
Unified entry-point for the Federated Learning RUL pipeline.

Usage examples
--------------
# Preprocess dataset only
python main.py --mode preprocess

# Train centralized baseline
python main.py --mode centralized

# Run federated simulation
python main.py --mode federated

# Evaluate a saved checkpoint
python main.py --mode evaluate --run centralized
python main.py --mode evaluate --run federated

# Generate comparison plots
python main.py --mode visualize

# Run all stages sequentially
python main.py --mode all

# Override config on the command line
python main.py --mode federated --dataset FD002 --clients 3 --rounds 10
"""

import argparse
import logging
import os
import sys
import time

# Pre-import torch to avoid TensorFlow DLL load failure on Windows
import torch

# ── Project root on sys.path ──────────────────────────────────────────────── #
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CFG


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, level: str = "INFO") -> None:
    """Configure root logger with both console and file handlers.

    Parameters
    ----------
    log_dir:
        Directory for the ``pipeline.log`` file.
    level:
        Logging level string (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Use UTF-8 for file handler; use errors='replace' for console on Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    # Wrap console stream to handle Unicode gracefully on cp1252 terminals
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        root.addHandler(console_handler)
        root.addHandler(file_handler)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_preprocess(args: argparse.Namespace) -> None:
    """Preprocess the raw C-MAPSS files and cache results."""
    from datasets.preprocessing import preprocess_dataset

    logger.info("[PREPROCESS] Dataset: %s", CFG.data.dataset_id)
    splits = preprocess_dataset(
        dataset_id=CFG.data.dataset_id,
        force=args.force_preprocess,
    )
    for split, (X, y) in splits.items():
        logger.info("  %-6s -> X=%s  y=%s", split, X.shape, y.shape)
    logger.info("[DONE] Preprocessing complete.")


def run_centralized(args: argparse.Namespace) -> None:
    """Train a centralized baseline model."""
    import torch
    from datasets.cmapss_dataset import build_datasets, make_dataloader
    from datasets.preprocessing import preprocess_dataset, get_n_features
    from models.lstm_model import build_model
    from training.train import train_model
    from training.utils import resolve_device, set_seed

    logger.info("[CENTRALIZED] Training: %s", CFG.data.dataset_id)
    set_seed(CFG.data.seed)
    device = resolve_device(CFG.train.device)

    splits = preprocess_dataset(dataset_id=CFG.data.dataset_id)
    n_features = get_n_features(CFG.data.dataset_id)
    CFG.model.input_size = n_features

    train_ds, val_ds, _ = build_datasets(splits, rul_cap=CFG.data.rul_cap)
    train_loader = make_dataloader(
        train_ds, batch_size=CFG.train.batch_size, shuffle=True,
        num_workers=CFG.train.num_workers,
    )
    val_loader = make_dataloader(
        val_ds, batch_size=CFG.train.batch_size, shuffle=False,
        num_workers=CFG.train.num_workers,
    )

    model = build_model(
        input_size=n_features,
        hidden_size=CFG.model.hidden_size,
        num_layers=CFG.model.num_layers,
        dropout=CFG.model.dropout,
        bidirectional=CFG.model.bidirectional,
        device=device,
    )
    logger.info("Model parameters: %d", model.count_parameters())

    t0 = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=CFG.train,
        device=device,
        run_name="centralized",
        checkpoint_dir=CFG.checkpoint_dir,
        log_dir=CFG.log_dir,
        rul_cap=CFG.data.rul_cap,
    )
    elapsed = time.time() - t0
    best_ep = history.get("best_epoch", "?")
    logger.info("[DONE] Centralized training | best_epoch=%s | %.1fs", best_ep, elapsed)


def run_federated(args: argparse.Namespace) -> None:
    """Run the federated learning simulation."""
    from federated.simulation import run_federated_simulation

    logger.info("[FEDERATED] Simulation: %s", CFG.data.dataset_id)
    t0 = time.time()
    results = run_federated_simulation(cfg=CFG)
    elapsed = time.time() - t0
    logger.info("[DONE] Federated simulation | %.1fs", elapsed)

    # Print final round metrics
    history = results.get("strategy_history", {})
    if history.get("eval_rmse"):
        final_rmse = history["eval_rmse"][-1]
        final_mae = history["eval_mae"][-1]
        logger.info("Final round  RMSE=%.3f  MAE=%.3f", final_rmse, final_mae)


def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a saved model checkpoint on the test split."""
    from training.evaluate import evaluate_checkpoint

    run_name = args.run or "centralized"
    logger.info("[EVALUATE] Checkpoint: %s | dataset: %s", run_name, CFG.data.dataset_id)
    metrics = evaluate_checkpoint(
        run_name=run_name,
        dataset_id=CFG.data.dataset_id,
    )
    logger.info("[DONE] Evaluation complete.")


def run_visualize(args: argparse.Namespace) -> None:
    """Generate training curves and comparison plots."""
    from visualization import (
        plot_centralized_curves,
        plot_federated_convergence,
        plot_comparison,
    )

    logger.info("[VISUALIZE] Generating plots...")
    plot_centralized_curves()
    plot_federated_convergence()
    plot_comparison()
    logger.info("[DONE] Plots saved to %s", CFG.results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated Learning RUL Pipeline — NASA C-MAPSS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["preprocess", "centralized", "federated", "evaluate", "visualize", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--dataset", default="FD001", help="C-MAPSS subset ID.")
    parser.add_argument("--clients", type=int, default=None, help="Number of federated clients.")
    parser.add_argument("--rounds", type=int, default=None, help="Number of federated rounds.")
    parser.add_argument("--epochs", type=int, default=None, help="Max local training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--window", type=int, default=None, help="Sliding window size.")
    parser.add_argument("--non-iid", action="store_true", help="Use non-IID data split.")
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet alpha for non-IID.")
    parser.add_argument("--run", default=None, help="Checkpoint run name for --mode evaluate.")
    parser.add_argument("--device", default=None, help="Compute device override.")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity.")
    parser.add_argument("--force-preprocess", action="store_true", help="Force re-preprocessing.")

    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    """Propagate CLI overrides into the global ``CFG`` object.

    Parameters
    ----------
    args:
        Parsed argument namespace.
    """
    if args.dataset:
        CFG.data.dataset_id = args.dataset
    if args.clients:
        CFG.fed.num_clients = args.clients
    if args.rounds:
        CFG.fed.num_rounds = args.rounds
    if args.epochs:
        CFG.train.epochs = args.epochs
    if args.lr:
        CFG.train.learning_rate = args.lr
    if args.batch_size:
        CFG.train.batch_size = args.batch_size
    if args.window:
        CFG.data.window_size = args.window
    if args.non_iid:
        CFG.fed.iid = False
    if args.alpha:
        CFG.fed.dirichlet_alpha = args.alpha
    if args.device:
        CFG.train.device = args.device


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main orchestration function."""
    args = parse_args()
    apply_args(args)
    setup_logging(CFG.log_dir, level=args.log_level)

    logger.info("Pipeline started | mode=%s | dataset=%s", args.mode, CFG.data.dataset_id)

    # Create required output directories
    for d in [CFG.processed_dir, CFG.log_dir, CFG.checkpoint_dir, CFG.results_dir]:
        os.makedirs(d, exist_ok=True)

    stage_map = {
        "preprocess": run_preprocess,
        "centralized": run_centralized,
        "federated": run_federated,
        "evaluate": run_evaluate,
        "visualize": run_visualize,
    }

    if args.mode == "all":
        run_preprocess(args)
        run_centralized(args)
        run_federated(args)
        args.run = "centralized"
        run_evaluate(args)
        args.run = "federated"
        run_evaluate(args)
        run_visualize(args)
    else:
        stage_map[args.mode](args)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
