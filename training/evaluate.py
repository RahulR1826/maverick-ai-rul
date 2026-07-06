"""
training/evaluate.py
--------------------
Standalone evaluation script for a trained model checkpoint.

Loads the best checkpoint for a given run name, runs inference on the test
DataLoader, and reports RMSE, MAE, and the NASA scoring function.  Also
serialises predictions to a CSV file for offline analysis.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from config import CFG
from datasets.cmapss_dataset import CMAPSSDataset, make_dataloader
from datasets.preprocessing import preprocess_dataset, get_n_features
from models.lstm_model import build_model
from training.train import predict
from training.utils import evaluate_predictions, load_checkpoint, resolve_device

logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    run_name: str = "centralized",
    dataset_id: str = "FD001",
    checkpoint_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> dict:
    """Load a saved checkpoint and evaluate it on the test split.

    Parameters
    ----------
    run_name:
        Name prefix used when saving the checkpoint (e.g. ``"centralized"``
        or ``"federated"``).
    dataset_id:
        C-MAPSS subset identifier.
    checkpoint_dir:
        Directory containing ``{run_name}_best.pt``.
    results_dir:
        Directory where prediction CSV is saved.

    Returns
    -------
    dict
        Metric dictionary with keys ``rmse``, ``mae``, ``nasa_score``.
    """
    checkpoint_dir = checkpoint_dir or CFG.checkpoint_dir
    results_dir = results_dir or CFG.results_dir
    os.makedirs(results_dir, exist_ok=True)

    device = resolve_device(CFG.train.device)

    # ── Load preprocessed data ──────────────────────────────────────────── #
    splits = preprocess_dataset(dataset_id=dataset_id)
    n_features = get_n_features(dataset_id)

    test_ds = CMAPSSDataset(
        *splits["test"],
        normalise_targets=False,
        rul_cap=CFG.data.rul_cap,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=CFG.train.batch_size,
        shuffle=False,
        num_workers=CFG.train.num_workers,
    )

    # ── Build model and load weights ─────────────────────────────────────── #
    model = build_model(
        input_size=n_features,
        hidden_size=CFG.model.hidden_size,
        num_layers=CFG.model.num_layers,
        dropout=CFG.model.dropout,
        bidirectional=CFG.model.bidirectional,
        device=device,
    )

    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_checkpoint(model, ckpt_path, device=device)

    # ── Run inference ────────────────────────────────────────────────────── #
    y_true, y_pred = predict(model, test_loader, device)

    # ── Compute metrics ──────────────────────────────────────────────────── #
    metrics = evaluate_predictions(y_true, y_pred, prefix=f"{run_name}/test/")

    # ── Save predictions CSV ─────────────────────────────────────────────── #
    pred_csv = os.path.join(results_dir, f"{run_name}_{dataset_id}_predictions.csv")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(pred_csv, index=False)
    logger.info("Predictions saved → %s", pred_csv)

    # ── Print summary ────────────────────────────────────────────────────── #
    print("\n" + "=" * 55)
    print(f"  Evaluation: {run_name}  |  Dataset: {dataset_id}")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<35s} {v:.4f}")
    print("=" * 55 + "\n")

    return metrics
