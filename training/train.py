"""
training/train.py
-----------------
Core training loop for the LSTM RUL model.

Provides
--------
``train_one_epoch`` — run one full pass over the training DataLoader.
``train_model``     — full training loop with validation, early stopping,
                      LR scheduling, checkpointing, and TensorBoard / CSV
                      logging.
``get_optimizer``   — build Adam optimizer.
``get_scheduler``   — build ReduceLROnPlateau scheduler.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import CFG, TrainConfig
from training.utils import (
    CSVLogger,
    EarlyStopping,
    evaluate_predictions,
    make_writer,
    resolve_device,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer / scheduler factories
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module, cfg: TrainConfig = None) -> Optimizer:
    """Build an Adam optimizer for the given model.

    Parameters
    ----------
    model:
        PyTorch model whose parameters to optimise.
    cfg:
        Training config; falls back to ``CFG.train``.

    Returns
    -------
    torch.optim.Optimizer
    """
    cfg = cfg or CFG.train
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def get_scheduler(optimizer: Optimizer, cfg: TrainConfig = None) -> ReduceLROnPlateau:
    """Build a ``ReduceLROnPlateau`` LR scheduler.

    Parameters
    ----------
    optimizer:
        Optimizer to schedule.
    cfg:
        Training config; falls back to ``CFG.train``.

    Returns
    -------
    ReduceLROnPlateau
    """
    cfg = cfg or CFG.train
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.lr_min,
    )


# ---------------------------------------------------------------------------
# Single-epoch training pass
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch of training over ``loader``.

    Parameters
    ----------
    model:
        The LSTM model to train.
    loader:
        Training DataLoader yielding ``(X, y)`` batches.
    optimizer:
        Gradient-based optimizer.
    criterion:
        Loss function (e.g. ``nn.MSELoss()``).
    device:
        Target compute device.

    Returns
    -------
    float
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(X_batch)                 # (batch,)
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig = None,
    device: Optional[torch.device] = None,
    run_name: str = "centralized",
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
    csv_logger: Optional[CSVLogger] = None,
    rul_cap: int = 125,
    normalise_targets: bool = False,
) -> Dict[str, Any]:
    """Full training loop with validation, early stopping, and logging.

    Parameters
    ----------
    model:
        Instantiated LSTM model (already on target device).
    train_loader:
        DataLoader for training data.
    val_loader:
        DataLoader for validation data.
    cfg:
        Training hyperparameters; falls back to ``CFG.train``.
    device:
        Compute device; resolved automatically if ``None``.
    run_name:
        Tag used for checkpoint filenames and log sub-directories.
    checkpoint_dir:
        Root directory for checkpoints.
    log_dir:
        Root directory for TensorBoard logs.
    writer:
        An existing ``SummaryWriter`` (created internally if ``None``).
    csv_logger:
        A ``CSVLogger`` instance (created internally if ``None``).
    rul_cap:
        Cap value to inverse-normalise targets when ``normalise_targets=True``.
    normalise_targets:
        If ``True``, multiply predictions by ``rul_cap`` before metric
        computation (targets were normalised during dataset construction).

    Returns
    -------
    Dict[str, Any]
        Training history: ``{"train_loss": [...], "val_loss": [...],
        "val_rmse": [...], "val_mae": [...], "best_epoch": int}``.
    """
    cfg = cfg or CFG.train
    checkpoint_dir = checkpoint_dir or CFG.checkpoint_dir
    log_dir = log_dir or CFG.log_dir

    if device is None:
        device = resolve_device(cfg.device)

    model = model.to(device)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    # Create logging helpers if not provided
    _own_writer = False
    if writer is None:
        writer = make_writer(log_dir, run_name)
        _own_writer = True

    if csv_logger is None:
        csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")
        csv_logger = CSVLogger(
            csv_path,
            fieldnames=["epoch", "train_loss", "val_loss", "val_rmse", "val_mae", "lr"],
        )

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        # ── Training pass ────────────────────────────────────────────────── #
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # ── Validation pass ──────────────────────────────────────────────── #
        val_loss, val_rmse, val_mae = evaluate_model(
            model, val_loader, device, rul_cap=rul_cap, normalise_targets=normalise_targets
        )

        # ── LR scheduling ────────────────────────────────────────────────── #
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── History ──────────────────────────────────────────────────────── #
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        # ── TensorBoard ──────────────────────────────────────────────────── #
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_rmse", val_rmse, epoch)
        writer.add_scalar("Metrics/val_mae", val_mae, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # ── CSV ──────────────────────────────────────────────────────────── #
        csv_logger.log({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_rmse": round(val_rmse, 4),
            "val_mae": round(val_mae, 4),
            "lr": current_lr,
        })

        # ── Checkpoint on improvement ─────────────────────────────────────── #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            save_checkpoint(
                model, optimizer, epoch,
                {"val_loss": val_loss, "val_rmse": val_rmse, "val_mae": val_mae},
                best_ckpt_path,
            )

        logger.info(
            "Epoch %3d | train_loss=%.5f | val_loss=%.5f | RMSE=%.3f | MAE=%.3f | lr=%.6f",
            epoch, train_loss, val_loss, val_rmse, val_mae, current_lr,
        )

        # ── Early stopping ────────────────────────────────────────────────── #
        if stopper(val_loss):
            logger.info("Early stopping at epoch %d (best epoch %d).", epoch, history.get("best_epoch", epoch))
            break

    if _own_writer:
        writer.close()

    return history


# ---------------------------------------------------------------------------
# Validation / inference helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rul_cap: int = 125,
    normalise_targets: bool = False,
) -> Tuple[float, float, float]:
    """Compute validation loss, RMSE, and MAE on a DataLoader.

    Parameters
    ----------
    model:
        Model to evaluate.
    loader:
        Validation or test DataLoader.
    device:
        Compute device.
    rul_cap:
        Used to inverse-normalise if targets were scaled.
    normalise_targets:
        Whether targets were normalised (divide by ``rul_cap``) during
        dataset construction.

    Returns
    -------
    Tuple[float, float, float]
        ``(val_loss, rmse, mae)`` all in RUL cycles.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(y_pred.cpu().numpy())
            all_true.append(y_batch.cpu().numpy())

    import numpy as np
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_true)

    if normalise_targets:
        preds = preds * rul_cap
        trues = trues * rul_cap

    from training.utils import compute_rmse, compute_mae
    rmse = compute_rmse(trues, preds)
    mae = compute_mae(trues, preds)
    val_loss = total_loss / max(n_batches, 1)

    return val_loss, rmse, mae


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rul_cap: int = 125,
    normalise_targets: bool = False,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Run inference and return predictions alongside ground-truth labels.

    Parameters
    ----------
    model:
        Trained model.
    loader:
        DataLoader to iterate.
    device:
        Compute device.
    rul_cap:
        Inverse-normalisation factor.
    normalise_targets:
        Whether targets were normalised during dataset construction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(y_true, y_pred)`` arrays of shape ``(N,)``.
    """
    import numpy as np
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_true.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_true)

    if normalise_targets:
        preds = preds * rul_cap
        trues = trues * rul_cap

    return trues, preds


# ---------------------------------------------------------------------------
# Extended training entry point (used by run_experiments.py)
# ---------------------------------------------------------------------------

def train_model_ex(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
    cfg: "TrainConfig" = None,
    device: Optional["torch.device"] = None,
    run_name: str = "centralized",
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    rul_cap: int = 125,
) -> dict:
    """Run training and return extended result dict including timing.

    This is a thin wrapper around ``train_model`` that additionally records:
    * wall-clock training time
    * per-sample inference time on test_loader (if provided)
    * final test RMSE, MAE, NASA score (if test_loader provided)

    All existing behaviour of ``train_model`` is preserved unchanged.

    Parameters
    ----------
    model:
        Instantiated LSTM model.
    train_loader:
        Training DataLoader.
    val_loader:
        Validation DataLoader.
    test_loader:
        Optional test DataLoader for final evaluation.
    cfg:
        Training config.
    device:
        Compute device.
    run_name:
        Tag for checkpoints and logs.
    checkpoint_dir:
        Checkpoint directory.
    log_dir:
        Log directory.
    rul_cap:
        RUL cap for metric computation.

    Returns
    -------
    dict
        Keys: ``history``, ``training_time_s``, ``inference_time_ms``,
        ``test_rmse``, ``test_mae``, ``test_nasa_score``.
    """
    import time as _time
    import numpy as _np
    from training.utils import compute_rmse, compute_mae, nasa_score, resolve_device

    if device is None:
        device = resolve_device((cfg or CFG.train).device)

    t_start = _time.perf_counter()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        rul_cap=rul_cap,
    )

    training_time_s = _time.perf_counter() - t_start

    # ── Inference timing ─────────────────────────────────────────────────── #
    inference_time_ms = float("nan")
    if test_loader is not None:
        # Warm-up
        model.eval()
        with torch.no_grad():
            for xb, _ in test_loader:
                _ = model(xb.to(device))
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_inf = _time.perf_counter()
        n_samples = 0
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                _ = model(xb)
                n_samples += xb.shape[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inf_elapsed = _time.perf_counter() - t_inf
        inference_time_ms = (inf_elapsed / max(n_samples, 1)) * 1000.0

    # ── Test evaluation ──────────────────────────────────────────────────── #
    test_rmse = test_mae = test_nasa = float("nan")
    if test_loader is not None:
        y_true, y_pred = predict(model, test_loader, device, rul_cap=rul_cap)
        test_rmse = compute_rmse(y_true, y_pred)
        test_mae = compute_mae(y_true, y_pred)
        test_nasa = nasa_score(y_true, y_pred)

    return {
        "history": history,
        "training_time_s": training_time_s,
        "inference_time_ms": inference_time_ms,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_nasa_score": test_nasa,
    }
