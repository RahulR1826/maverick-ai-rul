"""
training/utils.py
-----------------
Shared training utilities:
  * device resolution
  * metric computation (RMSE, MAE, NASA scoring function)
  * early stopping
  * model checkpointing
  * CSV metric logger
  * TensorBoard SummaryWriter factory
"""

import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve a device string to a ``torch.device``.

    Parameters
    ----------
    device_str:
        ``"auto"`` → CUDA if available else CPU.
        ``"cuda"`` / ``"cpu"`` / ``"mps"`` → explicit.

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logger.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in RUL cycles.

    Parameters
    ----------
    y_true:
        Ground-truth RUL values.
    y_pred:
        Model predictions.

    Returns
    -------
    float
        RMSE (same units as RUL: cycles).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in RUL cycles."""
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA asymmetric scoring function from the PHMAP 2008 competition.

    Penalises late predictions (overestimating RUL) more heavily than early
    predictions (underestimating RUL).

    .. math::

        S = \\sum_{i} \\begin{cases}
            e^{-d_i / 13} - 1 & d_i < 0 \\\\
            e^{d_i / 10} - 1  & d_i \\geq 0
        \\end{cases}

    where :math:`d_i = \\hat{y}_i - y_i` (error = prediction – truth).

    Parameters
    ----------
    y_true:
        Ground-truth RUL values.
    y_pred:
        Model predictions.

    Returns
    -------
    float
        NASA score (lower is better; 0 is perfect).
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(score))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute all evaluation metrics in one call.

    Parameters
    ----------
    y_true:
        Ground-truth RUL array.
    y_pred:
        Model prediction array.
    prefix:
        Optional string prepended to each metric key (e.g. ``"test/"``).

    Returns
    -------
    Dict[str, float]
        ``{"rmse": ..., "mae": ..., "nasa_score": ...}`` (with prefix).
    """
    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    score = nasa_score(y_true, y_pred)

    metrics = {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}nasa_score": score,
    }
    for k, v in metrics.items():
        logger.info("  %-30s %.4f", k, v)
    return metrics


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor a validation metric and signal when training should stop.

    Parameters
    ----------
    patience:
        Number of epochs to wait after the last improvement.
    min_delta:
        Minimum absolute improvement required to reset the counter.
    mode:
        ``"min"`` → lower is better (e.g. loss/RMSE);
        ``"max"`` → higher is better.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, current: float) -> bool:
        """Update state and return whether training should stop.

        Parameters
        ----------
        current:
            Current epoch metric value.

        Returns
        -------
        bool
            ``True`` if training should be halted.
        """
        improved = (
            self.best_score is None
            or (self.mode == "min" and current < self.best_score - self.min_delta)
            or (self.mode == "max" and current > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info("Early stopping triggered (patience=%d).", self.patience)
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset the stopper state (call between federation rounds if needed)."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


# ---------------------------------------------------------------------------
# Model checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    """Persist a model checkpoint to disk.

    Parameters
    ----------
    model:
        PyTorch model.
    optimizer:
        Optimizer (state saved for resumption; ``None`` → skip).
    epoch:
        Current epoch index.
    metrics:
        Metric dict to store alongside the weights.
    path:
        Full file path (e.g. ``checkpoints/best.pt``).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    logger.debug("Checkpoint saved → %s  (epoch %d)", path, epoch)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into ``model`` (and optionally ``optimizer``).

    Parameters
    ----------
    model:
        Model to load weights into.
    path:
        Checkpoint file path.
    optimizer:
        Optimizer to restore state into.
    device:
        Target device for weight loading.

    Returns
    -------
    Dict[str, Any]
        Full checkpoint payload (including ``epoch`` and ``metrics``).
    """
    if device is None:
        device = torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    logger.info("Checkpoint loaded from %s (epoch %d)", path, payload.get("epoch", -1))
    return payload


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Append metric rows to a CSV file after every epoch / round.

    Parameters
    ----------
    filepath:
        Destination CSV file (created if it doesn't exist).
    fieldnames:
        Column names.  The CSV header is written once when the file is new.
    """

    def __init__(self, filepath: str, fieldnames: List[str]) -> None:
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # Write header only for new files
        self._write_header = not os.path.exists(filepath)

    def log(self, row: Dict[str, Any]) -> None:
        """Append a single row to the CSV.

        Parameters
        ----------
        row:
            Dict mapping column names to values.
        """
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow(row)


# ---------------------------------------------------------------------------
# TensorBoard writer factory
# ---------------------------------------------------------------------------

def make_writer(log_dir: str, run_name: str = "") -> SummaryWriter:
    """Create a TensorBoard ``SummaryWriter`` for a named run.

    Parameters
    ----------
    log_dir:
        Root log directory.
    run_name:
        Sub-directory name (e.g. ``"centralized"`` or ``"federated"``).

    Returns
    -------
    SummaryWriter
    """
    run_dir = os.path.join(log_dir, run_name) if run_name else log_dir
    os.makedirs(run_dir, exist_ok=True)
    return SummaryWriter(log_dir=run_dir)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across numpy, random and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Global seed set to %d", seed)


def model_size_mb(model: nn.Module) -> float:
    """Estimate model weight size in megabytes.

    Parameters
    ----------
    model:
        PyTorch model.

    Returns
    -------
    float
        Approximate size in MB.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 ** 2)
