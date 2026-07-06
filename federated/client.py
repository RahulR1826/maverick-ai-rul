"""
federated/client.py
-------------------
Flower NumPy client representing a single airline (data owner).

Each client:
  1. Receives the current global model weights from the server.
  2. Updates its local LSTM model with those weights.
  3. Runs ``local_epochs`` of training on its private shard.
  4. Returns updated weights + training metrics to the server.
  5. Optionally evaluates on its local validation split.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from config import CFG, TrainConfig
from models.lstm_model import LSTMModel
from training.train import evaluate_model, train_one_epoch, get_optimizer, get_scheduler
from training.utils import compute_mae, compute_rmse, nasa_score

logger = logging.getLogger(__name__)


class RULClient(NumPyClient):
    """Flower client that trains an LSTM RUL model on a local data shard.

    Parameters
    ----------
    client_id:
        Integer client identifier (airline index).
    model:
        Shared LSTM architecture (weights will be set per round).
    train_loader:
        DataLoader for this client's private training shard.
    val_loader:
        DataLoader for this client's local validation shard.
    device:
        Compute device.
    cfg:
        Training hyperparameters.
    local_epochs:
        Number of local epochs per federation round.
    rul_cap:
        RUL cap used to inverse-normalise targets for metric computation.
    normalise_targets:
        Whether targets were normalised by ``rul_cap`` in the dataset.
    """

    def __init__(
        self,
        client_id: int,
        model: LSTMModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: TrainConfig = None,
        local_epochs: int = 5,
        rul_cap: int = 125,
        normalise_targets: bool = False,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg or CFG.train
        self.local_epochs = local_epochs
        self.rul_cap = rul_cap
        self.normalise_targets = normalise_targets

        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return the current model weights as a list of NumPy arrays.

        Parameters
        ----------
        config:
            Unused config dict (required by Flower interface).

        Returns
        -------
        NDArrays
            Ordered list of weight tensors as NumPy arrays.
        """
        return _get_params(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Update the local model with weights received from the server.

        Parameters
        ----------
        parameters:
            List of NumPy arrays (same order as ``get_parameters``).
        """
        _set_params(self.model, parameters)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Set global weights, run local training, return updated weights.

        Parameters
        ----------
        parameters:
            Global model weights from the server.
        config:
            Round-level config dict (may contain ``local_epochs``).

        Returns
        -------
        Tuple[NDArrays, int, Dict[str, Scalar]]
            Updated weights, number of training samples, training metrics.
        """
        # Allow server to override local_epochs per round
        local_epochs = int(config.get("local_epochs", self.local_epochs))

        self.set_parameters(parameters)
        optimizer = get_optimizer(self.model, self.cfg)
        scheduler = get_scheduler(optimizer, self.cfg)

        avg_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = train_one_epoch(
                self.model, self.train_loader, optimizer,
                self.criterion, self.device,
            )
            scheduler.step(epoch_loss)
            avg_loss = epoch_loss
            logger.debug(
                "Client %d | epoch %d/%d | loss=%.5f",
                self.client_id, epoch + 1, local_epochs, epoch_loss,
            )

        n_samples = len(self.train_loader.dataset)
        metrics: Dict[str, Scalar] = {
            "train_loss": float(avg_loss),
            "client_id": int(self.client_id),
        }

        logger.info(
            "Client %d fit complete | samples=%d | avg_loss=%.5f",
            self.client_id, n_samples, avg_loss,
        )

        return _get_params(self.model), n_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Set global weights and evaluate on the local validation split.

        Parameters
        ----------
        parameters:
            Global model weights from the server.
        config:
            Round-level config dict.

        Returns
        -------
        Tuple[float, int, Dict[str, Scalar]]
            Validation loss, number of samples, metric dictionary.
        """
        self.set_parameters(parameters)

        val_loss, rmse, mae = evaluate_model(
            self.model, self.val_loader, self.device,
            rul_cap=self.rul_cap,
            normalise_targets=self.normalise_targets,
        )

        # Compute NASA score separately
        preds, trues = _collect_predictions(
            self.model, self.val_loader, self.device,
            rul_cap=self.rul_cap,
            normalise_targets=self.normalise_targets,
        )
        nasa = nasa_score(trues, preds)

        n_samples = len(self.val_loader.dataset)
        metrics: Dict[str, Scalar] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "nasa_score": float(nasa),
            "client_id": int(self.client_id),
        }

        logger.info(
            "Client %d eval | samples=%d | loss=%.5f | RMSE=%.3f | MAE=%.3f | NASA=%.2f",
            self.client_id, n_samples, val_loss, rmse, mae, nasa,
        )

        return float(val_loss), n_samples, metrics


# ---------------------------------------------------------------------------
# Parameter conversion helpers
# ---------------------------------------------------------------------------

def _get_params(model: nn.Module) -> NDArrays:
    """Extract model parameters as ordered list of NumPy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def _set_params(model: nn.Module, parameters: NDArrays) -> None:
    """Load a list of NumPy arrays into a model's state dict."""
    state_dict = dict(zip(model.state_dict().keys(), parameters))
    model.load_state_dict(
        {k: torch.tensor(v) for k, v in state_dict.items()},
        strict=True,
    )


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rul_cap: int,
    normalise_targets: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all predictions and ground-truth labels from a DataLoader.

    Parameters
    ----------
    model:
        Model in evaluation mode.
    loader:
        DataLoader to iterate.
    device:
        Compute device.
    rul_cap:
        Inverse-normalisation factor.
    normalise_targets:
        Whether to invert target normalisation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(y_pred, y_true)`` arrays.
    """
    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_pred = model(X_batch).cpu().numpy()
            preds_list.append(y_pred)
            trues_list.append(y_batch.numpy())

    preds = np.concatenate(preds_list)
    trues = np.concatenate(trues_list)

    if normalise_targets:
        preds = preds * rul_cap
        trues = trues * rul_cap

    return preds, trues
