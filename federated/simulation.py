"""
federated/simulation.py
-----------------------
In-process Flower simulation for the federated RUL pipeline.

Uses ``fl.simulation.start_simulation`` (stable across Flower 1.x) to spawn
all clients within a single Python process via Ray.  This avoids any network
overhead and is ideal for research and reproducibility.

For Flower >= 1.8 the new ``run_simulation`` / ``ClientApp`` / ``ServerApp``
API is also available; that newer interface is best invoked via ``flwr run``
CLI.  This module uses ``start_simulation`` for maximum script compatibility.

Workflow
--------
1. Preprocess the dataset (or load from cache).
2. Partition training data into per-client shards.
3. Build an LSTM model and serialise its initial weights.
4. Create a ``FedAvgRUL`` strategy with per-round logging.
5. Launch the Flower simulation.
6. Save the final global model and return the server-side history.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import NDArrays
from flwr.simulation import start_simulation

from config import CFG, Config
from datasets.cmapss_dataset import CMAPSSDataset, make_dataloader
from datasets.preprocessing import preprocess_dataset, get_n_features
from datasets.split_clients import split_for_clients
from federated.client import RULClient, _get_params, _set_params
from federated.server import build_strategy, build_server_config
from models.lstm_model import build_model
from training.utils import resolve_device, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory (called once per selected client per round)
# ---------------------------------------------------------------------------

def _make_client_fn(
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    val_data: Tuple[np.ndarray, np.ndarray],
    cfg: Config,
    n_features: int,
    device: torch.device,
) -> fl.client.ClientFn:
    """Return a ``client_fn`` closure for the Flower simulation.

    The closure captures the per-client data shards so that the correct
    training data is used when a specific client is selected each round.

    Parameters
    ----------
    client_data:
        Dict mapping ``client_id`` to ``(X_train, y_train)`` shards.
    val_data:
        Global validation split ``(X_val, y_val)`` used by all clients.
    cfg:
        Full configuration object.
    n_features:
        Number of input features (determines model input size).
    device:
        Compute device.

    Returns
    -------
    fl.client.ClientFn
        Factory function ``(cid: str) → fl.client.Client``.
    """

    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)

        # Build a fresh model instance per client (weights set by server)
        model = build_model(
            input_size=n_features,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional=cfg.model.bidirectional,
            device=device,
        )

        X_train, y_train = client_data[cid_int]
        X_val, y_val = val_data

        train_ds = CMAPSSDataset(X_train, y_train, rul_cap=cfg.data.rul_cap)
        val_ds = CMAPSSDataset(X_val, y_val, rul_cap=cfg.data.rul_cap)

        train_loader = make_dataloader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        )
        val_loader = make_dataloader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )

        return RULClient(
            client_id=cid_int,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg.train,
            local_epochs=cfg.fed.local_epochs,
            rul_cap=cfg.data.rul_cap,
        ).to_client()

    return client_fn


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def run_federated_simulation(
    cfg: Optional[Config] = None,
    dataset_id: Optional[str] = None,
) -> Dict:
    """Run the complete federated learning simulation.

    Parameters
    ----------
    cfg:
        Full pipeline configuration; defaults to ``CFG``.
    dataset_id:
        Override the dataset subset (e.g. ``"FD002"``).

    Returns
    -------
    Dict
        Dictionary with Flower history and strategy history objects.
    """
    cfg = cfg or CFG
    if dataset_id is not None:
        cfg.data.dataset_id = dataset_id

    set_seed(cfg.data.seed)
    device = resolve_device(cfg.train.device)

    logger.info("=" * 60)
    logger.info("  Federated Learning Simulation")
    logger.info("  Dataset : %s", cfg.data.dataset_id)
    logger.info("  Clients : %d  |  Rounds : %d  |  Local epochs : %d",
                cfg.fed.num_clients, cfg.fed.num_rounds, cfg.fed.local_epochs)
    logger.info("  Split   : %s", "IID" if cfg.fed.iid else f"Non-IID (α={cfg.fed.dirichlet_alpha})")
    logger.info("=" * 60)

    # ── 1. Preprocess or load cached data ───────────────────────────────── #
    splits = preprocess_dataset(dataset_id=cfg.data.dataset_id, cfg=cfg.data)
    n_features = get_n_features(cfg.data.dataset_id)
    cfg.model.input_size = n_features

    X_train, y_train = splits["train"]
    val_data = splits["val"]

    # ── 2. Partition into client shards ──────────────────────────────────── #
    client_data = split_for_clients(X_train, y_train, cfg=cfg.fed, seed=cfg.data.seed)

    # ── 3. Build initial global model ────────────────────────────────────── #
    global_model = build_model(
        input_size=n_features,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
        device=device,
    )
    logger.info(
        "Model: %d trainable parameters (%.2f MB)",
        global_model.count_parameters(),
        sum(p.numel() * p.element_size() for p in global_model.parameters()) / 1e6,
    )

    # Serialise initial weights for the strategy
    initial_params = fl.common.ndarrays_to_parameters(_get_params(global_model))

    # ── 4. Server-side evaluation function ───────────────────────────────── #
    def server_evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: Dict,
    ) -> Optional[Tuple[float, Dict]]:
        """Evaluate global model on the shared validation set each round."""
        from training.train import evaluate_model
        _set_params(global_model, parameters)
        X_val, y_val = val_data
        val_ds = CMAPSSDataset(X_val, y_val, rul_cap=cfg.data.rul_cap)
        val_loader = make_dataloader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
        )
        val_loss, rmse, mae = evaluate_model(
            global_model, val_loader, device, rul_cap=cfg.data.rul_cap
        )
        logger.info(
            "Server eval round %d | val_loss=%.5f | RMSE=%.3f | MAE=%.3f",
            server_round, val_loss, rmse, mae,
        )
        return val_loss, {"rmse": rmse, "mae": mae}

    # ── 5. Build strategy ────────────────────────────────────────────────── #
    strategy = build_strategy(
        initial_parameters=initial_params,
        fed_cfg=cfg.fed,
        log_dir=cfg.log_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        run_name="federated",
        evaluate_fn=server_evaluate_fn,
    )

    # ── 6. Build client factory ──────────────────────────────────────────── #
    client_fn = _make_client_fn(
        client_data=client_data,
        val_data=val_data,
        cfg=cfg,
        n_features=n_features,
        device=device,
    )

    # ── 7. Launch Flower simulation ──────────────────────────────────────── #
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ``start_simulation`` is the stable script-mode API across Flower 1.x
    history = start_simulation(
        client_fn=client_fn,
        num_clients=cfg.fed.num_clients,
        config=build_server_config(cfg.fed.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False, "num_cpus": 2},
    )

    # ── 8. Save final global model ────────────────────────────────────────── #
    final_ckpt = os.path.join(cfg.checkpoint_dir, "federated_best.pt")
    _save_final_model(global_model, strategy, final_ckpt)

    strategy.finalize()

    logger.info("Federated simulation complete. Logs: %s", cfg.log_dir)
    return {"flower_history": history, "strategy_history": strategy.history}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_final_model(
    model: torch.nn.Module,
    strategy,
    path: str,
) -> None:
    """Save the final global model weights derived from the last aggregation.

    Parameters
    ----------
    model:
        Global model (already loaded with the last round's parameters by the
        ``server_evaluate_fn``).
    strategy:
        The ``FedAvgRUL`` strategy (holds metric history).
    path:
        Output checkpoint file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": strategy.history,
        },
        path,
    )
    logger.info("Final federated model saved → %s", path)
