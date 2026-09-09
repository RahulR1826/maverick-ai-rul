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


# ---------------------------------------------------------------------------
# Extended simulation entry point (used by run_experiments.py)
# ---------------------------------------------------------------------------

def run_federated_simulation_ex(
    num_clients: int = 5,
    num_rounds: int = 20,
    local_epochs: int = 5,
    iid: bool = True,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    dataset_id: str = "FD001",
    run_name: str = "fl_iid",
    client_data_override: dict = None,
    engine_labels: "np.ndarray" = None,
    engine_ids_per_client: dict = None,
) -> dict:
    """Run a fully parameterised federated simulation and return structured results.

    This is a non-breaking addition.  The original ``run_federated_simulation``
    is not modified.  This variant:
    * accepts explicit hyperparameters rather than mutating global CFG
    * measures per-round wall-clock time
    * evaluates final global model on the test split
    * returns a structured dict with all metrics needed by run_experiments.py

    Parameters
    ----------
    num_clients:
        Number of FL clients.
    num_rounds:
        Number of FL rounds.
    local_epochs:
        Local training epochs per round.
    iid:
        IID (True) or Non-IID (False) partition.
    dirichlet_alpha:
        Dirichlet α for Non-IID split (ignored when iid=True).
    seed:
        Random seed.
    dataset_id:
        C-MAPSS subset to use.
    run_name:
        Tag for log/checkpoint naming.
    client_data_override:
        If provided, use this pre-built {cid: (X,y)} dict instead of
        computing a new partition.  Used for engine-level split.
    engine_labels:
        Per-window engine ID array (only needed when client_data_override
        is also provided, for logging purposes).
    engine_ids_per_client:
        Mapping {cid: [engine_ids]}, used for engine-level logging.

    Returns
    -------
    dict
        Keys:
        ``strategy_history``  — per-round metrics dict from FedAvgRUL
        ``flower_history``    — raw Flower History object
        ``round_times_s``     — list of per-round wall-clock seconds
        ``total_time_s``      — total wall-clock seconds
        ``test_rmse``         — final global model RMSE on test split
        ``test_mae``          — final global model MAE on test split
        ``test_nasa_score``   — final global model NASA score on test split
        ``client_data``       — the per-client data dict used
        ``num_clients``       — K
        ``num_rounds``        — R
        ``local_epochs``      — E
    """
    import time as _time
    import copy as _copy

    from config import Config, DataConfig, ModelConfig, TrainConfig, FedConfig
    from training.utils import set_seed as _set_seed, resolve_device, compute_rmse, compute_mae, nasa_score
    from datasets.preprocessing import preprocess_dataset, get_n_features
    from datasets.cmapss_dataset import CMAPSSDataset, make_dataloader
    from datasets.split_clients import split_for_clients
    from models.lstm_model import build_model
    from federated.client import _get_params, _set_params
    from federated.server import build_strategy, build_server_config
    from training.train import evaluate_model, predict as _predict

    _set_seed(seed)

    # Build a local config copy — do NOT mutate global CFG
    cfg = Config(
        data=DataConfig(dataset_id=dataset_id, seed=seed),
        model=ModelConfig(),
        train=TrainConfig(),
        fed=FedConfig(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            iid=iid,
            dirichlet_alpha=dirichlet_alpha,
        ),
    )

    device = resolve_device(cfg.train.device)

    # ── Data ─────────────────────────────────────────────────────────────── #
    splits = preprocess_dataset(dataset_id=cfg.data.dataset_id, cfg=cfg.data)
    n_features = get_n_features(cfg.data.dataset_id)
    cfg.model.input_size = n_features

    X_train, y_train = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]

    # ── Partition ─────────────────────────────────────────────────────────── #
    if client_data_override is not None:
        client_data = client_data_override
    else:
        client_data = split_for_clients(X_train, y_train, cfg=cfg.fed, seed=seed)

    # ── Global model ──────────────────────────────────────────────────────── #
    global_model = build_model(
        input_size=n_features,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        device=device,
    )
    initial_params = fl.common.ndarrays_to_parameters(_get_params(global_model))

    # ── Per-round timing via wrapper strategy ─────────────────────────────── #
    round_start_times: list = []
    round_end_times: list = []

    # Server-side evaluation (validation set)
    def server_evaluate_fn(server_round, parameters, config):
        _set_params(global_model, parameters)
        X_val, y_val = val_data
        val_ds = CMAPSSDataset(X_val, y_val, rul_cap=cfg.data.rul_cap)
        val_loader = make_dataloader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
        val_loss, rmse, mae = evaluate_model(global_model, val_loader, device, rul_cap=cfg.data.rul_cap)
        round_end_times.append(_time.perf_counter())
        return val_loss, {"rmse": rmse, "mae": mae}

    import os as _os
    _os.makedirs(cfg.log_dir, exist_ok=True)
    _os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    strategy = build_strategy(
        initial_parameters=initial_params,
        fed_cfg=cfg.fed,
        log_dir=cfg.log_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        run_name=run_name,
        evaluate_fn=server_evaluate_fn,
    )

    # Patch aggregate_fit to record round start time
    _orig_aggregate_fit = strategy.aggregate_fit

    def _timed_aggregate_fit(server_round, results, failures):
        round_start_times.append(_time.perf_counter())
        return _orig_aggregate_fit(server_round, results, failures)

    strategy.aggregate_fit = _timed_aggregate_fit

    # ── Client factory ────────────────────────────────────────────────────── #
    def client_fn(cid: str):
        from models.lstm_model import build_model as _bm
        from datasets.cmapss_dataset import CMAPSSDataset as _DS, make_dataloader as _mdl
        from federated.client import RULClient as _RC
        cid_int = int(cid)
        _model = _bm(input_size=n_features, hidden_size=cfg.model.hidden_size,
                     num_layers=cfg.model.num_layers, dropout=cfg.model.dropout, device=device)
        X_c, y_c = client_data[cid_int]
        X_v, y_v = val_data
        tr_ds = _DS(X_c, y_c, rul_cap=cfg.data.rul_cap)
        vl_ds = _DS(X_v, y_v, rul_cap=cfg.data.rul_cap)
        tr_ldr = _mdl(tr_ds, batch_size=cfg.train.batch_size, shuffle=True)
        vl_ldr = _mdl(vl_ds, batch_size=cfg.train.batch_size, shuffle=False)
        return _RC(client_id=cid_int, model=_model, train_loader=tr_ldr,
                   val_loader=vl_ldr, device=device, cfg=cfg.train,
                   local_epochs=local_epochs, rul_cap=cfg.data.rul_cap).to_client()

    t_total_start = _time.perf_counter()

    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=build_server_config(num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "include_dashboard": False, "num_cpus": 2},
    )

    total_time_s = _time.perf_counter() - t_total_start

    # Per-round times (end_time - start_time for each round)
    round_times_s = []
    for i in range(min(len(round_start_times), len(round_end_times))):
        round_times_s.append(round_end_times[i] - round_start_times[i])
    # If lengths differ (last round may not have end recorded), pad with nan
    while len(round_times_s) < num_rounds:
        round_times_s.append(float("nan"))

    # ── Final test evaluation ─────────────────────────────────────────────── #
    X_test, y_test = test_data
    test_ds = CMAPSSDataset(X_test, y_test, rul_cap=cfg.data.rul_cap)
    test_loader = make_dataloader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)
    y_true, y_pred = _predict(global_model, test_loader, device, rul_cap=cfg.data.rul_cap)
    test_rmse = compute_rmse(y_true, y_pred)
    test_mae = compute_mae(y_true, y_pred)
    test_nasa = nasa_score(y_true, y_pred)

    strategy.finalize()

    return {
        "strategy_history": strategy.history,
        "flower_history": history,
        "round_times_s": round_times_s,
        "total_time_s": total_time_s,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_nasa_score": test_nasa,
        "client_data": client_data,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
    }
