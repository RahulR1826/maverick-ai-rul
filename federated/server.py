"""
federated/server.py
-------------------
Flower server utilities for the federated RUL pipeline.

Provides a factory function ``build_server_config`` that assembles the
``fl.server.ServerConfig`` and the custom ``FedAvgRUL`` strategy, and a
helper ``run_server`` that can be used to start a *real* gRPC server if
running clients in separate processes (as opposed to the simulation mode in
``simulation.py``).

For most research workflows, prefer ``simulation.py`` which uses
``fl.simulation.start_simulation`` to run everything in-process.
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server import ServerConfig

from config import CFG, FedConfig
from federated.strategy import FedAvgRUL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------

def build_strategy(
    initial_parameters: fl.common.Parameters,
    fed_cfg: FedConfig = None,
    log_dir: str = None,
    checkpoint_dir: str = None,
    run_name: str = "federated",
    evaluate_fn: Optional[Callable] = None,
) -> FedAvgRUL:
    """Construct the ``FedAvgRUL`` strategy with all federation settings.

    Parameters
    ----------
    initial_parameters:
        Starting model weights (Flower ``Parameters`` object).
    fed_cfg:
        Federation configuration; defaults to ``CFG.fed``.
    log_dir:
        Root log directory for TensorBoard and CSV output.
    checkpoint_dir:
        Root checkpoint directory.
    run_name:
        Tag for naming logs and checkpoints.
    evaluate_fn:
        Optional server-side evaluation function called after each round.
        Signature: ``(server_round, parameters, config) → (loss, metrics)``.

    Returns
    -------
    FedAvgRUL
        Configured strategy instance.
    """
    fed_cfg = fed_cfg or CFG.fed
    log_dir = log_dir or CFG.log_dir
    checkpoint_dir = checkpoint_dir or CFG.checkpoint_dir

    strategy = FedAvgRUL(
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        save_every=1,
        # FedAvg parameters
        fraction_fit=fed_cfg.fraction_fit,
        fraction_evaluate=fed_cfg.fraction_evaluate,
        min_fit_clients=fed_cfg.min_fit_clients,
        min_evaluate_clients=fed_cfg.min_evaluate_clients,
        min_available_clients=fed_cfg.min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        # Pass local_epochs to clients via on_fit_config_fn
        on_fit_config_fn=_make_fit_config(fed_cfg.local_epochs),
    )

    logger.info(
        "Strategy built: rounds=%d  clients=%d  fraction_fit=%.2f  iid=%s",
        fed_cfg.num_rounds,
        fed_cfg.num_clients,
        fed_cfg.fraction_fit,
        fed_cfg.iid,
    )

    return strategy


# ---------------------------------------------------------------------------
# Config functions
# ---------------------------------------------------------------------------

def _make_fit_config(local_epochs: int) -> Callable[[int], Dict]:
    """Return a fit-config factory that injects ``local_epochs`` each round.

    Parameters
    ----------
    local_epochs:
        Number of local training epochs per round.

    Returns
    -------
    Callable[[int], Dict]
        Function ``(server_round) → {"local_epochs": int}``.
    """
    def on_fit_config_fn(server_round: int) -> Dict:
        return {"local_epochs": local_epochs, "server_round": server_round}
    return on_fit_config_fn


def build_server_config(num_rounds: int = None) -> ServerConfig:
    """Create a Flower ``ServerConfig`` object.

    Parameters
    ----------
    num_rounds:
        Total federation rounds; defaults to ``CFG.fed.num_rounds``.

    Returns
    -------
    ServerConfig
    """
    num_rounds = num_rounds or CFG.fed.num_rounds
    return ServerConfig(num_rounds=num_rounds)


# ---------------------------------------------------------------------------
# (Optional) real gRPC server launcher — not used in simulation mode
# ---------------------------------------------------------------------------

def run_server(
    strategy: FedAvgRUL,
    server_address: str = "[::]:8080",
    num_rounds: int = None,
) -> fl.server.history.History:
    """Start a real Flower gRPC server (for multi-process or remote clients).

    This function is provided for completeness and distributed deployment.
    For single-machine research, use ``simulation.py`` instead.

    Parameters
    ----------
    strategy:
        Configured ``FedAvgRUL`` strategy.
    server_address:
        gRPC address string.
    num_rounds:
        Total federation rounds.

    Returns
    -------
    fl.server.history.History
        Flower training history.
    """
    num_rounds = num_rounds or CFG.fed.num_rounds
    server_cfg = build_server_config(num_rounds)

    logger.info("Starting Flower server at %s for %d rounds …", server_address, num_rounds)

    history = fl.server.start_server(
        server_address=server_address,
        config=server_cfg,
        strategy=strategy,
    )

    strategy.finalize()
    return history
