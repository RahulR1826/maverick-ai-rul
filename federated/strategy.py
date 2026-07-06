"""
federated/strategy.py
---------------------
Custom Flower strategy extending FedAvg with:
  * Per-round metric aggregation (RMSE, MAE, NASA score, loss)
  * TensorBoard and CSV logging of aggregated server-side metrics
  * Persistence of the globally aggregated model after every round

The strategy is intentionally thin — it delegates weight aggregation to the
parent ``FedAvg`` class and only adds instrumentation on top.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from training.utils import CSVLogger, make_writer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------

def _weighted_average(metrics_list: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute a sample-weighted average of client metrics.

    Parameters
    ----------
    metrics_list:
        List of ``(n_samples, metrics_dict)`` tuples returned by clients.

    Returns
    -------
    Metrics
        Aggregated metric dictionary (keys identical to client metrics).
    """
    if not metrics_list:
        return {}

    total_samples = sum(n for n, _ in metrics_list)
    aggregated: Dict[str, float] = {}

    # Collect all unique metric keys
    all_keys = set()
    for _, m in metrics_list:
        all_keys.update(m.keys())

    for key in all_keys:
        weighted_sum = sum(
            n * m[key] for n, m in metrics_list if key in m
        )
        aggregated[key] = weighted_sum / total_samples

    return aggregated


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class FedAvgRUL(FedAvg):
    """FedAvg strategy augmented with RUL-specific metric logging.

    Parameters
    ----------
    log_dir:
        Root directory for TensorBoard event files and CSV logs.
    checkpoint_dir:
        Directory where per-round global model weights are saved.
    run_name:
        String tag for naming log files (default ``"federated"``).
    save_every:
        Save a model checkpoint every ``save_every`` rounds
        (0 = only at the end).
    **kwargs:
        All remaining keyword arguments are forwarded to ``FedAvg.__init__``.
    """

    def __init__(
        self,
        log_dir: str,
        checkpoint_dir: str,
        run_name: str = "federated",
        save_every: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.save_every = save_every
        self._round = 0

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self._writer = make_writer(log_dir, run_name)
        self._csv = CSVLogger(
            filepath=os.path.join(log_dir, f"{run_name}_server_metrics.csv"),
            fieldnames=[
                "round", "fit_loss", "eval_loss",
                "eval_rmse", "eval_mae", "eval_nasa_score",
                "n_fit_clients", "n_eval_clients",
            ],
        )

        # Track history for convergence plots
        self.history: Dict[str, list] = {
            "round": [],
            "fit_loss": [],
            "eval_loss": [],
            "eval_rmse": [],
            "eval_mae": [],
            "eval_nasa_score": [],
        }

    # ------------------------------------------------------------------
    # FedAvg hooks
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates and log training metrics."""
        self._round = server_round

        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # ── Save global model ────────────────────────────────────────────── #
        if aggregated_params is not None and (
            self.save_every > 0 and server_round % self.save_every == 0
        ):
            self._save_global_model(aggregated_params, server_round)

        # ── Aggregate fit metrics from clients ───────────────────────────── #
        fit_metrics_list = [
            (res.num_examples, res.metrics)
            for _, res in results
            if res.metrics
        ]
        fit_agg = _weighted_average(fit_metrics_list)
        fit_loss = fit_agg.get("train_loss", float("nan"))

        logger.info(
            "Round %d | fit clients=%d | avg train_loss=%.5f",
            server_round, len(results), fit_loss,
        )

        self._writer.add_scalar("Fit/train_loss", fit_loss, server_round)

        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and log all metrics."""
        loss_agg, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        eval_metrics_list = [
            (res.num_examples, res.metrics)
            for _, res in results
            if res.metrics
        ]
        eval_agg = _weighted_average(eval_metrics_list)

        rmse = eval_agg.get("rmse", float("nan"))
        mae = eval_agg.get("mae", float("nan"))
        nasa = eval_agg.get("nasa_score", float("nan"))
        eval_loss = loss_agg if loss_agg is not None else float("nan")

        # ── TensorBoard ──────────────────────────────────────────────────── #
        self._writer.add_scalar("Eval/loss", eval_loss, server_round)
        self._writer.add_scalar("Eval/rmse", rmse, server_round)
        self._writer.add_scalar("Eval/mae", mae, server_round)
        self._writer.add_scalar("Eval/nasa_score", nasa, server_round)

        # ── History ──────────────────────────────────────────────────────── #
        self.history["round"].append(server_round)
        self.history["eval_loss"].append(eval_loss)
        self.history["eval_rmse"].append(rmse)
        self.history["eval_mae"].append(mae)
        self.history["eval_nasa_score"].append(nasa)

        # ── CSV ──────────────────────────────────────────────────────────── #
        self._csv.log({
            "round": server_round,
            "fit_loss": "",          # filled in aggregate_fit log only
            "eval_loss": round(eval_loss, 6),
            "eval_rmse": round(rmse, 4),
            "eval_mae": round(mae, 4),
            "eval_nasa_score": round(nasa, 2),
            "n_fit_clients": "",
            "n_eval_clients": len(results),
        })

        logger.info(
            "Round %d | eval clients=%d | loss=%.5f | RMSE=%.3f | MAE=%.3f | NASA=%.2f",
            server_round, len(results), eval_loss, rmse, mae, nasa,
        )

        return loss_agg, aggregated_metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_global_model(self, parameters: Parameters, server_round: int) -> None:
        """Persist the current global model weights to disk.

        Parameters
        ----------
        parameters:
            Flower Parameters object (list of NumPy arrays).
        server_round:
            Current federation round number.
        """
        import numpy as np
        import torch

        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        save_path = os.path.join(
            self.checkpoint_dir, f"{self.run_name}_round_{server_round:03d}.pt"
        )
        torch.save({"round": server_round, "parameters": ndarrays}, save_path)
        logger.debug("Global model saved → %s", save_path)

    def finalize(self) -> None:
        """Close TensorBoard writer at end of simulation."""
        self._writer.close()
        logger.info("FedAvgRUL strategy finalised.")
