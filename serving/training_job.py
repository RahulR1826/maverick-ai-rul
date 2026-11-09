"""
serving/training_job.py
-----------------------
A short, real FedAvg training run that streams its progress (one event per round)
for the dashboard. Uses the same data, partitions, model and trainer as the paper
experiments (research/), on the GPU when available. Nothing is saved; the run is a
demonstration and is not part of the paper's results.

Only one job runs at a time. Budgets are capped so a run finishes in about a minute.
"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from research import config as C
from research import trainer as TR
from research.comm import fedavg_traffic, payload_info
from research.data import get_data
from research.partition import client_stats, derive_seed, partition

LIMITS = {"num_clients": (1, 10), "num_rounds": (1, 30), "local_epochs": (1, 10), "max_passes": 100}
_lock = threading.Lock()


class Cancelled(Exception):
    pass


@dataclass
class JobParams:
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 2
    partition: str = "iid_window"
    alpha: Optional[float] = None
    seed: int = 42

    def validate(self) -> None:
        for k in ("num_clients", "num_rounds", "local_epochs"):
            lo, hi = LIMITS[k]
            if not lo <= getattr(self, k) <= hi:
                raise ValueError(f"{k} must be between {lo} and {hi}")
        if self.num_rounds * self.local_epochs > LIMITS["max_passes"]:
            raise ValueError(f"num_rounds × local_epochs must be at most {LIMITS['max_passes']} (the paper's budget)")
        if self.partition not in ("iid_window", "dirichlet_window", "engine"):
            raise ValueError("partition must be iid_window, dirichlet_window or engine")
        if self.partition == "dirichlet_window" and not (self.alpha and 0.01 <= self.alpha <= 100):
            raise ValueError("dirichlet_window needs alpha between 0.01 and 100")


def busy() -> bool:
    return _lock.locked()


def run(params: JobParams, emit: Callable[[Dict], None], cancelled: Callable[[], bool]) -> None:
    """Run FedAvg and call ``emit`` with events: start, partition, round (×R), done | error | cancelled."""
    params.validate()
    if not _lock.acquire(blocking=False):
        raise RuntimeError("A training job is already running")
    try:
        proto = C.load_protocol()
        base = proto["base"]
        fed = dict(base["fed"], num_clients=params.num_clients, num_rounds=params.num_rounds, local_epochs=params.local_epochs,
                   partition=params.partition, dirichlet_alpha=params.alpha if params.partition == "dirichlet_window" else None)
        TR.setup_determinism(True)
        device = TR.resolve_device("auto")
        data = get_data(base["data"], C.DATA_DIR)
        T = TR.Tensors(data, device)
        t0 = time.perf_counter()
        emit({"type": "start", "params": params.__dict__, "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
        part = partition(data.train, fed, base["data"]["rul_cap"], params.seed)
        cs = client_stats(data.train, part["indices"], base["data"]["rul_cap"], fed["n_rul_bins"])
        emit({"type": "partition", "clients": [{k: c[k] for k in ("client_id", "n_windows", "n_engines", "rul_mean")} for c in cs]})
        model = TR.init_model(base["model"], data.meta["n_features"], params.seed, device)
        payload = payload_info(model)["serialized_payload_bytes"]
        torch.manual_seed(derive_seed(params.seed, "train"))          # same stream as research/run.py

        def on_round(row: Dict) -> None:
            if cancelled():
                raise Cancelled()
            r = int(row["round"])
            emit({"type": "round", "round": r, "of": params.num_rounds, "val_rmse": row["val_rmse"],
                  "round_s": row["round_wall_s"], "cum_MB": 2 * params.num_clients * r * payload / 1e6,
                  "elapsed_s": time.perf_counter() - t0})

        out = TR.run_fedavg(model, part["indices"], T, base["train"], fed, params.seed, device, on_round=on_round)
        model.load_state_dict(out["best_state"])
        sel, _ = TR.test_eval(model, T)
        model.load_state_dict(out["final_state"])
        fin, _ = TR.test_eval(model, T)
        traffic = fedavg_traffic(payload, [params.num_clients] * params.num_rounds)
        emit({"type": "done", "best_round": out["best_round"], "best_val_rmse": out["best_val_rmse"],
              "test_rmse_selected": sel["rmse"], "test_rmse_final": fin["rmse"], "test_nasa_selected": sel["nasa_score"],
              "total_MB": traffic["total_bytes"] / 1e6, "wall_s": time.perf_counter() - t0,
              "note": "Demonstration run; single seed. Paper results use 5 seeds (results/v2)."})
    except Cancelled:
        emit({"type": "cancelled"})
    except Exception as ex:   # reported to the client; the server keeps running
        emit({"type": "error", "message": f"{type(ex).__name__}: {ex}"})
    finally:
        _lock.release()
