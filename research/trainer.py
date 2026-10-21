"""
research/trainer.py
-------------------
Deterministic training for the three methods compared in the paper.

All methods share: the LSTM from ``models/lstm_model.py``; Adam (lr, weight
decay from config); MSE loss; gradient-norm clipping; mini-batch size; the
17-feature/30-cycle windows; the same 10 validation engines; the same 100 test
engines.  Randomness streams (all derived from the run seed):

  init      -> model initialisation (identical initial weights for every
               method/client that uses the same seed)
  partition -> client partition (see research/partition.py)
  shuffle   -> mini-batch order, one torch.Generator per client / per model
  train     -> dropout masks (global torch RNG, consumed in a fixed order
               because clients are simulated sequentially)

Checkpoint-selection policy (identical for all methods unless the
``centralized_legacy_es`` reference protocol is requested):
  * the model is evaluated on the validation engines every ``E`` epochs
    (= after every FedAvg round; every ``select_every`` epochs for centralized
    and local-only training), giving the same number R of candidates;
  * the candidate with the lowest validation RMSE is the "selected" model;
  * the model after the full budget is the "final" model.
Both are evaluated once on the untouched test set and both are reported.

Unavoidable differences (documented, not hidden):
  * FedAvg clients start every round with a fresh Adam state (optimizer state is
    not communicated in FedAvg); centralized/local-only keep one Adam state for
    the whole run.
  * Validation for FedAvg is computed on the server with a held-out set of
    engines; a real deployment may not have such a server-side set.
"""

import copy
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.lstm_model import build_model
from research import metrics as M
from research.partition import derive_seed


class NonFiniteLossError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def setup_determinism(deterministic: bool) -> Dict[str, object]:
    # Must be in the environment before the first cuBLAS call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {"cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now(device: torch.device) -> float:
    _sync(device)
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Model / batches
# ---------------------------------------------------------------------------

def init_model(model_cfg: Dict, n_features: int, seed: int, device: torch.device) -> nn.Module:
    """Initialise on CPU from the 'init' stream so weights do not depend on device."""
    torch.manual_seed(derive_seed(seed, "init"))
    m = build_model(input_size=n_features, hidden_size=model_cfg["hidden_size"],
                    num_layers=model_cfg["num_layers"], dropout=model_cfg["dropout"],
                    bidirectional=model_cfg["bidirectional"])
    return m.to(device)


def make_optimizer(model: nn.Module, train_cfg: Dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"],
                            weight_decay=train_cfg["weight_decay"])


def train_epoch(model, X: torch.Tensor, y: torch.Tensor, opt, gen: torch.Generator,
                batch_size: int, grad_clip: Optional[float]) -> float:
    """One pass over (X, y) in shuffled mini-batches (last partial batch kept)."""
    model.train()
    n = X.shape[0]
    perm = torch.randperm(n, generator=gen).to(X.device)
    crit = nn.MSELoss()
    tot, cnt = 0.0, 0
    for s in range(0, n, batch_size):
        b = perm[s:s + batch_size]
        opt.zero_grad(set_to_none=True)
        loss = crit(model(X[b]), y[b])
        if not torch.isfinite(loss):
            raise NonFiniteLossError(f"Non-finite training loss {loss.item()}")
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()
        tot += loss.item() * len(b)
        cnt += len(b)
    return tot / cnt


@torch.no_grad()
def predict(model, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = [model(X[s:s + batch]).float().cpu() for s in range(0, X.shape[0], batch)]
    p = torch.cat(out).numpy().astype(np.float64)
    if not np.isfinite(p).all():
        raise NonFiniteLossError("Non-finite predictions")
    return p


def val_metrics(model, Xv: torch.Tensor, yv_np: np.ndarray) -> Dict[str, float]:
    p = predict(model, Xv)
    return {"val_rmse": M.rmse(yv_np, p), "val_mae": M.mae(yv_np, p), "val_mse": float(np.mean((p - yv_np) ** 2))}


def snapshot(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class Tensors:
    """Data moved to the compute device once per run."""

    def __init__(self, data, device):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.Xtr, self.ytr = t(data.train.X), t(data.train.y)
        self.Xv, self.yv_np = t(data.val.X), data.val.y.astype(np.float64)
        self.Xte, self.yte_np = t(data.test.X), data.test.y.astype(np.float64)


def test_eval(model, T: Tensors) -> Tuple[Dict[str, float], np.ndarray]:
    p = predict(model, T.Xte)
    return M.error_summary(T.yte_np, p), p


# ---------------------------------------------------------------------------
# Single-model training with periodic validation (centralized and local-only)
# ---------------------------------------------------------------------------

def train_single(model, X, y, T: Tensors, train_cfg: Dict, epochs: int, select_every: int,
                 gen: torch.Generator, device, early_stopping: bool = False, patience: int = 10,
                 min_delta: float = 1e-4, lr_plateau: bool = False,
                 log: Optional[Callable[[str], None]] = None) -> Dict:
    opt = make_optimizer(model, train_cfg)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
             if lr_plateau else None)
    hist, best, best_state, best_epoch = [], float("inf"), None, None
    es_best, es_wait, stopped_epoch = float("inf"), 0, None
    t0 = now(device)
    train_time = 0.0
    for ep in range(1, epochs + 1):
        a = now(device)
        loss = train_epoch(model, X, y, opt, gen, train_cfg["batch_size"], train_cfg["grad_clip"])
        train_time += now(device) - a
        vm = val_metrics(model, T.Xv, T.yv_np)
        is_candidate = (ep % select_every == 0) or (ep == epochs)
        row = {"epoch": ep, "train_loss": loss, **vm, "lr": opt.param_groups[0]["lr"],
               "candidate": int(is_candidate), "elapsed_s": now(device) - t0}
        hist.append(row)
        if sched is not None:
            sched.step(vm["val_mse"])
        if is_candidate and vm["val_rmse"] < best:
            best, best_state, best_epoch = vm["val_rmse"], snapshot(model), ep
        if early_stopping:
            if vm["val_mse"] < es_best - min_delta:
                es_best, es_wait = vm["val_mse"], 0
            else:
                es_wait += 1
                if es_wait >= patience:
                    stopped_epoch = ep
                    if not is_candidate and vm["val_rmse"] < best:
                        best, best_state, best_epoch = vm["val_rmse"], snapshot(model), ep
                    break
        if log and (ep % max(1, select_every) == 0 or ep == 1):
            log(f"    epoch {ep:3d}/{epochs} loss={loss:9.3f} val_rmse={vm['val_rmse']:.3f}")
    final_state = snapshot(model)
    return {"history": hist, "best_val_rmse": best, "best_epoch": best_epoch, "best_state": best_state,
            "final_state": final_state, "epochs_run": len(hist), "early_stopped_at": stopped_epoch,
            "train_compute_s": train_time, "wall_s": now(device) - t0}


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fedavg_aggregate(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    """Sample-size-weighted average (McMahan et al. 2017; Flower FedAvg default)."""
    if not states or len(states) != len(weights) or min(weights) <= 0:
        raise ValueError("fedavg_aggregate needs one positive weight per client state")
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    out = {}
    for k in states[0]:
        acc = sum(wi * s[k].to(torch.float64) for wi, s in zip(w.tolist(), states))
        out[k] = acc.to(states[0][k].dtype)
    return out


def run_fedavg(model, client_idx: List[np.ndarray], T: Tensors, train_cfg: Dict, fed_cfg: Dict,
               seed: int, device, log: Optional[Callable[[str], None]] = None) -> Dict:
    R, E, K = fed_cfg["num_rounds"], fed_cfg["local_epochs"], len(client_idx)
    idx_t = [torch.from_numpy(i).to(device) for i in client_idx]
    shards = [(T.Xtr[i], T.ytr[i]) for i in idx_t]
    gens = [torch.Generator().manual_seed(derive_seed(seed, "shuffle", "client", c)) for c in range(K)]
    sizes = [int(len(i)) for i in client_idx]
    global_state = snapshot(model)
    hist, client_rows = [], []
    best, best_state, best_round = float("inf"), None, None
    t0 = now(device)
    for r in range(1, R + 1):
        r0 = now(device)
        states, ctimes = [], []
        for c in range(K):
            model.load_state_dict(global_state)          # download
            opt = make_optimizer(model, train_cfg)        # fresh local optimizer each round
            c0 = now(device)
            losses = [train_epoch(model, shards[c][0], shards[c][1], opt, gens[c],
                                  train_cfg["batch_size"], train_cfg["grad_clip"]) for _ in range(E)]
            ct = now(device) - c0
            ctimes.append(ct)
            states.append(snapshot(model))                # upload
            client_rows.append({"round": r, "client_id": c, "n_windows": sizes[c],
                                "local_train_s": ct, "last_epoch_loss": losses[-1], "first_epoch_loss": losses[0]})
        a0 = now(device)
        global_state = fedavg_aggregate(states, sizes)
        agg_t = now(device) - a0
        model.load_state_dict(global_state)
        v0 = now(device)
        vm = val_metrics(model, T.Xv, T.yv_np)
        val_t = now(device) - v0
        if vm["val_rmse"] < best:
            best, best_state, best_round = vm["val_rmse"], snapshot(model), r
        hist.append({"round": r, **vm, "round_wall_s": now(device) - r0,
                     "sum_client_train_s": float(sum(ctimes)), "max_client_train_s": float(max(ctimes)),
                     "aggregation_s": agg_t, "server_val_s": val_t,
                     "parallel_round_estimate_s": float(max(ctimes)) + agg_t,
                     "mean_client_last_loss": float(np.mean([x["last_epoch_loss"] for x in client_rows[-K:]])),
                     "elapsed_s": now(device) - t0})
        if log:
            log(f"    round {r:3d}/{R} val_rmse={vm['val_rmse']:.3f} round_s={hist[-1]['round_wall_s']:.2f}")
    return {"history": hist, "client_rounds": client_rows, "best_val_rmse": best, "best_round": best_round,
            "best_state": best_state, "final_state": snapshot(model), "wall_s": now(device) - t0,
            "client_sizes": sizes}
