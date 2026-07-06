"""
demo_run.py
-----------
Quick demo script: runs all pipeline stages with reduced epochs/rounds
so you can see real output within a few minutes.

Run: python demo_run.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

from config import CFG
from datasets.cmapss_dataset import build_datasets, make_dataloader
from datasets.preprocessing import preprocess_dataset, get_n_features
from datasets.split_clients import split_for_clients
from federated.client import _get_params, _set_params
from federated.server import build_strategy, build_server_config
from federated.simulation import _make_client_fn, _save_final_model
from models.lstm_model import build_model
from training.train import train_model, evaluate_model, predict
from training.utils import resolve_device, set_seed, evaluate_predictions
from training.evaluate import evaluate_checkpoint
from visualization import (
    plot_centralized_curves,
    plot_federated_convergence,
    plot_comparison,
    plot_predictions,
    plot_client_distribution,
)

import flwr as fl
from flwr.simulation import start_simulation

# ── Reduced settings for demo speed ─────────────────────────────────────────
DEMO_CENTRALIZED_EPOCHS = 10
DEMO_FED_ROUNDS         = 5
DEMO_LOCAL_EPOCHS       = 3
DEMO_CLIENTS            = 3
DEMO_BATCH              = 512

DIVIDER = "=" * 62


def section(title):
    print()
    print(DIVIDER)
    print(f"  {title}")
    print(DIVIDER)


# ============================================================================
# STEP 1: PREPROCESS
# ============================================================================
section("STEP 1 — DATA PREPROCESSING  (FD001)")

splits = preprocess_dataset(dataset_id="FD001")
n_features = get_n_features("FD001")

for split, (X, y) in splits.items():
    print(f"  {split:6s}: shape={X.shape}   RUL in [{y.min():.0f}, {y.max():.0f}]")
print(f"  n_features       : {n_features}")
print(f"  window_size      : {CFG.data.window_size}")
print(f"  RUL cap          : {CFG.data.rul_cap}")

# ============================================================================
# STEP 2: CENTRALIZED BASELINE
# ============================================================================
section(f"STEP 2 — CENTRALIZED BASELINE  ({DEMO_CENTRALIZED_EPOCHS} epochs)")

set_seed(42)
device = resolve_device("auto")
print(f"  Device           : {device}")

CFG.train.epochs    = DEMO_CENTRALIZED_EPOCHS
CFG.train.patience  = 20
CFG.train.batch_size = DEMO_BATCH
CFG.model.input_size = n_features

train_ds, val_ds, test_ds = build_datasets(splits, rul_cap=CFG.data.rul_cap)
train_loader = make_dataloader(train_ds, batch_size=DEMO_BATCH, shuffle=True)
val_loader   = make_dataloader(val_ds,   batch_size=DEMO_BATCH, shuffle=False)
test_loader  = make_dataloader(test_ds,  batch_size=DEMO_BATCH, shuffle=False)

model = build_model(
    input_size=n_features,
    hidden_size=CFG.model.hidden_size,
    num_layers=CFG.model.num_layers,
    dropout=CFG.model.dropout,
    device=device,
)
print(f"  Model parameters : {model.count_parameters():,}")
print(f"  Training windows : {len(train_ds):,}")

t0 = time.time()
c_history = train_model(
    model=model, train_loader=train_loader, val_loader=val_loader,
    cfg=CFG.train, device=device, run_name="centralized",
    checkpoint_dir="checkpoints", log_dir="logs", rul_cap=CFG.data.rul_cap,
)
c_elapsed = time.time() - t0

print()
print(f"  [CENTRALIZED RESULTS]")
print(f"  Time elapsed     : {c_elapsed:.1f}s")
print(f"  Best epoch       : {c_history.get('best_epoch', '?')}")
print(f"  Best val RMSE    : {min(c_history['val_rmse']):.2f} cycles")
print(f"  Best val MAE     : {min(c_history['val_mae']):.2f} cycles")

# Test-set evaluation using best checkpoint
print()
print("  --- Test Set Evaluation ---")
y_true, y_pred = predict(model, test_loader, device)
c_metrics = evaluate_predictions(y_true, y_pred, prefix="centralized/test/")

# ============================================================================
# STEP 3: FEDERATED SIMULATION
# ============================================================================
section(
    f"STEP 3 — FEDERATED SIMULATION  "
    f"({DEMO_CLIENTS} clients, {DEMO_FED_ROUNDS} rounds, {DEMO_LOCAL_EPOCHS} local epochs)"
)

# Temporarily override federation config
CFG.fed.num_clients   = DEMO_CLIENTS
CFG.fed.num_rounds    = DEMO_FED_ROUNDS
CFG.fed.local_epochs  = DEMO_LOCAL_EPOCHS
CFG.train.batch_size  = DEMO_BATCH

X_train, y_train = splits["train"]
val_data = splits["val"]
client_data = split_for_clients(X_train, y_train, cfg=CFG.fed, seed=42)

print(f"  Data split       : {'IID' if CFG.fed.iid else 'Non-IID'}")
for cid, (Xc, yc) in client_data.items():
    print(f"  Client {cid}         : {len(Xc):,} windows  |  RUL in [{yc.min():.0f}, {yc.max():.0f}]")

# Fresh global model for federation
fed_model = build_model(
    input_size=n_features,
    hidden_size=CFG.model.hidden_size,
    num_layers=CFG.model.num_layers,
    dropout=CFG.model.dropout,
    device=device,
)
initial_params = fl.common.ndarrays_to_parameters(_get_params(fed_model))

# Server-side evaluation function
def server_eval_fn(server_round, parameters, config):
    _set_params(fed_model, parameters)
    from datasets.cmapss_dataset import CMAPSSDataset
    X_val, y_val = val_data
    from datasets.cmapss_dataset import CMAPSSDataset, make_dataloader as mdl
    vds = CMAPSSDataset(X_val, y_val, rul_cap=CFG.data.rul_cap)
    vl = mdl(vds, batch_size=DEMO_BATCH, shuffle=False)
    val_loss, rmse, mae = evaluate_model(fed_model, vl, device, rul_cap=CFG.data.rul_cap)
    print(f"  [Server] Round {server_round:2d} | val_loss={val_loss:.2f} | RMSE={rmse:.2f} | MAE={mae:.2f}")
    return val_loss, {"rmse": rmse, "mae": mae}

strategy = build_strategy(
    initial_parameters=initial_params,
    fed_cfg=CFG.fed,
    log_dir="logs",
    checkpoint_dir="checkpoints",
    run_name="federated",
    evaluate_fn=server_eval_fn,
)

client_fn = _make_client_fn(
    client_data=client_data,
    val_data=val_data,
    cfg=CFG,
    n_features=n_features,
    device=device,
)

print()
print("  Starting Flower simulation...")
t0 = time.time()
fl_history = start_simulation(
    client_fn=client_fn,
    num_clients=CFG.fed.num_clients,
    config=build_server_config(CFG.fed.num_rounds),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
    ray_init_args={"ignore_reinit_error": True, "include_dashboard": False, "num_cpus": 2},
)
fed_elapsed = time.time() - t0

print()
print(f"  [FEDERATED RESULTS]")
print(f"  Time elapsed     : {fed_elapsed:.1f}s")

sh = strategy.history
if sh.get("eval_rmse"):
    print(f"  Final RMSE       : {sh['eval_rmse'][-1]:.2f} cycles")
    print(f"  Final MAE        : {sh['eval_mae'][-1]:.2f} cycles")
    print(f"  Final NASA score : {sh['eval_nasa_score'][-1]:.2f}")

# Test-set evaluation with federated model
print()
print("  --- Federated Test Set Evaluation ---")
y_true_f, y_pred_f = predict(fed_model, test_loader, device)
f_metrics = evaluate_predictions(y_true_f, y_pred_f, prefix="federated/test/")

_save_final_model(fed_model, strategy, "checkpoints/federated_best.pt")
strategy.finalize()

# ============================================================================
# STEP 4: COMPARISON SUMMARY
# ============================================================================
section("STEP 4 — COMPARISON SUMMARY")

print(f"  {'Metric':<30} {'Centralized':>14} {'Federated':>12}")
print("  " + "-" * 58)
for key in ["rmse", "mae", "nasa_score"]:
    ck = f"centralized/test/{key}"
    fk = f"federated/test/{key}"
    cv = c_metrics.get(ck, float("nan"))
    fv = f_metrics.get(fk, float("nan"))
    diff = fv - cv
    sign = "+" if diff > 0 else ""
    print(f"  {key:<30} {cv:>14.3f} {fv:>12.3f}   (diff: {sign}{diff:.3f})")

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================
section("STEP 5 — GENERATING PLOTS")
os.makedirs("results", exist_ok=True)

p1 = plot_centralized_curves(dataset_id="FD001")
p2 = plot_federated_convergence(dataset_id="FD001")
p3 = plot_comparison(dataset_id="FD001")

import numpy as np
import pandas as pd
pred_csv = "results/centralized_FD001_predictions.csv"
pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(pred_csv, index=False)
p4 = plot_predictions(pred_csv, run_name="centralized", dataset_id="FD001")

import numpy as np
client_labels = {cid: yc for cid, (_, yc) in client_data.items()}
p5 = plot_client_distribution(client_labels, dataset_id="FD001")

saved = [p for p in [*p1, *p2, *p3, p4, p5] if p]
for p in saved:
    print(f"  Saved: {p}")

print()
print(DIVIDER)
print("  DEMO COMPLETE!")
print(f"  Centralized RMSE : {min(c_history['val_rmse']):.2f} cycles (val)")
if sh.get("eval_rmse"):
    print(f"  Federated RMSE   : {sh['eval_rmse'][-1]:.2f} cycles (val, round {DEMO_FED_ROUNDS})")
print(f"  Logs             : logs/")
print(f"  Checkpoints      : checkpoints/")
print(f"  Plots            : results/")
print(DIVIDER)
print()
print("To view TensorBoard curves, run:")
print("  tensorboard --logdir logs")
