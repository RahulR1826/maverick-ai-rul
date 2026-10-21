# Federated Learning for RUL Prediction — NASA C-MAPSS

A **research-quality** implementation of Federated Learning for **Remaining Useful Life (RUL) prediction** of aircraft engines using the NASA C-MAPSS dataset.  Multiple simulated airlines collaboratively train an LSTM model **without sharing raw sensor data**, using the [Flower](https://flower.ai/) federated learning framework.

> Keeping raw data on clients is **not** a formal privacy guarantee: FedAvg model updates can leak information about training data. No differential privacy or secure aggregation is implemented.

---

## Reproducible paper experiments (v2 protocol) — start here

The paper results (NICEDT-2027) are produced by the `research/` package, **not** by the older
`run_experiments.py` (kept unchanged for reference; its outputs in `results/` other than
`results/v2/` come from the earlier, non-budget-matched protocol). The v2 pipeline re-uses the
original LSTM (`models/lstm_model.py`) and reproduces the original FD001 preprocessing exactly
(verified by `tests/test_data_partition.py::test_fd001_matches_original_cached_arrays`), but runs
FedAvg as a deterministic, sequential in-process simulation so that every random choice is seeded.

| What | Where |
|---|---|
| All settings, experiments, sweeps, seed lists | `configs/fd001_protocol.json` |
| Data prep (engine-aware windows, leakage checks) | `research/data.py` |
| IID / Dirichlet / whole-engine partitions, min-size safeguard | `research/partition.py` |
| Centralized, FedAvg, local-only training, checkpoint selection | `research/trainer.py` |
| RMSE, MAE, NASA score, error summaries | `research/metrics.py` |
| Serialized-payload communication accounting | `research/comm.py` |
| Runner (resumable, never overwrites) | `research/run.py` |
| Tables / figures / report | `research/aggregate.py`, `research/plots.py`, `research/report.py` |
| Unit tests | `tests/` |

### Setup

Tested with Python 3.11.9, torch 2.7.1+cu118, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.3.0,
matplotlib 3.10.6 and pytest 9.1.1 on Windows 11 with an RTX 4060 Laptop GPU. The v2 pipeline doesn't need `flwr`.

```bash
pip install -r requirements.txt
```

### Commands

```bash
python -m pytest -q tests
```

```bash
python -m research.run --list
```

```bash
python -m research.run --suite essential --smoke
```

```bash
python -m research.run --suite essential
```

```bash
python -m research.run --suite sweeps
```

```bash
python -m research.run --suite essential --seed-set fallback
```

```bash
python -m research.run --experiments centralized fedavg_iid --seeds 42 123
```

```bash
python -m research.aggregate
```

```bash
python -m research.plots --checkpoint both
```

```bash
python -m research.report
```

```bash
python -m research.paper_tables
```

```bash
python -m research.verify_phase2
```

`verify_phase2` checks the four Phase-2 requirements (5 seeds, fair comparison, partition protocols,
repeated sweeps) against the run outputs on disk and writes `results/v2/PHASE2_VERIFICATION.md`.

Exact sequence used for the paper results (2026-10-21; 5 seeds; about 75 min of GPU time on an RTX 4060 Laptop):

```bash
python -m research.run --suite essential
```

```bash
python -m research.run --suite sweeps
```

```bash
python -m research.run --experiments ablation_fedavg_k1
```

```bash
python -m research.run --experiments centralized centralized_legacy_es fedavg_iid local_iid fedavg_engine local_engine --seeds 42 --out results/v2_timing
```

then `research.aggregate` (once for `results/v2`, once with `--out results/v2_timing --seeds 42`), `research.plots --checkpoint both`,
`research.report` and `research.paper_tables`. The optional `local_dirichlet` experiments (suite `heterogeneity_extra`) were not run.

**Results:** `results/v2/README.md` (index), `results/v2/EXPERIMENT_REPORT.md` (report),
`results/v2/CONCLUSION_CHECK.md` (original findings vs new results), `results/v2/paper_tables/PAPER_VALUES.md`.

**Manuscript:** `paper/updated_sections.tex` holds the revised abstract, protocol, results (tables and figures),
limitations and conclusion, ready to paste into the paper. `paper/figures/` has the figures, and
`python paper/check_numbers.py` checks every number in the text against `results/v2/tables`.

**Not in this repository (excluded by `.gitignore`):** the NASA C-MAPSS data (`CMAPSSData/`, download it from the
NASA Prognostics Data Repository and place the `train_/test_/RUL_FD00x.txt` files there), model checkpoints
(`checkpoints/`, about 850 MB) and cached arrays (`processed/`). Every result, table and figure can be inspected without
them; re-running experiments requires the dataset.

- `--smoke` uses a 2-round / 2-epoch budget and writes to `results/v2_smoke/` (never used in the paper).
- `--dry-run` lists every planned run and whether it is already complete.
- Runs are content-addressed (`<method>-<config-hash>-s<seed>`): identical configurations shared by several
  experiments (e.g. FedAvg-IID = R=20 point of the round sweep) run once. A finished run is skipped; `--force`
  recomputes it and keeps the old directory as `*.prev_<timestamp>`.

### Outputs

- `results/v2/runs/<run_id>/`: `config.json` (resolved config, seed streams, environment, git commit),
  `metrics.json` (test metrics for selected and final checkpoints, validation, timing, communication, partition),
  `history.csv` (per-round/epoch validation RMSE, timing, cumulative traffic), `test_predictions.csv` (per-engine
  predictions and errors), `clients.csv` (per-client windows, engines, RUL distribution), `client_rounds.csv`.
- `checkpoints/v2/<run_id>/selected.pt`, `final.pt` (local-only: one per client).
- `results/v2/tables/*.csv`, `results/v2/figures/*.{png,pdf}`, `results/v2/EXPERIMENT_REPORT.md`.
- `results/v2/logs/`: one log file per runner invocation.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset Placement](#dataset-placement)
4. [Configuration](#configuration)
5. [Preprocessing](#preprocessing)
6. [Centralized Baseline Training](#centralized-baseline-training)
7. [Federated Learning Simulation](#federated-learning-simulation)
8. [Evaluation](#evaluation)
9. [Visualization](#visualization)
10. [Expected Outputs](#expected-outputs)
11. [Architecture Overview](#architecture-overview)
12. [Extending to FD002–FD004](#extending-to-fd002fd004)

---

## Project Structure

```
federated_learning pipeline_rul/
│
├── CMAPSSData/                    ← Raw NASA C-MAPSS files (place here)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── datasets/
│   ├── preprocessing.py           ← Load, normalise, window, cache
│   ├── cmapss_dataset.py          ← PyTorch Dataset + DataLoader factory
│   └── split_clients.py          ← IID / Non-IID / engine-level partitioning
│
├── models/
│   └── lstm_model.py              ← Stacked LSTM + FC head
│
├── training/
│   ├── train.py                   ← Training loop, optimizer, scheduler
│   ├── evaluate.py                ← Checkpoint evaluation on test set
│   └── utils.py                  ← Metrics, early stopping, checkpointing, logging
│
├── federated/
│   ├── client.py                  ← Flower NumPyClient (airline)
│   ├── server.py                  ← Strategy builder + optional gRPC server
│   ├── strategy.py                ← FedAvgRUL — metric logging + checkpointing
│   └── simulation.py             ← In-process Flower simulation runner
│
├── visualization.py              ← Loss curves, convergence, comparison plots
├── config.py                      ← All hyperparameters and paths
├── main.py                        ← Unified CLI entry-point
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU recommended (CPU works but is slower)

### Steps

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU only) Install the correct torch+CUDA build from pytorch.org
#    e.g. for CUDA 12.1:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset Placement

The NASA C-MAPSS dataset must be placed inside the `CMAPSSData/` folder at the project root.

```
CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt   ← optional
├── test_FD002.txt    ← optional
├── RUL_FD002.txt     ← optional
... (FD003, FD004)
```

> The dataset is freely available from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

---

## Configuration

All hyperparameters live in `config.py`.  You can either edit the file directly or override any setting on the command line:

| Config class | Key parameters |
|---|---|
| `DataConfig` | `dataset_id`, `rul_cap`, `window_size`, `stride`, `drop_sensors` |
| `ModelConfig` | `hidden_size`, `num_layers`, `dropout`, `bidirectional` |
| `TrainConfig` | `epochs`, `batch_size`, `learning_rate`, `patience`, `device` |
| `FedConfig` | `num_clients`, `num_rounds`, `local_epochs`, `fraction_fit`, `iid`, `dirichlet_alpha` |

### Key defaults

```python
rul_cap      = 125    # piecewise-linear RUL cap
window_size  = 30     # sliding window length (cycles)
hidden_size  = 128    # LSTM hidden units
num_layers   = 2      # stacked LSTM layers
dropout      = 0.2
epochs       = 50     # max centralized epochs
num_clients  = 5      # simulated airlines
num_rounds   = 20     # federation rounds
local_epochs = 5      # local epochs per round
```

---

## Preprocessing

Preprocessing loads the raw space-delimited C-MAPSS files, computes piecewise-linear RUL labels (capped at `rul_cap`), drops low-variance sensors, normalises features with a training-only `MinMaxScaler`, and builds sliding windows.  Results are cached as `.npy` files.

```bash
# Preprocess FD001 (default)
python main.py --mode preprocess

# Force re-preprocessing
python main.py --mode preprocess --force-preprocess

# Different subset
python main.py --mode preprocess --dataset FD002
```

Cached output appears in `processed/FD001/`:
```
processed/FD001/
├── X_train.npy    (N_train, 30, 14)
├── y_train.npy    (N_train,)
├── X_val.npy
├── y_val.npy
├── X_test.npy     (100, 30, 14)  ← one window per engine
├── y_test.npy
└── scaler.joblib
```

### Dropped sensors (FD001/FD003)

Sensors `s1, s5, s6, s10, s16, s18, s19` are removed because they are near-constant and contribute no discriminative signal.  The remaining **14 features** include 3 operational settings + 11 active sensors.

---

## Centralized Baseline Training

```bash
python main.py --mode centralized
```

Optional overrides:

```bash
python main.py --mode centralized --dataset FD001 --epochs 80 --lr 5e-4 --batch-size 128
```

Outputs:
- `checkpoints/centralized_best.pt`  — best checkpoint (lowest val loss)
- `logs/centralized_metrics.csv`     — per-epoch metrics
- `logs/centralized/`               — TensorBoard event files

---

## Federated Learning Simulation

```bash
# IID split (default)
python main.py --mode federated

# Non-IID split (Dirichlet, α=0.5)
python main.py --mode federated --non-iid --alpha 0.5

# Custom scale
python main.py --mode federated --clients 10 --rounds 30 --local-epochs 3
```

Each client simulates one airline owning a private engine fleet.  The Flower simulation runs **in-process** (no network needed).

Outputs:
- `checkpoints/federated_best.pt`          — final global model
- `checkpoints/federated_round_XXX.pt`    — per-round snapshots
- `logs/federated_server_metrics.csv`      — round-wise aggregated metrics
- `logs/federated/`                        — TensorBoard event files

---

## Evaluation

```bash
# Evaluate centralized checkpoint
python main.py --mode evaluate --run centralized

# Evaluate federated checkpoint
python main.py --mode evaluate --run federated
```

Metrics reported:

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error (cycles) |
| **MAE** | Mean Absolute Error (cycles) |
| **NASA Score** | Asymmetric penalty function — late predictions penalised more |

Prediction CSV saved to `results/<run>_FD001_predictions.csv`.

---

## Visualization

```bash
python main.py --mode visualize
```

Generated plots (saved to `results/`):

| File | Content |
|---|---|
| `centralized_loss_FD001.png` | Train vs val loss curves |
| `centralized_metrics_FD001.png` | Val RMSE and MAE |
| `federated_convergence_FD001.png` | 4-panel: loss / RMSE / MAE / NASA score per round |
| `comparison_FD001.png` | Side-by-side centralized vs federated RMSE and MAE |

---

## Run All Stages

```bash
python main.py --mode all
```

Runs: preprocess → centralized → federated → evaluate (both) → visualize

---

## Expected Outputs

After a full run on FD001 with default settings:

| Run | RMSE (cycles) | MAE (cycles) | NASA Score |
|---|---|---|---|
| Centralized | ~14–18 | ~10–13 | ~200–600 |
| Federated (IID) | ~16–22 | ~12–16 | ~300–800 |

> Exact numbers vary by random seed and hardware.

### TensorBoard

```bash
tensorboard --logdir logs/
```

Navigate to `http://localhost:6006` to view live curves.

---

## Architecture Overview

```
Input: (batch, 30, 14)
       ↓
LSTM (128 hidden, 2 layers, dropout=0.2)
       ↓
Last hidden state h_n[-1]: (batch, 128)
       ↓
Dropout(0.2)
       ↓
Linear(128 → 64) + ReLU
       ↓
Linear(64 → 1)
       ↓
Predicted RUL (scalar, cycles)
```

**Loss**: MSELoss  
**Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)  
**Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)  
**Gradient clipping**: max_norm=1.0  

---

## Extending to FD002–FD004

1. **Change dataset** via CLI or `config.py`:
   ```bash
   python main.py --mode all --dataset FD003
   ```

2. **Sensor dropping**: FD002/FD004 have 6 operating conditions.  You may want to keep more sensors.  Override `DataConfig.drop_sensors` in `config.py`:
   ```python
   data: DataConfig = field(default_factory=lambda: DataConfig(
       dataset_id="FD002",
       drop_sensors=[1, 5, 6],  # keep more sensors for multi-regime
   ))
   ```

3. **More engines** → increase `num_clients` proportionally.

---

## Code Quality

- Full **type hints** on all public functions
- **Docstrings** (NumPy style) on all modules, classes, and functions
- **PEP 8** compliant
- No placeholder `pass` or `TODO` stubs — all functions are complete
- Modular design: data, model, training, federated, and viz are fully decoupled
