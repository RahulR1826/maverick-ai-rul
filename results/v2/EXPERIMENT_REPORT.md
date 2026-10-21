# Experiment Report — Federated Learning for Aircraft Engine Prognostics (v2 protocol)

Generated 2026-10-21 20:07 by `python -m research.report` from `results\v2\tables`. Every number in sections 3–8 is computed from per-run `metrics.json` files; nothing is imputed. Values are mean ± sample SD (ddof = 1) over seeds.

## 1. Protocol

| Item | Setting |
|---|---|
| Dataset | NASA C-MAPSS FD001: 100 training engines, 100 official test engines |
| Train / validation split | engine-level, 10% of training engines (10) held out, drawn with `default_rng(42)` (same engines as the original code); fixed for all seeds and methods |
| Features | 3 operational settings + 14 sensors (dropped sensors [1, 5, 6, 10, 16, 18, 19]) = 17; fixed a priori |
| Normalisation | MinMaxScaler(0,1) fitted on the 90 training engines only; applied unchanged to validation and test |
| Windows / target | 30 cycles, stride 1; label = piece-wise linear RUL capped at 125 at the window's last cycle |
| Test set | last 30 cycles of each of the 100 test engines; truth from RUL_FD001.txt, capped at 125 (yes); used only after model selection |
| Model | LSTM 2×128, dropout 0.2, FC 64→1 (`models/lstm_model.py`, unchanged) |
| Optimiser | Adam lr=0.001, weight decay=1e-05, batch 256, grad-norm clip 1.0, MSE loss, constant LR |
| FedAvg | K=5, R=20, E=5, full participation, sample-weighted averaging, fresh Adam state per client per round |
| Training budget | every method makes R×E = 100 passes over its training data (centralized: 100 epochs; local-only: 100 epochs on its shard) |
| Checkpoint selection | validation RMSE on the 10 held-out engines every E epochs / every round (20 candidates); lowest is 'selected'. The model after the full budget is 'final'. Both are reported. |
| Reference protocol | `centralized_legacy_es` = original code: ≤50 epochs, early stopping patience 10, ReduceLROnPlateau, selection every epoch. Not budget-matched; shown only to quantify the protocol change. |
| Seeds | [42, 123, 2026, 3407, 9001] (3-seed fallback: [42, 123, 2026]); identical lists for every method. A seed fixes model init, partition, mini-batch order and dropout via independent derived streams |
| Determinism | cuDNN deterministic, benchmark off, `torch.use_deterministic_algorithms(True, warn_only=True)`, CUBLAS_WORKSPACE_CONFIG=:4096:8 |

### Partition protocols

- **IID window**: all 15,630 training windows randomly permuted and split into K near-equal shards. Overlapping windows of one engine land on different clients, so every client sees fragments of almost every engine.
- **Dirichlet window (α)**: RUL labels binned into 10 equal-width bins on [0,125]; per bin, client shares ~ Dir(α·1_K). Changes the RUL distribution **and** client sizes (quantity skew), and still spreads engines across clients.
- **Whole-engine**: the 90 training engines randomly assigned (seeded) to K clients, 18 each for K=5; each client owns complete run-to-failure trajectories. Client window counts differ because engine lifetimes differ.
- Every window is assigned to exactly one client (asserted). Total training data is constant across K. Minimum client size: 256 windows; partitions violating it are redrawn (attempt count recorded), never truncated.

**Confounding note.** Window-level vs whole-engine partitions differ in more than heterogeneity: engine ownership (fragmented vs complete trajectories), cross-client overlap of near-duplicate windows, and per-client volume. Dirichlet partitions combine label skew with quantity skew. Differences between these conditions are therefore *not* attributed to heterogeneity alone.


## 2. Centralized vs FedAvg vs local-only — checkpoint = **selected**

| Configuration | n seeds | Test RMSE | Test MAE | NASA score | Best val RMSE | Wall time, clean runs (s) | Model traffic (MB) |
|---|---|---|---|---|---|---|---|
| centralized | 5 | 14.46 ± 0.44 | 10.67 ± 0.43 | 447.9 ± 35.4 | 12.69 ± 0.24 | 131 ± 15 | 0.0 |
| centralized_legacy_es | 5 | 14.55 ± 0.48 | 10.87 ± 0.24 | 441.4 ± 61.3 | 12.79 ± 0.25 | 36 ± 16 | 0.0 |
| ablation_fedavg_k1 | 5 | 13.94 ± 0.89 | 10.35 ± 0.66 | 388.1 ± 99.6 | 12.64 ± 0.23 | n/a (no clean run) | 34.6 |
| fedavg_iid | 5 | 13.24 ± 0.32 | 9.78 ± 0.21 | 304.0 ± 26.6 | 12.36 ± 0.11 | 53 ± 1 | 172.9 |
| local_iid | 5 | 14.58 ± 0.17 | 10.81 ± 0.09 | 414.6 ± 20.4 | 13.16 ± 0.18 | 109 ± 38 (n=4) | 0.0 |
| fedavg_engine | 5 | 13.87 ± 0.19 | 10.15 ± 0.13 | 349.0 ± 26.5 | 12.87 ± 0.34 | 65 ± 7 | 172.9 |
| local_engine | 5 | 16.28 ± 0.50 | 12.13 ± 0.26 | 651.8 ± 80.3 | 14.94 ± 0.29 | 73 ± 15 | 0.0 |

Local-only rows: mean over the K=5 client models of each model's metrics on all 100 test engines.


## 2b. Centralized vs FedAvg vs local-only — checkpoint = **final**

| Configuration | n seeds | Test RMSE | Test MAE | NASA score | Best val RMSE | Wall time, clean runs (s) | Model traffic (MB) |
|---|---|---|---|---|---|---|---|
| centralized | 5 | 14.82 ± 1.10 | 10.99 ± 0.63 | 417.6 ± 85.7 | 12.69 ± 0.24 | 131 ± 15 | 0.0 |
| centralized_legacy_es | 5 | 14.13 ± 0.55 | 10.31 ± 0.31 | 377.5 ± 88.7 | 12.79 ± 0.25 | 36 ± 16 | 0.0 |
| ablation_fedavg_k1 | 5 | 15.96 ± 1.62 | 11.66 ± 1.36 | 603.6 ± 233.1 | 12.64 ± 0.23 | n/a (no clean run) | 34.6 |
| fedavg_iid | 5 | 12.88 ± 0.30 | 9.50 ± 0.16 | 280.8 ± 27.6 | 12.36 ± 0.11 | 53 ± 1 | 172.9 |
| local_iid | 5 | 14.85 ± 0.66 | 10.93 ± 0.56 | 470.1 ± 102.5 | 13.16 ± 0.18 | 109 ± 38 (n=4) | 0.0 |
| fedavg_engine | 5 | 13.74 ± 0.68 | 10.17 ± 0.48 | 355.8 ± 96.5 | 12.87 ± 0.34 | 65 ± 7 | 172.9 |
| local_engine | 5 | 17.84 ± 0.65 | 13.07 ± 0.51 | 858.2 ± 138.2 | 14.94 ± 0.29 | 73 ± 15 | 0.0 |

Local-only rows: mean over the K=5 client models of each model's metrics on all 100 test engines.


## 3. Paired differences vs FedAvg-IID (matched seeds)

`diff = other − FedAvg-IID` per seed (negative = other method has lower error). The exact sign-flip permutation p-value is shown for transparency only: with n = 5 seeds the smallest attainable two-sided p is 0.0625, so **no comparison can be declared statistically significant at α = 0.05 with this design, and none is claimed.**

| Ckpt | Comparison | Metric | n | Mean diff ± SD | Other lower | Per-seed diffs | Sign-flip p |
|---|---|---|---|---|---|---|---|
| selected | centralized - fedavg_iid | rmse | 5 | +1.22 ± 0.56 | 0/5 | [0.286, 1.453, 1.727, 1.143, 1.498] | 0.062 |
| selected | centralized - fedavg_iid | nasa_score | 5 | +143.91 ± 37.40 | 0/5 | [90.403, 155.285, 194.342, 143.235, 136.283] | 0.062 |
| selected | centralized_legacy_es - fedavg_iid | rmse | 5 | +1.31 ± 0.47 | 0/5 | [1.389, 1.123, 0.665, 1.413, 1.957] | 0.062 |
| selected | centralized_legacy_es - fedavg_iid | nasa_score | 5 | +137.35 ± 59.47 | 0/5 | [158.583, 114.788, 53.421, 144.573, 215.398] | 0.062 |
| selected | local_iid - fedavg_iid | rmse | 5 | +1.34 ± 0.42 | 0/5 | [0.745, 1.686, 1.355, 1.162, 1.771] | 0.062 |
| selected | local_iid - fedavg_iid | nasa_score | 5 | +110.57 ± 40.29 | 0/5 | [47.1, 141.322, 125.447, 95.642, 143.351] | 0.062 |
| selected | fedavg_engine - fedavg_iid | rmse | 5 | +0.63 ± 0.42 | 0/5 | [0.212, 0.701, 0.835, 0.225, 1.189] | 0.062 |
| selected | fedavg_engine - fedavg_iid | nasa_score | 5 | +44.97 ± 45.80 | 1/5 | [5.587, 42.019, 86.111, -5.116, 96.25] | 0.125 |
| selected | local_engine - fedavg_iid | rmse | 5 | +3.04 ± 0.75 | 0/5 | [2.88, 3.56, 3.141, 1.857, 3.772] | 0.062 |
| selected | local_engine - fedavg_iid | nasa_score | 5 | +347.80 ± 98.41 | 0/5 | [340.127, 385.588, 410.805, 180.939, 421.565] | 0.062 |
| final | centralized - fedavg_iid | rmse | 5 | +1.94 ± 1.11 | 0/5 | [2.893, 0.556, 1.41, 1.591, 3.247] | 0.062 |
| final | centralized - fedavg_iid | nasa_score | 5 | +136.79 ± 92.67 | 0/5 | [153.169, 33.765, 69.492, 154.233, 273.303] | 0.062 |
| final | centralized_legacy_es - fedavg_iid | rmse | 5 | +1.24 ± 0.71 | 0/5 | [1.3, 0.82, 0.282, 1.999, 1.817] | 0.062 |
| final | centralized_legacy_es - fedavg_iid | nasa_score | 5 | +96.71 ± 105.30 | 1/5 | [138.418, 66.647, -29.971, 251.815, 56.637] | 0.125 |
| final | local_iid - fedavg_iid | rmse | 5 | +1.97 ± 0.67 | 0/5 | [2.125, 1.252, 1.657, 3.033, 1.78] | 0.062 |
| final | local_iid - fedavg_iid | nasa_score | 5 | +189.26 ± 104.37 | 0/5 | [167.41, 83.098, 149.225, 363.009, 183.577] | 0.062 |
| final | fedavg_engine - fedavg_iid | rmse | 5 | +0.86 ± 0.81 | 0/5 | [2.091, 0.574, 0.003, 0.418, 1.189] | 0.062 |
| final | fedavg_engine - fedavg_iid | nasa_score | 5 | +74.98 ± 106.85 | 0/5 | [253.211, 19.355, 4.23, 1.849, 96.25] | 0.062 |
| final | local_engine - fedavg_iid | rmse | 5 | +4.95 ± 0.76 | 0/5 | [5.488, 4.182, 4.113, 5.222, 5.756] | 0.062 |
| final | local_engine - fedavg_iid | nasa_score | 5 | +577.36 ± 148.60 | 0/5 | [732.127, 543.746, 469.607, 410.127, 731.205] | 0.062 |

## 4. Partition / heterogeneity experiments (K=5, R=20, E=5; selected checkpoint)

| Configuration | n seeds | Test RMSE | Test MAE | NASA score | Best val RMSE |
|---|---|---|---|---|---|
| fedavg_iid  | 5 | 13.24 ± 0.32 | 9.78 ± 0.21 | 304.0 ± 26.6 | 12.36 ± 0.11 |
| local_iid  | 5 | 14.58 ± 0.17 | 10.81 ± 0.09 | 414.6 ± 20.4 | 13.16 ± 0.18 |
| fedavg_engine  | 5 | 13.87 ± 0.19 | 10.15 ± 0.13 | 349.0 ± 26.5 | 12.87 ± 0.34 |
| local_engine  | 5 | 16.28 ± 0.50 | 12.13 ± 0.26 | 651.8 ± 80.3 | 14.94 ± 0.29 |
| fedavg_dirichlet dirichlet_alpha=1.0 | 5 | 13.35 ± 0.26 | 9.92 ± 0.34 | 309.0 ± 8.7 | 12.45 ± 0.13 |
| fedavg_dirichlet dirichlet_alpha=0.5 | 5 | 13.47 ± 0.32 | 9.91 ± 0.36 | 319.6 ± 32.1 | 12.69 ± 0.41 |
| fedavg_dirichlet dirichlet_alpha=0.1 | 5 | 14.06 ± 0.46 | 11.01 ± 0.60 | 346.4 ± 34.9 | 13.22 ± 0.49 |

### Client data per condition (mean over seeds; min–max across clients)

| Experiment | Grid | K | Windows/client (min–max over clients & seeds) | Engines touched/client | Min exclusive engines/client | Client mean RUL range | Seeds |
|---|---|---|---|---|---|---|---|
| fedavg_dirichlet | dirichlet_alpha=0.1 | 5 | 479–8555 | 90–90 | 0 | 20.8–112.8 | 42,123,2026,3407,9001 |
| fedavg_dirichlet | dirichlet_alpha=0.5 | 5 | 1433–7250 | 90–90 | 0 | 42.3–105.7 | 42,123,2026,3407,9001 |
| fedavg_dirichlet | dirichlet_alpha=1.0 | 5 | 1521–5855 | 90–90 | 0 | 45.6–102.2 | 42,123,2026,3407,9001 |
| fedavg_engine | default | 5 | 2751–3352 | 18–18 | 18 | 73.5–82.8 | 42,123,2026,3407,9001 |
| fedavg_iid | default | 5 | 3126–3126 | 90–90 | 0 | 78.1–81.8 | 42,123,2026,3407,9001 |
| sweep_clients | num_clients=10 | 10 | 1563–1563 | 90–90 | 0 | 77.3–82.3 | 42,123,2026,3407,9001 |
| sweep_clients | num_clients=2 | 2 | 7815–7815 | 90–90 | 0 | 79.5–80.0 | 42,123,2026,3407,9001 |
| sweep_clients | num_clients=5 | 5 | 3126–3126 | 90–90 | 0 | 78.1–81.8 | 42,123,2026,3407,9001 |
| sweep_clients_engine | num_clients=10 | 10 | 1231–1885 | 9–9 | 9 | 67.7–87.4 | 42,123,2026,3407,9001 |
| sweep_clients_engine | num_clients=2 | 2 | 7582–8048 | 45–45 | 45 | 78.3–81.1 | 42,123,2026,3407,9001 |
| sweep_clients_engine | num_clients=5 | 5 | 2751–3352 | 18–18 | 18 | 73.5–82.8 | 42,123,2026,3407,9001 |

Per-client, per-seed counts, RUL quantiles and 10-bin RUL histograms: `tables/partition_stats.csv`. Partition seeds are recorded per run in `runs/<id>/metrics.json` → `partition.partition_seed`.


## 5. Communication-round sweep (K=5, E=5, IID window)

| Configuration | n seeds | Test RMSE | Test MAE | NASA score | Best val RMSE | Wall time, clean runs (s) | Traffic (MB) | Selected round |
|---|---|---|---|---|---|---|---|---|
| num_rounds=5 | 5 | 14.83 ± 0.94 | 11.33 ± 0.75 | 415.7 ± 92.8 | 14.16 ± 0.35 | 19 ± 4 | 43.2 | 5.0 ± 0.0 |
| num_rounds=10 | 5 | 13.78 ± 0.26 | 10.03 ± 0.23 | 338.8 ± 19.1 | 12.90 ± 0.08 | 34 ± 2 | 86.4 | 10.0 ± 0.0 |
| num_rounds=20 | 5 | 13.24 ± 0.32 | 9.78 ± 0.21 | 304.0 ± 26.6 | 12.36 ± 0.11 | 53 ± 1 | 172.9 | 18.0 ± 1.6 |
| num_rounds=30 | 5 | 13.48 ± 0.31 | 10.01 ± 0.19 | 331.1 ± 29.8 | 12.28 ± 0.11 | 97 ± 6 | 259.3 | 21.4 ± 2.8 |

Final-round checkpoint (no validation-based selection): num_rounds=5: RMSE 14.83 ± 0.94; num_rounds=10: RMSE 13.78 ± 0.26; num_rounds=20: RMSE 12.88 ± 0.30; num_rounds=30: RMSE 13.33 ± 0.22


## 6. Local-epoch sweep (K=5, R=20, IID window)

| Configuration | n seeds | Test RMSE | Test MAE | NASA score | Best val RMSE | Wall time, clean runs (s) | Traffic (MB) | Selected round |
|---|---|---|---|---|---|---|---|---|
| local_epochs=1 | 5 | 17.58 ± 1.30 | 13.08 ± 0.83 | 713.9 ± 241.5 | 17.05 ± 1.37 | 15 ± 1 | 172.9 | 19.8 ± 0.4 |
| local_epochs=3 | 5 | 13.44 ± 0.30 | 9.95 ± 0.10 | 307.7 ± 28.3 | 12.61 ± 0.08 | 39 ± 1 | 172.9 | 19.2 ± 0.8 |
| local_epochs=5 | 5 | 13.24 ± 0.32 | 9.78 ± 0.21 | 304.0 ± 26.6 | 12.36 ± 0.11 | 53 ± 1 | 172.9 | 18.0 ± 1.6 |
| local_epochs=10 | 5 | 13.56 ± 0.37 | 10.21 ± 0.28 | 326.6 ± 30.4 | 12.37 ± 0.12 | 129 ± 5 | 172.9 | 13.2 ± 3.7 |

Final-round checkpoint (no validation-based selection): local_epochs=1: RMSE 17.42 ± 1.51; local_epochs=3: RMSE 13.11 ± 0.24; local_epochs=5: RMSE 12.88 ± 0.30; local_epochs=10: RMSE 13.67 ± 0.39


## 7a. Client-count sweep, IID window (R=20, E=5; total training data fixed)

| Configuration | n seeds | Test RMSE | Test MAE | NASA score | Best val RMSE | Wall time, clean runs (s) | Traffic (MB) | Selected round |
|---|---|---|---|---|---|---|---|---|
| num_clients=2 | 5 | 14.14 ± 0.62 | 10.47 ± 0.43 | 384.2 ± 77.1 | 12.30 ± 0.23 | 60 ± 4 | 69.1 | 16.2 ± 4.8 |
| num_clients=5 | 5 | 13.24 ± 0.32 | 9.78 ± 0.21 | 304.0 ± 26.6 | 12.36 ± 0.11 | 53 ± 1 | 172.9 | 18.0 ± 1.6 |
| num_clients=10 | 5 | 13.53 ± 0.32 | 9.91 ± 0.08 | 317.1 ± 30.4 | 12.79 ± 0.14 | 65 ± 6 | 345.7 | 20.0 ± 0.0 |

Final-round checkpoint (no validation-based selection): num_clients=2: RMSE 13.62 ± 0.40; num_clients=5: RMSE 12.88 ± 0.30; num_clients=10: RMSE 13.53 ± 0.32


## 7b. Client-count sweep, whole-engine (R=20, E=5; total training data fixed)

| Configuration | n seeds | Test RMSE | Test MAE | NASA score | Best val RMSE | Wall time, clean runs (s) | Traffic (MB) | Selected round |
|---|---|---|---|---|---|---|---|---|
| num_clients=2 | 5 | 13.65 ± 0.41 | 10.29 ± 0.43 | 327.6 ± 38.3 | 12.51 ± 0.20 | 58 ± 4 (n=4) | 69.1 | 11.0 ± 3.5 |
| num_clients=5 | 5 | 13.87 ± 0.19 | 10.15 ± 0.13 | 349.0 ± 26.5 | 12.87 ± 0.34 | 65 ± 7 | 172.9 | 15.2 ± 3.7 |
| num_clients=10 | 5 | 14.34 ± 0.59 | 10.63 ± 0.55 | 374.7 ± 62.6 | 13.25 ± 0.18 | n/a (no clean run) | 345.7 | 19.8 ± 0.4 |

Final-round checkpoint (no validation-based selection): num_clients=2: RMSE 13.98 ± 0.51; num_clients=5: RMSE 13.74 ± 0.68; num_clients=10: RMSE 14.53 ± 0.43


Note: in the round and local-epoch sweeps the total compute budget R×E changes with the swept variable, so accuracy differences mix the effect of communication frequency with the effect of more training.


## 8. Communication accounting

- Serialized model payload (Flower `ndarray_to_bytes` format, fp32): **864,260 bytes** per model transfer (raw fp32 tensor bytes = parameters × 4, plus one .npy header per tensor; exact parameter count in `runs/<id>/metrics.json` → `comm.model_payload`).
- Per round: K downloads + K uploads of one full model. Cumulative traffic = 2 · K · R · payload. The initial model download is counted.
- **Label: serialized-payload estimate.** Clients were simulated in one process, so no network traffic was measured. gRPC/protobuf framing, TLS, TCP/IP headers, retransmissions and config/metric messages are excluded.
- Centralized training exchanges no model, but would require centralising the raw data; the run metadata records an analytical float32 estimate of that volume (`comm.raw_training_data_float32_bytes_estimate`).
- Timing: clients train sequentially on one GPU. `wall_s` is measured. `parallel_estimate_total_s` (sum over rounds of slowest client + aggregation) is a derived estimate of a parallel deployment, not a measurement.


## 8b. Runtime (dedicated timing benchmark)

Measured by re-running seed 42 of each main configuration back-to-back on an otherwise idle GPU (`results/v2_timing/`), because wall-clock times in the main runs were affected by machine state, one suspend and two periods of concurrent runs (flagged per run in `tables/per_seed.csv` → `timing_clean`). Clients are simulated sequentially, so FedAvg wall time = sum of all client work; the parallel estimate (slowest client + aggregation, summed over rounds) is derived, not measured. Accuracy in the benchmark reproduced the main runs exactly (column 'RMSE matches').

| Configuration | Wall time (s) | Mean round (s) | Mean client local training per round (s) | Parallel estimate (s) | Passes over data | RMSE matches main run |
|---|---|---|---|---|---|---|
| centralized | 78.2 | — | — | — | 100 | yes |
| centralized_legacy_es | 25.0 | — | — | — | 33 | yes |
| fedavg_iid | 72.2 | 3.61 | 0.71 | 15.8 | 100 | yes |
| local_iid | 71.3 | — | — | — | 100 | yes |
| fedavg_engine | 59.1 | 2.95 | 0.58 | 12.5 | 100 | yes |
| local_engine | 73.1 | — | — | — | 100 | yes |

Hardware/software: NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 11.8, cuDNN 90100, torch 2.7.1+cu118, Python 3.11.9, Windows-10-10.0.26200-SP0.


## 9. Run status

| Experiment | Grid point | Completed seeds |
|---|---|---|
| ablation_fedavg_k1 | default | 42,123,2026,3407,9001 |
| centralized | default | 42,123,2026,3407,9001 |
| centralized_legacy_es | default | 42,123,2026,3407,9001 |
| fedavg_dirichlet | dirichlet_alpha=0.1 | 42,123,2026,3407,9001 |
| fedavg_dirichlet | dirichlet_alpha=0.5 | 42,123,2026,3407,9001 |
| fedavg_dirichlet | dirichlet_alpha=1.0 | 42,123,2026,3407,9001 |
| fedavg_engine | default | 42,123,2026,3407,9001 |
| fedavg_iid | default | 42,123,2026,3407,9001 |
| local_engine | default | 42,123,2026,3407,9001 |
| local_iid | default | 42,123,2026,3407,9001 |
| sweep_clients | num_clients=10 | 42,123,2026,3407,9001 |
| sweep_clients | num_clients=2 | 42,123,2026,3407,9001 |
| sweep_clients | num_clients=5 | 42,123,2026,3407,9001 |
| sweep_clients_engine | num_clients=10 | 42,123,2026,3407,9001 |
| sweep_clients_engine | num_clients=2 | 42,123,2026,3407,9001 |
| sweep_clients_engine | num_clients=5 | 42,123,2026,3407,9001 |
| sweep_local_epochs | local_epochs=1 | 42,123,2026,3407,9001 |
| sweep_local_epochs | local_epochs=10 | 42,123,2026,3407,9001 |
| sweep_local_epochs | local_epochs=3 | 42,123,2026,3407,9001 |
| sweep_local_epochs | local_epochs=5 | 42,123,2026,3407,9001 |
| sweep_rounds | num_rounds=10 | 42,123,2026,3407,9001 |
| sweep_rounds | num_rounds=20 | 42,123,2026,3407,9001 |
| sweep_rounds | num_rounds=30 | 42,123,2026,3407,9001 |
| sweep_rounds | num_rounds=5 | 42,123,2026,3407,9001 |

**Planned but not run (no results exist; not imputed):**

| Experiment | Grid point | Missing seeds |
|---|---|---|
| local_dirichlet | dirichlet_alpha=0.1 | 42,123,2026,3407,9001 |
| local_dirichlet | dirichlet_alpha=0.5 | 42,123,2026,3407,9001 |
| local_dirichlet | dirichlet_alpha=1.0 | 42,123,2026,3407,9001 |

---

## 10. Interpretation (written after inspecting the results above)

All statements below refer to numbers in sections 2–9 of this report or in `tables/*.csv`.
"Lower" means lower test error. With 5 seeds no difference is declared statistically
significant (see section 3); we describe how large and how consistent the effects are across matched seeds.

### 10.1 Centralized vs FedAvg vs local-only

* **FedAvg-IID had lower test error than budget-matched centralized training in all 5 matched seeds**
  (RMSE 13.24 ± 0.32 vs 14.46 ± 0.44 at the validation-selected checkpoint; per-seed gap +0.29 to +1.73 cycles,
  mean +1.22). The same ordering holds for MAE and the NASA score, and at the final checkpoint.
  The original report's gap (3.06 cycles: 12.95 vs 16.01) came mostly from comparing a single favourable FL seed with
  a single unfavourable centralized seed. Under matched seeds the gap is **about 60% smaller**.
* **Why FedAvg can beat centralized training here.** Federation itself adds no information, so the advantage has to
  come from the optimisation procedure. The convergence plot (fig. 2) shows the centralized model reaching its best
  validation RMSE around epoch 45 of 100 and then **overfitting** (validation RMSE rises from ≈13 to ≈15). FedAvg's
  validation RMSE keeps improving slowly through round 20. Two plausible regularising mechanisms are (i) model
  averaging across clients and (ii) the per-round reset of each client's Adam state.
  **Ablation (`ablation_fedavg_k1`, 5 seeds):** FedAvg with a single client holding all data is exactly centralized
  training with the Adam state reset every 5 epochs and the same selection rule. It reaches 13.94 ± 0.89 (selected)
  and 15.96 ± 1.62 (final), between centralized (14.46 / 14.82) and FedAvg-IID (13.24 / 12.88). It is lower than
  FedAvg-IID in only 2 of 5 seeds, selects early (round 7.2 on average) and overfits afterwards. **The optimizer
  reset alone therefore does not reproduce FedAvg's advantage.** The evidence is consistent with averaging five
  clients' updates acting as a regulariser: each client takes ≈65 optimizer steps per round instead of ≈310 on the
  pooled data, and their average moves the global model more conservatively. The centralized baseline is effectively
  over-trained at 100 epochs with these settings. This is a property of the optimisation set-up, not a benefit of
  data decentralisation, and the manuscript should say so.
* **Hyperparameters were not tuned for either method.** All methods use the original repository's settings
  (Adam 1e-3, batch 256, dropout 0.2, no LR schedule). A centralized model with tuned regularisation or a tuned epoch
  budget may close the gap. The defensible claim is "under identical, untuned hyperparameters and a matched
  training budget", not "FedAvg is more accurate than centralized learning".
* **The original early-stopping protocol does not explain the original gap either.** `centralized_legacy_es`
  (original early stopping + ReduceLROnPlateau) gives 14.55 ± 0.48, practically the same as the
  budget-matched centralized run (14.46 ± 0.44; difference well within one SD). Changing the centralized protocol therefore removes a
  confound in the paper without changing the centralized number much.
* **Local-only training is clearly worse than FedAvg**: IID-window local models 14.58 ± 0.17 (+1.34 vs FedAvg-IID,
  5/5 seeds). Under whole-engine ownership local models (18 engines each) reach 16.28 ± 0.50 vs 13.87 ± 0.19 for
  FedAvg on the same partition (−2.41 cycles), and the NASA score drops from 652 to 349. **This is the strongest and
  most practically relevant result:** an airline with 18 engines gains substantially from federating with four others.
  IID-window local-only is less informative: each local shard contains fragments of all 90 engines, which a real
  airline would not have.

### 10.2 Partition protocols and heterogeneity

* **Whole-engine ownership costs 0.63 ± 0.42 RMSE relative to IID-window FedAvg (5/5 seeds)**
  (13.87 vs 13.24). This should **not** be described as an effect of statistical heterogeneity. Client RUL
  distributions are almost identical under both partitions (client mean RUL 73.5–82.8 vs 78.1–81.8 cycles; fig. 9),
  but the partitions differ in (a) engine ownership (each IID-window client sees fragments of all 90 engines, each
  whole-engine client sees 18 complete trajectories), (b) cross-client near-duplicate windows (overlapping windows of
  one engine split across clients), and (c) volume (3,126 windows per client vs 2,751–3,352).
* **Dirichlet label skew degrades FedAvg only at α = 0.1**: 13.35 ± 0.26 (α=1.0), 13.47 ± 0.32 (α=0.5),
  14.06 ± 0.46 (α=0.1) vs 13.24 ± 0.32 (IID). The α=1.0 and α=0.5 differences are within one SD. At α=0.1 the
  partition is extremely skewed in **both** label and quantity (client means 20.8–112.8 cycles; client sizes 479–8,555
  windows; one redraw needed for one seed to satisfy the 256-window minimum). The degradation therefore cannot be
  attributed to label skew alone.
* Validation RMSE ranks the conditions the same way as test RMSE (IID 12.36 < α1.0 12.45 < α0.5 12.69 = whole-engine
  12.87 < α0.1 13.22), which supports the ordering.

### 10.3 Error characteristics

* All methods **over-predict RUL on average** (mean error +3.3 to +4.8 cycles; 58–69% of test engines predicted late).
  Late predictions are the costly direction under the NASA score, which is why the score separates methods more than
  RMSE does. FedAvg-IID has the smallest bias (+3.3) and the smallest 90th-percentile absolute error (21.8 cycles).
* The NASA score is a **sum over the 100 test engines** and is dominated by a few large late errors. Its seed-to-seed
  spread is large (e.g. local whole-engine 652 ± 80), so it should be reported with its SD and never on its own.

### 10.4 Communication and runtime

**Communication rounds** (K=5, E=5; final-round checkpoint, which is the natural one for a round budget):
R=5 14.83 ± 0.94, R=10 13.78 ± 0.26, R=20 12.88 ± 0.30, R=30 13.33 ± 0.22. Accuracy improves up to R=20 and
then **does not improve further**; R=30 is slightly worse than R=20 in mean while costing 50% more traffic
(259.3 vs 172.9 MB). The selected checkpoint shows the same pattern (13.24 vs 13.48). Because E is fixed, more
rounds also means more total training (R×E passes), so this sweep mixes communication frequency with training length.

**Local epochs** (K=5, R=20): E=1 is clearly insufficient (17.42 ± 1.51 final; NASA 689 ± 269). E=3 (13.11 ± 0.24),
E=5 (12.88 ± 0.30) and E=10 (13.67 ± 0.39) are close, with E=5 lowest on average and E=10 worse than E=5 in mean.
E=3 and E=5 differ by less than one SD. More local computation at a fixed communication budget helps only up to a point.

**Number of clients, total data fixed** (R=20, E=5):
* IID window: K=2 13.62 ± 0.40, K=5 12.88 ± 0.30, K=10 13.53 ± 0.32 (final). **Not monotone**; K=5 is lowest.
  Each K changes the per-client shard size and the number of local steps per round, so this is not a pure
  "more participants" effect.
* Whole-engine: K=2 13.65 ± 0.41, K=5 13.87 ± 0.19, K=10 14.34 ± 0.59 (selected). Error **rises** as the same 90
  engines are split over more owners (45 → 18 → 9 engines per client).
* Traffic grows linearly with K: 69.1, 172.9, 345.7 MB (serialized-payload estimate).

**Communication accounting**: one model transfer = 864,260 bytes (862,724 B of fp32 weights + 1,536 B of `.npy`
headers). Primary setting (K=5, R=20): 86.43 MB up + 86.43 MB down = 172.85 MB (= 164.84 MiB). The original report's
"164.55 MB" was the same quantity in MiB without the headers. All figures are payload estimates; no network was measured.

**Runtime** (section 8b, seed 42, clean benchmark): with the same 100 passes over the data, centralized training took
78 s and FedAvg-IID 72 s of sequential compute on one RTX 4060 Laptop GPU, i.e. **the same training cost**.
FedAvg's derived parallel estimate is 15.8 s (each client ≈0.7 s per round). The original report's "FL 131 s vs
centralized 49 s" compared a fixed 100-pass FL budget with an early-stopped centralized run, plus Ray/Flower overhead.
In this simulation FedAvg's wall time is the sum of all client
work on one GPU. It should not be read as the latency of a real deployment, which would depend on parallel
clients, network bandwidth and stragglers.

**Accuracy–communication trade-off**: the cheapest FedAvg setting tested that stays within ≈1 cycle of the best is
R=10 (13.78 RMSE, 86 MB), half the traffic of R=20 (12.88, 173 MB). Centralized training needs no model exchange,
but the raw training data would have to be moved instead: the 90-engine training table is ≈1.3 MB as float32
(analytical estimate, `comm.raw_training_data_float32_bytes_estimate`), about 130× smaller than FedAvg's model
traffic at R=20. **FedAvg is not a bandwidth saving here.** Its motivation is data ownership and governance,
not communication.

### 10.5 Validation-based selection vs final checkpoint

For FedAvg the final-round model was better on the test set than the validation-selected one (12.88 vs 13.24), while
for centralized training selection helped (14.46 vs 14.82). The validation set has only 10 engines, so checkpoint
selection is noisy. Both checkpoints are reported everywhere. **The method ordering is the same under both
policies**, so the conclusions do not depend on the selection rule.

### 10.6 Limitations

1. **One dataset subset (FD001)**: single operating condition, single fault mode, 100 training engines. Results may
   not transfer to FD002–FD004 (multiple conditions / fault modes), where client heterogeneity is more realistic.
2. **Small validation set**: 10 engines (2,101 windows), fixed across seeds. Selection noise is visible (10.5).
   Varying the validation split across seeds, or cross-validation, would give a fuller variance estimate.
3. **Five seeds**: enough to show consistent orderings, but too few for a significance test at α = 0.05 (the
   smallest attainable two-sided sign-flip p is 0.0625). Effects smaller than about one SD (e.g. α = 1.0 vs IID) are not
   resolved.
4. **Untuned hyperparameters**: no method was tuned. The centralized baseline overfits under the shared budget,
   which favours FedAvg in the comparison.
5. **Simulated federation**: clients run sequentially in one process; full participation, no stragglers, no client
   dropout, no network. Communication is a serialized-payload estimate, not measured traffic.
6. **Server-side validation set**: FedAvg checkpoint selection uses a held-out set of engines at the server, which
   a real deployment may not have.
7. **Synthetic partitions**: the "airlines" are random partitions of one fleet simulated under one condition.
   Real airline heterogeneity (operating profiles, maintenance practice, sensor calibration) is not represented.
8. **No privacy guarantee**: raw data stay on clients, but model updates can leak information. No differential
   privacy or secure aggregation is implemented, so no formal privacy claim can be made.
9. **Test labels capped at 125** (common FD001 practice); the uncapped truth is stored in every
   `test_predictions.csv` for re-scoring.
