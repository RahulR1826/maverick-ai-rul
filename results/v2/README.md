# results/v2 — reproducible multi-seed results (NICEDT-2027 FL-RUL paper)

Everything here was produced by `python -m research.run` (see the top-level README for commands).
The older files in `results/` (outside `v2/`) come from the earlier, non-budget-matched protocol and are kept
unchanged for reference only.

## Start here

| File | Contents |
|---|---|
| `EXPERIMENT_REPORT.md` | Research report: protocol, all mean ± SD tables, run status, interpretation, limitations |
| `CONCLUSION_CHECK.md` | The 7 original findings vs the new 5-seed results, with a verdict and manuscript edits |
| `figures/` | Publication figures (PNG 300 dpi + PDF) |
| `tables/` | Aggregated CSV tables (below) |
| `runs/` | Raw per-run results, one directory per (configuration, seed) |

## Raw results (`runs/<method>-<config-hash>-s<seed>/`)

Every configuration × seed has its own directory, including poor and failed-to-improve runs. Nothing was dropped.

| File | Contents |
|---|---|
| `config.json` | Fully resolved configuration, seed, derived seed streams, determinism flags, software/hardware, git commit |
| `metrics.json` | Test RMSE/MAE/NASA + error summary for the **selected** and **final** checkpoints, validation, timing, communication, partition |
| `history.csv` | Per round (FedAvg) or epoch (centralized/local): validation RMSE/MAE, timings, cumulative upload/download bytes |
| `test_predictions.csv` | Per test engine: true RUL (capped and uncapped), predictions and errors for each checkpoint |
| `clients.csv` | Per client: windows, engines, exclusive engines, RUL mean/SD/quantiles, 10-bin RUL histogram |
| `client_rounds.csv` | FedAvg: per client per round local training time and loss |
| `local_clients.csv` | Local-only: per client model metrics |

Checkpoints: `checkpoints/v2/<run_id>/selected.pt` and `final.pt` (local-only: `client<k>_selected.pt`).

## Aggregated tables (`tables/`)

| File | Contents |
|---|---|
| `per_seed.csv` | One row per experiment × grid point × seed × checkpoint (all raw numbers, plus timing-validity flags) |
| `summary.csv` | Mean, SD (ddof=1), min, max and n_seeds per experiment × grid point × checkpoint |
| `paired_vs_fedavg_iid.csv` | Per-seed paired differences against FedAvg-IID (matched seeds) |
| `convergence.csv` | Validation RMSE per round/epoch, mean ± SD over seeds, cumulative traffic |
| `partition_stats.csv` | Per-client partition statistics for every seed and partition |
| `per_engine_errors.csv` | Absolute test error per engine per run (main comparison) |
| `missing_runs.csv` | Planned runs without results (currently only the optional `local_dirichlet` extras) |

`timing_clean = False` in `per_seed.csv` marks runs whose wall-clock time is not a clean measurement: they overlapped
another run on the GPU, or spanned a laptop suspend. Their accuracy is unaffected (runs are deterministic). Clean
runtimes are in `../v2_timing/` and in report section 8b.

## Seeds

Primary: 42, 123, 2026, 3407, 9001 (3-seed fallback: 42, 123, 2026). The same list is used for every method,
so comparisons are paired by seed.
