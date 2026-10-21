"""
research/report.py
------------------
Builds ``<out>/EXPERIMENT_REPORT.md`` from the aggregated tables.  All numbers
in the generated tables come from ``tables/*.csv``; the interpretation section
is read from ``<out>/discussion.md`` (written by a human after inspecting the
results) and is clearly separated from the generated material.

python -m research.report [--out results/v2]
"""

import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from research import config as C


def fmt(m, s, n, d=2):
    if pd.isna(m):
        return "—"
    if n is None or n < 2 or pd.isna(s):
        return f"{m:.{d}f} (n={int(n) if n else 0})"
    return f"{m:.{d}f} ± {s:.{d}f}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def metric_rows(summ, sel, label_fn, extra_cols=()):
    rows = []
    for _, r in summ[sel].iterrows():
        rows.append([label_fn(r), int(r.n_seeds), fmt(r.rmse_mean, r.rmse_std, r.n_seeds),
                     fmt(r.mae_mean, r.mae_std, r.n_seeds), fmt(r.nasa_score_mean, r.nasa_score_std, r.n_seeds, 1),
                     fmt(r.best_val_rmse_mean, r.best_val_rmse_std, r.n_seeds)]
                    + [f(r) for f in extra_cols])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.DEFAULT_OUT_DIR)
    ap.add_argument("--seed-set", default="primary")
    a = ap.parse_args(argv)
    proto = C.load_protocol()
    t = os.path.join(a.out, "tables")
    summ = pd.read_csv(os.path.join(t, "summary.csv"))
    ps = pd.read_csv(os.path.join(t, "per_seed.csv"))
    missing = pd.read_csv(os.path.join(t, "missing_runs.csv")) if os.path.getsize(os.path.join(t, "missing_runs.csv")) > 1 else pd.DataFrame()
    paired = pd.read_csv(os.path.join(t, "paired_vs_fedavg_iid.csv")) if os.path.exists(os.path.join(t, "paired_vs_fedavg_iid.csv")) else pd.DataFrame()
    parts = pd.read_csv(os.path.join(t, "partition_stats.csv")) if os.path.exists(os.path.join(t, "partition_stats.csv")) else pd.DataFrame()
    base = proto["base"]
    _cw = ps[(ps.checkpoint == "selected") & ps.get("timing_clean", True).astype(bool)].groupby(["experiment", "grid_label"])["wall_s"]
    _cw = _cw.agg(["mean", "std", "count"])

    def clean_wall(r):
        """Wall time over runs with a clean timing measurement only (see per_seed.csv: timing_clean)."""
        k = (r.experiment, r.grid_label)
        if k not in _cw.index:
            return "n/a (no clean run)"
        m_, s_, n_ = _cw.loc[k]
        return (f"{m_:.0f} ± {s_:.0f}" if n_ > 1 else f"{m_:.0f}") + ("" if n_ == r.n_seeds else f" (n={int(n_)})")
    seeds = proto["seeds"][a.seed_set]
    H = ["Configuration", "n seeds", "Test RMSE", "Test MAE", "NASA score", "Best val RMSE"]
    L = []
    L.append("# Experiment Report — Federated Learning for Aircraft Engine Prognostics (v2 protocol)\n")
    L.append(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `python -m research.report` from `{os.path.relpath(t, ROOT)}`. "
             "Every number in sections 3–8 is computed from per-run `metrics.json` files; nothing is imputed. "
             "Values are mean ± sample SD (ddof = 1) over seeds.\n")

    # ------------------------------------------------------------- protocol
    d, m, tr, fed = base["data"], base["model"], base["train"], base["fed"]
    L.append("## 1. Protocol\n")
    L.append(table([
        ["Dataset", f"NASA C-MAPSS {d['dataset_id']}: 100 training engines, 100 official test engines"],
        ["Train / validation split", f"engine-level, {int(d['val_fraction']*100)}% of training engines (10) held out, drawn with `default_rng({d['val_split_seed']})` (same engines as the original code); fixed for all seeds and methods"],
        ["Features", f"3 operational settings + 14 sensors (dropped sensors {d['drop_sensors']}) = 17; fixed a priori"],
        ["Normalisation", "MinMaxScaler(0,1) fitted on the 90 training engines only; applied unchanged to validation and test"],
        ["Windows / target", f"{d['window_size']} cycles, stride {d['stride']}; label = piece-wise linear RUL capped at {d['rul_cap']} at the window's last cycle"],
        ["Test set", f"last {d['window_size']} cycles of each of the 100 test engines; truth from RUL_FD001.txt, capped at {d['rul_cap']} ({'yes' if d['cap_test_rul'] else 'no'}); used only after model selection"],
        ["Model", f"LSTM {m['num_layers']}×{m['hidden_size']}, dropout {m['dropout']}, FC 64→1 (`models/lstm_model.py`, unchanged)"],
        ["Optimiser", f"Adam lr={tr['learning_rate']}, weight decay={tr['weight_decay']}, batch {tr['batch_size']}, grad-norm clip {tr['grad_clip']}, MSE loss, constant LR"],
        ["FedAvg", f"K={fed['num_clients']}, R={fed['num_rounds']}, E={fed['local_epochs']}, full participation, sample-weighted averaging, fresh Adam state per client per round"],
        ["Training budget", "every method makes R×E = 100 passes over its training data (centralized: 100 epochs; local-only: 100 epochs on its shard)"],
        ["Checkpoint selection", "validation RMSE on the 10 held-out engines every E epochs / every round (20 candidates); lowest is 'selected'. The model after the full budget is 'final'. Both are reported."],
        ["Reference protocol", "`centralized_legacy_es` = original code: ≤50 epochs, early stopping patience 10, ReduceLROnPlateau, selection every epoch. Not budget-matched; shown only to quantify the protocol change."],
        ["Seeds", f"{seeds} (3-seed fallback: {proto['seeds']['fallback']}); identical lists for every method. A seed fixes model init, partition, mini-batch order and dropout via independent derived streams"],
        ["Determinism", "cuDNN deterministic, benchmark off, `torch.use_deterministic_algorithms(True, warn_only=True)`, CUBLAS_WORKSPACE_CONFIG=:4096:8"],
    ], ["Item", "Setting"]))
    L.append("\n### Partition protocols\n")
    L.append("- **IID window**: all 15,630 training windows randomly permuted and split into K near-equal shards. Overlapping windows of one engine land on different clients, so every client sees fragments of almost every engine.\n"
             "- **Dirichlet window (α)**: RUL labels binned into 10 equal-width bins on [0,125]; per bin, client shares ~ Dir(α·1_K). Changes the RUL distribution **and** client sizes (quantity skew), and still spreads engines across clients.\n"
             f"- **Whole-engine**: the 90 training engines randomly assigned (seeded) to K clients, 18 each for K=5; each client owns complete run-to-failure trajectories. Client window counts differ because engine lifetimes differ.\n"
             f"- Every window is assigned to exactly one client (asserted). Total training data is constant across K. Minimum client size: {fed['min_client_windows']} windows; partitions violating it are redrawn (attempt count recorded), never truncated.\n")
    L.append("**Confounding note.** Window-level vs whole-engine partitions differ in more than heterogeneity: engine ownership (fragmented vs complete trajectories), cross-client overlap of near-duplicate windows, and per-client volume. Dirichlet partitions combine label skew with quantity skew. Differences between these conditions are therefore *not* attributed to heterogeneity alone.\n")

    # ------------------------------------------------------------- main table
    for ck in ("selected", "final"):
        s = summ[summ.checkpoint == ck]
        names = ["centralized", "centralized_legacy_es", "ablation_fedavg_k1", "fedavg_iid", "local_iid", "fedavg_engine", "local_engine"]
        sel = s.experiment.isin(names) & (s.grid_label == "default")
        L.append(f"\n## {'2' if ck == 'selected' else '2b'}. Centralized vs FedAvg vs local-only — checkpoint = **{ck}**\n")
        rows = metric_rows(s, sel, lambda r: r.experiment, [lambda r: clean_wall(r),
                                                             lambda r: f"{r.comm_total_MB_mean:.1f}"])
        order = {n: i for i, n in enumerate(names)}
        rows.sort(key=lambda r: order.get(r[0], 99))
        L.append(table(rows, H + ["Wall time, clean runs (s)", "Model traffic (MB)"]))
        L.append("\nLocal-only rows: mean over the K=5 client models of each model's metrics on all 100 test engines.\n")

    if not paired.empty:
        L.append("\n## 3. Paired differences vs FedAvg-IID (matched seeds)\n")
        L.append("`diff = other − FedAvg-IID` per seed (negative = other method has lower error). The exact sign-flip permutation p-value is shown for transparency only: with n = 5 seeds the smallest attainable two-sided p is 0.0625, so **no comparison can be declared statistically significant at α = 0.05 with this design, and none is claimed.**\n")
        p = paired[(paired.metric.isin(["rmse", "nasa_score"]))]
        L.append(table([[r.checkpoint, r.comparison, r.metric, int(r.n_matched_seeds), f"{r.mean_diff:+.2f} ± {r.std_diff:.2f}",
                         f"{int(r.n_other_lower)}/{int(r.n_matched_seeds)}", r.per_seed_diff, f"{r.sign_flip_p_two_sided:.3f}"]
                        for r in p.itertuples()],
                       ["Ckpt", "Comparison", "Metric", "n", "Mean diff ± SD", "Other lower", "Per-seed diffs", "Sign-flip p"]))

    # ------------------------------------------------------------- heterogeneity
    s = summ[summ.checkpoint == "selected"]
    het = s[((s.experiment.isin(["fedavg_iid", "fedavg_engine", "fedavg_dirichlet", "local_dirichlet", "local_iid", "local_engine"])))]
    if len(het):
        L.append("\n## 4. Partition / heterogeneity experiments (K=5, R=20, E=5; selected checkpoint)\n")
        L.append(table(metric_rows(s, s.index.isin(het.index),
                                   lambda r: f"{r.experiment} {'' if r.grid_label == 'default' else r.grid_label}"), H))
        if not parts.empty:
            L.append("\n### Client data per condition (mean over seeds; min–max across clients)\n")
            pp = parts[parts.experiment.isin(["fedavg_iid", "fedavg_dirichlet", "fedavg_engine", "sweep_clients", "sweep_clients_engine"])]
            g = pp.groupby(["experiment", "grid_label", "seed"]).agg(
                kmin=("n_windows", "min"), kmax=("n_windows", "max"), emin=("n_engines", "min"), emax=("n_engines", "max"),
                exmin=("n_engines_exclusive", "min"), rmin=("rul_mean", "min"), rmax=("rul_mean", "max"),
                cap=("frac_at_cap", "max"), K=("client_id", "count")).reset_index()
            g2 = g.groupby(["experiment", "grid_label"]).agg(K=("K", "first"), seeds=("seed", lambda x: ",".join(map(str, sorted(x)))),
                                                             kmin=("kmin", "min"), kmax=("kmax", "max"), emin=("emin", "min"), emax=("emax", "max"),
                                                             exmin=("exmin", "min"), rmin=("rmin", "min"), rmax=("rmax", "max")).reset_index()
            L.append(table([[r.experiment, r.grid_label, int(r.K), f"{int(r.kmin)}–{int(r.kmax)}", f"{int(r.emin)}–{int(r.emax)}",
                             int(r.exmin), f"{r.rmin:.1f}–{r.rmax:.1f}", r.seeds] for r in g2.itertuples()],
                           ["Experiment", "Grid", "K", "Windows/client (min–max over clients & seeds)", "Engines touched/client",
                            "Min exclusive engines/client", "Client mean RUL range", "Seeds"]))
            L.append("\nPer-client, per-seed counts, RUL quantiles and 10-bin RUL histograms: `tables/partition_stats.csv`. "
                     "Partition seeds are recorded per run in `runs/<id>/metrics.json` → `partition.partition_seed`.\n")

    # ------------------------------------------------------------- sweeps
    def sweep_table(exp, key, title):
        sub = s[s.experiment == exp].sort_values(key)
        if sub.empty:
            return
        L.append(f"\n## {title}\n")
        L.append(table(metric_rows(s, s.experiment == exp, lambda r: f"{key}={int(r[key])}",
                                   [lambda r: clean_wall(r), lambda r: f"{r.comm_total_MB_mean:.1f}",
                                    lambda r: fmt(r.selected_at_mean, r.selected_at_std, r.n_seeds, 1)]),
                       H + ["Wall time, clean runs (s)", "Traffic (MB)", "Selected round"]))
        f = summ[(summ.checkpoint == "final") & (summ.experiment == exp)]
        L.append("\nFinal-round checkpoint (no validation-based selection): " + "; ".join(
            f"{key}={int(r[key])}: RMSE {fmt(r.rmse_mean, r.rmse_std, r.n_seeds)}" for _, r in f.sort_values(key).iterrows()) + "\n")
    sweep_table("sweep_rounds", "num_rounds", "5. Communication-round sweep (K=5, E=5, IID window)")
    sweep_table("sweep_local_epochs", "local_epochs", "6. Local-epoch sweep (K=5, R=20, IID window)")
    sweep_table("sweep_clients", "num_clients", "7a. Client-count sweep, IID window (R=20, E=5; total training data fixed)")
    sweep_table("sweep_clients_engine", "num_clients", "7b. Client-count sweep, whole-engine (R=20, E=5; total training data fixed)")
    L.append("\nNote: in the round and local-epoch sweeps the total compute budget R×E changes with the swept variable, so accuracy differences mix the effect of communication frequency with the effect of more training.\n")

    # ------------------------------------------------------------- communication
    pay = ps["payload_bytes"].dropna()
    if len(pay):
        L.append("\n## 8. Communication accounting\n")
        L.append(f"- Serialized model payload (Flower `ndarray_to_bytes` format, fp32): **{int(pay.iloc[0]):,} bytes** per model transfer "
                 "(raw fp32 tensor bytes = parameters × 4, plus one .npy header per tensor; exact parameter count in `runs/<id>/metrics.json` → `comm.model_payload`).\n"
                 "- Per round: K downloads + K uploads of one full model. Cumulative traffic = 2 · K · R · payload. The initial model download is counted.\n"
                 "- **Label: serialized-payload estimate.** Clients were simulated in one process, so no network traffic was measured. gRPC/protobuf framing, TLS, TCP/IP headers, retransmissions and config/metric messages are excluded.\n"
                 "- Centralized training exchanges no model, but would require centralising the raw data; the run metadata records an analytical float32 estimate of that volume (`comm.raw_training_data_float32_bytes_estimate`).\n"
                 "- Timing: clients train sequentially on one GPU. `wall_s` is measured. `parallel_estimate_total_s` (sum over rounds of slowest client + aggregation) is a derived estimate of a parallel deployment, not a measurement.\n")

    # ------------------------------------------------------------- timing benchmark
    tb = os.path.join(a.out + "_timing", "tables", "per_seed.csv")
    L.append("\n## 8b. Runtime (dedicated timing benchmark)\n")
    if os.path.exists(tb):
        tt = pd.read_csv(tb)
        bench = ["centralized", "centralized_legacy_es", "fedavg_iid", "local_iid", "fedavg_engine", "local_engine"]
        tt = tt[(tt.checkpoint == "selected") & tt.experiment.isin(bench) & (tt.grid_label == "default")]
        L.append("Measured by re-running seed 42 of each main configuration back-to-back on an otherwise idle GPU "
                 "(`results/v2_timing/`), because wall-clock times in the main runs were affected by machine state, one "
                 "suspend and two periods of concurrent runs (flagged per run in `tables/per_seed.csv` → `timing_clean`). "
                 "Clients are simulated sequentially, so FedAvg wall time = sum of all client work; the parallel estimate "
                 "(slowest client + aggregation, summed over rounds) is derived, not measured. "
                 "Accuracy in the benchmark reproduced the main runs exactly (column 'RMSE matches').\n")
        main_sel = ps[(ps.checkpoint == "selected") & (ps.seed == 42)].set_index("experiment")["rmse"]
        L.append(table([[r.experiment, f"{r.wall_s:.1f}", "—" if pd.isna(r.mean_round_wall_s) else f"{r.mean_round_wall_s:.2f}",
                         "—" if pd.isna(r.mean_client_local_train_s) else f"{r.mean_client_local_train_s:.2f}",
                         "—" if pd.isna(r.parallel_estimate_total_s) else f"{r.parallel_estimate_total_s:.1f}",
                         int(r.sample_passes) if not pd.isna(r.sample_passes) else "—",
                         "yes" if abs(main_sel.get(r.experiment, float("nan")) - r.rmse) < 1e-9 else "NO"]
                        for r in tt.itertuples()],
                       ["Configuration", "Wall time (s)", "Mean round (s)", "Mean client local training per round (s)",
                        "Parallel estimate (s)", "Passes over data", "RMSE matches main run"]))
        env = None
        for rid in tt.run_id:
            c = os.path.join(a.out + "_timing", "runs", rid, "config.json")
            if os.path.exists(c):
                env = json.load(open(c, encoding="utf-8"))["environment"]; break
        if env:
            L.append(f"\nHardware/software: {env.get('gpu')}, CUDA {env.get('cuda')}, cuDNN {env.get('cudnn')}, torch {env.get('torch')}, Python {env.get('python')}, {env.get('platform')}.\n")
    else:
        L.append("*Timing benchmark not run yet.*\n")

    # ------------------------------------------------------------- status
    L.append("\n## 9. Run status\n")
    done = ps[ps.checkpoint == "selected"].groupby(["experiment", "grid_label"]).seed.apply(lambda x: ",".join(map(str, sorted(x)))).reset_index()
    L.append(table([[r.experiment, r.grid_label, r.seed] for r in done.itertuples()], ["Experiment", "Grid point", "Completed seeds"]))
    if not missing.empty:
        mm = missing.groupby(["experiment", "grid_label"]).seed.apply(lambda x: ",".join(map(str, sorted(x)))).reset_index()
        L.append("\n**Planned but not run (no results exist; not imputed):**\n")
        L.append(table([[r.experiment, r.grid_label, r.seed] for r in mm.itertuples()], ["Experiment", "Grid point", "Missing seeds"]))
    fails = os.path.join(a.out, "failures.jsonl")
    if os.path.exists(fails):
        L.append("\n**Failed runs** (see `failures.jsonl` and logs):\n```\n" + open(fails, encoding="utf-8").read() + "```\n")

    disc = os.path.join(a.out, "discussion.md")
    L.append("\n---\n")
    if os.path.exists(disc):
        L.append(open(disc, encoding="utf-8").read())
    else:
        L.append("*No interpretation written yet (`discussion.md` missing).*\n")
    path = os.path.join(a.out, "EXPERIMENT_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("Report ->", path)


if __name__ == "__main__":
    main()
