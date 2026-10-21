"""
research/aggregate.py
---------------------
Collect per-run ``metrics.json`` files into tables.  Nothing is imputed: a run
that has not completed appears in ``missing_runs.csv`` and is excluded from
means; ``n_seeds`` is reported next to every mean ± std.

python -m research.aggregate [--seed-set primary] [--out results/v2]

Outputs (in <out>/tables/):
  per_seed.csv           one row per (experiment, grid point, seed, checkpoint)
  summary.csv            mean, sample std (ddof=1), min, max, n_seeds
  paired_vs_fedavg_iid.csv  per-seed paired differences against FedAvg-IID
  convergence.csv        validation RMSE per round/epoch, mean ± std over seeds
  partition_stats.csv    per-client windows / engines / RUL distribution, per seed
  per_engine_errors.csv  test error per engine per run (main comparison)
  missing_runs.csv       planned runs without results
"""

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research import config as C

METRICS = ["rmse", "mae", "nasa_score", "bias_mean_error"]
MAIN = ["centralized", "centralized_legacy_es", "fedavg_iid", "local_iid", "fedavg_engine", "local_engine"]


def summarize(df: pd.DataFrame, keys: List[str], metrics: List[str]) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False, sort=False)
    out = g.size().rename("n_seeds").reset_index()
    for m in metrics:
        a = g[m].agg(["mean", "std", "min", "max"]).reset_index(drop=True)
        a.columns = [f"{m}_{c}" for c in a.columns]
        out = pd.concat([out, a], axis=1)
    return out


def paired_differences(df: pd.DataFrame, metric: str, a: str, b: str, key: str = "experiment") -> pd.DataFrame:
    """Per-seed difference a - b on seeds present for both (matched seeds only)."""
    x = df[df[key] == a].set_index("seed")[metric]
    y = df[df[key] == b].set_index("seed")[metric]
    seeds = sorted(set(x.index) & set(y.index))
    return pd.DataFrame({"seed": seeds, a: x.loc[seeds].values, b: y.loc[seeds].values,
                         "diff": (x.loc[seeds] - y.loc[seeds]).values})


def sign_flip_p(diffs: np.ndarray) -> float:
    """Exact two-sided paired sign-flip permutation p-value of the mean difference.
    With n seeds the smallest attainable p is 2 / 2**n (0.0625 for n = 5)."""
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return float("nan")
    obs = abs(d.mean())
    stats = [abs((d * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean(np.array(stats) >= obs - 1e-12))


def flag_timing(ps: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Mark runs whose wall-clock time is not a clean measurement.

    ``timing_concurrent``: the run's [start, finish] interval overlaps another run's (GPU shared).
    ``timing_suspended``: wall time > 3x the median of the same configuration (machine slept).
    Accuracy is unaffected (runs are deterministic); only timing is.
    """
    roots = [os.path.join(out_dir, "runs")] + [os.path.join(out_dir + s, "runs") for s in ("_timing", "_smoke")]
    iv = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for rid in os.listdir(root):
            c, m = os.path.join(root, rid, "config.json"), os.path.join(root, rid, "metrics.json")
            if os.path.exists(c) and os.path.exists(m):
                with open(c, encoding="utf-8") as f1, open(m, encoding="utf-8") as f2:
                    iv.append((os.path.join(root, rid), pd.Timestamp(json.load(f1)["started"]), pd.Timestamp(json.load(f2)["finished"])))
    own = {os.path.basename(k): (s, e) for k, s, e in iv if k.startswith(os.path.join(out_dir, "runs"))}
    def concurrent(rid):
        if rid not in own:
            return None
        s, e = own[rid]
        return any(not (e2 <= s or s2 >= e) for k, s2, e2 in iv if os.path.basename(k) != rid or not k.startswith(os.path.join(out_dir, "runs")))
    ps = ps.copy()
    ps["timing_concurrent"] = ps["run_id"].map(concurrent)
    med = ps.groupby(["experiment", "grid_label"])["wall_s"].transform("median")
    ps["timing_suspended"] = ps["wall_s"] > 3 * med
    ps["timing_clean"] = ~(ps["timing_concurrent"].fillna(True).astype(bool) | ps["timing_suspended"])
    return ps


def load(out_dir: str, seeds: List[int], proto) -> Dict[str, pd.DataFrame]:
    rows, missing, conv, parts, engines = [], [], [], [], []
    for name in proto["experiments"]:
        for point in C.expand_experiment(proto, name):
            for seed in seeds:
                rid = C.run_id(point["config"]["method"], point["config_hash"], seed)
                rdir = os.path.join(out_dir, "runs", rid)
                mpath = os.path.join(rdir, "metrics.json")
                base = {"experiment": name, "grid_label": point["grid_label"], "method": point["config"]["method"],
                        "seed": seed, "run_id": rid, **{k.split(".")[-1]: v for k, v in point["grid_point"].items()}}
                if not os.path.exists(mpath):
                    missing.append(base)
                    continue
                with open(mpath, encoding="utf-8") as f:
                    r = json.load(f)
                if r.get("smoke") or r.get("status") != "complete":
                    missing.append(dict(base, note="smoke/incomplete"))
                    continue
                fed = point["config"].get("fed", {})
                common = dict(base, num_clients=fed.get("num_clients", 1), num_rounds=fed.get("num_rounds"),
                              local_epochs=fed.get("local_epochs"), partition=fed.get("partition", "pooled"),
                              alpha=fed.get("dirichlet_alpha"),
                              best_val_rmse=r["val"].get("best_val_rmse"),
                              selected_at=r["val"].get("selected_at_round", r["val"].get("selected_at_epoch")),
                              sample_passes=r["val"].get("sample_passes"),
                              wall_s=r["timing"].get("wall_s"),
                              mean_round_wall_s=r["timing"].get("mean_round_wall_s"),
                              mean_client_local_train_s=r["timing"].get("mean_client_local_train_s"),
                              parallel_estimate_total_s=r["timing"].get("parallel_estimate_total_s"),
                              comm_total_MB=r["comm"].get("total_bytes", 0) / 1e6,
                              comm_upload_MB=r["comm"].get("total_upload_bytes", 0) / 1e6,
                              comm_download_MB=r["comm"].get("total_download_bytes", 0) / 1e6,
                              payload_bytes=r["comm"]["model_payload"]["serialized_payload_bytes"],
                              partition_attempts=r.get("partition", {}).get("attempts"),
                              client_windows=json.dumps(r.get("partition", {}).get("client_windows")))
                for ck in ("selected", "final"):
                    rows.append(dict(common, checkpoint=ck, **{m: r[f"test_{ck}"].get(m) for m in METRICS},
                                     **({"abs_err_median": r[f"test_{ck}"].get("abs_err_median"),
                                         "abs_err_p90": r[f"test_{ck}"].get("abs_err_p90"),
                                         "frac_late": r[f"test_{ck}"].get("frac_late")} if r["method"] != "local" else {})))
                h = pd.read_csv(os.path.join(rdir, "history.csv"))
                step = "round" if "round" in h.columns else "epoch"
                if "client_id" in h.columns:   # local-only: mean over clients per epoch
                    h = h.groupby(step, as_index=False)["val_rmse"].mean()
                for _, hr in h.iterrows():
                    conv.append({"experiment": name, "grid_label": point["grid_label"], "seed": seed,
                                 "step_type": step, "step": int(hr[step]), "val_rmse": hr["val_rmse"],
                                 **({"cum_total_MB": hr["cum_total_bytes"] / 1e6} if "cum_total_bytes" in h.columns else {})})
                cp = os.path.join(rdir, "clients.csv")
                if os.path.exists(cp):
                    c = pd.read_csv(cp)
                    c.insert(0, "seed", seed); c.insert(0, "grid_label", point["grid_label"]); c.insert(0, "experiment", name)
                    parts.append(c)
                if name in MAIN:
                    tp = pd.read_csv(os.path.join(rdir, "test_predictions.csv"))
                    cols = [c for c in tp.columns if c.startswith("err_") and c.endswith("selected")]
                    tp["abs_err_selected"] = tp[cols].abs().mean(axis=1)
                    engines.append(tp[["unit", "rul_true", "abs_err_selected"]].assign(experiment=name, seed=seed))
    return {"per_seed": pd.DataFrame(rows), "missing": pd.DataFrame(missing), "conv": pd.DataFrame(conv),
            "parts": pd.concat(parts) if parts else pd.DataFrame(), "engines": pd.concat(engines) if engines else pd.DataFrame()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=C.DEFAULT_PROTOCOL)
    ap.add_argument("--seed-set", default="primary")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out", default=C.DEFAULT_OUT_DIR)
    a = ap.parse_args(argv)
    proto = C.load_protocol(a.protocol)
    seeds = a.seeds or proto["seeds"][a.seed_set]
    d = load(a.out, seeds, proto)
    tdir = os.path.join(a.out, "tables")
    os.makedirs(tdir, exist_ok=True)
    ps = d["per_seed"]
    d["missing"].to_csv(os.path.join(tdir, "missing_runs.csv"), index=False)
    if ps.empty:
        print("No completed runs found.")
        return 1
    ps = flag_timing(ps, a.out)
    ps.to_csv(os.path.join(tdir, "per_seed.csv"), index=False)
    keys = ["experiment", "grid_label", "method", "partition", "num_clients", "num_rounds", "local_epochs", "alpha", "checkpoint"]
    summ = summarize(ps, keys, METRICS + ["best_val_rmse", "wall_s", "comm_total_MB", "selected_at"])
    summ.to_csv(os.path.join(tdir, "summary.csv"), index=False)

    pr = []
    for ck in ("selected", "final"):
        sub = ps[(ps.checkpoint == ck) & (ps.grid_label == "default")]
        for other in [m for m in MAIN if m != "fedavg_iid"]:
            for m in ("rmse", "mae", "nasa_score"):
                p = paired_differences(sub, m, other, "fedavg_iid")
                if len(p) == 0:
                    continue
                pr.append({"checkpoint": ck, "comparison": f"{other} - fedavg_iid", "metric": m, "n_matched_seeds": len(p),
                           "mean_diff": p["diff"].mean(), "std_diff": p["diff"].std(ddof=1) if len(p) > 1 else np.nan,
                           "n_other_lower": int((p["diff"] < 0).sum()), "per_seed_diff": json.dumps([round(x, 3) for x in p["diff"]]),
                           "sign_flip_p_two_sided": sign_flip_p(p["diff"].values),
                           "min_attainable_p": 2 / 2 ** len(p)})
    pd.DataFrame(pr).to_csv(os.path.join(tdir, "paired_vs_fedavg_iid.csv"), index=False)

    if not d["conv"].empty:
        cv = d["conv"]
        g = cv.groupby(["experiment", "grid_label", "step_type", "step"])
        cs = g["val_rmse"].agg(["mean", "std", "count"]).reset_index()
        if "cum_total_MB" in cv.columns:
            cs = cs.merge(g["cum_total_MB"].mean().reset_index(), on=["experiment", "grid_label", "step_type", "step"])
        cs.to_csv(os.path.join(tdir, "convergence.csv"), index=False)
    if not d["parts"].empty:
        d["parts"].to_csv(os.path.join(tdir, "partition_stats.csv"), index=False)
    if not d["engines"].empty:
        d["engines"].to_csv(os.path.join(tdir, "per_engine_errors.csv"), index=False)
    print(f"{len(ps) // 2} completed runs aggregated; {len(d['missing'])} planned runs missing. Tables -> {tdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
