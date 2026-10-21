"""
research/run.py
---------------
Command-line runner.

Examples
--------
python -m research.run --list
python -m research.run --suite essential                      # 5 primary seeds
python -m research.run --suite essential --seed-set fallback  # 3-seed fallback
python -m research.run --experiments centralized fedavg_iid --seeds 42 123
python -m research.run --suite sweeps --dry-run
python -m research.run --suite essential --smoke              # tiny budget, results/v2_smoke

Runs are resumable: a run whose ``metrics.json`` exists is skipped (use
``--force`` to recompute; the old directory is renamed, never deleted).
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch/CUDA init

import argparse
import datetime as dt
import json
import logging
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research import config as C
from research import trainer as TR
from research.comm import fedavg_traffic, payload_info
from research.data import get_data
from research.partition import client_stats, derive_seed, partition

log = logging.getLogger("research")


def _setup_logging(out_dir: str) -> str:
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    path = os.path.join(out_dir, "logs", f"run_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return path


def environment_info(device: torch.device) -> Dict:
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                                             stderr=subprocess.DEVNULL).strip())
    except Exception:
        git, dirty = "unknown", None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "git_commit": git, "git_dirty": dirty}


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def _save_ckpt(path: str, state, meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": {k: v.cpu() for k, v in state.items()}, "meta": meta}, path)


def _test_rows(data, preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame({"unit": data.test.unit, "last_cycle": data.test.cycle, "rul_true": data.test.y,
                       "rul_true_uncapped": data.test_rul_uncapped})
    for k, p in preds.items():
        df[f"pred_{k}"] = p
        df[f"err_{k}"] = p - data.test.y
    return df


def execute_run(entry: Dict, out_dir: str, ckpt_dir: str, smoke: bool, force: bool) -> str:
    rid = entry["run_id"]
    run_dir = os.path.join(out_dir, "runs", rid)
    done = os.path.join(run_dir, "metrics.json")
    if os.path.exists(done) and not force:
        log.info("SKIP %s (complete)", rid)
        return "skipped"
    if os.path.exists(run_dir):
        keep = f"{run_dir}.prev_{dt.datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(run_dir, keep)
        log.info("Existing directory kept as %s", keep)
    os.makedirs(run_dir)

    cfg = C.smoke_override(entry["config"]) if smoke else entry["config"]
    seed = int(entry["seed"])
    det = TR.setup_determinism(cfg["train"]["deterministic"])
    device = TR.resolve_device(cfg["train"]["device"])
    data = get_data(cfg["data"], C.DATA_DIR)
    T = TR.Tensors(data, device)
    nF = data.meta["n_features"]
    meta = {"run_id": rid, "experiment": entry["experiment"], "grid_point": entry["grid_point"],
            "aliases": entry.get("aliases", []), "seed": seed, "config_hash": entry["config_hash"],
            "smoke": smoke, "config": cfg, "determinism": det, "environment": environment_info(device),
            "seed_streams": {s: derive_seed(seed, *s.split(":")) for s in ("init", "train")},
            "started": dt.datetime.now().isoformat(timespec="seconds")}
    _write_json(os.path.join(run_dir, "config.json"), meta)
    log.info("RUN %s | %s %s | seed=%d", rid, entry["experiment"], entry["grid_label"], seed)

    method = cfg["method"]
    result: Dict = {"run_id": rid, "method": method, "seed": seed, "experiment": entry["experiment"],
                    "grid_point": entry["grid_point"], "config_hash": entry["config_hash"], "smoke": smoke,
                    "data": data.meta | {"scaler_data_min": None, "scaler_data_max": None}}
    say = lambda m: log.info(m)
    model = TR.init_model(cfg["model"], nF, seed, device)
    pinfo = payload_info(model)
    result["model"] = {"n_parameters": pinfo["n_parameters"]}

    if method == "centralized":
        cen = cfg["centralized"]
        torch.manual_seed(derive_seed(seed, "train"))
        gen = torch.Generator().manual_seed(derive_seed(seed, "shuffle", "central"))
        out = TR.train_single(model, T.Xtr, T.ytr, T, cfg["train"], cen["epochs"], cen["select_every"], gen,
                              device, early_stopping=cen["early_stopping"], patience=cen["patience"],
                              min_delta=cen["min_delta"], lr_plateau=cen["lr_plateau"], log=say)
        model.load_state_dict(out["best_state"]); sel, p_sel = TR.test_eval(model, T)
        model.load_state_dict(out["final_state"]); fin, p_fin = TR.test_eval(model, T)
        pd.DataFrame(out["history"]).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        _test_rows(data, {"selected": p_sel, "final": p_fin}).to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
        _save_ckpt(os.path.join(ckpt_dir, rid, "selected.pt"), out["best_state"], {"epoch": out["best_epoch"]})
        _save_ckpt(os.path.join(ckpt_dir, rid, "final.pt"), out["final_state"], {"epoch": out["epochs_run"]})
        result.update({
            "test_selected": sel, "test_final": fin,
            "val": {"best_val_rmse": out["best_val_rmse"], "selected_at_epoch": out["best_epoch"],
                    "final_val_rmse": out["history"][-1]["val_rmse"], "epochs_run": out["epochs_run"],
                    "early_stopped_at": out["early_stopped_at"],
                    "sample_passes": out["epochs_run"]},
            "timing": {"wall_s": out["wall_s"], "train_compute_s": out["train_compute_s"]},
            "comm": {"model_payload": pinfo, "total_bytes": 0,
                     "note": "No model exchange. Raw training data would have to be centralised instead.",
                     "raw_training_data_float32_bytes_estimate": int(data.meta["raw_train_rows"] * (nF + 1) * 4)},
        })

    else:
        fed = cfg["fed"]
        part = partition(data.train, fed, cfg["data"]["rul_cap"], seed)
        cs = client_stats(data.train, part["indices"], cfg["data"]["rul_cap"], fed["n_rul_bins"])
        pd.DataFrame(cs).to_csv(os.path.join(run_dir, "clients.csv"), index=False)
        result["partition"] = {"kind": fed["partition"], "num_clients": fed["num_clients"],
                               "alpha": fed["dirichlet_alpha"], "attempts": part["attempts"],
                               "min_client_windows": fed["min_client_windows"],
                               "partition_seed": derive_seed(seed, "partition", fed["partition"], fed["num_clients"], fed["dirichlet_alpha"]),
                               "client_windows": [c["n_windows"] for c in cs],
                               "client_engines": [c["n_engines"] for c in cs],
                               "client_rul_mean": [round(c["rul_mean"], 2) for c in cs],
                               "engine_groups": part["engine_groups"]}
        R, E = fed["num_rounds"], fed["local_epochs"]
        if method == "fedavg":
            torch.manual_seed(derive_seed(seed, "train"))
            out = TR.run_fedavg(model, part["indices"], T, cfg["train"], fed, seed, device, log=say)
            model.load_state_dict(out["best_state"]); sel, p_sel = TR.test_eval(model, T)
            model.load_state_dict(out["final_state"]); fin, p_fin = TR.test_eval(model, T)
            hist = pd.DataFrame(out["history"])
            traffic = fedavg_traffic(pinfo["serialized_payload_bytes"], [fed["num_clients"]] * R)
            tr = pd.DataFrame(traffic["per_round"])
            hist = hist.merge(tr, on="round")
            hist.to_csv(os.path.join(run_dir, "history.csv"), index=False)
            pd.DataFrame(out["client_rounds"]).to_csv(os.path.join(run_dir, "client_rounds.csv"), index=False)
            _test_rows(data, {"selected": p_sel, "final": p_fin}).to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
            _save_ckpt(os.path.join(ckpt_dir, rid, "selected.pt"), out["best_state"], {"round": out["best_round"]})
            _save_ckpt(os.path.join(ckpt_dir, rid, "final.pt"), out["final_state"], {"round": R})
            result.update({
                "test_selected": sel, "test_final": fin,
                "val": {"best_val_rmse": out["best_val_rmse"], "selected_at_round": out["best_round"],
                        "final_val_rmse": out["history"][-1]["val_rmse"], "sample_passes": R * E},
                "timing": {"wall_s": out["wall_s"],
                           "mean_round_wall_s": float(hist["round_wall_s"].mean()),
                           "total_client_train_s": float(hist["sum_client_train_s"].sum()),
                           "mean_client_local_train_s": float(pd.DataFrame(out["client_rounds"])["local_train_s"].mean()),
                           "total_aggregation_s": float(hist["aggregation_s"].sum()),
                           "total_server_val_s": float(hist["server_val_s"].sum()),
                           "parallel_estimate_total_s": float(hist["parallel_round_estimate_s"].sum()),
                           "note": "Clients run sequentially on one device; parallel_estimate = sum over rounds of (slowest client + aggregation), a derived estimate."},
                "comm": {"model_payload": pinfo, "rounds": R, "clients_per_round": fed["num_clients"],
                         "precision": "float32", "protocol_overhead_included": False,
                         "total_upload_bytes": traffic["total_upload_bytes"],
                         "total_download_bytes": traffic["total_download_bytes"],
                         "total_bytes": traffic["total_bytes"], "label": traffic["label"]},
            })
        elif method == "local":
            rows, preds, hists = [], {}, []
            t0 = TR.now(device)
            for c, idx in enumerate(part["indices"]):
                model = TR.init_model(cfg["model"], nF, seed, device)     # same init for every client
                torch.manual_seed(derive_seed(seed, "train", "local", c))
                gen = torch.Generator().manual_seed(derive_seed(seed, "shuffle", "client", c))
                ii = torch.from_numpy(idx).to(device)
                say(f"  local client {c} ({len(idx)} windows)")
                out = TR.train_single(model, T.Xtr[ii], T.ytr[ii], T, cfg["train"], R * E, E, gen, device, log=None)
                model.load_state_dict(out["best_state"]); sel, p_sel = TR.test_eval(model, T)
                model.load_state_dict(out["final_state"]); fin, p_fin = TR.test_eval(model, T)
                preds[f"c{c}_selected"], preds[f"c{c}_final"] = p_sel, p_fin
                h = pd.DataFrame(out["history"]); h.insert(0, "client_id", c); hists.append(h)
                _save_ckpt(os.path.join(ckpt_dir, rid, f"client{c}_selected.pt"), out["best_state"], {"epoch": out["best_epoch"]})
                rows.append({"client_id": c, "n_windows": len(idx), "best_val_rmse": out["best_val_rmse"],
                             "selected_at_epoch": out["best_epoch"], "train_compute_s": out["train_compute_s"],
                             **{f"sel_{k}": v for k, v in sel.items()}, **{f"fin_{k}": v for k, v in fin.items()}})
                say(f"    client {c}: val={out['best_val_rmse']:.2f} test_rmse(sel)={sel['rmse']:.2f}")
            cdf = pd.DataFrame(rows)
            cdf.to_csv(os.path.join(run_dir, "local_clients.csv"), index=False)
            pd.concat(hists).to_csv(os.path.join(run_dir, "history.csv"), index=False)
            _test_rows(data, preds).to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
            mean_of = lambda pre: {k: float(cdf[f"{pre}_{k}"].mean()) for k in ("rmse", "mae", "nasa_score", "bias_mean_error")}
            result.update({
                "test_selected": mean_of("sel"), "test_final": mean_of("fin"),
                "test_aggregation_note": "Local-only metrics are the unweighted MEAN over the K client models, each evaluated on all 100 test engines.",
                "local_client_spread": {"sel_rmse_min": float(cdf["sel_rmse"].min()), "sel_rmse_max": float(cdf["sel_rmse"].max()),
                                        "sel_rmse_std_across_clients": float(cdf["sel_rmse"].std(ddof=0))},
                "val": {"best_val_rmse": float(cdf["best_val_rmse"].mean()), "sample_passes": R * E},
                "timing": {"wall_s": TR.now(device) - t0, "train_compute_s": float(cdf["train_compute_s"].sum())},
                "comm": {"model_payload": pinfo, "total_bytes": 0, "note": "No communication."},
            })
    result["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    result["status"] = "complete"
    _write_json(done, result)
    log.info("DONE %s | test RMSE selected=%.3f final=%.3f | NASA selected=%.1f",
             rid, result["test_selected"]["rmse"], result["test_final"]["rmse"], result["test_selected"]["nasa_score"])
    return "done"


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default=C.DEFAULT_PROTOCOL)
    ap.add_argument("--suite", nargs="*", default=[])
    ap.add_argument("--experiments", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--seed-set", default="primary", help="named seed list in the protocol (primary|fallback)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny budget; writes to results/v2_smoke")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args(argv)

    proto = C.load_protocol(a.protocol)
    if a.list:
        print("Suites:", ", ".join(C.all_suites(proto)))
        for n, e in proto["experiments"].items():
            pts = C.expand_experiment(proto, n)
            print(f"  {n:24s} [{','.join(e.get('suites', []))}] {len(pts)} config(s): {e['description']}")
        print("Seed sets:", proto["seeds"])
        return 0
    names = list(a.experiments) + (C.experiments_for_suites(proto, a.suite) if a.suite else [])
    names = list(dict.fromkeys(names))
    if not names:
        ap.error("give --suite and/or --experiments (or --list)")
    seeds = a.seeds if a.seeds else proto["seeds"][a.seed_set]
    out_dir = a.out or (C.DEFAULT_OUT_DIR + "_smoke" if a.smoke else C.DEFAULT_OUT_DIR)
    ckpt_dir = os.path.join(C.DEFAULT_CKPT_DIR + ("_smoke" if a.smoke else ""))
    plan = C.plan_runs(proto, names, seeds)
    logpath = _setup_logging(out_dir)
    log.info("Experiments: %s | seeds: %s | %d unique runs | out=%s | log=%s", names, seeds, len(plan), out_dir, logpath)
    if a.dry_run:
        for e in plan:
            status = "complete" if os.path.exists(os.path.join(out_dir, "runs", e["run_id"], "metrics.json")) else "pending"
            print(f"{e['run_id']:32s} {e['experiment']:22s} {e['grid_label']:22s} seed={e['seed']:<5d} {status}"
                  + (f"  (also: {e['aliases']})" if e["aliases"] else ""))
        return 0
    status = {"done": 0, "skipped": 0, "failed": 0}
    failures = []
    for i, e in enumerate(plan, 1):
        log.info("[%d/%d]", i, len(plan))
        try:
            status[execute_run(e, out_dir, ckpt_dir, a.smoke, a.force)] += 1
        except Exception as ex:   # record and continue; failed runs leave no metrics.json
            status["failed"] += 1
            failures.append({"run_id": e["run_id"], "experiment": e["experiment"], "grid": e["grid_label"],
                             "seed": e["seed"], "error": repr(ex)})
            log.error("FAILED %s: %s\n%s", e["run_id"], ex, traceback.format_exc())
            if a.stop_on_error:
                break
    if failures:
        with open(os.path.join(out_dir, "failures.jsonl"), "a", encoding="utf-8") as f:
            for x in failures:
                f.write(json.dumps(x) + "\n")
    log.info("Finished: %s", status)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
