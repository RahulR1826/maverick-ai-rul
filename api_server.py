"""
api_server.py
-------------
FastAPI backend for the Maverick AI dashboard (React frontend in ``frontend/``).

Serves real model output only:
* 5-seed ensembles of the v2 checkpoints, replaying the 100 official FD001 test
  engines cycle by cycle, with validation-calibrated prediction intervals;
* integrated-gradients explanations computed from the served models;
* the v2 experiment results (``results/v2/tables``).
Missing artifacts produce HTTP 503 with an explanation; nothing is simulated.

Run:  python api_server.py    (http://localhost:8000, docs at /docs)
"""

import math
import os
import threading
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from serving.model_store import DEFAULT_FAMILY, DEFAULT_THRESHOLDS, FAMILIES, FEATURE_INFO, ModelStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE_DIR, "results", "v2", "tables")
TIMING_TABLES = os.path.join(BASE_DIR, "results", "v2_timing", "tables")

app = FastAPI(title="Maverick AI — RUL serving API",
              description="Real FD001 predictions from the v2 federated/centralized checkpoints.")

# The dashboard is read-only and uses no cookies, so credentials are not allowed.
origins = os.environ.get("MAVERICK_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                   allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])

_store: Optional[ModelStore] = None
_store_error: Optional[str] = None
_lock = threading.Lock()


def store() -> ModelStore:
    global _store, _store_error
    with _lock:
        if _store is None and _store_error is None:
            try:
                _store = ModelStore()
            except (FileNotFoundError, AssertionError) as ex:
                _store_error = str(ex)
    if _store is None:
        raise HTTPException(503, f"Model store unavailable: {_store_error}")
    return _store


@app.on_event("startup")
def _warm() -> None:
    def _load():
        try:
            store().family_info(DEFAULT_FAMILY)
        except HTTPException:
            pass
    threading.Thread(target=_load, daemon=True).start()


def thresholds(critical: float, high: float, medium: float) -> Dict[str, float]:
    if not (0 <= critical <= high <= medium):
        raise HTTPException(422, "thresholds must satisfy 0 <= critical <= high <= medium")
    return {"critical": critical, "high": high, "medium": medium}


def family_param(family: str) -> str:
    if family not in FAMILIES:
        raise HTTPException(404, f"Unknown model family '{family}'. Available: {list(FAMILIES)}")
    return family


def clean(obj):
    """JSON-safe: NaN/inf -> None (pandas tables contain NaN for undefined values)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def table(name: str, root: str = TABLES) -> pd.DataFrame:
    path = os.path.join(root, name)
    if not os.path.exists(path):
        raise HTTPException(503, f"{os.path.relpath(path, BASE_DIR)} not found. Run `python -m research.aggregate`.")
    return pd.read_csv(path)


# ----------------------------------------------------------------- serving
@app.get("/api/health")
def health():
    ready = _store is not None and DEFAULT_FAMILY in _store._families
    return {"status": "ready" if ready else ("error" if _store_error else "loading"), "error": _store_error}


@app.get("/api/meta")
def meta(family: str = DEFAULT_FAMILY):
    s = store()
    info = s.family_info(family_param(family))
    return clean({
        "dataset": {"name": "NASA C-MAPSS", "subset": s.data_cfg["dataset_id"], "test_engines": len(s.engines),
                    "window": s.W, "rul_cap": s.cap, "features": [
                        {"key": c, "name": FEATURE_INFO[c][0], "description": FEATURE_INFO[c][1], "unit": FEATURE_INFO[c][2]}
                        for c in s.cols]},
        "model": {**info, "seeds": s.seeds, "checkpoint": "validation-selected (results/v2 protocol)",
                  "architecture": f"LSTM {s.model_cfg['num_layers']}x{s.model_cfg['hidden_size']}, dropout {s.model_cfg['dropout']}"},
        "families": FAMILIES, "default_family": DEFAULT_FAMILY,
        "interval": {"coverage_target": s.coverage,
                     "method": "ensemble mean ± k·SD, k calibrated on the 10 validation engines"},
        "risk": {"rule": "risk from the interval's LOWER RUL bound", "default_thresholds": DEFAULT_THRESHOLDS},
        "ground_truth_note": "True RUL is known for test engines (RUL_FD001.txt) and shown for evaluation only.",
    })


@app.get("/api/fleet")
def fleet(family: str = DEFAULT_FAMILY, critical: float = 20, high: float = 40, medium: float = 70):
    return clean(store().fleet(family_param(family), thresholds(critical, high, medium)))


@app.get("/api/engines/{engine_id}")
def engine(engine_id: int, family: str = DEFAULT_FAMILY, critical: float = 20, high: float = 40, medium: float = 70):
    s = store()
    if engine_id not in s.engines:
        raise HTTPException(404, f"Engine {engine_id} not found (valid: 1-{len(s.engines)})")
    return clean(s.engine(engine_id, family_param(family), thresholds(critical, high, medium)))


@app.get("/api/engines/{engine_id}/explain")
def explain(engine_id: int, cycle: Optional[int] = None, family: str = DEFAULT_FAMILY,
            steps: int = Query(256, ge=8, le=512)):
    s = store()
    if engine_id not in s.engines:
        raise HTTPException(404, f"Engine {engine_id} not found")
    cyc = int(s.engines[engine_id].cycles[-1]) if cycle is None else cycle
    try:
        return clean(s.explain(engine_id, cyc, family_param(family), steps=steps))
    except ValueError as ex:
        raise HTTPException(422, str(ex))


# ----------------------------------------------------------------- results
SUMMARY_COLS = ["experiment", "grid_label", "method", "partition", "num_clients", "num_rounds", "local_epochs", "alpha",
                "checkpoint", "n_seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "nasa_score_mean",
                "nasa_score_std", "best_val_rmse_mean", "comm_total_MB_mean", "selected_at_mean"]


@app.get("/api/results/summary")
def results_summary(checkpoint: str = Query("selected", pattern="^(selected|final)$")):
    s = table("summary.csv")
    return clean(s[s.checkpoint == checkpoint][SUMMARY_COLS].to_dict("records"))


@app.get("/api/results/per_seed")
def results_per_seed(experiment: Optional[str] = None, checkpoint: str = Query("selected", pattern="^(selected|final)$")):
    p = table("per_seed.csv")
    p = p[p.checkpoint == checkpoint]
    if experiment:
        p = p[p.experiment == experiment]
    cols = ["experiment", "grid_label", "seed", "rmse", "mae", "nasa_score", "num_clients", "num_rounds", "local_epochs",
            "alpha", "partition", "comm_total_MB", "timing_clean"]
    return clean(p[cols].to_dict("records"))


@app.get("/api/results/convergence")
def results_convergence(experiment: Optional[str] = None):
    c = table("convergence.csv")
    if experiment:
        c = c[c.experiment == experiment]
    return clean(c.to_dict("records"))


@app.get("/api/results/partitions")
def results_partitions(seed: int = 42):
    p = table("partition_stats.csv")
    p = p[(p.seed == seed) & p.experiment.isin(["fedavg_iid", "fedavg_dirichlet", "fedavg_engine"])]
    cols = ["experiment", "grid_label", "client_id", "n_windows", "n_engines", "n_engines_exclusive", "rul_mean", "rul_std"]
    return clean(p[cols].to_dict("records"))


@app.get("/api/results/paired")
def results_paired():
    return clean(table("paired_vs_fedavg_iid.csv").to_dict("records"))


@app.get("/api/results/runtime")
def results_runtime():
    t = table("per_seed.csv", TIMING_TABLES)
    t = t[(t.checkpoint == "selected") & (t.grid_label == "default") &
          t.experiment.isin(["centralized", "centralized_legacy_es", "fedavg_iid", "local_iid", "fedavg_engine", "local_engine"])]
    cols = ["experiment", "wall_s", "mean_round_wall_s", "mean_client_local_train_s", "parallel_estimate_total_s",
            "sample_passes", "payload_bytes", "comm_total_MB"]
    return clean(t[cols].to_dict("records"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
