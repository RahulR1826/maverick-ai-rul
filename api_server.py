"""
api_server.py
-------------
FastAPI backend for the Maverick AI dashboard (React frontend in ``frontend/``).

Serves real model output only:
* 5-seed ensembles of the v2 checkpoints, replaying the 100 official FD001 test
  engines cycle by cycle, with conformal (Mondrian) prediction intervals;
* uploaded engines in the C-MAPSS file format, with an out-of-distribution flag;
* integrated-gradients explanations (precomputed for every cycle when the cache exists);
* side-by-side comparison of the federated and centralized models;
* a short live FedAvg training run streamed round by round (server-sent events);
* the v2 experiment results (``results/v2/tables``).
Missing artifacts produce HTTP 503 with an explanation; nothing is simulated.

Run:  python api_server.py    (http://localhost:8000, docs at /docs)
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before any CUDA use (live training)

import asyncio
import json
import math
import queue
import subprocess
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional, Union

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from serving import training_job
from serving.model_store import DEFAULT_FAMILY, DEFAULT_THRESHOLDS, FAMILIES, FEATURE_INFO, MAX_UPLOADS, ModelStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE_DIR, "results", "v2", "tables")
TIMING_TABLES = os.path.join(BASE_DIR, "results", "v2_timing", "tables")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
API_VERSION = "2.1"

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


@asynccontextmanager
async def lifespan(_: FastAPI):
    def _warm():
        try:
            store().family_info(DEFAULT_FAMILY)
        except HTTPException:
            pass
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Maverick AI — RUL serving API", version=API_VERSION, lifespan=lifespan,
              description="Real FD001 predictions from the v2 federated/centralized checkpoints.")

# No cookies or credentials; only the dashboard's own origins may call the API.
origins = os.environ.get("MAVERICK_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return None


GIT_COMMIT = git_commit()


def thresholds(critical: float, high: float, medium: float) -> Dict[str, float]:
    if not (0 <= critical <= high <= medium):
        raise HTTPException(422, "thresholds must satisfy 0 <= critical <= high <= medium")
    return {"critical": critical, "high": high, "medium": medium}


def family_param(family: str) -> str:
    if family not in FAMILIES:
        raise HTTPException(404, f"Unknown model family '{family}'. Available: {list(FAMILIES)}")
    return family


def engine_ref(engine_id: str) -> Union[int, str]:
    """Test engines are numbered 1-100; uploads have ids like 'u-1a2b3c4d'."""
    if engine_id.isdigit():
        return int(engine_id)
    if engine_id.startswith("u-"):
        return engine_id
    raise HTTPException(404, f"Engine '{engine_id}' not found")


def clean(obj):
    """JSON-safe: NaN/inf -> None (pandas tables contain NaN for undefined values)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def table(name: str, root: str = TABLES) -> pd.DataFrame:
    path = os.path.join(root, name)
    if not os.path.exists(path):
        raise HTTPException(503, f"{os.path.relpath(path, BASE_DIR)} not found. Run `python -m research.aggregate`.")
    return pd.read_csv(path)


def _lookup(fn):
    try:
        return fn()
    except KeyError as ex:
        raise HTTPException(404, str(ex).strip("'\""))
    except LookupError as ex:
        raise HTTPException(409, str(ex))
    except ValueError as ex:
        raise HTTPException(422, str(ex))


# ----------------------------------------------------------------- serving
@app.get("/api/health")
def health():
    ready = _store is not None and DEFAULT_FAMILY in _store._families
    return {"status": "ready" if ready else ("error" if _store_error else "loading"), "error": _store_error,
            "families_loaded": sorted(_store._families) if _store else [], "api_version": API_VERSION,
            "git_commit": GIT_COMMIT, "cuda_available": torch.cuda.is_available(), "training_busy": training_job.busy()}


@app.get("/api/meta")
def meta(family: str = DEFAULT_FAMILY):
    s = store()
    info = s.family_info(family_param(family))
    return clean({
        "dataset": {"name": "NASA C-MAPSS", "subset": s.data_cfg["dataset_id"], "test_engines": len(s.engines),
                    "window": s.W, "rul_cap": s.cap, "features": [
                        {"key": c, "name": FEATURE_INFO[c][0], "description": FEATURE_INFO[c][1], "unit": FEATURE_INFO[c][2]}
                        for c in s.cols]},
        "model": {**{k: v for k, v in info.items() if k != "interval"}, "seeds": s.seeds,
                  "checkpoint": "validation-selected (results/v2 protocol)", "checkpoints": s.checkpoint_digests(family),
                  "architecture": f"LSTM {s.model_cfg['num_layers']}x{s.model_cfg['hidden_size']}, dropout {s.model_cfg['dropout']}"},
        "families": FAMILIES, "default_family": DEFAULT_FAMILY,
        "interval": {"coverage_target": s.coverage, "method": info["interval"]["served"]["method"],
                     "calibration": info["interval"]["served"], "baseline_global": info["interval"]["baseline_global"],
                     "evaluation": info["interval"]["evaluation"],
                     "note": "Calibrated on the 10 validation engines; coverage on test engines is measured, not guaranteed."},
        "risk": {"rule": "risk from the interval's LOWER RUL bound", "default_thresholds": DEFAULT_THRESHOLDS},
        "ground_truth_note": "True RUL is known for test engines (RUL_FD001.txt) and shown for evaluation only.",
        "version": {"api": API_VERSION, "git_commit": GIT_COMMIT, "cache_key": info["cache_key"], "cache": info["cache"]},
    })


@app.get("/api/fleet")
def fleet(family: str = DEFAULT_FAMILY, critical: float = 20, high: float = 40, medium: float = 70):
    return clean(store().fleet(family_param(family), thresholds(critical, high, medium)))


@app.get("/api/engines/{engine_id}")
def engine(engine_id: str, family: str = DEFAULT_FAMILY, critical: float = 20, high: float = 40, medium: float = 70):
    s, ref, th = store(), engine_ref(engine_id), thresholds(critical, high, medium)
    return clean(_lookup(lambda: s.engine(ref, family_param(family), th)))


@app.get("/api/engines/{engine_id}/explain")
def explain(engine_id: str, cycle: Optional[int] = None, family: str = DEFAULT_FAMILY,
            steps: Optional[int] = Query(None, ge=8, le=512)):
    s, ref = store(), engine_ref(engine_id)

    def run():
        e = s.series(ref)
        cyc = int(e.cycles[-1]) if cycle is None else cycle
        fam = s._load_family(family_param(family))
        # the precomputed cache answers instantly when it exists; otherwise compute with 256 steps
        st = steps or (fam["ig"]["steps"] if "ig" in fam and isinstance(ref, int) else 256)
        return s.explain(ref, cyc, family, steps=st)
    return clean(_lookup(run))


@app.get("/api/engines/{engine_id}/attributions")
def attributions(engine_id: str, family: str = DEFAULT_FAMILY):
    s, ref = store(), engine_ref(engine_id)
    if not isinstance(ref, int):
        raise HTTPException(409, "Per-cycle attributions are precomputed for the FD001 test engines only; use /explain for uploads.")
    return clean(_lookup(lambda: s.attribution_series(ref, family_param(family))))


@app.get("/api/engines/{engine_id}/compare")
def compare(engine_id: str, critical: float = 20, high: float = 40, medium: float = 70):
    s, ref, th = store(), engine_ref(engine_id), thresholds(critical, high, medium)
    return clean(_lookup(lambda: s.compare(ref, th)))


# ----------------------------------------------------------------- uploads
@app.post("/api/uploads")
async def upload(file: UploadFile = File(...), rul_at_end: Optional[float] = Form(None)):
    """Upload engine data in NASA's C-MAPSS text format (26 columns). Kept in memory only."""
    body = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // 1024} KB")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(422, "File must be UTF-8 text")
    if rul_at_end is not None and not (0 <= rul_at_end <= 1000):
        raise HTTPException(422, "rul_at_end must be between 0 and 1000")
    s = store()
    created = _lookup(lambda: s.add_upload(text, os.path.basename(file.filename or "upload.txt")[:80], rul_at_end))
    return {"uploads": created, "note": f"Kept in server memory only (at most {MAX_UPLOADS}); lost on restart.",
            "warning": ("Some cycles have inputs outside the training range; predictions there are extrapolations."
                        if any(c["max_ood_fraction"] > 0 for c in created) else None)}


@app.get("/api/uploads")
def list_uploads():
    s = store()
    return [{"upload_id": k, "source": v.source, "predicted_cycles": int(len(v.cycles)), "has_ground_truth": v.true_rul is not None}
            for k, v in reversed(s.uploads.items())]


# ----------------------------------------------------------------- live training
@app.get("/api/train/status")
def train_status():
    return {"busy": training_job.busy(), "limits": training_job.LIMITS, "cuda_available": torch.cuda.is_available()}


@app.get("/api/train/stream")
async def train_stream(request: Request, num_clients: int = 5, num_rounds: int = 10, local_epochs: int = 2,
                       partition: str = "iid_window", alpha: Optional[float] = None, seed: int = 42):
    """Server-sent events: start, partition, round (one per round), then done | error | cancelled."""
    params = training_job.JobParams(num_clients, num_rounds, local_epochs, partition, alpha, seed)
    try:
        params.validate()
    except ValueError as ex:
        raise HTTPException(422, str(ex))
    if training_job.busy():
        raise HTTPException(409, "A training job is already running")
    q: "queue.Queue[dict]" = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=_run_job, args=(params, q, stop), daemon=True).start()

    async def events():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                yield f"event: {ev['type']}\ndata: {json.dumps(clean(ev))}\n\n"
                if ev["type"] in ("done", "error", "cancelled"):
                    break
        finally:
            stop.set()        # client left or stream ended: stop training at the next round
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _run_job(params, q, stop) -> None:
    try:
        training_job.run(params, q.put, stop.is_set)
    except RuntimeError as ex:          # another job grabbed the lock first
        q.put({"type": "error", "message": str(ex)})


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
