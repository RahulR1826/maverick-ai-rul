"""
serving/model_store.py
----------------------
Loads the v2 checkpoints as 5-seed ensembles, replays the 100 official FD001 test
engines (and user-uploaded engines) cycle by cycle, calibrates prediction
intervals with conformal prediction and serves integrated-gradients attributions.

Data protocol (identical to research/data.py, asserted at start-up):
* MinMax scaler fitted on the 90 training engines only;
* a prediction at cycle c uses the 30 cycles ending at c (so c >= 30);
* ground truth for a test engine at cycle c is RUL_file + (T_engine - c), capped at
  125 like the training target. Test labels are used only for display and
  evaluation, never for fitting anything here.

Prediction interval: ensemble mean ± k(mu)·SD, with k from Mondrian (per predicted-RUL
bin) normalised split conformal calibration on the held-out validation engines
(see serving/conformal.py). Coverage on the test engines is measured and reported.

Caching: ensemble predictions (and, if precomputed, attributions for every cycle)
are stored under cache/serving/, keyed by a hash of the checkpoint files, the
data configuration and CACHE_VERSION, so stale results can never be served.
Build the full cache with ``python -m serving.precompute`` (uses the GPU if present).
"""

import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from models.lstm_model import build_model
from research import config as C
from research import metrics as M
from research.data import COLUMN_NAMES, add_train_rul, feature_columns, get_data, load_raw, split_engines
from serving.conformal import GlobalConformal, MondrianConformal, evaluate

CKPT_ROOT = os.path.join(C.BASE_DIR, "checkpoints", "v2")
CACHE_DIR = os.path.join(C.BASE_DIR, "cache", "serving")
CACHE_VERSION = "serving-cache-v1"
IG_STEPS = 128            # steps used for precomputed attributions (single requests use 256)

FAMILIES = {
    "fedavg_iid": "FedAvg, K=5, IID window partition",
    "fedavg_engine": "FedAvg, K=5, whole-engine ownership",
    "centralized": "Centralized (pooled data, same budget)",
}
DEFAULT_FAMILY = "fedavg_iid"

# Sensor descriptions from Saxena et al. (2008), Table 2.
FEATURE_INFO = {
    "os_1": ("Setting 1", "Operational setting 1", ""),
    "os_2": ("Setting 2", "Operational setting 2", ""),
    "os_3": ("Setting 3", "Operational setting 3 (constant in FD001)", ""),
    "s_2": ("T24", "Total temperature at LPC outlet", "°R"),
    "s_3": ("T30", "Total temperature at HPC outlet", "°R"),
    "s_4": ("T50", "Total temperature at LPT outlet", "°R"),
    "s_7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s_8": ("Nf", "Physical fan speed", "rpm"),
    "s_9": ("Nc", "Physical core speed", "rpm"),
    "s_11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s_12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s_13": ("NRf", "Corrected fan speed", "rpm"),
    "s_14": ("NRc", "Corrected core speed", "rpm"),
    "s_15": ("BPR", "Bypass ratio", ""),
    "s_17": ("htBleed", "Bleed enthalpy", ""),
    "s_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s_21": ("W32", "LPT coolant bleed", "lbm/s"),
}

DEFAULT_THRESHOLDS = {"critical": 20.0, "high": 40.0, "medium": 70.0}
MAX_UPLOADS = 20
OOD_MARGIN = 0.05          # scaled value outside [-margin, 1+margin] = outside the training range


def risk_level(lower_bound: float, th: Dict[str, float]) -> str:
    """Risk from the CONSERVATIVE end of the interval (lower RUL bound)."""
    if lower_bound < th["critical"]:
        return "Critical"
    if lower_bound < th["high"]:
        return "High"
    if lower_bound < th["medium"]:
        return "Medium"
    return "Low"


EngineRef = Union[int, str]


@dataclass
class EngineSeries:
    ref: EngineRef
    cycles: np.ndarray                   # cycles at which a prediction exists (>= window)
    X: np.ndarray                        # (n, W, F) scaled windows ending at each cycle
    raw: pd.DataFrame                    # raw feature values for every observed cycle
    true_rul: Optional[np.ndarray]       # capped ground truth (None for uploads without labels)
    true_rul_uncapped: Optional[np.ndarray]
    source: str = "FD001 test set"
    ood_fraction: Optional[np.ndarray] = None   # per predicted cycle: share of inputs outside the training range


def integrated_gradients(models, X: np.ndarray, baseline: np.ndarray, steps: int, device: torch.device,
                         seq_budget: Optional[int] = None) -> np.ndarray:
    """Integrated gradients (Sundararajan et al., 2017) averaged over the ensemble.

    X: (n, W, F); baseline: (1, W, F). Returns attributions (n, W, F) in RUL cycles.
    Windows are processed in batches of ``seq_budget // steps`` (all interpolation
    steps of a window in one batch). On CUDA, cuDNN is disabled for this pass because
    cuDNN's RNN backward requires training mode, which would activate dropout.
    """
    base = torch.from_numpy(baseline.astype(np.float32)).to(device)
    alphas = torch.linspace(0, 1, steps + 1, device=device)[1:].view(-1, 1, 1, 1)
    out = np.zeros_like(X, dtype=np.float32)
    # sequences per batch: the non-cuDNN LSTM keeps all activations for backward (~3 MB per sequence on GPU)
    seq_budget = seq_budget or (2048 if device.type == "cuda" else 8192)
    b = max(1, seq_budget // steps)
    with torch.backends.cudnn.flags(enabled=device.type != "cuda" and torch.backends.cudnn.enabled):
        for s in range(0, len(X), b):
            x = torch.from_numpy(np.ascontiguousarray(X[s:s + b], dtype=np.float32)).to(device)
            n = x.shape[0]
            path = (base.unsqueeze(0) + alphas * (x.unsqueeze(0) - base.unsqueeze(0))).reshape(steps * n, *x.shape[1:])
            path.requires_grad_(True)
            acc = torch.zeros_like(x)
            for m in models:
                g, = torch.autograd.grad(m(path).sum(), path)
                acc += g.reshape(steps, n, *x.shape[1:]).mean(0)
            out[s:s + n] = ((x - base) * acc / len(models)).detach().cpu().numpy()
    return out


class ModelStore:
    def __init__(self, coverage: float = 0.90, device: str = "cpu", use_cache: bool = True):
        proto = C.load_protocol()
        self.proto = proto
        self.data_cfg = proto["base"]["data"]
        self.model_cfg = proto["base"]["model"]
        self.seeds: List[int] = proto["seeds"]["primary"]
        self.coverage = coverage
        self.device = torch.device(device)
        self.use_cache = use_cache
        self.cap = float(self.data_cfg["rul_cap"])
        self.W = int(self.data_cfg["window_size"])
        self.cols = feature_columns(self.data_cfg["drop_sensors"])
        self._prepare_data()
        self._families: Dict[str, Dict] = {}
        self.uploads: "OrderedDict[str, EngineSeries]" = OrderedDict()

    # ------------------------------------------------------------------ data
    def _prepare_data(self) -> None:
        dc, ddir = self.data_cfg, C.DATA_DIR
        ref = get_data(dc, ddir)                     # validated research pipeline
        train_raw = add_train_rul(load_raw(dc["dataset_id"], "train", ddir), dc["rul_cap"])
        tr_units, _ = split_engines(sorted(train_raw["unit"].unique()), dc["val_fraction"], dc["val_split_seed"])
        self.scaler = MinMaxScaler((0, 1)).fit(train_raw[train_raw.unit.isin(tr_units)][self.cols].to_numpy())
        self.constant_features = [c for c, lo, hi in zip(self.cols, self.scaler.data_min_, self.scaler.data_max_) if hi == lo]
        # Typical healthy range per feature: 1st-99th percentile of training cycles at the RUL cap.
        healthy = train_raw[train_raw.unit.isin(tr_units) & (train_raw.rul >= dc["rul_cap"])]
        self.healthy_band = {c: (float(healthy[c].quantile(0.01)), float(healthy[c].quantile(0.99))) for c in self.cols}

        test_raw = load_raw(dc["dataset_id"], "test", ddir)
        rul_end = pd.read_csv(os.path.join(ddir, f"RUL_{dc['dataset_id']}.txt"), header=None, sep=r"\s+").iloc[:, 0].to_numpy()
        self.engines: Dict[int, EngineSeries] = {}
        for i, unit in enumerate(sorted(test_raw.unit.unique())):
            s = self._series(int(unit), test_raw[test_raw.unit == unit], rul_end=float(rul_end[i]))
            if s is not None:
                self.engines[int(unit)] = s
        for k, unit in enumerate(ref.test.unit):
            if not np.allclose(self.engines[int(unit)].X[-1], ref.test.X[k], atol=1e-6):
                raise AssertionError(f"Serving preprocessing diverges from research pipeline for engine {unit}")
        self.val_X, self.val_y = ref.val.X, ref.val.y.astype(float)
        self.train_mean_window = ref.train.X.mean(axis=0, keepdims=True).astype(np.float32)   # IG baseline
        self.test_units = sorted(self.engines)
        self.offsets = np.cumsum([0] + [len(self.engines[u].cycles) for u in self.test_units])

    def _series(self, ref: EngineRef, g: pd.DataFrame, rul_end: Optional[float] = None, source: str = "FD001 test set") -> Optional[EngineSeries]:
        g = g.sort_values("cycle")
        F = self.scaler.transform(g[self.cols].to_numpy(dtype=float)).astype(np.float32)
        T = len(g)
        if T < self.W:
            return None
        ends = np.arange(self.W, T + 1)
        X = np.stack([F[e - self.W:e] for e in ends])
        var = [j for j, c in enumerate(self.cols) if c not in self.constant_features]
        out_mask = (F[:, var] < -OOD_MARGIN) | (F[:, var] > 1 + OOD_MARGIN)
        const = [j for j, c in enumerate(self.cols) if c in self.constant_features]
        if const:   # a feature that was constant in training must keep that value
            out_mask = np.concatenate([out_mask, np.abs(F[:, const]) > 1e-6], axis=1)
        ood = out_mask.mean(axis=1)[ends - 1]
        true = unc = None
        if rul_end is not None:
            unc = (rul_end + (T - ends)).astype(float)
            true = np.minimum(unc, self.cap)
        return EngineSeries(ref, ends.astype(int), X, g[["cycle"] + self.cols].reset_index(drop=True), true, unc, source, ood)

    def series(self, ref: EngineRef) -> EngineSeries:
        if isinstance(ref, str):
            if ref not in self.uploads:
                raise KeyError(f"Upload '{ref}' not found (uploads are kept in memory, at most {MAX_UPLOADS})")
            return self.uploads[ref]
        if ref not in self.engines:
            raise KeyError(f"Engine {ref} not found (valid: 1-{len(self.engines)})")
        return self.engines[ref]

    # ----------------------------------------------------------------- models
    def _run_ids(self, family: str) -> List[str]:
        point = C.expand_experiment(self.proto, family)[0]
        return [C.run_id(point["config"]["method"], point["config_hash"], s) for s in self.seeds]

    def _ckpt_paths(self, family: str) -> List[str]:
        return [os.path.join(CKPT_ROOT, rid, "selected.pt") for rid in self._run_ids(family)]

    def _cache_key(self, family: str) -> str:
        h = hashlib.sha256()
        h.update(CACHE_VERSION.encode())
        h.update(json.dumps(self.data_cfg, sort_keys=True).encode())
        for p in self._ckpt_paths(family):
            with open(p, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        return h.hexdigest()

    def _cache_file(self, family: str, kind: str) -> str:
        return os.path.join(CACHE_DIR, f"{family}-{self._cache_key(family)[:16]}-{kind}.npz")

    def _load_models(self, family: str, device: Optional[torch.device] = None):
        models = []
        for p in self._ckpt_paths(family):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing checkpoint {p}. Run the v2 experiments first.")
            m = build_model(input_size=len(self.cols), hidden_size=self.model_cfg["hidden_size"],
                            num_layers=self.model_cfg["num_layers"], dropout=self.model_cfg["dropout"],
                            bidirectional=self.model_cfg["bidirectional"])
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"])
            models.append(m.to(device or self.device).eval())
        return models

    def _load_family(self, family: str) -> Dict:
        if family not in FAMILIES:
            raise KeyError(f"Unknown model family '{family}'. Available: {list(FAMILIES)}")
        if family in self._families:
            return self._families[family]
        t0 = time.perf_counter()
        models = self._load_models(family)
        fam = {"name": family, "description": FAMILIES[family], "run_ids": self._run_ids(family), "models": models,
               "cache_key": self._cache_key(family)[:16]}
        pred_file = self._cache_file(family, "pred")
        if self.use_cache and os.path.exists(pred_file):
            z = np.load(pred_file)
            members, val_members = z["members"], z["val_members"]
            fam["cache"] = "hit"
        else:
            members = self._predict_members(models, np.concatenate([self.engines[u].X for u in self.test_units]))
            val_members = self._predict_members(models, self.val_X)
            if self.use_cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                np.savez_compressed(pred_file, members=members, val_members=val_members)
            fam["cache"] = "miss (saved)" if self.use_cache else "disabled"
        fam["pred"] = {}
        for i, u in enumerate(self.test_units):
            mem = members[:, self.offsets[i]:self.offsets[i + 1]]
            fam["pred"][u] = {"mean": mem.mean(0), "std": mem.std(0, ddof=1), "members": mem}

        # ---- conformal calibration on the validation engines
        vm, vs = val_members.mean(0), val_members.std(0, ddof=1)
        alpha = 1 - self.coverage
        fam["cal_global"] = GlobalConformal(alpha).fit(self.val_y, vm, vs)
        fam["cal"] = MondrianConformal(alpha).fit(self.val_y, vm, vs)          # served
        fam["val_rmse"] = M.rmse(self.val_y, vm)

        # ---- test-set evaluation (reporting only)
        mean_all, std_all = members.mean(0), members.std(0, ddof=1)
        true_all = np.concatenate([self.engines[u].true_rul for u in self.test_units])
        last = self.offsets[1:] - 1
        member_last = members[:, last]
        fam["test"] = {
            "ensemble_last_cycle": M.error_summary(true_all[last], mean_all[last]),
            "members_last_cycle_rmse_mean": float(np.mean([M.rmse(true_all[last], member_last[j]) for j in range(len(models))])),
            "members_last_cycle_rmse_std": float(np.std([M.rmse(true_all[last], member_last[j]) for j in range(len(models))], ddof=1)),
            "rmse_all_cycles": M.rmse(true_all, mean_all),
            "n_engines": len(self.test_units), "n_cycle_predictions": int(true_all.size),
        }
        ev = {}
        for name, cal in (("mondrian", fam["cal"]), ("global", fam["cal_global"])):
            ev[name] = {"last_cycle": evaluate(cal, true_all[last], mean_all[last], std_all[last]),
                        "all_cycles": evaluate(cal, true_all, mean_all, std_all),
                        "validation": evaluate(cal, self.val_y, vm, vs)}
        fam["interval_eval"] = ev
        t = fam["test"]
        t["interval_coverage_last_cycle"] = ev["mondrian"]["last_cycle"]["coverage"]
        t["interval_coverage_all_cycles"] = ev["mondrian"]["all_cycles"]["coverage"]
        t["mean_interval_width_last_cycle"] = ev["mondrian"]["last_cycle"]["mean_width"]
        fam["val_coverage"] = ev["mondrian"]["validation"]["coverage"]

        # ---- precomputed attributions (optional)
        ig_file = self._cache_file(family, f"ig{IG_STEPS}")
        if self.use_cache and os.path.exists(ig_file):
            z = np.load(ig_file)
            fam["ig"] = {"attr": z["attr"], "f_base": float(z["f_base"]), "steps": int(z["steps"])}
        fam["latency_ms"] = self._latency(models)
        fam["load_s"] = time.perf_counter() - t0
        self._families[family] = fam
        return fam

    def precompute_attributions(self, family: str, device: torch.device, steps: int = IG_STEPS) -> str:
        """Integrated gradients for every cycle of every test engine (slow on CPU; fast on GPU)."""
        models = self._load_models(family, device)
        X = np.concatenate([self.engines[u].X for u in self.test_units])
        attr = integrated_gradients(models, X, self.train_mean_window, steps, device)
        with torch.no_grad():
            b = torch.from_numpy(self.train_mean_window).to(device)
            f_base = float(np.mean([float(m(b)) for m in models]))
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = self._cache_file(family, f"ig{steps}")
        np.savez_compressed(path, attr=attr.astype(np.float32), f_base=f_base, steps=steps)
        self._families.pop(family, None)       # reload with attributions next time
        return path

    @torch.no_grad()
    def _predict_members(self, models, X: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)
        return np.stack([np.concatenate([m(xt[s:s + 4096]).cpu().numpy() for s in range(0, len(xt), 4096)])
                         for m in models]).astype(float)

    def _latency(self, models) -> Dict[str, float]:
        """Measured wall-clock time of one ensemble prediction for one 30-cycle window."""
        x = torch.from_numpy(self.engines[self.test_units[0]].X[-1:]).to(self.device)
        with torch.no_grad():
            for m in models:
                m(x)
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                for m in models:
                    m(x)
                ts.append((time.perf_counter() - t0) * 1000)
        return {"ensemble_window_ms_median": float(np.median(ts)), "ensemble_window_ms_p95": float(np.percentile(ts, 95)),
                "device": str(self.device), "torch_threads": torch.get_num_threads()}

    # --------------------------------------------------------------- queries
    def interval(self, fam: Dict, mu: np.ndarray, sd: np.ndarray):
        half = fam["cal"].scale(mu) * np.maximum(sd, 1e-6)
        return np.clip(mu - half, 0, None), mu + half

    def _predict_series(self, fam: Dict, s: EngineSeries) -> Dict:
        if isinstance(s.ref, int) and s.ref in fam["pred"]:
            return fam["pred"][s.ref]
        mem = self._predict_members(fam["models"], s.X)
        return {"mean": mem.mean(0), "std": mem.std(0, ddof=1), "members": mem}

    def fleet(self, family: str, th: Dict[str, float]) -> List[Dict]:
        fam = self._load_family(family)
        rows = []
        for u in self.test_units:
            e, p = self.engines[u], fam["pred"][u]
            lo_all, hi_all = self.interval(fam, p["mean"], p["std"])
            since = {}
            for name in ("critical", "high"):
                below = lo_all < th[name]
                if below[-1]:
                    above = np.flatnonzero(~below)
                    since[name] = int(e.cycles[above[-1] + 1]) if len(above) else int(e.cycles[0])
                else:
                    since[name] = None
            mu = float(p["mean"][-1])
            rows.append({"engine_id": u, "last_cycle": int(e.cycles[-1]), "rul_mean": mu, "rul_std": float(p["std"][-1]),
                         "critical_since_cycle": since["critical"], "high_since_cycle": since["high"],
                         "rul_lower": float(lo_all[-1]), "rul_upper": float(hi_all[-1]),
                         "life_remaining_pct": float(np.clip(mu / self.cap, 0, 1) * 100),
                         "risk": risk_level(float(lo_all[-1]), th),
                         "true_rul": float(e.true_rul[-1]), "true_rul_uncapped": float(e.true_rul_uncapped[-1])})
        return rows

    def _sensors(self, e: EngineSeries) -> List[Dict]:
        out = []
        for c in self.cols:
            short, desc, unit_ = FEATURE_INFO.get(c, (c, c, ""))
            band = self.healthy_band[c]
            out.append({"key": c, "name": short, "description": desc, "unit": unit_,
                        "healthy_low": band[0], "healthy_high": band[1], "values": [float(v) for v in e.raw[c]]})
        return out

    def engine(self, ref: EngineRef, family: str, th: Dict[str, float]) -> Dict:
        fam = self._load_family(family)
        e = self.series(ref)
        p = self._predict_series(fam, e)
        lo, hi = self.interval(fam, p["mean"], p["std"])
        traj = []
        for i, c in enumerate(e.cycles):
            row = {"cycle": int(c), "rul_mean": float(p["mean"][i]), "rul_lower": float(lo[i]), "rul_upper": float(hi[i]),
                   "true_rul": float(e.true_rul[i]) if e.true_rul is not None else None, "risk": risk_level(float(lo[i]), th)}
            if e.ood_fraction is not None:
                row["ood_fraction"] = float(e.ood_fraction[i])
            traj.append(row)
        return {"engine_id": ref, "family": family, "source": e.source, "has_ground_truth": e.true_rul is not None,
                "observed_cycles": [int(c) for c in e.raw["cycle"]], "window": self.W, "trajectory": traj,
                "sensors": self._sensors(e),
                "attributions_available": isinstance(ref, int) and "ig" in fam}

    def compare(self, ref: EngineRef, th: Dict[str, float]) -> Dict:
        e = self.series(ref)
        out = {"engine_id": ref, "cycles": [int(c) for c in e.cycles],
               "true_rul": [float(v) for v in e.true_rul] if e.true_rul is not None else None, "families": {}}
        for f in FAMILIES:
            fam = self._load_family(f)
            p = self._predict_series(fam, e)
            lo, hi = self.interval(fam, p["mean"], p["std"])
            entry = {"description": FAMILIES[f], "rul_mean": p["mean"].round(3).tolist(), "rul_lower": lo.round(3).tolist(),
                     "rul_upper": hi.round(3).tolist(), "risk_last": risk_level(float(lo[-1]), th),
                     "fleet_test_rmse_last_cycle": fam["test"]["ensemble_last_cycle"]["rmse"]}
            if e.true_rul is not None:
                entry["rmse_this_engine_all_cycles"] = M.rmse(e.true_rul, p["mean"])
                entry["error_last_cycle"] = float(p["mean"][-1] - e.true_rul[-1])
            out["families"][f] = entry
        return out

    def _ig_payload(self, e: EngineSeries, idx: int, attr: np.ndarray, pred: float, f_base: float, family: str, steps: int, source: str) -> Dict:
        per_feature = attr.sum(axis=0)
        feats = []
        for j, c in enumerate(self.cols):
            short, desc, _ = FEATURE_INFO.get(c, (c, c, ""))
            feats.append({"key": c, "name": short, "description": desc, "attribution": float(per_feature[j])})
        feats.sort(key=lambda d: -abs(d["attribution"]))
        start = int(e.cycles[idx]) - self.W + 1
        return {"engine_id": e.ref, "cycle": int(e.cycles[idx]), "family": family, "method": "integrated_gradients",
                "baseline": "mean training window", "steps": steps, "source": source,
                "prediction": pred, "baseline_prediction": f_base, "sum_of_attributions": float(attr.sum()),
                "completeness_error": float(attr.sum() - (pred - f_base)), "features": feats,
                "heatmap": {"cycles": list(range(start, start + self.W)),
                            "features": [FEATURE_INFO.get(c, (c,))[0] for c in self.cols], "values": attr.T.round(4).tolist()}}

    def explain(self, ref: EngineRef, cycle: int, family: str, steps: int = 256) -> Dict:
        fam = self._load_family(family)
        e = self.series(ref)
        idx = int(np.searchsorted(e.cycles, cycle))
        if idx >= len(e.cycles) or e.cycles[idx] != cycle:
            raise ValueError(f"cycle must be between {int(e.cycles[0])} and {int(e.cycles[-1])}")
        p = self._predict_series(fam, e)
        if isinstance(ref, int) and "ig" in fam and steps == fam["ig"]["steps"]:
            i0 = self.offsets[self.test_units.index(ref)]
            return self._ig_payload(e, idx, fam["ig"]["attr"][i0 + idx], float(p["mean"][idx]), fam["ig"]["f_base"], family,
                                    fam["ig"]["steps"], "precomputed cache")
        attr = integrated_gradients(fam["models"], e.X[idx:idx + 1], self.train_mean_window, steps, self.device)[0]
        with torch.no_grad():
            f_base = float(np.mean([float(m(torch.from_numpy(self.train_mean_window).to(self.device))) for m in fam["models"]]))
        return self._ig_payload(e, idx, attr, float(p["mean"][idx]), f_base, family, steps, "computed on request")

    def attribution_series(self, ref: int, family: str) -> Dict:
        """Per-cycle, per-feature attribution (summed over the window) for live replay colouring."""
        fam = self._load_family(family)
        if "ig" not in fam:
            raise LookupError("Attributions for every cycle are not precomputed. Run: python -m serving.precompute")
        e = self.series(ref)
        i = self.test_units.index(ref)
        a = fam["ig"]["attr"][self.offsets[i]:self.offsets[i + 1]].sum(axis=1)       # (n_cycles, F)
        return {"engine_id": ref, "family": family, "steps": fam["ig"]["steps"], "cycles": [int(c) for c in e.cycles],
                "features": self.cols, "values": a.round(3).tolist()}

    # --------------------------------------------------------------- uploads
    def add_upload(self, text: str, name: str, rul_at_end: Optional[float] = None) -> List[Dict]:
        """Parse a C-MAPSS-format file (whitespace or comma separated, 26 columns, optional header).

        One engine per distinct unit id. Nothing is written to disk; uploads live in
        memory (the oldest are dropped beyond MAX_UPLOADS).
        """
        rows = [r for r in text.replace(",", " ").splitlines() if r.strip()]
        if not rows:
            raise ValueError("The file is empty.")
        first = rows[0].split()[0]
        try:
            float(first)
        except ValueError:
            rows = rows[1:]                                          # header line
        try:
            data = [[float(v) for v in r.split()] for r in rows]
        except ValueError as ex:
            raise ValueError(f"Non-numeric value in the file: {ex}") from None
        widths = {len(r) for r in data}
        if widths != {len(COLUMN_NAMES)}:
            raise ValueError(f"Expected {len(COLUMN_NAMES)} columns per row (unit, cycle, 3 settings, 21 sensors) "
                             f"as in NASA's train/test_FD00x.txt files; found rows with {sorted(widths)} columns.")
        arr = np.array(data, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("The file contains missing or non-finite values.")
        df = pd.DataFrame(arr, columns=COLUMN_NAMES)
        df["unit"], df["cycle"] = df["unit"].astype(int), df["cycle"].astype(int)
        units = sorted(df.unit.unique())
        if len(units) > 10:
            raise ValueError(f"At most 10 engines per upload ({len(units)} found).")
        if rul_at_end is not None and len(units) != 1:
            raise ValueError("A known final RUL can only be given for a single-engine upload.")
        series = []
        for u in units:
            g = df[df.unit == u]
            if g["cycle"].duplicated().any():
                raise ValueError(f"Engine {u}: duplicate cycle numbers.")
            s = self._series("pending", g, rul_end=rul_at_end, source=f"upload: {name} (unit {u})")
            if s is None:
                raise ValueError(f"Engine {u}: needs at least {self.W} cycles, has {len(g)}.")
            series.append((u, len(g), s))
        created = []
        for u, n, s in series:                     # register only after the whole file validated
            uid = f"u-{uuid.uuid4().hex[:8]}"
            s.ref = uid
            self.uploads[uid] = s
            while len(self.uploads) > MAX_UPLOADS:
                self.uploads.popitem(last=False)
            created.append({"upload_id": uid, "unit": int(u), "cycles": int(n), "predicted_cycles": int(len(s.cycles)),
                            "max_ood_fraction": float(s.ood_fraction.max()), "source": s.source,
                            "has_ground_truth": s.true_rul is not None})
        return created

    # --------------------------------------------------------------- info
    def family_info(self, family: str) -> Dict:
        fam = self._load_family(family)
        return {"name": fam["name"], "description": fam["description"], "run_ids": fam["run_ids"],
                "cache_key": fam["cache_key"], "cache": fam["cache"], "load_s": fam["load_s"],
                "attributions_precomputed": "ig" in fam,
                "interval": {"served": fam["cal"].describe(), "baseline_global": fam["cal_global"].describe(),
                             "evaluation": fam["interval_eval"]},
                "k": fam["cal_global"].k, "val_coverage": fam["val_coverage"], "val_rmse": fam["val_rmse"],
                "test": fam["test"], "latency_ms": fam["latency_ms"]}

    def checkpoint_digests(self, family: str) -> List[Dict]:
        out = []
        for rid, p in zip(self._run_ids(family), self._ckpt_paths(family)):
            with open(p, "rb") as f:
                out.append({"run_id": rid, "sha256": hashlib.sha256(f.read()).hexdigest()[:16]})
        return out
