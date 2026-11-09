"""
serving/model_store.py
----------------------
Loads the v2 checkpoints as 5-seed ensembles, replays the 100 official FD001 test
engines cycle by cycle, calibrates prediction intervals and computes
integrated-gradients attributions.

Data protocol (identical to research/data.py, asserted at start-up):
* MinMax scaler fitted on the 90 training engines only;
* a prediction at cycle c uses the 30 cycles ending at c (so c >= 30);
* ground truth for a test engine at cycle c is RUL_file + (T_engine - c), shown
  capped at 125 like the training target. The test labels are used only for
  display and evaluation, never for fitting anything here.

Prediction interval
* ensemble = the validation-selected checkpoint of each of the 5 seeds;
* interval = mean ± k · std across members, where k is chosen on the held-out
  VALIDATION engines so that ``coverage`` (default 90%) of validation windows fall
  inside. Empirical coverage on the test engines is reported, not assumed.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from models.lstm_model import build_model
from research import config as C
from research import metrics as M
from research.data import add_train_rul, feature_columns, get_data, load_raw, split_engines

CKPT_ROOT = os.path.join(C.BASE_DIR, "checkpoints", "v2")

# Families of models that can be served. Each is a protocol experiment whose
# validation-selected checkpoint exists for every primary seed.
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


def risk_level(lower_bound: float, th: Dict[str, float]) -> str:
    """Risk from the CONSERVATIVE end of the interval (lower RUL bound)."""
    if lower_bound < th["critical"]:
        return "Critical"
    if lower_bound < th["high"]:
        return "High"
    if lower_bound < th["medium"]:
        return "Medium"
    return "Low"


@dataclass
class EngineSeries:
    unit: int
    cycles: np.ndarray        # cycles at which a prediction exists (>= window)
    X: np.ndarray             # (n, W, F) scaled windows ending at each cycle
    raw: pd.DataFrame         # raw feature values for every observed cycle
    true_rul: np.ndarray      # capped ground truth at each predicted cycle
    true_rul_uncapped: np.ndarray


class ModelStore:
    def __init__(self, coverage: float = 0.90, device: str = "cpu"):
        proto = C.load_protocol()
        self.proto = proto
        self.data_cfg = proto["base"]["data"]
        self.model_cfg = proto["base"]["model"]
        self.seeds: List[int] = proto["seeds"]["primary"]
        self.coverage = coverage
        self.device = torch.device(device)
        self.cap = float(self.data_cfg["rul_cap"])
        self.W = int(self.data_cfg["window_size"])
        self.cols = feature_columns(self.data_cfg["drop_sensors"])
        self._prepare_data()
        self._families: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ data
    def _prepare_data(self) -> None:
        dc, ddir = self.data_cfg, C.DATA_DIR
        ref = get_data(dc, ddir)                     # validated research pipeline
        train_raw = add_train_rul(load_raw(dc["dataset_id"], "train", ddir), dc["rul_cap"])
        tr_units, va_units = split_engines(sorted(train_raw["unit"].unique()), dc["val_fraction"], dc["val_split_seed"])
        self.scaler = MinMaxScaler((0, 1)).fit(train_raw[train_raw.unit.isin(tr_units)][self.cols].to_numpy())
        # Typical healthy range per feature: 1st-99th percentile of training cycles
        # whose (capped) RUL is at the cap, i.e. early, healthy operation. By
        # construction ~2% of healthy readings fall outside it.
        healthy = train_raw[train_raw.unit.isin(tr_units) & (train_raw.rul >= dc["rul_cap"])]
        self.healthy_band = {c: (float(healthy[c].quantile(0.01)), float(healthy[c].quantile(0.99))) for c in self.cols}

        test_raw = load_raw(dc["dataset_id"], "test", ddir)
        rul_end = pd.read_csv(os.path.join(ddir, f"RUL_{dc['dataset_id']}.txt"), header=None, sep=r"\s+").iloc[:, 0].to_numpy()
        self.engines: Dict[int, EngineSeries] = {}
        for i, unit in enumerate(sorted(test_raw.unit.unique())):
            g = test_raw[test_raw.unit == unit].sort_values("cycle")
            F = self.scaler.transform(g[self.cols].to_numpy()).astype(np.float32)
            T = len(g)
            if T < self.W:
                continue   # not the case for FD001 (shortest test engine: 31 cycles)
            ends = np.arange(self.W, T + 1)
            X = np.stack([F[e - self.W:e] for e in ends])
            unc = rul_end[i] + (T - ends)
            self.engines[int(unit)] = EngineSeries(int(unit), ends.astype(int), X, g[["cycle"] + self.cols].reset_index(drop=True),
                                                   np.minimum(unc, self.cap).astype(float), unc.astype(float))
        # The last window of every engine must equal the research pipeline's test set.
        for k, unit in enumerate(ref.test.unit):
            if not np.allclose(self.engines[int(unit)].X[-1], ref.test.X[k], atol=1e-6):
                raise AssertionError(f"Serving preprocessing diverges from research pipeline for engine {unit}")
        self.val_X, self.val_y = ref.val.X, ref.val.y.astype(float)
        self.train_mean_window = ref.train.X.mean(axis=0, keepdims=True).astype(np.float32)   # IG baseline

    # ----------------------------------------------------------------- models
    def _run_ids(self, family: str) -> List[str]:
        point = C.expand_experiment(self.proto, family)[0]
        return [C.run_id(point["config"]["method"], point["config_hash"], s) for s in self.seeds]

    def _load_family(self, family: str) -> Dict:
        if family not in FAMILIES:
            raise KeyError(f"Unknown model family '{family}'. Available: {list(FAMILIES)}")
        if family in self._families:
            return self._families[family]
        models, rids = [], self._run_ids(family)
        for rid in rids:
            path = os.path.join(CKPT_ROOT, rid, "selected.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing checkpoint {path}. Run the v2 experiments first.")
            m = build_model(input_size=len(self.cols), hidden_size=self.model_cfg["hidden_size"],
                            num_layers=self.model_cfg["num_layers"], dropout=self.model_cfg["dropout"],
                            bidirectional=self.model_cfg["bidirectional"])
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
            models.append(m.to(self.device).eval())
        fam = {"name": family, "description": FAMILIES[family], "run_ids": rids, "models": models}
        # Calibrate k on validation windows.
        vm, vs = self._ensemble(models, self.val_X)
        z = np.abs(self.val_y - vm) / np.maximum(vs, 1e-6)
        fam["k"] = float(np.quantile(z, self.coverage))
        fam["val_coverage"] = float(np.mean(np.abs(self.val_y - vm) <= fam["k"] * vs))
        fam["val_rmse"] = M.rmse(self.val_y, vm)
        # Predictions for every cycle of every test engine.
        fam["pred"] = {}
        for u, e in self.engines.items():
            mu, sd, members = self._ensemble(models, e.X, return_members=True)
            fam["pred"][u] = {"mean": mu, "std": sd, "members": members}
        last_mean = np.array([fam["pred"][u]["mean"][-1] for u in self.engines])
        last_std = np.array([fam["pred"][u]["std"][-1] for u in self.engines])
        last_true = np.array([self.engines[u].true_rul[-1] for u in self.engines])
        all_mean = np.concatenate([fam["pred"][u]["mean"] for u in self.engines])
        all_std = np.concatenate([fam["pred"][u]["std"] for u in self.engines])
        all_true = np.concatenate([self.engines[u].true_rul for u in self.engines])
        member_last = np.stack([fam["pred"][u]["members"][:, -1] for u in self.engines], axis=1)  # (5, 100)
        fam["test"] = {
            "ensemble_last_cycle": M.error_summary(last_true, last_mean),
            "members_last_cycle_rmse_mean": float(np.mean([M.rmse(last_true, member_last[j]) for j in range(len(models))])),
            "members_last_cycle_rmse_std": float(np.std([M.rmse(last_true, member_last[j]) for j in range(len(models))], ddof=1)),
            "interval_coverage_last_cycle": float(np.mean(np.abs(last_true - last_mean) <= fam["k"] * last_std)),
            "interval_coverage_all_cycles": float(np.mean(np.abs(all_true - all_mean) <= fam["k"] * all_std)),
            "mean_interval_width_last_cycle": float(np.mean(2 * fam["k"] * last_std)),
            "rmse_all_cycles": M.rmse(all_true, all_mean),
            "n_engines": len(self.engines), "n_cycle_predictions": int(all_true.size),
        }
        fam["latency_ms"] = self._latency(models)
        self._families[family] = fam
        return fam

    @torch.no_grad()
    def _ensemble(self, models, X: np.ndarray, return_members: bool = False):
        xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(self.device)
        preds = np.stack([np.concatenate([m(xt[s:s + 4096]).cpu().numpy() for s in range(0, len(xt), 4096)])
                          for m in models]).astype(float)
        out = (preds.mean(0), preds.std(0, ddof=1))
        return out + (preds,) if return_members else out

    def _latency(self, models) -> Dict[str, float]:
        """Measured wall-clock time of one ensemble prediction for one 30-cycle window."""
        x = torch.from_numpy(self.engines[next(iter(self.engines))].X[-1:]).to(self.device)
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
        lo = np.clip(mu - fam["k"] * sd, 0, None)
        hi = mu + fam["k"] * sd
        return lo, hi

    def fleet(self, family: str, th: Dict[str, float], at: Optional[str] = None) -> List[Dict]:
        fam = self._load_family(family)
        rows = []
        for u, e in self.engines.items():
            p = fam["pred"][u]
            mu, sd = float(p["mean"][-1]), float(p["std"][-1])
            lo, hi = self.interval(fam, np.array([mu]), np.array([sd]))
            lo_all, _ = self.interval(fam, p["mean"], p["std"])
            # First cycle from which the lower bound stays below each threshold (None if it never does).
            since = {}
            for name in ("critical", "high"):
                below = lo_all < th[name]
                if below[-1]:
                    above = np.flatnonzero(~below)
                    since[name] = int(e.cycles[above[-1] + 1]) if len(above) else int(e.cycles[0])
                else:
                    since[name] = None
            rows.append({"engine_id": u, "last_cycle": int(e.cycles[-1]), "rul_mean": mu, "rul_std": sd,
                         "critical_since_cycle": since["critical"], "high_since_cycle": since["high"],
                         "rul_lower": float(lo[0]), "rul_upper": float(hi[0]),
                         "life_remaining_pct": float(np.clip(mu / self.cap, 0, 1) * 100),
                         "risk": risk_level(float(lo[0]), th),
                         "true_rul": float(e.true_rul[-1]), "true_rul_uncapped": float(e.true_rul_uncapped[-1])})
        return rows

    def engine(self, unit: int, family: str, th: Dict[str, float]) -> Dict:
        fam = self._load_family(family)
        e = self.engines[unit]
        p = fam["pred"][unit]
        lo, hi = self.interval(fam, p["mean"], p["std"])
        trajectory = [{"cycle": int(c), "rul_mean": float(m), "rul_lower": float(l), "rul_upper": float(h),
                       "true_rul": float(t), "risk": risk_level(float(l), th)}
                      for c, m, l, h, t in zip(e.cycles, p["mean"], lo, hi, e.true_rul)]
        sensors = []
        for c in self.cols:
            short, desc, unit_ = FEATURE_INFO.get(c, (c, c, ""))
            band = self.healthy_band[c]
            sensors.append({"key": c, "name": short, "description": desc, "unit": unit_,
                            "healthy_low": band[0], "healthy_high": band[1],
                            "values": [float(v) for v in e.raw[c]]})
        return {"engine_id": unit, "family": family, "observed_cycles": [int(c) for c in e.raw["cycle"]],
                "window": self.W, "trajectory": trajectory, "sensors": sensors}

    def explain(self, unit: int, cycle: int, family: str, steps: int = 256) -> Dict:
        """Integrated gradients (Sundararajan et al., 2017), averaged over the ensemble.

        Baseline: the mean training window. Attributions are in RUL cycles; their sum
        equals prediction(x) - prediction(baseline) up to the reported completeness error.
        """
        fam = self._load_family(family)
        e = self.engines[unit]
        idx = int(np.searchsorted(e.cycles, cycle))
        if idx >= len(e.cycles) or e.cycles[idx] != cycle:
            raise ValueError(f"cycle must be between {int(e.cycles[0])} and {int(e.cycles[-1])}")
        x = torch.from_numpy(e.X[idx:idx + 1])
        base = torch.from_numpy(self.train_mean_window)
        alphas = torch.linspace(0, 1, steps + 1)[1:].view(-1, 1, 1)
        path = (base + alphas * (x - base)).requires_grad_(True)
        attr = torch.zeros_like(x)
        f_x, f_b = [], []
        for m in fam["models"]:
            m_cpu = m.cpu().train(False)
            out = m_cpu(path).sum()
            g, = torch.autograd.grad(out, path)
            attr += (x - base) * g.mean(0, keepdim=True)
            with torch.no_grad():
                f_x.append(float(m_cpu(x)))
                f_b.append(float(m_cpu(base)))
            m.to(self.device)
        attr = (attr / len(fam["models"]))[0].detach().numpy()          # (W, F)
        delta = float(np.mean(f_x) - np.mean(f_b))
        per_feature = attr.sum(axis=0)
        feats = []
        for j, c in enumerate(self.cols):
            short, desc, unit_ = FEATURE_INFO.get(c, (c, c, ""))
            feats.append({"key": c, "name": short, "description": desc, "attribution": float(per_feature[j])})
        feats.sort(key=lambda d: -abs(d["attribution"]))
        start = int(e.cycles[idx]) - self.W + 1
        return {"engine_id": unit, "cycle": int(cycle), "family": family, "method": "integrated_gradients",
                "baseline": "mean training window", "steps": steps,
                "prediction": float(np.mean(f_x)), "baseline_prediction": float(np.mean(f_b)),
                "sum_of_attributions": float(attr.sum()),
                "completeness_error": float(attr.sum() - delta),
                "features": feats,
                "heatmap": {"cycles": list(range(start, start + self.W)),
                            "features": [FEATURE_INFO.get(c, (c,))[0] for c in self.cols],
                            "values": attr.T.round(4).tolist()}}

    def family_info(self, family: str) -> Dict:
        fam = self._load_family(family)
        return {k: fam[k] for k in ("name", "description", "run_ids", "k", "val_coverage", "val_rmse", "test", "latency_ms")}
