"""
research/config.py
------------------
Loads the JSON protocol file, expands experiments into concrete runs
(experiment x grid point x seed) and gives every run a content-addressed ID.

A run's ID is a hash of its *fully resolved* configuration plus its seed, so
two experiments that resolve to the same configuration (e.g. ``fedavg_iid`` and
the R=20 point of ``sweep_rounds``) share one run instead of training twice,
and changing any hyperparameter can never silently re-use stale results.
"""

import copy
import hashlib
import itertools
import json
import os
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROTOCOL = os.path.join(BASE_DIR, "configs", "fd001_protocol.json")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "results", "v2")
DEFAULT_CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v2")
DATA_DIR = os.path.join(BASE_DIR, "CMAPSSData")

METHODS = ("centralized", "fedavg", "local")
PARTITIONS = ("iid_window", "dirichlet_window", "engine")


def load_protocol(path: str = DEFAULT_PROTOCOL) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``upd`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: Dict[str, Any], key: str, value: Any) -> None:
    """Set ``cfg["a"]["b"] = value`` for ``key == "a.b"``."""
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d[p]
    if parts[-1] not in d:
        raise KeyError(f"Unknown config key in grid: {key}")
    d[parts[-1]] = value


def resolve_derived(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill derived values so that the stored config is fully explicit."""
    cfg = copy.deepcopy(cfg)
    fed, cen = cfg["fed"], cfg["centralized"]
    if cfg["method"] == "centralized":
        # Budget matching: centralized sees every training window R*E times,
        # exactly like FedAvg with full participation; selection granularity
        # (every E epochs) gives the same number of checkpoint candidates (R).
        if cen["early_stopping"]:
            cen["epochs"] = cen["max_epochs_es"]
        elif cen["epochs"] is None:
            cen["epochs"] = fed["num_rounds"] * fed["local_epochs"]
        if cen["select_every"] is None:
            cen["select_every"] = fed["local_epochs"]
        # Partition settings are irrelevant for the pooled baseline.
        cfg["fed"] = {k: fed[k] for k in ("num_rounds", "local_epochs")}
    else:
        cfg.pop("centralized", None)
        if fed["partition"] not in PARTITIONS:
            raise ValueError(f"Unknown partition {fed['partition']}")
        if fed["partition"] == "dirichlet_window" and fed["dirichlet_alpha"] is None:
            raise ValueError("dirichlet_window partition requires fed.dirichlet_alpha")
        if fed["partition"] != "dirichlet_window":
            fed["dirichlet_alpha"] = None
        if fed["fraction_fit"] != 1.0:
            raise NotImplementedError("Only full participation (fraction_fit=1.0) is implemented.")
    if cfg["method"] not in METHODS:
        raise ValueError(f"Unknown method {cfg['method']}")
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def grid_label(point: Dict[str, Any]) -> str:
    if not point:
        return "default"
    return ",".join(f"{k.split('.')[-1]}={v}" for k, v in point.items())


def expand_experiment(protocol: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    """Return one resolved config per grid point of experiment ``name``."""
    exp = protocol["experiments"][name]
    base = deep_update(protocol["base"], exp.get("overrides", {}))
    base["method"] = exp["method"]
    grid = exp.get("grid", {})
    keys = list(grid.keys())
    points = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])] or [{}]
    out = []
    for point in points:
        cfg = copy.deepcopy(base)
        for k, v in point.items():
            set_dotted(cfg, k, v)
        cfg = resolve_derived(cfg)
        out.append({"experiment": name, "grid_point": point, "grid_label": grid_label(point),
                    "config": cfg, "config_hash": config_hash(cfg)})
    return out


def run_id(method: str, cfg_hash: str, seed: int) -> str:
    return f"{method}-{cfg_hash}-s{seed}"


def experiments_for_suites(protocol: Dict[str, Any], suites: List[str]) -> List[str]:
    names = [n for n, e in protocol["experiments"].items() if set(e.get("suites", [])) & set(suites)]
    if not names:
        raise ValueError(f"No experiments match suites {suites}")
    return names


def all_suites(protocol: Dict[str, Any]) -> List[str]:
    return sorted({s for e in protocol["experiments"].values() for s in e.get("suites", [])})


def plan_runs(protocol: Dict[str, Any], experiments: List[str], seeds: List[int]) -> List[Dict[str, Any]]:
    """Expand experiments x grid x seeds; identical resolved configs are de-duplicated."""
    plan, seen = [], {}
    for name in experiments:
        if name not in protocol["experiments"]:
            raise KeyError(f"Unknown experiment '{name}'. Known: {list(protocol['experiments'])}")
        for point in expand_experiment(protocol, name):
            for seed in seeds:
                rid = run_id(point["config"]["method"], point["config_hash"], seed)
                entry = seen.get(rid)
                if entry is None:
                    entry = dict(point, seed=seed, run_id=rid, aliases=[])
                    seen[rid] = entry
                    plan.append(entry)
                else:
                    entry["aliases"].append((name, point["grid_label"]))
    return plan


def smoke_override(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shrink the budget for a quick end-to-end smoke test (never used for paper results)."""
    cfg = copy.deepcopy(cfg)
    if cfg["method"] == "centralized":
        cfg["centralized"]["epochs"] = 2
        cfg["centralized"]["max_epochs_es"] = 2
        cfg["centralized"]["select_every"] = 1
        cfg["fed"] = {"num_rounds": 2, "local_epochs": 1}
    else:
        cfg["fed"]["num_rounds"] = 2
        cfg["fed"]["local_epochs"] = 1
    return cfg
