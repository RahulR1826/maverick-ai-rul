"""
research/data.py
----------------
Leakage-audited C-MAPSS preprocessing that keeps engine identity for every
window (needed for whole-engine partitions and per-engine test errors).

Protocol (FD001 defaults; see configs/fd001_protocol.json)
---------------------------------------------------------
1. Training RUL label: ``min(max_cycle(unit) - cycle, rul_cap)`` (piece-wise linear).
2. Features: 3 operational settings + 14 sensors (sensors 1,5,6,10,16,18,19
   dropped) = 17 features.  The sensor list is fixed a priori from the
   literature, not selected on validation or test data.
3. Train/validation split at the ENGINE level: ``val_fraction`` of the 100
   training engines (10 engines) drawn with ``np.random.default_rng(val_split_seed)``
   -- identical to the original ``datasets/preprocessing.py`` so the same 10
   engines are held out.  The split is FIXED across run seeds and methods so
   that all methods select checkpoints on exactly the same engines.
4. MinMaxScaler fitted on the 90 training engines only; applied unchanged to
   validation and official test data.
5. Windows: length ``window_size``, stride 1, label = RUL at the window's last
   cycle.  Engines shorter than the window are reported, never silently dropped.
6. Official test set: the last ``window_size`` cycles of each of the 100 test
   engines; ground truth from RUL_FD001.txt, capped at ``rul_cap`` when
   ``cap_test_rul`` is true (common practice; uncapped labels are kept as well).
   The test set is only used for final evaluation of already-selected models.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMN_NAMES: List[str] = (
    ["unit", "cycle"] + [f"os_{i}" for i in range(1, 4)] + [f"s_{i}" for i in range(1, 22)]
)


@dataclass
class WindowSet:
    X: np.ndarray            # (N, W, F) float32
    y: np.ndarray            # (N,) float32, RUL at last window step
    unit: np.ndarray         # (N,) int32, engine id of each window
    cycle: np.ndarray        # (N,) int32, last cycle of each window

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "WindowSet":
        return WindowSet(self.X[idx], self.y[idx], self.unit[idx], self.cycle[idx])


@dataclass
class PreparedData:
    train: WindowSet
    val: WindowSet
    test: WindowSet
    test_rul_uncapped: np.ndarray
    feature_cols: List[str]
    train_units: List[int]
    val_units: List[int]
    test_units: List[int]
    raw_train_rows: int
    meta: Dict = field(default_factory=dict)


def load_raw(dataset_id: str, split: str, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{split}_{dataset_id}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(f"{path}: expected {len(COLUMN_NAMES)} columns, found {df.shape[1]}")
    df.columns = COLUMN_NAMES
    if df.isna().any().any():
        raise ValueError(f"{path}: contains missing values")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    for unit, g in df.groupby("unit"):
        c = g["cycle"].to_numpy()
        if not np.array_equal(c, np.arange(1, len(c) + 1)):
            raise ValueError(f"{path}: engine {unit} cycles are not 1..T in order")
    return df


def add_train_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=rul_cap).astype(np.float32)
    return df


def feature_columns(drop_sensors: List[int]) -> List[str]:
    drop = {f"s_{i}" for i in drop_sensors}
    return [c for c in COLUMN_NAMES if c not in ("unit", "cycle") and c not in drop]


def split_engines(units: List[int], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """Same draw as the original preprocessing (np.random.default_rng(seed).choice)."""
    units = sorted(units)
    n_val = max(1, int(len(units) * val_fraction))
    rng = np.random.default_rng(seed)
    val = sorted(int(u) for u in rng.choice(units, size=n_val, replace=False))
    train = [int(u) for u in units if u not in set(val)]
    return train, val


def make_windows(df: pd.DataFrame, cols: List[str], window: int, stride: int) -> WindowSet:
    Xs, ys, us, cs = [], [], [], []
    short = []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        F = g[cols].to_numpy(dtype=np.float32)
        Y = g["rul"].to_numpy(dtype=np.float32)
        C = g["cycle"].to_numpy()
        T = len(g)
        if T < window:
            short.append(int(unit))
            continue
        for s in range(0, T - window + 1, stride):
            e = s + window
            Xs.append(F[s:e]); ys.append(Y[e - 1]); us.append(int(unit)); cs.append(int(C[e - 1]))
    if short:
        raise ValueError(f"Training/validation engines shorter than window {window}: {short}")
    return WindowSet(np.stack(Xs).astype(np.float32), np.asarray(ys, np.float32),
                     np.asarray(us, np.int32), np.asarray(cs, np.int32))


def make_test_windows(test_df: pd.DataFrame, rul: np.ndarray, cols: List[str], window: int,
                      rul_cap: Optional[int]) -> Tuple[WindowSet, np.ndarray]:
    units = sorted(test_df["unit"].unique())
    if len(units) != len(rul):
        raise ValueError(f"Test engines ({len(units)}) != RUL rows ({len(rul)})")
    Xs, cs = [], []
    for unit in units:
        g = test_df[test_df["unit"] == unit].sort_values("cycle")
        F = g[cols].to_numpy(dtype=np.float32)
        if len(F) < window:
            # Not triggered for FD001 (shortest test engine has 31 cycles); kept
            # for other subsets.  Padding repeats the first observed row rather
            # than inserting zeros, which would be out-of-distribution.
            F = np.vstack([np.repeat(F[:1], window - len(F), axis=0), F])
        Xs.append(F[-window:])
        cs.append(int(g["cycle"].max()))
    y_unc = rul.astype(np.float32)
    y = np.minimum(y_unc, rul_cap).astype(np.float32) if rul_cap is not None else y_unc
    ws = WindowSet(np.stack(Xs).astype(np.float32), y, np.asarray(units, np.int32), np.asarray(cs, np.int32))
    return ws, y_unc


def prepare_data(data_cfg: Dict, data_dir: str) -> PreparedData:
    ds = data_cfg["dataset_id"]
    cols = feature_columns(data_cfg["drop_sensors"])
    train_raw = add_train_rul(load_raw(ds, "train", data_dir), data_cfg["rul_cap"])
    test_raw = load_raw(ds, "test", data_dir)
    rul = pd.read_csv(os.path.join(data_dir, f"RUL_{ds}.txt"), header=None, sep=r"\s+").iloc[:, 0].to_numpy()

    tr_units, va_units = split_engines(sorted(train_raw["unit"].unique()), data_cfg["val_fraction"],
                                       data_cfg["val_split_seed"])
    tr_df = train_raw[train_raw["unit"].isin(tr_units)]
    va_df = train_raw[train_raw["unit"].isin(va_units)]

    scaler = MinMaxScaler((0, 1)).fit(tr_df[cols].to_numpy())   # training engines only
    def _scale(df):
        df = df.copy()
        df[cols] = scaler.transform(df[cols].to_numpy()).astype(np.float32)
        return df

    W, S = data_cfg["window_size"], data_cfg["stride"]
    train = make_windows(_scale(tr_df), cols, W, S)
    val = make_windows(_scale(va_df), cols, W, S)
    test, y_unc = make_test_windows(_scale(test_raw), rul, cols, W,
                                    data_cfg["rul_cap"] if data_cfg["cap_test_rul"] else None)

    pd_ = PreparedData(train=train, val=val, test=test, test_rul_uncapped=y_unc, feature_cols=cols,
                       train_units=tr_units, val_units=va_units,
                       test_units=[int(u) for u in test.unit], raw_train_rows=len(tr_df))
    pd_.meta = {
        "dataset_id": ds, "n_features": len(cols), "feature_cols": cols,
        "n_train_engines": len(tr_units), "n_val_engines": len(va_units), "n_test_engines": len(test),
        "val_units": va_units, "n_train_windows": len(train), "n_val_windows": len(val),
        "scaler": "MinMaxScaler(0,1) fitted on training engines only",
        "scaler_data_min": scaler.data_min_.tolist(), "scaler_data_max": scaler.data_max_.tolist(),
        "constant_train_features": [c for c, lo, hi in zip(cols, scaler.data_min_, scaler.data_max_) if hi == lo],
        "test_rul_capped": bool(data_cfg["cap_test_rul"]),
        "raw_train_rows": len(tr_df),
    }
    validate_prepared(pd_, data_cfg)
    return pd_


def validate_prepared(d: PreparedData, data_cfg: Dict) -> None:
    """Sanity checks that would expose leakage or malformed data."""
    F = len(d.feature_cols)
    for name, ws in (("train", d.train), ("val", d.val), ("test", d.test)):
        if ws.X.ndim != 3 or ws.X.shape[1] != data_cfg["window_size"] or ws.X.shape[2] != F:
            raise ValueError(f"{name}: bad window shape {ws.X.shape}")
        if not (np.isfinite(ws.X).all() and np.isfinite(ws.y).all()):
            raise ValueError(f"{name}: non-finite values")
        if (ws.y < 0).any():
            raise ValueError(f"{name}: negative RUL labels")
        if name != "test" and (ws.y > data_cfg["rul_cap"]).any():
            raise ValueError(f"{name}: RUL above cap")
    if set(d.train_units) & set(d.val_units):
        raise AssertionError("Engine overlap between train and validation")
    if set(np.unique(d.train.unit)) != set(d.train_units) or set(np.unique(d.val.unit)) != set(d.val_units):
        raise AssertionError("Window engine ids do not match the engine split")
    if len(np.unique(d.test.unit)) != len(d.test):
        raise AssertionError("Test set must contain exactly one window per test engine")


_CACHE: Dict[str, PreparedData] = {}


def get_data(data_cfg: Dict, data_dir: str) -> PreparedData:
    """In-process cache keyed on the data config (preprocessing takes ~1 s)."""
    import json
    key = json.dumps(data_cfg, sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = prepare_data(data_cfg, data_dir)
    return _CACHE[key]
