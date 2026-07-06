"""
datasets/preprocessing.py
--------------------------
Handles all raw-data loading and feature engineering for the NASA C-MAPSS dataset.

Steps
-----
1. Load the space-delimited raw text files.
2. Drop low-information sensors (configurable).
3. Compute piece-wise linear Remaining Useful Life (RUL) labels.
4. Normalise feature columns using *training* statistics only.
5. Build sliding-window samples (X, y) ready for PyTorch DataLoader.
6. Persist processed arrays to disk for fast re-use.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import CFG, DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COLUMN_NAMES: List[str] = (
    ["unit", "cycle"]
    + [f"os_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(dataset_id: str, split: str, data_dir: str) -> pd.DataFrame:
    """Load a raw C-MAPSS text file into a DataFrame.

    Parameters
    ----------
    dataset_id:
        One of ``FD001``, ``FD002``, ``FD003``, ``FD004``.
    split:
        ``"train"`` or ``"test"``.
    data_dir:
        Directory containing the raw ``.txt`` files.

    Returns
    -------
    pd.DataFrame
        Raw data with named columns, no normalisation applied.
    """
    filename = f"{split}_{dataset_id}.txt"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"C-MAPSS file not found: {filepath}\n"
            "Please place the raw dataset files inside the configured data directory."
        )

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=_COLUMN_NAMES,
        engine="python",
    )
    logger.info("Loaded %s (%d rows, %d cols)", filepath, len(df), df.shape[1])
    return df


def compute_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """Append a ``rul`` column with piece-wise linear RUL labels.

    For training data the true final cycle per engine is known (max cycle).
    RUL is capped at ``rul_cap`` to avoid overfitting on the flat early-life region.

    Parameters
    ----------
    df:
        DataFrame with ``unit`` and ``cycle`` columns.
    rul_cap:
        Maximum RUL value (cycles) — default 125.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``rul`` column.
    """
    # Compute the last observed cycle for each engine
    max_cycles = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit")

    # RUL = max_cycle – current_cycle, capped at rul_cap
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def get_feature_columns(df: pd.DataFrame, drop_sensors: List[int]) -> List[str]:
    """Return the list of feature columns after dropping low-variance sensors.

    Operational settings are always kept because they encode regime information.
    The sensor indices in ``drop_sensors`` are 1-based (matching the paper).

    Parameters
    ----------
    df:
        DataFrame with full column set.
    drop_sensors:
        1-based indices of sensors to exclude (e.g. ``[1, 5, 6, ...]``).

    Returns
    -------
    List[str]
        Ordered list of feature column names.
    """
    drop_cols = {f"s_{i}" for i in drop_sensors}
    feature_cols = [
        c for c in df.columns
        if c not in {"unit", "cycle", "rul"} and c not in drop_cols
    ]
    logger.debug("Feature columns (%d): %s", len(feature_cols), feature_cols)
    return feature_cols


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> MinMaxScaler:
    """Fit a ``MinMaxScaler`` on training features only.

    Parameters
    ----------
    train_df:
        Training DataFrame (already has RUL column but scaling targets features).
    feature_cols:
        Columns to scale.

    Returns
    -------
    MinMaxScaler
        Fitted scaler (fitted on training statistics only).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols].values)
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: List[str], scaler: MinMaxScaler) -> pd.DataFrame:
    """Apply a pre-fitted scaler to a DataFrame in-place.

    Parameters
    ----------
    df:
        DataFrame to normalise.
    feature_cols:
        Feature columns to transform.
    scaler:
        Pre-fitted ``MinMaxScaler``.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalised feature columns.
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    stride: int,
    rul_col: str = "rul",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window (X, y) tensors from a normalised engine DataFrame.

    For each engine, windows of length ``window_size`` are extracted every
    ``stride`` cycles.  The label for each window is the RUL at its last cycle.

    Parameters
    ----------
    df:
        Normalised DataFrame containing feature columns and RUL labels.
    feature_cols:
        Column names to include in X.
    window_size:
        Number of time steps per sample.
    stride:
        Step between consecutive windows.
    rul_col:
        Name of the target column.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X of shape ``(N, window_size, n_features)`` and
        y of shape ``(N,)``.
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for unit_id, group in df.groupby("unit"):
        group = group.sort_values("cycle")
        features = group[feature_cols].values      # (T, F)
        labels = group[rul_col].values             # (T,)
        T = len(features)

        # Need at least one full window
        if T < window_size:
            logger.warning(
                "Engine %d has only %d cycles (<window_size=%d), skipped.",
                unit_id, T, window_size,
            )
            continue

        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            X_list.append(features[start:end])     # (window_size, F)
            y_list.append(labels[end - 1])         # RUL at last step

    X = np.array(X_list, dtype=np.float32)        # (N, W, F)
    y = np.array(y_list, dtype=np.float32)        # (N,)
    logger.info("Sliding windows: X=%s  y=%s", X.shape, y.shape)
    return X, y


def make_test_windows(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    rul_cap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build test windows by taking the *last* ``window_size`` cycles per engine.

    The ground-truth RUL for test engines comes from the RUL_FD00x.txt file.

    Parameters
    ----------
    test_df:
        Normalised test DataFrame (no ``rul`` column from preprocessing).
    rul_df:
        DataFrame with a single ``rul`` column, one row per engine.
    feature_cols:
        Feature column names.
    window_size:
        Window length.
    rul_cap:
        Cap applied to ground-truth labels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X ``(n_engines, window_size, n_features)`` and y ``(n_engines,)``.
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    true_ruls = rul_df["rul"].values.clip(max=rul_cap)

    for idx, (unit_id, group) in enumerate(test_df.groupby("unit")):
        group = group.sort_values("cycle")
        features = group[feature_cols].values

        # Pad with zeros if shorter than window_size
        if len(features) < window_size:
            pad = np.zeros((window_size - len(features), features.shape[1]), dtype=np.float32)
            features = np.vstack([pad, features])
        else:
            features = features[-window_size:]

        X_list.append(features)
        y_list.append(float(true_ruls[idx]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info("Test windows: X=%s  y=%s", X.shape, y.shape)
    return X, y


def preprocess_dataset(
    dataset_id: str = "FD001",
    data_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    cfg: Optional[DataConfig] = None,
    force: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """End-to-end preprocessing pipeline for one C-MAPSS subset.

    Loads, cleans, normalises and windows the data.  Results are cached as
    NumPy ``.npy`` files to speed up subsequent runs.

    Parameters
    ----------
    dataset_id:
        One of ``FD001``–``FD004``.
    data_dir:
        Path to raw text files.  Defaults to ``CFG.data_dir``.
    processed_dir:
        Output directory for cached arrays.  Defaults to ``CFG.processed_dir``.
    cfg:
        ``DataConfig`` instance; falls back to ``CFG.data``.
    force:
        Re-run preprocessing even if cached files exist.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Keys: ``"train"``, ``"val"``, ``"test"``.
        Values: ``(X, y)`` tuples.
    """
    cfg = cfg or CFG.data
    data_dir = data_dir or CFG.data_dir
    processed_dir = processed_dir or CFG.processed_dir

    out_dir = os.path.join(processed_dir, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    cache_paths = {
        split: {
            "X": os.path.join(out_dir, f"X_{split}.npy"),
            "y": os.path.join(out_dir, f"y_{split}.npy"),
        }
        for split in ("train", "val", "test")
    }
    scaler_path = os.path.join(out_dir, "scaler.joblib")

    # Return cached results if available and not forced
    if not force and all(
        os.path.exists(v["X"]) and os.path.exists(v["y"])
        for v in cache_paths.values()
    ):
        logger.info("Loading cached preprocessed data from %s", out_dir)
        return {
            split: (np.load(paths["X"]), np.load(paths["y"]))
            for split, paths in cache_paths.items()
        }

    logger.info("Preprocessing %s …", dataset_id)

    # --- 1. Load raw files ---------------------------------------------------
    train_raw = load_raw(dataset_id, "train", data_dir)
    test_raw = load_raw(dataset_id, "test", data_dir)

    rul_file = os.path.join(data_dir, f"RUL_{dataset_id}.txt")
    rul_df = pd.read_csv(rul_file, header=None, names=["rul"])

    # --- 2. Compute training RUL and identify feature columns ----------------
    train_raw = compute_rul(train_raw, cfg.rul_cap)
    feature_cols = get_feature_columns(train_raw, cfg.drop_sensors)

    # --- 3. Train/validation split at the engine level -----------------------
    rng = np.random.default_rng(cfg.seed)
    all_units = sorted(train_raw["unit"].unique())
    n_val = max(1, int(len(all_units) * cfg.test_split))
    val_units = set(rng.choice(all_units, size=n_val, replace=False).tolist())
    train_units = [u for u in all_units if u not in val_units]

    train_df = train_raw[train_raw["unit"].isin(train_units)].copy()
    val_df = train_raw[train_raw["unit"].isin(val_units)].copy()

    # --- 4. Fit scaler on training engines only ------------------------------
    scaler = fit_scaler(train_df, feature_cols)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    train_df = apply_scaler(train_df, feature_cols, scaler)
    val_df = apply_scaler(val_df, feature_cols, scaler)
    test_norm = apply_scaler(test_raw, feature_cols, scaler)

    # --- 5. Build sliding windows --------------------------------------------
    X_train, y_train = make_windows(train_df, feature_cols, cfg.window_size, cfg.stride)
    X_val, y_val = make_windows(val_df, feature_cols, cfg.window_size, cfg.stride)
    X_test, y_test = make_test_windows(test_norm, rul_df, feature_cols, cfg.window_size, cfg.rul_cap)

    # --- 6. Cache to disk ----------------------------------------------------
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    for split, (X, y) in splits.items():
        np.save(cache_paths[split]["X"], X)
        np.save(cache_paths[split]["y"], y)
        logger.info("Saved %s split: X=%s  y=%s", split, X.shape, y.shape)

    # Persist number of features so ModelConfig can be set automatically
    n_features_path = os.path.join(out_dir, "n_features.txt")
    with open(n_features_path, "w") as f:
        f.write(str(X_train.shape[2]))

    return splits


def load_processed(
    dataset_id: str = "FD001",
    processed_dir: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load already-preprocessed data from the cache directory.

    Parameters
    ----------
    dataset_id:
        Dataset identifier.
    processed_dir:
        Root processed data directory.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        ``{"train": (X, y), "val": (X, y), "test": (X, y)}``.
    """
    processed_dir = processed_dir or CFG.processed_dir
    out_dir = os.path.join(processed_dir, dataset_id)

    splits = {}
    for split in ("train", "val", "test"):
        X_path = os.path.join(out_dir, f"X_{split}.npy")
        y_path = os.path.join(out_dir, f"y_{split}.npy")
        if not os.path.exists(X_path):
            raise FileNotFoundError(
                f"Processed file not found: {X_path}\n"
                "Run preprocess_dataset() first."
            )
        splits[split] = (np.load(X_path), np.load(y_path))

    return splits


def get_n_features(dataset_id: str = "FD001", processed_dir: Optional[str] = None) -> int:
    """Read the cached number of input features for a dataset.

    Parameters
    ----------
    dataset_id:
        Dataset identifier.
    processed_dir:
        Root processed data directory.

    Returns
    -------
    int
        Number of feature dimensions.
    """
    processed_dir = processed_dir or CFG.processed_dir
    path = os.path.join(processed_dir, dataset_id, "n_features.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"n_features.txt not found at {path}. Run preprocess_dataset() first."
        )
    with open(path) as f:
        return int(f.read().strip())
