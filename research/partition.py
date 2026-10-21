"""
research/partition.py
---------------------
Client partitioning of the training windows.  Every partition assigns EVERY
training window to exactly one client (no window is discarded or duplicated);
this is asserted.  Total training data is therefore constant across K.

iid_window
    Random permutation of all windows, split into K near-equal shards.  Windows
    of one engine are spread over all clients (an engine's trajectory is NOT
    owned by a single client).
dirichlet_window
    Label-skew partition: RUL is binned into ``n_rul_bins`` equal-width bins on
    [0, rul_cap]; for each bin, client proportions ~ Dirichlet(alpha * 1_K).
    This changes BOTH the RUL distribution and the number of windows per client
    (quantity skew), and still spreads engines over clients.
engine
    Whole-engine ownership: training engines are randomly permuted (seeded) and
    split into K near-equal groups; a client holds all windows of its engines.

Minimum-client-size safeguard: if any client receives fewer than
``min_client_windows`` windows, the partition is redrawn with the same RNG
stream (attempt counter recorded).  After ``max_partition_attempts`` failures a
``PartitionError`` is raised -- data are never silently dropped.
"""

from typing import Dict, List, Optional

import numpy as np

from research.data import WindowSet


class PartitionError(RuntimeError):
    pass


def derive_seed(seed: int, *keys) -> int:
    """Deterministic, independent sub-seed for a named randomness stream."""
    ints = [int(seed)] + [sum((i + 1) * ord(ch) for i, ch in enumerate(str(k))) for k in keys]
    return int(np.random.SeedSequence(ints).generate_state(1)[0])


def iid_window(n: int, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    return [np.sort(c) for c in np.array_split(rng.permutation(n), k)]


def rul_bins(y: np.ndarray, rul_cap: float, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, float(rul_cap), n_bins + 1)
    return np.clip(np.digitize(y, edges[1:-1], right=False), 0, n_bins - 1)


def dirichlet_window(y: np.ndarray, k: int, alpha: float, rul_cap: float, n_bins: int,
                     rng: np.random.Generator) -> List[np.ndarray]:
    bins = rul_bins(y, rul_cap, n_bins)
    parts: List[List[int]] = [[] for _ in range(k)]
    for b in range(n_bins):
        idx = np.flatnonzero(bins == b)
        if len(idx) == 0:
            continue
        idx = rng.permutation(idx)
        p = rng.dirichlet(np.full(k, float(alpha)))
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for cid, chunk in enumerate(np.split(idx, cuts)):
            parts[cid].extend(chunk.tolist())
    return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]


def engine_groups(units: List[int], k: int, rng: np.random.Generator) -> List[List[int]]:
    if k > len(units):
        raise PartitionError(f"Cannot give {len(units)} engines to {k} clients")
    perm = rng.permutation(np.asarray(sorted(units)))
    return [sorted(int(u) for u in g) for g in np.array_split(perm, k)]


def partition(train: WindowSet, fed_cfg: Dict, rul_cap: float, seed: int) -> Dict:
    """Return ``{"indices": [idx_per_client], "attempts": int, "engine_groups": ...}``."""
    k = int(fed_cfg["num_clients"])
    if k < 1:
        raise PartitionError("num_clients must be >= 1")
    kind = fed_cfg["partition"]
    min_n = int(fed_cfg.get("min_client_windows", 1))
    rng = np.random.default_rng(derive_seed(seed, "partition", kind, k, fed_cfg.get("dirichlet_alpha")))
    n = len(train)
    groups: Optional[List[List[int]]] = None
    for attempt in range(1, int(fed_cfg.get("max_partition_attempts", 100)) + 1):
        if kind == "iid_window":
            idx = iid_window(n, k, rng)
        elif kind == "dirichlet_window":
            idx = dirichlet_window(train.y, k, fed_cfg["dirichlet_alpha"], rul_cap, fed_cfg["n_rul_bins"], rng)
        elif kind == "engine":
            groups = engine_groups(sorted(set(train.unit.tolist())), k, rng)
            idx = [np.flatnonzero(np.isin(train.unit, g)) for g in groups]
        else:
            raise PartitionError(f"Unknown partition '{kind}'")
        if min(len(i) for i in idx) >= min_n:
            check_partition(idx, n)
            return {"indices": idx, "attempts": attempt, "engine_groups": groups}
    raise PartitionError(
        f"{kind} (K={k}, alpha={fed_cfg.get('dirichlet_alpha')}): could not satisfy "
        f"min_client_windows={min_n} after {attempt} attempts")


def check_partition(idx: List[np.ndarray], n: int) -> None:
    allidx = np.concatenate(idx) if idx else np.array([], dtype=np.int64)
    if len(allidx) != n or len(np.unique(allidx)) != n:
        raise PartitionError("Partition must cover every training window exactly once")
    if any(len(i) == 0 for i in idx):
        raise PartitionError("Empty client")


def client_stats(train: WindowSet, idx: List[np.ndarray], rul_cap: float, n_bins: int) -> List[Dict]:
    """Per-client sample/engine counts and RUL-label distribution."""
    owners: Dict[int, set] = {}
    for cid, ii in enumerate(idx):
        for u in np.unique(train.unit[ii]):
            owners.setdefault(int(u), set()).add(cid)
    rows = []
    for cid, ii in enumerate(idx):
        y = train.y[ii]
        units = np.unique(train.unit[ii])
        hist = np.bincount(rul_bins(y, rul_cap, n_bins), minlength=n_bins)
        rows.append({
            "client_id": cid, "n_windows": int(len(ii)), "share": float(len(ii) / len(train)),
            "n_engines": int(len(units)),
            "n_engines_exclusive": int(sum(1 for u in units if len(owners[int(u)]) == 1)),
            "rul_mean": float(y.mean()), "rul_std": float(y.std()), "rul_min": float(y.min()),
            "rul_q25": float(np.percentile(y, 25)), "rul_median": float(np.median(y)),
            "rul_q75": float(np.percentile(y, 75)), "rul_max": float(y.max()),
            "frac_at_cap": float(np.mean(y >= rul_cap)),
            "rul_bin_counts": hist.tolist(),
            "engines": [int(u) for u in units] if len(units) <= 30 else f"{len(units)} engines",
        })
    return rows
