"""
datasets/split_clients.py
--------------------------
Partitions the preprocessed training dataset into per-client shards that
simulate multiple airlines (data owners) in the federated learning scenario.

Supports two split strategies
------------------------------
IID (``iid=True``)
    Shuffle all windows randomly then allocate equal-sized contiguous blocks
    to each client.  All clients see roughly the same feature distribution.

Non-IID (``iid=False``)
    Use a Dirichlet distribution over the RUL range to create heterogeneous
    shards where different clients tend to see engines in different health
    states.  The concentration parameter ``alpha`` controls heterogeneity
    (lower → more skewed).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from config import CFG, FedConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: sort windows into RUL-based classes for Non-IID splits
# ---------------------------------------------------------------------------

def _rul_to_class(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Discretise continuous RUL values into ``n_classes`` buckets.

    Parameters
    ----------
    y:
        RUL array of shape ``(N,)``.
    n_classes:
        Number of discrete classes.

    Returns
    -------
    np.ndarray
        Integer class labels ``[0, n_classes)``.
    """
    bins = np.linspace(y.min(), y.max() + 1e-6, n_classes + 1)
    return np.digitize(y, bins) - 1   # 0-indexed


# ---------------------------------------------------------------------------
# IID split
# ---------------------------------------------------------------------------

def iid_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Randomly shuffle and divide data equally among clients.

    Parameters
    ----------
    X:
        Feature windows ``(N, W, F)``.
    y:
        RUL targets ``(N,)``.
    num_clients:
        Number of clients to partition into.
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    Dict[int, Tuple[np.ndarray, np.ndarray]]
        ``{client_id: (X_client, y_client)}`` (0-indexed client IDs).
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    idx = rng.permutation(N)

    # Split indices into equal-sized chunks (last client may get fewer samples)
    chunks = np.array_split(idx, num_clients)
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for cid, chunk in enumerate(chunks):
        client_data[cid] = (X[chunk], y[chunk])
        logger.info(
            "IID client %d → %d windows (%.1f%%)",
            cid, len(chunk), 100.0 * len(chunk) / N,
        )

    return client_data


# ---------------------------------------------------------------------------
# Non-IID split (Dirichlet)
# ---------------------------------------------------------------------------

def non_iid_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    n_rul_classes: int = 10,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Partition data using a Dirichlet distribution over RUL classes.

    Each client receives a fraction of samples from each RUL class
    drawn from ``Dirichlet(alpha)``.  Small ``alpha`` creates highly
    imbalanced, health-state-skewed client datasets.

    Parameters
    ----------
    X:
        Feature windows ``(N, W, F)``.
    y:
        RUL targets ``(N,)``.
    num_clients:
        Number of clients.
    alpha:
        Dirichlet concentration parameter (default 0.5).
    seed:
        NumPy random seed.
    n_rul_classes:
        Number of discrete RUL buckets used for partitioning.

    Returns
    -------
    Dict[int, Tuple[np.ndarray, np.ndarray]]
        ``{client_id: (X_client, y_client)}``.
    """
    rng = np.random.default_rng(seed)
    classes = _rul_to_class(y, n_classes=n_rul_classes)
    class_indices: Dict[int, List[int]] = {
        c: np.where(classes == c)[0].tolist()
        for c in range(n_rul_classes)
    }

    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    for cls, idx_list in class_indices.items():
        if not idx_list:
            continue
        rng.shuffle(idx_list)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        proportions = (proportions / proportions.sum() * len(idx_list)).astype(int)

        # Fix rounding so all samples are assigned
        remainder = len(idx_list) - proportions.sum()
        for i in range(abs(remainder)):
            proportions[i % num_clients] += int(np.sign(remainder))

        ptr = 0
        for cid in range(num_clients):
            end = ptr + proportions[cid]
            client_indices[cid].extend(idx_list[ptr:end])
            ptr = end

    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    N = len(y)
    for cid, idx_list in client_indices.items():
        idx_arr = np.array(idx_list, dtype=int)
        client_data[cid] = (X[idx_arr], y[idx_arr])
        logger.info(
            "Non-IID client %d → %d windows (%.1f%%)",
            cid, len(idx_arr), 100.0 * len(idx_arr) / N,
        )

    return client_data


# ---------------------------------------------------------------------------
# Engine-level split (simulates airline data ownership)
# ---------------------------------------------------------------------------

def engine_split(
    X: np.ndarray,
    y: np.ndarray,
    engine_ids_per_client: Dict[int, List[int]],
    engine_labels: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Assign windows to clients based on engine (unit) ownership.

    Each window is tagged with its originating engine; all windows from the
    same engine go to the same client.  This is the most realistic simulation
    of an airline-owning-its-fleet scenario.

    Parameters
    ----------
    X:
        Feature windows ``(N, W, F)``.
    y:
        RUL targets ``(N,)``.
    engine_ids_per_client:
        ``{client_id: [engine_id, …]}``.
    engine_labels:
        Array of length ``N`` mapping each window to its engine ID.

    Returns
    -------
    Dict[int, Tuple[np.ndarray, np.ndarray]]
        ``{client_id: (X_client, y_client)}``.
    """
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, engines in engine_ids_per_client.items():
        mask = np.isin(engine_labels, engines)
        client_data[cid] = (X[mask], y[mask])
        logger.info(
            "Engine-split client %d → engines %s → %d windows",
            cid, engines, mask.sum(),
        )
    return client_data


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def split_for_clients(
    X: np.ndarray,
    y: np.ndarray,
    cfg: FedConfig = None,
    seed: int = 42,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Create per-client dataset partitions according to federation config.

    Parameters
    ----------
    X:
        Training feature windows.
    y:
        Training RUL targets.
    cfg:
        ``FedConfig`` controlling ``num_clients``, ``iid``, ``dirichlet_alpha``.
        Defaults to ``CFG.fed``.
    seed:
        Random seed.

    Returns
    -------
    Dict[int, Tuple[np.ndarray, np.ndarray]]
        Client-indexed data shards.
    """
    cfg = cfg or CFG.fed

    if cfg.iid:
        logger.info(
            "Splitting %d samples (IID) across %d clients.",
            len(y), cfg.num_clients,
        )
        return iid_split(X, y, num_clients=cfg.num_clients, seed=seed)
    else:
        logger.info(
            "Splitting %d samples (Non-IID, α=%.2f) across %d clients.",
            len(y), cfg.dirichlet_alpha, cfg.num_clients,
        )
        return non_iid_split(
            X, y,
            num_clients=cfg.num_clients,
            alpha=cfg.dirichlet_alpha,
            seed=seed,
        )


def get_engine_splits(
    num_engines: int,
    num_clients: int,
) -> Dict[int, List[int]]:
    """Evenly divide engine IDs (1-indexed) among clients.

    Parameters
    ----------
    num_engines:
        Total number of engines (e.g. 100 for FD001).
    num_clients:
        Number of clients.

    Returns
    -------
    Dict[int, List[int]]
        ``{client_id: [engine_id, …]}``.
    """
    engine_ids = list(range(1, num_engines + 1))
    chunks = np.array_split(engine_ids, num_clients)
    return {cid: chunk.tolist() for cid, chunk in enumerate(chunks)}
