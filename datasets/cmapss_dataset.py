"""
datasets/cmapss_dataset.py
--------------------------
PyTorch Dataset wrapper for preprocessed C-MAPSS sliding-window arrays.

Wraps (X, y) NumPy arrays into a ``torch.utils.data.Dataset`` so they can be
fed directly to a ``DataLoader`` with batching, shuffling and multi-worker
prefetching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class CMAPSSDataset(Dataset):
    """PyTorch Dataset for C-MAPSS RUL regression.

    Parameters
    ----------
    X:
        Window features of shape ``(N, window_size, n_features)``.
    y:
        RUL targets of shape ``(N,)``.
    normalise_targets:
        If ``True``, divide targets by ``rul_cap`` so they lie in ``[0, 1]``.
        The inverse transform must be applied to predictions before computing
        physical metrics (RMSE in cycles).
    rul_cap:
        Maximum RUL cap used during preprocessing. Required when
        ``normalise_targets=True``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalise_targets: bool = False,
        rul_cap: int = 125,
    ) -> None:
        super().__init__()
        # Store as float32 tensors
        self.X = torch.from_numpy(X.astype(np.float32))  # (N, W, F)
        y_arr = y.astype(np.float32)

        if normalise_targets:
            y_arr = y_arr / float(rul_cap)

        self.y = torch.from_numpy(y_arr)                 # (N,)
        self.normalise_targets = normalise_targets
        self.rul_cap = rul_cap

    # ------------------------------------------------------------------ #
    #  Required Dataset interface                                          #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (x, y) pair.

        Parameters
        ----------
        idx:
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``x`` of shape ``(window_size, n_features)`` and scalar ``y``.
        """
        return self.X[idx], self.y[idx]

    # ------------------------------------------------------------------ #
    #  Convenience properties                                              #
    # ------------------------------------------------------------------ #

    @property
    def n_features(self) -> int:
        """Number of input features (last dimension of X)."""
        return self.X.shape[-1]

    @property
    def window_size(self) -> int:
        """Window length (second dimension of X)."""
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        """Total number of windows."""
        return len(self.X)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: CMAPSSDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Construct a ``DataLoader`` from a ``CMAPSSDataset``.

    Parameters
    ----------
    dataset:
        Instantiated dataset.
    batch_size:
        Mini-batch size.
    shuffle:
        Whether to shuffle samples each epoch.
    num_workers:
        Number of worker processes for data loading.
    pin_memory:
        Pin CPU memory for faster GPU transfer.
    drop_last:
        Drop the final incomplete batch if ``True``.

    Returns
    -------
    DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )


def build_datasets(
    splits: dict,
    normalise_targets: bool = False,
    rul_cap: int = 125,
) -> Tuple["CMAPSSDataset", "CMAPSSDataset", "CMAPSSDataset"]:
    """Build train/val/test CMAPSSDatasets from a pre-loaded splits dict.

    Parameters
    ----------
    splits:
        Dict returned by ``preprocessing.preprocess_dataset`` or
        ``preprocessing.load_processed``.  Expected keys: ``"train"``,
        ``"val"``, ``"test"``.
    normalise_targets:
        Divide RUL targets by ``rul_cap``.
    rul_cap:
        Maximum RUL for normalisation.

    Returns
    -------
    Tuple[CMAPSSDataset, CMAPSSDataset, CMAPSSDataset]
        Train, validation and test datasets.
    """
    train_ds = CMAPSSDataset(*splits["train"], normalise_targets=normalise_targets, rul_cap=rul_cap)
    val_ds = CMAPSSDataset(*splits["val"], normalise_targets=normalise_targets, rul_cap=rul_cap)
    test_ds = CMAPSSDataset(*splits["test"], normalise_targets=normalise_targets, rul_cap=rul_cap)
    return train_ds, val_ds, test_ds
