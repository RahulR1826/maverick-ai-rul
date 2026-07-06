"""
config.py
---------
Central configuration for the Federated Learning RUL prediction pipeline.
All hyperparameters, paths, and experimental settings are defined here.
"""

import os
# Prevent PyTorch from using all CPUs per process and freezing the system
os.environ["OMP_NUM_THREADS"] = "1"

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "CMAPSSData")   # raw NASA C-MAPSS files
PROCESSED_DIR: str = os.path.join(BASE_DIR, "processed")  # preprocessed numpy arrays
LOG_DIR: str = os.path.join(BASE_DIR, "logs")             # TensorBoard + CSV logs
CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")  # saved model weights
RESULTS_DIR: str = os.path.join(BASE_DIR, "results")      # metric JSON / plots


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Parameters governing dataset loading and windowing."""

    dataset_id: str = "FD001"          # one of FD001 / FD002 / FD003 / FD004
    rul_cap: int = 125                  # piecewise-linear RUL cap (cycles)
    window_size: int = 30               # sliding-window length (time steps)
    stride: int = 1                     # step between consecutive windows
    test_split: float = 0.10           # fraction of training engines held out for validation
    seed: int = 42

    # Sensors identified as low-variance / non-informative in the literature
    # for FD001 / FD003 (single operating condition).
    # Override for FD002 / FD004 (six conditions) if desired.
    drop_sensors: List[int] = field(default_factory=lambda: [1, 5, 6, 10, 16, 18, 19])

    # Column names in the raw text files
    col_names: List[str] = field(default_factory=lambda: (
        ["unit", "cycle"]
        + [f"os_{i}" for i in range(1, 4)]
        + [f"s_{i}" for i in range(1, 22)]
    ))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """LSTM model architecture hyperparameters."""

    input_size: int = 14            # set automatically after preprocessing
    hidden_size: int = 128          # LSTM hidden dimension
    num_layers: int = 2             # stacked LSTM layers
    dropout: float = 0.2            # dropout probability (applied between LSTM layers)
    bidirectional: bool = False     # bidirectional LSTM flag


# ---------------------------------------------------------------------------
# Training (local / centralized)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyperparameters for local / centralized training."""

    epochs: int = 50                # maximum training epochs
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5      # L2 regularisation

    # Early stopping
    patience: int = 10              # epochs without improvement before stopping
    min_delta: float = 1e-4         # minimum improvement threshold

    # LR scheduling (ReduceLROnPlateau)
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6

    device: str = "auto"            # "auto" | "cpu" | "cuda" | "mps"
    num_workers: int = 0            # DataLoader workers (0 = main process)
    pin_memory: bool = False


# ---------------------------------------------------------------------------
# Federated Learning
# ---------------------------------------------------------------------------
@dataclass
class FedConfig:
    """Flower federated learning settings."""

    num_clients: int = 5            # total simulated clients (airlines)
    num_rounds: int = 20            # global federation rounds
    local_epochs: int = 5           # local epochs per round per client
    fraction_fit: float = 1.0       # fraction of clients selected each round
    fraction_evaluate: float = 1.0  # fraction of clients used for evaluation
    min_fit_clients: int = 2        # minimum clients required to start a round
    min_evaluate_clients: int = 2
    min_available_clients: int = 2

    # Data heterogeneity
    iid: bool = True                # True → uniform IID split; False → non-IID

    # Non-IID Dirichlet concentration (lower α → more skewed)
    dirichlet_alpha: float = 0.5


# ---------------------------------------------------------------------------
# Aggregated config object
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Top-level configuration composed of sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fed: FedConfig = field(default_factory=FedConfig)

    # Derived paths (populated at runtime)
    processed_dir: str = PROCESSED_DIR
    log_dir: str = LOG_DIR
    checkpoint_dir: str = CHECKPOINT_DIR
    results_dir: str = RESULTS_DIR
    data_dir: str = DATA_DIR


# ---------------------------------------------------------------------------
# Default singleton – import and use across the project
# ---------------------------------------------------------------------------
CFG = Config()
