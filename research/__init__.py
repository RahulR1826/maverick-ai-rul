"""
research
--------
Reproducible experimental pipeline for the NICEDT-2027 paper
"Federated Learning for Aircraft Engine Prognostics: An Analysis of Accuracy,
Communication, and Data Heterogeneity Trade-offs".

This package sits next to the original Flower-based code (``federated/``,
``training/``, ``run_experiments.py``) and re-uses the original LSTM
architecture (``models/lstm_model.py``) and preprocessing rules
(``datasets/preprocessing.py``).  It replaces the Ray/Flower simulation with a
deterministic, sequential in-process FedAvg simulation so that every source of
randomness is controlled by an explicit seed and runs can be repeated across
matched seed lists.

Entry points
------------
python -m research.run        -- run experiments (suites / individual experiments)
python -m research.aggregate  -- aggregate per-seed results into mean ± std tables
python -m research.plots      -- regenerate publication figures from result files
python -m research.report     -- regenerate the experiment report (markdown)
"""
