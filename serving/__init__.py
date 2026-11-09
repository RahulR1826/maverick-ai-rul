"""
serving
-------
Model-serving backend for the Maverick AI dashboard.

Everything served here is computed from real artifacts: the v2 checkpoints
(``checkpoints/v2``), the NASA C-MAPSS FD001 files (``CMAPSSData``) and the
v2 result tables (``results/v2/tables``). Nothing is simulated; if an artifact
is missing the API answers 503 instead of inventing data.
"""
