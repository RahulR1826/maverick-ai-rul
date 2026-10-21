"""
paper/check_numbers.py
----------------------
Verifies that every "mean ± SD" pair in paper/updated_sections.tex exists in the
generated results (results/v2/tables/summary.csv), so no manuscript number can
drift from the experiments.

python paper/check_numbers.py
"""

import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tex = open(os.path.join(ROOT, "paper", "updated_sections.tex"), encoding="utf-8").read()
s = pd.read_csv(os.path.join(ROOT, "results", "v2", "tables", "summary.csv"))

known = set()
for m in ("rmse", "mae", "nasa_score"):
    for _, r in s.iterrows():
        for d in (1, 2):
            known.add((f"{r[m + '_mean']:.{d}f}", f"{r[m + '_std']:.{d}f}"))

paired = pd.read_csv(os.path.join(ROOT, "results", "v2", "tables", "paired_vs_fedavg_iid.csv"))
for _, r in paired.iterrows():
    known.add((f"{abs(r.mean_diff):.2f}", f"{r.std_diff:.2f}"))

pairs = re.findall(r"(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)", tex)
bad = [p for p in pairs if p not in known]
print(f"{len(pairs)} mean +/- SD pairs in the manuscript; {len(pairs) - len(bad)} found in summary.csv / paired_vs_fedavg_iid.csv")
for p in bad:
    print("NOT FOUND:", p)
sys.exit(1 if bad else 0)
