"""
serving/precompute.py
---------------------
Builds the serving cache: ensemble predictions and integrated-gradients
attributions for every cycle of every FD001 test engine, for each model family.

python -m serving.precompute                  # all families, GPU if available
python -m serving.precompute --family fedavg_iid --device cpu

Results go to cache/serving/ (git-ignored) and are keyed by checkpoint hashes,
so they are recomputed automatically when the models change.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from serving.model_store import FAMILIES, IG_STEPS, ModelStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), action="append")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--steps", type=int, default=IG_STEPS)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    store = ModelStore()
    for fam in a.family or list(FAMILIES):
        t0 = time.perf_counter()
        info = store.family_info(fam)                     # predictions (cached on first run)
        t1 = time.perf_counter()
        path = store.precompute_attributions(fam, device, steps=a.steps)
        t2 = time.perf_counter()
        x = store.explain(store.test_units[0], int(store.engines[store.test_units[0]].cycles[-1]), fam, steps=a.steps)
        print(f"{fam}: predictions {info['cache']} ({t1 - t0:.1f}s); attributions {t2 - t1:.1f}s on {device} -> {os.path.relpath(path, ROOT)}; "
              f"completeness error (engine 1, last cycle) {x['completeness_error']:+.3f} cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
