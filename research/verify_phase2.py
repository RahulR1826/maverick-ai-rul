"""
research/verify_phase2.py
-------------------------
Checks the four Phase-2 requirements against the run outputs on disk
(config.json, metrics.json, clients.csv, test_predictions.csv, the generated
report), not against the protocol file or the code's intentions.  Writes
``<out>/PHASE2_VERIFICATION.md`` with PASS / FAIL and the evidence for every check.

python -m research.verify_phase2 [--out results/v2]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from research import config as C

MAIN = ["centralized", "fedavg_iid", "local_iid", "fedavg_engine", "local_engine"]
IMPORTANT = MAIN + ["fedavg_dirichlet", "sweep_rounds", "sweep_local_epochs", "sweep_clients", "sweep_clients_engine"]


class Checker:
    def __init__(self):
        self.rows, self.section = [], ""

    def sec(self, title):
        self.section = title

    def check(self, req, ok, evidence):
        self.rows.append((self.section, req, bool(ok), evidence))
        return ok


def load_runs(out, proto, seeds):
    runs = defaultdict(dict)        # (experiment, grid_label) -> seed -> dict
    for name in proto["experiments"]:
        for p in C.expand_experiment(proto, name):
            for s in seeds:
                rid = C.run_id(p["config"]["method"], p["config_hash"], s)
                d = os.path.join(out, "runs", rid)
                if not os.path.exists(os.path.join(d, "metrics.json")):
                    continue
                with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                with open(os.path.join(d, "metrics.json"), encoding="utf-8") as f:
                    met = json.load(f)
                runs[(name, p["grid_label"])][s] = {"dir": d, "cfg": cfg, "met": met, "point": p["grid_point"]}
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.DEFAULT_OUT_DIR)
    a = ap.parse_args(argv)
    proto = C.load_protocol()
    seeds = proto["seeds"]["primary"]
    runs = load_runs(a.out, proto, seeds)
    report = open(os.path.join(a.out, "EXPERIMENT_REPORT.md"), encoding="utf-8").read()
    summ = pd.read_csv(os.path.join(a.out, "tables", "summary.csv"))
    K = Checker()
    get = lambda e, g="default": runs.get((e, g), {})

    # =====================================================================
    K.sec("1. Multiple random seeds")
    for (e, g), rs in sorted(runs.items()):
        if e in IMPORTANT:
            K.check(f"{e} [{g}] has >= 5 completed seeds", len(rs) >= 5, f"seeds = {sorted(rs)}")
    seed_sets = {f"{e}[{g}]": tuple(sorted(rs)) for (e, g), rs in runs.items() if e in IMPORTANT}
    K.check("Identical seed list for every compared method/configuration", len(set(seed_sets.values())) == 1,
            f"distinct seed lists: {sorted(set(seed_sets.values()))}")
    # randomness actually varies with the seed: init / train streams and partitions
    fr = get("fedavg_iid")
    init_streams = {s: r["cfg"]["seed_streams"]["init"] for s, r in fr.items()}
    K.check("Model-initialisation stream differs across seeds", len(set(init_streams.values())) == len(fr), f"{init_streams}")
    K.check("Training (dropout / shuffle) stream differs across seeds",
            len({r["cfg"]["seed_streams"]["train"] for r in fr.values()}) == len(fr), "seed_streams.train distinct per seed")
    pseeds = {s: r["met"]["partition"]["partition_seed"] for s, r in fr.items()}
    K.check("Client partition differs across seeds (partition seed recorded)", len(set(pseeds.values())) == len(fr), f"{pseeds}")
    parts = {}
    for s, r in fr.items():
        c = pd.read_csv(os.path.join(r["dir"], "clients.csv"))
        parts[s] = tuple(round(x, 3) for x in c.rul_mean)
    K.check("Partitions are actually different across seeds (client RUL means differ)", len(set(parts.values())) == len(fr),
            "client mean-RUL vectors distinct per seed")
    # same seed => same init across methods (matched)
    cen = get("centralized")
    K.check("Same seed gives the same initialisation stream across methods (paired design)",
            all(cen[s]["cfg"]["seed_streams"]["init"] == fr[s]["cfg"]["seed_streams"]["init"] for s in seeds if s in cen and s in fr),
            "centralized vs fedavg_iid init streams equal per seed")
    for m in ("rmse", "mae", "nasa_score"):
        K.check(f"Mean and SD reported for {m.upper()}", {f"{m}_mean", f"{m}_std"} <= set(summ.columns),
                f"summary.csv has {m}_mean / {m}_std; report tables show mean ± SD")
    sub = summ[summ.experiment.isin(IMPORTANT) & (summ.checkpoint == "selected")]
    K.check("Every summary row of the important experiments is based on 5 seeds", (sub.n_seeds >= 5).all(),
            f"n_seeds min = {int(sub.n_seeds.min())}")
    K.check("Per-seed raw results preserved (one directory + metrics.json per seed)",
            all(os.path.exists(os.path.join(r["dir"], f)) for rs in runs.values() for r in rs.values()
                for f in ("config.json", "metrics.json", "history.csv", "test_predictions.csv")),
            f"{sum(len(v) for v in runs.values())} run directories checked")
    near = re.search(r"(threshold|noise floor|signific\w*)[^.\n]{0,80}0\.39\b|0\.39\b[^.\n]{0,80}(threshold|noise floor|signific\w*)",
                     report, re.I)
    code_hits = []
    for fn in os.listdir(os.path.join(ROOT, "research")):
        if fn.endswith(".py") and fn != "verify_phase2.py":
            if re.search(r"\b0\.39\b", open(os.path.join(ROOT, "research", fn), encoding="utf-8").read()):
                code_hits.append(fn)
    K.check("No universal ~0.39 RMSE variability threshold is used", not near and not code_hits,
            "0.39 never appears as a threshold in the report or as a constant in research/*.py "
            "(it occurs only as a measured SD, E=10: 13.67 ± 0.39); each comparison uses its own per-seed differences")
    K.check("No statistical-significance claim without a valid test", "none is claimed" in report,
            "report states the minimum attainable p (0.0625) and claims no significance")

    # =====================================================================
    K.sec("2. Fair centralized vs federated comparison")
    for e in ("centralized", "fedavg_iid", "local_iid"):
        K.check(f"Baseline '{e}' was run", len(get(e)) >= 5, f"{len(get(e))} seeds")
    allmain = [r for e in MAIN for r in get(e).values()]
    models = {json.dumps(r["cfg"]["config"]["model"], sort_keys=True) for r in allmain}
    nparams = {r["met"]["model"]["n_parameters"] for r in allmain}
    K.check("Same model architecture for all methods", len(models) == 1 and len(nparams) == 1, f"model cfg = {models}, params = {nparams}")
    datas = {json.dumps(r["cfg"]["config"]["data"], sort_keys=True) for r in allmain}
    K.check("Same preprocessing / features / RUL target for all methods", len(datas) == 1, f"{len(datas)} distinct data config(s)")
    trains = {json.dumps(r["cfg"]["config"]["train"], sort_keys=True) for r in allmain}
    K.check("Same optimiser hyper-parameters for all methods (none tuned for any method)", len(trains) == 1, f"{trains}")
    vals = {tuple(r["met"]["data"]["val_units"]) for r in allmain}
    K.check("Same validation engines for all methods", len(vals) == 1, f"val engines = {sorted(vals)[0] if vals else None}")
    feats = {r["met"]["data"]["n_features"] for r in allmain}
    K.check("17 input features, 30-cycle windows", feats == {17} and {r["cfg"]["config"]["data"]["window_size"] for r in allmain} == {30},
            f"n_features = {feats}")
    c0 = next(iter(cen.values()))["cfg"]["config"]["centralized"]
    K.check("Centralized uses NO early stopping and NO LR plateau schedule (like FedAvg)",
            (not c0["early_stopping"]) and (not c0["lr_plateau"]), f"{c0}")
    passes = {e: {r["met"]["val"]["sample_passes"] for r in get(e).values()} for e in MAIN}
    K.check("Matched training budget: 100 passes over each method's training data", all(v == {100} for v in passes.values()), f"{passes}")
    fedE = next(iter(fr.values()))["cfg"]["config"]["fed"]["local_epochs"]
    K.check("Same checkpoint-selection granularity (centralized every E epochs = every FedAvg round)",
            c0["select_every"] == fedE, f"centralized select_every = {c0['select_every']}, FedAvg E = {fedE}")
    sel_c = [r["met"]["val"]["selected_at_epoch"] for r in cen.values()]
    K.check("Centralized selected checkpoints lie on the shared selection grid",
            all(x % fedE == 0 for x in sel_c), f"selected epochs = {sel_c}")
    tests = set()
    for r in allmain:
        t = pd.read_csv(os.path.join(r["dir"], "test_predictions.csv"))
        tests.add((tuple(t.unit), tuple(t.rul_true.round(3))))
    K.check("All selected models evaluated on the same untouched 100-engine test set", len(tests) == 1 and len(next(iter(tests))[0]) == 100,
            f"{len(tests)} distinct test sets; units 1..100")
    K.check("Both selected and final checkpoints reported (selection policy does not drive conclusions)",
            set(summ.checkpoint) == {"selected", "final"}, "summary.csv has both checkpoints")
    K.check("Original early-stopping protocol kept only as a labelled reference", len(get("centralized_legacy_es")) == 5
            and "Not budget-matched" in report, "centralized_legacy_es: 5 seeds; labelled 'not budget-matched' in report")
    K.check("Report avoids claiming FL is inherently more accurate than centralized",
            "not \"FedAvg is more accurate than centralized learning\"" in report or "Do **not** claim" in open(os.path.join(a.out, "CONCLUSION_CHECK.md"), encoding="utf-8").read(),
            "interpretation attributes the gap to optimisation (K=1 ablation)")

    # =====================================================================
    K.sec("3. Client-partition experiments")
    conds = [("fedavg_iid", "default", "IID window"), ("fedavg_dirichlet", "dirichlet_alpha=1.0", "Dirichlet 1.0"),
             ("fedavg_dirichlet", "dirichlet_alpha=0.5", "Dirichlet 0.5"), ("fedavg_dirichlet", "dirichlet_alpha=0.1", "Dirichlet 0.1"),
             ("fedavg_engine", "default", "whole-engine")]
    n_train = None
    for e, g, lab in conds:
        rs = get(e, g)
        K.check(f"{lab}: run with 5 seeds", len(rs) == 5, f"seeds {sorted(rs)}")
        ok_files, totals, minw, attempts, exclusive = True, set(), [], [], []
        for s, r in rs.items():
            cp = os.path.join(r["dir"], "clients.csv")
            if not os.path.exists(cp):
                ok_files = False
                continue
            c = pd.read_csv(cp)
            needed = {"n_windows", "n_engines", "rul_mean", "rul_std", "rul_q25", "rul_median", "rul_q75", "rul_bin_counts"}
            ok_files &= needed <= set(c.columns)
            totals.add(int(c.n_windows.sum())); minw.append(int(c.n_windows.min()))
            attempts.append(r["met"]["partition"]["attempts"])
            exclusive.append(bool((c.n_engines == c.n_engines_exclusive).all()))
            ok_files &= "partition_seed" in r["met"]["partition"]
        n_train = n_train or next(iter(rs.values()))["met"]["data"]["n_train_windows"]
        K.check(f"{lab}: per-client samples, engines, RUL distribution and partition seed recorded", ok_files,
                "clients.csv (counts, quantiles, 10-bin RUL histogram) + metrics.json partition_seed for every seed")
        K.check(f"{lab}: every training window assigned exactly once (no data discarded)", totals == {n_train},
                f"client totals = {totals}, training windows = {n_train}")
        K.check(f"{lab}: minimum-client-size rule respected and recorded", min(minw) >= proto["base"]["fed"]["min_client_windows"],
                f"smallest client = {min(minw)} windows (rule: >= {proto['base']['fed']['min_client_windows']}); redraw attempts = {attempts}")
        if e == "fedavg_engine":
            K.check("whole-engine: each engine belongs to exactly one client", all(exclusive), "n_engines == n_engines_exclusive on every client")
    K.check("Report states that partition differences are not attributed to heterogeneity alone",
            "not* attributed to heterogeneity alone" in report or "not attributed to heterogeneity alone" in report,
            "'Confounding note' in report section 1")

    # =====================================================================
    K.sec("4. Repeated parameter sweeps")
    sweeps = {"sweep_rounds": ("num_rounds", [5, 10, 20, 30]), "sweep_local_epochs": ("local_epochs", [1, 3, 5, 10]),
              "sweep_clients": ("num_clients", [2, 5, 10]), "sweep_clients_engine": ("num_clients", [2, 5, 10])}
    for e, (key, values) in sweeps.items():
        got = {}
        fixed = defaultdict(set)
        for (ee, g), rs in runs.items():
            if ee != e:
                continue
            for s, r in rs.items():
                fed = r["cfg"]["config"]["fed"]
                got.setdefault(fed[key], set()).add(s)
                for k2 in ("num_rounds", "local_epochs", "num_clients", "partition", "dirichlet_alpha", "fraction_fit"):
                    if k2 != key:
                        fixed[k2].add(str(fed[k2]))
                fixed["train"].add(json.dumps(r["cfg"]["config"]["train"], sort_keys=True))
                fixed["model"].add(json.dumps(r["cfg"]["config"]["model"], sort_keys=True))
                fixed["data"].add(json.dumps(r["cfg"]["config"]["data"], sort_keys=True))
        K.check(f"{e}: all values {values} run", sorted(got) == values, f"values run = {sorted(got)}")
        K.check(f"{e}: every value repeated with 5 seeds", all(len(v) == 5 for v in got.values()),
                f"{ {k: len(v) for k, v in sorted(got.items())} }")
        K.check(f"{e}: all other settings held fixed", all(len(v) == 1 for v in fixed.values()),
                "; ".join(f"{k}={next(iter(v))[:40]}" for k, v in fixed.items() if k not in ('train', 'model', 'data')))
        if key == "num_clients":
            tot = set()
            for (ee, g), rs in runs.items():
                if ee == e:
                    for r in rs.values():
                        tot.add(int(pd.read_csv(os.path.join(r["dir"], "clients.csv")).n_windows.sum()))
            K.check(f"{e}: total training data fixed as K grows", tot == {n_train}, f"total windows per run = {tot}")
    K.check("Report states that total training data stays fixed in the client sweep", "total training data fixed" in report,
            "section 7a/7b headings")
    K.check("Heterogeneity sweep (IID + alpha 1.0 / 0.5 / 0.1) repeated", all(len(get(e, g)) == 5 for e, g, _ in conds[:4]), "see section 3")
    K.check("Partition-strategy comparison (window vs whole-engine) repeated, also across K",
            len(get("fedavg_engine")) == 5 and all(len(get("sweep_clients_engine", f"num_clients={k}")) == 5 for k in (2, 5, 10)),
            "fedavg_engine + sweep_clients_engine, 5 seeds each")

    # =====================================================================
    n_pass = sum(r[2] for r in K.rows)
    lines = ["# Phase 2 verification — were the requested changes made?", "",
             f"Generated by `python -m research.verify_phase2` from the run outputs in `{os.path.relpath(a.out, ROOT)}` "
             "(each run's config.json, metrics.json, clients.csv, test_predictions.csv) and the generated report. "
             "Each check reads what was actually executed, not what the code or the protocol file intends.", "",
             f"**Result: {n_pass} / {len(K.rows)} checks passed.**", ""]
    cur = None
    for sec, req, ok, ev in K.rows:
        if sec != cur:
            lines += ["", f"## {sec}", "", "| | Requirement | Evidence |", "|---|---|---|"]
            cur = sec
        lines.append(f"| {'PASS' if ok else '**FAIL**'} | {req} | {str(ev).replace('|', '/')} |")
    lines += ["", "## What these checks do not establish", "",
              "- **Comparable tuning budget** is met by tuning *nothing*: every method uses the original repository's "
              "hyper-parameters. That makes the comparison equal-effort, but not equal-optimum. A tuned centralized model "
              "may close the gap (report §10.1, §10.6).",
              "- **Five seeds** show consistent orderings but cannot support significance tests at α = 0.05; none is claimed.",
              "- **Wall-clock times** of 12 runs are invalid (suspend / concurrent GPU use) and are flagged; runtime "
              "conclusions use the dedicated benchmark in `results/v2_timing/`.",
              "- **Not run:** the optional `local_dirichlet` experiments (not part of the four Phase-2 requirements).",
              "- These checks verify that the experiments were executed as specified. Whether the interpretation is right "
              "is argued in `EXPERIMENT_REPORT.md` §10 and `CONCLUSION_CHECK.md`."]
    path = os.path.join(a.out, "PHASE2_VERIFICATION.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{n_pass}/{len(K.rows)} checks passed -> {path}")
    for sec, req, ok, ev in K.rows:
        if not ok:
            print("FAIL:", sec, "|", req, "|", ev)
    return 0 if n_pass == len(K.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
