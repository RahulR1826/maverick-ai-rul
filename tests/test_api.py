"""API tests: every served number must come from the real models and result tables."""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from research import config as C

HAVE = os.path.exists(os.path.join(C.DATA_DIR, "test_FD001.txt")) and os.path.isdir(
    os.path.join(C.BASE_DIR, "checkpoints", "v2", "fedavg-3985e8aca1-s42"))
pytestmark = pytest.mark.skipif(not HAVE, reason="needs FD001 data and v2 checkpoints")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import api_server
    with TestClient(api_server.app) as c:
        yield c


def test_fleet_is_the_real_test_set(client):
    r = client.get("/api/fleet")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 100 and sorted(x["engine_id"] for x in rows) == list(range(1, 101))
    truth = pd.read_csv(os.path.join(C.DATA_DIR, "RUL_FD001.txt"), header=None, sep=r"\s+").iloc[:, 0].to_numpy()
    got = np.array([x["true_rul_uncapped"] for x in sorted(rows, key=lambda x: x["engine_id"])])
    assert np.array_equal(got, truth)
    assert all(x["rul_lower"] <= x["rul_mean"] <= x["rul_upper"] for x in rows)


def test_fleet_is_deterministic(client):
    a, b = client.get("/api/fleet").json(), client.get("/api/fleet").json()
    assert a == b                       # no random resampling between requests


def test_members_reproduce_paper_rmse(client):
    m = client.get("/api/meta").json()["model"]["test"]
    s = pd.read_csv(os.path.join(C.BASE_DIR, "results", "v2", "tables", "summary.csv"))
    paper = s[(s.experiment == "fedavg_iid") & (s.checkpoint == "selected")].rmse_mean.iloc[0]
    assert m["members_last_cycle_rmse_mean"] == pytest.approx(paper, abs=1e-3)


def test_risk_uses_lower_bound_and_thresholds(client):
    rows = client.get("/api/fleet", params={"critical": 30, "high": 60, "medium": 90}).json()
    for x in rows:
        lo = x["rul_lower"]
        exp = "Critical" if lo < 30 else "High" if lo < 60 else "Medium" if lo < 90 else "Low"
        assert x["risk"] == exp
    assert client.get("/api/fleet", params={"critical": 50, "high": 10}).status_code == 422


def test_engine_trajectory_and_truth(client):
    e = client.get("/api/engines/34").json()
    tr = e["trajectory"]
    assert tr[0]["cycle"] == 30 and len(tr) == len(e["observed_cycles"]) - 29
    # ground truth decreases by one per cycle (until capped)
    t = np.array([p["true_rul"] for p in tr])
    assert np.all(np.diff(t)[t[1:] < 125] == -1)
    assert len(e["sensors"]) == 17
    assert client.get("/api/engines/999").status_code == 404


def test_explain_completeness(client):
    x = client.get("/api/engines/34/explain", params={"cycle": 100}).json()
    assert abs(x["completeness_error"]) < 0.5
    assert len(x["features"]) == 17 and len(x["heatmap"]["values"]) == 17
    assert client.get("/api/engines/34/explain", params={"cycle": 5}).status_code == 422


def test_results_endpoints(client):
    for path in ("/api/results/summary", "/api/results/convergence", "/api/results/partitions",
                 "/api/results/paired", "/api/results/runtime", "/api/results/per_seed"):
        r = client.get(path)
        assert r.status_code == 200 and len(r.json()) > 0, path


# ---------------------------------------------------------------- v2.1 features
def _engine_rows(unit):
    rows = [l for l in open(os.path.join(C.DATA_DIR, "test_FD001.txt")).read().splitlines() if l.split() and int(l.split()[0]) == unit]
    return "\n".join(rows)


def test_upload_reproduces_builtin_engine(client):
    r = client.post("/api/uploads", files={"file": ("engine34.txt", _engine_rows(34), "text/plain")})
    assert r.status_code == 200, r.text
    up = r.json()["uploads"]
    assert len(up) == 1
    a = client.get(f"/api/engines/{up[0]['upload_id']}").json()
    b = client.get("/api/engines/34").json()
    # identical pipeline: same predictions and the same out-of-distribution flags as the built-in engine
    # (engine 34 has one W31 reading just outside the training range, which is correctly flagged)
    assert [p["rul_mean"] for p in a["trajectory"]] == pytest.approx([p["rul_mean"] for p in b["trajectory"]], abs=1e-6)
    assert [p["ood_fraction"] for p in a["trajectory"]] == [p["ood_fraction"] for p in b["trajectory"]]
    assert up[0]["max_ood_fraction"] == max(p["ood_fraction"] for p in b["trajectory"])
    assert a["has_ground_truth"] is False and b["has_ground_truth"] is True


def test_upload_flags_out_of_distribution_and_rejects_bad_files(client):
    rows = [l.split() for l in _engine_rows(34).splitlines()]
    for r_ in rows[-10:]:
        r_[11] = str(float(r_[11]) * 1.5)            # s_7 far outside the training range
    r = client.post("/api/uploads", files={"file": ("ood.txt", "\n".join(" ".join(r_) for r_ in rows), "text/plain")})
    assert r.status_code == 200 and r.json()["uploads"][0]["max_ood_fraction"] > 0 and r.json()["warning"]
    assert client.post("/api/uploads", files={"file": ("bad.txt", "1 2 3\n4 5 6", "text/plain")}).status_code == 422
    short = "\n".join(_engine_rows(34).splitlines()[:10])
    assert client.post("/api/uploads", files={"file": ("short.txt", short, "text/plain")}).status_code == 422
    assert client.get("/api/engines/u-doesnotexist").status_code == 404


def test_compare_has_all_families(client):
    c = client.get("/api/engines/34/compare").json()
    assert set(c["families"]) == {"fedavg_iid", "fedavg_engine", "centralized"}
    for f in c["families"].values():
        assert len(f["rul_mean"]) == len(c["cycles"]) == len(c["true_rul"])
        assert all(lo <= m <= hi for lo, m, hi in zip(f["rul_lower"], f["rul_mean"], f["rul_upper"]))


def test_conformal_interval_reported(client):
    m = client.get("/api/meta").json()
    assert "Mondrian" in m["interval"]["method"]
    assert len(m["interval"]["calibration"]["bins"]) == 4
    assert 0.8 <= m["model"]["test"]["interval_coverage_all_cycles"] <= 1.0


def test_attribution_series_when_precomputed(client):
    r = client.get("/api/engines/34/attributions")
    assert r.status_code in (200, 409)
    if r.status_code == 200:
        d = r.json()
        assert len(d["values"]) == len(d["cycles"]) and len(d["values"][0]) == 17
        x = client.get("/api/engines/34/explain").json()
        assert x["source"] == "precomputed cache" and abs(x["completeness_error"]) < 1.5


def test_live_training_stream(client):
    events = []
    with client.stream("GET", "/api/train/stream", params={"num_clients": 2, "num_rounds": 2, "local_epochs": 1}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("event: "):
                events.append(line[7:])
    assert events[0] == "start" and events.count("round") == 2 and events[-1] == "done"
    assert client.get("/api/train/stream", params={"num_rounds": 50}).status_code == 422
