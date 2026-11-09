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
