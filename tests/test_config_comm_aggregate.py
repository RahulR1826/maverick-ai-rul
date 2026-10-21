import numpy as np
import pytest

from research import config as C


# ---------------------------------------------------------------- config
def test_protocol_expands_and_dedupes():
    proto = C.load_protocol()
    plan = C.plan_runs(proto, ["fedavg_iid", "sweep_rounds", "sweep_local_epochs", "sweep_clients"], [1, 2])
    # fedavg_iid == R20 == E5 == K5 -> 1 shared config; plus 3 + 3 + 2 others = 9 configs x 2 seeds
    assert len(plan) == 9 * 2
    shared = [e for e in plan if e["experiment"] == "fedavg_iid"]
    assert all(len(e["aliases"]) == 3 for e in shared)


def test_centralized_budget_matches_fedavg():
    proto = C.load_protocol()
    cen = C.expand_experiment(proto, "centralized")[0]["config"]
    assert cen["centralized"]["epochs"] == 20 * 5
    assert cen["centralized"]["select_every"] == 5
    leg = C.expand_experiment(proto, "centralized_legacy_es")[0]["config"]
    assert leg["centralized"]["early_stopping"] and leg["centralized"]["epochs"] == 50


def test_sweeps_vary_only_one_setting():
    proto = C.load_protocol()
    for name, key in (("sweep_rounds", "num_rounds"), ("sweep_local_epochs", "local_epochs"),
                      ("sweep_clients", "num_clients")):
        cfgs = [p["config"] for p in C.expand_experiment(proto, name)]
        for other in ("num_rounds", "local_epochs", "num_clients", "partition"):
            if other != key:
                assert len({c["fed"][other] for c in cfgs}) == 1, (name, other)
        assert len({c["fed"][key] for c in cfgs}) == len(cfgs)


def test_config_hash_changes_with_any_setting():
    proto = C.load_protocol()
    a = C.expand_experiment(proto, "fedavg_iid")[0]
    cfg = dict(a["config"]); cfg["train"] = dict(cfg["train"], learning_rate=0.002)
    assert C.config_hash(cfg) != a["config_hash"]


def test_dirichlet_requires_alpha():
    proto = C.load_protocol()
    base = C.deep_update(proto["base"], {"fed": {"partition": "dirichlet_window"}})
    base["method"] = "fedavg"
    with pytest.raises(ValueError):
        C.resolve_derived(base)


# ---------------------------------------------------------------- comm + aggregation (torch)
def test_payload_matches_flower_serialisation_and_param_count():
    torch = pytest.importorskip("torch")
    from models.lstm_model import build_model
    from research.comm import payload_info, serialized_payload_bytes, state_arrays
    m = build_model(input_size=17)
    info = payload_info(m)
    assert info["n_parameters"] == sum(p.numel() for p in m.parameters())
    assert info["raw_tensor_bytes"] == info["analytic_bytes_params_x4"]
    assert info["serialized_payload_bytes"] > info["raw_tensor_bytes"]   # .npy headers
    try:
        from flwr.common import ndarray_to_bytes
    except Exception:
        pytest.skip("flwr not importable")
    assert serialized_payload_bytes(state_arrays(m)) == sum(len(ndarray_to_bytes(a)) for a in state_arrays(m))


def test_fedavg_traffic_accounting():
    from research.comm import fedavg_traffic
    t = fedavg_traffic(1000, [5, 5, 5])
    assert t["total_upload_bytes"] == t["total_download_bytes"] == 15000
    assert t["per_round"][-1]["cum_total_bytes"] == 30000


def test_fedavg_aggregate_weighted_mean():
    torch = pytest.importorskip("torch")
    from research.trainer import fedavg_aggregate
    s1 = {"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([1.0])}
    s2 = {"w": torch.tensor([4.0, 6.0]), "b": torch.tensor([3.0])}
    out = fedavg_aggregate([s1, s2], [1, 3])
    assert torch.allclose(out["w"], torch.tensor([3.0, 5.0]))
    assert torch.allclose(out["b"], torch.tensor([2.5]))
    same = fedavg_aggregate([s1, s1, s1], [5, 7, 9])
    assert torch.allclose(same["w"], s1["w"])
    with pytest.raises(ValueError):
        fedavg_aggregate([s1, s2], [1, 0])


def test_aggregate_mean_std_and_paired():
    from research.aggregate import paired_differences, summarize
    import pandas as pd
    df = pd.DataFrame({"experiment": ["a"] * 3 + ["b"] * 3, "grid_label": ["default"] * 6,
                       "seed": [1, 2, 3] * 2, "rmse": [10.0, 12.0, 14.0, 11.0, 12.0, 16.0]})
    s = summarize(df, ["experiment", "grid_label"], ["rmse"])
    a = s[s.experiment == "a"].iloc[0]
    assert a["rmse_mean"] == pytest.approx(12.0) and a["rmse_std"] == pytest.approx(2.0) and a["n_seeds"] == 3
    p = paired_differences(df, "rmse", "a", "b")
    assert list(p["diff"]) == [-1.0, 0.0, -2.0]
