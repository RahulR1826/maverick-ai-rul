import os

import numpy as np
import pandas as pd
import pytest

from research import config as C
from research.data import (WindowSet, add_train_rul, get_data, make_test_windows, make_windows,
                           split_engines)
from research.partition import (PartitionError, check_partition, client_stats, derive_seed,
                                partition, rul_bins)

HAVE_DATA = os.path.exists(os.path.join(C.DATA_DIR, "train_FD001.txt"))
BASE = C.load_protocol()["base"]


def _toy_df(lengths):
    rows = []
    for u, T in enumerate(lengths, start=1):
        for c in range(1, T + 1):
            rows.append({"unit": u, "cycle": c, "f": float(c)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- preprocessing
def test_rul_target_piecewise_linear():
    df = add_train_rul(_toy_df([200]), rul_cap=125)
    assert df["rul"].iloc[-1] == 0
    assert df["rul"].iloc[-2] == 1
    assert df["rul"].max() == 125
    assert (df["rul"].iloc[:75] == 125).all()   # 199..125 capped


def test_windows_label_is_last_step_and_count():
    df = add_train_rul(_toy_df([40, 35]), rul_cap=125)
    ws = make_windows(df, ["f"], window=30, stride=1)
    assert len(ws) == (40 - 30 + 1) + (35 - 30 + 1)
    assert ws.X.shape == (17, 30, 1)
    # first window of engine 1 ends at cycle 30 -> RUL 10
    assert ws.cycle[0] == 30 and ws.y[0] == 10 and ws.X[0, -1, 0] == 30
    assert ws.y[16] == 0 and ws.unit[16] == 2


def test_short_engine_is_an_error_not_silently_dropped():
    df = add_train_rul(_toy_df([40, 20]), rul_cap=125)
    with pytest.raises(ValueError):
        make_windows(df, ["f"], window=30, stride=1)


def test_test_windows_take_last_cycles_and_cap():
    df = _toy_df([50, 31])
    ws, unc = make_test_windows(df, np.array([140.0, 20.0]), ["f"], 30, rul_cap=125)
    assert ws.X[0, -1, 0] == 50 and ws.X[0, 0, 0] == 21
    assert list(ws.y) == [125.0, 20.0] and list(unc) == [140.0, 20.0]


def test_engine_split_matches_original_rng_and_is_disjoint():
    units = list(range(1, 101))
    tr, va = split_engines(units, 0.10, 42)
    assert len(va) == 10 and len(tr) == 90 and not set(tr) & set(va)
    rng = np.random.default_rng(42)
    assert sorted(rng.choice(units, size=10, replace=False).tolist()) == va


@pytest.mark.skipif(not HAVE_DATA, reason="FD001 not present")
def test_fd001_protocol_and_no_leakage():
    d = get_data(BASE["data"], C.DATA_DIR)
    assert d.meta["n_features"] == 17
    assert d.train.X.shape[1:] == (30, 17)
    assert len(d.train_units) == 90 and len(d.val_units) == 10 and len(d.test) == 100
    assert not set(np.unique(d.train.unit)) & set(np.unique(d.val.unit))
    # scaler fitted on train engines only -> train features exactly span [0, 1]
    flat = d.train.X.reshape(-1, 17)
    nonconst = [i for i, c in enumerate(d.feature_cols) if c not in d.meta["constant_train_features"]]
    assert np.allclose(flat[:, nonconst].min(0), 0, atol=1e-6)
    assert np.allclose(flat[:, nonconst].max(0), 1, atol=1e-6)


@pytest.mark.skipif(not HAVE_DATA, reason="FD001 not present")
def test_fd001_matches_original_cached_arrays():
    proc = os.path.join(C.BASE_DIR, "processed", "FD001")
    if not os.path.exists(os.path.join(proc, "X_train.npy")):
        pytest.skip("original processed cache absent")
    d = get_data(BASE["data"], C.DATA_DIR)
    for split, ws in (("train", d.train), ("val", d.val), ("test", d.test)):
        X0, y0 = np.load(os.path.join(proc, f"X_{split}.npy")), np.load(os.path.join(proc, f"y_{split}.npy"))
        assert X0.shape == ws.X.shape
        assert np.allclose(X0, ws.X, atol=1e-5) and np.allclose(y0, ws.y)


# ---------------------------------------------------------------- partitioning
def _toy_windows(n_eng=20, per=50):
    unit = np.repeat(np.arange(1, n_eng + 1), per).astype(np.int32)
    y = np.tile(np.linspace(125, 0, per), n_eng).astype(np.float32)
    X = np.zeros((len(y), 2, 1), np.float32)
    return WindowSet(X, y, unit, np.zeros_like(unit))


def _fed(kind, k=5, alpha=None, min_n=1):
    return {"num_clients": k, "partition": kind, "dirichlet_alpha": alpha, "n_rul_bins": 10,
            "min_client_windows": min_n, "max_partition_attempts": 100}


@pytest.mark.parametrize("kind,alpha", [("iid_window", None), ("dirichlet_window", 0.5),
                                        ("dirichlet_window", 0.1), ("engine", None)])
def test_partitions_cover_everything_once_and_are_seeded(kind, alpha):
    ws = _toy_windows()
    a = partition(ws, _fed(kind, alpha=alpha), 125, seed=7)
    b = partition(ws, _fed(kind, alpha=alpha), 125, seed=7)
    c = partition(ws, _fed(kind, alpha=alpha), 125, seed=8)
    check_partition(a["indices"], len(ws))
    assert all(np.array_equal(x, y) for x, y in zip(a["indices"], b["indices"]))
    assert not all(np.array_equal(x, y) for x, y in zip(a["indices"], c["indices"]))


def test_engine_partition_has_exclusive_ownership():
    ws = _toy_windows()
    p = partition(ws, _fed("engine", k=4), 125, seed=1)
    owners = [set(np.unique(ws.unit[i])) for i in p["indices"]]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not owners[i] & owners[j]
    stats = client_stats(ws, p["indices"], 125, 10)
    assert all(s["n_engines"] == s["n_engines_exclusive"] == 5 for s in stats)


def test_iid_window_spreads_engines_and_equal_sizes():
    ws = _toy_windows()
    p = partition(ws, _fed("iid_window", k=4), 125, seed=1)
    sizes = [len(i) for i in p["indices"]]
    assert max(sizes) - min(sizes) <= 1
    stats = client_stats(ws, p["indices"], 125, 10)
    assert all(s["n_engines_exclusive"] < s["n_engines"] for s in stats)


def test_min_client_size_redraw_and_failure():
    ws = _toy_windows()
    ok = partition(ws, _fed("dirichlet_window", alpha=0.1, min_n=50), 125, seed=3)
    assert min(len(i) for i in ok["indices"]) >= 50
    with pytest.raises(PartitionError):
        partition(ws, _fed("iid_window", k=5, min_n=10_000), 125, seed=3)
    with pytest.raises(PartitionError):
        partition(ws, _fed("engine", k=50), 125, seed=3)


def test_dirichlet_skew_increases_as_alpha_decreases():
    ws = _toy_windows(n_eng=60)
    def spread(alpha):
        vals = []
        for s in range(10):
            p = partition(ws, _fed("dirichlet_window", alpha=alpha, min_n=1), 125, seed=s)
            vals.append(np.std([ws.y[i].mean() for i in p["indices"] if len(i)]))
        return np.mean(vals)
    assert spread(0.1) > spread(100.0)


def test_rul_bins_edges():
    b = rul_bins(np.array([0.0, 12.49, 12.5, 124.9, 125.0]), 125, 10)
    assert list(b) == [0, 0, 1, 9, 9]


def test_derive_seed_independent_streams():
    assert derive_seed(42, "init") != derive_seed(42, "train")
    assert derive_seed(42, "init") == derive_seed(42, "init")
    assert derive_seed(42, "init") != derive_seed(43, "init")
