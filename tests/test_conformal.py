import numpy as np
import pytest

from serving.conformal import GlobalConformal, MondrianConformal, conformal_quantile, evaluate


def test_finite_sample_quantile():
    s = np.arange(1, 20, dtype=float)           # n = 19
    # ceil((19+1)*0.9) = 18th smallest
    assert conformal_quantile(s, 0.10) == 18.0
    assert conformal_quantile(np.arange(1, 5, dtype=float), 0.10) == np.inf   # too few points for 90%
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)


def _synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0, 125, n)
    sd = rng.uniform(1, 3, n)
    noise_scale = np.where(mu < 30, 1.0, 3.0)    # heteroscedastic by predicted RUL
    y = mu + rng.normal(0, 1, n) * sd * noise_scale
    return y, mu, sd


def test_global_and_mondrian_cover_on_exchangeable_data():
    y, mu, sd = _synthetic(seed=1)
    yt, mt, st = _synthetic(seed=2)
    for cal in (GlobalConformal(0.1), MondrianConformal(0.1)):
        cal.fit(y, mu, sd)
        ev = evaluate(cal, yt, mt, st)
        assert 0.88 <= ev["coverage"] <= 0.93


def test_mondrian_adapts_per_bin():
    y, mu, sd = _synthetic(seed=3)
    yt, mt, st = _synthetic(seed=4)
    g, m = GlobalConformal(0.1).fit(y, mu, sd), MondrianConformal(0.1).fit(y, mu, sd)
    eg, em = evaluate(g, yt, mt, st), evaluate(m, yt, mt, st)
    low_g, low_m = eg["per_bin"][0], em["per_bin"][0]
    assert low_m["mean_width"] < low_g["mean_width"]          # narrower where residuals are small
    assert all(b["coverage"] >= 0.86 for b in em["per_bin"])   # per-bin coverage near target
    assert m.ks[0] < m.ks[-1]


def test_mondrian_falls_back_for_sparse_bins():
    y, mu, sd = _synthetic(n=400, seed=5)
    m = MondrianConformal(0.1, min_bin=10_000).fit(y, mu, sd)
    assert all(k == m.k_global for k in m.ks)
