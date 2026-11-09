"""
serving/conformal.py
--------------------
Prediction intervals for the ensemble via split conformal prediction.

Score: normalised residual  s = |y - mu| / max(sd, eps), where mu and sd are the
ensemble mean and standard deviation (Lei et al., 2018; "normalised" conformal).

Two calibrators are provided:

* ``GlobalConformal``   one scale k for all predictions (the original method).
* ``MondrianConformal`` one scale per bin of *predicted* RUL (Vovk et al., 2005,
  "Mondrian" / group-conditional conformal). Residuals near failure behave
  differently from residuals on the healthy plateau, so per-bin calibration gives
  intervals whose coverage holds within each RUL range rather than only on
  average, and that are narrower where the model is reliably accurate.

Both use the finite-sample quantile ceil((n+1)(1-alpha))/n, which gives
marginal coverage >= 1-alpha *if calibration and test points are exchangeable*.
Here the calibration set is the windows of the 10 held-out validation engines.
Windows of one engine are correlated, and the same engines were used for
checkpoint selection, so the guarantee is approximate; coverage on the test
engines is therefore measured and reported, never assumed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

EPS = 1e-6


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of nonconformity scores."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf) if rank > n else float(s[rank - 1])


def scores(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y) - np.asarray(mu)) / np.maximum(np.asarray(sd), EPS)


@dataclass
class GlobalConformal:
    alpha: float
    k: float = float("nan")
    n: int = 0

    def fit(self, y, mu, sd) -> "GlobalConformal":
        sc = scores(y, mu, sd)
        self.k, self.n = conformal_quantile(sc, self.alpha), len(sc)
        return self

    def scale(self, mu: np.ndarray) -> np.ndarray:
        return np.full(np.shape(mu), self.k, dtype=float)

    def describe(self) -> Dict:
        return {"method": "global normalised split conformal", "alpha": self.alpha, "k": self.k, "n_calibration": self.n}


@dataclass
class MondrianConformal:
    """Per-bin scale k_b, bins defined on the predicted RUL (known at prediction time)."""
    alpha: float
    edges: Sequence[float] = (30.0, 60.0, 90.0)
    min_bin: int = 100
    ks: List[float] = field(default_factory=list)
    ns: List[int] = field(default_factory=list)
    k_global: float = float("nan")

    def bins(self, mu: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(mu, dtype=float), self.edges)

    def fit(self, y, mu, sd) -> "MondrianConformal":
        sc = scores(y, mu, sd)
        b = self.bins(mu)
        self.k_global = conformal_quantile(sc, self.alpha)
        self.ks, self.ns = [], []
        for i in range(len(self.edges) + 1):
            s_i = sc[b == i]
            self.ns.append(int(len(s_i)))
            # too few calibration points in a bin: fall back to the global scale
            self.ks.append(conformal_quantile(s_i, self.alpha) if len(s_i) >= self.min_bin else self.k_global)
        return self

    def scale(self, mu: np.ndarray) -> np.ndarray:
        return np.asarray(self.ks, dtype=float)[self.bins(mu)]

    def describe(self) -> Dict:
        labels = [f"< {self.edges[0]:g}"] + [f"{a:g}–{b:g}" for a, b in zip(self.edges[:-1], self.edges[1:])] + [f">= {self.edges[-1]:g}"]
        return {"method": "Mondrian (per predicted-RUL bin) normalised split conformal", "alpha": self.alpha,
                "bins": [{"predicted_rul": l, "k": k, "n_calibration": n, "fallback_to_global": n < self.min_bin}
                         for l, k, n in zip(labels, self.ks, self.ns)],
                "k_global": self.k_global}


def evaluate(cal, y: np.ndarray, mu: np.ndarray, sd: np.ndarray, edges: Sequence[float] = (30.0, 60.0, 90.0)) -> Dict:
    """Coverage and width overall and per predicted-RUL bin (for reporting only, never for fitting)."""
    half = cal.scale(mu) * np.maximum(sd, EPS)
    lo, hi = np.clip(mu - half, 0, None), mu + half
    inside = (y >= lo) & (y <= hi)
    b = np.digitize(mu, edges)
    labels = [f"< {edges[0]:g}"] + [f"{a:g}–{c:g}" for a, c in zip(edges[:-1], edges[1:])] + [f">= {edges[-1]:g}"]
    per = [{"predicted_rul": l, "n": int((b == i).sum()),
            "coverage": float(inside[b == i].mean()) if (b == i).any() else None,
            "mean_width": float((hi - lo)[b == i].mean()) if (b == i).any() else None} for i, l in enumerate(labels)]
    return {"coverage": float(inside.mean()), "mean_width": float((hi - lo).mean()), "n": int(len(y)), "per_bin": per}
