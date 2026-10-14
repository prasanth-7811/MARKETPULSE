"""Rule-based regime labels and a causally filtered Gaussian HMM."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

TRENDS, VOLS = ["Bullish", "Sideways", "Bearish"], ["Low", "High"]
CLASSES = [f"{t} / {v} Volatility" for t in TRENDS for v in VOLS]
# Okabe-Ito based, colour-blind conscious, all distinct
COLORS = {"Bullish / Low Volatility": "#009E73", "Bullish / High Volatility": "#56B4E9",
          "Sideways / Low Volatility": "#94A3B8", "Sideways / High Volatility": "#E69F00",
          "Bearish / Low Volatility": "#CC79A7", "Bearish / High Volatility": "#D55E00"}
STATE_PALETTE = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#F0E442"]


class RegimeLabeler:
    """Trend (trailing return quantiles) x volatility (vs training median). Thresholds fit on TRAIN only."""

    def __init__(self, window=60, q=(0.3, 0.7)): self.window, self.q = window, tuple(q)

    def signals(self, price: pd.Series) -> pd.DataFrame:
        r = np.log(price).diff()
        return pd.DataFrame({"sig_mom": price.pct_change(self.window),
                             "sig_vol": r.rolling(self.window).std() * np.sqrt(252)})

    def fit(self, sig: pd.DataFrame):
        s = sig.dropna()
        self.lo, self.hi = s["sig_mom"].quantile(self.q[0]), s["sig_mom"].quantile(self.q[1])
        self.vol_thr = s["sig_vol"].median()
        return self

    def label(self, sig: pd.DataFrame) -> pd.Series:
        trend = pd.Series(np.where(sig["sig_mom"] > self.hi, "Bullish", np.where(sig["sig_mom"] < self.lo, "Bearish", "Sideways")), index=sig.index)
        vol = pd.Series(np.where(sig["sig_vol"] > self.vol_thr, "High", "Low"), index=sig.index)
        lab = trend + " / " + vol + " Volatility"
        return lab.where(sig.notna().all(axis=1))

    def describe(self) -> dict:
        return {"bearish_below_momentum": float(self.lo), "bullish_above_momentum": float(self.hi),
                "high_vol_above": float(self.vol_thr), "window": self.window}


def _full_covs(m) -> np.ndarray:
    c, K, n = np.asarray(m.covars_), m.n_components, m.n_features
    if m.covariance_type == "full": return c
    if m.covariance_type == "tied": return np.stack([c] * K)
    if m.covariance_type == "diag": return np.stack([np.diag(c[k]) for k in range(K)])
    return np.stack([np.eye(n) * c[k] for k in range(K)])  # spherical


class FilteredHMM:
    """GaussianHMM whose state probabilities are FILTERED (forward algorithm): P(s_t | x_1..x_t).
    hmmlearn's predict_proba is smoothed (uses future data) and is deliberately not used."""

    def __init__(self, n_states=3, covariance_type="full", n_iter=200, seed=42):
        self.K, self.ct, self.n_iter, self.seed = n_states, covariance_type, n_iter, seed

    def fit(self, X: np.ndarray):
        from hmmlearn.hmm import GaussianHMM
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z, best = (X - self.mu) / self.sd, None
        for s in range(4):  # restarts: EM is sensitive to initialisation
            try:
                m = GaussianHMM(self.K, covariance_type=self.ct, n_iter=self.n_iter, min_covar=1e-4, random_state=self.seed + s).fit(Z)
                sc = m.score(Z)
                if best is None or sc > best[0]: best = (sc, m)
            except Exception:
                continue
        if best is None: raise RuntimeError("HMM failed to converge for every restart.")
        self.train_loglik, self.model = best
        return self

    @property
    def transmat(self) -> np.ndarray: return self.model.transmat_

    def filter(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (filtered P(s_t|x_<=t), predictive P(s_{t+1}|x_<=t))."""
        Z, m = (X - self.mu) / self.sd, self.model
        covs = _full_covs(m)
        logB = np.column_stack([multivariate_normal(m.means_[k], covs[k], allow_singular=True).logpdf(Z) for k in range(self.K)])
        A, prior = m.transmat_, m.startprob_
        filt, nxt = np.zeros((len(Z), self.K)), np.zeros((len(Z), self.K))
        for t in range(len(Z)):
            a = prior * np.exp(logB[t] - logB[t].max())
            s = a.sum()
            a = a / s if s > 0 and np.isfinite(s) else np.full(self.K, 1 / self.K)
            filt[t], prior = a, a @ A
            nxt[t] = prior
        return filt, nxt


def label_states(stats: pd.DataFrame) -> list[str]:
    """Descriptive names from TRAINING-period state statistics (cols: mom, vol). State ids are arbitrary.
    Trend is assigned by rank of mean momentum (lowest -> Bearish, highest -> Bullish, others Sideways);
    volatility is High when above the median state volatility."""
    K = len(stats)
    order = stats["mom"].rank(method="first").astype(int).to_numpy() - 1
    vthr = stats["vol"].median() if K > 1 else np.inf
    names = []
    for rk, v in zip(order, stats["vol"]):
        tr = "Bullish" if rk == K - 1 else "Bearish" if rk == 0 else "Sideways"
        if K == 1: tr = "Sideways"
        names.append(f"{tr} / {'High' if v > vthr else 'Low'} Volatility")
    return [n if names.count(n) == 1 else f"{n} (S{i})" for i, n in enumerate(names)]


def run_lengths(s: pd.Series) -> pd.DataFrame:
    grp = (s != s.shift()).cumsum()
    return s.groupby(grp).agg(regime="first", duration="size")
