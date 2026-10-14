"""Estimators, class-aligned wrappers, ensembles, calibration and metrics."""
from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALGOS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
DEFAULTS = {
    "Logistic Regression": dict(C=1.0),
    "Random Forest": dict(n_estimators=300, max_depth=8, min_samples_leaf=5),
    "XGBoost": dict(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    "LightGBM": dict(n_estimators=250, num_leaves=15, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20),
    "Transition": dict(n_estimators=200, num_leaves=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_samples=30),
}
GRIDS = {
    "Logistic Regression": dict(C=[0.01, 0.1, 1, 10, 100]),
    "Random Forest": dict(n_estimators=[150, 300, 500], max_depth=[4, 6, 8, 12], min_samples_leaf=[2, 5, 10, 20]),
    "XGBoost": dict(n_estimators=[150, 250, 400], max_depth=[2, 3, 4, 6], learning_rate=[0.02, 0.05, 0.1], subsample=[0.7, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0]),
    "LightGBM": dict(n_estimators=[150, 250, 400], num_leaves=[7, 15, 31], learning_rate=[0.02, 0.05, 0.1], subsample=[0.7, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], min_child_samples=[10, 20, 40]),
}


def make_estimator(name: str, seed: int, params: dict | None = None):
    p = {**DEFAULTS[name], **(params or {})}
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed, **p))
    if name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **p)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(eval_metric="mlogloss", random_state=seed, n_jobs=-1, verbosity=0, **p)
    from lightgbm import LGBMClassifier  # LightGBM / Transition
    return LGBMClassifier(subsample_freq=1, random_state=seed, verbose=-1, n_jobs=-1, **p)


class Clf:
    """Wraps an estimator so probabilities always span ALL n_classes (absent training classes get 0)."""

    def __init__(self, name, est, cols, n_classes): self.name, self.est, self.cols, self.n_classes = name, est, list(cols), n_classes

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.present = np.unique(y)
        if len(self.present) < 2: raise ValueError("Training window contains a single class.")
        self.est.fit(X[self.cols], np.searchsorted(self.present, y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X), self.n_classes))
        out[:, self.present] = self.est.predict_proba(X[self.cols])
        return out

    def predict(self, X): return self.predict_proba(X).argmax(1)


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    if abs(T - 1) < 1e-9: return P
    lp = np.log(np.clip(P, 1e-9, 1)) / T
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling fitted on VALIDATION rows only (minimises log loss)."""
    nll = lambda T: -np.mean(np.log(np.clip(apply_temperature(P, T)[np.arange(len(y)), y], 1e-9, 1)))
    return float(minimize_scalar(nll, bounds=(0.3, 4.0), method="bounded").x)


class Ens:
    """Soft-voting ensemble (mean of member probabilities) with optional temperature calibration."""

    def __init__(self, members: list[Clf], temperature: float = 1.0):
        self.members, self.T = members, temperature
        self.cols = list(dict.fromkeys(c for m in members for c in m.cols))
        self.n_classes = members[0].n_classes

    def predict_proba(self, X): return apply_temperature(np.mean([m.predict_proba(X) for m in self.members], 0), self.T)
    def predict(self, X): return self.predict_proba(X).argmax(1)


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, ok = P.max(1), P.argmax(1) == y
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): tot += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(tot)


def multiclass_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    n = P.shape[1]
    Pn = np.clip(P, 1e-6, 1); Pn = Pn / Pn.sum(1, keepdims=True)
    pred = P.argmax(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                    macro_f1=f1_score(y, pred, average="macro", zero_division=0),
                    weighted_f1=f1_score(y, pred, average="weighted", zero_division=0),
                    log_loss=log_loss(y, Pn, labels=list(range(n))),
                    brier=float(np.mean(np.sum((Pn - np.eye(n)[y]) ** 2, axis=1))), ece=ece(P, y), n=len(y))


class ProbCal:
    """Binary probability calibrator fitted on validation predictions.

    method: none | platt | isotonic | auto. `auto` picks the method with the lowest Brier score under a
    forward-chained split of the VALIDATION rows (fit on the first half, score on the second), then refits on all
    validation rows. Calibrating on a window that differs from deployment can hurt, so 'none' is a candidate.
    """

    def __init__(self, method: str = "isotonic"): self.method, self.note, self.requested = method, "", method

    def fit(self, p, y):
        p, y = np.asarray(p), np.asarray(y)
        if self.method != "none" and (y.sum() < 20 or (1 - y).sum() < 20):
            self.note = "Too few validation events/non-events to calibrate; raw probabilities used."
            self.method = "none"
        if self.method == "auto":
            h = len(p) // 2; best = ("none", brier_score_loss(y[h:], np.clip(p[h:], 0, 1)))
            if 0 < y[:h].sum() and y[:h].sum() < h and len(np.unique(y[h:])) == 2:
                for m in ("platt", "isotonic"):
                    q = ProbCal(m).fit(p[:h], y[:h]).transform(p[h:]); b = brier_score_loss(y[h:], np.clip(q, 0, 1))
                    if b < best[1] - 1e-4: best = (m, b)
            self.method = best[0]; self.note = f"auto-selected '{best[0]}' on validation (forward-chained Brier)."
        if self.method == "isotonic": self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        elif self.method == "platt": self.m = LogisticRegression().fit(self._lg(p), y)
        return self

    @staticmethod
    def _lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, p):
        if self.method == "isotonic": return self.m.predict(p)
        if self.method == "platt": return self.m.predict_proba(self._lg(p))[:, 1]
        return np.asarray(p)


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    two = len(np.unique(y)) == 2
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return dict(n=len(y), events=int(y.sum()), prevalence=float(y.mean()) if len(y) else np.nan,
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                pr_auc=average_precision_score(y, p) if two else np.nan,
                roc_auc=roc_auc_score(y, p) if two else np.nan,
                brier=brier_score_loss(y, np.clip(p, 0, 1)), confusion=cm)


def hyper_search(name, X, y, cols, n_classes, n_iter, cv_splits, seed) -> pd.DataFrame:
    """Random search scored by walk-forward (expanding window) CV on TRAINING rows only."""
    grid = GRIDS[name]
    size = int(np.prod([len(v) for v in grid.values()]))
    rows = []
    for i, params in enumerate(ParameterSampler(grid, min(n_iter, size), random_state=seed)):
        sc, t0 = [], time.time()
        for tr, va in TimeSeriesSplit(cv_splits).split(X):
            try:
                m = Clf(name, make_estimator(name, seed, params), cols, n_classes).fit(X.iloc[tr], y[tr])
                sc.append(f1_score(y[va], m.predict(X.iloc[va]), average="macro", zero_division=0))
            except ValueError:
                continue
        rows.append(dict(trial=i, **params, cv_macro_f1=float(np.mean(sc)) if sc else np.nan,
                         cv_std=float(np.std(sc)) if sc else np.nan, seconds=round(time.time() - t0, 2)))
    return pd.DataFrame(rows)
