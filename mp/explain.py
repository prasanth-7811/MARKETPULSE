"""Explainability: SHAP for tree models, permutation importance fallback. Associations, not causation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

TREE = ("Random Forest", "XGBoost", "LightGBM")


def _norm(sv, n_rows: int, n_feat: int):
    """Return SHAP array as (rows, features, classes) across shap versions."""
    if isinstance(sv, list): sv = np.stack(sv, axis=2)
    sv = np.asarray(sv)
    if sv.ndim == 2: sv = sv[:, :, None]
    return sv


def shap_for(clf, X: pd.DataFrame, binary_positive: bool = False):
    """SHAP values for a Clf wrapping a tree model. Returns (values[rows,feat,present_classes], clf.present)."""
    import shap
    Xs = X[clf.cols]
    sv = _norm(shap.TreeExplainer(clf.est).shap_values(Xs), len(Xs), len(clf.cols))
    if binary_positive and sv.shape[2] == 2: sv = sv[:, :, 1:2]
    return sv, clf.present


def perm_importance(model, X: pd.DataFrame, y: np.ndarray, seed: int = 0, repeats: int = 5) -> pd.Series:
    class _W:  # sklearn-compatible shim exposing predict() on the DataFrame
        def __init__(s, m): s.m = m
        def fit(s, *a): return s
        def predict(s, Xd): return s.m.predict(pd.DataFrame(Xd, columns=model.cols))
    r = permutation_importance(_W(model), X[model.cols].to_numpy(), y, n_repeats=repeats, random_state=seed,
                               scoring=lambda e, Xd, yd: f1_score(yd, e.predict(Xd), average="macro", zero_division=0))
    return pd.Series(r.importances_mean, index=model.cols).sort_values(ascending=False)


def subsample(idx: np.ndarray, k: int = 400) -> np.ndarray:
    return idx if len(idx) <= k else idx[np.linspace(0, len(idx) - 1, k).astype(int)]
