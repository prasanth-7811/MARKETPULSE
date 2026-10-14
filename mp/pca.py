"""Dimensionality reduction: PCA on engineered features.

Provides explained-variance analysis, feature loadings, and regime-coloured
projections so the latent factors driving the feature space are visible.
PCA is fit on the training split only (no look-ahead); the same rotation is
applied to validation/test rows for fair comparison."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca(X: pd.DataFrame) -> PCA:
    """Fit PCA on the engineered feature matrix."""
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return PCA(n_components=min(X.shape), svd_solver="full").fit(StandardScaler().fit_transform(X))


def explained_variance(pca: PCA, top_k: int = 12) -> pd.DataFrame:
    """Top components by explained variance + cumulative share."""
    ev = pca.explained_variance_ratio_
    df = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(len(ev))],
        "Explained variance ratio": np.round(ev, 4),
        "Cumulative explained variance": np.round(np.cumsum(ev), 4),
    }).sort_values("Explained variance ratio", ascending=False)
    return df.head(top_k)


def cumulative_components(pca: PCA, threshold: float = 0.90) -> int:
    """Minimum number of PCs needed to explain `threshold` of variance."""
    ev = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(ev, threshold) + 1) or 1


def feature_loadings(pca: PCA, feature_names: list[str], n_top: int = 12) -> dict:
    """Absolute loading of each feature on each component."""
    load = np.abs(pca.components_)
    return {
        f"PC{i + 1}": pd.DataFrame({
            "Feature": [feature_names[j] for j in np.argsort(load[i])[::-1][:n_top]],
            "Loading (abs)": np.round(load[i, np.argsort(load[i])[::-1][:n_top]], 4),
        })
        for i in range(pca.n_components_)
    }


def projection(X: pd.DataFrame, pca: PCA, regime: np.ndarray | None, n: int = 600) -> pd.DataFrame:
    """Projection of observations onto PC1/PC2, optionally labelled by regime."""
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    idx = np.arange(len(X))
    if len(idx) > n:
        idx = np.linspace(0, len(idx) - 1, n, dtype=int)
    proj = pd.DataFrame(pca.transform(X).iloc[idx, :2], columns=["PC1", "PC2"], index=X.index[idx])
    if regime is not None:
        proj["Regime"] = np.asarray(regime)[idx]
    return proj


def scree(pca: PCA) -> pd.DataFrame:
    """Bar-plot data: component index vs explained variance ratio."""
    ev = pca.explained_variance_ratio_
    return pd.DataFrame({"Component": np.arange(1, len(ev) + 1), "Explained variance ratio": ev})
