"""Dimensionality reduction for feature diagnostics and as a modelling option.

PCA is fitted on TRAINING rows only (scaler + components) and applied to the remaining rows, so no
information from validation/test leaks into the projection. t-SNE is a visual diagnostic only
(non-parametric, transductive) and is never used as model input.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class PCAReducer:
    def __init__(self, variance: float = 0.95, clip: float = 5.0, seed: int = 42):
        self.variance, self.clip, self.seed = variance, clip, seed

    def fit(self, X: pd.DataFrame):
        self.cols = list(X.columns)
        self.sc = StandardScaler().fit(X)
        Z = np.clip(self.sc.transform(X), -self.clip, self.clip)
        full = PCA(random_state=self.seed).fit(Z)
        self.k = int(min(np.searchsorted(np.cumsum(full.explained_variance_ratio_), self.variance - 1e-12) + 1, len(self.cols)))
        self.pca = PCA(n_components=self.k, random_state=self.seed).fit(Z)
        self.explained_ = full.explained_variance_ratio_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = np.clip(self.sc.transform(X[self.cols]), -self.clip, self.clip)
        return pd.DataFrame(self.pca.transform(Z), index=X.index, columns=[f"PC{i + 1}" for i in range(self.k)])

    def loadings(self, n: int = 4) -> pd.DataFrame:
        return pd.DataFrame(self.pca.components_[:n].T, index=self.cols, columns=[f"PC{i + 1}" for i in range(n)])

    def scree(self) -> pd.DataFrame:
        return pd.DataFrame({"component": np.arange(1, len(self.explained_) + 1), "explained": self.explained_,
                             "cumulative": np.cumsum(self.explained_)})


def tsne_embed(Xz: pd.DataFrame, max_n: int = 1500, perplexity: float = 30, seed: int = 42) -> pd.DataFrame:
    """2-D t-SNE of (already scaled) features on an evenly spaced subsample. Visual diagnostic only."""
    idx = np.linspace(0, len(Xz) - 1, min(max_n, len(Xz))).astype(int)
    sub = Xz.iloc[idx]
    emb = TSNE(2, perplexity=min(perplexity, max(5, len(sub) // 4)), init="pca", learning_rate="auto", random_state=seed).fit_transform(sub.to_numpy())
    return pd.DataFrame(emb, index=sub.index, columns=["tsne1", "tsne2"])


def silhouette_by_regime(Z: pd.DataFrame, labels: pd.Series) -> float:
    """How separable the regimes are in PCA space (silhouette, -1..1). Subsampled for speed."""
    from sklearn.metrics import silhouette_score
    m = labels.notna().to_numpy(); Zv, lv = Z[m], labels[m]
    idx = np.linspace(0, len(Zv) - 1, min(2500, len(Zv))).astype(int)
    return float(silhouette_score(Zv.iloc[idx], lv.iloc[idx]))
