"""Publication figures (matplotlib, static PNG/PDF) for the report. Colour-blind-safe Okabe-Ito palette."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mp.regimes import CLASSES, COLORS
from mp.errors import SHORT

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif", "axes.grid": True, "grid.alpha": .25})
C6 = [COLORS[c] for c in CLASSES]
ACC, GREY = "#0072B2", "#7F7F7F"


def _save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(out / f"{name}.png", bbox_inches="tight"); plt.close(fig); return name


def price_regimes(R, df, out):
    d = R["data"]; fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
    lab = R["labels"]; px = df["adj_close"].reindex(d.index)
    for i, c in enumerate(CLASSES):
        m = (lab == c).to_numpy(); ax[0].scatter(d.index[m], px[m], s=1.5, color=C6[i], label=SHORT[i], rasterized=True)
    ax[0].set_yscale("log"); ax[0].set_ylabel("S&P 500 (log)"); ax[0].legend(ncol=6, markerscale=5, loc="upper left", frameon=False, columnspacing=.8, handletextpad=.2)
    if "vix" in df: ax[1].plot(d.index, df["vix"].reindex(d.index), color=GREY, lw=.7); ax[1].set_ylabel("VIX")
    for k, col in (("validation", "#F0E442"), ("test", "#CC79A7")):
        ix = R["idx"][k]; [a.axvspan(d.index[ix[0]], d.index[ix[-1]], color=col, alpha=.12, lw=0) for a in ax]
    ax[1].text(d.index[R["idx"]["validation"][0]], ax[1].get_ylim()[1] * .8, " validation", fontsize=7); ax[1].text(d.index[R["idx"]["test"][0]], ax[1].get_ylim()[1] * .8, " test", fontsize=7)
    return _save(fig, out, "fig_price_regimes")


def correlation(R, out):
    X = R["Xb"].iloc[R["idx"]["train"]]; c = X.corr(); order = c.abs().sum().sort_values(ascending=False).index
    from scipy.cluster.hierarchy import leaves_list, linkage
    order = c.columns[leaves_list(linkage(1 - c.abs().to_numpy()[np.triu_indices(len(c), 1)], "average"))]
    c = c.loc[order, order]; fig, ax = plt.subplots(figsize=(6.4, 5.6)); im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90, fontsize=5.5); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index, fontsize=5.5)
    fig.colorbar(im, shrink=.8, label="Pearson r (train rows)"); return _save(fig, out, "fig_correlation")


def pca_fig(R, red, Z, out):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5)); n = min(15, len(red.explained_)); e = red.explained_[:n]
    ax[0].bar(range(1, n + 1), e, color=ACC); ax[0].plot(range(1, n + 1), np.cumsum(red.explained_)[:n], "o-", color="#D55E00", ms=2.5); ax[0].axhline(.95, ls=":", color=GREY)
    ax[0].set_xlabel("component"); ax[0].set_ylabel("explained variance"); ax[0].set_title(f"(a) Scree: {red.k} PCs reach 95%")
    lab = R["labels"].reindex(Z.index)
    for i, c in enumerate(CLASSES):
        m = (lab == c).to_numpy(); ax[1].scatter(Z["PC1"][m], Z["PC2"][m], s=2, color=C6[i], label=SHORT[i], alpha=.7, rasterized=True)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].set_title("(b) PC1-PC2 by regime"); ax[1].legend(markerscale=3, fontsize=5.5, frameon=False)
    L = red.loadings(2); top = L.abs().sum(1).sort_values(ascending=False).head(10).index; y = np.arange(len(top))
    ax[2].barh(y - .2, L.loc[top, "PC1"], .4, color=ACC, label="PC1"); ax[2].barh(y + .2, L.loc[top, "PC2"], .4, color="#E69F00", label="PC2")
    ax[2].set_yticks(y); ax[2].set_yticklabels(top, fontsize=6); ax[2].invert_yaxis(); ax[2].legend(frameon=False); ax[2].set_title("(c) Top loadings")
    return _save(fig, out, "fig_pca")


def tsne_fig(R, emb, out):
    fig, ax = plt.subplots(figsize=(3.5, 3.2)); lab = R["labels"].reindex(emb.index)
    for i, c in enumerate(CLASSES):
        m = (lab == c).to_numpy(); ax.scatter(emb.tsne1[m], emb.tsne2[m], s=3, color=C6[i], label=SHORT[i], alpha=.8, rasterized=True)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(markerscale=3, fontsize=5.5, frameon=False); ax.grid(False)
    return _save(fig, out, "fig_tsne")


def confusion(E, out):
    cm = E["confusion"]; z = cm.div(cm.sum(1).clip(lower=1), axis=0); fig, ax = plt.subplots(figsize=(3.5, 3.2)); ax.imshow(z, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(6):
        for j in range(6): ax.text(j, i, f"{z.iloc[i, j]:.2f}\n({cm.iloc[i, j]})", ha="center", va="center", fontsize=5.2, color="white" if z.iloc[i, j] > .55 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=6); ax.set_yticks(range(6)); ax.set_yticklabels(cm.index, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual (t+1)"); return _save(fig, out, "fig_confusion")


def reliability(R, E, P, out):
    te = R["idx"]["test"]; y = R["y_next"][te]; fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for name, col in (("Calibrated hybrid", ACC), ("HMM-augmented ensemble", "#E69F00"), ("Persistence baseline", "#D55E00"), ("Logistic Regression", "#009E73")):
        if name not in P: continue
        Q = P[name][te]; conf, ok = Q.max(1), Q.argmax(1) == y; bins = np.linspace(.2, 1, 9); xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10: xs.append(conf[m].mean()); ys.append(ok[m].mean())
        ax[0].plot(xs, ys, "o-", ms=3, color=col, label=name)
    ax[0].plot([.2, 1], [.2, 1], ":", color="k"); ax[0].set_xlabel("mean confidence"); ax[0].set_ylabel("empirical accuracy"); ax[0].set_title("(a) Reliability (top-class), test"); ax[0].legend(frameon=False, fontsize=6)
    h = sorted(R["transition"]); w = .27
    for i, hh in enumerate(h):
        e = R["transition"][hh]
        if e.get("model") is None: continue
        t = e["idx"]["test"]; yt = e["y"][t]; bins = np.linspace(0, 1, 7); xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (e["p_cal"][t] > lo) & (e["p_cal"][t] <= hi)
            if m.sum() >= 15: xs.append(e["p_cal"][t][m].mean()); ys.append(yt[m].mean())
        ax[1].plot(xs, ys, "o-", ms=3, label=f"{hh}-day")
    ax[1].plot([0, 1], [0, 1], ":", color="k"); ax[1].set_xlabel("predicted P(regime change)"); ax[1].set_ylabel("observed frequency"); ax[1].set_title("(b) Transition reliability, test"); ax[1].legend(frameon=False)
    return _save(fig, out, "fig_reliability")


def model_compare(main, out):
    d = main.sort_values("log_loss", ascending=False); fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True); y = np.arange(len(d))
    col = ["#D55E00" if m == "Calibrated hybrid" else ACC for m in d.model]
    ax[0].barh(y, d.log_loss, xerr=[d.log_loss - d.log_loss_lo, d.log_loss_hi - d.log_loss], color=col, ecolor="k", error_kw=dict(lw=.6, capsize=1.5)); ax[0].set_yticks(y); ax[0].set_yticklabels(d.model, fontsize=6.5)
    ax[0].set_xlabel("log loss (lower better)"); ax[0].set_title("(a) Probabilistic quality")
    ax[1].barh(y, d.macro_f1, xerr=[d.macro_f1 - d.macro_f1_lo, d.macro_f1_hi - d.macro_f1], color=col, ecolor="k", error_kw=dict(lw=.6, capsize=1.5)); ax[1].set_xlabel("macro-F1 (higher better)"); ax[1].set_title("(b) Hard-label quality")
    return _save(fig, out, "fig_model_compare")


def error_fig(E, out):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4)); m = E["by_margin"]
    ax[0].bar(range(len(m)), m.error_rate, color=ACC); ax[0].set_xticks(range(len(m))); ax[0].set_xticklabels([s.replace(" ", "\n") for s in m.margin.astype(str)], fontsize=6); ax[0].set_ylabel("error rate"); ax[0].set_title("(a) By distance to threshold")
    if "by_vix" in E:
        v = E["by_vix"].set_index("vix_bucket").reindex(sorted(E["by_vix"].vix_bucket.astype(str), key=lambda s: ["low", "mid", "high"].index(s.split()[1]))); ax[1].bar(range(len(v)), v.error_rate, color="#E69F00")
        ax[1].set_xticks(range(len(v))); ax[1].set_xticklabels([s.replace(" (", "\n(") for s in v.index.astype(str)], fontsize=6); ax[1].set_title("(b) By VIX regime")
    y = E["by_year"]; ax[2].plot(y.year, y.model_error, "o-", color=ACC, ms=3, label="hybrid"); ax[2].plot(y.year, y.persistence_error, "s--", color="#D55E00", ms=3, label="persistence"); ax[2].legend(frameon=False); ax[2].set_title("(c) By year")
    return _save(fig, out, "fig_errors")


def tuning_fig(studies: dict, out, summary=None):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5)); col = ["#0072B2", "#009E73", "#D55E00", "#E69F00"]
    for (name, (df, direction)), c in zip(studies.items(), col):
        d = df[df.state == "COMPLETE"]; ax[0].plot(d.trial, d.value.cummax(), "-", color=c, label=name); ax[0].scatter(d.trial, d.value, s=5, color=c, alpha=.4)
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("walk-forward CV score"); ax[0].set_title("(a) Optuna optimisation history"); ax[0].legend(frameon=False, fontsize=6)
    if summary is not None:
        x, w = np.arange(len(summary)), .38
        ax[1].bar(x - w / 2, summary.val_logloss_default, w, color=GREY, label="default"); ax[1].bar(x + w / 2, summary.val_logloss_tuned, w, color=ACC, label="Optuna-tuned")
        ax[1].set_xticks(x); ax[1].set_xticklabels([m.replace(" ", "\n") for m in summary.model], fontsize=6.5); ax[1].set_ylabel("validation log loss"); ax[1].set_title("(b) Default vs tuned (validation)"); ax[1].legend(frameon=False)
    return fig, ax


def importance_fig(imp: pd.Series, out, title="Mean |SHAP|"):
    imp = imp.sort_values().tail(15); fig, ax = plt.subplots(figsize=(3.5, 3.0)); ax.barh(imp.index, imp.values, color=ACC); ax.set_xlabel(title); ax.tick_params(axis="y", labelsize=6)
    return _save(fig, out, "fig_shap")


def feature_ablation_fig(fa, out):
    t = fa[fa.split == "test"].reset_index(drop=True); fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4)); x = np.arange(len(t))
    ax[0].plot(x, t.macro_f1, "o-", color=ACC); ax[0].set_ylabel("test macro-F1"); ax[1].plot(x, t.log_loss, "o-", color="#D55E00"); ax[1].set_ylabel("test log loss")
    for a in ax: a.set_xticks(x); a.set_xticklabels(t.feature_set, rotation=40, ha="right", fontsize=6)
    return _save(fig, out, "fig_feature_ablation")


def hmm_fig(R, out, start="2007-06-01", end="2010-06-30"):
    H = R["hmm_filt"][[c for c in R["hmm_filt"].columns if c.startswith("hmm_p")]].loc[start:end]; fig, ax = plt.subplots(figsize=(7.2, 2.1))
    ax.stackplot(H.index, H.T, labels=R["state_names"], colors=["#56B4E9", "#D55E00", "#94A3B8", "#009E73"][:H.shape[1]], alpha=.9); ax.set_ylim(0, 1); ax.set_ylabel("filtered P(state)"); ax.legend(ncol=3, frameon=False, loc="lower left", fontsize=6)
    return _save(fig, out, "fig_hmm_states")
