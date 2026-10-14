"""Benchmark suite: our hybrid vs. baselines drawn from the regime / volatility-forecasting literature.

All methods are trained on TRAIN rows, probability post-processing (where needed) is fitted on VALIDATION rows,
and every number reported is on the held-out TEST split. Uncertainty is quantified with a moving-block
bootstrap (block = 20 trading days, preserving serial dependence) - never an i.i.d. bootstrap.

The methods are *re-implementations of method families*, not replications of any paper's exact setup or numbers.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score,
                             accuracy_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mp.models import Clf, make_estimator, multiclass_metrics, ProbCal
from mp.reduce import PCAReducer
from mp.regimes import CLASSES

warnings.filterwarnings("ignore")
NC = len(CLASSES)

LITERATURE = pd.DataFrame([
    ["Hamilton (1989), Econometrica 57(2):357-384", "Markov-switching regimes in returns", "Gaussian HMM (multivariate generalisation) with filtered probabilities -> 'HMM-based approach'"],
    ["Rabiner (1989), Proc. IEEE 77(2):257-286", "HMM inference (forward algorithm)", "Forward filtering (no smoothing / look-ahead) in mp.regimes.FilteredHMM"],
    ["Ang & Bekaert (2002), Rev. Financial Studies 15(4):1137-1187", "Regime shifts in asset allocation", "Regime-conditioned state-to-next-regime mapping (HMM-based approach)"],
    ["Kritzman, Page & Turkington (2012), Financial Analysts J. 68(3):22-39", "HMM regimes for dynamic strategies", "HMM states used as features for supervised models (HMM-augmented ensemble)"],
    ["J.P. Morgan/Reuters (1996), RiskMetrics Technical Document", "EWMA volatility (lambda = 0.94)", "EWMA sigma baseline for the volatility-regime sub-task"],
    ["Bollerslev (1986), J. Econometrics 31(3):307-327", "GARCH(1,1) volatility forecasting", "GARCH(1,1) one-step sigma forecast (arch package) for the volatility sub-task"],
    ["Faber (2007), J. Wealth Management 9(4):69-79", "Moving-average trend rule", "SMA-50/200 trend rule for the trend sub-task"],
    ["Chen & Guestrin (2016), KDD; Ke et al. (2017), NeurIPS", "Gradient-boosted trees", "XGBoost and LightGBM classifiers (tuned with Optuna, Akiba et al. 2019, KDD)"],
    ["Guo et al. (2017), ICML", "Temperature scaling for calibration", "Temperature-scaled soft-voting ensemble ('Calibrated hybrid')"],
    ["Lundberg & Lee (2017), NeurIPS", "SHAP explanations", "TreeSHAP in the Explainable-AI page and report"],
], columns=["Reference", "Method family", "Implementation here"])


# ----------------------------------------------------------------------------- bootstrap
def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / block)); starts = rng.integers(0, max(n - block, 1), nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def boot_ci(fn, n: int, B: int = 300, block: int = 20, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        try: v.append(fn(block_indices(n, block, rng)))
        except Exception: continue
    v = np.array(v, float)
    return (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def paired_boot(fa, fb, n: int, B: int = 500, block: int = 20, seed: int = 0) -> dict:
    """Paired block bootstrap of metric(A)-metric(B) on identical resamples. p = two-sided sign probability."""
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        ix = block_indices(n, block, rng)
        try: d.append(fa(ix) - fb(ix))
        except Exception: continue
    d = np.array(d, float); d = d[~np.isnan(d)]
    return dict(delta=float(np.mean(d)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def _mf1(y, P): return lambda ix: f1_score(y[ix], P[ix].argmax(1), average="macro", zero_division=0)
def _ll(y, P):
    Pn = np.clip(P, 1e-6, 1); Pn = Pn / Pn.sum(1, keepdims=True)
    return lambda ix: log_loss(y[ix], Pn[ix], labels=list(range(P.shape[1])))


# ----------------------------------------------------------------------------- extra models
def _extra_models(R: dict, seed: int) -> dict[str, np.ndarray]:
    """Additional full-task baselines, all trained on train rows: majority class, MLP (neural), PCA+LR, PCA+LightGBM."""
    Xb, y, idx = R["Xb"], R["y_next"], R["idx"]; tr = idx["train"]; out = {}
    maj = np.bincount(y[tr], minlength=NC) / len(tr)
    out["Majority class"] = np.tile(0.9 * maj + 0.1 / NC, (len(Xb), 1))
    va = idx["validation"]; best = None                      # small, fair search: L2 strength chosen by VALIDATION log loss
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 3.0):
        m = Clf("MLP", make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=alpha, max_iter=400, random_state=seed)), list(Xb.columns), NC).fit(Xb.iloc[tr], y[tr])
        ll = log_loss(y[va], np.clip(m.predict_proba(Xb.iloc[va]), 1e-6, 1) / np.clip(m.predict_proba(Xb.iloc[va]), 1e-6, 1).sum(1, keepdims=True), labels=list(range(NC)))
        if best is None or ll < best[0]: best = (ll, alpha, m)
    out["MLP (neural network)"] = best[2].predict_proba(Xb)
    red = PCAReducer(0.95, seed=seed).fit(Xb.iloc[tr]); Z = red.transform(Xb)
    lr = Clf("PCA+LR", LogisticRegression(max_iter=3000, C=1.0), list(Z.columns), NC)
    out["PCA + Logistic Regression"] = lr.fit(Z.iloc[tr], y[tr]).predict_proba(Z)
    return out


def full_task(R: dict, seed: int = 42, B: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """6-class next-day regime task: metrics + 95% block-bootstrap CIs on TEST, and paired tests vs 'Calibrated hybrid'."""
    y, te = R["y_next"], R["idx"]["test"]
    P = dict(R["proba"]); P.update(_extra_models(R, seed))
    yt, rows = y[te], []
    for name, Pa in P.items():
        Pt = Pa[te]; m = multiclass_metrics(yt, Pt)
        f1lo, f1hi = boot_ci(_mf1(yt, Pt), len(te), B, seed=seed); lllo, llhi = boot_ci(_ll(yt, Pt), len(te), B, seed=seed)
        rows.append(dict(model=name, accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], macro_f1=m["macro_f1"], macro_f1_lo=f1lo, macro_f1_hi=f1hi,
                         log_loss=m["log_loss"], log_loss_lo=lllo, log_loss_hi=llhi, brier=m["brier"], ece=m["ece"]))
    main = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    ref = "Calibrated hybrid"; pr = []
    for name in P:
        if name == ref: continue
        a, b = _ll(yt, P[ref][te]), _ll(yt, P[name][te]); fa, fb = _mf1(yt, P[ref][te]), _mf1(yt, P[name][te])
        l, f = paired_boot(a, b, len(te), seed=seed), paired_boot(fa, fb, len(te), seed=seed)
        pr.append(dict(comparison=f"{ref} vs {name}", d_log_loss=l["delta"], ll_lo=l["lo"], ll_hi=l["hi"], p_log_loss=l["p"],
                       d_macro_f1=f["delta"], f1_lo=f["lo"], f1_hi=f["hi"], p_macro_f1=f["p"]))
    return main, pd.DataFrame(pr), P


# ----------------------------------------------------------------------------- sub-tasks
def _platt(score_tr_val: tuple[np.ndarray, np.ndarray], score_all: np.ndarray) -> np.ndarray:
    s, y = score_tr_val
    m = LogisticRegression(C=100).fit(s.reshape(-1, 1), y)
    return m.predict_proba(score_all.reshape(-1, 1))[:, 1]


def _bin_metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(accuracy=accuracy_score(y, p >= .5), roc_auc=roc_auc_score(y, p), log_loss=log_loss(y, p), brier=brier_score_loss(y, p))


def volatility_task(R: dict, df: pd.DataFrame, P: dict) -> pd.DataFrame:
    """Next-day volatility regime (High/Low): literature volatility forecasters vs our model collapsed to P(High)."""
    from arch import arch_model
    y = R["y_next"] % 2; idx, data = R["idx"], R["data"]
    r = np.log(df["adj_close"]).diff().reindex(data.index)
    tr, va, te = idx["train"], idx["validation"], idx["test"]
    rows = {}
    cur = (R["y"] % 2).astype(float); rows["Persistence (today's regime)"] = np.where(cur == 1, .98, .02)
    ew = np.sqrt(r.pow(2).ewm(alpha=0.06, adjust=False).mean()).to_numpy()
    sc = {"RiskMetrics EWMA sigma": ew, "VIX level (implied vol)": df["vix"].reindex(data.index).to_numpy() if "vix" in df else None,
          "Realised vol (60d)": data["sig_vol"].to_numpy()}
    try:
        am = arch_model(r.iloc[tr].dropna() * 100, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
        res = am.fit(disp="off"); om, al, be = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
        full = arch_model(r.fillna(0) * 100, vol="GARCH", p=1, q=1, dist="t", mean="Constant").fix(res.params)
        s2 = full.conditional_volatility.to_numpy() ** 2; rr = (r.fillna(0).to_numpy() * 100 - res.params["mu"]) ** 2
        sc["GARCH(1,1)-t sigma forecast"] = np.sqrt(om + al * rr + be * s2)         # sigma_{t+1|t}
    except Exception:
        pass
    for k, s in sc.items():
        if s is None or np.isnan(s).all(): continue
        s = np.log(np.clip(s, 1e-8, None)); fit = np.r_[tr, va]
        rows[k + " (+Platt on train+val)"] = _platt((s[fit], y[fit]), s)
    pv = P["Calibrated hybrid"].reshape(len(y), -1)[:, 1::2].sum(1); rows["Calibrated hybrid (collapsed to P(High))"] = pv
    pv2 = P["Soft-voting ensemble"][:, 1::2].sum(1); rows["Soft-voting ensemble (collapsed)"] = pv2
    out = [dict(model=k, **_bin_metrics(y[te], v[te])) for k, v in rows.items()]
    return pd.DataFrame(out).sort_values("log_loss").reset_index(drop=True)


def trend_task(R: dict, df: pd.DataFrame, P: dict) -> pd.DataFrame:
    """Next-day trend regime (Bullish/Sideways/Bearish): SMA-cross rule vs persistence vs our model collapsed to 3 classes."""
    y = R["y_next"] // 2; data, te = R["data"], R["idx"]["test"]
    p = df["adj_close"].reindex(data.index); s50, s200 = df["adj_close"].rolling(50).mean().reindex(data.index), df["adj_close"].rolling(200).mean().reindex(data.index)
    rule = np.where((p > s200) & (s50 > s200), 0, np.where((p < s200) & (s50 < s200), 2, 1))        # 0 Bullish, 1 Sideways, 2 Bearish
    rows = {"SMA-50/200 rule (Faber-style)": rule, "Persistence (today's regime)": R["y"] // 2,
            "Majority class": np.full(len(y), np.bincount(y[R["idx"]["train"]]).argmax())}
    for name in ("Calibrated hybrid", "Soft-voting ensemble", "HMM-based approach", "Persistence baseline"):
        Q = P[name].reshape(len(y), 3, 2).sum(2); rows[f"{name} (collapsed to 3 trend classes)"] = Q.argmax(1)
    return pd.DataFrame([dict(model=k, accuracy=accuracy_score(y[te], v[te]), macro_f1=f1_score(y[te], v[te], average="macro", zero_division=0)) for k, v in rows.items()]
                        ).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def transition_task(R: dict, seed: int = 42, B: int = 200) -> pd.DataFrame:
    """Regime-change-within-h-days: baselines vs our calibrated LightGBM. Metrics on TEST with block-bootstrap CI for PR-AUC."""
    from sklearn.pipeline import make_pipeline
    Xt, rows = R["Xt"], []
    A = R["hmm"].transmat; filt = R["hmm_filt"][[c for c in R["hmm_filt"].columns if c.startswith("hmm_p")]].to_numpy()
    dir_ = R["Xt"]["days_in_regime"].to_numpy()
    for h, e in R["transition"].items():
        if e.get("model") is None: continue
        yt, tr, va, te = e["y"], e["idx"]["train"], e["idx"]["validation"], e["idx"]["test"]
        P = {}
        P["Constant (train prevalence)"] = np.full(len(yt), yt[tr].mean())
        bins = np.array([0, 1, 2, 3, 5, 10, 20, 40, 10 ** 6]); b = np.digitize(dir_, bins[1:-1], right=True)
        haz = pd.Series(yt[tr]).groupby(b[tr]).mean().reindex(range(len(bins) - 1)).fillna(yt[tr].mean()).to_numpy(); P["Empirical hazard by days-in-regime"] = haz[b]
        Ah = np.linalg.matrix_power(A, h); hs = 1 - (filt * np.diag(Ah)[None, :]).sum(1)
        P["HMM-implied state-change prob. (+Platt)"] = _platt((hs[np.r_[tr, va]], yt[np.r_[tr, va]]), hs)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(Xt.iloc[tr], yt[tr]); P["Logistic Regression (all features)"] = lr.predict_proba(Xt)[:, 1]
        P["LightGBM (raw)"] = e["p_raw"]; P["LightGBM + calibration (ours)"] = e["p_cal"]
        for k, p in P.items():
            y_, p_ = yt[te], np.clip(p[te], 1e-6, 1 - 1e-6)
            lo, hi = boot_ci(lambda ix: average_precision_score(y_[ix], p_[ix]), len(te), B, seed=seed)
            rows.append(dict(horizon=h, model=k, prevalence=float(y_.mean()), pr_auc=average_precision_score(y_, p_), pr_lo=lo, pr_hi=hi, roc_auc=roc_auc_score(y_, p_),
                             brier=brier_score_loss(y_, p_), log_loss=log_loss(y_, p_)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- feature engineering / dimensionality reduction studies
def feature_ablation(R: dict, seed: int = 42) -> pd.DataFrame:
    """LightGBM trained on cumulative feature groups -> value of each engineered group (validation and test)."""
    g, Xb, y, idx = R["groups"], R["Xb"], R["y_next"], R["idx"]; tr = idx["train"]
    order = [k for k in ("returns", "volatility", "momentum", "volume", "advanced_vol", "trend", "cross_asset") if k in g]
    rows, cols = [], []
    for k in order:
        cols = cols + g[k]
        m = Clf("LightGBM", make_estimator("LightGBM", seed), cols, NC).fit(Xb.iloc[tr], y[tr]); P = m.predict_proba(Xb)
        for s in ("validation", "test"):
            ix = idx[s]; mm = multiclass_metrics(y[ix], P[ix])
            rows.append(dict(feature_set="+ " + k, n_features=len(cols), split=s, macro_f1=mm["macro_f1"], log_loss=mm["log_loss"], accuracy=mm["accuracy"]))
    return pd.DataFrame(rows)


def pca_study(R: dict, seed: int = 42, variance: float = 0.95) -> pd.DataFrame:
    """Effect of PCA input dimensionality on LR and LightGBM (PCA fitted on train only)."""
    Xb, y, idx = R["Xb"], R["y_next"], R["idx"]; tr, te = idx["train"], idx["test"]
    red = PCAReducer(variance, seed=seed).fit(Xb.iloc[tr]); Zfull = PCAReducer(1.0, seed=seed).fit(Xb.iloc[tr]); Z = Zfull.transform(Xb); rows = []
    for k in sorted({3, 5, 8, red.k, Z.shape[1]}):
        cols = list(Z.columns[:k])
        for name, est in (("Logistic Regression", LogisticRegression(max_iter=3000)), ("LightGBM", make_estimator("LightGBM", seed))):
            m = Clf(name, est, cols, NC).fit(Z.iloc[tr], y[tr]); mm = multiclass_metrics(y[te], m.predict_proba(Z)[te])
            rows.append(dict(input=f"PCA-{k}" + (" (95% var.)" if k == red.k else ""), n_inputs=k, model=name, macro_f1=mm["macro_f1"], log_loss=mm["log_loss"], accuracy=mm["accuracy"]))
    for name, est in (("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))), ("LightGBM", make_estimator("LightGBM", seed))):
        m = Clf(name, est, list(Xb.columns), NC).fit(Xb.iloc[tr], y[tr]); mm = multiclass_metrics(y[te], m.predict_proba(Xb)[te])
        rows.append(dict(input=f"Raw features ({Xb.shape[1]})", n_inputs=Xb.shape[1], model=name, macro_f1=mm["macro_f1"], log_loss=mm["log_loss"], accuracy=mm["accuracy"]))
    return pd.DataFrame(rows)


def _ece_bin(y, p, bins=10) -> float:
    e = np.linspace(0, 1, bins + 1); t = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): t += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(t)


def calibration_study(R: dict) -> pd.DataFrame:
    """Transition models: raw vs Platt vs isotonic calibration, fitted on VALIDATION, scored on TEST."""
    rows = []
    for h, e in R["transition"].items():
        if e.get("model") is None: continue
        yt, va, te, raw = e["y"], e["idx"]["validation"], e["idx"]["test"], e["p_raw"]
        for meth in ("none", "platt", "isotonic"):
            p = ProbCal(meth).fit(raw[va], yt[va]).transform(raw)[te]; p = np.clip(p, 1e-6, 1 - 1e-6)
            rows.append(dict(horizon=h, calibration=meth, brier=brier_score_loss(yt[te], p), log_loss=log_loss(yt[te], p), ece=_ece_bin(yt[te], p), roc_auc=roc_auc_score(yt[te], p)))
    return pd.DataFrame(rows)


def transition_feature_study(R: dict, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Does advanced / cross-asset feature engineering help the (harder) regime-change task? LightGBM, 3 seeds, TEST metrics."""
    g, Xt, rows = R["groups"], R["Xt"], []
    drop_adv = [c for k in ("advanced_vol", "trend", "cross_asset") for c in g.get(k, [])]
    drop_x = [c for c in g.get("cross_asset", [])]
    sets = {"core + HMM + regime state": [c for c in Xt.columns if c not in drop_adv], "+ advanced vol / trend": [c for c in Xt.columns if c not in drop_x], "+ cross-asset (VIX, Brent) = full": list(Xt.columns)}
    for h, e in R["transition"].items():
        if e.get("model") is None: continue
        yt, tr, te = e["y"], e["idx"]["train"], e["idx"]["test"]
        for name, cols in sets.items():
            pr, br = [], []
            for sd in seeds:
                m = Clf("Transition", make_estimator("Transition", sd), cols, 2).fit(Xt.iloc[tr], yt[tr]); p = np.clip(m.predict_proba(Xt)[:, 1][te], 1e-6, 1 - 1e-6)
                pr.append(average_precision_score(yt[te], p)); br.append(brier_score_loss(yt[te], p))
            rows.append(dict(horizon=h, feature_set=name, n_features=len(cols), pr_auc=np.mean(pr), pr_auc_sd=np.std(pr), brier=np.mean(br), brier_sd=np.std(br)))
    return pd.DataFrame(rows)


def run_all(R: dict, df: pd.DataFrame, seed: int = 42, B: int = 300) -> dict:
    main, paired, P = full_task(R, seed, B)
    return dict(main=main, paired=paired, volatility=volatility_task(R, df, P), trend=trend_task(R, df, P), transition=transition_task(R, seed),
                feature_ablation=feature_ablation(R, seed), calibration=calibration_study(R), transition_features=transition_feature_study(R), pca=pca_study(R, seed), literature=LITERATURE, proba_all=P)
