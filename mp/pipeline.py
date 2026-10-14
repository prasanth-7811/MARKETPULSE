"""End-to-end training / evaluation pipeline. Chronological splits with embargo; no test-set tuning."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from mp.features import build_features, mid
from mp.models import (ALGOS, DEFAULTS, Clf, Ens, ProbCal, binary_metrics, fit_temperature, make_estimator,
                       multiclass_metrics)
from mp.regimes import CLASSES, FilteredHMM, RegimeLabeler, label_states, run_lengths

warnings.filterwarnings("ignore")
C2I = {c: i for i, c in enumerate(CLASSES)}


class PipelineError(Exception):
    pass


def days_in_regime(codes: np.ndarray) -> np.ndarray:
    s = pd.Series(codes)
    return (s.groupby((s != s.shift()).cumsum()).cumcount() + 1).to_numpy()


def split_positions(n: int, cfg: dict, embargo: int) -> pd.Series:
    s = cfg["split"]
    i1, i2 = int(n * s["train"]), int(n * (s["train"] + s["val"]))
    lab = np.array(["train"] * n, dtype=object)
    lab[i1 - embargo:i1] = "embargo"; lab[i1:i2] = "validation"
    lab[i2 - embargo:i2] = "embargo"; lab[i2:] = "test"
    return pd.Series(lab)


def run_pipeline(df: pd.DataFrame, cfg: dict) -> dict:
    seed, mp_ = int(cfg["split"]["seed"]), cfg["models"]["params"]
    horizons = sorted(int(h) for h in cfg["transition"]["horizons"])
    embargo = max(int(cfg["split"]["embargo"]), max(horizons))
    feats, groups = build_features(df, cfg)
    labeler = RegimeLabeler(cfg["labels"]["window"], cfg["labels"]["trend_quantiles"])
    sig = labeler.signals(df["adj_close"])
    data = feats.join(sig).dropna()
    n = len(data)
    if n < 500: raise PipelineError(f"Only {n} usable observations after feature warm-up; need >= 500. Widen the date range.")
    split = split_positions(n, cfg, embargo); split.index = data.index
    tr = np.where(split == "train")[0]; va = np.where(split == "validation")[0]; te = np.where(split == "test")[0]
    labeler.fit(sig.loc[data.index[tr]])
    lab = labeler.label(data[["sig_mom", "sig_vol"]])
    y = lab.map(C2I).to_numpy().astype(int)
    y_next = np.full(n, -1); y_next[:-1] = y[1:]            # target: regime at t+1 given features at t
    Xb = data[feats.columns].copy()
    ok = y_next >= 0
    trv, vav, tev = tr[ok[tr]], va[ok[va]], te[ok[te]]

    # ---- HMM (fit on training rows only; filtered probabilities for all rows) ----
    f = cfg["features"]
    hcols = ["ret", f"vol_{mid(f['vol_windows'])}", f"mom_{mid(f['mom_windows'])}"]
    hmm = FilteredHMM(cfg["hmm"]["n_states"], cfg["hmm"]["covariance_type"], cfg["hmm"]["n_iter"], seed).fit(Xb[hcols].to_numpy()[tr])
    filt, nxt = hmm.filter(Xb[hcols].to_numpy())
    K = hmm.K
    st_tr = filt[tr].argmax(1)
    stats = pd.DataFrame({k: {"mom": Xb[hcols[2]].to_numpy()[tr][st_tr == k].mean() if (st_tr == k).any() else 0.0,
                              "vol": Xb[hcols[1]].to_numpy()[tr][st_tr == k].mean() if (st_tr == k).any() else 0.0,
                              "ret": Xb["ret"].to_numpy()[tr][st_tr == k].mean() if (st_tr == k).any() else 0.0,
                              "n_train": int((st_tr == k).sum())} for k in range(K)}).T
    state_names = label_states(stats)
    hmm_cols = [f"hmm_p{k}" for k in range(K)] + [f"hmm_n{k}" for k in range(K)]
    Hdf = pd.DataFrame(np.hstack([filt, nxt]), index=data.index, columns=hmm_cols)
    Xa = Xb.join(Hdf)
    hmm_state = pd.Series(np.array(state_names, dtype=object)[filt.argmax(1)], index=data.index)

    # ---- supervised models ----
    base_cols, aug_cols = list(Xb.columns), list(Xa.columns)
    nC = len(CLASSES)
    proba, models, fitted = {}, {}, {}
    sel = [m for m in cfg["models"]["selected"] if m in ALGOS]
    for name in sel:
        for tag, cols, X in (("base", base_cols, Xb), ("aug", aug_cols, Xa)):
            m = Clf(name, make_estimator(name, seed, mp_.get(name)), cols, nC).fit(X.iloc[trv], y_next[trv])
            fitted[(name, tag)] = m
            if tag == "base": models[name] = m; proba[name] = m.predict_proba(X)
    # persistence + HMM-based classifier
    eps = 0.02
    proba["Persistence baseline"] = np.eye(nC)[y] * (1 - eps) + eps / (nC - 1) * (1 - np.eye(nC)[y])
    M = np.zeros((K, nC))
    for t in trv: M += filt[t][:, None] * np.eye(nC)[y_next[t]][None, :]
    M = (M + 0.5) / (M + 0.5).sum(1, keepdims=True)
    proba["HMM-based approach"] = filt @ M
    # ensembles
    ens = Ens([fitted[(m, "base")] for m in sel]); models["Soft-voting ensemble"] = ens
    aens = Ens([fitted[(m, "aug")] for m in sel]); models["HMM-augmented ensemble"] = aens
    proba["Soft-voting ensemble"], proba["HMM-augmented ensemble"] = ens.predict_proba(Xb), aens.predict_proba(Xa)
    T = fit_temperature(proba["HMM-augmented ensemble"][vav], y_next[vav]) if cfg["calibration"]["method"] == "temperature" else 1.0
    cal = Ens(aens.members, T); models["Calibrated hybrid"] = cal; proba["Calibrated hybrid"] = cal.predict_proba(Xa)

    # ---- metrics (train / validation / test reported separately) ----
    rows = []
    for name, P in proba.items():
        for sname, idx in (("train", trv), ("validation", vav), ("test", tev)):
            if len(idx): rows.append(dict(model=name, split=sname, **multiclass_metrics(y_next[idx], P[idx])))
    metrics = pd.DataFrame(rows)

    # ---- ablation (LightGBM on growing feature sets; ensembles reused) ----
    abl_models, abl = {}, []
    gr = lambda *k: [c for g in k for c in groups[g]]
    for label, cols in (("Supervised baseline (returns only)", gr("returns")),
                        ("Baseline + volatility features", gr("returns", "volatility")),
                        ("Baseline + volatility + momentum", gr("returns", "volatility", "momentum"))):
        m = Clf("LightGBM", make_estimator("LightGBM", seed, mp_.get("LightGBM")), cols, nC).fit(Xb.iloc[trv], y_next[trv])
        abl_models[label] = m.predict_proba(Xb)
    abl_models["Ensemble without HMM information"] = proba["Soft-voting ensemble"]
    abl_models["HMM-augmented ensemble"] = proba["HMM-augmented ensemble"]
    abl_models["Calibrated hybrid model"] = proba["Calibrated hybrid"]
    for label, P in abl_models.items():
        for sname, idx in (("validation", vav), ("test", tev)):
            abl.append(dict(configuration=label, split=sname, **{k: v for k, v in multiclass_metrics(y_next[idx], P[idx]).items() if k in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier", "ece")}))
    ablation = pd.DataFrame(abl)

    # ---- transition models (one per horizon, trained separately) ----
    reg_oh = pd.DataFrame(np.eye(nC)[y], index=data.index, columns=[f"regime_{i}" for i in range(nC)])
    Xt = Xa.join(reg_oh); Xt["days_in_regime"] = days_in_regime(y)
    tcols, trans = list(Xt.columns), {}
    for h in horizons:
        ch = np.zeros(n, bool)
        for k in range(1, h + 1): ch[:n - k] |= y[k:] != y[:-k]
        yt, valid = ch.astype(int), np.arange(n) + h < n
        a, b, c = tr[valid[tr]], va[valid[va]], te[valid[te]]
        entry = dict(horizon=h, y=yt, valid=valid, features=tcols)
        try:
            m = Clf("Transition", make_estimator("Transition", seed, mp_.get("Transition")), tcols, 2).fit(Xt.iloc[a], yt[a])
            raw = m.predict_proba(Xt)[:, 1]
            pc = ProbCal(cfg["calibration"]["transition_method"]).fit(raw[b], yt[b])
            entry.update(model=m, cal=pc, p_raw=raw, p_cal=pc.transform(raw), cal_method=pc.method, cal_note=pc.note,
                         idx={"train": a, "validation": b, "test": c})
        except ValueError as exc:
            entry.update(model=None, error=str(exc))
        trans[h] = entry

    reg_runs = run_lengths(hmm_state)
    return dict(data=data, split=split, idx=dict(train=trv, validation=vav, test=tev), y=y, y_next=y_next, labels=lab,
                labeler=labeler, hmm=hmm, hmm_cols=hcols, hmm_state=hmm_state, hmm_filt=Hdf, state_names=state_names,
                state_stats=stats, hmm_runs=reg_runs, Xb=Xb, Xa=Xa, Xt=Xt, groups=groups, proba=proba, models=models,
                metrics=metrics, ablation=ablation, transition=trans, temperature=T, embargo=embargo,
                feature_cols=dict(base=base_cols, aug=aug_cols, transition=tcols),
                params={a: {**DEFAULTS[a], **mp_.get(a, {})} for a in ALGOS})
