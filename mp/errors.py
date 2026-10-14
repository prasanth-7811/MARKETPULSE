"""Error analysis for the 6-class next-day regime task. Everything is computed on a chosen split (default: TEST)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from mp.regimes import CLASSES

SHORT = [c.replace(" Volatility", "").replace(" / ", "/") for c in CLASSES]


def error_analysis(R: dict, df: pd.DataFrame, model: str = "Calibrated hybrid", split: str = "test", P: dict | None = None) -> dict:
    P = P or R["proba"]; ix = R["idx"][split]; data = R["data"]
    y, Pm = R["y_next"][ix], P[model][ix]; pred = Pm.argmax(1); cur = R["y"][ix]; pers = cur
    wrong = pred != y; conf = Pm.max(1); dates = data.index[ix]
    out: dict = {}
    # --- per class
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    out["per_class"] = pd.DataFrame(dict(regime=SHORT, precision=pr, recall=rc, f1=f1, support=sup))
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES))); out["confusion"] = pd.DataFrame(cm, index=SHORT, columns=SHORT)
    off = [(SHORT[a], SHORT[b], int(cm[a, b])) for a in range(len(SHORT)) for b in range(len(SHORT)) if a != b and cm[a, b] > 0]
    out["top_confusions"] = pd.DataFrame(sorted(off, key=lambda t: -t[2])[:8], columns=["actual", "predicted", "count"]).assign(share_of_errors=lambda d: d["count"] / max(wrong.sum(), 1))
    # --- which dimension is wrong?
    tw, vw = (pred // 2) != (y // 2), (pred % 2) != (y % 2)
    out["dimension"] = pd.DataFrame({"error type": ["trend only", "volatility only", "both"], "count": [int((tw & ~vw).sum()), int((vw & ~tw).sum()), int((tw & vw).sum())]}).assign(share=lambda d: d["count"] / max(d["count"].sum(), 1))
    # --- change vs stay days (the rows where a model can add value over persistence)
    chg = y != cur
    out["change_vs_stay"] = pd.DataFrame([
        dict(group="Regime stays (t+1 = t)", n=int((~chg).sum()), model_acc=float((~wrong[~chg]).mean()), persistence_acc=1.0),
        dict(group="Regime changes (t+1 != t)", n=int(chg.sum()), model_acc=float((~wrong[chg]).mean()) if chg.any() else np.nan, persistence_acc=0.0)])
    pc = pred != cur
    out["change_detection"] = dict(true_changes=int(chg.sum()), predicted_changes=int(pc.sum()), caught=int((pc & chg).sum()),
                                   precision=float((pc & chg).sum() / max(pc.sum(), 1)), recall=float((pc & chg).sum() / max(chg.sum(), 1)))
    # --- boundary proximity (labels are threshold rules -> errors concentrate near thresholds)
    lab = R["labeler"]; s = data[["sig_mom", "sig_vol"]].iloc[ix]
    mom_m = np.minimum((s.sig_mom - lab.lo).abs(), (s.sig_mom - lab.hi).abs()); vol_m = np.abs(np.log(s.sig_vol / lab.vol_thr))
    marg = np.minimum(mom_m / mom_m.std(), vol_m / vol_m.std()).to_numpy()
    q = pd.qcut(marg, 4, labels=["closest 25%", "25-50%", "50-75%", "farthest 25%"], duplicates="drop")
    out["by_margin"] = pd.DataFrame({"margin": q, "error": wrong}).groupby("margin", observed=True)["error"].agg(error_rate="mean", n="size").reset_index()
    # --- volatility environment (VIX terciles from TRAIN rows)
    if "vix" in df:
        v = df["vix"].reindex(data.index); lo, hi = v.iloc[R["idx"]["train"]].quantile([1 / 3, 2 / 3])
        vb = pd.cut(v.iloc[ix], [-np.inf, lo, hi, np.inf], labels=[f"VIX low (<{lo:.1f})", f"VIX mid ({lo:.1f}-{hi:.1f})", f"VIX high (>{hi:.1f})"])
        out["by_vix"] = pd.DataFrame({"vix_bucket": vb.to_numpy(), "error": wrong}).groupby("vix_bucket", observed=True)["error"].agg(error_rate="mean", n="size").reset_index()
    # --- calendar
    out["by_year"] = pd.DataFrame({"year": dates.year, "error": wrong, "persist_err": pers != y}).groupby("year").agg(model_error=("error", "mean"), persistence_error=("persist_err", "mean"), n=("error", "size")).reset_index()
    # --- confidence vs error
    cb = pd.cut(conf, [0, .4, .5, .6, .7, .8, .9, 1.0])
    out["by_confidence"] = pd.DataFrame({"confidence": cb.astype(str), "error": wrong, "conf": conf}).groupby("confidence").agg(error_rate=("error", "mean"), mean_conf=("conf", "mean"), n=("error", "size")).reset_index()
    hc = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "actual_next": [SHORT[i] for i in y], "predicted": [SHORT[i] for i in pred], "confidence": conf})[wrong]
    out["worst_errors"] = hc.sort_values("confidence", ascending=False).head(10).reset_index(drop=True)
    out["meta"] = dict(model=model, split=split, n=len(y), error_rate=float(wrong.mean()), persistence_error=float((pers != y).mean()))
    return out


def findings(E: dict) -> list[str]:
    """Plain-language takeaways generated from the numbers (nothing hard-coded)."""
    m, cd = E["meta"], E["change_detection"]; out = []
    out.append(f"On {m['n']} {m['split']} days the {m['model']} errs on {m['error_rate']:.1%} of days versus {m['persistence_error']:.1%} for persistence.")
    d = E["dimension"].set_index("error type")["share"]
    out.append(f"Error composition: {d['trend only']:.0%} trend-only, {d['volatility only']:.0%} volatility-only, {d['both']:.0%} both.")
    out.append(f"Regime changes: {cd['true_changes']} true changes, {cd['predicted_changes']} flagged, {cd['caught']} caught (recall {cd['recall']:.0%}, precision {cd['precision']:.0%}); persistence catches none by construction.")
    bm = E["by_margin"].set_index("margin")["error_rate"]
    if len(bm) >= 2: out.append(f"Errors concentrate near label thresholds: {bm.iloc[0]:.1%} error in the closest quartile vs {bm.iloc[-1]:.1%} in the farthest.")
    if "by_vix" in E:
        vb = E["by_vix"].set_index("vix_bucket")["error_rate"]; out.append("Error by VIX regime: " + ", ".join(f"{k} {v:.1%}" for k, v in vb.items()) + ".")
    tc = E["top_confusions"]
    if len(tc): out.append(f"Most frequent confusion: {tc.iloc[0]['actual']} -> {tc.iloc[0]['predicted']} ({tc.iloc[0]['count']} days, {tc.iloc[0]['share_of_errors']:.0%} of errors).")
    return out
