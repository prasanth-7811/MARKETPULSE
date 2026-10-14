#!/usr/bin/env python
"""Automated, reproducible training pipeline:  data -> features -> (Optuna tuning) -> models -> benchmarks -> errors -> artifacts.

Examples
  python train.py                       # default hyper-parameters, bundled dataset, writes artifacts/ + docs/figures/
  python train.py --tune --trials 25    # Optuna walk-forward tuning per algorithm, keep-if-better on VALIDATION
  python train.py --data my.csv         # your own OHLCV CSV (Date, Open, High, Low, Close, [Adj Close, Volume, VIX])
  python train.py --quick               # smoke run (few HMM iterations, small bootstrap)

Everything is seeded (config split.seed). Each run appends a record to experiments/runs.jsonl.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from mp import benchmarks, config as C, errors, figures, reduce
from mp.data import clean, load_bundled, load_csv
from mp.inference import make_bundle, save_bundle
from mp.models import ALGOS
from mp.pipeline import run_pipeline
from mp.regimes import CLASSES
from mp.tracking import Tracker, fingerprint

ROOT = Path(__file__).resolve().parent


def metric_row(R: dict, model: str = "Calibrated hybrid") -> dict:
    m = R["metrics"]; out = {}
    for s in ("validation", "test"):
        r = m[(m.model == model) & (m.split == s)].iloc[0]
        out.update({f"{s}_{k}": float(r[k]) for k in ("accuracy", "macro_f1", "log_loss", "brier", "ece")})
    return out


def tune_all(R: dict, cfg: dict, tracker: Tracker, out: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Optuna per algorithm on TRAIN rows (walk-forward CV with embargo gap). Keep tuned params only if VALIDATION log loss improves."""
    from mp.tuning import tune
    from mp.models import Clf, make_estimator, multiclass_metrics
    t, seed = cfg["tuning"], int(cfg["split"]["seed"]); tr, va = R["idx"]["train"], R["idx"]["validation"]
    X, y, cols = R["Xb"], R["y_next"], R["feature_cols"]["base"]
    best, rows, studies = {}, [], {}
    for name in [m for m in cfg["models"]["selected"] if m in ALGOS]:
        with tracker.start(f"optuna_{name}", kind="tuning", algo=name) as run:
            bp, tdf, st = tune(name, X.iloc[tr], y[tr], cols, len(CLASSES), n_trials=int(t["n_trials"]), cv_splits=int(t["cv_splits"]), gap=R["embargo"],
                               metric=t["metric"], seed=seed, timeout=float(t["timeout_s"]))
            tdf.to_csv(out / "tables" / f"optuna_{name.replace(' ', '_')}.csv", index=False)
            studies[name] = (tdf, "maximize")
            def val(params):
                m = Clf(name, make_estimator(name, seed, params), cols, len(CLASSES)).fit(X.iloc[tr], y[tr])
                return multiclass_metrics(y[va], m.predict_proba(X.iloc[va]))
            d, b = val(None), val(bp); keep = b["log_loss"] < d["log_loss"]
            if keep: best[name] = bp
            rows.append(dict(model=name, cv_best=st.best_value, trials=len(tdf), pruned=int((tdf.state == "PRUNED").sum()), val_logloss_default=d["log_loss"], val_logloss_tuned=b["log_loss"],
                             val_f1_default=d["macro_f1"], val_f1_tuned=b["macro_f1"], adopted=bool(keep), best_params=json.dumps(bp)))
            run.log_params(algo=name, n_trials=len(tdf), metric=t["metric"], cv_splits=t["cv_splits"], best_params=bp)
            run.log_metrics(cv_best=st.best_value, val_logloss_default=d["log_loss"], val_logloss_tuned=b["log_loss"], adopted=bool(keep))
    return best, pd.DataFrame(rows), studies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml")); ap.add_argument("--data", default="bundled", help="'bundled' or path to a CSV")
    ap.add_argument("--tune", action="store_true"); ap.add_argument("--trials", type=int); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts")); ap.add_argument("--figs", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--no-bench", action="store_true"); ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    out, figs = Path(a.out), Path(a.figs); (out / "tables").mkdir(parents=True, exist_ok=True)
    import yaml
    cfg = C.deep_update(C.load_defaults(), yaml.safe_load(Path(a.config).read_text()) or {})
    cfg["data"]["end"] = "2100-01-01"
    if a.quick: cfg = C.deep_update(cfg, {"hmm": {"n_iter": 30}, "tuning": {"n_trials": 4, "timeout_s": 120}})
    if a.trials: cfg["tuning"]["n_trials"] = a.trials
    errs = C.validate(cfg)
    if errs: raise SystemExit("Invalid config: " + " ".join(errs))
    raw = load_bundled() if a.data == "bundled" else load_csv(Path(a.data).read_bytes())
    df, q = clean(raw); df = df.loc[str(cfg["data"]["start"]):]
    cfg["data"]["ticker"] = "S&P 500 (bundled)" if a.data == "bundled" else Path(a.data).stem
    tracker = Tracker(ROOT / cfg["tracking"]["dir"], bool(cfg["tracking"]["mlflow"]))
    data_fp = fingerprint(df); print(f"[data] {len(df)} rows {df.index[0].date()}..{df.index[-1].date()} aux={q['aux_columns']} fp={data_fp}")

    with tracker.start("baseline_default_params", kind="train", data=data_fp) as run:
        R0 = run_pipeline(df, cfg); run.log_params(config=cfg, data_rows=len(df)); run.log_metrics(**metric_row(R0))
    print("[baseline]", {k: round(v, 4) for k, v in metric_row(R0).items() if k.startswith("test")})

    best, tune_df, studies, R = {}, pd.DataFrame(), {}, R0
    if a.tune:
        best, tune_df, studies = tune_all(R0, cfg, tracker, out); tune_df.to_csv(out / "tables" / "tuning_summary.csv", index=False)
        (out / "best_params.json").write_text(json.dumps(best, indent=2)); print("[tuning]\n", tune_df.drop(columns="best_params").round(4).to_string())
        cfg = C.deep_update(cfg, {"models": {"params": best}})
        with tracker.start("final_tuned_params", kind="train", data=data_fp) as run:
            R = run_pipeline(df, cfg); run.log_params(models_params=best); run.log_metrics(**metric_row(R))
        eff = pd.DataFrame([dict(config="default hyper-parameters", **metric_row(R0)), dict(config="Optuna-tuned (validation-gated)", **metric_row(R))]); eff.to_csv(out / "tables" / "tuning_effect.csv", index=False)
        print("[tuned]", {k: round(v, 4) for k, v in metric_row(R).items() if k.startswith("test")})

    R["metrics"].to_csv(out / "tables" / "metrics_all_splits.csv", index=False); R["ablation"].to_csv(out / "tables" / "ablation.csv", index=False)
    Z = None
    with tracker.start("analysis", kind="analysis", data=data_fp) as run:
        tr = R["idx"]["train"]; red = reduce.PCAReducer(cfg["reduction"]["pca_variance"], seed=cfg["split"]["seed"]).fit(R["Xb"].iloc[tr]); Z = red.transform(R["Xb"])
        sil = reduce.silhouette_by_regime(Z.iloc[:, :red.k], R["labels"].reindex(Z.index))
        red.scree().to_csv(out / "tables" / "pca_scree.csv", index=False); red.loadings(5).to_csv(out / "tables" / "pca_loadings.csv")
        run.log_metrics(pca_components_95=red.k, pca_pc1=float(red.explained_[0]), regime_silhouette_pca=sil)
        bench = {}; E = None
        if not a.no_bench:
            bench = benchmarks.run_all(R, df, int(cfg["split"]["seed"]), B=60 if a.quick else 300)
            for k, v in bench.items():
                if isinstance(v, pd.DataFrame): v.to_csv(out / "tables" / f"bench_{k}.csv", index=False)
            E = errors.error_analysis(R, df, P=bench["proba_all"])
            for k, v in E.items():
                if isinstance(v, pd.DataFrame): v.to_csv(out / "tables" / f"errors_{k}.csv", index=False)
            (out / "error_findings.txt").write_text("\n".join(errors.findings(E)))
            run.log_metrics(**{f"hybrid_test_{k}": v for k, v in metric_row(R).items() if k.startswith("test")})
            print("[errors]\n- " + "\n- ".join(errors.findings(E)))
    if not a.no_figs and E is not None:
        import matplotlib.pyplot as plt
        from mp.explain import shap_for, subsample
        emb = reduce.tsne_embed(((R["Xb"].iloc[tr] - R["Xb"].iloc[tr].mean()) / R["Xb"].iloc[tr].std()).clip(-5, 5), 1200, cfg["reduction"]["tsne_perplexity"], cfg["split"]["seed"])
        figures.price_regimes(R, df, figs); figures.correlation(R, figs); figures.pca_fig(R, red, Z.iloc[tr], figs); figures.tsne_fig(R, emb, figs)
        figures.confusion(E, figs); figures.reliability(R, E, bench["proba_all"], figs); figures.model_compare(bench["main"], figs); figures.error_fig(E, figs)
        figures.feature_ablation_fig(bench["feature_ablation"], figs); figures.hmm_fig(R, figs)
        try:
            xm = [m for m in R["models"].values() if getattr(m, "name", "") == "XGBoost"][0]; ix = subsample(R["idx"]["test"], 400)
            sv, present = shap_for(xm, R["Xb"].iloc[ix]); figures.importance_fig(pd.Series(np.abs(sv).mean((0, 2)), index=xm.cols), figs)
        except Exception as exc:
            print("[shap] skipped:", exc)
        if studies:
            fig, ax = figures.tuning_fig(studies, figs, tune_df)
            try:
                import optuna
                from optuna.importance import get_param_importances
                lg = studies.get("LightGBM")
            except Exception:
                pass
            fig.tight_layout(); fig.savefig(figs / "fig_tuning.png", bbox_inches="tight"); plt.close(fig)
        print(f"[figures] {figs}")

    bundle = make_bundle(R, cfg, df); path = save_bundle(bundle, out / "model_bundle.joblib")
    summary = dict(data=dict(rows=len(df), first=str(df.index[0].date()), last=str(df.index[-1].date()), fingerprint=data_fp), splits=bundle["splits"], temperature=R["temperature"],
                   label_thresholds=R["labeler"].describe(), hybrid=metric_row(R), pca=dict(k95=red.k, pc1=float(red.explained_[0]), silhouette=sil), best_params=best,
                   hmm_states=R["state_names"], runtime_s=round(time.time() - t0, 1))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    with tracker.start("bundle", kind="artifact") as run: run.log_artifact(path); run.log_metrics(**metric_row(R))
    print(f"[done] {time.time() - t0:.0f}s  bundle={path} ({path.stat().st_size / 1e6:.1f} MB)  runs log={tracker.file}")


if __name__ == "__main__":
    main()
