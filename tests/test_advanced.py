"""Data engineering, dimensionality reduction, tuning, tracking, benchmarking and error-analysis tests."""
import json
import numpy as np, pandas as pd, pytest
from mp import benchmarks as B, errors as E
from mp.features import build_features
from mp.models import ProbCal
from mp.reduce import PCAReducer
from mp.tracking import Tracker


def test_bundled_dataset_is_real_and_clean(bundled, df):
    assert len(df) > 7000 and df.index.is_unique and df.index.is_monotonic_increasing
    assert {"vix", "brent"} <= set(df.columns) and (df["vix"] > 5).all()
    assert df.index[0].year == 1990 and df.index[-1].year == 2018
    assert (df["high"] >= df["low"]).all()


def test_all_features_causal_including_advanced_and_cross_asset(df, cfg):
    f1, g = build_features(df, cfg)
    assert {"advanced_vol", "trend", "cross_asset"} <= set(g)
    df2 = df.copy(); df2.iloc[3000:, :] = df2.iloc[3000:, :] * 1.7           # corrupt the future of every column (incl. vix, brent)
    f2, _ = build_features(df2, cfg)
    pd.testing.assert_frame_equal(f1.iloc[:3000], f2.iloc[:3000])


def test_feature_toggles(df, cfg):
    from mp import config as C
    f, g = build_features(df, C.deep_update(cfg, {"features": {"advanced": False, "use_cross_asset": False}}))
    assert "advanced_vol" not in g and "cross_asset" not in g and f.shape[1] < 20


def test_pca_fitted_on_train_only_and_orthogonal(R):
    tr, te = R["idx"]["train"], R["idx"]["test"]
    red = PCAReducer(0.95).fit(R["Xb"].iloc[tr]); Z = red.transform(R["Xb"])
    X2 = R["Xb"].copy(); X2.iloc[te] = X2.iloc[te] * 3                         # change test rows
    red2 = PCAReducer(0.95).fit(X2.iloc[tr])
    np.testing.assert_allclose(red.pca.components_, red2.pca.components_)      # projection independent of test data
    assert red.k < R["Xb"].shape[1] and np.cumsum(red.explained_)[red.k - 1] >= 0.95 - 1e-9
    C = np.cov(Z.iloc[tr].T); np.testing.assert_allclose(C - np.diag(np.diag(C)), 0, atol=1e-6)


def test_probcal_auto_never_worse_than_none_on_its_own_validation():
    rng = np.random.default_rng(0); p = rng.uniform(0.05, 0.95, 600); y = (rng.uniform(size=600) < p).astype(int)
    pc = ProbCal("auto").fit(p, y); assert pc.method in ("none", "platt", "isotonic") and "auto-selected" in pc.note
    out = pc.transform(p); assert out.shape == p.shape and (0 <= out).all() and (out <= 1).all()


def test_probcal_few_events_falls_back():
    pc = ProbCal("auto").fit(np.linspace(0, 1, 100), np.r_[np.ones(5), np.zeros(95)].astype(int)); assert pc.method == "none"


def test_optuna_tuning_walk_forward_smoke(R):
    pytest.importorskip("optuna")
    from mp.tuning import tune
    tr = R["idx"]["train"][:1500]
    bp, tdf, st = tune("Logistic Regression", R["Xb"].iloc[tr], R["y_next"][tr], R["feature_cols"]["base"], 6, n_trials=4, cv_splits=3, gap=20, seed=1)
    assert "C" in bp and len(tdf) == 4 and tdf.value.notna().any()


def test_tracker_roundtrip(tmp_path):
    t = Tracker(tmp_path)
    with t.start("demo", kind="unit") as r: r.log_params(a=1, arr=np.arange(3)); r.log_metrics(x=np.float64(0.5))
    with pytest.raises(RuntimeError):
        with t.start("boom") as r: raise RuntimeError("x")
    h = t.history(); assert len(h) == 2 and h.loc[0, "metrics.x"] == 0.5 and h.loc[1, "status"] == "failed"
    json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])


def test_block_bootstrap_preserves_length_and_blocks():
    ix = B.block_indices(100, 20, np.random.default_rng(0)); assert len(ix) == 100
    assert (np.diff(ix[:20]) == 1).all()                                       # contiguous within a block


def test_paired_bootstrap_detects_difference():
    y = np.random.default_rng(1).integers(0, 2, 400); good = lambda ix: float((y[ix] == y[ix]).mean()); bad = lambda ix: float((y[ix] == 1 - y[ix]).mean())
    r = B.paired_boot(good, bad, 400, B=50); assert r["delta"] == 1.0 and r["p"] <= 0.05


def test_benchmarks_and_errors_end_to_end(R, df):
    pytest.importorskip("arch")
    res = B.run_all(R, df, B=20)
    assert {"Calibrated hybrid", "Persistence baseline", "MLP (neural network)", "Majority class", "PCA + Logistic Regression"} <= set(res["main"].model)
    assert res["main"].macro_f1_lo.le(res["main"].macro_f1).all() and res["main"].macro_f1_hi.ge(res["main"].macro_f1).all()
    assert any("GARCH" in m for m in res["volatility"].model) and any("SMA" in m for m in res["trend"].model)
    assert set(res["transition"].horizon) == set(R["transition"]) and len(res["literature"]) >= 8
    Ea = E.error_analysis(R, df, P=res["proba_all"])
    assert Ea["confusion"].to_numpy().sum() == len(R["idx"]["test"]) and abs(Ea["dimension"]["share"].sum() - 1) < 1e-9
    assert Ea["change_detection"]["caught"] <= Ea["change_detection"]["true_changes"]
    assert len(E.findings(Ea)) >= 4


def test_hmm_state_names_rank_based():
    from mp.regimes import label_states
    st = pd.DataFrame({"mom": [0.02, -0.03, 0.001], "vol": [0.1, 0.4, 0.15]}); n = label_states(st)
    assert n[0].startswith("Bullish") and n[1].startswith("Bearish") and n[2].startswith("Sideways") and "High" in n[1]
