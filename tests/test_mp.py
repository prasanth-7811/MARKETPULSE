import sys, pathlib
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mp import config as C
from mp.data import DataError, clean
from mp.features import build_features
from mp.regimes import CLASSES, RegimeLabeler, run_lengths
from mp.pipeline import days_in_regime, split_positions


def synth(n=1800, seed=0):
    """Regime-switching synthetic OHLCV used ONLY for tests (never shown in the app)."""
    rng = np.random.default_rng(seed)
    mu, sd, s, r = [0.0006, -0.0004, 0.0001], [0.007, 0.018, 0.011], 0, []
    for _ in range(n):
        if rng.random() < 0.02: s = rng.integers(0, 3)
        r.append(rng.normal(mu[s], sd[s]))
    p = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"open": p, "high": p * 1.004, "low": p * 0.996, "close": p, "adj_close": p, "volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)


CFG = C.load_defaults(); CFG["data"]["end"] = "2030-01-01"


def test_default_config_valid(): assert C.validate(CFG) == []


def test_validation_catches_bad_split():
    bad = C.deep_update(CFG, {"split": {"train": 0.9, "val": 0.2, "test": 0.2}})
    assert any("sum to 1" in e for e in C.validate(bad))


def test_clean_removes_duplicates_and_invalid():
    df = synth(50); df = pd.concat([df, df.iloc[[3]]]); df.iloc[5, df.columns.get_loc("close")] = -1
    out, q = clean(df)
    assert q["duplicate_dates"] == 1 and q["invalid_rows"] >= 1 and out.index.is_unique and (out["close"] > 0).all()


def test_clean_rejects_empty():
    with pytest.raises(DataError): clean(pd.DataFrame({"close": [-1.0]}, index=pd.to_datetime(["2020-01-01"])))


def test_features_are_causal():
    df = synth(400); f1, _ = build_features(df, CFG)
    df2 = df.copy(); df2.iloc[300:, :] = df2.iloc[300:, :] * 1.5  # change the future
    f2, _ = build_features(df2, CFG)
    pd.testing.assert_frame_equal(f1.iloc[:300], f2.iloc[:300])


def test_labeler_thresholds_use_train_only_and_cover_classes():
    df = synth(); lb = RegimeLabeler(60, (0.3, 0.7)); sig = lb.signals(df["adj_close"])
    lb.fit(sig.iloc[:1000]); lab = lb.label(sig).dropna()
    assert set(lab).issubset(CLASSES) and len(set(lab)) >= 4
    lb2 = RegimeLabeler(60, (0.3, 0.7)).fit(sig.iloc[:1000]); assert lb2.hi == lb.hi


def test_split_has_embargo_and_order():
    s = split_positions(1000, CFG, 20).to_numpy()
    first = {k: np.where(s == k)[0] for k in ("train", "validation", "test")}
    assert first["train"].max() < first["validation"].min() < first["validation"].max() < first["test"].min()
    assert (s == "embargo").sum() == 40


def test_days_in_regime():
    assert days_in_regime(np.array([1, 1, 2, 2, 2, 1])).tolist() == [1, 2, 1, 2, 3, 1]


def test_run_lengths():
    r = run_lengths(pd.Series(list("aabbbc"))); assert r.duration.tolist() == [2, 3, 1]


def test_hmm_filtering_is_causal():
    pytest.importorskip("hmmlearn")
    from mp.regimes import FilteredHMM
    rng = np.random.default_rng(1); X = np.r_[rng.normal(0, 1, (300, 2)), rng.normal(3, 1, (300, 2))]
    h = FilteredHMM(2, "full", 50, 0).fit(X); f_full, _ = h.filter(X); f_part, _ = h.filter(X[:400])
    np.testing.assert_allclose(f_full[:400], f_part, atol=1e-9)
    np.testing.assert_allclose(f_full.sum(1), 1, atol=1e-9)


def test_temperature_scaling_does_not_increase_val_loss():
    from mp.models import apply_temperature, fit_temperature
    rng = np.random.default_rng(0); y = rng.integers(0, 3, 500)
    logits = np.eye(3)[y] * 1.0 + rng.normal(0, 1, (500, 3)); P = np.exp(logits * 4); P /= P.sum(1, keepdims=True)  # overconfident
    T = fit_temperature(P, y); nll = lambda Q: -np.log(Q[np.arange(500), y]).mean()
    assert nll(apply_temperature(P, T)) <= nll(P) + 1e-9 and T > 1


def test_pipeline_end_to_end_smoke():
    for m in ("hmmlearn", "lightgbm", "xgboost"): pytest.importorskip(m)
    from mp.pipeline import run_pipeline
    cfg = C.deep_update(CFG, {"models": {"selected": ["Logistic Regression", "LightGBM"]}, "hmm": {"n_iter": 30}})
    R = run_pipeline(synth(), cfg)
    assert {"Calibrated hybrid", "HMM-based approach", "Persistence baseline"} <= set(R["proba"])
    for P in R["proba"].values(): np.testing.assert_allclose(P.sum(1), 1, atol=1e-6)
    assert set(R["metrics"].split) == {"train", "validation", "test"}
    assert all(e["idx"]["train"].max() < e["idx"]["validation"].min() for e in R["transition"].values() if e.get("model"))
