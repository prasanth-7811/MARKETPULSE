"""Model bundle (save/load) and a stateless Predictor used by the FastAPI service and the CLI.

A bundle holds everything needed at inference time: config, fitted HMM, label thresholds, calibrated ensemble,
per-horizon transition models + calibrators, feature column lists and held-out test metrics for the model card.
Inference never refits anything and uses causal (filtered) HMM probabilities, so a prediction for day t uses data <= t.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from mp import config as C
from mp.data import AUX, clean
from mp.features import build_features
from mp.pipeline import days_in_regime
from mp.regimes import CLASSES

BUNDLE_VERSION = 2
MIN_ROWS = 320          # 200-day SMA + 60-day label window + margin for the HMM filter to forget its prior
DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / "artifacts" / "model_bundle.joblib"
DISCLAIMER = "Educational research output. Not financial advice, not a forecast, not a trading system."


class InferenceError(ValueError):
    pass


def make_bundle(R: dict, cfg: dict, df: pd.DataFrame, test_metrics: dict | None = None) -> dict:
    tm = R["metrics"][R["metrics"].split == "test"].set_index("model")
    trans = {int(h): dict(model=e["model"], cal=e["cal"], cal_method=e["cal_method"]) for h, e in R["transition"].items() if e.get("model") is not None}
    return dict(version=BUNDLE_VERSION, cfg=cfg, classes=CLASSES, hmm=R["hmm"], hmm_cols=R["hmm_cols"], labeler=R["labeler"], state_names=R["state_names"],
                model=R["models"]["Calibrated hybrid"], temperature=R["temperature"], transition=trans, feature_cols=R["feature_cols"],
                aux_used=[c for c in AUX if c in df.columns and any(f.startswith(c) or f.startswith("vrp") for f in R["feature_cols"]["base"])],
                trained_on=dict(first=str(df.index[0].date()), last=str(df.index[-1].date()), rows=int(len(df))),
                splits={k: [str(R["data"].index[v[0]].date()), str(R["data"].index[v[-1]].date()), int(len(v))] for k, v in R["idx"].items()},
                test_metrics={k: float(v) for k, v in tm.loc["Calibrated hybrid", ["accuracy", "macro_f1", "log_loss", "brier", "ece"]].items()},
                persistence_test_metrics={k: float(v) for k, v in tm.loc["Persistence baseline", ["accuracy", "macro_f1", "log_loss", "brier", "ece"]].items()},
                extra=test_metrics or {})


def save_bundle(bundle: dict, path: str | Path = DEFAULT_BUNDLE) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


class Predictor:
    def __init__(self, bundle: dict):
        if bundle.get("version") != BUNDLE_VERSION: raise InferenceError("Incompatible model bundle version; retrain with train.py")
        self.b, self.threshold = bundle, float(bundle["cfg"]["calibration"]["confidence_threshold"])

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BUNDLE) -> "Predictor":
        p = Path(path)
        if not p.exists(): raise FileNotFoundError(f"Model bundle not found at {p}. Train one with: python train.py")
        return cls(joblib.load(p))

    def info(self) -> dict:
        b = self.b
        return dict(bundle_version=b["version"], classes=b["classes"], trained_on=b["trained_on"], splits=b["splits"], min_history_rows=MIN_ROWS,
                    required_columns=["date", "open", "high", "low", "close", "volume"] + [f"{c} (needed)" for c in b["aux_used"]],
                    horizons=sorted(b["transition"]), n_features=len(b["feature_cols"]["aug"]), temperature=b["temperature"],
                    test_metrics=b["test_metrics"], persistence_test_metrics=b["persistence_test_metrics"],
                    transition_calibration={h: v["cal_method"] for h, v in b["transition"].items()}, disclaimer=DISCLAIMER)

    def predict(self, raw: pd.DataFrame) -> dict:
        """raw: OHLCV (+ vix/brent when the model uses them) indexed by date, ascending, >= MIN_ROWS rows."""
        b = self.b; cfg = b["cfg"]
        try:
            df, q = clean(raw)
        except Exception as exc:
            raise InferenceError(f"Could not clean input: {exc}") from exc
        missing = [c for c in b["aux_used"] if c not in df or df[c].isna().all()]
        if missing: raise InferenceError(f"This model needs column(s) {missing} in the input history.")
        if len(df) < MIN_ROWS: raise InferenceError(f"Need at least {MIN_ROWS} daily rows of history; got {len(df)}.")
        feats, _ = build_features(df, cfg)
        lab = b["labeler"]; sig = lab.signals(df["adj_close"])
        data = feats.join(sig)
        ok = data[b["feature_cols"]["base"] + ["sig_mom", "sig_vol"]].notna().all(axis=1)
        data = data[ok]
        if len(data) < 60: raise InferenceError("Too few usable rows after feature warm-up (NaNs in input?).")
        regime = lab.label(data[["sig_mom", "sig_vol"]]); y = regime.map({c: i for i, c in enumerate(CLASSES)}).to_numpy().astype(int)
        hmm, hc = b["hmm"], b["hmm_cols"]
        filt, nxt = hmm.filter(data[hc].to_numpy())
        K = hmm.K; Hdf = pd.DataFrame(np.hstack([filt, nxt]), index=data.index, columns=[f"hmm_p{k}" for k in range(K)] + [f"hmm_n{k}" for k in range(K)])
        Xa = data[b["feature_cols"]["base"]].join(Hdf)
        P = b["model"].predict_proba(Xa.iloc[[-1]])[0]
        order = np.argsort(-P); conf = float(P[order[0]])
        Xt = Xa.join(pd.DataFrame(np.eye(len(CLASSES))[y], index=data.index, columns=[f"regime_{i}" for i in range(len(CLASSES))]))
        Xt["days_in_regime"] = days_in_regime(y)
        trans = {}
        for h, t in sorted(b["transition"].items()):
            p = float(t["cal"].transform(t["model"].predict_proba(Xt.iloc[[-1]])[:, 1])[0]); trans[str(h)] = round(min(max(p, 0.0), 1.0), 4)
        names = b["state_names"]; asof = data.index[-1].date().isoformat()
        sp = b["splits"]                                      # boundary-based: the final row has no t+1 label, so it is not listed in a split
        period = ("train" if asof <= sp["train"][1] else "embargo" if asof < sp["validation"][0] else "validation" if asof <= sp["validation"][1]
                  else "embargo" if asof < sp["test"][0] else "test" if asof <= b["trained_on"]["last"] else "after_training_data")
        return dict(as_of=asof, data_split=period, in_sample=period in ("train", "validation", "embargo"), history_rows=int(len(df)),
                    current_regime=CLASSES[y[-1]], days_in_current_regime=int(Xt["days_in_regime"].iloc[-1]),
                    hmm_state=names[int(filt[-1].argmax())], hmm_state_probabilities={n: round(float(p), 4) for n, p in zip(names, filt[-1])},
                    next_day_regime_probabilities={CLASSES[i]: round(float(P[i]), 4) for i in order},
                    predicted_next_regime=CLASSES[order[0]], confidence=round(conf, 4), low_confidence=conf < self.threshold,
                    regime_change_probability=trans, trained_through=b["trained_on"]["last"], disclaimer=DISCLAIMER)
