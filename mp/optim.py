"""Optuna-based hyperparameter optimisation.

Warp the same walk-forward CV used in ``mp.pipeline`` into an Optuna study,
maximising mean macro F1 across the training split. Best parameters can be
applied live to the Streamlit session and retrained."""
from __future__ import annotations
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from mp.models import Clf, make_estimator

# Continuous/expansive search space (wider than the fixed grid for Optuna).
ALGO_BOUNDS = {
    "Logistic Regression": {"C": (1e-3, 3e2, True)},
    "Random Forest": {"n_estimators": (50, 800, False), "max_depth": (2, 20, False), "min_samples_leaf": (1, 30, False)},
    "XGBoost": {"n_estimators": (50, 800, False), "max_depth": (2, 8, False),
                "learning_rate": (1e-3, 0.3, True), "subsample": (0.5, 1.0, False),
                "colsample_bytree": (0.5, 1.0, False), "min_child_weight": (1, 20, False)},
    "LightGBM": {"n_estimators": (50, 800, False), "num_leaves": (7, 63, False),
                 "learning_rate": (1e-3, 0.3, True), "subsample": (0.5, 1.0, False),
                 "colsample_bytree": (0.5, 1.0, False), "min_child_samples": (5, 80, False)},
}


def optuna_search(name: str, X, y, cols, n_classes: int, n_trials: int = 30,
                  cv_splits: int = 3, seed: int = 42) -> tuple[dict, object, pd.DataFrame]:
    """Run an Optuna TPE search (maximise mean walk-forward macro F1).

    Returns (best_params, study, trials DataFrame with CV scores)."""
    space = ALGO_BOUNDS.get(name, {})
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for param_name, (lo, hi, log) in space.items():
            params[param_name] = trial.suggest_float(param_name, lo, hi, log=log) if log \
                else trial.suggest_int(param_name, int(lo), int(hi))
        est = make_estimator(name, seed, params)
        scores = []
        for tr, va in tscv.split(X):
            try:
                m = Clf(name, est, cols, n_classes).fit(X.iloc[tr], y[tr])
                scores.append(f1_score(y[va], m.predict(X.iloc[va]), average="macro", zero_division=0))
            except Exception:  # infeasible hyperparameter combo in a fold
                return 0.0
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"marketpulse_{name}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = []
    for s in study.trials:
        if s.state != optuna.trial.TrialState.COMPLETE:
            continue
        params = s.params
        trials.append({
            "Trial": int(s.number + 1),
            **{f"HP_{k}": (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()},
            "CV mean macro F1": round(float(s.value), 4),
            "Best val so far": round(float(study.best_value), 4),
        })
    return dict(study.best_params), study, pd.DataFrame(trials)


def apply_best(study: optuna.study.Study, model_name: str) -> None:
    """Push the best trial's parameters into the Streamlit session state so the
    next 'Apply and retrain' uses them."""
    import streamlit as st
    if not st.session_state.get("cfg_over"):
        st.session_state["cfg_over"] = {}
    st.session_state["cfg_over"]["models"] = {
        "params": {model_name: {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                                for k, v in study.best_params.items()}},
    }
