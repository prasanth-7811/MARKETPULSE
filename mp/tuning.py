"""Hyperparameter optimisation with Optuna (TPE sampler + median pruner), scored by walk-forward CV.

Leakage controls: (1) only TRAINING rows are ever passed in; (2) folds are expanding-window and a `gap`
(= embargo) separates each training fold from its validation fold; (3) the test split is never seen.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from mp.models import Clf, make_estimator

SPACES = {
    "Logistic Regression": lambda t: dict(C=t.suggest_float("C", 1e-3, 1e2, log=True)),
    "Random Forest": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 100, 500, step=50), max_depth=t.suggest_int("max_depth", 3, 14),
                                    min_samples_leaf=t.suggest_int("min_samples_leaf", 2, 30), max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5])),
    "XGBoost": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 100, 500, step=50), max_depth=t.suggest_int("max_depth", 2, 7),
                              learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                              colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0), reg_lambda=t.suggest_float("reg_lambda", 1e-2, 20, log=True)),
    "LightGBM": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 100, 500, step=50), num_leaves=t.suggest_int("num_leaves", 5, 48),
                               learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                               colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0), min_child_samples=t.suggest_int("min_child_samples", 5, 60),
                               reg_lambda=t.suggest_float("reg_lambda", 1e-2, 20, log=True)),
}


def _score(metric: str, y, P, pred) -> float:
    if metric == "neg_log_loss":
        Pn = np.clip(P, 1e-6, 1); Pn /= Pn.sum(1, keepdims=True)
        return -log_loss(y, Pn, labels=list(range(P.shape[1])))
    return f1_score(y, pred, average="macro", zero_division=0)


def tune(name: str, X: pd.DataFrame, y: np.ndarray, cols: list[str], n_classes: int, *, n_trials: int = 25, cv_splits: int = 4,
         gap: int = 20, metric: str = "macro_f1", seed: int = 42, timeout: float | None = None):
    """Returns (best_params, trials_df, study). Maximises the chosen walk-forward CV metric."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(TimeSeriesSplit(cv_splits, gap=gap).split(X))

    def objective(trial: "optuna.Trial") -> float:
        params, scores = SPACES[name](trial), []
        for i, (tr, va) in enumerate(folds):
            try:
                m = Clf(name, make_estimator(name, seed, params), cols, n_classes).fit(X.iloc[tr], y[tr])
                P = m.predict_proba(X.iloc[va]); scores.append(_score(metric, y[va], P, P.argmax(1)))
            except ValueError:
                continue
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune(): raise optuna.TrialPruned()
        if not scores: raise optuna.TrialPruned()
        trial.set_user_attr("cv_std", float(np.std(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=False),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    t0 = time.time(); study.optimize(objective, n_trials=n_trials, timeout=timeout)
    rows = [dict(trial=t.number, state=t.state.name, value=t.value, cv_std=t.user_attrs.get("cv_std"), **t.params,
                 seconds=round((t.datetime_complete - t.datetime_start).total_seconds(), 2) if t.datetime_complete else None) for t in study.trials]
    df = pd.DataFrame(rows)
    return dict(study.best_params), df, study
