"""Configuration loading, merging and validation."""
from __future__ import annotations
import copy
from datetime import date
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_defaults() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate(cfg: dict) -> list[str]:
    e: list[str] = []
    f, s, h, t, c = cfg["features"], cfg["split"], cfg["hmm"], cfg["transition"], cfg["calibration"]
    if not 2 <= int(f["return_window"]) <= 60: e.append("Return window must be 2-60.")
    for k in ("vol_windows", "mom_windows"):
        w = f[k]
        if not w or any(int(x) < 2 or int(x) > 250 for x in w): e.append(f"{k}: values must be integers in 2-250.")
        if len(set(w)) != len(w): e.append(f"{k}: duplicate windows.")
    if abs(s["train"] + s["val"] + s["test"] - 1) > 1e-6: e.append("Train/validation/test proportions must sum to 1.")
    if min(s["train"], s["val"], s["test"]) < 0.1: e.append("Each split must be at least 10%.")
    if int(s["cv_splits"]) < 2: e.append("Walk-forward CV needs at least 2 splits.")
    if not 2 <= int(h["n_states"]) <= 6: e.append("HMM states must be 2-6.")
    if h["covariance_type"] not in ("full", "diag", "tied", "spherical"): e.append("Invalid covariance type.")
    if not cfg["models"]["selected"]: e.append("Select at least one model.")
    if not t["horizons"] or any(int(x) < 1 for x in t["horizons"]): e.append("Horizons must be positive integers.")
    if not 0 < float(t["threshold"]) < 1: e.append("Classification threshold must be in (0, 1).")
    if not 0.2 <= float(c["confidence_threshold"]) <= 1: e.append("Confidence threshold must be in [0.2, 1].")
    if cfg["labels"]["window"] < 10: e.append("Label window must be >= 10.")
    q = cfg["labels"]["trend_quantiles"]
    if not 0 < q[0] < q[1] < 1: e.append("Trend quantiles must satisfy 0 < low < high < 1.")
    try:
        if date.fromisoformat(str(cfg["data"]["start"])) >= date.fromisoformat(str(cfg["data"]["end"])):
            e.append("Start date must precede end date.")
    except Exception:
        e.append("Invalid dates.")
    return e
