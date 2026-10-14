"""Lightweight experiment tracking: one JSON record per run in <dir>/runs.jsonl (+ optional MLflow mirror).

A record holds: run id, UTC time, config hash, data fingerprint, package versions, parameters, metrics,
artifact paths, tags and wall-clock seconds. Human-readable, diff-able and committed with the repo.
"""
from __future__ import annotations
import hashlib, json, platform, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd


def _json_safe(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    if isinstance(o, (pd.Timestamp,)): return str(o)
    return str(o)


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12]


def versions() -> dict:
    out = {"python": platform.python_version()}
    for m in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "hmmlearn", "optuna", "shap"):
        try:
            from importlib.metadata import version
            out[m] = version(m)
        except Exception:
            pass
    return out


class Run:
    def __init__(self, tracker: "Tracker", name: str, tags: dict | None):
        self.t, self.rec = tracker, dict(run_id=uuid.uuid4().hex[:10], name=name, started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         tags=tags or {}, params={}, metrics={}, artifacts=[], versions=versions())
        self._t0 = time.time()

    def log_params(self, **kw): self.rec["params"].update(kw)
    def log_metrics(self, **kw): self.rec["metrics"].update(kw)
    def log_artifact(self, path): self.rec["artifacts"].append(str(path))

    def __enter__(self): return self

    def __exit__(self, et, ev, tb):
        self.rec.update(seconds=round(time.time() - self._t0, 2), status="failed" if et else "finished", error=repr(ev) if et else None)
        self.t._write(self.rec)
        return False


class Tracker:
    def __init__(self, directory: str | Path = "experiments", mlflow: bool = False):
        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)
        self.file, self.use_mlflow = self.dir / "runs.jsonl", mlflow

    def start(self, name: str, **tags) -> Run: return Run(self, name, tags)

    def _write(self, rec: dict):
        with self.file.open("a") as fh: fh.write(json.dumps(rec, default=_json_safe) + "\n")
        if self.use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(f"file:{(self.dir / 'mlruns').resolve()}")
                with mlflow.start_run(run_name=rec["name"]):
                    mlflow.log_params({k: str(v)[:250] for k, v in rec["params"].items()})
                    mlflow.log_metrics({k: float(v) for k, v in rec["metrics"].items() if isinstance(v, (int, float)) and v is not None})
            except Exception:
                pass

    def history(self) -> pd.DataFrame:
        if not self.file.exists(): return pd.DataFrame()
        rows = [json.loads(l) for l in self.file.read_text().splitlines() if l.strip()]
        return pd.json_normalize(rows)
