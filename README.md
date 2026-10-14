---
title: MarketPulse Regime API
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# MarketPulse - Financial Market Regime Intelligence (v2.0)

Leakage-controlled classification of six market regimes (Bullish/Sideways/Bearish x Low/High volatility) and calibrated regime-change probabilities (5/10/20 days), with a FastAPI inference service, Docker image, Streamlit dashboard and IEEE report.
Educational research tool - not financial advice, not a forecast, not a trading system. Daily data only.

**Live demo:** _(add your Hugging Face Space URL after deploying - see `docs/DEPLOYMENT.md`)_

## Headline results (test split 2013-05..2018-12, n = 1,420, real S&P 500 + VIX + Brent)
| Model | Accuracy | Macro-F1 | Log loss [95% CI] | ECE |
|---|---|---|---|---|
| Persistence baseline | **0.861** | **0.837** | 0.783 [0.655, 0.939] | 0.119 |
| **Calibrated hybrid (ours)** | 0.843 | 0.800 | **0.391** [0.335, 0.453] | **0.020** |
Persistence wins hard-label metrics; the hybrid wins probabilistic quality. HMM states, advanced and cross-asset features gave no measurable gain (reported in the ablations, not hidden). Full tables: `artifacts/tables/`, report: `report/`.

## Rubric map
| Area | Where |
|---|---|
| 1. Data engineering & visualization | `data/bundled/` (+ `DATA_CARD.md`), `mp/features.py` (33 causal features, 7 groups), `mp/reduce.py` (PCA, t-SNE), `mp/figures.py`, dashboard pages *Feature Lab*, *Data Explorer* |
| 2. Modeling & pipeline optimisation | `train.py` (automated pipeline), `mp/tuning.py` (Optuna, walk-forward + gap, pruning), `mp/tracking.py` (`experiments/runs.jsonl`, optional MLflow), `mp/models.py`, `mp/pipeline.py` |
| 3. Benchmarking & error analysis | `mp/benchmarks.py` (literature baselines, block-bootstrap CIs, paired tests), `mp/errors.py`, dashboard *Benchmarks & Errors*, `artifacts/tables/bench_*.csv`, `errors_*.csv` |
| 4. Deployment | `api/main.py` (FastAPI), `Dockerfile`, `docker-compose.yml`, `docs/DEPLOYMENT.md` (HF Spaces / AWS) |
| 5. Documentation | `report/main.tex` (IEEEtran), this README, `tests/` (30 tests), `.github/workflows/ci.yml`, `Makefile` |

## Quick start
```bash
pip install -r requirements.txt
pytest -q                               # 30 tests
python train.py --tune --trials 25      # regenerate artifacts/, docs/figures/, experiments/ (~17 min, 2 cores)
uvicorn api.main:app --port 8000        # API + demo page: http://localhost:8000  (/docs, /model, /demo)
streamlit run app.py                    # dashboard
docker build -t marketpulse-api . && docker run --rm -p 7860:7860 marketpulse-api
```
API example: `curl "localhost:8000/demo?as_of=2016-02-11"`; `POST /predict` takes >= 320 daily rows with date, open, high, low, close, volume, vix, brent (see `/model`).

## Method (summary)
- **Labels:** trend (trailing 60-day return quantiles, fit on train) x volatility (vs train median).
- **Splits:** chronological 60/20/20 with embargo >= max horizon; HMM, scaler, PCA, thresholds fit on train; calibration on validation; test never tuned on.
- **HMM:** Gaussian HMM, forward-**filtered** probabilities (no look-ahead; unit-tested).
- **Models:** LR, RF, XGBoost, LightGBM, HMM-based, soft-voting, HMM-augmented, temperature-calibrated hybrid; Optuna tuning gated on validation.
- **Transition task:** per-horizon LightGBM; calibration (none/Platt/isotonic) auto-selected on validation only.
- **Explainability:** TreeSHAP; associations, not causation.

## Limitations
Rule-based labels make accuracy largely a persistence measure; single instrument; test window excludes 2020/2022; bundled S&P file ends 2018-12-21 (use yfinance/CSV for newer data); S&P redistribution licence unstated (see data card). The bundled model pickle is version-sensitive (pins in `requirements-api.txt`).
