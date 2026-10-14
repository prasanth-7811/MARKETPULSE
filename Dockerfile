# MarketPulse inference API - works locally, on Hugging Face Spaces (sdk: docker, port 7860) and on AWS (ECR/App Runner/ECS).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
# libgomp1: OpenMP runtime needed by LightGBM / XGBoost wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements-api.txt .
RUN pip install --user -r requirements-api.txt

COPY --chown=user mp/ mp/
COPY --chown=user api/ api/
COPY --chown=user config.yaml .
COPY --chown=user data/bundled/ data/bundled/
COPY --chown=user artifacts/model_bundle.joblib artifacts/model_bundle.joblib

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD curl -fs http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
