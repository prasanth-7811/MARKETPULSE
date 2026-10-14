# Deployment guide

The inference service is a FastAPI app (`api/main.py`) that loads `artifacts/model_bundle.joblib` (4 MB) at start-up.

## 1. Local (no Docker)
```bash
pip install -r requirements.txt
python train.py                      # creates artifacts/model_bundle.joblib (skip if the repo already ships one)
uvicorn api.main:app --port 8000     # http://localhost:8000   (demo page)   /docs (Swagger)   /model   /health
curl localhost:8000/demo?as_of=2018-12-21
```
`POST /predict` takes `{"rows": [{"date":"2018-12-21","open":..,"high":..,"low":..,"close":..,"volume":..,"vix":..}, ...]}` with >= 320 ascending daily rows (the model uses VIX, so `vix` is required; see `GET /model`).

## 2. Docker (local)
```bash
docker build -t marketpulse-api .
docker run --rm -p 7860:7860 marketpulse-api         # http://localhost:7860
docker compose up --build                            # API on :8000 ; add  --profile full  for the Streamlit dashboard on :8501
```

## 3. Hugging Face Spaces (Docker SDK)  - free CPU tier
The root `README.md` already carries the Spaces front-matter (`sdk: docker`, `app_port: 7860`).
```bash
pip install -U huggingface_hub && huggingface-cli login
huggingface-cli repo create marketpulse --type space --space_sdk docker
git remote add space https://huggingface.co/spaces/<your-username>/marketpulse
git push space main                                   # Space builds the Dockerfile and starts uvicorn on 7860
```
Then record the live URL in the README "Live demo" line and in the report (Section VII). The model bundle (4 MB) is below the 10 MB non-LFS limit; if you retrain with larger models run `git lfs track "*.joblib"`.

## 4. AWS alternative (App Runner)
```bash
aws ecr create-repository --repository-name marketpulse-api
docker tag marketpulse-api <acct>.dkr.ecr.<region>.amazonaws.com/marketpulse-api:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker push <acct>.dkr.ecr.<region>.amazonaws.com/marketpulse-api:latest
# App Runner console: create service from the ECR image, port 7860, health check path /health
```

## Operational notes
* Pickles are version-sensitive; `requirements-api.txt` pins the training versions. Retrain (`python train.py`) after changing them.
* The bundled demo data ends 2018-12-21; `/demo?as_of=` dates inside the train/validation windows are flagged `in_sample: true`.
* No state is stored; each request is independent.
