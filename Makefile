.PHONY: install data train tune test api app docker report
install:  ; pip install -r requirements.txt
data:     ; python scripts/build_bundled_data.py
train:    ; python train.py                       # default params -> artifacts/ + docs/figures/
tune:     ; python train.py --tune --trials 25    # Optuna walk-forward tuning
test:     ; pytest -q
api:      ; uvicorn api.main:app --reload --port 8000
app:      ; streamlit run app.py
docker:   ; docker build -t marketpulse-api . && docker run --rm -p 7860:7860 marketpulse-api
report:   ; cd report && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
