"""Inference module + FastAPI service tests (uses a bundle trained in a fixture, never a stored one)."""
import numpy as np, pandas as pd, pytest
pytest.importorskip("fastapi"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from mp.inference import MIN_ROWS, Predictor


@pytest.fixture(scope="module")
def client(bundle_path, monkeypatch_module):
    import api.main as m
    monkeypatch_module.setattr(m, "BUNDLE", bundle_path)
    with TestClient(m.app) as c: yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch(); yield mp; mp.undo()


def rows(bundled, n=400, end="2017-06-30"):
    d = bundled.loc[:end].tail(n).reset_index(); d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return d[["date", "open", "high", "low", "close", "volume", "vix", "brent"]].to_dict("records")


def test_predictor_matches_training_pipeline(R, bundle_path, bundled):
    out = Predictor.load(bundle_path).predict(bundled.tail(1500))
    last = len(R["data"]) - 1
    np.testing.assert_allclose(sorted(out["next_day_regime_probabilities"].values()), sorted(np.round(R["proba"]["Calibrated hybrid"][last], 4)), atol=2e-4)
    for h, e in R["transition"].items(): assert abs(out["regime_change_probability"][str(h)] - e["p_cal"][last]) < 2e-4


def test_health_and_model(client):
    assert client.get("/health").json()["status"] == "ok"
    j = client.get("/model").json(); assert j["horizons"] == [5, 10, 20] and "log_loss" in j["test_metrics"] and j["min_history_rows"] == MIN_ROWS


def test_demo_and_flags(client):
    j = client.get("/demo").json(); assert j["as_of"] == "2018-12-21" and j["data_split"] == "test" and j["in_sample"] is False
    assert abs(sum(j["next_day_regime_probabilities"].values()) - 1) < 1e-2 and all(0 <= v <= 1 for v in j["regime_change_probability"].values())
    old = client.get("/demo", params={"as_of": "2001-03-01"}).json(); assert old["in_sample"] is True and old["as_of"] <= "2001-03-01"


def test_predict_valid(client, bundled):
    r = client.post("/predict", json={"rows": rows(bundled)}); assert r.status_code == 200, r.text
    j = r.json(); assert j["predicted_next_regime"] in j["next_day_regime_probabilities"] and 0 < j["confidence"] <= 1 and "disclaimer" in j


def test_predict_rejects_bad_input(client, bundled):
    assert client.post("/predict", json={"rows": rows(bundled)[:50]}).status_code == 422                   # too short
    bad = rows(bundled); bad[10]["high"], bad[10]["low"] = 1.0, 2.0
    assert client.post("/predict", json={"rows": bad}).status_code == 422                                 # high < low
    nov = [{k: v for k, v in r.items() if k != "vix"} for r in rows(bundled)]
    r = client.post("/predict", json={"rows": nov}); assert r.status_code == 422 and "vix" in r.text.lower()   # model needs VIX
    uns = rows(bundled)[::-1]; assert client.post("/predict", json={"rows": uns}).status_code == 422       # not ascending
    assert client.post("/predict", json={"nope": 1}).status_code == 422


def test_index_page(client):
    r = client.get("/"); assert r.status_code == 200 and "MarketPulse" in r.text
