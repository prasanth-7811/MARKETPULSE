import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mp import config as C
from mp.data import clean, load_bundled

FAST = {"models": {"selected": ["Logistic Regression", "LightGBM"]}, "hmm": {"n_iter": 30}}


@pytest.fixture(scope="session")
def cfg():
    c = C.deep_update(C.load_defaults(), FAST); c["data"]["end"] = "2100-01-01"; return c


@pytest.fixture(scope="session")
def bundled():
    return load_bundled()


@pytest.fixture(scope="session")
def df(bundled):
    return clean(bundled)[0]


@pytest.fixture(scope="session")
def R(df, cfg):
    pytest.importorskip("hmmlearn"); pytest.importorskip("lightgbm")
    from mp.pipeline import run_pipeline
    return run_pipeline(df, cfg)


@pytest.fixture(scope="session")
def bundle_path(R, cfg, df, tmp_path_factory):
    from mp.inference import make_bundle, save_bundle
    return save_bundle(make_bundle(R, cfg, df), tmp_path_factory.mktemp("art") / "model_bundle.joblib")
