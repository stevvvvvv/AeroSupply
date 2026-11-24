"""Tests run on a fresh copy of the (seeded, deterministic) synthetic data in a temp folder,
with SQLite + embedded MongoDB, and reuse the trained artifacts shipped in backend/artifacts."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="aerosupply-test-"))
os.environ["AEROSUPPLY_DATA"] = str(TMP / "data")
os.environ["AEROSUPPLY_ARTIFACTS"] = str(TMP / "artifacts")
os.environ["DATABASE_URL"] = f"sqlite:///{TMP / 'test.db'}"
os.environ.pop("MONGO_URL", None)
sys.path.insert(0, str(BACKEND))

shutil.copytree(BACKEND / "artifacts", TMP / "artifacts")


@pytest.fixture(scope="session")
def pipeline():
    from aerosupply.pipeline import steps
    run_id = "test"
    steps.extract(run_id, force=True)
    for name, fn in steps.STEPS[1:]:
        if name == "train_models":
            continue          # shipped artifacts (training is exercised by `make train`)
        fn(run_id)
    return steps


@pytest.fixture(scope="session")
def D(pipeline):
    return pipeline.ctx().D


@pytest.fixture(scope="session")
def client(pipeline):
    from fastapi.testclient import TestClient
    from aerosupply.api.main import app
    return TestClient(app)
