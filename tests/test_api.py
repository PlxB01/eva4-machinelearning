"""Pruebas de la API: endpoints, formato de respuesta y manejo de errores."""
import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.constants import CLASSES
from app.main import MODEL_PATH, app

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_PATH),
                                reason="Primero ejecuta train.py, export_onnx.py y quantize.py")


def fake_png(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (28, 28), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" ejecuta la carga del modelo (lifespan)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_ok(client):
    r = client.post("/predict", files={"file": ("prenda.png", fake_png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["label"] in CLASSES
    assert 0.0 <= body["prediction"]["confidence"] <= 1.0
    assert len(body["prediction"]["top3"]) == 3
    assert body["latency_ms"] >= 0


def test_predict_batch(client):
    files = [("files", (f"img{i}.png", fake_png(i), "image/png")) for i in range(4)]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200
    assert r.json()["batch_size"] == 4
    assert len(r.json()["predictions"]) == 4


def test_invalid_image_returns_400(client):
    r = client.post("/predict", files={"file": ("malo.png", b"esto no es una imagen", "image/png")})
    assert r.status_code == 400


def test_missing_file_returns_422(client):
    assert client.post("/predict").status_code == 422


def test_drift_endpoint(client):
    r = client.get("/monitor/drift")
    assert r.status_code == 200
    assert r.json()["samples"] > 0
