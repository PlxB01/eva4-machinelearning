"""Pruebas automáticas del modelo: métricas mínimas, pérdida por cuantización y latencia.

Los umbrales se pueden ajustar con variables de entorno (el CI usa valores más holgados).
"""
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from data_utils import load_fashion

INT8 = Path("models/model_int8.onnx")
FP32 = Path("models/model_fp32.onnx")
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.85"))
MAX_ACC_DROP = float(os.getenv("MAX_ACC_DROP", "0.02"))
MAX_LATENCY_MS = float(os.getenv("MAX_LATENCY_MS", "20"))

pytestmark = pytest.mark.skipif(not INT8.exists(), reason="Falta models/model_int8.onnx")


@pytest.fixture(scope="module")
def test_data():
    return load_fashion(train=False, n=2000)


def predict(path, x):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x})[0].argmax(1)


def test_accuracy_minima(test_data):
    x, y = test_data
    acc = float((predict(INT8, x) == y).mean())
    assert acc >= MIN_ACCURACY, f"Accuracy INT8 {acc:.3f} < {MIN_ACCURACY}"


def test_perdida_por_cuantizacion(test_data):
    x, y = test_data
    drop = float((predict(FP32, x) == y).mean() - (predict(INT8, x) == y).mean())
    assert drop <= MAX_ACC_DROP, f"La cuantización bajó la accuracy en {drop:.3f}"


def test_latencia_p95(test_data):
    x, _ = test_data
    sess = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
    one = x[:1]
    for _ in range(10):
        sess.run(None, {"input": one})
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {"input": one})
        times.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(times, 95))
    assert p95 <= MAX_LATENCY_MS, f"Latencia p95 {p95:.2f} ms > {MAX_LATENCY_MS} ms"
