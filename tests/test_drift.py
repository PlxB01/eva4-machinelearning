"""Pruebas de detección de drift: datos normales no alertan, datos alterados sí."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.drift import PSI_ALERT, PSI_WARNING, histogram, psi
from data_utils import load_fashion

REF = Path("models/reference_stats.json")
pytestmark = pytest.mark.skipif(not REF.exists(), reason="Falta models/reference_stats.json")


@pytest.fixture(scope="module")
def reference():
    return json.loads(REF.read_text(encoding="utf-8"))


def test_datos_de_test_sin_drift(reference):
    raw, _ = load_fashion(train=False, n=2000, normalized=False)
    assert psi(reference["hist"], histogram(raw)) < PSI_WARNING


def test_datos_alterados_detectan_drift(reference):
    raw, _ = load_fashion(train=False, n=2000, normalized=False)
    shifted = np.clip(raw * 0.5 + 0.4, 0, 1)  # imágenes más claras y con menos contraste
    assert psi(reference["hist"], histogram(shifted)) > PSI_ALERT
