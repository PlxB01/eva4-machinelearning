"""Detección simple de data drift con PSI (Population Stability Index).

Compara la distribución de intensidad de píxeles de los datos nuevos contra la
distribución de referencia guardada durante el entrenamiento.
Regla común: PSI < 0.1 estable | 0.1 - 0.25 cambio moderado | > 0.25 drift.
"""
import json

import numpy as np

BINS = np.linspace(0.0, 1.0, 11)  # 10 tramos de intensidad (píxeles en [0, 1])
PSI_WARNING = 0.1
PSI_ALERT = 0.25


def histogram(pixels: np.ndarray) -> np.ndarray:
    """Cuenta cuántos píxeles caen en cada tramo de intensidad."""
    counts, _ = np.histogram(np.clip(pixels, 0.0, 1.0), bins=BINS)
    return counts.astype(np.float64)


def psi(reference_counts, current_counts, eps: float = 1e-6) -> float:
    ref = np.asarray(reference_counts, dtype=np.float64)
    cur = np.asarray(current_counts, dtype=np.float64)
    p = np.clip(ref / ref.sum(), eps, None)
    q = np.clip(cur / cur.sum(), eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def status(value: float) -> str:
    if value < PSI_WARNING:
        return "estable"
    if value < PSI_ALERT:
        return "cambio_moderado"
    return "drift"


def load_reference(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
