"""API de inferencia (FastAPI + ONNX Runtime con modelo cuantizado INT8).

Endpoints:
  GET  /health          estado del servicio y del modelo
  POST /predict         clasifica 1 imagen
  POST /predict/batch   clasifica varias imágenes en UNA sola llamada al modelo (batching)
  GET  /monitor/drift   PSI de las últimas imágenes recibidas vs. datos de entrenamiento
Documentación interactiva: http://localhost:8000/docs
"""
import io
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.constants import CLASSES, IMG_SIZE, MEAN, STD
from app.drift import histogram, load_reference, psi, status

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_int8.onnx")
REF_STATS_PATH = os.getenv("REF_STATS_PATH", "models/reference_stats.json")
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB por imagen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")
state: dict = {}


# ---------- Esquemas de entrada/salida ----------
class ClassScore(BaseModel):
    label: str
    probability: float


class Prediction(BaseModel):
    filename: str
    class_id: int
    label: str
    confidence: float
    top3: List[ClassScore]


class PredictResponse(BaseModel):
    prediction: Prediction
    latency_ms: float
    model: str


class BatchResponse(BaseModel):
    predictions: List[Prediction]
    batch_size: int
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    error: str | None = None


class DriftResponse(BaseModel):
    samples: int
    psi: float | None
    status: str


# ---------- Ciclo de vida: el modelo se carga UNA vez al iniciar ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = ORT_THREADS
        session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
        state["session"] = session
        state["input_name"] = session.get_inputs()[0].name
        session.run(None, {state["input_name"]: np.zeros((1, 1, IMG_SIZE, IMG_SIZE), np.float32)})  # warm-up
        log.info("Modelo cargado: %s", MODEL_PATH)
    except Exception as e:
        state["error"] = f"No se pudo cargar el modelo: {e}"
        log.error(state["error"])
    state["reference"] = load_reference(REF_STATS_PATH) if os.path.exists(REF_STATS_PATH) else None
    state["recent"] = deque(maxlen=500)  # histogramas de las últimas 500 imágenes
    yield
    state.clear()


app = FastAPI(title="Clasificador Fashion-MNIST (ONNX INT8)", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    log.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor"})


# ---------- Utilidades ----------
def get_session() -> ort.InferenceSession:
    if "session" not in state:
        raise HTTPException(status_code=503, detail=state.get("error", "Modelo no disponible"))
    return state["session"]


async def read_image(file: UploadFile, invert: bool) -> np.ndarray:
    """Lee la imagen, la pasa a escala de grises 28x28 y devuelve píxeles en [0, 1]."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"'{file.filename}' está vacío")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"'{file.filename}' supera 5 MB")
    try:
        img = Image.open(io.BytesIO(content)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail=f"'{file.filename}' no es una imagen válida")
    pixels = np.asarray(img, dtype=np.float32) / 255.0
    if invert:  # Fashion-MNIST usa prenda clara sobre fondo negro
        pixels = 1.0 - pixels
    return pixels


def run_model(pixels: np.ndarray) -> np.ndarray:
    """pixels: [N, 28, 28] en [0, 1] -> probabilidades [N, 10]."""
    x = ((pixels - MEAN) / STD)[:, None, :, :].astype(np.float32)
    logits = get_session().run(None, {state["input_name"]: x})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def to_prediction(filename: str, probs: np.ndarray) -> Prediction:
    top = np.argsort(probs)[::-1][:3]
    return Prediction(
        filename=filename or "sin_nombre",
        class_id=int(top[0]),
        label=CLASSES[top[0]],
        confidence=round(float(probs[top[0]]), 4),
        top3=[ClassScore(label=CLASSES[i], probability=round(float(probs[i]), 4)) for i in top],
    )


def track_drift(pixels: np.ndarray):
    for p in pixels:
        state["recent"].append(histogram(p))


# ---------- Endpoints ----------
@app.get("/health", response_model=HealthResponse)
def health():
    loaded = "session" in state
    body = HealthResponse(status="ok" if loaded else "error", model_loaded=loaded,
                          model=os.path.basename(MODEL_PATH), error=state.get("error"))
    if not loaded:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...),
                  invert: bool = Query(False, description="Invertir colores (para fotos con fondo blanco)")):
    get_session()
    pixels = await read_image(file, invert)
    t0 = time.perf_counter()
    probs = run_model(pixels[None])
    latency = (time.perf_counter() - t0) * 1000
    track_drift(pixels[None])
    return PredictResponse(prediction=to_prediction(file.filename, probs[0]),
                           latency_ms=round(latency, 3), model=os.path.basename(MODEL_PATH))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...),
                        invert: bool = Query(False, description="Invertir colores")):
    get_session()
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Máximo {MAX_BATCH} imágenes por petición")
    pixels = np.stack([await read_image(f, invert) for f in files])
    t0 = time.perf_counter()
    probs = run_model(pixels)  # una sola inferencia para todo el lote
    latency = (time.perf_counter() - t0) * 1000
    track_drift(pixels)
    return BatchResponse(predictions=[to_prediction(f.filename, p) for f, p in zip(files, probs)],
                         batch_size=len(files), latency_ms=round(latency, 3),
                         model=os.path.basename(MODEL_PATH))


@app.get("/monitor/drift", response_model=DriftResponse)
def monitor_drift():
    recent, ref = state.get("recent"), state.get("reference")
    if ref is None:
        return DriftResponse(samples=0, psi=None, status="sin_referencia")
    if not recent:
        return DriftResponse(samples=0, psi=None, status="sin_datos")
    value = psi(ref["hist"], np.sum(recent, axis=0))
    return DriftResponse(samples=len(recent), psi=round(value, 4), status=status(value))
