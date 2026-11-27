# Imagen liviana: solo ONNX Runtime + FastAPI (sin PyTorch, ~10 veces más chica)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model_int8.onnx \
    REF_STATS_PATH=/app/models/reference_stats.json \
    ORT_THREADS=2

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY models/model_int8.onnx models/reference_stats.json ./models/

# Usuario sin privilegios (buena práctica de seguridad)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
