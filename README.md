# Eva 4 Machine Learning — Despliegue de un modelo Deep Learning optimizado

Clasificador de prendas de ropa (Fashion-MNIST, 10 clases) con una CNN en PyTorch,
**exportada a ONNX**, **cuantizada a INT8**, servida con **FastAPI** dentro de **Docker**
y con un pipeline **CI/CD en GitHub Actions** (MLflow + DVC para versionado).

```
Entrenar (PyTorch) → Exportar ONNX → Validar → Cuantizar INT8 → Benchmark → API FastAPI → Docker → CI/CD
```

## Qué cubre cada archivo (vs. la pauta)

| Archivo | Qué hace | Criterio |
|---|---|---|
| `train.py` | Entrena la CNN y registra parámetros/métricas en MLflow | 4.1.3 |
| `export_onnx.py` | Exporta a ONNX (batch dinámico) y valida que dé lo mismo que PyTorch | 4.1.2 |
| `quantize.py` | Cuantización estática FP32 → INT8 con calibración | 4.1.1 |
| `benchmark.py` | Compara latencia, throughput (batching), tamaño y RAM | 4.1.1 |
| `app/main.py` | API FastAPI: `/health`, `/predict`, `/predict/batch`, `/monitor/drift` | 4.1.4 |
| `Dockerfile` | Imagen liviana solo con ONNX Runtime (sin PyTorch) | 4.1.4 |
| `tests/` | Tests de accuracy mínima, pérdida por cuantización, latencia p95, drift y API | 4.1.3 |
| `.github/workflows/ci.yml` | Pipeline: entrena → valida → cuantiza → testea → build Docker → publica | 4.1.3 |
| `dvc.yaml` | Pipeline reproducible y versionado de modelos con DVC | 4.1.3 |

---

## Paso a paso

### 0. Requisitos
- Python **3.11** (o 3.10/3.12)
- **Docker Desktop** abierto
- Cuenta de **GitHub** (para el CI/CD)

### 1. Crear entorno e instalar dependencias
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Mac/Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline del modelo (en orden)
```bash
python train.py        # ~2-5 min en CPU. Descarga Fashion-MNIST sola. Accuracy esperada ~90%
python export_onnx.py  # crea models/model_fp32.onnx y lo valida
python quantize.py     # crea models/model_int8.onnx
python benchmark.py    # crea results/benchmark.md  ← TABLA PARA EL INFORME
```

### 3. Correr los tests
```bash
pytest -v
```
Si el test de latencia falla en tu PC, sube el umbral: `MAX_LATENCY_MS=50 pytest -v`
(en PowerShell: `$env:MAX_LATENCY_MS="50"; pytest -v`).

### 4. Probar la API localmente (sin Docker)
```bash
uvicorn app.main:app --reload
```
Abre **http://localhost:8000/docs** → `POST /predict` → "Try it out" → sube una imagen.
> Las imágenes de Fashion-MNIST son prenda **clara sobre fondo negro**. Si subes una foto
> normal (fondo blanco), activa `invert=true`.

### 5. Docker
```bash
docker build -t fashion-api .
docker run -d -p 8000:8000 --name fashion-api fashion-api
```
Abre **http://localhost:8000/docs** y prueba de nuevo. Para ver el tamaño de la imagen: `docker images fashion-api`.
Para detenerlo: `docker stop fashion-api && docker rm fashion-api`.

Prueba con curl:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@mi_imagen.png"
curl http://localhost:8000/monitor/drift
```

### 6. MLflow (versionado de experimentos)
```bash
mlflow ui
```
Abre **http://localhost:5000** y verás el run con sus parámetros, métricas por época y el modelo guardado.

### 7. CI/CD en GitHub
```bash
git init
git add .
git commit -m "Eva 4: modelo DL optimizado + API + Docker + CI/CD"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/eva4-ml.git
git push -u origin main
```
En GitHub → pestaña **Actions** verás el pipeline corriendo. Al terminar en verde, la imagen
queda publicada en **Packages** (GitHub Container Registry).

### 8. (Opcional) DVC
```bash
pip install dvc
dvc init
dvc repro      # ejecuta el pipeline de dvc.yaml
dvc metrics show
```

---

## Capturas que debes sacar para el informe
1. Salida de `train.py` (accuracy por época) y la UI de MLflow.
2. Salida de `export_onnx.py` (validación: diferencia máxima y 100% de coincidencia).
3. Salida de `quantize.py` (accuracy FP32 vs INT8 y reducción de tamaño).
4. Tabla de `results/benchmark.md`.
5. `pytest -v` con todos los tests en verde.
6. `docker build`, `docker images` (tamaño) y `docker ps`.
7. Swagger (`/docs`) con una predicción exitosa, y una petición con error (archivo inválido → 400).
8. GitHub Actions con el pipeline en verde.

## Nota sobre GPU / VRAM
El proyecto corre 100% en CPU. Si no tienes GPU, en el informe reporta RAM y explica
de forma teórica que con GPU se usaría `onnxruntime-gpu` (CUDA/TensorRT Execution Provider),
donde INT8/FP16 reduce el uso de VRAM en proporción similar a la reducción de tamaño.
