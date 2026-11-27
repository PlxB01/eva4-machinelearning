"""Fase 4: benchmark de velocidad y recursos.

Compara PyTorch FP32 vs ONNX Runtime FP32 vs ONNX Runtime INT8:
  - latencia por petición (p50 / p95) con batch 1
  - throughput (imágenes/segundo) con batching de 1, 8, 32 y 64
  - tamaño del archivo y RAM usada al cargar el modelo
Genera results/benchmark.json y results/benchmark.md (tabla lista para el informe).
"""
import gc
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import torch

from data_utils import load_fashion
from model import FashionCNN

THREADS = int(os.getenv("BENCH_THREADS", "1"))  # 1 hilo = comparación justa
BATCH_SIZES = [1, 8, 32, 64]
RUNS, WARMUP = 200, 20
MODELS = Path("models")
proc = psutil.Process()


def rss_mb() -> float:
    gc.collect()
    return proc.memory_info().rss / 1024 ** 2


def time_it(fn, x, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def bench(name, load_fn, size_kb, data):
    before = rss_mb()
    predict = load_fn()
    predict(data[:1])
    ram = rss_mb() - before
    row = {"modelo": name, "tamano_kb": round(size_kb, 1), "ram_carga_mb": round(max(ram, 0), 1)}
    t1 = time_it(predict, data[:1])
    row["latencia_p50_ms"] = round(float(np.percentile(t1, 50)), 3)
    row["latencia_p95_ms"] = round(float(np.percentile(t1, 95)), 3)
    for bs in BATCH_SIZES:
        t = time_it(predict, data[:bs], runs=50, warmup=5)
        row[f"img_por_s_batch{bs}"] = round(bs / (np.median(t) / 1000), 0)
    return row


def load_torch():
    torch.set_num_threads(THREADS)
    m = FashionCNN()
    m.load_state_dict(torch.load(MODELS / "model.pt", map_location="cpu"))
    m.eval()

    def predict(x):
        with torch.no_grad():
            return m(torch.from_numpy(x)).numpy()
    return predict


def load_onnx(path):
    def loader():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        return lambda x: sess.run(None, {"input": x})[0]
    return loader


def to_markdown(rows):
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
    return "\n".join(lines)


def main():
    data, _ = load_fashion(train=False, n=64)
    rows = [
        bench("PyTorch FP32", load_torch, (MODELS / "model.pt").stat().st_size / 1024, data),
        bench("ONNX Runtime FP32", load_onnx(MODELS / "model_fp32.onnx"),
              (MODELS / "model_fp32.onnx").stat().st_size / 1024, data),
        bench("ONNX Runtime INT8", load_onnx(MODELS / "model_int8.onnx"),
              (MODELS / "model_int8.onnx").stat().st_size / 1024, data),
    ]
    quant = {}
    if Path("results/quantization.json").exists():
        quant = json.loads(Path("results/quantization.json").read_text(encoding="utf-8"))
        rows[0]["accuracy"] = rows[1]["accuracy"] = quant.get("accuracy_fp32")
        rows[2]["accuracy"] = quant.get("accuracy_int8")

    env = {"cpu": platform.processor() or platform.machine(), "nucleos": os.cpu_count(),
           "hilos_usados": THREADS, "python": platform.python_version(),
           "torch": torch.__version__, "onnxruntime": ort.__version__}
    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"entorno": env, "resultados": rows}, f, indent=2, ensure_ascii=False)
    md = f"## Benchmark de inferencia (CPU, {THREADS} hilo/s)\n\nEntorno: {env}\n\n{to_markdown(rows)}\n"
    Path("results/benchmark.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
