## Benchmark de inferencia (CPU, 1 hilo/s)

Entorno: {'cpu': 'x86_64', 'nucleos': 2, 'hilos_usados': 1, 'python': '3.14.2', 'torch': '2.15.0+cpu', 'onnxruntime': '1.32.0'}

| modelo | tamano_kb | ram_carga_mb | latencia_p50_ms | latencia_p95_ms | img_por_s_batch1 | img_por_s_batch8 | img_por_s_batch32 | img_por_s_batch64 | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 811.5 | 5.8 | 0.25 | 0.316 | 4137.0 | 6339.0 | 7886.0 | 8188.0 | 0.8922 |
| ONNX Runtime FP32 | 810.3 | 8.5 | 0.055 | 0.071 | 18428.0 | 31662.0 | 34160.0 | 34289.0 | 0.8922 |
| ONNX Runtime INT8 | 214.3 | 0.6 | 0.062 | 0.079 | 15874.0 | 22765.0 | 24014.0 | 23985.0 | 0.8909 |
