"""Fase 3: cuantización post-entrenamiento FP32 -> INT8 (estática, con calibración).

Se usan 300 imágenes de entrenamiento para calibrar los rangos de las activaciones.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)

from data_utils import load_fashion

MODELS = Path("models")
FP32 = MODELS / "model_fp32.onnx"
PREP = MODELS / "model_fp32_prep.onnx"
INT8 = MODELS / "model_int8.onnx"


class FashionCalibrationReader(CalibrationDataReader):
    def __init__(self, n: int = 300):
        x, _ = load_fashion(train=True, n=n)
        self.data = iter([{"input": x[i:i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self.data, None)


def accuracy(path, x, y) -> float:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    preds = np.concatenate([sess.run(None, {"input": x[i:i + 500]})[0].argmax(1)
                            for i in range(0, len(x), 500)])
    return float((preds == y).mean())


def main():
    # Pre-procesado recomendado por ONNX Runtime (inferencia de formas + fusión de nodos)
    source = FP32
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(FP32), str(PREP))
        source = PREP
    except Exception as e:  # si falla, se cuantiza directamente el modelo original
        print(f"Aviso: pre-procesado omitido ({e})")

    quantize_static(
        model_input=str(source),
        model_output=str(INT8),
        calibration_data_reader=FashionCalibrationReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    x, y = load_fashion(train=False)
    acc_fp32, acc_int8 = accuracy(FP32, x, y), accuracy(INT8, x, y)
    report = {
        "accuracy_fp32": acc_fp32,
        "accuracy_int8": acc_int8,
        "accuracy_drop": round(acc_fp32 - acc_int8, 4),
        "size_fp32_kb": round(FP32.stat().st_size / 1024, 1),
        "size_int8_kb": round(INT8.stat().st_size / 1024, 1),
    }
    report["size_reduction_pct"] = round(100 * (1 - report["size_int8_kb"] / report["size_fp32_kb"]), 1)
    Path("results").mkdir(exist_ok=True)
    with open("results/quantization.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    PREP.unlink(missing_ok=True)
    print(f"Modelo INT8 guardado en {INT8}  ->  {report}")


if __name__ == "__main__":
    main()
