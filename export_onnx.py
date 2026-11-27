"""Fase 2: exportación a ONNX + validación del artefacto.

Valida que el modelo ONNX sea estructuralmente correcto y que entregue las mismas
predicciones que el modelo PyTorch original.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from data_utils import load_fashion
from model import FashionCNN

MODELS = Path("models")
ONNX_PATH = MODELS / "model_fp32.onnx"
TOLERANCE = 1e-3


def export(model):
    dummy = torch.randn(1, 1, 28, 28)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # batch variable
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(ONNX_PATH), dynamo=False, **kwargs)
    except TypeError:  # versiones antiguas de torch no tienen el parámetro "dynamo"
        torch.onnx.export(model, dummy, str(ONNX_PATH), **kwargs)


def main():
    model = FashionCNN()
    model.load_state_dict(torch.load(MODELS / "model.pt", map_location="cpu"))
    model.eval()
    export(model)

    # Validación 1: estructura del grafo
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))

    # Validación 2: equivalencia numérica PyTorch vs ONNX Runtime
    x, y = load_fashion(train=False, n=1000)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    agreement = float((torch_out.argmax(1) == onnx_out.argmax(1)).mean())
    report = {"max_abs_diff": max_diff, "prediction_agreement": agreement,
              "onnx_accuracy_1000": float((onnx_out.argmax(1) == y).mean()),
              "valid": max_diff < TOLERANCE and agreement == 1.0}
    Path("results").mkdir(exist_ok=True)
    with open("results/export_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"ONNX exportado en {ONNX_PATH}  ->  {report}")
    if not report["valid"]:
        raise SystemExit("ERROR: el modelo ONNX no coincide con PyTorch")


if __name__ == "__main__":
    main()
