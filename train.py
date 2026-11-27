"""Fase 1: entrenamiento de la CNN + registro de experimento en MLflow.

Uso:  python train.py            (3 épocas por defecto)
      EPOCHS=1 python train.py   (rápido, lo usa el pipeline de CI)
"""
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.drift import BINS, histogram
from data_utils import load_fashion
from model import FashionCNN

EPOCHS = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = 128
LR = 1e-3
SEED = 42
MODELS = Path("models")


@torch.no_grad()
def evaluate(model, x, y) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), 1000):
        logits = model(torch.from_numpy(x[i:i + 1000]))
        correct += (logits.argmax(1).numpy() == y[i:i + 1000]).sum()
    return float(correct / len(x))


def main():
    MODELS.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train, y_train = load_fashion(train=True)
    x_test, y_test = load_fashion(train=False)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = FashionCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {n_params:,}")

    # MLflow es opcional: si no está instalado, el entrenamiento igual funciona
    try:
        import mlflow
        mlflow.set_experiment("eva4-fashion-cnn")
        run_ctx = mlflow.start_run()
    except ImportError:
        mlflow, run_ctx = None, contextlib.nullcontext()

    with run_ctx:
        if mlflow:
            mlflow.log_params({"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                               "seed": SEED, "params": n_params, "arquitectura": "CNN 2conv+2fc"})
        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(xb)
            train_loss = running / len(x_train)
            test_acc = evaluate(model, x_test, y_test)
            print(f"Época {epoch}/{EPOCHS} | loss={train_loss:.4f} | accuracy test={test_acc:.4f}")
            if mlflow:
                mlflow.log_metrics({"train_loss": train_loss, "test_accuracy": test_acc}, step=epoch)

        torch.save(model.state_dict(), MODELS / "model.pt")

        # Estadísticas de referencia para detectar drift en producción
        raw, _ = load_fashion(train=True, n=5000, normalized=False)
        reference = {"bins": BINS.tolist(), "hist": histogram(raw).tolist(),
                     "mean": float(raw.mean()), "std": float(raw.std())}
        with open(MODELS / "reference_stats.json", "w", encoding="utf-8") as f:
            json.dump(reference, f, indent=2)

        metrics = {"test_accuracy": test_acc, "train_loss": train_loss, "epochs": EPOCHS,
                   "params": n_params, "train_seconds": round(time.time() - t0, 1)}
        with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if mlflow:
            mlflow.log_artifact(str(MODELS / "model.pt"))
            mlflow.log_artifact(str(MODELS / "reference_stats.json"))

    print(f"Modelo guardado en {MODELS / 'model.pt'}  ->  {metrics}")


if __name__ == "__main__":
    main()
