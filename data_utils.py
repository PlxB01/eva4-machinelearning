"""Carga de Fashion-MNIST como arreglos numpy (se descarga sola la primera vez en ./data)."""
import numpy as np

from app.constants import MEAN, STD


def load_fashion(train: bool, n: int | None = None, normalized: bool = True):
    """Devuelve (x, y) con x de forma [N, 1, 28, 28] float32 e y de forma [N] int64."""
    from torchvision import datasets

    ds = datasets.FashionMNIST(root="data", train=train, download=True)
    x = ds.data.numpy().astype(np.float32) / 255.0
    y = ds.targets.numpy().astype(np.int64)
    if n is not None:
        x, y = x[:n], y[:n]
    if normalized:
        x = (x - MEAN) / STD
    return x[:, None, :, :].astype(np.float32), y
