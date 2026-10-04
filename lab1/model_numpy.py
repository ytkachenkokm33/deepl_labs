"""Мережа 4 → 8 → 3: тільки NumPy, без автоматичного диференціювання."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Cache:
    """Проміжні значення forward, які повторно використовує backward."""

    x: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    logits: np.ndarray
    p: np.ndarray


def initialize_parameters() -> dict[str, np.ndarray]:
    """Створити початкові параметри у float64 за фіксованим рецептом."""

    rng = np.random.default_rng(0)
    w1 = rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8))
    w2 = rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3))
    return {"W1": w1, "b1": np.zeros(8), "W2": w2, "b2": np.zeros(3)}


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Повернути середню крос-ентропію та ймовірності класів."""


    shifted = logits - logits.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


    loss = -log_p[np.arange(len(y)), y].mean()
    return float(loss), np.exp(log_p)


def forward(x: np.ndarray, y: np.ndarray, params: dict) -> tuple[float, Cache]:
    """Перенести вхід через два афінні шари та обчислити втрату."""

    z1 = x @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)

    logits = a1 @ params["W2"] + params["b2"]
    loss, p = cross_entropy(logits, y)
    return loss, Cache(x=x, z1=z1, a1=a1, logits=logits, p=p)


def backward(cache: Cache, y: np.ndarray, params: dict, *, bug: bool = False) -> dict:
    """Обчислити всі 67 похідних вручну за ланцюговим правилом."""
    n = len(y)


    dz2 = cache.p.copy()
    dz2[np.arange(n), y] -= 1.0
    if not bug:
        dz2 /= n


    dw2 = cache.a1.T @ dz2


    db2 = dz2.sum(axis=0)


    da1 = dz2 @ params["W2"].T


    dz1 = da1 * (cache.z1 > 0)
    dw1 = cache.x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}
