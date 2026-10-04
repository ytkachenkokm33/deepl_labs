"""Підготовка Iris: відтворюваний поділ і стандартизація без витоку даних."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris


@dataclass
class Dataset:
    """Дані та статистики, потрібні для пояснення і перевірки підготовки."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def prepare_data() -> Dataset:
    """Виконати саме алгоритм поділу з умови, а не train_test_split."""
    iris = load_iris()
    x = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)


    rng = np.random.default_rng(0)
    train_parts, test_parts = [], []
    for label in (0, 1, 2):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        train_parts.append(indices[:35])
        test_parts.append(indices[35:])
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)


    mean = x[train_indices].mean(axis=0)
    std = x[train_indices].std(axis=0, ddof=0)
    if np.any(std == 0):
        raise ValueError("Неможливо стандартизувати ознаку з нульовим std")

    return Dataset(
        x_train=(x[train_indices] - mean) / std,
        y_train=y[train_indices],
        x_test=(x[test_indices] - mean) / std,
        y_test=y[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        mean=mean,
        std=std,
    )
