"""Два незалежні еталони: autograd PyTorch і центральні різниці."""

import numpy as np
import torch
from torch import nn

from model_numpy import forward

AUTOGRAD_TOL = 1e-12
NUMERICAL_TOL = 1e-7
EPSILON = 1e-6
SELECTED = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))


def torch_reference(x: np.ndarray, y: np.ndarray, params: dict) -> tuple[float, dict]:
    """Скопіювати саме NumPy-ваги й отримати loss/градієнти."""

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()
    with torch.no_grad():
        for layer, suffix in ((model[0], "1"), (model[2], "2")):

            layer.weight.copy_(torch.from_numpy(params["W" + suffix].T))
            layer.bias.copy_(torch.from_numpy(params["b" + suffix]))
    logits = model(torch.from_numpy(x))
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(y), reduction="mean")
    loss.backward()
    grads = {}
    for layer, suffix in ((model[0], "1"), (model[2], "2")):
        grads["W" + suffix] = layer.weight.grad.detach().numpy().T.copy()
        grads["b" + suffix] = layer.bias.grad.detach().numpy().copy()
    return loss.item(), grads


def compare_autograd(loss: float, grads: dict, reference_loss: float, reference: dict) -> list:
    """Перевірити КОЖНИЙ елемент; у таблицю винести найбільшу різницю."""
    rows = []
    for name, a, b in [("Loss", loss, reference_loss)] + [
        (name, grads[name], reference[name]) for name in grads
    ]:
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise ValueError(f"Несумісні форми {name}: {a.shape}, {b.shape}")
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        difference = np.abs(a - b)
        rows.append({"name": name, "max_abs_diff": float(difference.max()),
                     "finite": finite, "passed": finite and bool((difference <= AUTOGRAD_TOL).all())})
    return rows


def numerical_checks(x: np.ndarray, y: np.ndarray, params: dict, grads: dict) -> list:
    """Збурювати по одному елементу та гарантовано відновлювати його."""
    _, base_cache = forward(x, y, params)
    base_mask = base_cache.z1 > 0
    rows = []
    for name, index in SELECTED:
        original = params[name][index].copy()
        try:
            params[name][index] = original + EPSILON
            loss_plus, plus_cache = forward(x, y, params)

            params[name][index] = original - EPSILON
            loss_minus, minus_cache = forward(x, y, params)
        finally:

            params[name][index] = original
        numerical = (loss_plus - loss_minus) / (2 * EPSILON)
        manual = float(grads[name][index])
        difference = abs(numerical - manual)


        mask_stable = bool(np.array_equal(base_mask, plus_cache.z1 > 0)
                           and np.array_equal(base_mask, minus_cache.z1 > 0))
        finite = bool(np.isfinite([loss_plus, loss_minus, numerical, manual]).all())
        rows.append({"name": name + "[" + ",".join(map(str, index)) + "]",
                     "manual": manual, "numerical": numerical, "abs_diff": difference,
                     "finite": finite, "relu_mask_stable": mask_stable,
                     "passed": finite and difference <= NUMERICAL_TOL})
    return rows
