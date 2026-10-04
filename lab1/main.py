"""Запуск лабораторної: дані → forward → backward → незалежні перевірки."""

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
import torch

from checks import compare_autograd, numerical_checks, torch_reference
from data import prepare_data
from model_numpy import backward, forward, initialize_parameters


PREDICTION = (
    "Прогноз ДО обчислення помилкового backward: loss не зміниться; "
    "ненульові градієнти збільшаться у 105 разів. Звірка loss пройде, "
    "а обидва способи перевірки градієнтів виявлять помилку. "
)


def as_markdown(result: dict) -> str:
    """Сформувати читабельні таблиці із фактичних чисел."""
    status = lambda value: "так" if value else "ні"
    lines = [
        "## Результати: " + ("навмисна помилка" if result["bug"] else "правильний backward"),
        "", f"Loss NumPy: `{result['loss_numpy']:.17g}`.",
        f"Loss PyTorch: `{result['loss_torch']:.17g}`.", "",
        "| Величина | Максимальна абсолютна різниця | Скінченні | Пройдено (1e-12) |",
        "|---|---:|---|---|",
    ]
    for row in result["autograd"]:
        lines.append(f"| {row['name']} | {row['max_abs_diff']:.12e} | {status(row['finite'])} | {status(row['passed'])} |")
    lines += ["", "Центральні різниці: epsilon = 1e-6.", "",
              "| Параметр | Ручний градієнт | Чисельна похідна | Абсолютна різниця | Пройдено (1e-7) | Маска ReLU стала |",
              "|---|---:|---:|---:|---|---|"]
    for row in result["numerical"]:
        lines.append(f"| {row['name']} | {row['manual']:.12e} | {row['numerical']:.12e} | {row['abs_diff']:.12e} | {status(row['passed'])} | {status(row['relu_mask_stable'])} |")
    if result["bug"]:
        lines += ["", "Прогноз, зафіксований до запуску:", "", PREDICTION, "",
                  "Перевірка масштабу відносно правильного ручного backward:", "",
                  "| Градієнт | min відношення ненульових | max відношення ненульових | max abs(g_bug - 105*g_correct) |",
                  "|---|---:|---:|---:|"]
        for row in result["scaling"]:
            lines.append(f"| {row['name']} | {row['ratio_min']:.12g} | {row['ratio_max']:.12g} | {row['max_residual']:.12e} |")
    lines += ["", f"Параметри не змінені: {status(result['parameters_unchanged'])}.",
              f"Очікуваний результат експерименту отримано: {status(result['experiment_passed'])}.", ""]
    return "\n".join(lines)


def run(*, bug: bool = False) -> dict:
    """Виконати дослід на початкових параметрах."""
    data = prepare_data()
    params = initialize_parameters()
    original = {name: value.copy() for name, value in params.items()}
    loss, cache = forward(data.x_train, data.y_train, params)
    correct = backward(cache, data.y_train, params)
    grads = backward(cache, data.y_train, params, bug=bug)
    reference_loss, reference = torch_reference(data.x_train, data.y_train, params)
    autograd = compare_autograd(loss, grads, reference_loss, reference)
    numerical = numerical_checks(data.x_train, data.y_train, params, grads)
    unchanged = all(np.array_equal(params[k], original[k]) for k in params)

    scaling = []
    for name in grads:
        nonzero = correct[name] != 0
        ratios = grads[name][nonzero] / correct[name][nonzero]
        scaling.append({"name": name, "ratio_min": float(ratios.min()),
                        "ratio_max": float(ratios.max()),
                        "max_residual": float(np.max(np.abs(grads[name] - len(data.y_train) * correct[name])))})

    finite = all(r["finite"] for r in autograd + numerical)
    masks_stable = all(r["relu_mask_stable"] for r in numerical)
    if bug:


        baseline_ok = all(r["passed"] for r in compare_autograd(loss, correct, reference_loss, reference))
        expected = (baseline_ok and autograd[0]["passed"]
                    and all(not r["passed"] for r in autograd[1:])
                    and all(not r["passed"] for r in numerical)
                    and all(np.allclose(grads[k], 105 * correct[k], atol=1e-12, rtol=1e-12) for k in grads))
    else:
        expected = all(r["passed"] for r in autograd + numerical)

    return {
        "bug": bug, "loss_numpy": loss, "loss_torch": reference_loss,
        "autograd": autograd, "numerical": numerical,
        "scaling": scaling if bug else [], "parameters_unchanged": unchanged,
        "experiment_passed": bool(expected and unchanged and finite and masks_stable),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__},
        "data": {"train_size": len(data.y_train), "test_size": len(data.y_test),
                 "train_counts": np.bincount(data.y_train).tolist(),
                 "test_counts": np.bincount(data.y_test).tolist(),
                 "mean": data.mean.tolist(), "std": data.std.tolist(),
                 "train_indices": data.train_indices.tolist(), "test_indices": data.test_indices.tolist()},
    }


def main() -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bug", action="store_true", help="Навмисно прибрати 1/N у backward")
    args = parser.parse_args()
    if args.bug:
        print(PREDICTION, flush=True)
    result = run(bug=args.bug)
    report = as_markdown(result)
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    stem = "bug" if args.bug else "correct"


    (output / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / f"{stem}.md").write_text(report, encoding="utf-8")
    print(report)
    raise SystemExit(0 if result["experiment_passed"] else 1)


if __name__ == "__main__":
    main()
