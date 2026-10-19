"""Перевірити здані артефакти й повторний аналіз без доступу до CSV."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import confusion_matrix
from analyze import analyze
from thresholds import confusion


def main():
    root = Path(__file__).resolve().parent
    results = root / "results"
    files = [root / "README.md", *sorted(results.iterdir())]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with np.load(results / "predictions.npz", allow_pickle=False) as data:
        ids = [data[f"{name}_row_ids"] for name in ("train", "validation", "test")]
        all_ids = np.concatenate(ids)
        if not np.array_equal(np.sort(all_ids), np.arange(284807)):
            raise AssertionError("Поділ має пропуски або перетини")
        result = json.loads((results / "analysis.json").read_text(encoding="utf-8"))
        threshold = result["test"]["threshold"]
        threshold = float("inf") if threshold == "+inf" else threshold
        for split in ("validation", "test"):
            y, scores = data[f"{split}_y"], data[f"{split}_scores"]
            row = confusion(y, scores, threshold)
            reference = confusion_matrix(y, scores >= threshold, labels=[0, 1])
            if not np.array_equal(reference, [[row.TN, row.FP], [row.FN, row.TP]]):
                raise AssertionError("Матриця не збігається зі sklearn")

    checking = True
    def forbid_csv_reads(event, args):
        if checking and event == "open" and isinstance(args[0], (str, bytes)):
            path = str(args[0]).lower()
            mode = args[1]
            if Path(args[0]).resolve() == root / "README.md":
                flags = args[2]
                if (mode is not None and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                    raise AssertionError("Аналіз не повинен перезаписувати README")
            if path.endswith(".csv") and (mode is None or "r" in mode or "+" in mode):
                raise AssertionError("Аналіз не повинен читати CSV")

    sys.addaudithook(forbid_csv_reads)
    analyze(results, results)
    checking = False
    changed = [str(p) for p in files if hashlib.sha256(p.read_bytes()).hexdigest() != before[p]]
    if changed:
        raise AssertionError(f"Повторний аналіз змінив результати: {changed}")
    print("PASS: повний поділ, sklearn confusion matrix, аналіз без читання CSV і запису в README, побайтове відтворення всіх результатів.")


if __name__ == "__main__":
    main()
