"""Відтворюваний аналіз лише з predictions.npz, без CSV і навчання."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from checks import run_checks
from thresholds import candidates, best_candidate, confusion


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def analyze(saved, output):
    # Перевірки виконуються до тестової оцінки; їх збій перериває аналіз.
    checks = run_checks(output)
    with np.load(saved / "predictions.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    summary = []
    for split in ("train", "validation", "test"):
        y = data[f"{split}_y"]
        summary.append(dict(split=split, n=len(y), positives=int(y.sum()), positive_fraction=float(y.mean())))
    write_csv(output / "split_summary.csv", summary)
    y, scores = data["validation_y"], data["validation_scores"]
    rows = candidates(y, scores)
    best = best_candidate(rows)
    write_csv(output / "threshold_candidates.csv", [row.to_dict() for row in rows])
    # Фіксуємо поріг до обчислення тестових метрик.
    write_json(output / "threshold.json", {"threshold": best.threshold, "source": "validation", "rule": "score >= threshold", "cost": "10*FN + FP", "tie_break": "largest threshold"})
    validation = [dict(rule=rule, **confusion(y, scores, threshold).to_dict()) for rule, threshold in [("standard", 0.5), ("optimal", best.threshold), ("all_negative", float("inf"))]]
    write_csv(output / "validation_summary.csv", validation)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
    precision, recall, _ = precision_recall_curve(y, scores)
    fpr, tpr, _ = roc_curve(y, scores)
    auc = float(roc_auc_score(y, scores))
    for stem, x, z, xlabel, ylabel in [
        ("pr", recall, precision, "Recall", "Precision"),
        ("roc", fpr, tpr, "False positive rate", "True positive rate"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(x, z, label="Validation" + (f"; AUC={auc:.6f}" if stem == "roc" else ""))
        if stem == "roc":
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
        else:
            ax.axhline(y.mean(), linestyle="--", color="gray", label="Positive fraction")
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{stem}.png", dpi=160)
        plt.close(fig)

    yt, st = data["test_y"], data["test_scores"]
    test = confusion(yt, st, best.threshold)
    test_metrics = dict(**test.to_dict(), precision=test.TP / (test.TP + test.FP) if test.TP + test.FP else None,
                        recall=test.TP / (test.TP + test.FN) if test.TP + test.FN else None)
    # Лише описове порівняння із заздалегідь заданим 0.5; поріг уже зафіксовано.
    baseline = confusion(yt, st, 0.5)
    error_rows = []
    for kind, mask in [("FP", (yt == 0) & (st >= best.threshold)), ("FN", (yt == 1) & (st < best.threshold))]:
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(data["test_row_ids"][indices])][:5]
        for index in indices:
            error_rows.append(dict(row_id=int(data["test_row_ids"][index]), error=kind, label=int(yt[index]), score=float(st[index]),
                                   Amount=float(data["test_amount"][index]), distance=abs(float(st[index]) - best.threshold)))
    if error_rows:
        write_csv(output / "errors.csv", error_rows)
    else:
        (output / "errors.csv").write_text("row_id,error,label,score,Amount,distance\n", encoding="utf-8")
    write_json(output / "analysis.json", dict(validation=validation, roc_auc=auc, test=test_metrics, test_baseline=baseline.to_dict(), checks=checks))
    print(f"Checks: PASS; validation t*={best.threshold!r}, C={best.C}; test C={test.C}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    analyze(args.saved, args.out)


if __name__ == "__main__":
    main()
