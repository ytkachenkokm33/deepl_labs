"""Незалежний повільний перебір дозволений лише для малих перевірок."""
import csv
from pathlib import Path
from thresholds import candidates, best_candidate


CASES = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0], 0.4),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0], float("inf")),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 11, 0.8),
}


def direct_search(y, scores):
    # Список кандидатів і лічильники незалежні від швидкого алгоритму.
    rows = []
    for threshold in sorted(set(scores) | {float("inf")}, reverse=True):
        tp = fp = fn = tn = 0
        for label, score in zip(y, scores, strict=True):
            if score >= threshold:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
            elif label == 1:
                fn += 1
            else:
                tn += 1
        rows.append(dict(threshold=threshold, TP=tp, FP=fp, FN=fn, TN=tn, C=10 * fn + fp))
    minimum = min(row["C"] for row in rows)
    threshold = max(row["threshold"] for row in rows if row["C"] == minimum)
    return rows, threshold


def run_checks(output=None):
    reports = {}
    for name, (scores, y, expected) in CASES.items():
        slow, slow_best = direct_search(y, scores)
        fast = candidates(y, scores)
        if [row.to_dict() for row in fast] != slow:
            raise AssertionError(f"{name}: лічильники/вартості не збігаються")
        if best_candidate(fast).threshold != slow_best or slow_best != expected:
            raise AssertionError(f"{name}: неправильний вибір порога")
        reports[name] = {"rows": slow, "threshold": slow_best, "passed": True}
        if output is not None:
            path = Path(output)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"check_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(slow[0]))
                writer.writeheader()
                writer.writerows(slow)
    return reports


if __name__ == "__main__":
    for name, report in run_checks().items():
        print(f"{name}: PASS, threshold={report['threshold']}")
