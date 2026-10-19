"""Підрахунок помилок і точний пошук порога; без sklearn та PyTorch."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Candidate:
    threshold: float
    TP: int
    FP: int
    FN: int
    TN: int
    C: int

    def to_dict(self):
        return asdict(self)


def validate(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Потрібні непорожні одновимірні масиви однакової довжини.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Мітки повинні бути 0/1, оцінки — скінченні.")
    return y.astype(np.int64), scores.astype(np.float64)


def confusion(y, scores, threshold):
    y, scores = validate(y, scores)
    if np.isnan(threshold):
        raise ValueError("Поріг не може бути NaN.")
    positive = scores >= threshold
    tp = int(np.count_nonzero(positive & (y == 1)))
    fp = int(np.count_nonzero(positive & (y == 0)))
    fn = int(np.count_nonzero(~positive & (y == 1)))
    tn = int(np.count_nonzero(~positive & (y == 0)))
    return Candidate(float(threshold), tp, fp, fn, tn, 10 * fn + fp)


def candidates(y, scores):
    """Пороги за спаданням; оцінюємо тільки кінці груп рівних score."""
    y, scores = validate(y, scores)
    order = np.argsort(-scores, kind="stable")
    ordered_scores, ordered_y = scores[order], y[order]
    cumulative_tp = np.cumsum(ordered_y, dtype=np.int64)
    ends = np.flatnonzero(np.r_[ordered_scores[:-1] != ordered_scores[1:], True])
    total_positive = int(y.sum())
    total_negative = len(y) - total_positive
    rows = [Candidate(float("inf"), 0, 0, total_positive, total_negative, 10 * total_positive)]
    for index in ends:
        tp = int(cumulative_tp[index])
        fp = int(index + 1 - tp)
        fn, tn = total_positive - tp, total_negative - fp
        rows.append(Candidate(float(ordered_scores[index]), tp, fp, fn, tn, 10 * fn + fp))
    return rows


def best_candidate(rows):
    return min(rows, key=lambda row: (row.C, -row.threshold))
