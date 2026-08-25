"""Official-style record matching for MNER, EEG, and GMNER."""

from __future__ import annotations

from functools import lru_cache

from .geometry import region_correct
from .parsing import EntityTriple


def _maximum_matching(
    prediction_count: int,
    gold_count: int,
    edges: dict[tuple[int, int], float],
) -> int:
    """Maximize matched records, using overlap only as a tie-breaker."""

    @lru_cache(None)
    def search(prediction_index: int, used_gold: int) -> tuple[int, float]:
        if prediction_index == prediction_count:
            return 0, 0.0
        best = search(prediction_index + 1, used_gold)
        for gold_index in range(gold_count):
            if used_gold & (1 << gold_index):
                continue
            score = edges.get((prediction_index, gold_index))
            if score is None:
                continue
            count, total = search(prediction_index + 1, used_gold | (1 << gold_index))
            candidate = (count + 1, total + score)
            if candidate > best:
                best = candidate
        return best

    return search(0, 0)[0]


def count_correct_gmner(
    predictions: list[EntityTriple],
    gold: list[EntityTriple],
    iou_threshold: float = 0.5,
) -> int:
    edges: dict[tuple[int, int], float] = {}
    for pred_index, prediction in enumerate(predictions):
        for gold_index, target in enumerate(gold):
            if prediction.text != target.text or prediction.etype != target.etype:
                continue
            correct, score = region_correct(prediction, target, iou_threshold)
            if correct:
                edges[(pred_index, gold_index)] = score
    return _maximum_matching(len(predictions), len(gold), edges)


def count_correct_mner(predictions: list[EntityTriple], gold: list[EntityTriple]) -> int:
    edges = {
        (pred_index, gold_index): 1.0
        for pred_index, prediction in enumerate(predictions)
        for gold_index, target in enumerate(gold)
        if prediction.text == target.text and prediction.etype == target.etype
    }
    return _maximum_matching(len(predictions), len(gold), edges)


def count_correct_eeg(
    predictions: list[EntityTriple],
    gold: list[EntityTriple],
    iou_threshold: float = 0.5,
) -> int:
    edges: dict[tuple[int, int], float] = {}
    for pred_index, prediction in enumerate(predictions):
        for gold_index, target in enumerate(gold):
            if prediction.text != target.text:
                continue
            correct, score = region_correct(prediction, target, iou_threshold)
            if correct:
                edges[(pred_index, gold_index)] = score
    return _maximum_matching(len(predictions), len(gold), edges)


def prf(correct: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


__all__ = [
    "count_correct_eeg",
    "count_correct_gmner",
    "count_correct_mner",
    "prf",
]
