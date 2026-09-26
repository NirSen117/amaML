"""Validation metrics for the challenge's macro F-beta objective."""
from typing import Iterable

def fbeta(true_ids: set[str], predicted_ids: set[str], beta: float = 0.5) -> float:
    if not true_ids and not predicted_ids:
        return 1.0
    if not true_ids or not predicted_ids:
        return 0.0
    precision = len(true_ids & predicted_ids) / len(predicted_ids)
    recall = len(true_ids & predicted_ids) / len(true_ids)
    denominator = beta * beta * precision + recall
    return 0.0 if denominator == 0 else (1 + beta * beta) * precision * recall / denominator

def macro_fbeta(rows: Iterable[tuple[set[str], set[str]]], beta: float = 0.5) -> float:
    values = [fbeta(a, b, beta) for a, b in rows]
    return sum(values) / len(values) if values else 0.0
