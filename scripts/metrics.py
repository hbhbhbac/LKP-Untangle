"""
metrics.py — Evaluation metrics for commit decomposition.

Provides:
- accuracy_a: Line-level accuracy (permutation-invariant).
- accuracy_c: Case-level success rate at threshold tau.

These metrics implement Equation (1) from the paper:
    Accuracy_C@tau = (1/N) * sum_{i=1}^{N} I[s_i >= tau/100]

where s_i is the best changed-line agreement score for case i after
optimal group matching via exhaustive permutation search.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Optional


def find_best_mapping(
    pred: List[int], true: List[int],
    num_pred_groups: int, num_true_groups: int
) -> Dict[int, int]:
    """
    Find optimal pred_group -> true_group mapping via exhaustive permutation.
    Works efficiently for small group counts (<=6).
    """
    pred_ids = sorted(set(p for p in pred if p is not None))
    true_ids = sorted(set(t for t in true if t is not None))

    if not pred_ids or not true_ids:
        return {}

    best_correct = -1
    best_map = {}

    for perm in permutations(true_ids, min(len(pred_ids), len(true_ids))):
        mapping = dict(zip(pred_ids[:len(perm)], perm))
        correct = sum(
            1 for p, t in zip(pred, true)
            if t is not None and p is not None and mapping.get(p) == t
        )
        if correct > best_correct:
            best_correct = correct
            best_map = mapping

    return best_map


def accuracy_a(
    pred_groups: List[Optional[int]],
    true_groups: List[Optional[int]]
) -> float:
    """
    Line-level accuracy (Accuracy_A): fraction of changed lines
    correctly assigned after optimal permutation matching.
    """
    valid_pred = []
    valid_true = []
    for p, t in zip(pred_groups, true_groups):
        if t is not None:
            valid_pred.append(p)
            valid_true.append(t)

    if not valid_true:
        return 0.0

    num_pred = len(set(p for p in valid_pred if p is not None))
    num_true = len(set(valid_true))

    mapping = find_best_mapping(valid_pred, valid_true, num_pred, num_true)

    correct = sum(
        1 for p, t in zip(valid_pred, valid_true)
        if p is not None and mapping.get(p) == t
    )
    return correct / len(valid_true)


def accuracy_c(
    predictions: List[List[Optional[int]]],
    references: List[List[Optional[int]]],
    threshold: float = 1.0
) -> float:
    """
    Case-level success rate (Accuracy_C@tau).

    A case is counted as successful if its line-level accuracy >= threshold.
    threshold=1.0 corresponds to Accuracy_C@100 (exact match).
    threshold=0.9 corresponds to Accuracy_C@90.
    threshold=0.8 corresponds to Accuracy_C@80.
    """
    if not predictions:
        return 0.0
    correct = sum(
        1 for pred, true in zip(predictions, references)
        if accuracy_a(pred, true) >= threshold
    )
    return correct / len(predictions)


def evaluate_dataset(
    predictions: List[List[Optional[int]]],
    references: List[List[Optional[int]]]
) -> Dict[str, float]:
    """
    Full evaluation returning all three threshold metrics.
    """
    return {
        "Accuracy_C@100": round(accuracy_c(predictions, references, 1.0), 4),
        "Accuracy_C@90": round(accuracy_c(predictions, references, 0.9), 4),
        "Accuracy_C@80": round(accuracy_c(predictions, references, 0.8), 4),
    }


def evaluate_all(
    predictions: List[List[Optional[int]]],
    references: List[List[Optional[int]]],
    levels: List[str] = None
) -> Dict:
    """
    Full evaluation returning overall and per-difficulty-level results.
    """
    import statistics

    all_accs = [accuracy_a(p, t) for p, t in zip(predictions, references)]

    result = {
        'total_cases': len(all_accs),
        'accuracy_a_mean': round(statistics.mean(all_accs), 4) if all_accs else 0,
        'accuracy_a_median': round(statistics.median(all_accs), 4) if all_accs else 0,
        'accuracy_c_100': round(accuracy_c(predictions, references, 1.0), 4),
        'accuracy_c_90': round(accuracy_c(predictions, references, 0.9), 4),
        'accuracy_c_80': round(accuracy_c(predictions, references, 0.8), 4),
    }

    if levels:
        level_groups = {}
        for acc, level in zip(all_accs, levels):
            level_groups.setdefault(level, []).append(acc)

        result['by_level'] = {}
        for level in sorted(level_groups.keys()):
            accs = level_groups[level]
            result['by_level'][level] = {
                'count': len(accs),
                'accuracy_a_mean': round(statistics.mean(accs), 4),
                'accuracy_c_100': round(sum(1 for a in accs if a >= 1.0) / len(accs), 4),
            }

    return result
