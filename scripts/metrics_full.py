"""
metrics.py — EvaluateMetric

 accuracy_a (Accuracy) accuracy_c (caseAccuracy) Compute
EvaluateHungarian algorithm
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Optional, Tuple
from collections import Counter


def find_best_mapping(pred: List[int], true: List[int],
                      num_pred_groups: int, num_true_groups: int) -> Dict[int, int]:
    """
 pred_group -> true_group
 group <=6
    """
    pred_ids = sorted(set(p for p in pred if p is not None))
    true_ids = sorted(set(t for t in true if t is not None))

    if not pred_ids or not true_ids:
        return {}

    best_correct = -1
    best_map = {}

    for perm in permutations(true_ids, min(len(pred_ids), len(true_ids))):
        mapping = dict(zip(pred_ids[:len(perm)], perm))
        correct = sum(1 for p, t in zip(pred, true)
                      if t is not None and p is not None and mapping.get(p) == t)
        if correct > best_correct:
            best_correct = correct
            best_map = mapping

    return best_map


def accuracy_a(pred_groups: List[Optional[int]],
               true_groups: List[Optional[int]]) -> float:
    """
 Accuracy (Accuracy_A)Change / Change
 Evaluate
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

    correct = sum(1 for p, t in zip(valid_pred, valid_true)
                  if p is not None and mapping.get(p) == t)
    return correct / len(valid_true)


def accuracy_c(predictions: List[List[Optional[int]]],
               references: List[List[Optional[int]]],
               threshold: float = 1.0) -> float:
    """
 Case Accuracy (Accuracy_C)Threshold case
 threshold=1.0 Match
    """
    if not predictions:
        return 0.0
    correct = sum(1 for pred, true in zip(predictions, references)
                  if accuracy_a(pred, true) >= threshold)
    return correct / len(predictions)


def evaluate_all(predictions: List[List[Optional[int]]],
                 references: List[List[Optional[int]]],
                 levels: List[str] = None) -> Dict:
    """
 EvaluateReturnLayerResults
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
