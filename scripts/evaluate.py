"""
evaluate.py — Run case-level evaluation on LKP-Untangle.

Usage:
    python evaluate.py --dataset ../data/lkp_untangle.json --predictions predictions.json

Input format for predictions.json:
    A JSON list where each element is a list of integers representing
    the predicted group label for each changed line in the corresponding case.

Output:
    Case-level success rates: Accuracy_C@100, Accuracy_C@90, Accuracy_C@80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from metrics import accuracy_a, accuracy_c, evaluate_dataset


def load_ground_truth(dataset_path: Path) -> List[List[int]]:
    """Extract ground-truth line-level group labels from the dataset.

    Uses the 'ground_truth' field which contains aligned line-level labels.
    Only changed lines (type='add' or 'del') are included.
    """
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    references = []
    for entry in dataset:
        gt = entry["ground_truth"]
        labels = []
        for f in gt["files"]:
            for la in f["lines"]:
                if la["type"] in ("add", "del"):
                    labels.append(la["group_id"])
        references.append(labels)
    return references


def classify_difficulty(entry: dict) -> str:
    """
    Classify a case into Easy/Medium/Hard based on file and function overlap.
    - Easy: no file overlap between any pair of target groups.
    - Medium: file overlap exists but no function overlap.
    - Hard: function overlap exists between at least one pair.
    """
    targets = entry.get("targets", [])
    if len(targets) < 2:
        return "Easy"

    def get_files(diff_text):
        files = set()
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                parts = line.split(" b/")
                if len(parts) >= 2:
                    files.add(parts[-1])
        return files

    def get_functions(diff_text):
        import re
        funcs = set()
        current_file = ""
        hunk_re = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@\s*(.*)')
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                parts = line.split(" b/")
                if len(parts) >= 2:
                    current_file = parts[-1]
            elif line.startswith("+++ "):
                m = re.match(r'^\+\+\+ (?:b/)?(.*)', line)
                if m:
                    current_file = m.group(1).strip()
            else:
                m = hunk_re.match(line)
                if m:
                    suffix = m.group(1).strip()
                    if suffix:
                        funcs.add(f"{current_file}::{suffix}")
        return funcs

    from itertools import combinations
    atom_files = [get_files(t["diff"]) for t in targets]
    atom_funcs = [get_functions(t["diff"]) for t in targets]

    any_file_overlap = False
    any_func_overlap = False
    for i, j in combinations(range(len(targets)), 2):
        if atom_files[i] & atom_files[j]:
            any_file_overlap = True
        if atom_funcs[i] & atom_funcs[j]:
            any_func_overlap = True

    if not any_file_overlap:
        return "Easy"
    elif not any_func_overlap:
        return "Medium"
    else:
        return "Hard"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate commit decomposition predictions."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset JSON (e.g., lkp_untangle.json)"
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to predictions JSON (list of group-label lists)"
    )
    parser.add_argument(
        "--by-difficulty", action="store_true",
        help="Also report results stratified by difficulty tier"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    pred_path = Path(args.predictions)

    # Load data
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    references = load_ground_truth(dataset_path)

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    assert len(predictions) == len(references), (
        f"Mismatch: {len(predictions)} predictions vs {len(references)} cases"
    )

    # Overall evaluation
    results = evaluate_dataset(predictions, references)
    print("=== Overall Results ===")
    for metric, value in results.items():
        print(f"  {metric}: {value*100:.1f}%")

    # By difficulty
    if args.by_difficulty:
        difficulties = [classify_difficulty(e) for e in dataset]
        for tier in ["Easy", "Medium", "Hard"]:
            tier_preds = [p for p, d in zip(predictions, difficulties) if d == tier]
            tier_refs = [r for r, d in zip(references, difficulties) if d == tier]
            if tier_preds:
                tier_results = evaluate_dataset(tier_preds, tier_refs)
                print(f"\n=== {tier} (n={len(tier_preds)}) ===")
                for metric, value in tier_results.items():
                    print(f"  {metric}: {value*100:.1f}%")


if __name__ == "__main__":
    main()
