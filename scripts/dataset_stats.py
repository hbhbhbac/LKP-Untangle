"""
dataset_stats.py — Compute dataset statistics reported in Table I of the paper.

Usage:
    python dataset_stats.py --dataset ../data/lkp_untangle.json

Computes:
- Number of cases
- Target groups range
- Changed lines (avg / median)
- Files touched (avg / median)
- Difficulty distribution (Easy:Medium:Hard)
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path


def get_files_from_diff(diff_text: str) -> set:
    files = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) >= 2:
                files.add(parts[-1])
    return files


def get_functions_from_diff(diff_text: str) -> set:
    """Extract file::function signatures from hunk headers."""
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


def count_changed_lines(diff_text: str) -> int:
    count = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            count += 1
        elif line.startswith("-") and not line.startswith("---"):
            count += 1
    return count


def classify_difficulty(entry: dict) -> str:
    """
    Classify difficulty using the 5-level overlap scheme merged to 3 tiers.
    L1 -> Easy (no file overlap)
    L2, L3 -> Medium (file overlap, no function overlap)
    L4, L5 -> Hard (function overlap)
    """
    targets = entry.get("targets", [])
    if len(targets) < 2:
        return "Easy"

    atom_files = [get_files_from_diff(t["diff"]) for t in targets]
    atom_funcs = [get_functions_from_diff(t["diff"]) for t in targets]

    pairs = list(combinations(range(len(targets)), 2))
    pair_file = [bool(atom_files[i] & atom_files[j]) for i, j in pairs]
    pair_func = [bool(atom_funcs[i] & atom_funcs[j]) for i, j in pairs]

    any_file = any(pair_file)
    any_func = any(pair_func)

    if not any_file:
        return "Easy"  # L1
    elif not any_func:
        return "Medium"  # L2 or L3
    else:
        return "Hard"  # L4 or L5


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics.")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    n_cases = len(dataset)
    all_changed_lines = []
    all_files = []
    all_groups = []
    difficulties = []

    for entry in dataset:
        targets = entry["targets"]
        all_groups.append(len(targets))

        # Combine all diffs from source
        source_diff = entry.get("source", {}).get("diff", "")
        total_lines = count_changed_lines(source_diff)
        all_changed_lines.append(total_lines)

        source_files = get_files_from_diff(source_diff)
        all_files.append(len(source_files))

        difficulties.append(classify_difficulty(entry))

    diff_counter = Counter(difficulties)
    e_pct = round(100 * diff_counter["Easy"] / n_cases)
    m_pct = round(100 * diff_counter["Medium"] / n_cases)
    h_pct = round(100 * diff_counter["Hard"] / n_cases)

    print(f"Dataset: {args.dataset}")
    print(f"Cases: {n_cases}")
    print(f"Target groups: {min(all_groups)}--{max(all_groups)}")
    print(f"Changed lines (Avg / Med): {statistics.mean(all_changed_lines):.1f} / "
          f"{statistics.median(all_changed_lines):.1f}")
    print(f"Files (Avg / Med): {statistics.mean(all_files):.1f} / "
          f"{statistics.median(all_files):.1f}")
    print(f"Difficulty E:M:H: {e_pct}:{m_pct}:{h_pct}")
    print(f"  Easy: {diff_counter['Easy']} ({e_pct}%)")
    print(f"  Medium: {diff_counter['Medium']} ({m_pct}%)")
    print(f"  Hard: {diff_counter['Hard']} ({h_pct}%)")


if __name__ == "__main__":
    main()
