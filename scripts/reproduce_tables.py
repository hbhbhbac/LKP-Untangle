"""
reproduce_tables.py — Reproduce paper Tables II-V by running baseline methods
and evaluating against ground_truth labels.

This script:
1. Loads the LKP-Untangle dataset (350 cases)
2. Runs each baseline predict function on every case
3. Evaluates predictions against ground_truth line-level labels
4. Outputs CSV tables matching the paper

Usage:
    cd replication/scripts
    python reproduce_tables.py

Requirements:
    - tree-sitter, tree-sitter-c (for baseline methods)
    - numpy, scipy (for graph methods)
    - The baselines/ directory must be importable

For cross-language results (Java/C#):
    python reproduce_tables.py --cross-dataset

For Table V (fine-tuning scaling):
    Requires running Qwen3-32B variants. See llm_finetune/README.md.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
BASELINES_DIR = REPO_DIR / "baselines"

sys.path.insert(0, str(SCRIPT_DIR))
# Add REPO_DIR so 'baselines' can be imported as a package (needed for .. imports)
sys.path.insert(0, str(REPO_DIR))

from metrics import accuracy_a, accuracy_c


# ---------------------------------------------------------------------------
# Ground truth extraction (matches evaluate.py)
# ---------------------------------------------------------------------------

def get_true_labels(entry: dict) -> List[Optional[int]]:
    """Extract ground-truth line labels from the dataset entry."""
    gt = entry['ground_truth']
    fk = 'files' if 'files' in gt else 'file_assignments'
    return [la['group_id'] for fi in gt[fk] for la in fi['lines']
            if la['type'] in ('add', 'del')]


def get_num_changed(entry: dict) -> int:
    gt = entry['ground_truth']
    fk = 'files' if 'files' in gt else 'file_assignments'
    return sum(1 for fi in gt[fk] for la in fi['lines']
               if la['type'] in ('add', 'del'))


# ---------------------------------------------------------------------------
# Difficulty classification
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@\s*(.*)')


def get_files_from_diff(diff_text: str) -> set:
    files = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) >= 2:
                files.add(parts[-1])
    return files


def get_functions_from_diff(diff_text: str) -> set:
    funcs = set()
    current_file = ""
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
            m = _HUNK_RE.match(line)
            if m:
                suffix = m.group(1).strip()
                if suffix:
                    funcs.add(f"{current_file}::{suffix}")
    return funcs


def classify_difficulty(entry: dict) -> str:
    targets = entry.get("targets", [])
    if len(targets) < 2:
        return "Easy"
    atom_files = [get_files_from_diff(t["diff"]) for t in targets]
    atom_funcs = [get_functions_from_diff(t["diff"]) for t in targets]
    pairs = list(combinations(range(len(targets)), 2))
    any_file = any(bool(atom_files[i] & atom_files[j]) for i, j in pairs)
    any_func = any(bool(atom_funcs[i] & atom_funcs[j]) for i, j in pairs)
    if not any_file:
        return "Easy"
    elif not any_func:
        return "Medium"
    else:
        return "Hard"


STMT_BINS = [("1-20", 1, 20), ("21-50", 21, 50), ("51-100", 51, 100),
             ("101-200", 101, 200), ("201+", 201, float("inf"))]


def get_size_bin(n: int) -> str:
    for label, lo, hi in STMT_BINS:
        if lo <= n <= hi:
            return label
    return "201+"


# ---------------------------------------------------------------------------
# Run predictions
# ---------------------------------------------------------------------------

def run_method(method_name: str, predict_fn, dataset: list) -> List[List[int]]:
    """Run a predict function on all dataset entries, aligned to ground_truth."""
    predictions = []
    errors = 0
    for i, entry in enumerate(dataset):
        true = get_true_labels(entry)
        try:
            pred = predict_fn(entry)
        except Exception as e:
            pred = [0] * len(true)
            errors += 1

        # Align length
        if len(pred) < len(true):
            pred.extend([0] * (len(true) - len(pred)))
        elif len(pred) > len(true):
            pred = pred[:len(true)]
        predictions.append(pred)

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(dataset)}]", flush=True)

    if errors:
        print(f"    {method_name}: {errors} errors (fallback to all-zero)")
    return predictions


def eval_thresholds(preds, refs):
    c100 = accuracy_c(preds, refs, 1.0)
    c90 = accuracy_c(preds, refs, 0.9)
    c80 = accuracy_c(preds, refs, 0.8)
    return c100, c90, c80


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_tables(dataset: list, methods: dict, output_dir: Path):
    """Run all methods and generate Tables II, III, IV."""
    references = [get_true_labels(e) for e in dataset]
    difficulties = [classify_difficulty(e) for e in dataset]
    sizes = [get_num_changed(e) for e in dataset]
    size_bins = [get_size_bin(s) for s in sizes]

    all_results = {}

    # Run each method
    for method_name, predict_fn in methods.items():
        print(f"\n  Running {method_name}...", flush=True)
        t0 = time.time()
        preds = run_method(method_name, predict_fn, dataset)
        elapsed = time.time() - t0
        all_results[method_name] = preds
        c100, c90, c80 = eval_thresholds(preds, references)
        print(f"    {method_name}: @100={c100*100:.1f}% @90={c90*100:.1f}% "
              f"@80={c80*100:.1f}% ({elapsed:.1f}s)")

    # --- Table II (LKP-Untangle rows) ---
    print("\n=== Table II: Overall Results (LKP-Untangle) ===")
    rows_t2 = [["Dataset", "Method", "Accuracy_C@100", "Accuracy_C@90", "Accuracy_C@80"]]
    for method_name, preds in all_results.items():
        c100, c90, c80 = eval_thresholds(preds, references)
        rows_t2.append(["LKP-Untangle", method_name,
                        f"{c100*100:.1f}%", f"{c90*100:.1f}%", f"{c80*100:.1f}%"])
        print(f"  {method_name}: {c100*100:.1f}% / {c90*100:.1f}% / {c80*100:.1f}%")

    with open(output_dir / "table2_overall_results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_t2)

    # --- Table III: By difficulty ---
    print("\n=== Table III: By Difficulty ===")
    header = ["Method", "Easy_C@100", "Medium_C@100", "Hard_C@100",
              "Easy_C@90", "Medium_C@90", "Hard_C@90",
              "Easy_C@80", "Medium_C@80", "Hard_C@80"]
    rows_t3 = [header]
    for method_name, preds in all_results.items():
        row = [method_name]
        for threshold in [1.0, 0.9, 0.8]:
            for tier in ["Easy", "Medium", "Hard"]:
                t_preds = [p for p, d in zip(preds, difficulties) if d == tier]
                t_refs = [r for r, d in zip(references, difficulties) if d == tier]
                score = accuracy_c(t_preds, t_refs, threshold) if t_preds else 0
                row.append(f"{score*100:.1f}%")
        rows_t3.append(row)

    with open(output_dir / "table3_difficulty_results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_t3)

    # --- Table IV: By size (Acc_C@100 only) ---
    print("\n=== Table IV: By Size (Exact Success) ===")
    rows_t4 = [["Method", "1-20", "21-50", "51-100", "101-200", "201+"]]
    for method_name, preds in all_results.items():
        row = [method_name]
        for bin_label, _, _ in STMT_BINS:
            b_preds = [p for p, b in zip(preds, size_bins) if b == bin_label]
            b_refs = [r for r, b in zip(references, size_bins) if b == bin_label]
            score = accuracy_c(b_preds, b_refs, 1.0) if b_preds else 0
            row.append(f"{score*100:.1f}%")
        rows_t4.append(row)

    with open(output_dir / "table4_size_results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_t4)

    print(f"\n  Tables saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce paper tables.")
    parser.add_argument("--dataset", default=None,
                        help="LKP-Untangle dataset path (default: ../data/lkp_untangle.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../results)")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Methods to run (default: all). Options: barnett herzig flexeme utango hdgnn smartcommit")
    parser.add_argument("--cross-dataset", action="store_true",
                        help="Also run cross-dataset evaluation (Java/C#)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else REPO_DIR / "data" / "lkp_untangle.json"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_DIR / "results" / "reproduced"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"  {len(dataset)} cases loaded")

    # Import baseline predict functions
    available_methods = {}
    method_names = args.methods or ["barnett", "herzig", "flexeme", "utango", "hdgnn", "smartcommit"]

    for m in method_names:
        try:
            if m == "barnett":
                from baselines.barnett.barnett_faithful import predict_barnett_faithful
                available_methods["Barnett"] = predict_barnett_faithful
            elif m == "herzig":
                from baselines.herzig.herzig_faithful import predict_herzig_faithful
                available_methods["Herzig"] = predict_herzig_faithful
            elif m == "flexeme":
                from baselines.flexeme.flexeme_faithful import predict_flexeme_faithful
                available_methods["Flexeme"] = predict_flexeme_faithful
            elif m == "utango":
                from baselines.utango.utango_faithful import predict_utango_faithful
                available_methods["UTANGO"] = predict_utango_faithful
            elif m == "hdgnn":
                # Paper uses per-dataset trained GNN model; fall back to heuristic
                model_path = BASELINES_DIR / "hdgnn" / "hdgnn_lkp_model.pt"
                if model_path.exists():
                    try:
                        import torch
                        from baselines.hdgnn.hdgnn_source import HDGNNModel, predict_hdgnn_source
                        device = torch.device("cpu")
                        model = HDGNNModel()
                        checkpoint = torch.load(str(model_path), map_location=device)
                        state = checkpoint.get("model_state_dict", checkpoint)
                        model.load_state_dict(state)
                        model.to(device).eval()
                        available_methods["HD-GNN"] = lambda entry, _m=model, _d=device: predict_hdgnn_source(entry, _m, _d)
                        print(f"  Loaded: hdgnn (GNN model from {model_path.name})")
                    except Exception as e:
                        from baselines.hdgnn.hdgnn_faithful import predict_hdgnn_faithful
                        available_methods["HD-GNN"] = predict_hdgnn_faithful
                        print(f"  Loaded: hdgnn (heuristic fallback, GNN load failed: {e})")
                else:
                    from baselines.hdgnn.hdgnn_faithful import predict_hdgnn_faithful
                    available_methods["HD-GNN"] = predict_hdgnn_faithful
                    print(f"  Loaded: hdgnn (heuristic — no model file, train with train_hdgnn_source.py)")
                continue  # skip generic print below
            elif m == "smartcommit":
                from baselines.smartcommit.smartcommit_faithful import predict_smartcommit_faithful
                available_methods["SmartCommit"] = predict_smartcommit_faithful
            print(f"  Loaded: {m}")
        except ImportError as e:
            print(f"  [SKIP] {m}: {e}")

    if not available_methods:
        print("\nERROR: No baseline methods could be imported.")
        print("Make sure dependencies are installed: pip install tree-sitter tree-sitter-c numpy scipy")
        print("\nAlternatively, run evaluate.py from the project root for the full evaluation.")
        sys.exit(1)

    generate_tables(dataset, available_methods, output_dir)

    if args.cross_dataset:
        print("\n" + "=" * 60)
        print("Cross-dataset evaluation requires running methods on Java/C# val splits.")
        print("Use: python ../baselines/eval_all_datasets.py")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("NOTE: Table V (fine-tuning scaling) requires Qwen3-32B inference.")
    print("See scripts/llm_finetune/README.md for instructions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
