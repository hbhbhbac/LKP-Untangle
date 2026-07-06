"""
evaluate_all.py — Evaluate all methods on lkp_untangle.json

Methods: Barnett, Herzig, Flexeme, UTANGO, HD-GNN, SmartCommit + LLM (Qwen3-32B)
"""

from __future__ import annotations

import json
import sys
import time
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import accuracy_a, accuracy_c, evaluate_all, find_best_mapping

# build_tangled_dataset is at project root
from build_tangled_dataset import get_file_paths, get_func_signatures

from faithful.barnett_faithful import predict_barnett_faithful
from faithful.herzig_faithful import predict_herzig_faithful
from faithful.flexeme_faithful import predict_flexeme_faithful
from faithful.utango_faithful import predict_utango_faithful
from faithful.hdgnn_faithful import predict_hdgnn_faithful
from faithful.smartcommit_faithful import predict_smartcommit_faithful

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "data" / "lkp_manual" / "lkp_untangle.json"

# --- Overlap classification ---

def classify_overlap_level(entry: dict) -> str:
    targets = entry.get("targets", [])
    if len(targets) < 2:
        return "L1"
    atom_files = [get_file_paths(t["diff"]) for t in targets]
    atom_funcs = [get_func_signatures(t["diff"]) for t in targets]
    pairs = list(combinations(range(len(targets)), 2))
    pair_file = [bool(atom_files[i] & atom_files[j]) for i, j in pairs]
    pair_func = [bool(atom_funcs[i] & atom_funcs[j]) for i, j in pairs]
    any_file = any(pair_file)
    all_file = all(pair_file)
    any_func = any(pair_func)
    all_func = all(pair_func)
    if not any_file:
        return "L1"
    if not all_file and not any_func:
        return "L2"
    if all_file and not any_func:
        return "L3"
    if any_func and not all_func:
        return "L4"
    return "L5"


OVERLAP_MERGED = {"L1": "L1", "L2": "L2-3", "L3": "L2-3", "L4": "L4-5", "L5": "L4-5"}
OVERLAP_MERGED_ORDER = ["L1", "L2-3", "L4-5"]

STMT_BINS = [("1-20", 1, 20), ("21-50", 21, 50), ("51-100", 51, 100),
             ("101-200", 101, 200), ("201+", 201, float("inf"))]


def get_stmt_bin(n: int) -> str:
    for label, lo, hi in STMT_BINS:
        if lo <= n <= hi:
            return label
    return "201+"


def get_true_labels(entry: dict) -> List[Optional[int]]:
    gt = entry['ground_truth']
    fk = 'files' if 'files' in gt else 'file_assignments'
    return [la['group_id'] for fi in gt[fk] for la in fi['lines']
            if la['type'] in ('add', 'del')]


def get_num_changed(entry: dict) -> int:
    gt = entry['ground_truth']
    fk = 'files' if 'files' in gt else 'file_assignments'
    return sum(1 for fi in gt[fk] for la in fi['lines'] if la['type'] in ('add', 'del'))


# --- Run predictions ---

PREDICT_FNS = {
    'barnett': predict_barnett_faithful,
    'herzig': predict_herzig_faithful,
    'flexeme': predict_flexeme_faithful,
    'utango': predict_utango_faithful,
    'hdgnn': predict_hdgnn_faithful,
    'smartcommit': predict_smartcommit_faithful,
}


def run_method(method: str, dataset: List[Dict]) -> List[List[int]]:
    predict_fn = PREDICT_FNS[method]
    predictions = []
    errors = 0
    for i, entry in enumerate(dataset):
        try:
            pred = predict_fn(entry)
        except Exception:
            n = get_num_changed(entry)
            pred = [0] * n
            errors += 1
        true = get_true_labels(entry)
        if len(pred) < len(true):
            pred.extend([0] * (len(true) - len(pred)))
        elif len(pred) > len(true):
            pred = pred[:len(true)]
        predictions.append(pred)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(dataset)}]", flush=True)
    print(f"    errors={errors}")
    return predictions


def load_llm_predictions(name: str, dataset: List[Dict]) -> Optional[List[List[int]]]:
    """Load LLM predictions and align to ground truth by id."""
    llm_dir = Path(__file__).resolve().parent / "llm" / "results"
    path = llm_dir / f"predictions_{name}.json"
    if not path.exists():
        print(f"    [SKIP] {path} not found")
        return None

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    preds_list = data.get("predictions", data if isinstance(data, list) else [])
    id_to_pred = {}
    for p in preds_list:
        pred = p.get("prediction") or p.get("pred_groups")
        if pred is not None:
            id_to_pred[p["id"]] = pred

    predictions = []
    missing = 0
    for entry in dataset:
        true = get_true_labels(entry)
        eid = entry["id"]
        pred = id_to_pred.get(eid)
        if pred is None:
            pred = [0] * len(true)
            missing += 1
        else:
            if len(pred) < len(true):
                pred.extend([0] * (len(true) - len(pred)))
            elif len(pred) > len(true):
                pred = pred[:len(true)]
        predictions.append(pred)

    print(f"    loaded {len(id_to_pred)} predictions, missing={missing}")
    return predictions


def compute_detailed_metrics(predictions, references, overlaps, stmt_bins):
    """Compute per-case metrics and group by overlap and stmt_bin."""
    per_case_acc = [accuracy_a(p, r) for p, r in zip(predictions, references)]

    # Overall
    acc_a_mean = statistics.mean(per_case_acc)
    acc_a_median = statistics.median(per_case_acc)
    acc_c100 = sum(1 for a in per_case_acc if a >= 1.0) / len(per_case_acc)
    acc_c90 = sum(1 for a in per_case_acc if a >= 0.9) / len(per_case_acc)
    acc_c80 = sum(1 for a in per_case_acc if a >= 0.8) / len(per_case_acc)

    # By overlap (merged)
    by_overlap = defaultdict(list)
    for acc, ovlp in zip(per_case_acc, overlaps):
        by_overlap[OVERLAP_MERGED.get(ovlp, ovlp)].append(acc)

    # By stmt_bin
    by_stmt = defaultdict(list)
    for acc, sb in zip(per_case_acc, stmt_bins):
        by_stmt[sb].append(acc)

    # Cross: overlap x stmt
    by_cross = defaultdict(list)
    for acc, ovlp, sb in zip(per_case_acc, overlaps, stmt_bins):
        by_cross[(OVERLAP_MERGED.get(ovlp, ovlp), sb)].append(acc)

    return {
        'acc_a_mean': acc_a_mean,
        'acc_a_median': acc_a_median,
        'acc_c100': acc_c100,
        'acc_c90': acc_c90,
        'acc_c80': acc_c80,
        'by_overlap': dict(by_overlap),
        'by_stmt': dict(by_stmt),
        'by_cross': dict(by_cross),
        'per_case': per_case_acc,
    }


def main():
    print(f"Loading dataset: {DATASET_PATH}")
    with open(DATASET_PATH, encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entries")

    # Compute metadata
    references = [get_true_labels(e) for e in dataset]
    overlaps = [classify_overlap_level(e) for e in dataset]
    num_changed = [get_num_changed(e) for e in dataset]
    stmt_bins_list = [get_stmt_bin(n) for n in num_changed]

    # Dataset stats
    overlap_counts = Counter(OVERLAP_MERGED.get(o, o) for o in overlaps)
    stmt_counts = Counter(stmt_bins_list)
    print(f"\nOverlap distribution: {dict(sorted(overlap_counts.items()))}")
    print(f"Stmt-bin distribution: {dict(sorted(stmt_counts.items(), key=lambda x: [b[0] for b in STMT_BINS].index(x[0])))}")

    all_results = {}

    # Run faithful methods
    methods = ['barnett', 'herzig', 'hdgnn', 'utango', 'smartcommit', 'flexeme']
    for method in methods:
        print(f"\n[{method}] Running...")
        start = time.time()
        predictions = run_method(method, dataset)
        elapsed = time.time() - start
        print(f"    elapsed: {elapsed:.1f}s")

        metrics = compute_detailed_metrics(predictions, references, overlaps, stmt_bins_list)
        metrics['elapsed_s'] = round(elapsed, 1)
        all_results[method] = metrics
        print(f"    acc_a={metrics['acc_a_mean']:.4f}, acc_c100={metrics['acc_c100']:.4f}")

    # Load LLM predictions
    llm_models = {'qwen3_32b': 'Qwen3-32B'}
    for key, name in llm_models.items():
        print(f"\n[{name}] Loading predictions...")
        predictions = load_llm_predictions(key, dataset)
        if predictions is None:
            continue
        metrics = compute_detailed_metrics(predictions, references, overlaps, stmt_bins_list)
        all_results[name] = metrics
        print(f"    acc_a={metrics['acc_a_mean']:.4f}, acc_c100={metrics['acc_c100']:.4f}")

    # Save raw results
    save_results = {}
    for name, m in all_results.items():
        save_results[name] = {
            'acc_a_mean': round(m['acc_a_mean'], 4),
            'acc_a_median': round(m['acc_a_median'], 4),
            'acc_c100': round(m['acc_c100'], 4),
            'acc_c90': round(m['acc_c90'], 4),
            'acc_c80': round(m['acc_c80'], 4),
            'by_overlap': {k: {'count': len(v), 'acc_a_mean': round(statistics.mean(v), 4),
                               'acc_c100': round(sum(1 for a in v if a >= 1.0)/len(v), 4)}
                          for k, v in m['by_overlap'].items()},
            'by_stmt': {k: {'count': len(v), 'acc_a_mean': round(statistics.mean(v), 4),
                            'acc_c100': round(sum(1 for a in v if a >= 1.0)/len(v), 4)}
                       for k, v in m['by_stmt'].items()},
        }

    with open(OUTPUT_DIR / "results_all.json", 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results_all.json")

    # Generate markdown report
    generate_report(all_results, dataset, overlaps, stmt_bins_list, overlap_counts, stmt_counts, num_changed)


def generate_report(all_results, dataset, overlaps, stmt_bins_list, overlap_counts, stmt_counts, num_changed):
    """Generate evaluation_report.md"""
    lines = []
    W = lines.append

    W("# Commit Untangling MethodEvaluate ")
    W("")
    W("## Data")
    W("")
    W("| | |")
    W("|------|------|")
    W(f"| Data | `lkp_untangle.json` |")
    W(f"| | C (Linux Kernel) |")
    W(f"| | {len(dataset)} |")
    W(f"| Alignment | 100% FULL |")
    W(f"| Change (mean/median) | {statistics.mean(num_changed):.1f} / {statistics.median(num_changed):.0f} |")

    # Overlap distribution
    W("")
    W("### ")
    W("")
    W("| Level | | |")
    W("|-------|------|------|")
    for level in OVERLAP_MERGED_ORDER:
        cnt = overlap_counts.get(level, 0)
        W(f"| {level} | {cnt} | {cnt/len(dataset)*100:.1f}% |")

    # Stmt-bin distribution
    W("")
    W("### Change")
    W("")
    W("| | |")
    W("|------|------|")
    for label, _, _ in STMT_BINS:
        W(f"| {label} | {stmt_counts.get(label, 0)} |")

    W("")
    W("---")
    W("")

    # Section 1: Overall results
    W("## Results")
    W("")
    W("| Method | Acc_A (mean) | Acc_A (median) | Acc_C@100 | Acc_C@90 | Acc_C@80 |")
    W("|------|:---:|:---:|:---:|:---:|:---:|")

    # Sort by acc_a_mean descending
    sorted_methods = sorted(all_results.items(), key=lambda x: x[1]['acc_a_mean'], reverse=True)
    for name, m in sorted_methods:
        bold = "**" if name == sorted_methods[0][0] else ""
        W(f"| {bold}{name}{bold} | {m['acc_a_mean']:.4f} | {m['acc_a_median']:.4f} | "
          f"{m['acc_c100']:.4f} | {m['acc_c90']:.4f} | {m['acc_c80']:.4f} |")

    W("")
    W("---")
    W("")

    # Section 2: By overlap - Acc_A
    W("## Layer — Acc_A (mean)")
    W("")
    header = "| Method |"
    for level in OVERLAP_MERGED_ORDER:
        cnt = overlap_counts.get(level, 0)
        header += f" {level} (n={cnt}) |"
    header += " ALL |"
    W(header)
    W("|------|" + ":---:|" * (len(OVERLAP_MERGED_ORDER) + 1))

    for name, m in sorted_methods:
        row = f"| {name} |"
        for level in OVERLAP_MERGED_ORDER:
            accs = m['by_overlap'].get(level, [])
            if accs:
                row += f" {statistics.mean(accs):.4f} |"
            else:
                row += " - |"
        row += f" {m['acc_a_mean']:.4f} |"
        W(row)

    W("")
    W("## Layer — Acc_C@100")
    W("")
    header = "| Method |"
    for level in OVERLAP_MERGED_ORDER:
        cnt = overlap_counts.get(level, 0)
        header += f" {level} (n={cnt}) |"
    header += " ALL |"
    W(header)
    W("|------|" + ":---:|" * (len(OVERLAP_MERGED_ORDER) + 1))

    for name, m in sorted_methods:
        row = f"| {name} |"
        for level in OVERLAP_MERGED_ORDER:
            accs = m['by_overlap'].get(level, [])
            if accs:
                c100 = sum(1 for a in accs if a >= 1.0) / len(accs)
                row += f" {c100:.4f} |"
            else:
                row += " - |"
        row += f" {m['acc_c100']:.4f} |"
        W(row)

    W("")
    W("## Layer — Acc_C@80")
    W("")
    header = "| Method |"
    for level in OVERLAP_MERGED_ORDER:
        cnt = overlap_counts.get(level, 0)
        header += f" {level} (n={cnt}) |"
    header += " ALL |"
    W(header)
    W("|------|" + ":---:|" * (len(OVERLAP_MERGED_ORDER) + 1))

    for name, m in sorted_methods:
        row = f"| {name} |"
        for level in OVERLAP_MERGED_ORDER:
            accs = m['by_overlap'].get(level, [])
            if accs:
                c80 = sum(1 for a in accs if a >= 0.8) / len(accs)
                row += f" {c80:.4f} |"
            else:
                row += " - |"
        row += f" {m['acc_c80']:.4f} |"
        W(row)

    W("")
    W("---")
    W("")

    # Section 5: By stmt-bin - Acc_A
    stmt_bin_order = [b[0] for b in STMT_BINS]
    W("## ChangeLayer — Acc_A (mean)")
    W("")
    header = "| Method |"
    for sb in stmt_bin_order:
        cnt = stmt_counts.get(sb, 0)
        header += f" {sb} (n={cnt}) |"
    W(header)
    W("|------|" + ":---:|" * len(stmt_bin_order))

    for name, m in sorted_methods:
        row = f"| {name} |"
        for sb in stmt_bin_order:
            accs = m['by_stmt'].get(sb, [])
            if accs:
                row += f" {statistics.mean(accs):.4f} |"
            else:
                row += " - |"
        W(row)

    W("")
    W("## ChangeLayer — Acc_C@100")
    W("")
    header = "| Method |"
    for sb in stmt_bin_order:
        cnt = stmt_counts.get(sb, 0)
        header += f" {sb} (n={cnt}) |"
    W(header)
    W("|------|" + ":---:|" * len(stmt_bin_order))

    for name, m in sorted_methods:
        row = f"| {name} |"
        for sb in stmt_bin_order:
            accs = m['by_stmt'].get(sb, [])
            if accs:
                c100 = sum(1 for a in accs if a >= 1.0) / len(accs)
                row += f" {c100:.4f} |"
            else:
                row += " - |"
        W(row)

    W("")
    W("---")
    W("")

    # Section 7: Cross table (overlap x stmt) for top methods
    W("## × Change — Acc_A (mean)")
    W("")

    top_methods = [name for name, _ in sorted_methods[:6]]
    for method_name in top_methods:
        m = all_results[method_name]
        W(f"### {method_name}")
        W("")
        header = "| Overlap \\ Stmt |"
        for sb in stmt_bin_order:
            header += f" {sb} |"
        header += " ALL |"
        W(header)
        W("|------|" + ":---:|" * (len(stmt_bin_order) + 1))

        for level in OVERLAP_MERGED_ORDER:
            row = f"| {level} |"
            for sb in stmt_bin_order:
                accs = m['by_cross'].get((level, sb), [])
                if accs:
                    row += f" {statistics.mean(accs):.4f} |"
                else:
                    row += " - |"
            level_accs = m['by_overlap'].get(level, [])
            if level_accs:
                row += f" {statistics.mean(level_accs):.4f} |"
            else:
                row += " - |"
            W(row)

        row = "| **ALL** |"
        for sb in stmt_bin_order:
            accs = m['by_stmt'].get(sb, [])
            if accs:
                row += f" **{statistics.mean(accs):.4f}** |"
            else:
                row += " - |"
        row += f" **{m['acc_a_mean']:.4f}** |"
        W(row)
        W("")

    W("---")
    W("")

    # Write report
    report_path = OUTPUT_DIR / "evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
