"""
compute_dataset_stats.py dataset_judged.json

  1. dataset_judged.json + LLM judger
  2.
     - file_interleave
     - func_interleave
     - Hunk intra_hunk_interleave
     - line_switch_ratio
     -
  3.
     - L0_no_interleave: group hunk
     - L1_file_interleave: group
     - L2_func_interleave: group
     - L3_hunk_interleave: hunk hunk group
  4. key
  5. dataset_with_stats.json

  python compute_dataset_stats.py [--input dataset_judged.json] [--out dataset_with_stats.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Diff build_tangled_dataset.py
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(
    r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)'
)


def get_func_signatures_from_diff(diff_text: str) -> Dict[str, Set[str]]:
    """
     diff
     {filepath: {func_sig1, func_sig2, ...}}
    """
    file_funcs: Dict[str, Set[str]] = defaultdict(set)
    current_file = None

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            parts = line.split(' b/')
            if len(parts) >= 2:
                current_file = parts[-1].strip()
        elif line.startswith('+++ b/'):
            current_file = line[6:].strip()
        elif line.startswith('@@') and current_file:
            m = _HUNK_RE.match(line)
            if m:
                suffix = m.group(5).strip()
                if suffix:
                    file_funcs[current_file].add(suffix)

    return dict(file_funcs)


def get_hunk_func_for_lines(diff_text: str) -> List[Optional[str]]:
    """
     diff +/- hunk
     = None hunk
    """
    result = []
    current_func = None

    for line in diff_text.splitlines():
        if line.startswith('@@'):
            m = _HUNK_RE.match(line)
            if m:
                suffix = m.group(5).strip()
                current_func = suffix if suffix else None
        elif (line.startswith('+') and not line.startswith('+++')) or \
             (line.startswith('-') and not line.startswith('---')):
            result.append(current_func)

    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_entry_stats(entry: Dict) -> Dict:
    """
     dataset_judged.json
     stats dict
    """
    gt = entry['ground_truth']
    num_groups = gt['num_groups']
    files = gt['files']
    source_diff = entry['source'].get('diff', '')

    # --- ---
    total_files = len(files)
    split_files = [f for f in files if f.get('split', False)]
    n_split_files = len(split_files)

    total_changed = 0
    unassigned = 0
    for f in files:
        for la in f.get('lines', []):
            if la['type'] in ('add', 'del'):
                total_changed += 1
                if la['group_id'] is None:
                    unassigned += 1

    # --- ---
    # group
    file_split_ratio = n_split_files / max(total_files, 1)
    has_file_interleave = n_split_files > 0

    # --- ---
    # hunk header group
    hunk_func_list = get_hunk_func_for_lines(source_diff)
    has_func_interleave = False
    func_group_map: Dict[str, Set[int]] = defaultdict(set)

    # ground_truth.files hunk
    change_idx = 0
    for f in files:
        for la in f.get('lines', []):
            if la['type'] in ('add', 'del'):
                gid = la['group_id']
                if gid is not None and change_idx < len(hunk_func_list):
                    func_sig = hunk_func_list[change_idx]
                    if func_sig:
                        # file::func
                        func_key = f"{f['file']}::{func_sig}"
                        func_group_map[func_key].add(gid)
                change_idx += 1

    # group
    func_interleave_count = sum(1 for groups in func_group_map.values() if len(groups) > 1)
    has_func_interleave = func_interleave_count > 0

    # --- Hunk group_id ---
    # hunk group
    has_intra_hunk_interleave = False
    total_switches = 0
    max_switches_in_hunk = 0

    for f in files:
        # hunk hunk header
        current_hunk_groups: List[int] = []
        for la in f.get('lines', []):
            if la['type'] == 'hunk':
                # hunk
                switches = _count_switches(current_hunk_groups)
                total_switches += switches
                if switches > 0:
                    has_intra_hunk_interleave = True
                max_switches_in_hunk = max(max_switches_in_hunk, switches)
                current_hunk_groups = []
            elif la['type'] in ('add', 'del'):
                gid = la['group_id']
                if gid is not None:
                    current_hunk_groups.append(gid)
        # hunk
        switches = _count_switches(current_hunk_groups)
        total_switches += switches
        if switches > 0:
            has_intra_hunk_interleave = True
        max_switches_in_hunk = max(max_switches_in_hunk, switches)

    line_switch_ratio = total_switches / max(total_changed - 1, 1)

    # --- ---
    # group
    group_file_map: Dict[int, Set[str]] = defaultdict(set)
    for f in files:
        for la in f.get('lines', []):
            if la['type'] in ('add', 'del') and la['group_id'] is not None:
                group_file_map[la['group_id']].add(f['file'])

    avg_files_per_group = (
        statistics.mean([len(fs) for fs in group_file_map.values()])
        if group_file_map else 0
    )

    # --- ---
    interleave_level = classify_interleave_level(
        has_file_interleave, has_func_interleave, has_intra_hunk_interleave
    )

    return {
        'total_files': total_files,
        'split_files': n_split_files,
        'file_split_ratio': round(file_split_ratio, 4),
        'has_file_interleave': has_file_interleave,
        'has_func_interleave': has_func_interleave,
        'func_interleave_count': func_interleave_count,
        'has_intra_hunk_interleave': has_intra_hunk_interleave,
        'total_changed_lines': total_changed,
        'unassigned_lines': unassigned,
        'unassigned_ratio': round(unassigned / max(total_changed, 1), 4),
        'total_group_switches': total_switches,
        'max_switches_in_hunk': max_switches_in_hunk,
        'line_switch_ratio': round(line_switch_ratio, 4),
        'num_groups': num_groups,
        'avg_files_per_group': round(avg_files_per_group, 2),
        'interleave_level': interleave_level,
    }


def _count_switches(group_seq: List[int]) -> int:
    """"""
    if len(group_seq) < 2:
        return 0
    return sum(1 for i in range(1, len(group_seq)) if group_seq[i] != group_seq[i - 1])


def classify_interleave_level(
    has_file: bool, has_func: bool, has_intra_hunk: bool
) -> str:
    """
      L0: group
      L1: group
      L2: group
      L3: Hunk hunk group
    """
    if has_intra_hunk:
        return 'L3_hunk_interleave'
    if has_func:
        return 'L2_func_interleave'
    if has_file:
        return 'L1_file_interleave'
    return 'L0_no_interleave'


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_aggregate_stats(dataset: List[Dict]) -> Dict:
    """"""
    total = len(dataset)

    level_counter = Counter(d['entry_stats']['interleave_level'] for d in dataset)

    # difficulty.level
    difficulty_counter = Counter(d['difficulty']['level'] for d in dataset)

    switch_ratios = [d['entry_stats']['line_switch_ratio'] for d in dataset]
    file_split_ratios = [d['entry_stats']['file_split_ratio'] for d in dataset]
    changed_lines = [d['entry_stats']['total_changed_lines'] for d in dataset]
    num_groups_list = [d['entry_stats']['num_groups'] for d in dataset]

    def _dist(vals):
        if not vals:
            return {}
        return {
            'mean': round(statistics.mean(vals), 4),
            'median': round(statistics.median(vals), 4),
            'std': round(statistics.stdev(vals), 4) if len(vals) > 1 else 0,
            'min': round(min(vals), 4),
            'max': round(max(vals), 4),
        }

    return {
        'total_cases': total,
        'interleave_level_distribution': {
            k: {'count': v, 'ratio': round(v / total, 4)}
            for k, v in sorted(level_counter.items())
        },
        'difficulty_level_distribution': {
            k: {'count': v, 'ratio': round(v / total, 4)}
            for k, v in sorted(difficulty_counter.items())
        },
        'line_switch_ratio': _dist(switch_ratios),
        'file_split_ratio': _dist(file_split_ratios),
        'changed_lines_per_case': _dist([float(x) for x in changed_lines]),
        'num_groups': _dist([float(x) for x in num_groups_list]),
        'num_groups_distribution': dict(sorted(Counter(num_groups_list).items())),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=" dataset_judged.json "
    )
    parser.add_argument('--input', type=str, default='dataset_judged.json',
                        help='')
    parser.add_argument('--out', type=str, default='dataset_with_stats.json',
                        help='')
    parser.add_argument('--print-summary', action='store_true',
                        help='')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] : {input_path}")
        return

    with open(input_path, encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"[stats] {len(dataset)} ...")

    for entry in dataset:
        entry['entry_stats'] = compute_entry_stats(entry)

    aggregate = compute_aggregate_stats(dataset)

    output = {
        'aggregate_stats': aggregate,
        'data': dataset,
    }

    out_path = Path(args.out)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[stats] {out_path}")

    if args.print_summary:
        _print_summary(aggregate)
    else:
        print(f"\n[] :")
        for level, info in aggregate['interleave_level_distribution'].items():
            print(f"  {level}: {info['count']} ({info['ratio']*100:.1f}%)")
        print(f"\n[] : mean={aggregate['line_switch_ratio']['mean']:.4f}, "
              f"median={aggregate['line_switch_ratio']['median']:.4f}")


def _print_summary(agg: Dict) -> None:
    print("\n" + "=" * 60)
    print(" ")
    print("=" * 60)

    print(f"\n: {agg['total_cases']} ")

    print("\n")
    for level, info in agg['interleave_level_distribution'].items():
        print(f"  {level}: {info['count']} ({info['ratio']*100:.1f}%)")

    print("\n")
    for level, info in agg['difficulty_level_distribution'].items():
        print(f"  {level}: {info['count']} ({info['ratio']*100:.1f}%)")

    print("\n")
    sr = agg['line_switch_ratio']
    print(f"  mean={sr['mean']:.4f}, median={sr['median']:.4f}, "
          f"min={sr['min']:.4f}, max={sr['max']:.4f}")

    print("\n/case")
    cl = agg['changed_lines_per_case']
    print(f"  mean={cl['mean']:.1f}, median={cl['median']:.1f}, "
          f"min={cl['min']:.0f}, max={cl['max']:.0f}")

    print("\nnum_groups ")
    for k, v in agg['num_groups_distribution'].items():
        print(f"  {k} groups: {v}")

    print("=" * 60)


if __name__ == "__main__":
    main()
