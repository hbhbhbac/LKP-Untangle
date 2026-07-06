"""
build_tangled_dataset.py LKML series tangled commit

 --strategy series
  2wayfile 2-patch
  2wayfunction 2-patch
  connected_file
  connected_func
  all_file series patch
  all_func series patch
  all_direct series patch ≥ 2

  python build_tangled_dataset.py --mode build
      [--strategy 1 0 0 0 0 0 0] # 2wayfile
      [--out dataset_tangled.json] [--series-limit N] [--case-limit N]

  python build_tangled_dataset.py --mode analyze [--input dataset_tangled.json]

  python build_tangled_dataset.py --mode filter  [--input dataset_tangled.json]
      [--min-stmts N] [--max-stmts N] [--min-atoms N] [--max-atoms N]
      [--out dataset_tangled_filtered.json]
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# & build_version_split_dataset.py
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE / "annotation_tool" / "backend"
_CODE_DIR = _HERE / "code"

sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_CODE_DIR))

from core import STRUCTURE_DIR                                    # noqa: E402
from processors.sqlite.db_utils import get_db_path               # noqa: E402
from relation_builder.series_builder import get_global_series_db_path  # noqa: E402
from diff_grouper import extract_diff                             # noqa: E402

# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(
    r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)'
)
_FILE_RE = re.compile(r'^(?:---|\+\+\+) (?:a/|b/)?(.*)')


def parse_diff(diff_text: str) -> List[Dict]:
    """
     unified diff file file hunks
    hunk: {old_start, old_count, new_start, new_count, header_suffix, lines: [{type, content}]}
    """
    files = []
    cur_file = None
    cur_hunk = None
    old_path = new_path = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith('diff --git'):
            if cur_file is not None:
                if cur_hunk is not None:
                    cur_file['hunks'].append(cur_hunk)
                    cur_hunk = None
                files.append(cur_file)
            cur_file = {'old_path': None, 'new_path': None, 'hunks': []}
            old_path = new_path = None
            continue

        if raw_line.startswith('--- '):
            m = re.match(r'^--- (?:a/)?(.*)', raw_line)
            if m:
                old_path = m.group(1).strip()
                if cur_file is not None:
                    cur_file['old_path'] = old_path
            continue

        if raw_line.startswith('+++ '):
            m = re.match(r'^\+\+\+ (?:b/)?(.*)', raw_line)
            if m:
                new_path = m.group(1).strip()
                if cur_file is not None:
                    cur_file['new_path'] = new_path
            continue

        m = _HUNK_RE.match(raw_line)
        if m:
            if cur_file is None:
                cur_file = {'old_path': old_path, 'new_path': new_path, 'hunks': []}
            if cur_hunk is not None:
                cur_file['hunks'].append(cur_hunk)
            cur_hunk = {
                'old_start': int(m.group(1)),
                'old_count': int(m.group(2)) if m.group(2) is not None else 1,
                'new_start': int(m.group(3)),
                'new_count': int(m.group(4)) if m.group(4) is not None else 1,
                'header_suffix': m.group(5),
                'lines': [],
            }
            continue

        if cur_hunk is not None:
            if raw_line.startswith('+'):
                cur_hunk['lines'].append({'type': 'add', 'content': raw_line[1:]})
            elif raw_line.startswith('-'):
                cur_hunk['lines'].append({'type': 'del', 'content': raw_line[1:]})
            elif raw_line.startswith(' ') or raw_line == '':
                cur_hunk['lines'].append({'type': 'ctx', 'content': raw_line[1:] if raw_line else ''})

    if cur_file is not None:
        if cur_hunk is not None:
            cur_file['hunks'].append(cur_hunk)
        files.append(cur_file)

    return files


def get_file_paths(diff_text: str) -> Set[str]:
    """ diff """
    paths = set()
    for f in parse_diff(diff_text):
        if f['new_path'] and f['new_path'] != '/dev/null':
            paths.add(f['new_path'])
        elif f['old_path'] and f['old_path'] != '/dev/null':
            paths.add(f['old_path'])
    return paths


def get_func_signatures(diff_text: str) -> Set[str]:
    """ diff hunk header @@ ... @@ func_name"""
    sigs = set()
    for f in parse_diff(diff_text):
        fpath = f.get('new_path') or f.get('old_path') or ''
        for h in f['hunks']:
            suffix = h.get('header_suffix', '').strip()
            if suffix:
                sigs.add(f"{fpath}::{suffix}")
    return sigs


# ---------------------------------------------------------------------------
# Hunk rebase patch_k hunk old_start
# ---------------------------------------------------------------------------

def rebase_hunk_old_start(
    hunk_old_start: int,
    preceding_hunks: List[Dict],
) -> int:
    """
    patch_k hunk.old_start patch_1..patch_{k-1}

       offset = sum of (new_count - old_count) for each preceding hunk
      whose old_start (in original coords) < current hunk's original old_start.

     preceding_hunks old_start
     offset
    """
    offset = 0
    # preceding_hunks rebase
    for ph in preceding_hunks:
        ph_orig_start = ph['orig_old_start']
        if ph_orig_start < hunk_old_start - offset:
            offset += ph['new_count'] - ph['old_count']
    return hunk_old_start - offset


def merge_diffs(patch_diffs: List[str]) -> Optional[str]:
    """
     patch diff tangled diff

      1. patch hunks
      2. patch hunk old_start rebase
      3. orig_old_start unified diff

     diff None
    """
    # patch hunks
    # file_hunks[filepath] = list of (patch_idx, hunk_with_orig_coords)
    file_hunks: Dict[str, List[Dict]] = defaultdict(list)
    file_meta: Dict[str, Dict] = {}  # filepath -> {old_path, new_path}

    for patch_idx, diff_text in enumerate(patch_diffs):
        parsed = parse_diff(diff_text)
        for fobj in parsed:
            fpath = fobj.get('new_path') or fobj.get('old_path') or ''
            if not fpath or fpath == '/dev/null':
                continue
            if fpath not in file_meta:
                file_meta[fpath] = {
                    'old_path': fobj.get('old_path', fpath),
                    'new_path': fobj.get('new_path', fpath),
                }
            # patch hunks rebase
            # preceding_hunks = patch rebase hunks
            preceding = [h for h in file_hunks[fpath]]  # already rebased
            for hunk in fobj['hunks']:
                orig_start = rebase_hunk_old_start(hunk['old_start'], preceding)
                rebased = dict(hunk)
                rebased['orig_old_start'] = orig_start
                rebased['patch_idx'] = patch_idx
                file_hunks[fpath].append(rebased)
                preceding.append(rebased)

    if not file_hunks:
        return None

    # diff
    lines = []
    for fpath in sorted(file_hunks.keys()):
        meta = file_meta[fpath]
        old_p = meta['old_path']
        new_p = meta['new_path']
        lines.append(f'diff --git a/{old_p} b/{new_p}')
        lines.append(f'--- a/{old_p}')
        lines.append(f'+++ b/{new_p}')

        # hunks
        sorted_hunks = sorted(file_hunks[fpath], key=lambda h: h['orig_old_start'])

        # new_start
        new_offset = 0 # add-del
        for h in sorted_hunks:
            orig_start = h['orig_old_start']
            old_count = h['old_count']
            new_count = h['new_count']
            new_start = orig_start + new_offset
            suffix = h.get('header_suffix', '')
            lines.append(f'@@ -{orig_start},{old_count} +{new_start},{new_count} @@{suffix}')
            for ln in h['lines']:
                t = ln['type']
                c = ln['content']
                if t == 'add':
                    lines.append(f'+{c}')
                elif t == 'del':
                    lines.append(f'-{c}')
                else:
                    lines.append(f' {c}')
            new_offset += new_count - old_count

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_ALL_EPOCHS: Optional[List[int]] = None


def _get_all_epochs() -> List[int]:
    global _ALL_EPOCHS
    if _ALL_EPOCHS is None:
        epochs = []
        for db_file in STRUCTURE_DIR.glob("lkml_epoch_*.sqlite"):
            m = re.search(r"epoch_(\d+)", db_file.name)
            if m:
                epochs.append(int(m.group(1)))
        _ALL_EPOCHS = sorted(epochs)
    return _ALL_EPOCHS


def fetch_multi_patch_series(limit: Optional[int] = None) -> List[Dict]:
    """
     lkml_series_global.sqlite ≥2 patch series
     [{series_id, patch_email_ids, representative_epoch, patch_version}]
    """
    db_path = get_global_series_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql = """
            SELECT id, patch_email_ids, representative_epoch, patch_version
            FROM series
            WHERE json_array_length(patch_email_ids) >= 2
        """
        if limit:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql).fetchall()
        result = []
        for row in rows:
            try:
                eids = json.loads(row['patch_email_ids'])
            except Exception:
                continue
            result.append({
                'series_id': row['id'],
                'patch_email_ids': eids,
                'representative_epoch': row['representative_epoch'],
                'patch_version': row['patch_version'],
            })
        return result
    finally:
        conn.close()


def _count_single_patch_series() -> int:
    """ series DB 1 patch series """
    db_path = get_global_series_db_path()
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM series WHERE json_array_length(patch_email_ids) = 1"
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def fetch_patch_info(email_id: int, epoch: int) -> Optional[Dict]:
    """
     patch subject, diff
    email_id DB iddiff body extract_diff()
     representative_epoch epoch
    """
    all_epochs = _get_all_epochs()
    search_epochs = [epoch] + [e for e in all_epochs if e != epoch]

    for ep in search_epochs:
        db_path = get_db_path(ep)
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT id, subject, body FROM emails WHERE id = ? AND is_patch = 1",
                    (email_id,)
                ).fetchone()
                if row:
                    diff = extract_diff(row['body'] or '')
                    return {
                        'email_id': row['id'],
                        'subject': row['subject'] or '',
                        'diff': diff,
                    }
            finally:
                conn.close()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def patches_share_file(diff_a: str, diff_b: str) -> bool:
    """ patch """
    return bool(get_file_paths(diff_a) & get_file_paths(diff_b))


def patches_share_func(diff_a: str, diff_b: str) -> bool:
    """ patch hunk header"""
    return bool(get_func_signatures(diff_a) & get_func_signatures(diff_b))


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

_STMT_RE = re.compile(r'^\s*(if|for|while|switch|return|break|continue|goto|do)\b|[;{}]')


def count_code_lines(diff_text: str) -> int:
    """ diff /"""
    count = 0
    _COMMENT_RE = re.compile(r'^\s*(//|/\*|\*|#)')
    for line in diff_text.splitlines():
        if line.startswith('+') or line.startswith('-'):
            content = line[1:].strip()
            if content and not _COMMENT_RE.match(content):
                count += 1
    return count


def count_statements(diff_text: str) -> int:
    """ diff ; { } """
    count = 0
    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            content = line[1:]
            if _STMT_RE.search(content):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------

def build_line_assignments(
    merged_diff: str,
    atom_diffs: List[str],
) -> List[Dict]:
    """
     merged_diff atom patch
     [{type, content, group_id}] group_id 1 None
    """
    # atom
    from collections import Counter
    atom_add_counters: List[Counter] = []
    atom_del_counters: List[Counter] = []
    for diff in atom_diffs:
        add_c: Counter = Counter()
        del_c: Counter = Counter()
        for line in diff.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                add_c[line[1:]] += 1
            elif line.startswith('-') and not line.startswith('---'):
                del_c[line[1:]] += 1
        atom_add_counters.append(add_c)
        atom_del_counters.append(del_c)

    # merged_diff
    assignments = []
    remaining_add = [Counter(c) for c in atom_add_counters]
    remaining_del = [Counter(c) for c in atom_del_counters]

    for line in merged_diff.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            content = line[1:]
            group_id = None
            for i, rc in enumerate(remaining_add):
                if rc[content] > 0:
                    rc[content] -= 1
                    group_id = i + 1
                    break
            assignments.append({'type': 'add', 'content': line, 'group_id': group_id})
        elif line.startswith('-') and not line.startswith('---'):
            content = line[1:]
            group_id = None
            for i, rc in enumerate(remaining_del):
                if rc[content] > 0:
                    rc[content] -= 1
                    group_id = i + 1
                    break
            assignments.append({'type': 'del', 'content': line, 'group_id': group_id})

    return assignments


def compute_ground_truth_quality(assignments: List[Dict], num_groups: int) -> Dict:
    """ ground truth """
    total = len(assignments)
    if total == 0:
        return {'all_assigned': False, 'all_groups_covered': False, 'perfect': False,
                'unassigned_count': 0, 'unassigned_ratio': 0.0}
    assigned = sum(1 for a in assignments if a['group_id'] is not None)
    groups_seen = {a['group_id'] for a in assignments if a['group_id'] is not None}
    all_assigned = assigned == total
    all_covered = len(groups_seen) == num_groups
    return {
        'all_assigned': all_assigned,
        'all_groups_covered': all_covered,
        'perfect': all_assigned and all_covered,
        'unassigned_count': total - assigned,
        'unassigned_ratio': round((total - assigned) / total, 4),
    }


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------

def make_tangled_case(
    series_id: int,
    atom_patches: List[Dict],  # [{message_id, subject, diff, patch_idx}]
    overlap_type: str,  # 'file' or 'func'
) -> Optional[Dict]:
    """
     atom patch tangled case
     case dict None
    """
    diffs = [p['diff'] for p in atom_patches]
    merged = merge_diffs(diffs)
    if not merged:
        return None

    num_groups = len(atom_patches)
    assignments = build_line_assignments(merged, diffs)
    gt_quality = compute_ground_truth_quality(assignments, num_groups)

    # atom
    atom_stmts = [count_statements(p['diff']) for p in atom_patches]
    total_stmts = count_statements(merged)

    return {
        'series_id': series_id,
        'label': 'tangled',
        'overlap_type': overlap_type,
        'num_atoms': num_groups,
        'tangled_diff': merged,
        'atoms': [
            {
                'message_id': p['message_id'],
                'subject': p['subject'],
                'diff': p['diff'],
                'patch_idx': p['patch_idx'],
                'stmt_count': atom_stmts[i],
            }
            for i, p in enumerate(atom_patches)
        ],
        'ground_truth': {
            'num_groups': num_groups,
            'assignments': assignments,
            'quality': gt_quality,
        },
        'stats': {
            'total_stmts': total_stmts,
            'atom_stmts': atom_stmts,
            'total_change_lines': len(assignments),
            'unassigned_count': gt_quality['unassigned_count'],
        },
    }


def make_negative_case(patch: Dict) -> Dict:
    """ case patch"""
    return {
        'series_id': patch.get('series_id', 0),
        'label': 'no_split',
        'overlap_type': None,
        'num_atoms': 1,
        'tangled_diff': patch['diff'],
        'atoms': [
            {
                'message_id': patch['message_id'],
                'subject': patch['subject'],
                'diff': patch['diff'],
                'patch_idx': patch.get('patch_idx', 0),
                'stmt_count': count_statements(patch['diff']),
            }
        ],
        'ground_truth': None,
        'stats': {
            'total_stmts': count_statements(patch['diff']),
            'atom_stmts': [count_statements(patch['diff'])],
            'total_change_lines': sum(
                1 for l in patch['diff'].splitlines()
                if (l.startswith('+') and not l.startswith('+++'))
                or (l.startswith('-') and not l.startswith('---'))
            ),
            'unassigned_count': 0,
        },
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

STRATEGY_NAMES = [
    '2wayfile', '2wayfunction',
    'connected_file', 'connected_func',
    'all_file', 'all_func', 'all_direct',
]


def assign_series_to_strategies(
    all_series: List[Dict],
    ratios: List[float],
) -> Dict[str, List[Dict]]:
    """
     series
    ratios 7 STRATEGY_NAMES
    """
    total_r = sum(ratios)
    if total_r == 0:
        raise ValueError("strategy ratios 0")
    norm = [r / total_r for r in ratios]
    n = len(all_series)
    result: Dict[str, List[Dict]] = {}
    cursor = 0
    for i, (name, r) in enumerate(zip(STRATEGY_NAMES, norm)):
        if i < len(STRATEGY_NAMES) - 1:
            count = round(r * n)
        else:
            count = n - cursor #
        result[name] = all_series[cursor: cursor + count]
        cursor += count
    return result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _overlap_type_label(patches: List[Dict]) -> str:
    """ patch 'func' 'file'"""
    n = len(patches)
    for i in range(n):
        for j in range(i + 1, n):
            if patches_share_func(patches[i]['diff'], patches[j]['diff']):
                return 'func'
    return 'file'


def greedy_2way_cases(
    series_id: int,
    patches: List[Dict],
    used_ids: Set,
    overlap_fn,
) -> Tuple[List[Dict], Set]:
    """
     2-patch overlap_fn patches_share_file patches_share_func
     used_ids patch
     (new_cases, updated_used_ids)
    """
    n = len(patches)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = patches[i], patches[j]
            if pi['message_id'] in used_ids or pj['message_id'] in used_ids:
                continue
            if overlap_fn(pi['diff'], pj['diff']):
                shared_count = len(
                    get_file_paths(pi['diff']) & get_file_paths(pj['diff'])
                )
                overlap_type = 'func' if patches_share_func(pi['diff'], pj['diff']) else 'file'
                pairs.append((shared_count, i, j, overlap_type))

    pairs.sort(key=lambda x: -x[0])

    new_cases = []
    local_used: Set = set()

    for _, i, j, overlap_type in pairs:
        pi, pj = patches[i], patches[j]
        mid_i, mid_j = pi['message_id'], pj['message_id']
        if mid_i in used_ids or mid_j in used_ids:
            continue
        if mid_i in local_used or mid_j in local_used:
            continue
        case = make_tangled_case(series_id, [pi, pj], overlap_type)
        if case is not None:
            new_cases.append(case)
            local_used.add(mid_i)
            local_used.add(mid_j)

    used_ids.update(local_used)
    return new_cases, used_ids


def build_connected_cases(
    series_id: int,
    patches: List[Dict],
    overlap_fn,
) -> List[Dict]:
    """
     overlap_fn ≥2 patch case
     patch used_ids
     case patch patch_idx
    """
    n = len(patches)
    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if overlap_fn(patches[i]['diff'], patches[j]['diff']):
                union(i, j)

    components: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    cases = []
    for indices in components.values():
        if len(indices) < 2:
            continue
        atom_patches = sorted([patches[i] for i in indices], key=lambda p: p['patch_idx'])
        overlap_type = _overlap_type_label(atom_patches)
        case = make_tangled_case(series_id, atom_patches, overlap_type)
        if case is not None:
            cases.append(case)
    return cases


def build_all_cases(
    series_id: int,
    patches: List[Dict],
    trigger_fn,
) -> Optional[Dict]:
    """
    all_file / all_func trigger_fn series patch
    """
    n = len(patches)
    triggered = any(
        trigger_fn(patches[i]['diff'], patches[j]['diff'])
        for i in range(n)
        for j in range(i + 1, n)
    )
    if not triggered:
        return None
    atom_patches = sorted(patches, key=lambda p: p['patch_idx'])
    overlap_type = _overlap_type_label(atom_patches)
    return make_tangled_case(series_id, atom_patches, overlap_type)


def build_direct_case(
    series_id: int,
    patches: List[Dict],
) -> Optional[Dict]:
    """
    all_direct patch ≥ 2
    """
    if len(patches) < 2:
        return None
    atom_patches = sorted(patches, key=lambda p: p['patch_idx'])
    overlap_type = _overlap_type_label(atom_patches)
    return make_tangled_case(series_id, atom_patches, overlap_type)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _dispatch_strategy(
    strategy: str,
    series_id: int,
    patches: List[Dict],
    used_patch_ids: Set,
) -> Tuple[List[Dict], Set]:
    """
     (new_cases, updated_used_patch_ids)
     connected/all/direct patch used_patch_ids
     case strategy
    """
    cases: List[Dict] = []

    if strategy == '2wayfile':
        cases, used_patch_ids = greedy_2way_cases(series_id, patches, used_patch_ids, patches_share_file)

    elif strategy == '2wayfunction':
        cases, used_patch_ids = greedy_2way_cases(series_id, patches, used_patch_ids, patches_share_func)

    elif strategy == 'connected_file':
        cases = build_connected_cases(series_id, patches, patches_share_file)
        for case in cases:
            for atom in case['atoms']:
                used_patch_ids.add(atom['message_id'])

    elif strategy == 'connected_func':
        cases = build_connected_cases(series_id, patches, patches_share_func)
        for case in cases:
            for atom in case['atoms']:
                used_patch_ids.add(atom['message_id'])

    elif strategy == 'all_file':
        case = build_all_cases(series_id, patches, patches_share_file)
        if case:
            for p in patches:
                used_patch_ids.add(p['message_id'])
            cases = [case]

    elif strategy == 'all_func':
        case = build_all_cases(series_id, patches, patches_share_func)
        if case:
            for p in patches:
                used_patch_ids.add(p['message_id'])
            cases = [case]

    elif strategy == 'all_direct':
        case = build_direct_case(series_id, patches)
        if case:
            for p in patches:
                used_patch_ids.add(p['message_id'])
            cases = [case]

    for case in cases:
        case['strategy'] = strategy

    return cases, used_patch_ids


def build_dataset(
    strategy_ratios: Optional[List[float]] = None,
    limit: Optional[int] = None,
    series_limit: Optional[int] = None,
    min_code_lines: int = 5,
    neg_min_code_lines: int = 10,
    max_stmts: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
     tangled dataset
    strategy_ratios: 7 STRATEGY_NAMES series
     (positive_cases, negative_cases)
    """
    if strategy_ratios is None:
        strategy_ratios = [1.0, 0, 0, 0, 0, 0, 0]

    print("[build] multi-patch series...")
    all_series = fetch_multi_patch_series(limit=series_limit)
    print(f"[build] {len(all_series)} multi-patch series")

    # patch series
    single_patch_series_count = _count_single_patch_series()
    print(f"[build] patch series: {single_patch_series_count} ")

    strategy_map = assign_series_to_strategies(all_series, strategy_ratios)
    for name, slist in strategy_map.items():
        if slist:
            print(f"[build]   {name}: {len(slist)} series ({len(slist)/len(all_series)*100:.1f}%)")

    positive_cases: List[Dict] = []
    used_patch_ids: Set = set()
    all_single_patches: List[Dict] = [] #

    processed = 0
    skipped_no_diff = 0

    for strategy, series_list in strategy_map.items():
        for series in series_list:
            if limit and len(positive_cases) >= limit:
                break

            epoch = series['representative_epoch']
            eids = series['patch_email_ids']
            series_id = series['series_id']

            # patch diff
            patches = []
            for idx, eid in enumerate(eids):
                info = fetch_patch_info(eid, epoch)
                if info is None or not info['diff']:
                    skipped_no_diff += 1
                    continue
                if count_code_lines(info['diff']) < min_code_lines:
                    continue
                patches.append({
                    'message_id': eid,
                    'subject': info['subject'],
                    'diff': info['diff'],
                    'patch_idx': idx,
                    'series_id': series_id,
                })

            if len(patches) < 2:
                for p in patches:
                    all_single_patches.append(p)
                processed += 1
                continue

            new_cases, used_patch_ids = _dispatch_strategy(
                strategy, series_id, patches, used_patch_ids
            )
            if max_stmts is not None:
                filtered = [c for c in new_cases if c.get('stats', {}).get('total_stmts', 0) <= max_stmts]
                skipped = len(new_cases) - len(filtered)
                if skipped:
                    print(f"[build] series {series_id}: {skipped} total_stmts > {max_stmts} case")
                new_cases = filtered
            positive_cases.extend(new_cases)

            # patch
            for p in patches:
                if p['message_id'] not in used_patch_ids:
                    all_single_patches.append(p)

            processed += 1
            if processed % 1000 == 0:
                print(f"[build] {processed} series {len(positive_cases)} ")

    print(f"[build] {processed} series {len(positive_cases)} ")
    print(f"[build] diff patch: {skipped_no_diff}")

    neg_candidates = [
        p for p in all_single_patches
        if p['message_id'] not in used_patch_ids
        and count_code_lines(p['diff']) >= neg_min_code_lines
    ]
    seen_neg: Set = set()
    unique_neg = []
    for p in neg_candidates:
        if p['message_id'] not in seen_neg:
            seen_neg.add(p['message_id'])
            unique_neg.append(p)

    unique_neg.sort(key=lambda p: count_code_lines(p['diff']), reverse=True)
    neg_count = min(len(unique_neg), len(positive_cases))
    negative_cases = [make_negative_case(p) for p in unique_neg[:neg_count]]

    print(f"[build] {len(negative_cases)} {len(unique_neg)} ")
    return positive_cases, negative_cases


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def compute_entanglement_metrics(pos: List[Dict]) -> Dict:
    """
     tangled case atom
    """
    if not pos:
        return {}

    n = len(pos)

    has_file_overlap = 0 # atom
    has_func_overlap = 0 # atom
    all_atoms_share_file = 0 # atom
    cross_file_only = 0 #

    # case
    shared_file_counts = []
    shared_func_counts = []

    # intra-hunk tangled diff hunk atom
    intra_hunk_cases = 0

    for d in pos:
        atoms = d['atoms']
        num_atoms = d['num_atoms']
        diffs = [a['diff'] for a in atoms]

        # atom
        atom_files = [get_file_paths(diff) for diff in diffs]
        atom_funcs = [get_func_signatures(diff) for diff in diffs]

        # pair /
        any_file = False
        any_func = False
        all_file = True
        shared_files_union: Set[str] = set()
        shared_funcs_union: Set[str] = set()

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                sf = atom_files[i] & atom_files[j]
                sfn = atom_funcs[i] & atom_funcs[j]
                if sf:
                    any_file = True
                    shared_files_union |= sf
                else:
                    all_file = False
                if sfn:
                    any_func = True
                    shared_funcs_union |= sfn

        if any_file:
            has_file_overlap += 1
        if any_func:
            has_func_overlap += 1
        if all_file and num_atoms > 1:
            all_atoms_share_file += 1
        if any_file and not any_func:
            cross_file_only += 1

        shared_file_counts.append(len(shared_files_union))
        shared_func_counts.append(len(shared_funcs_union))

        # intra-hunk tangled diff hunk atom
        assignments = d.get('ground_truth', {}).get('assignments', [])
        if assignments:
            # hunk hunk header
            # assignments add/del hunk tangled_diff
            tangled = d.get('tangled_diff', '')
            case_has_intra = False
            current_hunk_groups: Set = set()
            assign_iter = iter(assignments)
            cur_assign = None
            try:
                cur_assign = next(assign_iter)
            except StopIteration:
                cur_assign = None

            for line in tangled.splitlines():
                if line.startswith('@@'):
                    if len(current_hunk_groups) > 1:
                        case_has_intra = True
                        break
                    current_hunk_groups = set()
                elif (line.startswith('+') and not line.startswith('+++')) or \
                     (line.startswith('-') and not line.startswith('---')):
                    if cur_assign is not None:
                        gid = cur_assign.get('group_id')
                        if gid is not None:
                            current_hunk_groups.add(gid)
                        try:
                            cur_assign = next(assign_iter)
                        except StopIteration:
                            cur_assign = None
            if len(current_hunk_groups) > 1:
                case_has_intra = True
            if case_has_intra:
                intra_hunk_cases += 1

    import statistics as _s

    def _pct(x, total):
        return f"{x/total*100:.1f}%" if total else "0.0%"

    def _stats(vals):
        if not vals:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0}
        return {
            'mean': round(_s.mean(vals), 2),
            'median': round(_s.median(vals), 2),
            'min': min(vals),
            'max': max(vals),
        }

    return {
        'has_file_overlap': has_file_overlap,
        'has_file_overlap_ratio': _pct(has_file_overlap, n),
        'has_func_overlap': has_func_overlap,
        'has_func_overlap_ratio': _pct(has_func_overlap, n),
        'all_atoms_share_file': all_atoms_share_file,
        'all_atoms_share_file_ratio': _pct(all_atoms_share_file, n),
        'cross_file_only': cross_file_only,
        'cross_file_only_ratio': _pct(cross_file_only, n),
        'intra_hunk_split': intra_hunk_cases,
        'intra_hunk_split_ratio': _pct(intra_hunk_cases, n),
        'shared_file_count_stats': _stats(shared_file_counts),
        'shared_func_count_stats': _stats(shared_func_counts),
    }


def collect_dataset_stats(dataset: List[Dict]) -> Dict:
    """
     dict
    """
    import statistics as stats_mod
    from collections import Counter

    total = len(dataset)
    pos = [d for d in dataset if d['label'] == 'tangled']
    neg = [d for d in dataset if d['label'] == 'no_split']

    stats: Dict = {
        'total': total,
        'positive': len(pos),
        'negative': len(neg),
    }

    if pos:
        atom_dist = dict(sorted(Counter(d['num_atoms'] for d in pos).items()))
        ov_dist = dict(Counter(d['overlap_type'] for d in pos).most_common())
        strategy_dist = dict(sorted(Counter(d.get('strategy', 'unknown') for d in pos).items()))
        stmts = [d['stats']['total_stmts'] for d in pos]
        change_lines = [d['stats']['total_change_lines'] for d in pos]
        gt_perfect = sum(1 for d in pos if d.get('ground_truth') and d['ground_truth']['quality']['perfect'])
        unassigned = [d['ground_truth']['quality']['unassigned_ratio']
                      for d in pos if d.get('ground_truth')]

        stats['positive_stats'] = {
            'strategy_distribution': strategy_dist,
            'atom_count_distribution': atom_dist,
            'overlap_type_distribution': ov_dist,
            'stmt_count': {
                'mean': round(stats_mod.mean(stmts), 2),
                'median': round(stats_mod.median(stmts), 2),
                'min': min(stmts),
                'max': max(stmts),
            },
            'change_lines': {
                'mean': round(stats_mod.mean(change_lines), 2),
                'median': round(stats_mod.median(change_lines), 2),
                'min': min(change_lines),
                'max': max(change_lines),
            },
            'ground_truth_perfect': gt_perfect,
            'ground_truth_perfect_ratio': f"{gt_perfect/len(pos)*100:.1f}%",
            'unassigned_ratio': {
                'mean': round(stats_mod.mean(unassigned), 4) if unassigned else 0,
                'median': round(stats_mod.median(unassigned), 4) if unassigned else 0,
            },
            'entanglement': compute_entanglement_metrics(pos),
        }

    if neg:
        stmts_neg = [d['stats']['total_stmts'] for d in neg]
        neg_series = {d['series_id'] for d in neg}
        stats['negative_stats'] = {
            'stmt_count': {
                'mean': round(stats_mod.mean(stmts_neg), 2),
                'median': round(stats_mod.median(stmts_neg), 2),
                'min': min(stmts_neg),
                'max': max(stmts_neg),
            },
            'source_series_count': len(neg_series),
            'avg_per_series': round(len(neg) / len(neg_series), 2) if neg_series else 0,
        }

    return stats


def format_stats(stats: Dict, title: str = '') -> str:
    """ collect_dataset_stats """
    lines = []
    sep = '=' * 60
    lines.append(sep)
    if title:
        lines.append(f"  {title}")
    lines.append(f" Tangled Dataset ( {stats['total']} )")
    lines.append(sep)

    lines.append(f"\n")
    total = stats['total']
    pos_n = stats['positive']
    neg_n = stats['negative']
    lines.append(f" (tangled): {pos_n} ({pos_n/total*100:.1f}%)")
    lines.append(f" (no_split): {neg_n} ({neg_n/total*100:.1f}%)")

    ps = stats.get('positive_stats')
    if ps:
        lines.append(f"\n - ")
        for k, v in ps['strategy_distribution'].items():
            lines.append(f"  {k}: {v} ({v/pos_n*100:.1f}%)")

        lines.append(f"\n - Atom ")
        for k, v in ps['atom_count_distribution'].items():
            lines.append(f"  {k} atoms: {v} ({v/pos_n*100:.1f}%)")

        lines.append(f"\n - ")
        for k, v in ps['overlap_type_distribution'].items():
            lines.append(f"  {k}: {v} ({v/pos_n*100:.1f}%)")

        sc = ps['stmt_count']
        lines.append(f"\n - ")
        lines.append(f"  mean={sc['mean']}, median={sc['median']}, min={sc['min']}, max={sc['max']}")

        cl = ps['change_lines']
        lines.append(f"\n - ")
        lines.append(f"  mean={cl['mean']}, median={cl['median']}, min={cl['min']}, max={cl['max']}")

        lines.append(f"\nGround Truth ")
        lines.append(f" : {ps['ground_truth_perfect']} ({ps['ground_truth_perfect_ratio']})")
        ur = ps['unassigned_ratio']
        lines.append(f" : mean={ur['mean']}, median={ur['median']}")

        em = ps.get('entanglement', {})
        if em:
            lines.append(f"\n")
            lines.append(f" case: {em['has_file_overlap']} ({em['has_file_overlap_ratio']})")
            lines.append(f" case: {em['has_func_overlap']} ({em['has_func_overlap_ratio']})")
            lines.append(f" atom : {em['all_atoms_share_file']} ({em['all_atoms_share_file_ratio']})")
            lines.append(f" :{em['cross_file_only']} ({em['cross_file_only_ratio']})")
            lines.append(f" hunk atom : {em['intra_hunk_split']} ({em['intra_hunk_split_ratio']})")
            sf = em['shared_file_count_stats']
            lines.append(f" : mean={sf['mean']}, median={sf['median']}, min={sf['min']}, max={sf['max']}")
            sfn = em['shared_func_count_stats']
            lines.append(f" : mean={sfn['mean']}, median={sfn['median']}, min={sfn['min']}, max={sfn['max']}")

    ns = stats.get('negative_stats')
    if ns:
        sc = ns['stmt_count']
        lines.append(f"\n - ")
        lines.append(f"  mean={sc['mean']}, median={sc['median']}, min={sc['min']}, max={sc['max']}")
        lines.append(f"\n")
        lines.append(f" series : {ns['source_series_count']}")
        lines.append(f" series : {ns['avg_per_series']}")

    lines.append(f"\n{sep}")
    return '\n'.join(lines)


def analyze_dataset(dataset: List[Dict], title: str = '',
                    out_path: Optional[str] = None) -> None:
    """ out_path """
    stats = collect_dataset_stats(dataset)
    text = format_stats(stats, title)
    print(text)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as _f:
            _f.write(text + '\n')


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_dataset(
    dataset: List[Dict],
    min_stmts: Optional[int] = None,
    max_stmts: Optional[int] = None,
    min_atoms: Optional[int] = None,
    max_atoms: Optional[int] = None,
) -> List[Dict]:
    """
    - min_stmts / max_stmts: tangled diff
    - min_atoms / max_atoms: atom patch
    """
    filtered = []
    for d in dataset:
        stmts = d['stats']['total_stmts']
        if min_stmts is not None and stmts < min_stmts:
            continue
        if max_stmts is not None and stmts > max_stmts:
            continue
        if d['label'] == 'tangled':
            n_atoms = d['num_atoms']
            if min_atoms is not None and n_atoms < min_atoms:
                continue
            if max_atoms is not None and n_atoms > max_atoms:
                continue
        filtered.append(d)
    return filtered


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="// tangled commit ")
    parser.add_argument(
        '--mode', choices=['build', 'analyze', 'filter'], default='build',
        help=': build=, analyze=, filter='
    )
    parser.add_argument('--input', type=str, default='dataset_tangled.json',
                        help='analyze/filter ')
    parser.add_argument('--out', type=str, default='dataset_tangled.json',
                        help='')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='analyze <input_stem>_stats.txt')

    # build
    parser.add_argument('--series-limit', type=int, default=None,
                        help=' series ')
    parser.add_argument('--case-limit', type=int, default=None,
                        help='')
    parser.add_argument('--min-code-lines', type=int, default=5,
                        help='atom patch patch')
    parser.add_argument('--neg-min-code-lines', type=int, default=10,
                        help='')
    parser.add_argument('--max-build-stmts', type=int, default=None,
                        help='build total_stmts ')
    parser.add_argument(
        '--strategy', nargs=7, type=float,
        default=[1.0, 0, 0, 0, 0, 0, 0],
        metavar=('2WAYFILE', '2WAYFUNC', 'CONN_FILE', 'CONN_FUNC',
                 'ALL_FILE', 'ALL_FUNC', 'ALL_DIRECT'),
        help=(
            ' series \n'
            '  2wayfile 2wayfunction connected_file connected_func '
            'all_file all_func all_direct\n'
            ': --strategy 0.3 0.1 0.1 0.1 0.2 0.1 0.1'
        )
    )

    # filter
    parser.add_argument('--min-stmts', type=int, default=None,
                        help='')
    parser.add_argument('--max-stmts', type=int, default=None,
                        help='')
    parser.add_argument('--min-atoms', type=int, default=None,
                        help=' atom patch ')
    parser.add_argument('--max-atoms', type=int, default=None,
                        help=' atom patch ')

    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
        if not os.path.exists(out_path.parent):
            os.makedirs(out_path.parent)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        
    if args.mode == 'build':
        ratios = args.strategy
        ratio_str = ' '.join(f'{STRATEGY_NAMES[i]}={ratios[i]:.2f}' for i in range(7))
        print(f"[main] : build : {ratio_str}")
        pos_cases, neg_cases = build_dataset(
            strategy_ratios=ratios,
            limit=args.case_limit,
            series_limit=args.series_limit,
            min_code_lines=args.min_code_lines,
            neg_min_code_lines=args.neg_min_code_lines,
            max_stmts=args.max_build_stmts,
        )
        dataset = pos_cases + neg_cases
        out_path = Path(args.out)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"[main] {out_path} {len(dataset)} "
              f" {len(pos_cases)} {len(neg_cases)}")
        analyze_dataset(dataset, title=f'build ({ratio_str})',
                        out_path=Path(args.out_dir) / (Path(args.out).stem + '_stats.txt')
                        if args.out_dir else None)

    elif args.mode == 'analyze':
        print(f"[main] : analyze {args.input}")
        import ijson
        dataset = []
        with open(args.input, 'rb') as f:
            for d in ijson.items(f, 'item', use_float=True):
                dataset.append(d)
        out_path = (Path(args.out_dir) / (Path(args.input).stem + '_stats.txt')
                    if args.out_dir else None)
        analyze_dataset(dataset, title=Path(args.input).name, out_path=out_path)
        if out_path:
            print(f"[main] {out_path}")

    elif args.mode == 'filter':
        print(f"[main] : filter {args.input}")
        import ijson
        out_path = Path(args.out)
        before = 0
        after = 0

        with open(args.input, 'rb') as fin, \
             open(out_path, 'w', encoding='utf-8') as fout:
            fout.write('[\n')
            first_item = True
            for d in ijson.items(fin, 'item', use_float=True):
                before += 1
                stmts = d['stats']['total_stmts']
                if args.min_stmts is not None and stmts < args.min_stmts:
                    continue
                if args.max_stmts is not None and stmts > args.max_stmts:
                    continue
                if d['label'] == 'tangled':
                    n_atoms = d['num_atoms']
                    if args.min_atoms is not None and n_atoms < args.min_atoms:
                        continue
                    if args.max_atoms is not None and n_atoms > args.max_atoms:
                        continue
                if not first_item:
                    fout.write(',\n')
                fout.write(json.dumps(d, ensure_ascii=False, indent=2))
                first_item = False
                after += 1
                if before % 10000 == 0:
                    print(f"[filter] {before} {after} ...")
            fout.write('\n]')

        print(f"[main] {before} → {after} ( {before - after} )")
        print(f"[main] {out_path}")
        if args.out_dir:
            with open(args.input, 'rb') as fin2:
                filtered_data = list(ijson.items(fin2, 'item', use_float=True))
            analyze_dataset(filtered_data, title=f'filter → {out_path.name}',
                            out_path=Path(args.out_dir) / (out_path.stem + '_stats.txt'))


if __name__ == '__main__':
    main()
