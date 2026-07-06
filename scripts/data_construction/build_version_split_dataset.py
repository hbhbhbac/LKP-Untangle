"""
build_version_split_dataset.py commit untangle

   lkml_series_global.sqlite previous_version_series_id_*
   vN → vN+1 series patch

Pipeline：
  1. series DB series
  2. patch
  3. vN vN+1 patchessubject +
  4. sratio / tratio
  5. ground truth export_dataset.py
  6. + dataset_version_split.json

  python build_version_split_dataset.py [--out dataset_version_split.json]
      [--sratio 0.65] [--tratio 0.15] [--subset-ratio 0.40]
      [--min-src-lines 10] [--match-methods body thread subject]
      [--limit N] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE / "annotation_tool" / "backend"
_CODE_DIR = _HERE / "code"

sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_CODE_DIR))

from core import STRUCTURE_DIR
from processors.sqlite.db_utils import get_db_path
from relation_builder.series_builder import get_global_series_db_path
from diff_grouper import extract_diff, extract_files, build_lkml_url, classify_line
from stats_coverage import extract_code_lines

# export_dataset.py ground truth
from export_dataset import (
    build_ground_truth,
    judge_case as _judge_case,
    LineAssignment,
    FileAssignment,
    JudgerResult,
    Verdict,
    _split_diff_by_file,
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# →
# →
MATCH_METHODS = {
    "rep_body":     "previous_version_series_id_rep_body_match",
    "body":         "previous_version_series_id_body_match",
    "thread":       "previous_version_series_id_thread_match",
    "subject_fuzz": "previous_version_series_id_subject_fuzz_match",
    "subject":      "previous_version_series_id_subject_match",
}

DEFAULT_SRATIO = 0.65 # source targets
DEFAULT_TRATIO = 0.15 # target source
DEFAULT_SUBSET_RATIO = 0.40 # target source
DEFAULT_MIN_SRC_LINES = 10 # source patch
DEFAULT_MAX_TARGET_OVERLAP = 0.40 # target patch


# ---------------------------------------------------------------------------
# series DB
# ---------------------------------------------------------------------------

def _global_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_global_series_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def load_version_pairs(
    methods: List[str],
    limit: Optional[int] = None,
) -> List[Dict]:
    """
     lkml_series_global.sqlite series

    Returns:
        list of {
            curr_series_id, curr_epoch, curr_patch_ids, curr_version,
            prev_series_id, prev_epoch, prev_patch_ids, prev_version,
            match_method,
        }
    """
    conn = _global_conn()
    try:
        # series
        cursor = conn.execute("PRAGMA table_info(series)")
        existing_cols = {row[1] for row in cursor.fetchall()}
    finally:
        conn.close()

    available_methods = [m for m in methods if MATCH_METHODS[m] in existing_cols]
    if not available_methods:
        print(f"[WARN] series DB : {existing_cols}")
        return []

    seen_pairs: Set[Tuple[int, int]] = set()
    results: List[Dict] = []

    conn = _global_conn()
    try:
        for method in available_methods:
            col = MATCH_METHODS[method]
            sql = f"""
                SELECT
                    curr.id                  AS curr_series_id,
                    curr.representative_epoch AS curr_epoch,
                    curr.patch_email_ids      AS curr_patch_ids,
                    curr.patch_version        AS curr_version,
                    prev.id                  AS prev_series_id,
                    prev.representative_epoch AS prev_epoch,
                    prev.patch_email_ids      AS prev_patch_ids,
                    prev.patch_version        AS prev_version
                FROM series curr
                JOIN series prev ON curr.{col} = prev.id
                WHERE curr.{col} IS NOT NULL
                ORDER BY curr.id
                {f'LIMIT {limit}' if limit else ''}
            """
            rows = conn.execute(sql).fetchall()
            for row in rows:
                pair_key = (row["prev_series_id"], row["curr_series_id"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                curr_ids = json.loads(row["curr_patch_ids"] or "[]")
                prev_ids = json.loads(row["prev_patch_ids"] or "[]")

                # patch
                if len(curr_ids) <= len(prev_ids):
                    continue

                results.append({
                    "curr_series_id": row["curr_series_id"],
                    "curr_epoch":     row["curr_epoch"],
                    "curr_patch_ids": curr_ids,
                    "curr_version":   row["curr_version"],
                    "prev_series_id": row["prev_series_id"],
                    "prev_epoch":     row["prev_epoch"],
                    "prev_patch_ids": prev_ids,
                    "prev_version":   row["prev_version"],
                    "match_method":   method,
                })
    finally:
        conn.close()

    return results


# ---------------------------------------------------------------------------
# Epoch DB
# ---------------------------------------------------------------------------

def _get_available_epochs() -> List[int]:
    epochs = []
    for db_file in STRUCTURE_DIR.glob("lkml_epoch_*.sqlite"):
        m = re.search(r"epoch_(\d+)", db_file.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def _fetch_patches_for_series(
    series_id: int,
    email_ids: List[int],
    epoch: int,
    all_epochs: List[int],
) -> List[Dict]:
    """
     series patch
     representative_epoch epoch

    Returns:
        list of {email_id, patch_current, patch_total, subject, body, message_id, from_addr, date_raw}
         patch_current cover letter (patch_current=0)
    """
    if not email_ids:
        return []

    id_set = set(email_ids)
    patches: Dict[int, Dict] = {}  # email_id -> info

    # representative_epoch
    search_epochs = [epoch] + [e for e in all_epochs if e != epoch]

    for ep in search_epochs:
        if len(patches) == len(id_set):
            break
        db_path = get_db_path(ep)
        if not db_path.exists():
            continue
        remaining = id_set - set(patches.keys())
        placeholders = ",".join("?" * len(remaining))
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"""
                SELECT id, patch_current, patch_total, subject, body,
                       message_id, from_addr, date_raw
                FROM emails
                WHERE id IN ({placeholders})
                  AND is_patch = 1
                """,
                list(remaining),
            ).fetchall()
            for row in rows:
                patches[row["id"]] = dict(row)
        finally:
            conn.close()

    # cover letter patch_current
    result = [
        p for p in patches.values()
        if p.get("patch_current") and p["patch_current"] > 0
    ]
    result.sort(key=lambda x: x.get("patch_current") or 999)
    return result


# ---------------------------------------------------------------------------
# Subject patch
# ---------------------------------------------------------------------------

_PATCH_PREFIX_RE = re.compile(
    r"^\[(?:PATCH|RFC)[^\]]*\]\s*",
    re.IGNORECASE,
)


def _base_subject(subject: str) -> str:
    """ [PATCH vN k/m] base subject"""
    s = _PATCH_PREFIX_RE.sub("", subject or "").strip()
    return s.lower()


def _subject_similarity(s1: str, s2: str) -> float:
    """ base subject 0~1"""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return SequenceMatcher(None, s1, s2).ratio()


# ---------------------------------------------------------------------------
# Patch vN patch vN+1 patch
# ---------------------------------------------------------------------------

def align_patches(
    prev_patches: List[Dict],
    curr_patches: List[Dict],
    all_epochs: List[int],
    subject_threshold: float = 0.75,
) -> List[Dict]:
    """
     vN (prev) patch vN+1 (curr) patches
     1:N patch

      1. prev patch curr patches subject >= threshold
      2. >= 2
      3. == 0 curr patches

    Returns:
        list of {
            src_patch: Dict, # vN patch
            tgt_patches: List[Dict], # vN+1 patches>=2
            align_method: str,        # "subject" / "code"
        }
    """
    # patch
    def _get_code_lines(patch: Dict) -> Set[str]:
        diff = extract_diff(patch.get("body", ""))
        return extract_code_lines(diff)

    prev_code: List[Set[str]] = [_get_code_lines(p) for p in prev_patches]
    curr_code: List[Set[str]] = [_get_code_lines(p) for p in curr_patches]

    # curr patch prev patch
    curr_assigned: Dict[int, int] = {}  # curr_idx -> prev_idx

    candidates: List[Dict] = []

    for pi, prev_p in enumerate(prev_patches):
        prev_base = _base_subject(prev_p.get("subject", ""))
        prev_lines = prev_code[pi]

        # Step 1: subject
        subject_matches: List[Tuple[int, float]] = []  # (curr_idx, similarity)
        for ci, curr_p in enumerate(curr_patches):
            curr_base = _base_subject(curr_p.get("subject", ""))
            sim = _subject_similarity(prev_base, curr_base)
            if sim >= subject_threshold:
                subject_matches.append((ci, sim))

        subject_matches.sort(key=lambda x: x[1], reverse=True)

        if len(subject_matches) >= 2:
            tgt_indices = [ci for ci, _ in subject_matches]
            tgt_patches = [curr_patches[ci] for ci in tgt_indices]
            candidates.append({
                "src_patch": prev_p,
                "src_code": prev_lines,
                "tgt_patches": tgt_patches,
                "tgt_code": [curr_code[ci] for ci in tgt_indices],
                "align_method": "subject",
            })
            for ci in tgt_indices:
                curr_assigned[ci] = pi
            continue

        # Step 2: subject
        if not prev_lines:
            continue

        overlap_scores: List[Tuple[int, float]] = []
        for ci, curr_lines in enumerate(curr_code):
            if not curr_lines:
                continue
            overlap = len(prev_lines & curr_lines) / len(prev_lines)
            if overlap > 0.05: # 5%
                overlap_scores.append((ci, overlap))

        overlap_scores.sort(key=lambda x: x[1], reverse=True)

        # > 0.1 prev patch curr patches
        code_matches = [
            (ci, sc) for ci, sc in overlap_scores
            if sc > 0.10
        ]

        if len(code_matches) >= 2:
            tgt_indices = [ci for ci, _ in code_matches]
            tgt_patches = [curr_patches[ci] for ci in tgt_indices]
            candidates.append({
                "src_patch": prev_p,
                "src_code": prev_lines,
                "tgt_patches": tgt_patches,
                "tgt_code": [curr_code[ci] for ci in tgt_indices],
                "align_method": "code",
            })

    return candidates


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def verify_split_by_code(
    candidate: Dict,
    sratio_min: float = DEFAULT_SRATIO,
    tratio_min: float = DEFAULT_TRATIO,
    subset_ratio_min: float = DEFAULT_SUBSET_RATIO,
    min_src_lines: int = DEFAULT_MIN_SRC_LINES,
    max_target_overlap: float = DEFAULT_MAX_TARGET_OVERLAP,
) -> Optional[Dict]:
    """
     None

      1. source
      2. sratio: source targets >= sratio_min
      3. target tratio source >= tratio_min
      4. target subset_ratio source >= subset_ratio_min
      5. target max_target_overlap patch

    Returns:
        {
            src_patch, tgt_patches,
            sratio, per_tratio, per_subset_ratio,
            align_method,
        } None
    """
    src_lines: Set[str] = candidate["src_code"]
    tgt_lines_list: List[Set[str]] = candidate["tgt_code"]

    # 1source
    if len(src_lines) < min_src_lines:
        return None

    # target
    valid_tgts = [
        (p, t) for p, t in zip(candidate["tgt_patches"], tgt_lines_list)
        if len(t) > 0
    ]
    if len(valid_tgts) < 2:
        return None

    tgt_patches_valid = [p for p, _ in valid_tgts]
    tgt_lines_valid = [t for _, t in valid_tgts]

    # 2sratio
    union_tgt = set().union(*tgt_lines_valid)
    sratio = len(src_lines & union_tgt) / len(src_lines)
    if sratio < sratio_min:
        return None

    # 3 & 4per-target tratio subset_ratio
    per_tratio: List[float] = []
    per_subset_ratio: List[float] = []
    for tgt_lines in tgt_lines_valid:
        intersection = src_lines & tgt_lines
        tratio = len(intersection) / len(src_lines)
        subset_ratio = len(intersection) / len(tgt_lines)
        per_tratio.append(round(tratio, 4))
        per_subset_ratio.append(round(subset_ratio, 4))

    if min(per_tratio) < tratio_min:
        return None
    if min(per_subset_ratio) < subset_ratio_min:
        return None

    # 5target
    for i in range(len(tgt_lines_valid)):
        for j in range(i + 1, len(tgt_lines_valid)):
            ti, tj = tgt_lines_valid[i], tgt_lines_valid[j]
            if not ti or not tj:
                continue
            overlap = len(ti & tj) / min(len(ti), len(tj))
            if overlap > max_target_overlap:
                return None

    return {
        "src_patch":        candidate["src_patch"],
        "tgt_patches":      tgt_patches_valid,
        "sratio":           round(sratio, 4),
        "per_tratio":       per_tratio,
        "per_subset_ratio": per_subset_ratio,
        "align_method":     candidate["align_method"],
    }


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------

def _get_diff_and_meta(patch: Dict) -> Dict:
    """ patch diff """
    body = patch.get("body", "")
    diff = extract_diff(body)
    files = extract_files(body)
    mid = patch.get("message_id", "")
    return {
        "diff":       diff,
        "message_id": mid,
        "lore_url":   build_lkml_url(mid) if mid else None,
        "subject":    patch.get("subject", ""),
        "from_addr":  patch.get("from_addr", ""),
        "date_raw":   patch.get("date_raw", ""),
        "files":      files or [],
        "patch_num":  patch.get("patch_current"),
        "patch_total": patch.get("patch_total"),
    }


def build_case(
    pair: Dict,
    verified: Dict,
    all_epochs: List[int],
) -> Optional[Dict]:
    """
     case
     export_dataset.py dataset.json
    """
    src_patch = verified["src_patch"]
    tgt_patches = verified["tgt_patches"]

    source_meta = _get_diff_and_meta(src_patch)
    target_metas = [_get_diff_and_meta(p) for p in tgt_patches]

    source_diff = source_meta["diff"]
    target_diffs = [m["diff"] for m in target_metas]

    if not source_diff:
        return None
    if all(not d for d in target_diffs):
        return None

    # ground truth
    file_assignments, groups, unassigned_count = build_ground_truth(
        source_diff, target_diffs, target_metas
    )

    # not_in_ratiostarget source
    from export_dataset import _split_diff_by_file
    src_lines_set: Set[str] = set()
    for line in source_diff.splitlines():
        t = classify_line(line)
        if t in ("add", "del"):
            s = line[1:].strip()
            if s:
                src_lines_set.add(s)

    not_in_ratios: List[float] = []
    for tgt_diff in target_diffs:
        total, not_in = 0, 0
        for line in tgt_diff.splitlines():
            t = classify_line(line)
            if t in ("add", "del"):
                s = line[1:].strip()
                if s:
                    total += 1
                    if s not in src_lines_set:
                        not_in += 1
        not_in_ratios.append(round(not_in / total, 4) if total > 0 else 0.0)

    # similaritysource target
    # sratio
    similarity = verified["sratio"]

    # Judger export_dataset.py
    # judge_case case dict
    fake_case = {
        "label_source": "version_split",
        "not_in_ratios_json": json.dumps(not_in_ratios),
        "similarity": similarity,
        "split_emails_json": "[]",
    }
    judger = _judge_case(
        fake_case, source_meta, target_metas, file_assignments, unassigned_count
    )

    if judger.verdict == Verdict.INVALID:
        return None

    # file_assignments
    fa_list = []
    for fa in file_assignments:
        fa_list.append({
            "file":      fa.file,
            "split":     fa.split,
            "group_ids": fa.group_ids,
            "lines": [
                {
                    "content":     la.content,
                    "type":        la.line_type,
                    "group_id":    la.group_id,
                    "match_score": la.match_score,
                }
                for la in fa.lines
            ],
        })

    return {
        "label_source": "version_split",
        "split_type":   "VERSION_ITERATION",
        "version_info": {
            "source_version": pair["prev_version"],
            "target_version": pair["curr_version"],
            "match_method":   pair["match_method"],
            "align_method":   verified["align_method"],
        },

        # Source patch（vN）
        "source": {
            "series_id":   pair["prev_series_id"],
            "version":     pair["prev_version"],
            "patch_num":   source_meta["patch_num"],
            "patch_total": source_meta["patch_total"],
            "subject":     source_meta["subject"],
            "from_addr":   source_meta["from_addr"],
            "date_raw":    source_meta["date_raw"],
            "message_id":  source_meta["message_id"],
            "lore_url":    source_meta["lore_url"],
            "files":       source_meta["files"],
            "diff":        source_diff,
        },

        # Target patches（vN+1）
        "target": {
            "series_id": pair["curr_series_id"],
            "version":   pair["curr_version"],
            "patches": [
                {
                    "group_id":    idx,
                    "patch_num":   m["patch_num"],
                    "patch_total": m["patch_total"],
                    "subject":     m["subject"],
                    "from_addr":   m["from_addr"],
                    "date_raw":    m["date_raw"],
                    "message_id":  m["message_id"],
                    "lore_url":    m["lore_url"],
                    "files":       m["files"],
                    "diff":        m["diff"],
                }
                for idx, m in enumerate(target_metas)
            ],
        },

        # Ground truth
        "ground_truth": {
            "num_groups":       len(groups),
            "file_assignments": fa_list,
            "groups":           groups,
        },

        "split_evidence": {
            "emails":       [],
            "similarity":   similarity,
            "not_in_ratios": not_in_ratios,
            "sratio":       verified["sratio"],
            "per_tratio":   verified["per_tratio"],
            "per_subset_ratio": verified["per_subset_ratio"],
        },

        # Judger
        "judger": {
            "verdict":            judger.verdict.value,
            "confidence":         judger.confidence,
            "issues":             judger.issues,
            "modifications":      judger.modifications,
            "path_rename":        judger.path_rename,
            "has_file_split":     judger.has_file_split,
            "unassigned_line_count": judger.unassigned_line_count,
        },
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def deduplicate(cases: List[Dict]) -> List[Dict]:
    """
     (prev_series_id, src_patch_num, curr_series_id)
     match_method body > thread > subject
    """
    method_priority = {"rep_body": 0, "body": 1, "thread": 2, "subject_fuzz": 3, "subject": 4}
    seen: Dict[Tuple, Dict] = {}

    for case in cases:
        key = (
            case["source"]["series_id"],
            case["source"]["patch_num"],
            case["target"]["series_id"],
        )
        method = case["version_info"]["match_method"]
        priority = method_priority.get(method, 99)

        if key not in seen:
            seen[key] = (priority, case)
        else:
            existing_priority, _ = seen[key]
            if priority < existing_priority:
                seen[key] = (priority, case)

    return [case for _, case in seen.values()]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=" commit untangle ")
    parser.add_argument("--out", default="dataset_version_split.json",
                        help=" dataset_version_split.json")
    parser.add_argument("--sratio", type=float, default=DEFAULT_SRATIO,
                        help=f"source {DEFAULT_SRATIO}")
    parser.add_argument("--tratio", type=float, default=DEFAULT_TRATIO,
                        help=f" target source {DEFAULT_TRATIO}")
    parser.add_argument("--subset-ratio", type=float, default=DEFAULT_SUBSET_RATIO,
                        dest="subset_ratio",
                        help=f" target source {DEFAULT_SUBSET_RATIO}")
    parser.add_argument("--min-src-lines", type=int, default=DEFAULT_MIN_SRC_LINES,
                        dest="min_src_lines",
                        help=f"source patch {DEFAULT_MIN_SRC_LINES}")
    parser.add_argument("--match-methods", nargs="+",
                        default=["rep_body", "body", "thread", "subject_fuzz", "subject"],
                        choices=["rep_body", "body", "thread", "subject_fuzz", "subject"],
                        dest="match_methods",
                        help="")
    parser.add_argument("--limit", type=int, default=None,
                        help=" series ")
    parser.add_argument("--verbose", action="store_true",
                        help=" case ")
    args = parser.parse_args()

    all_epochs = _get_available_epochs()
    print(f"[init] epoch: {all_epochs}")

    # Step 1:
    print(f"\n[step1] series DB : {args.match_methods}...")
    pairs = load_version_pairs(args.match_methods, limit=args.limit)
    print(f"[step1] {len(pairs)} patch ")

    if not pairs:
        print("[ERROR] lkml_series_global.sqlite ")
        return

    # Step 2-4: patch +
    print(f"\n[step2-4] patches ...")
    print(f" : sratio>={args.sratio}, tratio>={args.tratio}, "
          f"subset_ratio>={args.subset_ratio}, min_src_lines>={args.min_src_lines}")

    all_cases: List[Dict] = []
    stats = {
        "total_pairs":       len(pairs),
        "pairs_with_candidates": 0,
        "candidates_total":  0,
        "candidates_passed": 0,
        "cases_built":       0,
        "cases_invalid":     0,
    }

    for i, pair in enumerate(pairs):
        if (i + 1) % 500 == 0:
            print(f" : {i+1}/{len(pairs)}, : {stats['candidates_passed']}")

        # prev curr patch
        prev_patches = _fetch_patches_for_series(
            pair["prev_series_id"],
            pair["prev_patch_ids"],
            pair["prev_epoch"],
            all_epochs,
        )
        curr_patches = _fetch_patches_for_series(
            pair["curr_series_id"],
            pair["curr_patch_ids"],
            pair["curr_epoch"],
            all_epochs,
        )

        if not prev_patches or not curr_patches:
            continue

        # Step 2: patch
        candidates = align_patches(prev_patches, curr_patches, all_epochs)
        if not candidates:
            continue

        stats["pairs_with_candidates"] += 1
        stats["candidates_total"] += len(candidates)

        # Step 3:
        for candidate in candidates:
            verified = verify_split_by_code(
                candidate,
                sratio_min=args.sratio,
                tratio_min=args.tratio,
                subset_ratio_min=args.subset_ratio,
                min_src_lines=args.min_src_lines,
            )
            if verified is None:
                continue

            stats["candidates_passed"] += 1

            # Step 4: case
            case = build_case(pair, verified, all_epochs)
            if case is None:
                stats["cases_invalid"] += 1
                continue

            stats["cases_built"] += 1
            all_cases.append(case)

            if args.verbose:
                src = case["source"]
                print(
                    f"  [PASS] prev={pair['prev_series_id']} "
                    f"patch={src['patch_num']}/{src['patch_total']} "
                    f"→ {len(case['target']['patches'])} targets  "
                    f"sratio={verified['sratio']:.2f}  "
                    f"method={pair['match_method']}/{verified['align_method']}  "
                    f"subject: {src['subject'][:60]}"
                )

    # Step 5:
    print(f"\n[step5] : {len(all_cases)} ")
    all_cases = deduplicate(all_cases)
    print(f"[step5] : {len(all_cases)} ")

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {out_path}{len(all_cases)} ")

    print("\n" + "=" * 55)
    print(" ")
    print("=" * 55)
    print(f" : {stats['total_pairs']:>8,}")
    print(f" : {stats['pairs_with_candidates']:>8,}")
    print(f" : {stats['candidates_total']:>8,}")
    print(f" : {stats['candidates_passed']:>8,}")
    print(f" case: {stats['cases_built']:>8,}")
    print(f" Judger case: {stats['cases_invalid']:>8,}")
    print(f" : {len(all_cases):>8,}")

    method_counts: Dict[str, int] = defaultdict(int)
    align_counts: Dict[str, int] = defaultdict(int)
    verdict_counts: Dict[str, int] = defaultdict(int)
    for case in all_cases:
        method_counts[case["version_info"]["match_method"]] += 1
        align_counts[case["version_info"]["align_method"]] += 1
        verdict_counts[case["judger"]["verdict"]] += 1

    print(f"\n :")
    for m, cnt in sorted(method_counts.items()):
        print(f"    {m}: {cnt}")
    print(f"\n patch :")
    for m, cnt in sorted(align_counts.items()):
        print(f"    {m}: {cnt}")
    print(f"\n Judger verdict :")
    for v, cnt in sorted(verdict_counts.items()):
        print(f"    {v}: {cnt}")
    print("=" * 55)


if __name__ == "__main__":
    main()
