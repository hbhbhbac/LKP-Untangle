"""
export_dataset.py

  1. "" YES + nc_tagged ~350
  2. epoch source/target patch diff
  3. hunk ground truth patch
  4. Judger case verdict
  5. dataset.json

  python export_dataset.py [--target 350] [--out dataset.json] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE / "annotation_tool" / "backend"
_CODE_DIR = _HERE / "code"

sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_CODE_DIR))

import db as db_module
from diff_grouper import (
    assign_diff_groups,
    extract_diff,
    extract_files,
    build_lkml_url,
    classify_line,
)
from stats_coverage import (
    extract_code_lines,
    compute_ratios,
    _has_re_subject,
    _has_re_matching_source,
    process_case as coverage_process_case,
)

ANNOTATIONS_DB = _BACKEND / "annotations.sqlite"


# ---------------------------------------------------------------------------
# Judger
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    VALID = "VALID"
    VALID_MODIFIED = "VALID_MODIFIED"
    SUSPICIOUS = "SUSPICIOUS"
    INVALID = "INVALID"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class LineAssignment:
    content: str # +/-
    line_type: str            # add / del / context / meta / hunk
    group_id: Optional[int] # None = unassigned add/del
    match_score: float = 0.0 # 0~1


@dataclass
class FileAssignment:
    file: str
    split: bool # group
    group_ids: List[int] # group_id
    lines: List[LineAssignment] = field(default_factory=list)


@dataclass
class JudgerResult:
    verdict: Verdict
    confidence: float
    issues: List[str] = field(default_factory=list)
    modifications: List[Dict] = field(default_factory=list)
    path_rename: bool = False
    has_file_split: bool = False
    unassigned_line_count: int = 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _ann_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(ANNOTATIONS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_positive_cases(target: int = 350) -> List[Dict]:
    """
      A. YESq1 IN YES_REVIEWER_REQUEST, YES_AUTHOR_SELF
      B. nc_tagged=1 AND status='pending'
      C. A+B < target pending
    """
    conn = _ann_conn()

    # A: YES
    yes_rows = conn.execute("""
        SELECT ac.*, hl.q1 AS label_q1, hl.q2 AS label_q2, hl.q3 AS label_q3,
               hl.notes AS label_notes
        FROM annotation_cases ac
        JOIN human_labels hl ON ac.id = hl.case_id
        WHERE hl.q1 IN ('YES_REVIEWER_REQUEST', 'YES_AUTHOR_SELF')
          AND hl.id = (SELECT MAX(id) FROM human_labels WHERE case_id = ac.id)
        ORDER BY ac.id
    """).fetchall()

    # B: nc_tagged=1 AND pending
    nc_pending_rows = conn.execute("""
        SELECT ac.*, NULL AS label_q1, NULL AS label_q2, NULL AS label_q3,
               NULL AS label_notes
        FROM annotation_cases ac
        WHERE ac.nc_tagged = 1 AND ac.status = 'pending'
        ORDER BY ac.id
    """).fetchall()

    conn.close()

    seen_ids: Set[int] = set()
    cases: List[Dict] = []

    def add_case(row, label_source: str):
        d = dict(row)
        d["label_source"] = label_source
        cases.append(d)
        seen_ids.add(d["id"])

    for row in yes_rows:
        add_case(row, "human_yes")

    for row in nc_pending_rows:
        if row["id"] not in seen_ids:
            add_case(row, "nc_auto")

    current = len(cases)
    print(f"[fetch] human_yes={sum(1 for c in cases if c['label_source']=='human_yes')}, "
          f"nc_auto={sum(1 for c in cases if c['label_source']=='nc_auto')}, "
          f"total={current}")

    # C: target pending
    # case
    if current < target:
        needed = 1500 # expansion pool
        cases += _fetch_expansion_cases(seen_ids, needed)
        print(f"[fetch] expanded by {len(cases) - current} cases, total={len(cases)}")

    return cases


def _fetch_expansion_cases(exclude_ids: Set[int], needed: int) -> List[Dict]:
    """
     pending
    0~1
      - similarity 0.4
      - max(not_in_ratios) 0.4 target
      - split_email keyword 0.2
    """
    conn = _ann_conn()
    rows = conn.execute("""
        SELECT ac.*, NULL AS label_q1, NULL AS label_q2, NULL AS label_q3,
               NULL AS label_notes
        FROM annotation_cases ac
        WHERE ac.status = 'pending'
          AND ac.nc_tagged = 0
        ORDER BY ac.similarity DESC
    """).fetchall()
    conn.close()

    scored: List[Tuple[float, Dict]] = []
    for row in rows:
        if row["id"] in exclude_ids:
            continue
        d = dict(row)

        sim = d.get("similarity") or 0.0
        not_in_ratios = json.loads(d.get("not_in_ratios_json") or "[]")
        max_not_in = max(not_in_ratios) if not_in_ratios else 0.0

        split_emails = json.loads(d.get("split_emails_json") or "[]")
        has_keyword = any(e.get("keyword", "").strip() for e in split_emails)
        keyword_score = 0.2 if has_keyword else 0.0

        score = 0.4 * sim + 0.4 * max_not_in + keyword_score
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = []
    for score, d in scored[:needed]:
        d["label_source"] = "expansion"
        d["expansion_score"] = round(score, 4)
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def get_patch_diff_and_meta(series_id: int, patch_num: int, epochs: List[int]) -> Dict:
    """ {diff, message_id, lore_url, subject, from_addr, date_raw, files}"""
    email = db_module._fetch_email_by_series_and_num(series_id, patch_num, epochs)
    if not email:
        return {"diff": "", "message_id": None, "lore_url": None,
                "subject": "", "from_addr": "", "date_raw": "", "files": []}
    body = email.get("body", "")
    diff = extract_diff(body)
    files = extract_files(body)
    mid = email.get("message_id", "")
    return {
        "diff": diff,
        "message_id": mid,
        "lore_url": build_lkml_url(mid) if mid else None,
        "subject": email.get("subject", ""),
        "from_addr": email.get("from_addr", ""),
        "date_raw": email.get("date_raw", ""),
        "files": files,
    }


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------

_FILE_SECTION_RE = re.compile(r"^diff\s+--git\s+a/(.+?)\s+b/(.+?)$", re.MULTILINE)


def _split_diff_by_file(diff: str) -> Dict[str, str]:
    """ unified diff {filepath: file_diff}"""
    result: Dict[str, str] = {}
    sections = _FILE_SECTION_RE.split(diff)
    # sections: [pre, a_path, b_path, content, a_path, b_path, content, ...]
    i = 1
    while i + 2 < len(sections):
        b_path = sections[i + 1].strip()
        content = "diff --git a/" + sections[i] + " b/" + sections[i + 1] + sections[i + 2]
        result[b_path] = content
        i += 3
    return result


def _resolve_target_file_diff(filepath: str, tgt_file_map: Dict[str, str]) -> str:
    """ target filepathbasename """
    if filepath in tgt_file_map:
        return tgt_file_map[filepath]
    basename = filepath.split("/")[-1]
    for tgt_path, tgt_fd in tgt_file_map.items():
        if tgt_path.split("/")[-1] == basename:
            return tgt_fd
    return ""


def build_ground_truth(
    source_diff: str,
    target_diffs: List[str],
    target_metas: List[Dict],
) -> Tuple[List[FileAssignment], List[Dict], int]:
    """
     ground truth

     source diff add/del +
     target patchgroup_id
    context/meta/hunk group_id None

    Returns:
        file_assignments:
        groups: target patch
        unassigned_count: add/del
    """
    src_by_file = _split_diff_by_file(source_diff)
    tgt_by_file: List[Dict[str, str]] = [_split_diff_by_file(td) for td in target_diffs]

    groups = [
        {
            "group_id": idx,
            "patch_num": m.get("patch_num"),
            "patch_total": m.get("patch_total"),
            "subject": m.get("subject", ""),
            "files": m.get("files", []),
        }
        for idx, m in enumerate(target_metas)
    ]

    file_assignments: List[FileAssignment] = []
    unassigned_count = 0

    for filepath, file_diff in src_by_file.items():
        # target add/del
        tgt_line_sets: List[Set[str]] = []
        tgt_line_lists: List[List[str]] = [] #
        for gid, tgt_file_map in enumerate(tgt_by_file):
            tgt_fd = _resolve_target_file_diff(filepath, tgt_file_map)
            lines_set: Set[str] = set()
            lines_list: List[str] = []
            for line in tgt_fd.splitlines():
                t = classify_line(line)
                if t in ("add", "del"):
                    s = line[1:].strip()
                    if s:
                        lines_set.add(s)
                        lines_list.append(s)
            tgt_line_sets.append(lines_set)
            tgt_line_lists.append(lines_list)

        line_assignments: List[LineAssignment] = []
        group_ids_for_file: Set[int] = set()

        # hunk group
        current_hunk_votes: Dict[int, int] = defaultdict(int)

        for raw_line in file_diff.splitlines():
            line_type = classify_line(raw_line)

            if line_type == "hunk":
                # hunk
                current_hunk_votes = defaultdict(int)
                line_assignments.append(LineAssignment(
                    content=raw_line, line_type=line_type,
                    group_id=None, match_score=0.0,
                ))
                continue

            if line_type not in ("add", "del"):
                line_assignments.append(LineAssignment(
                    content=raw_line, line_type=line_type,
                    group_id=None, match_score=0.0,
                ))
                continue

            stripped = raw_line[1:].strip()
            if not stripped:
                # group
                line_assignments.append(LineAssignment(
                    content=raw_line, line_type=line_type,
                    group_id=None, match_score=-1.0, # -1
                ))
                continue

            best_group: Optional[int] = None
            best_score = 0.0

            # 1. target
            exact_matches = [gid for gid, line_set in enumerate(tgt_line_sets)
                             if stripped in line_set]

            if len(exact_matches) == 1:
                best_group = exact_matches[0]
                best_score = 1.0
            elif len(exact_matches) > 1:
                # `{``}` target
                # hunk
                if current_hunk_votes:
                    best_in_hunk = max(
                        (g for g in exact_matches if g in current_hunk_votes),
                        key=lambda g: current_hunk_votes[g],
                        default=None,
                    )
                    best_group = best_in_hunk if best_in_hunk is not None else exact_matches[0]
                else:
                    best_group = exact_matches[0]
                best_score = 1.0

            # 2.
            if best_group is None:
                from difflib import SequenceMatcher
                for gid, lines_list in enumerate(tgt_line_lists):
                    for tl in lines_list:
                        score = SequenceMatcher(None, stripped, tl).ratio()
                        if score > best_score:
                            best_score = score
                            best_group = gid
                if best_score < 0.45:
                    best_group = None

            if best_group is None:
                unassigned_count += 1
                line_assignments.append(LineAssignment(
                    content=raw_line, line_type=line_type,
                    group_id=None, match_score=0.0,
                ))
            else:
                group_ids_for_file.add(best_group)
                current_hunk_votes[best_group] += 1
                line_assignments.append(LineAssignment(
                    content=raw_line, line_type=line_type,
                    group_id=best_group,
                    match_score=round(best_score, 3),
                ))

        # group match_score == -1
        last_group: Optional[int] = None
        for la in line_assignments:
            if la.line_type in ("add", "del") and la.match_score != -1.0 and la.group_id is not None:
                last_group = la.group_id
            elif la.line_type in ("add", "del") and la.match_score == -1.0:
                if last_group is not None:
                    la.group_id = last_group
                    la.match_score = 0.0 # 0.0
                    group_ids_for_file.add(last_group)

        next_group: Optional[int] = None
        for la in reversed(line_assignments):
            if la.line_type in ("add", "del") and la.match_score != -1.0 and la.group_id is not None:
                next_group = la.group_id
            elif la.line_type in ("add", "del") and la.match_score == -1.0:
                if next_group is not None:
                    la.group_id = next_group
                    la.match_score = 0.0
                    group_ids_for_file.add(next_group)
                else:
                    # null match_score
                    la.match_score = 0.0
                    unassigned_count += 1

        file_assignments.append(FileAssignment(
            file=filepath,
            split=len(group_ids_for_file) > 1,
            group_ids=sorted(group_ids_for_file),
            lines=line_assignments,
        ))

    return file_assignments, groups, unassigned_count


# ---------------------------------------------------------------------------
# Judger
# ---------------------------------------------------------------------------

def _detect_path_rename(source_files: List[str], target_files_all: List[str]) -> bool:
    """ staging/ → staging/"""
    src_basenames = {f.split("/")[-1] for f in source_files}
    tgt_basenames = {f.split("/")[-1] for f in target_files_all}
    common = src_basenames & tgt_basenames
    if not common:
        return False
    for sf in source_files:
        for tf in target_files_all:
            if sf.split("/")[-1] == tf.split("/")[-1] and sf != tf:
                return True
    return False


def judge_case(
    case: Dict,
    source_meta: Dict,
    target_metas_with_diff: List[Dict],
    file_assignments: List[FileAssignment],
    unassigned_count: int,
) -> JudgerResult:
    """ case JudgerResult"""
    issues: List[str] = []
    modifications: List[Dict] = []

    # YES
    label_source = case.get("label_source", "")
    confidence = 1.0 if label_source == "human_yes" else 0.85

    num_targets = len(target_metas_with_diff)
    not_in_ratios = json.loads(case.get("not_in_ratios_json") or "[]")
    similarity = case.get("similarity") or 0.0
    split_emails = json.loads(case.get("split_emails_json") or "[]")

    # 1target patch
    if num_targets < 2:
        return JudgerResult(
            verdict=Verdict.INVALID,
            confidence=0.0,
            issues=["target_count < 2: "],
        )

    # 2source diff
    if not source_meta.get("diff"):
        return JudgerResult(
            verdict=Verdict.INVALID,
            confidence=0.0,
            issues=["source_diff_empty: source patch diff"],
        )

    # 3target diff
    empty_targets = [i for i, m in enumerate(target_metas_with_diff) if not m.get("diff")]
    if len(empty_targets) == num_targets:
        return JudgerResult(
            verdict=Verdict.INVALID,
            confidence=0.0,
            issues=["all_target_diffs_empty: target patch diff "],
        )
    if empty_targets:
        issues.append(f"missing_target_diffs: target patch {empty_targets} diff ")
        confidence -= 0.15 * len(empty_targets)
        modifications.append({
            "type": "drop_empty_targets",
            "detail": f"Target patch indices {empty_targets} diff ground truth ",
            "affected_indices": empty_targets,
        })

    # 4not_in_ratios 0
    if not_in_ratios and all(r < 0.05 for r in not_in_ratios):
        issues.append("low_not_in_ratios: target patch source ")
        confidence -= 0.2

    # 5 target not_in_ratio 1 target patch
    high_not_in = [(i, r) for i, r in enumerate(not_in_ratios) if r > 0.9]
    if high_not_in:
        for idx, r in high_not_in:
            issues.append(f"high_not_in_ratio: target[{idx}] not_in_ratio={r:.2f} patch")
            confidence -= 0.1

    # 6unassigned lines
    if unassigned_count > 0:
        total_change_lines = sum(
            1 for fa in file_assignments
            for la in fa.lines
            if la.line_type in ("add", "del") and la.content.strip()[1:].strip()
        )
        ratio = unassigned_count / max(total_change_lines, 1)
        if ratio > 0.3:
            issues.append(f"high_unassigned_lines: {unassigned_count}/{total_change_lines} target")
            confidence -= 0.2
            modifications.append({
                "type": "drop_unmatched_lines",
                "detail": f"{unassigned_count} target patch group_id=null",
                "unassigned_count": unassigned_count,
            })
        else:
            issues.append(f"minor_unassigned_lines: {unassigned_count} {ratio:.0%}")
            confidence -= 0.05

    # 7split_emails Re: keyword
    if split_emails:
        all_re = all(re.match(r"^\s*re:", e.get("subject", ""), re.IGNORECASE) for e in split_emails)
        all_vague = all(not e.get("keyword_category", "").strip() for e in split_emails)
        no_keyword = all(not e.get("keyword", "").strip() for e in split_emails)
        if all_re and no_keyword:
            issues.append("vague_split_evidence: Re: ")
            confidence -= 0.1
        elif all_re and all_vague:
            issues.append("vague_split_evidence: Re: ")

    # 8
    src_files = source_meta.get("files", [])
    tgt_files_all = [f for m in target_metas_with_diff for f in m.get("files", [])]
    path_rename = _detect_path_rename(src_files, tgt_files_all)
    if path_rename:
        issues.append("path_rename: source target ")

    # 9
    has_file_split = any(fa.split for fa in file_assignments)

    confidence = max(0.0, min(1.0, confidence))
    if confidence >= 0.75:
        verdict = Verdict.VALID if not modifications else Verdict.VALID_MODIFIED
    elif confidence >= 0.4:
        verdict = Verdict.VALID_MODIFIED if modifications else Verdict.SUSPICIOUS
    else:
        verdict = Verdict.SUSPICIOUS

    return JudgerResult(
        verdict=verdict,
        confidence=round(confidence, 3),
        issues=issues,
        modifications=modifications,
        path_rename=path_rename,
        has_file_split=has_file_split,
        unassigned_line_count=unassigned_count,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def export_case(case: Dict, epochs: List[int]) -> Optional[Dict]:
    """ case None"""
    case_id = case["id"]

    # source patch
    source_meta = get_patch_diff_and_meta(
        case["source_series_id"], case["source_patch_num"], epochs
    )

    # target patches
    target_patches_meta = json.loads(case.get("target_patches_json") or "[]")
    if not target_patches_meta:
        return None

    target_metas_with_diff: List[Dict] = []
    for meta in target_patches_meta:
        tgt = get_patch_diff_and_meta(
            case["target_series_id"], meta["patch_num"], epochs
        )
        tgt.update({
            "patch_num": meta["patch_num"],
            "patch_total": meta["patch_total"],
            "subject": meta.get("subject", "") or tgt.get("subject", ""),
        })
        target_metas_with_diff.append(tgt)

    # ground truth
    source_diff = source_meta.get("diff", "")
    target_diffs = [m.get("diff", "") for m in target_metas_with_diff]

    file_assignments, groups, unassigned_count = build_ground_truth(
        source_diff, target_diffs, target_metas_with_diff
    )

    # : ground truth 1 group
    # target patch group
    actual_group_ids = set()
    for fa in file_assignments:
        for la in fa.lines:
            if la.line_type in ("add", "del") and la.group_id is not None:
                actual_group_ids.add(la.group_id)
    if len(actual_group_ids) <= 1:
        return None

    # Judger
    judger = judge_case(
        case, source_meta, target_metas_with_diff, file_assignments, unassigned_count
    )

    if judger.verdict == Verdict.INVALID:
        return None

    # split_evidence
    split_emails_raw = json.loads(case.get("split_emails_json") or "[]")
    not_in_ratios = json.loads(case.get("not_in_ratios_json") or "[]")

    # file_assignments
    fa_list = []
    for fa in file_assignments:
        fa_list.append({
            "file": fa.file,
            "split": fa.split,
            "group_ids": fa.group_ids,
            "lines": [
                {
                    "content": la.content,
                    "type": la.line_type,
                    "group_id": la.group_id,
                    "match_score": la.match_score,
                }
                for la in fa.lines
            ],
        })

    return {
        "id": case_id,
        "label_source": case.get("label_source", "unknown"),
        "split_type": case.get("label_q1") or (
            "nc_auto" if case.get("label_source") in ("nc_auto", "expansion") else "unknown"
        ),
        "nc_tagged": bool(case.get("nc_tagged")),
        "expansion_score": case.get("expansion_score"),

        "source": {
            "series_id": case["source_series_id"],
            "version": case.get("source_version"),
            "patch_num": case.get("source_patch_num"),
            "patch_total": case.get("source_patch_total"),
            "subject": source_meta.get("subject") or case.get("source_subject", ""),
            "from_addr": source_meta.get("from_addr"),
            "date_raw": source_meta.get("date_raw"),
            "message_id": source_meta.get("message_id"),
            "lore_url": source_meta.get("lore_url"),
            "files": source_meta.get("files") or json.loads(case.get("source_files") or "[]"),
            "diff": source_diff,
        },

        "target": {
            "series_id": case["target_series_id"],
            "version": case.get("target_version"),
            "patches": [
                {
                    "group_id": idx,
                    "patch_num": m.get("patch_num"),
                    "patch_total": m.get("patch_total"),
                    "subject": m.get("subject", ""),
                    "from_addr": m.get("from_addr"),
                    "date_raw": m.get("date_raw"),
                    "message_id": m.get("message_id"),
                    "lore_url": m.get("lore_url"),
                    "files": m.get("files", []),
                    "diff": m.get("diff", ""),
                }
                for idx, m in enumerate(target_metas_with_diff)
            ],
        },

        "ground_truth": {
            "num_groups": len(groups),
            "file_assignments": fa_list,
            "groups": groups,
        },

        "split_evidence": {
            "emails": split_emails_raw,
            "similarity": case.get("similarity"),
            "not_in_ratios": not_in_ratios,
        },

        "judger": {
            "verdict": judger.verdict.value,
            "confidence": judger.confidence,
            "issues": judger.issues,
            "modifications": judger.modifications,
            "path_rename": judger.path_rename,
            "has_file_split": judger.has_file_split,
            "unassigned_line_count": judger.unassigned_line_count,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=" commit untangle ")
    parser.add_argument("--target", type=int, default=350, help=" 350")
    parser.add_argument("--out", type=str, default="dataset.json", help="")
    parser.add_argument("--verbose", action="store_true", help="")
    args = parser.parse_args()

    print(f"[export] : {args.target}")
    print(f"[export] : {args.out}")

    cases = fetch_positive_cases(target=args.target)
    epochs = db_module._get_available_epochs()
    print(f"[export] epoch : {epochs}")

    dataset: List[Dict] = []
    skipped = 0
    verdict_counts: Dict[str, int] = defaultdict(int)

    for i, case in enumerate(cases):
        if args.verbose:
            print(f"  [{i+1}/{len(cases)}] case_id={case['id']} source={case['source_series_id']} ...", end=" ")

        try:
            record = export_case(case, epochs)
        except Exception as e:
            print(f"\n  [ERROR] case_id={case['id']}: {e}")
            skipped += 1
            continue

        if record is None:
            if args.verbose:
                print("INVALID (skipped)")
            skipped += 1
            continue

        verdict = record["judger"]["verdict"]
        verdict_counts[verdict] += 1
        dataset.append(record)

        if args.verbose:
            print(f"{verdict} (conf={record['judger']['confidence']:.2f})")

    print(f"\n[export] :")
    print(f" cases: {len(cases)}")
    print(f" (INVALID/ERROR): {skipped}")
    print(f" cases (): {len(dataset)}")
    for v, cnt in sorted(verdict_counts.items()):
        print(f"    {v}: {cnt}")

    # targethuman_yes > nc_auto > expansion
    if len(dataset) > args.target:
        priority = {"human_yes": 0, "nc_auto": 1, "expansion": 2, "unknown": 3}
        dataset.sort(key=lambda r: (
            priority.get(r["label_source"], 9),
            -r["judger"]["confidence"],
        ))
        dataset = dataset[:args.target]
        print(f" target={args.target}")

    print(f" : {len(dataset)}")

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n[export] {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
