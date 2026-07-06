"""
diff_grouper.py

 patch diff target patchgroup_index
Group
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

def classify_line(line: str) -> str:
    """Return diff Type: meta / hunk / add / del / context"""
    if line.startswith("diff ") or line.startswith("index ") \
            or line.startswith("--- ") or line.startswith("+++ "):
        return "meta"
    if line.startswith("@@"):
        return "hunk"
    if line.startswith("+"):
        return "add"
    if line.startswith("-"):
        return "del"
    return "context"


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------

def assign_diff_groups(
    source_diff: str,
    target_diffs: List[str],
    fuzzy_threshold: float = 0.55,
) -> List[dict]:
    """ patch Match target patch group_index

    Args:
        source_diff: patch diff
    target_diffs: target patch diff group_index
    fuzzy_threshold: MatchSimilarityThreshold

    Returns:
        list of {
            "content": str, #
            "type": str,          # meta/hunk/add/del/context
        "group": int | None # target patch group_indexNone
        }
    """
    if not source_diff:
        return []

    # ProcessExtract target diff Match
    target_line_sets: List[set] = []
    target_add_lines: List[List[str]] = [] # +/- Match
    for td in target_diffs:
        lines = td.splitlines()
        stripped = {l.lstrip("+-").strip() for l in lines if l.startswith(("+", "-")) and l.strip() not in ("+", "-")}
        target_line_sets.append(stripped)
        target_add_lines.append([l for l in lines if l.startswith(("+", "-"))])

    result = []
    for raw_line in source_diff.splitlines():
        line_type = classify_line(raw_line)
        group: Optional[int] = None

        if line_type in ("add", "del"):
            stripped = raw_line.lstrip("+-").strip()

            if stripped:
                # 1. Match
                for idx, line_set in enumerate(target_line_sets):
                    if stripped in line_set:
                        group = idx
                        break

                # 2. MatchMatchFailure
                if group is None:
                    best_score = 0.0
                    best_idx = None
                    for idx, add_lines in enumerate(target_add_lines):
                        for tl in add_lines:
                            tl_stripped = tl.lstrip("+-").strip()
                            if not tl_stripped:
                                continue
                            score = SequenceMatcher(None, stripped, tl_stripped).ratio()
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                    if best_score >= fuzzy_threshold:
                        group = best_idx

        result.append({
            "content": raw_line,
            "type": line_type,
            "group": group,
        })

    return result


def build_source_line_set(source_diff: str) -> set:
    """Extract diff add/del +/- strip"""
    result = set()
    for line in source_diff.splitlines():
        if line.startswith(("+", "-")) and classify_line(line) in ("add", "del"):
            stripped = line.lstrip("+-").strip()
            if stripped:
                result.add(stripped)
    return result




_DIFF_HEADER_RE = re.compile(r"^diff\s+--git\s+", re.MULTILINE)
_FILE_RE = re.compile(r"^diff\s+--git\s+a/(.+?)\s+b/(.+?)$", re.MULTILINE)


def extract_diff(body: str) -> str:
    """Extract diff diff --git Start"""
    m = _DIFF_HEADER_RE.search(body)
    return body[m.start():] if m else ""


def extract_files(body: str) -> List[str]:
    """ExtractModificationFile"""
    return [m.group(2) for m in _FILE_RE.finditer(body)]


def build_lkml_url(message_id: str) -> str:
    """ lkml.kernel.org message_id """
    mid = message_id.strip().strip("<>")
    return f"https://lkml.kernel.org/r/{mid}"
