"""
judge_dataset.py — LLM dataset.json 350 DataVerifyFix


 1. dataset.json
 2. Data prompt OpenAI API
 a. Verify source→target Error
 b. source target DiffVariable
 Output source
 c. split_evidence
 3. LLM ground truth repair_log
 4. Compute difficulty score
 5. Output dataset_judged.json evaluation
 6. Process id checkpoint FileSkip


  python judge_dataset.py [--input dataset.json] [--out dataset_judged.json]
                          [--checkpoint judge_checkpoint.json]
                          [--model claude-sonnet-4-6]
                          [--max-tokens-per-entry 8000]
                          [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import process
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
import openai as _openai_module

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
API_BASE_URL = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

SYSTEM_PROMPT = """\
You are an expert Linux kernel code reviewer specializing in patch split analysis.

You will receive a "tangled" source patch that was later split into N target patches,
along with evidence emails discussing the split. Your job has FOUR parts:

━━━ PART 1: Line Assignment Verification ━━━
Each changed line (+/-) in the source diff has a proposed group_id (0-indexed target).
Verify and correct these assignments. A line belongs to group G if its content
(or a semantically equivalent version) appears in target[G]'s diff.
- Use surrounding context to resolve ambiguous lines (e.g. `{`, `}`, blank lines).
- Lines absent from all targets → group_id: null.

━━━ PART 2: Content Alignment Fixes ━━━
Detect minor cosmetic differences between source lines and their counterparts in targets.
Fix ONLY when the difference is minor and clearly cosmetic (does NOT change semantics):
  • Variable/identifier renamed (e.g. typo fix, style normalization)
  • Whitespace/indentation reformat
  • Added or removed blank lines
  • Added/removed inline comment (e.g. `/* fix */`)
  • Line wrapped/unwrapped (one logical line split into two or merged)
Do NOT fix semantic changes (different logic, different values, added conditions).
For each fix, provide the corrected source line content that aligns with the target.

━━━ PART 3: Split Authenticity Judgment ━━━
Based on the split evidence emails, judge whether this is a genuine split request.
Look for keywords like: "split", "separate", "break", "divide", "too large",
"should be N patches", etc. in the email subjects or context snippets.

━━━ PART 4: Full Correspondence Verification ━━━
After applying all corrections from Parts 1 and 2, verify that the source patch and
the union of all target patches are in COMPLETE correspondence — i.e., every changed
line (+/-) in the source is accounted for in exactly one target, and no target
introduces content that has no counterpart in the source.

Specifically check:
A. COVERAGE: Every non-null assigned source line has a matching line in its target.
B. NO_EXTRA: No target patch contains changed lines that are entirely absent from source
   (i.e., the targets don't add new logic beyond what the source contains).
C. NO_MISSING: No source changed line is left unassigned (group_id: null) unless it is
   a pure formatting artifact (blank line, brace) with no semantic content.

Based on this, set "alignment" to one of:
  "FULL"     — complete correspondence, source ≡ union(targets) after cosmetic fixes
  "PARTIAL"  — most lines match but there are residual gaps or extras (< 10% of lines)
  "MISMATCH" — significant content present in targets but absent from source, or vice versa

For each gap found, describe it in "alignment_issues".

━━━ OUTPUT FORMAT ━━━
YOU MUST OUTPUT ONLY A SINGLE JSON OBJECT. No prose, no explanation, no markdown
outside the JSON. Your response MUST start with `{` and end with `}`.

{
  "verdict": "CORRECT" | "FIXED" | "UNCERTAIN",
  "line_corrections": [
    {
      "file": "<filepath>",
      "line_idx": <int>,
      "old_group": <int|null>,
      "new_group": <int|null>,
      "reason": "<max 15 words>"
    }
  ],
  "content_fixes": [
    {
      "file": "<filepath>",
      "line_idx": <int>,
      "original_content": "<exact original line>",
      "fixed_content": "<corrected line aligned with target>",
      "fix_type": "rename" | "whitespace" | "blank_line" | "comment" | "line_wrap",
      "reason": "<max 15 words>"
    }
  ],
  "split_authentic": true | false | null,
  "split_confidence": 0.0,
  "split_reason": "<max 30 words>",
  "alignment": "FULL" | "PARTIAL" | "MISMATCH",
  "alignment_issues": [
    {
      "issue_type": "missing_in_target" | "extra_in_target" | "unassigned_source",
      "file": "<filepath>",
      "description": "<max 30 words describing the gap>"
    }
  ],
  "notes": "<max 50 words>"
}

Rules:
- "verdict": CORRECT if no changes needed; FIXED if any corrections/fixes applied;
  UNCERTAIN if you cannot determine assignments confidently.
- "line_corrections": only list lines where group assignment changes.
- "content_fixes": only list lines with minor cosmetic differences; skip semantic changes.
- "split_authentic": true=genuine split request, false=not a split, null=cannot determine.
- "alignment": assessed AFTER applying all corrections from Parts 1 and 2.
- "alignment_issues": empty [] if alignment is FULL.
- Empty arrays [] are valid for line_corrections, content_fixes, and alignment_issues.
"""


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------

def _classify_line(line: str) -> str:
    if line.startswith('+') and not line.startswith('+++'):
        return 'add'
    if line.startswith('-') and not line.startswith('---'):
        return 'del'
    if line.startswith('@@'):
        return 'hunk'
    if line.startswith('diff ') or line.startswith('index ') or \
       line.startswith('--- ') or line.startswith('+++ ') or \
       line.startswith('new file') or line.startswith('deleted file') or \
       line.startswith('rename ') or line.startswith('similarity '):
        return 'meta'
    return 'context'


def _changed_lines_only(diff: str) -> str:
    """Keep only +/- lines and hunk headers (strip context) to reduce tokens."""
    out = []
    for line in diff.splitlines():
        t = _classify_line(line)
        if t in ('add', 'del', 'hunk', 'meta'):
            out.append(line)
    return '\n'.join(out)


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(entry: Dict, max_tokens: int) -> str:
    source = entry['source']
    targets = entry['target']['patches']
    fa_list = entry['ground_truth']['file_assignments']
    split_evidence = entry.get('split_evidence', {})

    src_diff = source.get('diff', '')
    tgt_diffs = [p.get('diff', '') or '' for p in targets]

    # Token budget: strip context lines if over limit
    budget_per_diff = max_tokens // (1 + len(tgt_diffs))
    if _estimate_tokens(src_diff) > budget_per_diff:
        src_diff = _changed_lines_only(src_diff)
    tgt_diffs_out = []
    for d in tgt_diffs:
        if _estimate_tokens(d) > budget_per_diff:
            d = _changed_lines_only(d)
        tgt_diffs_out.append(d)

    lines = []

    # ── Source patch ──
    lines.append("## Source Patch")
    lines.append(f"Subject: {source.get('subject', '')}")
    lines.append(f"Files: {', '.join(source.get('files', []))}")
    lines.append("```diff")
    lines.append(src_diff)
    lines.append("```")
    lines.append("")

    # ── Target patches ──
    lines.append("## Target Patches (the correct split result)")
    for i, (patch, diff) in enumerate(zip(targets, tgt_diffs_out)):
        lines.append(f"### Target {i} (group_id={i})")
        lines.append(f"Subject: {patch.get('subject', '')}")
        lines.append("```diff")
        lines.append(diff)
        lines.append("```")
        lines.append("")

    # ── Split evidence emails ──
    emails = split_evidence.get('emails', [])
    if emails:
        lines.append("## Split Evidence Emails")
        lines.append("These emails discuss the patch split. Use them for Part 3 judgment.")
        for em in emails[:5]:  # cap at 5 to save tokens
            lines.append(f"- Subject: {em.get('subject', '')}")
            kw = em.get('keyword', '')
            ctx = em.get('context', '')
            if kw:
                lines.append(f"  Keyword found: \"{kw}\"")
            if ctx:
                # Truncate context to 200 chars
                lines.append(f"  Context: {ctx[:200]}")
        lines.append("")

    # ── Proposed line assignments ──
    proposed = []
    for fa in fa_list:
        for idx, la in enumerate(fa['lines']):
            if la['type'] in ('add', 'del'):
                proposed.append({
                    "file": fa['file'],
                    "line_idx": idx,
                    "content": la['content'],
                    "proposed_group": la['group_id'],
                })

    lines.append("## Proposed Line Assignments")
    lines.append("Format: {file, line_idx, content, proposed_group}")
    lines.append("```json")
    lines.append(json.dumps(proposed, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    # ── Correspondence summary for Part 4 ──
    # Count changed lines per group in source assignments vs target diffs
    src_group_counts: Dict[Optional[int], int] = {}
    for p in proposed:
        g = p['proposed_group']
        src_group_counts[g] = src_group_counts.get(g, 0) + 1

    tgt_changed_counts = []
    for d in tgt_diffs_out:
        cnt = sum(1 for ln in d.splitlines()
                  if (ln.startswith('+') and not ln.startswith('+++'))
                  or (ln.startswith('-') and not ln.startswith('---')))
        tgt_changed_counts.append(cnt)

    lines.append("## Correspondence Summary (for Part 4)")
    lines.append("Source changed lines assigned per group:")
    for g, cnt in sorted((k, v) for k, v in src_group_counts.items() if k is not None):
        lines.append(f"  group {g}: {cnt} lines assigned  (target {g} has {tgt_changed_counts[g] if g < len(tgt_changed_counts) else '?'} changed lines)")
    unassigned_cnt = src_group_counts.get(None, 0)
    if unassigned_cnt:
        lines.append(f"  unassigned (null): {unassigned_cnt} lines")
    lines.append("")
    lines.append("Now output your JSON response (start with `{`, end with `}`):")

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Extract the first complete JSON object from text, even if surrounded by prose."""
    cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()
    if cleaned.startswith('{'):
        return cleaned
    # Brace-counting fallback
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise ValueError("Unbalanced braces in response")


def call_llm(client: OpenAI, prompt: str, model: str) -> Dict:
    """Call the API and return parsed JSON. Raises on failure."""
    raw = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            json_str = _extract_json(raw)
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == MAX_RETRIES - 1:
                raise ValueError(
                    f"LLM returned invalid JSON after {MAX_RETRIES} attempts: {e}\n"
                    f"Raw (first 400 chars): {raw[:400]}"
                )
            time.sleep(RETRY_DELAY)
        except _openai_module.RateLimitError:
            wait = RETRY_DELAY * (attempt + 2)
            print(f"    [rate limit] waiting {wait}s...")
            time.sleep(wait)
        except _openai_module.APIError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY)
    raise RuntimeError("LLM call failed after retries")


# ---------------------------------------------------------------------------
# Apply corrections
# ---------------------------------------------------------------------------

def apply_corrections(entry: Dict, llm_result: Dict) -> Tuple[Dict, List[Dict]]:
    """
    Apply LLM line_corrections and content_fixes to ground_truth.file_assignments.
    Returns (updated_ground_truth, repair_log).
    """
    line_corrections = llm_result.get('line_corrections', [])
    content_fixes    = llm_result.get('content_fixes', [])

    if not line_corrections and not content_fixes:
        return entry['ground_truth'], []

    # Build lookup maps keyed by (file, line_idx)
    lc_map: Dict[Tuple, Dict] = {}
    for c in line_corrections:
        lc_map[(c.get('file', ''), c['line_idx'])] = c

    cf_map: Dict[Tuple, Dict] = {}
    for c in content_fixes:
        cf_map[(c.get('file', ''), c['line_idx'])] = c

    repair_log = []
    fa_list = entry['ground_truth']['file_assignments']

    for fa in fa_list:
        fname = fa['file']
        for idx, la in enumerate(fa['lines']):
            key = (fname, idx)

            # Apply group assignment correction
            lc = lc_map.get(key)
            if lc is not None:
                old_group = la['group_id']
                new_group = lc.get('new_group')
                if old_group != new_group:
                    la['group_id'] = new_group
                    la['match_score'] = -1.0  # mark as LLM-corrected
                    repair_log.append({
                        "type": "group_reassign",
                        "file": fname,
                        "line_idx": idx,
                        "content": la['content'],
                        "old_group_id": old_group,
                        "new_group_id": new_group,
                        "reason": lc.get('reason', ''),
                    })

            # Apply content alignment fix
            cf = cf_map.get(key)
            if cf is not None:
                original = cf.get('original_content', '')
                fixed    = cf.get('fixed_content', '')
                if fixed and fixed != la['content']:
                    old_content = la['content']
                    la['content'] = fixed
                    la['match_score'] = -2.0  # mark as content-fixed
                    repair_log.append({
                        "type": "content_fix",
                        "fix_type": cf.get('fix_type', 'unknown'),
                        "file": fname,
                        "line_idx": idx,
                        "old_content": old_content,
                        "new_content": fixed,
                        "reason": cf.get('reason', ''),
                    })

    # Recompute split flag and group_ids for each file after corrections
    for fa in fa_list:
        gids = set()
        for la in fa['lines']:
            if la['type'] in ('add', 'del') and la['group_id'] is not None:
                gids.add(la['group_id'])
        fa['group_ids'] = sorted(gids)
        fa['split'] = len(gids) > 1

    return entry['ground_truth'], repair_log


# ---------------------------------------------------------------------------
# Difficulty score
# ---------------------------------------------------------------------------

def compute_difficulty(ground_truth: Dict) -> Dict:
    """
    Compute difficulty score based on code entanglement metrics.

    Dimensions:
      file_split_ratio  : files split across groups / total files          weight 0.35
      interleave_ratio  : group_id switches / total changed lines          weight 0.35
      unassigned_ratio  : unassigned changed lines / total changed lines   weight 0.10
      num_groups_factor : (num_groups - 2) / 3, capped at 1.0             weight 0.20
    """
    fa_list    = ground_truth.get('file_assignments', [])
    num_groups = ground_truth.get('num_groups', 2)

    total_files  = len(fa_list)
    split_files  = sum(1 for fa in fa_list if fa.get('split', False))
    file_split_ratio = split_files / max(total_files, 1)

    total_changed = 0
    unassigned    = 0
    switches      = 0

    for fa in fa_list:
        prev_group = None
        for la in fa.get('lines', []):
            if la['type'] not in ('add', 'del'):
                continue
            total_changed += 1
            gid = la['group_id']
            if gid is None:
                unassigned += 1
            else:
                if prev_group is not None and gid != prev_group:
                    switches += 1
                prev_group = gid

    interleave_ratio  = switches   / max(total_changed, 1)
    unassigned_ratio  = unassigned / max(total_changed, 1)
    num_groups_factor = min((num_groups - 2) / 3.0, 1.0)

    score = (
        0.35 * file_split_ratio +
        0.35 * interleave_ratio +
        0.10 * unassigned_ratio +
        0.20 * num_groups_factor
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    level = "easy" if score < 0.3 else ("medium" if score < 0.6 else "hard")

    return {
        "score": score,
        "level": level,
        "breakdown": {
            "file_split_ratio":  round(file_split_ratio,  4),
            "interleave_ratio":  round(interleave_ratio,  4),
            "unassigned_ratio":  round(unassigned_ratio,  4),
            "num_groups_factor": round(num_groups_factor, 4),
        },
    }


# ---------------------------------------------------------------------------
# Output format builder
# ---------------------------------------------------------------------------

def build_output_entry(entry: Dict, llm_result: Dict, repair_log: List[Dict]) -> Dict:
    source = entry['source']
    target = entry['target']
    gt     = entry['ground_truth']
    difficulty = compute_difficulty(gt)

    return {
        "id":           entry['id'],
        "label_source": entry.get('label_source'),
        "split_type":   entry.get('split_type'),
        "difficulty":   difficulty,

        # ── LLM split authenticity judgment ──
        "split_authentic": {
            "judgment":   llm_result.get('split_authentic'),
            "confidence": llm_result.get('split_confidence'),
            "reason":     llm_result.get('split_reason', ''),
        },

        "source": {
            "series_id":   source.get('series_id'),
            "version":     source.get('version'),
            "patch_num":   source.get('patch_num'),
            "patch_total": source.get('patch_total'),
            "subject":     source.get('subject'),
            "from_addr":   source.get('from_addr'),
            "date_raw":    source.get('date_raw'),
            "message_id":  source.get('message_id'),
            "lore_url":    source.get('lore_url'),
            "files":       source.get('files', []),
            "diff":        source.get('diff', ''),
        },

        "targets": [
            {
                "group_id":    p.get('group_id', i),
                "patch_num":   p.get('patch_num'),
                "patch_total": p.get('patch_total'),
                "subject":     p.get('subject'),
                "from_addr":   p.get('from_addr'),
                "date_raw":    p.get('date_raw'),
                "message_id":  p.get('message_id'),
                "lore_url":    p.get('lore_url'),
                "files":       p.get('files', []),
                "diff":        p.get('diff', ''),
            }
            for i, p in enumerate(target.get('patches', []))
        ],

        # ── Ground truth: line-level, easy to evaluate ──
        "ground_truth": {
            "num_groups": gt.get('num_groups'),
            "groups":     gt.get('groups', []),
            "files": [
                {
                    "file":      fa['file'],
                    "split":     fa.get('split', False),
                    "group_ids": fa.get('group_ids', []),
                    # Each line: idx, content (possibly LLM-corrected), type, group_id, match_score
                    "lines": [
                        {
                            "idx":         idx,
                            "content":     la['content'],
                            "type":        la['type'],
                            "group_id":    la['group_id'],
                            "match_score": la.get('match_score', 0.0),
                        }
                        for idx, la in enumerate(fa.get('lines', []))
                    ],
                }
                for fa in gt.get('file_assignments', [])
            ],
        },

        "split_evidence": entry.get('split_evidence', {}),

        "llm_judge": {
            "verdict":          llm_result.get('verdict', 'SKIPPED'),
            "model":            llm_result.get('model', ''),
            "line_corrections": llm_result.get('line_corrections', []),
            "content_fixes":    llm_result.get('content_fixes', []),
            "alignment":        llm_result.get('alignment', 'UNKNOWN'),
            "alignment_issues": llm_result.get('alignment_issues', []),
            "notes":            llm_result.get('notes', ''),
        },

        # Full audit trail of every change made
        "repair_log": repair_log,

        "original_judger": entry.get('judger', {}),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> Dict[int, Dict]:
    """Returns {entry_id: output_entry}. Skips ERROR entries so they get retried."""
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        records = json.load(f)
    return {
        r['id']: r for r in records
        if r.get('llm_judge', {}).get('verdict') not in ('ERROR',)
    }


def save_checkpoint(path: Path, done: Dict[int, Dict]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(list(done.values()), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM judge for dataset.json")
    parser.add_argument('--input',      default='dataset.json',          help='Input dataset')
    parser.add_argument('--out',        default='dataset_judged.json',   help='Output dataset')
    parser.add_argument('--checkpoint', default='judge_checkpoint.json', help='Checkpoint file ()')
    parser.add_argument('--model',      default=MODEL,                   help='Model ID')
    parser.add_argument('--max-tokens-per-entry', type=int, default=8000,
                        help='Max diff tokens per entry before stripping context lines')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build prompts and estimate tokens without calling API')
    parser.add_argument('--ids', nargs='+', type=int, default=None,
                        help='Only process specific entry IDs (for debugging)')
    parser.add_argument('--max-items', type=int, default=50, help='How many items to process')

    args = parser.parse_args()

    api_key = (os.environ.get('ANTHROPIC_API_KEY')
               or os.environ.get('OPENAI_API_KEY')
               or os.environ.get('LLM_API_KEY', ''))
    if not api_key and not args.dry_run:
        print("ERROR: API key not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=API_BASE_URL) if not args.dry_run else None

    print(f"[judge] Reading {args.input}...")
    with open(args.input, encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"[judge] Loaded {len(dataset)} entries")

    checkpoint_path = Path(args.checkpoint)
    done = load_checkpoint(checkpoint_path)
    print(f"[judge] Checkpoint: {len(done)} entries already processed")

    if args.ids:
        dataset = [e for e in dataset if e['id'] in set(args.ids)]
        print(f"[judge] Filtered to {len(dataset)} entries by --ids")
    
    max_items = args.max_items
    processed = 0

    total               = len(dataset)
    skipped             = 0
    errors              = 0
    total_input_tokens  = 0
    total_output_tokens = 0

    for i, entry in enumerate(dataset):
        print(f'processed: {processed} max_items: {max_items}')
        if processed >= max_items:
            break

        eid = entry['id']

        if eid in done:
            skipped += 1
            continue

        prompt       = build_prompt(entry, args.max_tokens_per_entry)
        prompt_tokens = _estimate_tokens(SYSTEM_PROMPT + prompt)
        total_input_tokens += prompt_tokens
        prefix = f"[{i+1}/{total}] id={eid}"

        if args.dry_run:
            print(f"{prefix}  ~{prompt_tokens} tokens  (dry-run)")
            llm_result = {
                "verdict": "DRY_RUN", "line_corrections": [], "content_fixes": [],
                "split_authentic": None, "split_confidence": None,
                "split_reason": "", "alignment": "UNKNOWN", "alignment_issues": [],
                "notes": "", "model": args.model,
            }
            gt, repair_log = apply_corrections(entry, llm_result)
            entry['ground_truth'] = gt
            done[eid] = build_output_entry(entry, llm_result, repair_log)
            continue

        print(f"{prefix}  ~{prompt_tokens} tokens", end="  ", flush=True)

        try:
            llm_result = call_llm(client, prompt, args.model)
            llm_result['model'] = args.model
            verdict      = llm_result.get('verdict', '?')
            n_lc         = len(llm_result.get('line_corrections', []))
            n_cf         = len(llm_result.get('content_fixes', []))
            authentic    = llm_result.get('split_authentic')
            alignment    = llm_result.get('alignment', '?')
            n_ai         = len(llm_result.get('alignment_issues', []))
            total_output_tokens += 600
            print(f"{verdict}  lc={n_lc}  cf={n_cf}  authentic={authentic}  align={alignment}({n_ai})")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
            llm_result = {
                "verdict": "ERROR", "line_corrections": [], "content_fixes": [],
                "split_authentic": None, "split_confidence": None,
                "split_reason": "", "alignment": "UNKNOWN", "alignment_issues": [],
                "notes": str(e)[:200], "model": args.model,
            }

        gt, repair_log = apply_corrections(entry, llm_result)
        entry['ground_truth'] = gt
        done[eid] = build_output_entry(entry, llm_result, repair_log)

        # Checkpoint after every entry
        save_checkpoint(checkpoint_path, done)
        processed += 1

    # Write final output in original dataset order
    id_to_pos = {e['id']: i for i, e in enumerate(dataset)}
    all_done  = sorted(done.values(), key=lambda r: id_to_pos.get(r['id'], 99999))

    out_path = Path(args.out)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_done, f, ensure_ascii=False, indent=2)

    print(f"\n[judge] Done.")
    print(f"  Total entries:    {total}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors:           {errors}")
    print(f"  Written to:       {out_path}  ({len(all_done)} entries)")
    if not args.dry_run:
        print(f"  Est. input tokens:  {total_input_tokens:,}")
        print(f"  Est. output tokens: {total_output_tokens:,}")
        cost = total_input_tokens * 3e-6 + total_output_tokens * 15e-6
        print(f"  Est. cost:          ${cost:.4f}")


if __name__ == '__main__':
    main()
