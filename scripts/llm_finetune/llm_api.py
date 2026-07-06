"""
llm_api.py — LLM-based commit untangling evaluation

Workflow:
1. Load evaluation dataset (lkp_untangle.json)
2. Send each tangled diff to LLM for group assignment
3. Collect line-level group predictions
4. Evaluate with metrics.py

Features:
 - Concurrent API calls with configurable workers
 - Retry with timeout handling
 - Checkpoint/resume support
 - Post-processing for group merging

Usage:
    python llm_api.py --input ../../data/lkp_manual/lkp_untangle.json \
        --out results/predictions.json \
        --checkpoint results/predictions.ckpt.json \
        --model qwen3-32b \
        --workers 8 \
        --log-dir results/logs \
        --postprocess
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FutureTimeoutError
)

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
import openai as _openai_module

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import evaluate_all

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

MODEL = os.environ.get("LLM_MODEL", "qwen3-32b")

API_BASE_URL = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")

MAX_RETRIES = 3
RETRY_DELAY = 5

API_TIMEOUT = 300

# SampleExecute
FUTURE_TIMEOUT = 600

# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

SYSTEM_PROMPT = r"""You are an expert at untangling commits.

A "tangled commit" is a single commit that contains multiple independent logical changes mixed together. Your job is to split the changed lines into groups, where each group represents one atomic, independent change.

Rules:
1. Semantically related changes belong to the same group.
2. Formatting-only lines follow nearby semantic lines.
3. Use surrounding diff context to resolve ambiguous lines.
4. Output ONLY valid JSON — no explanation, no markdown fences.
5. group_id is an integer starting from 0.

Output format:
{"groups": [g0, g1, g2, ...]}

Where the array length equals the total number of changed lines, and each element is the group_id (integer) for that line in sequential order.

Example:
- Input: 6 changed lines, split into 2 groups (0 and 1)
- Output: {"groups": [0, 0, 1, 1, 0, 1]}
"""

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------

def _extract_json(text: str) -> str:

    cleaned = re.sub(
        r'^```(?:json)?\s*',
        '',
        text.strip()
    )

    cleaned = re.sub(
        r'\s*```$',
        '',
        cleaned
    ).strip()

    if cleaned.startswith('{'):
        return cleaned

    start = cleaned.find('{')

    if start == -1:
        raise ValueError("No JSON found")

    depth = 0

    for i, ch in enumerate(cleaned[start:], start):

        if ch == '{':
            depth += 1

        elif ch == '}':
            depth -= 1

            if depth == 0:
                return cleaned[start:i+1]

    raise ValueError("Unbalanced JSON")


def _get_changed_lines(entry: Dict) -> List[Dict]:
    """Extract changed lines with seq_id from entry."""
    gt_files = entry["ground_truth"]["files"]
    changed = []
    seq = 0
    for fa in gt_files:
        fname = fa["file"]
        for line in fa["lines"]:
            if line["type"] not in ("add", "del"):
                continue
            changed.append({
                "seq": seq,
                "file": fname,
                "line_idx": line["idx"],
                "content": line["content"],
            })
            seq += 1
    return changed


# ----------------------------------------------------------------------
# Prompt Builder
# ----------------------------------------------------------------------

def build_prompt(entry: Dict, hint_groups: bool = True) -> str:

    source = entry["source"]
    num_groups = entry["ground_truth"]["num_groups"]
    changed_lines = _get_changed_lines(entry)
    num_lines = len(changed_lines)

    # Build file-grouped dict for compact representation
    file_groups = defaultdict(list)
    for cl in changed_lines:
        file_groups[cl["file"]].append(
            [cl["seq"], cl["content"]]
        )

    lines = []

    lines.append("## Task")
    if hint_groups:
        lines.append(
            f"Split this tangled patch into exactly "
            f"{num_groups} independent groups."
        )
    else:
        lines.append(
            "Split this tangled patch into independent groups. "
            "Each group is one atomic logical change."
        )
    lines.append("")

    lines.append("## Patch")
    lines.append(f"Subject: {source.get('subject', '')}")
    lines.append("```diff")
    lines.append(source.get("diff", ""))
    lines.append("```")
    lines.append("")

    if hint_groups:
        lines.append(
            f"## Changed Lines "
            f"(total: {num_lines} lines, "
            f"assign to {num_groups} groups: 0..{num_groups - 1})"
        )
    else:
        lines.append(
            f"## Changed Lines (total: {num_lines} lines)"
        )
    lines.append("```json")
    lines.append(
        json.dumps(
            dict(file_groups),
            ensure_ascii=False
        )
    )
    lines.append("```")
    lines.append("")

    lines.append("## Output")
    lines.append(
        f'Return a JSON object: {{"groups": [g0, g1, g2, ...]}}'
    )
    lines.append(f"- Length must be exactly {num_lines}")
    if hint_groups:
        lines.append(
            f"- Each value is an integer from 0 to {num_groups - 1}"
        )
    else:
        lines.append(
            "- Each value is an integer starting from 0"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    prompt: str,
    model: str
) -> Tuple[Dict, str]:
    """Call LLM and return (parsed_json, raw_response)."""

    raw = ""

    for attempt in range(MAX_RETRIES):

        try:

            resp = client.chat.completions.create(
                model=model,

                # max_tokens=163840,

                timeout=API_TIMEOUT,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            raw = (
                resp
                .choices[0]
                .message
                .content
                .strip()
            )

            json_str = _extract_json(raw)

            return json.loads(json_str), raw

        except (
            json.JSONDecodeError,
            ValueError
        ) as e:

            if attempt == MAX_RETRIES - 1:

                raise ValueError(
                    f"Invalid JSON: {e}\n"
                    f"RAW:\n{raw[:500]}"
                )

            time.sleep(RETRY_DELAY)

        except _openai_module.RateLimitError:

            wait = RETRY_DELAY * (attempt + 2)

            print(f"[rate limit] sleep {wait}s")

            time.sleep(wait)

        except _openai_module.APITimeoutError:

            print(
                f"[timeout] API timeout "
                f"(attempt {attempt+1}/{MAX_RETRIES})"
            )

            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(RETRY_DELAY)

        except _openai_module.APIError:

            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(RETRY_DELAY)

    raise RuntimeError("LLM failed")


# ----------------------------------------------------------------------
# Build GT & Prediction
# ----------------------------------------------------------------------

def build_reference(entry: Dict):

    refs = []

    gt_files = entry["ground_truth"]["files"]

    for fa in gt_files:

        for line in fa["lines"]:

            if line["type"] not in ("add", "del"):
                continue

            refs.append(line["group_id"])

    return refs


def build_prediction(
    entry: Dict,
    llm_output: Dict,
    num_changed_lines: int
) -> Tuple[List, Dict]:
    """Build prediction from LLM output.

    Supports two formats:
    - New: {"groups": [g0, g1, g2, ...]} — direct array
    - Legacy: {"assignments": [{file, line_idx, group_id}, ...]}
    """

    # New format: direct groups array
    if "groups" in llm_output:
        groups = llm_output["groups"]

        # Length alignment
        if len(groups) < num_changed_lines:
            groups = groups + [None] * (num_changed_lines - len(groups))

        preds = groups[:num_changed_lines]

        # Build pred_by_file
        gt_files = entry["ground_truth"]["files"]
        pred_by_file = {}
        idx = 0
        for fa in gt_files:
            fname = fa["file"]
            file_preds = []
            for line in fa["lines"]:
                if line["type"] not in ("add", "del"):
                    continue
                file_preds.append(preds[idx])
                idx += 1
            pred_by_file[fname] = file_preds

        return preds, pred_by_file

    # Legacy fallback: assignments format
    gt_files = entry["ground_truth"]["files"]

    pred_map = {}

    for item in llm_output.get("assignments", []):

        key = (
            item["file"],
            item["line_idx"]
        )

        pred_map[key] = item.get("group_id")

    preds = []

    pred_by_file = {}

    for fa in gt_files:

        fname = fa["file"]

        file_preds = []

        for line in fa["lines"]:

            if line["type"] not in ("add", "del"):
                continue

            key = (
                fname,
                line["idx"]
            )

            p = pred_map.get(key)

            preds.append(p)

            file_preds.append(p)

        pred_by_file[fname] = file_preds

    return preds, pred_by_file


# ----------------------------------------------------------------------
# Postprocessing: merge excess groups
# ----------------------------------------------------------------------

def postprocess_merge_groups(
    preds: List,
    num_groups: int
) -> List:
    """If LLM produced more unique groups than expected, merge small ones."""

    non_null = [p for p in preds if p is not None]
    if not non_null:
        return preds

    unique_groups = set(non_null)
    if len(unique_groups) <= num_groups:
        return preds

    # Count lines per group
    counts = Counter(non_null)
    # Keep the top num_groups groups, merge the rest
    top_groups = [
        g for g, _ in counts.most_common(num_groups)
    ]
    top_set = set(top_groups)

    # Map top groups to 0..num_groups-1
    remap = {g: i for i, g in enumerate(sorted(top_groups))}

    # For small groups, find the nearest top group by position
    result = []
    for i, p in enumerate(preds):
        if p is None:
            result.append(None)
        elif p in top_set:
            result.append(remap[p])
        else:
            # Find nearest line with a top group
            best_dist = float('inf')
            best_group = top_groups[0]
            for j, q in enumerate(preds):
                if q is not None and q in top_set:
                    dist = abs(i - j)
                    if dist < best_dist:
                        best_dist = dist
                        best_group = q
            result.append(remap[best_group])

    return result


# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------

def load_checkpoint(path: Path):

    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data

    else:
        records = data.get("records", [])

    done = {}

    for r in records:

        done[r["id"]] = r

    return done


def save_checkpoint(
    path: Path,
    done: Dict[int, Dict]
):

    payload = {
        "records": list(done.values()),
        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    tmp = path.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2
        )

    tmp.replace(path)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def write_log(
    log_dir: Optional[Path],
    entry_id: int,
    model: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    parsed_json: Optional[Dict],
    parse_error: Optional[str],
    prediction: Optional[List],
    reference: Optional[List],
    accuracy_a: Optional[float],
    num_null: int
):
    """Write detailed log for a single LLM call."""
    if log_dir is None:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "_").replace("/", "_")
    filename = f"{safe_model}_{entry_id}_{timestamp}.json"

    log_entry = {
        "id": entry_id,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        "output": {
            "raw_response": raw_response,
            "parsed_json": parsed_json,
            "parse_error": parse_error,
        },
        "result": {
            "prediction": prediction,
            "reference": reference,
            "accuracy_a": accuracy_a,
            "num_null": num_null,
        }
    }

    log_path = log_dir / filename
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[write_log] WARNING: could not write log {filename}: {e}")


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

def process_one_entry(
    entry: Dict,
    model: str,
    api_key: str,
    log_dir: Optional[Path] = None,
    do_postprocess: bool = False,
    hint_groups: bool = True,
):

    eid = entry["id"]

    print(f"[start] id={eid}")

    prompt = build_prompt(entry, hint_groups=hint_groups)
    changed_lines = _get_changed_lines(entry)
    num_changed_lines = len(changed_lines)
    num_groups = entry["ground_truth"]["num_groups"]

    client = OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
        timeout=API_TIMEOUT
    )

    raw_response = ""
    parse_error = None

    try:
        llm_output, raw_response = call_llm(
            client,
            prompt,
            model
        )
    except Exception as e:
        parse_error = str(e)
        llm_output = {}

    pred_groups, pred_by_file = build_prediction(
        entry,
        llm_output,
        num_changed_lines
    )

    # Optional postprocessing
    if do_postprocess and pred_groups:
        pred_groups = postprocess_merge_groups(
            pred_groups, num_groups
        )
        # Rebuild pred_by_file from postprocessed preds
        gt_files = entry["ground_truth"]["files"]
        pred_by_file = {}
        idx = 0
        for fa in gt_files:
            fname = fa["file"]
            file_preds = []
            for line in fa["lines"]:
                if line["type"] not in ("add", "del"):
                    continue
                file_preds.append(pred_groups[idx])
                idx += 1
            pred_by_file[fname] = file_preds

    ref_groups = build_reference(entry)

    # Compute per-sample accuracy_a for logging
    num_null = sum(1 for p in pred_groups if p is None)
    accuracy_a = None
    if not parse_error:
        # Simple accuracy: fraction of non-null correct matches
        # (full permutation-invariant eval happens in evaluate_all)
        non_null_count = len(pred_groups) - num_null
        if non_null_count > 0:
            # Just count matching pairs for a rough estimate
            matches = sum(
                1 for p, r in zip(pred_groups, ref_groups)
                if p is not None and p == r
            )
            accuracy_a = matches / len(pred_groups)

    # Write log
    write_log(
        log_dir=log_dir,
        entry_id=eid,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        raw_response=raw_response,
        parsed_json=llm_output if not parse_error else None,
        parse_error=parse_error,
        prediction=pred_groups,
        reference=ref_groups,
        accuracy_a=accuracy_a,
        num_null=num_null,
    )

    if parse_error:
        raise ValueError(parse_error)

    true_by_file = {}

    for fa in entry["ground_truth"]["files"]:

        true_by_file[fa["file"]] = [
            line["group_id"]
            for line in fa["lines"]
            if line["type"] in ("add", "del")
        ]

    result = {
        "id": eid,

        "pred_groups": pred_groups,

        "pred_by_file": pred_by_file,

        "true_groups": ref_groups,

        "true_by_file": true_by_file,

        "num_groups": num_groups,

        "llm_output": llm_output,
    }

    print(
        f"[done] id={eid} "
        f"null={num_null}/{num_changed_lines}"
    )

    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="../../data/lkp_manual/lkp_untangle.json"
    )

    parser.add_argument(
        "--out",
        default="results/predictions.json"
    )

    parser.add_argument(
        "--checkpoint",
        default="results/predictions.ckpt.json"
    )

    parser.add_argument(
        "--model",
        default=MODEL
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1
    )

    parser.add_argument(
        "--log-dir",
        default="results/logs",
        help="Directory for detailed per-call logs"
    )

    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Merge excess groups via nearest-neighbor"
    )

    parser.add_argument(
        "--no-hint-groups",
        action="store_true",
        help="Do not tell LLM the expected number of groups"
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    args = parser.parse_args()

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY", "")
    )

    # Setup log directory
    log_dir = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:

        dataset = json.load(f)

    if args.max_items:
        dataset = dataset[:args.max_items]

    out_path = Path(args.out)

    ckpt_path = Path(args.checkpoint)

    done = load_checkpoint(ckpt_path)

    print(
        f"[resume] "
        f"{len(done)} entries already done"
    )

    pending = [
        e for e in dataset
        if e["id"] not in done
    ]

    print(
        f"[dataset] "
        f"total={len(dataset)} "
        f"pending={len(pending)} "
        f"workers={args.workers}"
    )

    # ------------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_map = {

            executor.submit(
                process_one_entry,
                entry,
                args.model,
                api_key,
                log_dir,
                args.postprocess,
                not args.no_hint_groups,
            ): entry["id"]

            for entry in pending
        }

        completed_since_save = 0

        total_pending = len(pending)

        for idx, future in enumerate(
            as_completed(future_map),
            1
        ):

            eid = future_map[future]

            print(
                f"[{idx}/{total_pending}] "
                f"id={eid}"
            )

            try:

                record = future.result(
                    timeout=FUTURE_TIMEOUT
                )

                done[eid] = record

            except FutureTimeoutError:

                print(
                    f"[future timeout] "
                    f"id={eid}"
                )

                entry = next(
                    e for e in dataset
                    if e["id"] == eid
                )

                refs = build_reference(entry)

                done[eid] = {
                    "id": eid,
                    "pred_groups": [None] * len(refs),
                    "pred_by_file": {},
                    "true_groups": refs,
                    "true_by_file": {},
                    "num_groups": entry["ground_truth"]["num_groups"],
                    "llm_output": {
                        "error": "future timeout"
                    }
                }

            except Exception as e:

                print(
                    f"[error] "
                    f"id={eid} "
                    f"{e}"
                )

                entry = next(
                    e for e in dataset
                    if e["id"] == eid
                )

                refs = build_reference(entry)

                done[eid] = {
                    "id": eid,
                    "pred_groups": [None] * len(refs),
                    "pred_by_file": {},
                    "true_groups": refs,
                    "true_by_file": {},
                    "num_groups": entry["ground_truth"]["num_groups"],
                    "llm_output": {
                        "error": str(e)
                    }
                }

            completed_since_save += 1

            if completed_since_save >= args.save_every:

                save_checkpoint(
                    ckpt_path,
                    done
                )

                completed_since_save = 0

    # ------------------------------------------------------------------
    # Final Save
    # ------------------------------------------------------------------

    save_checkpoint(
        ckpt_path,
        done
    )

    ordered = []

    id_to_record = {
        r["id"]: r
        for r in done.values()
    }

    for entry in dataset:

        ordered.append(
            id_to_record[entry["id"]]
        )

    predictions = []
    references = []

    for r in ordered:

        predictions.append(r["pred_groups"])

        references.append(r["true_groups"])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    results = evaluate_all(
        predictions,
        references
    )

    print("\n================ EVALUATION ================")

    print(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        )
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    final_output = {
        "metrics": results,
        "predictions": ordered,
    }

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nSaved to {out_path}"
    )

    print(
        f"Checkpoint saved to {ckpt_path}"
    )


if __name__ == "__main__":
    main()