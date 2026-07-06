"""
Evaluate Qwen3-32B on Java and C# val sets via API.
High concurrency, checkpoint support, final acc@100/90/80 report.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from pathlib import Path

from openai import OpenAI
import openai as _openai_module

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import evaluate_all, accuracy_a, find_best_mapping

# --- Config ---
API_KEY = os.environ.get("LLM_API_KEY", "")
API_BASE_URL = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
MODEL = os.environ.get("LLM_MODEL", "Qwen3-32B")
MAX_RETRIES = 3
RETRY_DELAY = 5
API_TIMEOUT = 300
FUTURE_TIMEOUT = 600

# --- Reuse prompt/parsing from llm_api.py ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_api import (
    SYSTEM_PROMPT, build_prompt, build_reference, build_prediction,
    _get_changed_lines, _extract_json, call_llm
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def process_one(entry, hint_groups=False):
    eid = entry["id"]
    prompt = build_prompt(entry, hint_groups=hint_groups)
    changed_lines = _get_changed_lines(entry)
    num_changed = len(changed_lines)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=API_TIMEOUT)

    try:
        llm_output, raw = call_llm(client, prompt, MODEL)
    except Exception as e:
        refs = build_reference(entry)
        return {
            "id": eid,
            "pred": [None] * len(refs),
            "ref": refs,
            "status": "error",
            "error": str(e)
        }

    pred_groups, _ = build_prediction(entry, llm_output, num_changed)
    ref_groups = build_reference(entry)

    return {
        "id": eid,
        "pred": pred_groups,
        "ref": ref_groups,
        "status": "ok"
    }


def load_checkpoint(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", []) if isinstance(data, dict) else data
    return {r["id"]: r for r in records}


def save_checkpoint(path, done):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"records": list(done.values())}, f, ensure_ascii=False)
    tmp.replace(path)


def run_eval(dataset_path, ckpt_path, workers=32, hint_groups=False):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    ckpt = Path(ckpt_path)
    done = load_checkpoint(ckpt)
    print(f"[{dataset_path}] total={len(dataset)} done={len(done)} pending={len(dataset)-len(done)}")

    pending = [e for e in dataset if e["id"] not in done]
    if not pending:
        print("  All done, skipping.")
        return done

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one, entry, hint_groups): entry["id"]
            for entry in pending
        }
        completed = 0
        for future in as_completed(futures):
            eid = futures[future]
            completed += 1
            try:
                result = future.result(timeout=FUTURE_TIMEOUT)
                done[eid] = result
            except FutureTimeoutError:
                entry = next(e for e in dataset if e["id"] == eid)
                refs = build_reference(entry)
                done[eid] = {"id": eid, "pred": [None]*len(refs), "ref": refs, "status": "timeout"}
            except Exception as e:
                entry = next(e for e in dataset if e["id"] == eid)
                refs = build_reference(entry)
                done[eid] = {"id": eid, "pred": [None]*len(refs), "ref": refs, "status": "error", "error": str(e)}

            if completed % 50 == 0:
                save_checkpoint(ckpt, done)
                print(f"  [{completed}/{len(pending)}] checkpoint saved")

    save_checkpoint(ckpt, done)
    print(f"  Finished. Total records: {len(done)}")
    return done


def compute_metrics(done, dataset_path):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    scores = []
    errors = 0
    for entry in dataset:
        eid = entry["id"]
        if eid not in done:
            errors += 1
            continue
        r = done[eid]
        pred = r["pred"]
        ref = r["ref"]
        filtered = [(p, t) for p, t in zip(pred, ref) if t is not None]
        if not filtered:
            scores.append(0.0)
            continue
        pred_f = [p for p, _ in filtered]
        ref_f = [t for _, t in filtered]
        acc = accuracy_a(pred_f, ref_f)
        scores.append(acc)

    n = len(scores)
    if n == 0:
        return {}
    acc100 = sum(1 for s in scores if s >= 1.0) / n
    acc90 = sum(1 for s in scores if s >= 0.9) / n
    acc80 = sum(1 for s in scores if s >= 0.8) / n
    return {
        "n": n, "errors": errors,
        "acc@100": round(acc100, 4),
        "acc@90": round(acc90, 4),
        "acc@80": round(acc80, 4),
        "mean_acc": round(sum(scores)/n, 4)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--hint-groups", action="store_true")
    parser.add_argument("--results-dir", default=None,
                        help="Output directory for results (default: scripts/llm_finetune/results/)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "java": {
            "path": str(PROJECT_ROOT / "data" / "java_dataset.json"),
            "ckpt": str(results_dir / "qwen3_32b_java.ckpt.json"),
            "out": str(results_dir / "qwen3_32b_java.json"),
        },
        "csharp": {
            "path": str(PROJECT_ROOT / "data" / "csharp_dataset.json"),
            "ckpt": str(results_dir / "qwen3_32b_csharp.ckpt.json"),
            "out": str(results_dir / "qwen3_32b_csharp.json"),
        },
    }

    for name, cfg in datasets.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"{'='*60}")
        done = run_eval(cfg["path"], cfg["ckpt"], workers=args.workers, hint_groups=args.hint_groups)
        metrics = compute_metrics(done, cfg["path"])
        print(f"\n  Results ({name}):")
        print(f"    N={metrics['n']}, errors={metrics['errors']}")
        print(f"    Acc@100: {metrics['acc@100']}")
        print(f"    Acc@90:  {metrics['acc@90']}")
        print(f"    Acc@80:  {metrics['acc@80']}")
        print(f"    Mean:    {metrics['mean_acc']}")

        with open(cfg["out"], "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "records": list(done.values())}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
