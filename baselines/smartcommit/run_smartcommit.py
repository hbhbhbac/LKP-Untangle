"""
run_smartcommit.py — SmartCommit faithful
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_baselines_dir = Path(__file__).resolve().parent.parent
_replication_dir = _baselines_dir.parent
sys.path.insert(0, str(_baselines_dir))
sys.path.insert(0, str(_replication_dir))
sys.path.insert(0, str(_replication_dir / "scripts"))

from metrics import evaluate_all
from faithful.smartcommit_faithful import predict_smartcommit_faithful


def main():
    project_root = Path(__file__).resolve().parent.parent.parent  # replication/
    dataset_path = project_root / "data" / "lkp_untangle.json"
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"[smartcommit-faithful] {len(dataset)} ")

    predictions = []
    references = []
    levels = []
    per_case = []
    start_time = time.time()

    for i, entry in enumerate(dataset):
        try:
            pred = predict_smartcommit_faithful(entry)
        except Exception as e:
            n_lines = sum(1 for f in entry['ground_truth']['files']
                          for la in f['lines'] if la['type'] in ('add', 'del'))
            pred = [0] * n_lines
            if i < 10:
                print(f"  [WARN] entry {entry.get('id', i)} failed: {e}")

        true = [la['group_id'] for f in entry['ground_truth']['files']
                for la in f['lines'] if la['type'] in ('add', 'del')]

        if len(pred) < len(true):
            pred.extend([0] * (len(true) - len(pred)))
        elif len(pred) > len(true):
            pred = pred[:len(true)]

        predictions.append(pred)
        references.append(true)
        level = entry.get('entry_stats', {}).get('interleave_level', 'unknown')
        levels.append(level)
        per_case.append({
            'id': entry.get('id', i),
            'prediction': pred,
            'reference': true,
            'level': level,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(dataset)}] elapsed: {elapsed:.1f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"[smartcommit-faithful] {elapsed:.1f}s")

    results = evaluate_all(predictions, references, levels)

    output_dir = Path(__file__).resolve().parent
    with open(output_dir / "results_smartcommit.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(output_dir / "predictions_smartcommit.json", 'w', encoding='utf-8') as f:
        json.dump(per_case, f, ensure_ascii=False, indent=2)

    print(f"[smartcommit-faithful] ")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
