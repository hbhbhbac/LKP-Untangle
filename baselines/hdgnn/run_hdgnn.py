"""
run_hdgnn.py — HD-GNN faithful

 HD-GNN
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# metrics faithful
_baselines_dir = Path(__file__).resolve().parent.parent
_replication_dir = _baselines_dir.parent
sys.path.insert(0, str(_baselines_dir))
sys.path.insert(0, str(_replication_dir))
sys.path.insert(0, str(_replication_dir / "scripts"))

from metrics import evaluate_all
from faithful.hdgnn_faithful import predict_hdgnn_faithful, set_repo_path


def main():
    project_root = Path(__file__).resolve().parent.parent.parent  # replication/
    dataset_path = project_root / "data" / "lkp_untangle.json"

    # Linux bare repo
    repo_path = project_root / "data" / "subjects" / "linux.git"
    if repo_path.exists():
        print(f"[hdgnn-faithful] full-project entity graph (repo: {repo_path})")
        set_repo_path(repo_path)
    else:
        print("[hdgnn-faithful] git repo diff-only entity graph (fallback)")

    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"[hdgnn-faithful] {len(dataset)} ")

    predictions = []
    references = []
    levels = []
    per_case = []

    start_time = time.time()

    for i, entry in enumerate(dataset):
        try:
            pred = predict_hdgnn_faithful(entry)
        except Exception as e:
            # Fallback:
            n_lines = sum(1 for f in entry['ground_truth']['files']
                          for la in f['lines'] if la['type'] in ('add', 'del'))
            pred = [0] * n_lines
            if i < 5:
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
    print(f"[hdgnn-faithful] {elapsed:.1f}s")

    results = evaluate_all(predictions, references, levels)

    output_dir = Path(__file__).resolve().parent
    results_file = output_dir / "results_hdgnn.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    predictions_file = output_dir / "predictions_hdgnn.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(per_case, f, ensure_ascii=False, indent=2)

    print(f"[hdgnn-faithful] : {results_file}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
