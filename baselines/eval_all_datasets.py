"""
eval_all_datasets.py — Evaluate baseline methods on Java and C# datasets

Each baseline method is run on the test split of Java and C# datasets.
HD-GNN uses per-dataset trained models.

Output: results_csharp_{method}.json, results_java_{method}.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_baselines_dir = Path(__file__).resolve().parent
_replication_dir = _baselines_dir.parent
sys.path.insert(0, str(_baselines_dir))
sys.path.insert(0, str(_replication_dir))
sys.path.insert(0, str(_replication_dir / "scripts"))

from metrics import evaluate_all

from faithful.barnett_faithful import predict_barnett_faithful
from faithful.herzig_faithful import predict_herzig_faithful
from faithful.flexeme_faithful import predict_flexeme_faithful
from faithful.utango_faithful import predict_utango_faithful
from faithful.hdgnn_faithful import predict_hdgnn_faithful
from faithful.smartcommit_faithful import predict_smartcommit_faithful

PROJECT_ROOT = _replication_dir
OUTPUT_DIR = Path(__file__).resolve().parent

PREDICT_FNS = {
    'barnett': predict_barnett_faithful,
    'herzig': predict_herzig_faithful,
    'flexeme': predict_flexeme_faithful,
    'utango': predict_utango_faithful,
    'hdgnn': predict_hdgnn_faithful,
    'smartcommit': predict_smartcommit_faithful,
}


def load_dataset(name: str) -> List[Dict]:
    """Load dataset from data/ directory."""
    if name == "csharp":
        path = PROJECT_ROOT / "data" / "csharp_dataset.json"
    elif name == "java":
        path = PROJECT_ROOT / "data" / "java_dataset.json"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_predict_fn(method: str, dataset_name: str = None) -> Callable:
    """Get baseline predict function. For HD-GNN, loads per-dataset trained model."""
    if method == 'hdgnn' and dataset_name:
        model_path = OUTPUT_DIR / "hdgnn" / f"hdgnn_{dataset_name}_model.pt"
        if model_path.exists():
            try:
                import torch
                from hdgnn.hdgnn_source import HDGNNModel, predict_hdgnn_source
                device = torch.device("cpu")
                model = HDGNNModel()
                checkpoint = torch.load(str(model_path), map_location=device)
                state = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state)
                model.to(device).eval()
                return lambda entry, _m=model, _d=device: predict_hdgnn_source(entry, _m, _d)
            except Exception:
                pass
    return PREDICT_FNS[method]


def run_single_eval(
    method: str, dataset_name: str, dataset: List[Dict], predict_fn: Callable
) -> Dict:
    """Method+DataRunEvaluate"""
    predictions = []
    references = []
    errors = 0
    start_time = time.time()

    for i, entry in enumerate(dataset):
        try:
            pred = predict_fn(entry)
        except Exception:
            n_lines = sum(
                1 for f in entry['ground_truth']['files']
                for la in f['lines'] if la['type'] in ('add', 'del')
            )
            pred = [0] * n_lines
            errors += 1

        true = [
            la['group_id'] for f in entry['ground_truth']['files']
            for la in f['lines'] if la['type'] in ('add', 'del')
        ]

        # Align lengths
        if len(pred) < len(true):
            pred.extend([0] * (len(true) - len(pred)))
        elif len(pred) > len(true):
            pred = pred[:len(true)]

        predictions.append(pred)
        references.append(true)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"    [{i+1}/{len(dataset)}] {elapsed:.0f}s, errors={errors}")

    elapsed = time.time() - start_time
    results = evaluate_all(predictions, references)
    results['errors'] = errors
    results['elapsed_s'] = round(elapsed, 1)

    # Per-repo breakdown
    repo_preds = defaultdict(list)
    repo_refs = defaultdict(list)
    for entry, pred, ref in zip(dataset, predictions, references):
        repo = entry.get('metadata', {}).get('repo', 'unknown')
        repo_preds[repo].append(pred)
        repo_refs[repo].append(ref)

    results['by_repo'] = {}
    for repo in sorted(repo_preds.keys()):
        r = evaluate_all(repo_preds[repo], repo_refs[repo])
        results['by_repo'][repo] = {
            'count': r['total_cases'],
            'accuracy_a_mean': r['accuracy_a_mean'],
            'accuracy_c_100': r['accuracy_c_100'],
        }

    return results


def main():
    all_methods = ['barnett', 'herzig', 'flexeme', 'utango', 'hdgnn', 'smartcommit']
    all_datasets_names = ['csharp', 'java']

    # Parse args: support --method and --dataset flags
    datasets = []
    methods = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--method' and i + 1 < len(args):
            methods.append(args[i+1])
            i += 2
        elif args[i] == '--dataset' and i + 1 < len(args):
            datasets.append(args[i+1])
            i += 2
        elif args[i] in all_datasets_names:
            datasets.append(args[i])
            i += 1
        elif args[i] in all_methods:
            methods.append(args[i])
            i += 1
        else:
            i += 1

    if not datasets:
        datasets = all_datasets_names
    if not methods:
        methods = all_methods

    print("[eval_all_datasets] Multi-Dataset Baseline Evaluation")
    print("=" * 70)

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        dataset = load_dataset(ds_name)
        print(f"  Loaded {len(dataset)} entries")

        for method in methods:
            print(f"\n  [{method}] Running on {ds_name}...")
            try:
                predict_fn = load_predict_fn(method, ds_name)
            except Exception as e:
                print(f"    [ERROR] Failed to load {method}: {e}")
                continue

            results = run_single_eval(method, ds_name, dataset, predict_fn)

            # Save individual results
            out_path = OUTPUT_DIR / f"results_{ds_name}_{method}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            all_results[f"{ds_name}_{method}"] = results

            print(f"    acc_a={results['accuracy_a_mean']:.4f}, "
                  f"acc_c100={results['accuracy_c_100']:.4f}, "
                  f"errors={results['errors']}, "
                  f"time={results['elapsed_s']}s")

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<12s}", end="")
    for ds in datasets:
        print(f"  {ds+' acc_a':>14s} {ds+' acc_c100':>14s}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in datasets:
        print(f"  {'-'*14} {'-'*14}", end="")
    print()

    for method in methods:
        print(f"  {method:<12s}", end="")
        for ds in datasets:
            key = f"{ds}_{method}"
            if key in all_results:
                r = all_results[key]
                print(f"  {r['accuracy_a_mean']:>14.4f} {r['accuracy_c_100']:>14.4f}", end="")
            else:
                print(f"  {'N/A':>14s} {'N/A':>14s}", end="")
        print()

    print(f"\n[eval_all_datasets] Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
