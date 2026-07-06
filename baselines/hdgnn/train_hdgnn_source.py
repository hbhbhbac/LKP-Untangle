"""
train_hdgnn_source.py — Train full HD-GNN (paper-faithful GNN architecture)

Trains the hierarchical GNN end-to-end on entity reference graphs.
Evaluates on all 3 datasets with cross-dataset evaluation.

Usage:
  python train_hdgnn_source.py --dataset java --name hdgnn_source_java
  python train_hdgnn_source.py --dataset csharp --name hdgnn_source_csharp
  python train_hdgnn_source.py --dataset lkp --name hdgnn_source_lkp

Paper hyperparameters (Section 5.3):
  Epochs: 100, Batch: 50, LR: 0.003, Embed: 64, Hidden: 128, Iters: 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # replication/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))  # scripts/

from faithful.hdgnn_source import (
    HDGNNModel, build_graph_from_entry, predict_hdgnn_source,
)
from metrics import accuracy_a, accuracy_c, evaluate_all, find_best_mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # replication/
OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    'java': {
        'train_path': PROJECT_ROOT / 'data' / 'java_dataset.json',
        'train_split': 0.8,
        'split_seed': 42,
        'is_synthetic': False,
    },
    'csharp': {
        'train_path': PROJECT_ROOT / 'data' / 'csharp_dataset.json',
        'train_split': 0.8,
        'split_seed': 42,
        'is_synthetic': False,
    },
    'lkp': {
        'train_path': PROJECT_ROOT / 'data' / 'lkp_synth_sample_2k.json',
        'eval_path': PROJECT_ROOT / 'data' / 'lkp_untangle.json',
        'train_split': None,  # use all for training, separate eval set
        'split_seed': None,
        'is_synthetic': True,
    },
}

EVAL_DATASETS = {
    'java': {
        'path': PROJECT_ROOT / 'data' / 'java_dataset.json',
        'is_synthetic': False,
    },
    'csharp': {
        'path': PROJECT_ROOT / 'data' / 'csharp_dataset.json',
        'is_synthetic': False,
    },
    'lkp_manual': {
        'path': PROJECT_ROOT / 'data' / 'lkp_untangle.json',
        'is_synthetic': False,
    },
}


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_dataset(config: dict) -> tuple:
    """Load and split dataset into train/val."""
    path = config['train_path']
    print(f"[data] Loading {path.name}...")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Total entries: {len(data)}")

    is_synthetic = config.get('is_synthetic', False)

    if config.get('train_split') is not None:
        # Split into train/val
        seed = config['split_seed']
        rng = random.Random(seed)
        indices = list(range(len(data)))
        rng.shuffle(indices)
        split_idx = int(len(data) * config['train_split'])
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        print(f"  Train: {len(train_data)}, Val: {len(val_data)} (seed={seed})")
    else:
        # Use all for training; separate eval set
        train_data = data
        val_data = None
        if config.get('eval_path'):
            print(f"[data] Loading eval set from {config['eval_path'].name}...")
            with open(config['eval_path'], encoding='utf-8') as f:
                val_data = json.load(f)
            print(f"  Eval set: {len(val_data)} entries")

    return train_data, val_data, is_synthetic


# ---------------------------------------------------------------------------
# Graph caching for training
# ---------------------------------------------------------------------------

def precompute_graphs(data: list, is_synthetic: bool, max_entries: int = None,
                      desc: str = "train") -> list:
    """Build graphs for all entries (skip failures)."""
    graphs = []
    errors = 0
    t0 = time.time()
    n = len(data) if max_entries is None else min(len(data), max_entries)

    for i in range(n):
        entry = data[i]
        try:
            g = build_graph_from_entry(entry, is_synthetic=is_synthetic)
            if g is not None and g['pair_labels'] is not None:
                graphs.append(g)
        except Exception:
            errors += 1

        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(f"  [{desc}] {i+1}/{n} graphs built, "
                  f"{len(graphs)} valid, {errors} errors ({elapsed:.1f}s)")

    return graphs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model: HDGNNModel, train_graphs: list, val_graphs: list,
                device: torch.device, epochs: int = 100, batch_size: int = 50,
                lr: float = 0.003, patience: int = 10) -> dict:
    """
    Train HD-GNN with gradient accumulation over variable-size graphs.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.to(device)
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n[train] Starting training: {epochs} epochs, batch_size={batch_size}, "
          f"lr={lr}, device={device}")
    print(f"  Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs) if val_graphs else 0}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        optimizer.zero_grad()

        # Shuffle training data
        indices = list(range(len(train_graphs)))
        random.shuffle(indices)

        for step, idx in enumerate(indices):
            g = train_graphs[idx]

            # Move tensors to device
            gd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in g.items()}

            try:
                pair_probs = model(gd)  # [n_pairs, 2]
                # P(same_concern) = pair_probs[:, 0]
                pred_same = pair_probs[:, 0]
                labels = gd['pair_labels']

                loss = criterion(pred_same, labels)
                # Scale loss for gradient accumulation
                loss = loss / batch_size
                loss.backward()

                epoch_loss += loss.item() * batch_size
                n_samples += 1

            except Exception:
                continue

            # Step optimizer every batch_size entries
            if (step + 1) % batch_size == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Final step for remaining
        if n_samples % batch_size != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / max(n_samples, 1)
        history['train_loss'].append(avg_train_loss)

        # Validation
        val_loss = _compute_val_loss(model, val_graphs, device, criterion) if val_graphs else avg_train_loss
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best={best_epoch}, val_loss={best_val_loss:.4f})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print(f"  Training complete. Best epoch: {best_epoch}, "
          f"best val_loss: {best_val_loss:.4f}")

    return history


def _compute_val_loss(model: HDGNNModel, val_graphs: list,
                      device: torch.device, criterion: nn.Module) -> float:
    """Compute average loss on validation set."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for g in val_graphs:
            gd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in g.items()}
            try:
                pair_probs = model(gd)
                pred_same = pair_probs[:, 0]
                labels = gd['pair_labels']
                loss = criterion(pred_same, labels)
                total_loss += loss.item()
                n_samples += 1
            except Exception:
                continue

    return total_loss / max(n_samples, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_entries(model: HDGNNModel, dataset: list,
                        dataset_name: str, device: torch.device,
                        max_eval: int = None, is_synthetic: bool = False) -> dict:
    """Evaluate trained model on a list of entries (already in memory)."""
    print(f"\n[eval] {dataset_name} ({len(dataset)} entries)...")

    if max_eval and len(dataset) > max_eval:
        dataset = dataset[:max_eval]

    return _evaluate_entries(model, dataset, dataset_name, device, is_synthetic)


def evaluate_on_dataset(model: HDGNNModel, dataset_path: Path,
                        dataset_name: str, device: torch.device,
                        max_eval: int = None, is_synthetic: bool = False) -> dict:
    """Evaluate trained model on a dataset file."""
    print(f"\n[eval] {dataset_name} ({dataset_path.name})...")
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)

    if max_eval and len(dataset) > max_eval:
        dataset = dataset[:max_eval]

    return _evaluate_entries(model, dataset, dataset_name, device, is_synthetic)


def _evaluate_entries(model: HDGNNModel, dataset: list,
                      dataset_name: str, device: torch.device,
                      is_synthetic: bool = False) -> dict:
    """Core evaluation logic on a list of entries."""
    print(f"  Evaluating {len(dataset)} entries...")
    predictions = []
    references = []
    errors = 0
    t0 = time.time()

    for i, entry in enumerate(dataset):
        gt = entry.get('ground_truth', {})

        if is_synthetic:
            from faithful.train_hdgnn import convert_synthetic_entry
            entry_std = convert_synthetic_entry(entry)
            gt = entry_std['ground_truth']
            entry_for_pred = entry_std
        else:
            entry_for_pred = entry

        fk = 'files' if 'files' in gt else 'file_assignments'
        if fk not in gt:
            continue

        true_labels = []
        for fi in gt[fk]:
            for la in fi['lines']:
                if la['type'] in ('add', 'del'):
                    true_labels.append(la.get('group_id', 0))

        if not true_labels:
            continue

        try:
            pred = predict_hdgnn_source(entry_for_pred, model, device)
        except Exception:
            pred = [0] * len(true_labels)
            errors += 1

        # Align lengths
        if len(pred) < len(true_labels):
            pred.extend([0] * (len(true_labels) - len(pred)))
        elif len(pred) > len(true_labels):
            pred = pred[:len(true_labels)]

        predictions.append(pred)
        references.append(true_labels)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(dataset)} ({elapsed:.1f}s, {errors} errors)")

    elapsed = time.time() - t0
    print(f"  Done: {len(predictions)} evaluated, {errors} errors ({elapsed:.1f}s)")

    if not predictions:
        return {'error': 'no valid predictions'}

    results = evaluate_all(predictions, references)
    results['errors'] = errors
    results['elapsed_sec'] = round(elapsed, 1)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train HD-GNN (paper-faithful GNN)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['java', 'csharp', 'lkp'],
                        help='Dataset to train on')
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name for output files')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-iters', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-train', type=int, default=None,
                        help='Max training entries (for quick testing)')
    parser.add_argument('--max-eval', type=int, default=None,
                        help='Max eval entries per dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cross-eval', action='store_true',
                        help='Skip cross-dataset evaluation')
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[config] Device: {device}")
    print(f"[config] Dataset: {args.dataset}")
    print(f"[config] Hyperparams: embed={args.embed_dim}, hidden={args.hidden_dim}, "
          f"iters={args.n_iters}, lr={args.lr}, batch={args.batch_size}")

    # Load data
    config = DATASET_CONFIG[args.dataset]
    train_data, val_data, is_synthetic = load_dataset(config)

    # Build graphs
    print("\n[graph] Building training graphs...")
    train_graphs = precompute_graphs(
        train_data, is_synthetic, max_entries=args.max_train, desc="train"
    )

    val_graphs = None
    if val_data:
        print("[graph] Building validation graphs...")
        # For lkp: val is standard format (not synthetic)
        val_is_synthetic = False if args.dataset == 'lkp' else is_synthetic
        val_graphs = precompute_graphs(
            val_data, val_is_synthetic, max_entries=args.max_eval, desc="val"
        )

    if not train_graphs:
        print("[ERROR] No valid training graphs. Exiting.")
        sys.exit(1)

    # Create model
    model = HDGNNModel(
        n_entity_types=5,
        n_rel_types=3,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_iters=args.n_iters,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] HDGNNModel: {n_params:,} parameters")

    # Train
    history = train_model(
        model, train_graphs, val_graphs, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
    )

    # Save model
    model_path = OUTPUT_DIR / f'{args.name}_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'history': history,
    }, model_path)
    print(f"\n[save] Model saved to {model_path}")

    # Evaluate
    results = {'config': vars(args), 'history': history, 'eval': {}}

    # Evaluate on own validation set
    if args.dataset == 'lkp':
        eval_path = config['eval_path']
        r = evaluate_on_dataset(model, eval_path, 'lkp_manual', device,
                                max_eval=args.max_eval)
        results['eval']['lkp_manual'] = r
        print(f"  → Acc_A={r.get('accuracy_a_mean',0):.4f}, "
              f"Acc_C={r.get('accuracy_c_100',0):.4f}")
    else:
        # Eval on own val split directly (already loaded)
        if val_data:
            r = evaluate_on_entries(model, val_data, f'{args.dataset}_val', device,
                                    max_eval=args.max_eval)
            results['eval'][f'{args.dataset}_val'] = r
            print(f"  → Acc_A={r.get('accuracy_a_mean',0):.4f}, "
                  f"Acc_C={r.get('accuracy_c_100',0):.4f}")

    # Cross-dataset evaluation
    if not args.no_cross_eval:
        print("\n[cross-eval] Cross-dataset evaluation...")
        for eval_name, eval_cfg in EVAL_DATASETS.items():
            if eval_name == args.dataset:
                continue
            r = evaluate_on_dataset(
                model, eval_cfg['path'], eval_name, device,
                max_eval=args.max_eval or 500,
                is_synthetic=eval_cfg.get('is_synthetic', False),
            )
            results['eval'][eval_name] = r
            print(f"  {eval_name}: Acc_A={r.get('accuracy_a_mean',0):.4f}, "
                  f"Acc_C={r.get('accuracy_c_100',0):.4f}")

    # Save results
    results_path = OUTPUT_DIR / f'{args.name}_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[save] Results saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  SUMMARY: {args.name}")
    print("=" * 60)
    for dname, dres in results['eval'].items():
        if 'error' in dres:
            print(f"  {dname:15s}: ERROR")
        else:
            print(f"  {dname:15s}: Acc_A={dres.get('accuracy_a_mean',0):.4f}  "
                  f"Acc_C={dres.get('accuracy_c_100',0):.4f}  "
                  f"Acc_C90={dres.get('accuracy_c_90',0):.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

