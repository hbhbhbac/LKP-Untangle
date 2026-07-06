"""
train_utango.py — Train UTANGO GCN Weight (proper backprop)

Train:
1. LoadData (balanced_N1500 N2000)
2. entry utango_faithful BuildGraph + Feature
3. 2-layer GCN with learnable W1, W2 embeddings
4. Pairwise cosine similarity → BCE loss (same_group=1, diff_group=0)
5. Update W1, W2
6. Evaluate lkp_untangle.json

Usage:
  python train_utango.py --data balanced_N1500.json --name n1500_trained
  python train_utango.py --data balanced_N2000.json --name n2000_trained
"""

from __future__ import annotations

import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from faithful.source_resolver import reconstruct_pre_post_from_diff
from faithful.ast_parser import parse_c_file, is_c_file, map_line_to_function
from faithful.utango_faithful import (
    _build_graph_faithful, _build_features_faithful,
    _cosine_similarity, _clone_postprocess
)
from faithful.train_hdgnn import convert_synthetic_entry


# --- Trainable GCN ---

class TrainableGCN:
    """2-layer GCN with learnable weight matrices (numpy, proper backprop)."""

    def __init__(self, hidden_dim=64):
        self.hidden_dim = hidden_dim
        self.W1 = None  # initialized on first forward with actual input_dim
        self.W2 = None
        self.input_dim = None

    def _init_weights(self, input_dim):
        self.input_dim = input_dim
        scale1 = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        self.W1 = np.random.randn(input_dim, self.hidden_dim) * scale1
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * scale2

    def _compute_adj_hat(self, adj):
        """Symmetric normalized adjacency + self-loop: D^{-1/2}(A+I)D^{-1/2}"""
        n = adj.shape[0]
        A = adj + np.eye(n)
        D = A.sum(axis=1)
        D[D == 0] = 1
        D_inv_sqrt = 1.0 / np.sqrt(D)
        # D^{-1/2} A D^{-1/2}
        A_hat = A * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
        return A_hat

    def forward(self, features, adj):
        """
        Forward: h1 = ReLU(A_hat @ X @ W1), h2 = A_hat @ h1 @ W2, output = L2_norm(h2)
        Caches intermediate values for backprop.
        """
        if self.W1 is None:
            self._init_weights(features.shape[1])
        elif features.shape[1] != self.input_dim:
            # Pad or truncate features to match
            if features.shape[1] < self.input_dim:
                pad = np.zeros((features.shape[0], self.input_dim - features.shape[1]))
                features = np.hstack([features, pad])
            else:
                features = features[:, :self.input_dim]

        self.A_hat = self._compute_adj_hat(adj)
        self.X = features

        # Layer 1
        self.Z1 = self.A_hat @ features @ self.W1  # [n, hidden]
        self.H1 = np.maximum(0, self.Z1)  # ReLU

        # Layer 2
        self.Z2 = self.A_hat @ self.H1 @ self.W2  # [n, hidden]

        # L2 normalize
        self.norms2 = np.linalg.norm(self.Z2, axis=1, keepdims=True)
        self.norms2[self.norms2 == 0] = 1
        self.H2 = self.Z2 / self.norms2

        return self.H2

    def backward(self, dH2):
        """
        Backprop through the 2-layer GCN.
        dH2: gradient of loss w.r.t. normalized output [n, hidden]
        Returns gradients for W1, W2.
        """
        # Backprop through L2 normalization
        # d/dZ2 of (Z2 / ||Z2||) = (I - h*h^T) / ||Z2|| per row
        n = self.Z2.shape[0]
        dZ2 = np.zeros_like(self.Z2)
        for i in range(n):
            norm = self.norms2[i, 0]
            h = self.H2[i:i+1].T  # [d, 1]
            # Jacobian: (I - h*h^T) / norm
            dZ2[i] = (dH2[i] - self.H2[i] * np.dot(dH2[i], self.H2[i])) / norm

        # Layer 2: Z2 = A_hat @ H1 @ W2
        # dW2 = (A_hat @ H1)^T @ dZ2
        AH1 = self.A_hat @ self.H1
        dW2 = AH1.T @ dZ2

        # dH1 = A_hat^T @ dZ2 @ W2^T
        dH1 = self.A_hat.T @ dZ2 @ self.W2.T

        # Backprop through ReLU
        dZ1 = dH1 * (self.Z1 > 0)

        # Layer 1: Z1 = A_hat @ X @ W1
        AX = self.A_hat @ self.X
        dW1 = AX.T @ dZ1

        return dW1, dW2

    def save(self, path):
        np.savez(path, W1=self.W1, W2=self.W2,
                 input_dim=self.input_dim, hidden_dim=self.hidden_dim)

    def load(self, path):
        data = np.load(path)
        self.W1 = data['W1']
        self.W2 = data['W2']
        self.input_dim = int(data['input_dim'])
        self.hidden_dim = int(data['hidden_dim'])


# --- Training utilities ---

def compute_pairwise_loss_and_grad(embeddings, labels, max_pairs=1000):
    """
    Compute pairwise contrastive loss (BCE on cosine similarity).
    Returns (loss, dH) where dH is gradient w.r.t embeddings.
    """
    n = embeddings.shape[0]
    sim_matrix = embeddings @ embeddings.T  # cosine sim (already L2-normed)

    # Sample pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    if len(pairs) > max_pairs:
        idx = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    eps = 1e-7
    total_loss = 0.0
    dH = np.zeros_like(embeddings)

    for i, j in pairs:
        sim = sim_matrix[i, j]
        target = 1.0 if labels[i] == labels[j] else 0.0

        # Sigmoid-scaled similarity for BCE
        # Map sim from [-1,1] to [0,1] via (sim+1)/2
        p = np.clip((sim + 1) / 2, eps, 1 - eps)
        loss = -(target * np.log(p) + (1 - target) * np.log(1 - p))
        total_loss += loss

        # Gradient of loss w.r.t. sim
        # dp/dsim = 0.5
        # dloss/dp = -target/p + (1-target)/(1-p)
        dloss_dp = -target / p + (1 - target) / (1 - p)
        dloss_dsim = dloss_dp * 0.5

        # dsim/dH[i] = H[j], dsim/dH[j] = H[i]
        dH[i] += dloss_dsim * embeddings[j]
        dH[j] += dloss_dsim * embeddings[i]

    n_pairs = len(pairs)
    total_loss /= n_pairs
    dH /= n_pairs

    return total_loss, dH


def prepare_training_data(synth_path, max_entries=2000):
    """Load synthetic data and extract (features, adj, labels) for each entry."""
    print(f"[train-utango] Loading: {synth_path}")
    with open(synth_path, encoding='utf-8') as f:
        synth_data = json.load(f)

    train_raw = [e for e in synth_data if e['num_atoms'] > 1 and e['ground_truth']]
    print(f"  Multi-atom entries: {len(train_raw)}")

    train_items = []  # (features, adj, labels)
    max_dim = 0
    t0 = time.time()

    for i, synth_entry in enumerate(train_raw[:max_entries]):
        entry = convert_synthetic_entry(synth_entry)
        try:
            diff_text = entry['source']['diff']
            file_sources = reconstruct_pre_post_from_diff(diff_text)
            parsed_files = {}
            for fs in file_sources:
                source = fs.post_content or fs.pre_content
                if source and is_c_file(fs.filepath):
                    parsed_files[fs.filepath] = parse_c_file(source)

            nodes, adj = _build_graph_faithful(diff_text, parsed_files)
            if len(nodes) < 2:
                continue

            features = _build_features_faithful(nodes)

            # Get ground truth labels
            gt_lines = [la['group_id'] for f in entry['ground_truth']['files']
                        for la in f['lines'] if la['type'] in ('add', 'del')]
            if len(gt_lines) < len(nodes):
                gt_lines.extend([0] * (len(nodes) - len(gt_lines)))
            labels = np.array(gt_lines[:len(nodes)])

            # Ensure at least 2 groups
            if len(set(labels)) < 2:
                continue

            train_items.append((features, adj, labels))
            max_dim = max(max_dim, features.shape[1])

        except Exception:
            pass

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{min(max_entries, len(train_raw))}] "
                  f"{len(train_items)} valid, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  Done: {len(train_items)} valid entries, max_dim={max_dim}, {elapsed:.1f}s")
    return train_items, max_dim


def train_gcn(train_items, max_dim, epochs=30, lr=0.005):
    """Train the GCN model."""
    gcn = TrainableGCN(hidden_dim=64)

    # Initialize with first entry's dimension (padded to max_dim)
    gcn._init_weights(max_dim)

    best_loss = float('inf')
    patience = 8
    no_improve = 0

    for epoch in range(epochs):
        np.random.shuffle(train_items)
        total_loss = 0.0
        n_samples = 0

        for features, adj, labels in train_items:
            # Pad features to max_dim
            if features.shape[1] < max_dim:
                pad = np.zeros((features.shape[0], max_dim - features.shape[1]))
                features = np.hstack([features, pad])
            elif features.shape[1] > max_dim:
                features = features[:, :max_dim]

            # Skip very large graphs for speed
            if features.shape[0] > 300:
                continue

            # Forward
            embeddings = gcn.forward(features, adj)

            # Compute loss and gradient
            loss, dH = compute_pairwise_loss_and_grad(
                embeddings, labels, max_pairs=800)

            # Backward
            dW1, dW2 = gcn.backward(dH)

            # Gradient clipping
            grad_norm1 = np.linalg.norm(dW1)
            grad_norm2 = np.linalg.norm(dW2)
            max_norm = 5.0
            if grad_norm1 > max_norm:
                dW1 = dW1 * max_norm / grad_norm1
            if grad_norm2 > max_norm:
                dW2 = dW2 * max_norm / grad_norm2

            # Update
            gcn.W1 -= lr * dW1
            gcn.W2 -= lr * dW2

            total_loss += loss
            n_samples += 1

        if n_samples == 0:
            continue

        avg_loss = total_loss / n_samples
        print(f"  Epoch {epoch}: loss={avg_loss:.4f} ({n_samples} samples)")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Reduce lr
        if epoch > 0 and epoch % 10 == 0:
            lr *= 0.7
            print(f"  LR reduced to {lr:.5f}")

    return gcn


def predict_utango_trained(entry, gcn):
    """Predict using trained GCN (same pipeline as utango_faithful but with learned weights)."""
    diff_text = entry['source']['diff']
    num_groups = entry['ground_truth']['num_groups']

    file_sources = reconstruct_pre_post_from_diff(diff_text)
    parsed_files = {}
    for fs in file_sources:
        source = fs.post_content or fs.pre_content
        if source and is_c_file(fs.filepath):
            parsed_files[fs.filepath] = parse_c_file(source)

    nodes, adj = _build_graph_faithful(diff_text, parsed_files)
    n = len(nodes)

    if n == 0:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    features = _build_features_faithful(nodes)

    # Pad/truncate to match model input_dim
    if features.shape[1] < gcn.input_dim:
        pad = np.zeros((features.shape[0], gcn.input_dim - features.shape[1]))
        features = np.hstack([features, pad])
    elif features.shape[1] > gcn.input_dim:
        features = features[:, :gcn.input_dim]

    # GCN forward (with learned weights)
    if n <= 500:
        embeddings = gcn.forward(features, adj)
    else:
        # For large graphs, use untrained propagation as fallback
        from faithful.utango_faithful import _gcn_propagate
        embeddings = _gcn_propagate(features, adj, rounds=2)

    sim = embeddings @ embeddings.T

    # Clustering
    if n > 50:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        dist = 1.0 - np.clip(sim, -1, 1)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        dist = np.clip(dist, 0, 2)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='average')
        labels = fcluster(Z, t=num_groups, criterion='maxclust')
        cluster_ids = [int(l - 1) for l in labels]
    else:
        from faithful.utango_faithful import _agglomerative_avg
        cluster_ids = _agglomerative_avg(sim, num_groups)

    cluster_ids = _clone_postprocess(nodes, cluster_ids)

    # Map to output lines
    result = []
    idx = 0
    fk = 'files' if 'files' in entry['ground_truth'] else 'file_assignments'
    for f in entry['ground_truth'][fk]:
        for la in f['lines']:
            if la['type'] in ('add', 'del'):
                if idx < len(cluster_ids):
                    result.append(cluster_ids[idx])
                else:
                    result.append(0)
                idx += 1

    return result


def evaluate_on_lkp(gcn, dataset_path, output_name):
    """Evaluate trained model on lkp_untangle.json"""
    print(f"\n[eval] Evaluating on {dataset_path.name}...")
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"  {len(dataset)} entries")

    from metrics import find_best_mapping

    all_acc_a = []
    all_acc_c = []
    errors = 0
    t0 = time.time()

    for i, entry in enumerate(dataset):
        try:
            pred = predict_utango_trained(entry, gcn)
        except Exception:
            n_lines = sum(1 for f in entry['ground_truth']['files']
                          for la in f['lines'] if la['type'] in ('add', 'del'))
            pred = [0] * n_lines
            errors += 1

        # Get ground truth
        gt = entry['ground_truth']
        fk = 'files' if 'files' in gt else 'file_assignments'
        true_labels = []
        n_context = 0
        for fi in gt[fk]:
            for la in fi['lines']:
                if la['type'] in ('add', 'del'):
                    true_labels.append(la['group_id'])
                elif la['type'] == 'context':
                    n_context += 1

        # Align lengths
        if len(pred) < len(true_labels):
            pred.extend([0] * (len(true_labels) - len(pred)))
        elif len(pred) > len(true_labels):
            pred = pred[:len(true_labels)]

        # Compute accuracy
        valid_pred, valid_true = [], []
        for p, t in zip(pred, true_labels):
            if t is not None:
                valid_pred.append(p)
                valid_true.append(t)

        total_changed = len(valid_true)
        if total_changed == 0:
            all_acc_a.append(1.0)
            all_acc_c.append(1.0)
            continue

        num_pred = len(set(p for p in valid_pred if p is not None))
        num_true = len(set(valid_true))
        mapping = find_best_mapping(valid_pred, valid_true, num_pred, num_true)
        correct = sum(1 for p, t in zip(valid_pred, valid_true)
                      if p is not None and mapping.get(p) == t)

        acc_c = correct / total_changed
        acc_a = (correct + n_context) / (total_changed + n_context) if (total_changed + n_context) > 0 else 1.0
        all_acc_a.append(acc_a)
        all_acc_c.append(acc_c)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(dataset)}] acc_c_mean={np.mean(all_acc_c):.4f}, "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    acc_a_mean = float(np.mean(all_acc_a))
    acc_c_mean = float(np.mean(all_acc_c))
    acc_c_median = float(np.median(all_acc_c))
    c100 = sum(1 for a in all_acc_c if a >= 1.0) / len(all_acc_c)
    c90 = sum(1 for a in all_acc_c if a >= 0.9) / len(all_acc_c)
    c80 = sum(1 for a in all_acc_c if a >= 0.8) / len(all_acc_c)

    results = {
        'method': f'utango_{output_name}',
        'dataset': dataset_path.name,
        'n_entries': len(dataset),
        'acc_a_mean': round(acc_a_mean, 4),
        'acc_a_median': round(float(np.median(all_acc_a)), 4),
        'acc_c_mean': round(acc_c_mean, 4),
        'acc_c_median': round(acc_c_median, 4),
        'acc_c@100': round(c100, 4),
        'acc_c@90': round(c90, 4),
        'acc_c@80': round(c80, 4),
        'errors': errors,
        'elapsed_s': round(elapsed, 1),
    }

    print(f"\n  Results ({output_name}):")
    print(f"    Acc_A (mean): {acc_a_mean:.4f}")
    print(f"    Acc_C (mean): {acc_c_mean:.4f}")
    print(f"    Acc_C (median): {acc_c_median:.4f}")
    print(f"    Acc_C@100: {c100:.4f}")
    print(f"    Acc_C@90: {c90:.4f}")
    print(f"    Acc_C@80: {c80:.4f}")
    print(f"    Errors: {errors}, Time: {elapsed:.1f}s")

    # Save results
    output_path = Path(__file__).resolve().parent / f"results_utango_{output_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='balanced_N1500.json',
                        help='Training data filename (in data/lkp_synthesize/balanced/)')
    parser.add_argument('--name', type=str, default='n1500_trained',
                        help='Model name suffix for output files')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--max-train', type=int, default=1500,
                        help='Max training entries to use')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    synth_path = project_root / "data" / "lkp_synthesize" / "balanced" / args.data
    eval_path = project_root / "data" / "lkp_manual" / "lkp_untangle.json"

    print(f"{'='*60}")
    print(f"  UTANGO Trained: {args.name}")
    print(f"  Training data: {synth_path.name}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Hidden: {args.hidden}")
    print(f"{'='*60}")

    # 1. Prepare training data
    t_start = time.time()
    train_items, max_dim = prepare_training_data(synth_path, max_entries=args.max_train)

    if not train_items:
        print("[ERROR] No valid training data!")
        return

    t_prep = time.time() - t_start
    print(f"\n  Data preparation: {t_prep:.1f}s")

    # 2. Train
    t_train_start = time.time()
    gcn = TrainableGCN(hidden_dim=args.hidden)
    gcn._init_weights(max_dim)
    gcn = train_gcn(train_items, max_dim, epochs=args.epochs, lr=args.lr)
    t_train = time.time() - t_train_start
    print(f"\n  Training: {t_train:.1f}s")

    # Save model
    model_path = Path(__file__).resolve().parent / f"utango_model_{args.name}.npz"
    gcn.save(str(model_path))
    print(f"  Model saved: {model_path}")

    # 3. Evaluate on LKP-Untangle
    results = evaluate_on_lkp(gcn, eval_path, args.name)

    # Total time
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total time: {total:.1f}s")
    print(f"    Data prep: {t_prep:.1f}s")
    print(f"    Training: {t_train:.1f}s")
    print(f"    Evaluation: {results['elapsed_s']:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
