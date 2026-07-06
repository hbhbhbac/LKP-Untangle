"""
train_hdgnn.py — HD-GNN edge classifier

1. (balanced_N2000.json)
2. entry edge features ( hdgnn_faithful )
3. pairwise (same_group=1, diff_group=0)
4. MLP edge classifier
5.
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from faithful.source_resolver import reconstruct_pre_post_from_diff
from faithful.ast_parser import parse_c_file, is_c_file
from faithful.entity_extractor import extract_entities
from faithful.hdgnn_faithful import (
    extract_code_changes_faithful, compute_node_features,
    compute_edge_scores, _jaccard, cluster_from_scores,
    _connected_components, _force_merge, _force_split
)


def convert_synthetic_entry(synth_entry: dict) -> dict:
    """ faithful """
    diff_text = synth_entry['tangled_diff']
    assignments = synth_entry['ground_truth']['assignments']
    num_groups = synth_entry['ground_truth']['num_groups']

    # files
    files_dict = defaultdict(list)
    current_file = None

    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            parts = line.split(' b/')
            current_file = parts[-1].strip() if len(parts) >= 2 else ''
        elif (line.startswith('+') and not line.startswith('+++')) or \
             (line.startswith('-') and not line.startswith('---')):
            if current_file:
                files_dict[current_file].append(line)

    # assignments diff
    assign_idx = 0
    file_lines = defaultdict(list)

    current_file = None
    for line in diff_text.splitlines():
        if line.startswith('diff --git'):
            parts = line.split(' b/')
            current_file = parts[-1].strip() if len(parts) >= 2 else ''
            continue
        if line.startswith('+++ b/'):
            current_file = line[6:].strip()
            continue

        if (line.startswith('+') and not line.startswith('+++')) or \
           (line.startswith('-') and not line.startswith('---')):
            line_type = 'add' if line.startswith('+') else 'del'
            # assignment
            group_id = 0
            if assign_idx < len(assignments):
                group_id = assignments[assign_idx].get('group_id', 0)
                assign_idx += 1
            file_lines[current_file].append({
                'type': line_type,
                'content': line,
                'group_id': group_id,
            })

    # ground_truth
    files = []
    for filepath, lines in file_lines.items():
        files.append({
            'file': filepath,
            'lines': lines,
        })

    return {
        'id': synth_entry.get('series_id', 0),
        'source': {'diff': diff_text},
        'ground_truth': {
            'num_groups': num_groups,
            'files': files,
        },
    }


def extract_edge_features(entry: dict) -> tuple:
    """
     entry pairwise edge features
     (features_matrix, labels_vector)
    features: [n_pairs, n_features]
    labels: [n_pairs] (1=same group, 0=different group)
    """
    diff_text = entry['source']['diff']
    file_sources = reconstruct_pre_post_from_diff(diff_text)
    entity_graph = extract_entities(file_sources)
    changes = extract_code_changes_faithful(diff_text, file_sources, entity_graph)

    if len(changes) < 2:
        return None, None

    node_features = compute_node_features(changes, entity_graph)
    n = len(changes)

    # ground truth group assignments (hunk-level)
    # line-level hunk-level
    hunk_groups = _get_hunk_groups(entry, diff_text, n)
    if hunk_groups is None:
        return None, None

    # pairwise features
    features_list = []
    labels_list = []

    # entity graph
    ref_pairs = set()
    for ref in entity_graph.references:
        ref_pairs.add((ref.src, ref.dst))
        ref_pairs.add((ref.dst, ref.src))

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = changes[i], changes[j]

            # 7 edge features ( compute_edge_scores )
            entity_sim = _jaccard(ci['entities'], cj['entities'])
            ast_entity_sim = _jaccard(ci['ast_entities'], cj['ast_entities'])
            call_sim = _jaccard(ci['func_calls'], cj['func_calls'])
            same_file = 1.0 if ci['file'] == cj['file'] else 0.0
            same_func = 1.0 if (ci['func'] and ci['func'] == cj['func']) else 0.0
            cos_sim = float(np.dot(node_features[i], node_features[j])) if node_features.shape[1] > 0 else 0.0
            shared_refs = ci['ast_entities'] & cj['ast_entities']
            ref_score = min(len(shared_refs) / 3.0, 1.0) if shared_refs else 0.0

            features_list.append([
                entity_sim, ast_entity_sim, call_sim,
                same_file, same_func, cos_sim, ref_score
            ])

            # Label: 1 if same group, 0 if different
            label = 1 if hunk_groups[i] == hunk_groups[j] else 0
            labels_list.append(label)

    if not features_list:
        return None, None

    return np.array(features_list), np.array(labels_list)


def _get_hunk_groups(entry: dict, diff_text: str, n_hunks: int) -> list:
    """ ground_truth hunk-level group assignments"""
    # hunk group: hunk majority group
    hunk_groups = []
    hunk_idx = 0
    current_hunk_groups = []

    import re
    hunk_re = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@')

    gt_lines = []
    for f in entry['ground_truth']['files']:
        for la in f['lines']:
            if la['type'] in ('add', 'del'):
                gt_lines.append(la['group_id'])

    line_idx = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
            continue

        if hunk_re.match(line):
            if in_hunk and current_hunk_groups:
                # Majority vote for this hunk
                from collections import Counter
                most_common = Counter(current_hunk_groups).most_common(1)[0][0]
                hunk_groups.append(most_common)
            current_hunk_groups = []
            in_hunk = True
            continue

        if in_hunk:
            if (line.startswith('+') and not line.startswith('+++')) or \
               (line.startswith('-') and not line.startswith('---')):
                if line_idx < len(gt_lines):
                    current_hunk_groups.append(gt_lines[line_idx])
                    line_idx += 1

    # Last hunk
    if current_hunk_groups:
        from collections import Counter
        most_common = Counter(current_hunk_groups).most_common(1)[0][0]
        hunk_groups.append(most_common)

    if len(hunk_groups) != n_hunks:
        # Fallback:
        if len(hunk_groups) > n_hunks:
            hunk_groups = hunk_groups[:n_hunks]
        else:
            hunk_groups.extend([0] * (n_hunks - len(hunk_groups)))

    return hunk_groups


class MLPEdgeClassifier:
    """ MLP edge classifier (numpy PyTorch)"""

    def __init__(self, input_dim=7, hidden_dim=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Xavier
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(1)

    def forward(self, X):
        """"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = 1.0 / (1.0 + np.exp(-np.clip(self.z3, -500, 500)))  # Sigmoid
        return self.out.flatten()

    def predict(self, X):
        return self.forward(X)

    def train(self, X, y, epochs=100, lr=0.01, batch_size=256):
        """Mini-batch SGD """
        n = len(X)
        best_loss = float('inf')
        patience = 10
        no_improve = 0

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n)
            total_loss = 0.0

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx].reshape(-1, 1)

                # Forward
                pred = self.forward(X_batch).reshape(-1, 1)

                # Binary cross-entropy loss
                eps = 1e-7
                loss = -np.mean(y_batch * np.log(pred + eps) +
                                (1 - y_batch) * np.log(1 - pred + eps))
                total_loss += loss * len(batch_idx)

                # Backward (gradient computation)
                dz3 = pred - y_batch  # [batch, 1]
                dW3 = self.a2.T @ dz3 / len(batch_idx)
                db3 = np.mean(dz3, axis=0)

                da2 = dz3 @ self.W3.T
                dz2 = da2 * (self.z2 > 0)
                dW2 = self.a1.T @ dz2 / len(batch_idx)
                db2 = np.mean(dz2, axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (self.z1 > 0)
                dW1 = X_batch.T @ dz1 / len(batch_idx)
                db1 = np.mean(dz1, axis=0)

                # Update
                self.W3 -= lr * dW3
                self.b3 -= lr * db3
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            avg_loss = total_loss / n
            if epoch % 20 == 0:
                print(f'    Epoch {epoch}: loss={avg_loss:.4f}')

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']


def predict_with_trained_model(entry: dict, model: MLPEdgeClassifier) -> list:
    """"""
    diff_text = entry['source']['diff']
    num_groups = entry['ground_truth']['num_groups']
    file_sources = reconstruct_pre_post_from_diff(diff_text)
    entity_graph = extract_entities(file_sources)
    changes = extract_code_changes_faithful(diff_text, file_sources, entity_graph)

    if not changes:
        n_lines = sum(1 for f in entry['ground_truth']['files']
                      for la in f['lines'] if la['type'] in ('add', 'del'))
        return [0] * n_lines

    node_features = compute_node_features(changes, entity_graph)
    n = len(changes)

    if n <= num_groups:
        change_clusters = list(range(n))
    else:
        # edge scores
        ref_pairs = set()
        for ref in entity_graph.references:
            ref_pairs.add((ref.src, ref.dst))

        scores = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = changes[i], changes[j]
                feat = np.array([[
                    _jaccard(ci['entities'], cj['entities']),
                    _jaccard(ci['ast_entities'], cj['ast_entities']),
                    _jaccard(ci['func_calls'], cj['func_calls']),
                    1.0 if ci['file'] == cj['file'] else 0.0,
                    1.0 if (ci['func'] and ci['func'] == cj['func']) else 0.0,
                    float(np.dot(node_features[i], node_features[j])) if node_features.shape[1] > 0 else 0.0,
                    min(len(ci['ast_entities'] & cj['ast_entities']) / 3.0, 1.0),
                ]])
                score = model.predict(feat)[0]
                scores[i, j] = score
                scores[j, i] = score

        change_clusters = cluster_from_scores(scores, num_groups)

    # Map to lines
    from faithful.hdgnn_faithful import _map_to_lines
    return _map_to_lines(diff_text, change_clusters)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent

    # ()
    synth_path = project_root / "data" / "lkp_synthesize" / "balanced" / "balanced_N2000.json"
    print(f"[train-hdgnn] : {synth_path}")
    with open(synth_path, encoding='utf-8') as f:
        synth_data = json.load(f)

    # multi-atom entries ( ground truth)
    train_entries = [e for e in synth_data if e['num_atoms'] > 1 and e['ground_truth']]
    print(f"[train-hdgnn] : {len(train_entries)} (multi-atom)")

    print("[train-hdgnn] edge features...")
    all_features = []
    all_labels = []
    t0 = time.time()

    for i, synth_entry in enumerate(train_entries[:2000]):
        entry = convert_synthetic_entry(synth_entry)
        try:
            feats, labels = extract_edge_features(entry)
            if feats is not None and len(feats) > 0:
                all_features.append(feats)
                all_labels.append(labels)
        except Exception as e:
            if i < 3:
                print(f"  WARN: entry {i} failed: {e}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/2000] {len(all_features)} valid, elapsed: {elapsed:.1f}s")

    if not all_features:
        print("[ERROR] No valid training data")
        return

    X_train = np.vstack(all_features)
    y_train = np.concatenate(all_labels)
    print(f"[train-hdgnn] : {X_train.shape[0]} pairs, {X_train.shape[1]} features")
    print(f"  Positive ratio: {y_train.mean():.3f}")

    # MLP
    print("[train-hdgnn] MLP edge classifier...")
    model = MLPEdgeClassifier(input_dim=X_train.shape[1], hidden_dim=32)
    model.train(X_train, y_train, epochs=100, lr=0.01)

    model_path = Path(__file__).resolve().parent / "hdgnn_model.npz"
    model.save(str(model_path))
    print(f"[train-hdgnn] : {model_path}")

    print("\n[train-hdgnn] ...")
    stats_path = project_root / "dataset_with_stats.json"
    with open(stats_path, encoding='utf-8') as f:
        raw = json.load(f)
    val_dataset = raw['data']

    from metrics import evaluate_all

    predictions = []
    references = []
    levels = []

    for i, entry in enumerate(val_dataset):
        try:
            pred = predict_with_trained_model(entry, model)
        except Exception as e:
            n_lines = sum(1 for f in entry['ground_truth']['files']
                          for la in f['lines'] if la['type'] in ('add', 'del'))
            pred = [0] * n_lines

        true = [la['group_id'] for f in entry['ground_truth']['files']
                for la in f['lines'] if la['type'] in ('add', 'del')]

        if len(pred) < len(true):
            pred.extend([0] * (len(true) - len(pred)))
        elif len(pred) > len(true):
            pred = pred[:len(true)]

        predictions.append(pred)
        references.append(true)
        levels.append(entry.get('entry_stats', {}).get('interleave_level', 'unknown'))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(val_dataset)}]", flush=True)

    results = evaluate_all(predictions, references, levels)

    output_dir = Path(__file__).resolve().parent
    with open(output_dir / "results_hdgnn_trained.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[train-hdgnn] :")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
