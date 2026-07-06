# Baseline Methods

This document describes the baseline methods evaluated in the paper and how to obtain/run them.

## Program-Analysis and Heuristic Methods

### Barnett (ICSE 2015)
- **Paper**: "Helping Developers Help Themselves: Automatic Decomposition of Code Review Changesets"
- **Approach**: Dependency-based clustering using program slicing. Groups changed lines based on data and control dependencies, then merges groups that share dependencies.
- **Key assumption**: Assumes access to program dependency information extractable from the diff context.

### Herzig (ESE 2016)
- **Paper**: "The Impact of Tangled Code Changes on Defect Prediction Models"
- **Approach**: Confidence-voter framework that combines multiple heuristic signals (file proximity, temporal locality, author patterns) to determine whether changed lines belong to the same concern.
- **Key assumption**: Uses confidence thresholds to determine cluster boundaries.

## Graph and Learning-Based Methods

### Flexeme (FSE 2020)
- **Paper**: "Flexeme: Untangling Commits Using Lexical Flows"
- **Approach**: Constructs a delta Program Dependence Graph (delta-PDG) from the changed code and applies graph clustering to separate concerns.
- **Key assumption**: Relies on full PDG construction, which requires parseable source code.

### SmartCommit (FSE 2021)
- **Paper**: "SmartCommit: A Graph-Based Interactive Assistant for Activity-Oriented Commits"
- **Approach**: Builds a change dependency graph considering structural, semantic, and co-evolution relationships, then partitions it into activity-oriented groups.
- **Key assumption**: Uses predefined relationship types and thresholds for edge construction.

### UTANGO (FSE 2022)
- **Paper**: "UTANGO: Untangling Commits with Context-Aware, Graph-Based, Code Change Clustering Learning"
- **Approach**: Graph neural network that learns to predict whether pairs of changed lines belong to the same group, using code context and structural features.
- **Key assumption**: Requires training data; evaluated with 80/20 split on Java/C# datasets.

### HD-GNN (FSE 2024)
- **Paper**: "HD-GNN: Hierarchical Dependency Graph Neural Network for Commit Untangling"
- **Approach**: Hierarchical graph neural network that models dependencies at multiple granularity levels (token, statement, function, file) for pairwise line clustering.
- **Key assumption**: Requires training data; trained on Java dataset, cross-evaluated on other datasets.

## Language-Model Baselines

### Qwen3-32B
- **Model**: Qwen3-32B (32 billion parameters)
- **Context window**: 128K tokens (accommodates large diffs without truncation)
- **Variants**:
  - **base**: Zero-shot inference with structured prompting
  - **ft-2K**: QLoRA fine-tuned on 2,000 LKP-Synth examples
  - **ft-10K**: QLoRA fine-tuned on 10,000 LKP-Synth examples
- **Fine-tuning details**:
  - Method: QLoRA (rank=16, alpha=32)
  - Training data: Sampled from LKP-Synth with pos:neg = 3:1 ratio
  - Positive examples balanced across Easy:Medium:Hard = 1:1:1
  - Learning rate: 2e-4
  - Batch size: 8

## Evaluation Protocol

All methods receive identical inputs:
- The tangled diff (source) containing all changed lines
- For methods that require it: the number of target groups (k)

All methods are evaluated with the same:
- Case-level threshold metrics (Accuracy_C@100, @90, @80)
- Permutation-invariant group matching
- Changed-line counting (only `+` and `-` lines, excluding `+++`/`---` headers)

Methods that assume a fixed number of output groups (e.g., graph methods trained with known k) state this assumption explicitly. The LLM baselines infer the number of groups from the input.

## Directory Structure

```
baselines/
├── README.md                    # This file
├── ast_parser.py               # Shared: tree-sitter AST parsing
├── delta_pdg.py                # Shared: delta Program Dependence Graph
├── pdg_builder.py              # Shared: PDG construction
├── entity_extractor.py         # Shared: entity/reference extraction
├── source_resolver.py          # Shared: source code resolution
├── eval_all_datasets.py        # Cross-dataset evaluation runner
├── barnett/
│   ├── barnett_faithful.py     # Implementation
│   └── run_barnett.py          # Evaluation runner
├── herzig/
│   ├── herzig_faithful.py      # Implementation
│   └── run_herzig.py           # Evaluation runner
├── flexeme/
│   ├── flexeme_faithful.py     # Implementation
│   └── run_flexeme.py          # Evaluation runner
├── utango/
│   ├── utango_faithful.py      # Implementation
│   ├── run_utango.py           # Evaluation runner
│   └── train_utango.py         # GCN training script
├── hdgnn/
│   ├── hdgnn_faithful.py       # Implementation
│   ├── hdgnn_source.py         # Full GNN source (cross-dataset)
│   ├── run_hdgnn.py            # Evaluation runner
│   ├── train_hdgnn.py          # Edge classifier training
│   └── train_hdgnn_source.py   # Cross-dataset GNN training
└── smartcommit/
    ├── smartcommit_faithful.py # Implementation
    └── run_smartcommit.py      # Evaluation runner
```

## Dependencies

```bash
pip install tree-sitter tree-sitter-c numpy scipy scikit-learn
```

- `tree-sitter` + `tree-sitter-c`: AST parsing for function boundary detection
- `numpy`: Graph feature computation
- `scipy`: Sparse matrix operations (for graph methods)
- `scikit-learn`: Clustering (agglomerative, spectral)

## Running Baselines

### Evaluate All Methods on LKP-Untangle

```bash
# Run individual baselines
cd barnett && python run_barnett.py
cd herzig && python run_herzig.py
cd flexeme && python run_flexeme.py
cd utango && python run_utango.py
cd hdgnn && python run_hdgnn.py
cd smartcommit && python run_smartcommit.py
```

### Evaluate on Java/C# Datasets

```bash
python eval_all_datasets.py
```

This evaluates all 6 baseline methods on both the Java and C# validation splits (seed=42, 20% test).

### Training Graph Methods

HD-GNN and UTANGO require training before evaluation:

```bash
# Train HD-GNN edge classifier on LKP-Synth
cd hdgnn
python train_hdgnn.py \
    --data ../../data/lkp_synth_sample_10k.json \
    --epochs 100 --batch-size 50 --lr 0.003 \
    --embed-dim 64 --hidden-dim 128

# Train HD-GNN with cross-dataset evaluation
python train_hdgnn_source.py \
    --dataset java \
    --epochs 100 --patience 10 --seed 42

# Train UTANGO GCN
cd utango
python train_utango.py \
    --data ../../data/lkp_synth_sample_10k.json \
    --epochs 100 --lr 0.01
```

## Implementation Notes

### Barnett (barnett_faithful.py)
- Implements ClusterChanges algorithm with 3 rules:
  1. SameEnclosingMethod: lines in same function → same group
  2. Def-Use: definition and usage of same variable → same group
  3. Use-Use: two usages of same variable → same group
- Uses tree-sitter for precise function boundary detection

### Herzig (herzig_faithful.py)
- Implements 5 confidence voters:
  1. FileDistance: file path similarity
  2. PackageDistance: directory/namespace proximity
  3. CallGraph: function call relationships
  4. ChangeCouplings: co-change history
  5. DataDependency: shared variable access
- Combines votes with weighted confidence for clustering

### Flexeme (flexeme_faithful.py)
- Constructs delta-PDG from changed code
- Applies WL subtree kernel (k=10 iterations) for node similarity
- Uses complete-linkage hierarchical clustering

### UTANGO (utango_faithful.py)
- Label-GCN with AST-derived node labels
- 2-layer GCN with learnable weight matrices
- Hierarchical agglomerative clustering on learned embeddings

### HD-GNN (hdgnn_faithful.py)
- Entity Reference Graph: captures code entity relationships
- Code Change Relation Graph: models change-level dependencies
- Edge features + MLP classifier for pairwise grouping
- Threshold-based clustering (default threshold=0.5)

### SmartCommit (smartcommit_faithful.py)
- Diff Hunk Graph with 4 link types:
  1. Hard links: structural dependencies
  2. Soft links: semantic similarity
  3. Refactoring links: rename/move patterns
  4. Cosmetic links: formatting changes
- Kruskal-like edge-shrinking clustering (threshold=0.6)
