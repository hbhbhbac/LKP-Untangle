# Replication Package: Rethinking Commit Decomposition Evaluation: Realistic and Scalable Benchmarks

This replication package accompanies the paper *"Rethinking Commit Decomposition Evaluation: Realistic and Scalable Benchmarks"* and provides the data, scripts, and results needed to reproduce the experimental findings.

## Overview

This paper presents:
1. **LKP-Untangle**: A review-driven benchmark of 350 manually verified commit decomposition cases from Linux kernel patch review.
2. **LKP-Synth**: A scalable synthetic dataset of ~1.07M training cases constructed from kernel patch series.
3. An **aligned evaluation protocol** for consistent comparison across commit decomposition methods.

## Repository Structure

```
replication/
├── README.md                              # This file
├── data/
│   ├── dataset_description.json           # Dataset metadata and statistics
│   └── lkp_untangle.json                 # LKP-Untangle benchmark (350 cases)
├── scripts/
│   ├── metrics.py                         # Core evaluation metrics (Eq. 1)
│   ├── metrics_full.py                    # Extended metrics with level breakdown
│   ├── evaluate.py                        # Run evaluation on predictions
│   ├── evaluate_all.py                    # Full evaluation (all methods, all datasets)
│   ├── reproduce_tables.py                # Reproduce Tables II-IV from scratch
│   ├── dataset_stats.py                   # Compute dataset statistics (Table I)
│   ├── prepare_splits.py                  # Create 80/20 train/test splits
│   ├── sample_training_mixture.py         # Sample fine-tuning mixtures
│   ├── data_construction/                 # LKP-Synth construction pipeline
│   │   ├── README.md                      # Pipeline documentation
│   │   ├── lkml_preprocess.py            # Step 1-2: LKML → SQLite
│   │   ├── core.py                        # Shared constants
│   │   ├── relation_builder.py           # Step 3: Series relations
│   │   ├── relation_builder/             # Relation builder modules
│   │   ├── build_tangled_dataset.py      # Step 4: 7-strategy merging
│   │   ├── run_all_strategies.py         # Step 4: Parallel orchestration
│   │   ├── build_balanced_dataset.py     # Step 6: Difficulty balancing
│   │   ├── export_dataset.py             # Dataset export with ground truth
│   │   ├── diff_grouper.py              # Diff parsing utility
│   │   ├── compute_dataset_stats.py     # Statistics computation
│   │   ├── judge_dataset.py             # Quality auto-judging
│   │   └── processors/                   # I/O utilities
│   └── llm_finetune/                     # Qwen3-32B fine-tuning
│       ├── README.md                      # Fine-tuning documentation
│       ├── train_qlora_q3_32b.py         # QLoRA training script (2x A800)
│       ├── eval_local_q3_32b.py          # Local model evaluation
│       ├── eval_cross_dataset.py        # Cross-dataset evaluation
│       ├── config.py                      # Configuration
│       ├── llm_api.py                     # Prompt construction
│       ├── build_env.sh                   # Environment setup
│       ├── dl_qwen3_32b.sh              # Model download
│       └── train_eval_chain_n2000.sh     # End-to-end pipeline
├── baselines/
│   ├── README.md                          # Baseline descriptions & usage
│   ├── ast_parser.py                      # Shared: AST parsing
│   ├── delta_pdg.py                       # Shared: delta-PDG
│   ├── pdg_builder.py                     # Shared: PDG construction
│   ├── entity_extractor.py               # Shared: entity extraction
│   ├── source_resolver.py                # Shared: source resolution
│   ├── eval_all_datasets.py              # Cross-dataset evaluation
│   ├── barnett/                           # Barnett (ICSE 2015)
│   ├── herzig/                            # Herzig (ESE 2016)
│   ├── flexeme/                           # Flexeme (FSE 2020)
│   ├── utango/                            # UTANGO (FSE 2022)
│   ├── hdgnn/                             # HD-GNN (FSE 2024) + per-dataset pre-trained models
│   └── smartcommit/                       # SmartCommit (FSE 2021)
├── scripts/
│   └── llm/results/                       # Pre-computed LLM predictions
│       └── predictions_qwen3_32b.json     # Qwen3-32B ft-10K on LKP-Untangle
└── results/
    ├── table2_overall_results.csv         # Table II: Overall performance
    ├── table3_difficulty_results.csv      # Table III: By difficulty tier
    ├── table4_size_results.csv            # Table IV: By changed-line size
    └── table5_finetuning_results.csv      # Table V: Fine-tuning scaling
```

## Datasets

### LKP-Untangle (Primary Benchmark)

- **350** manually confirmed review-driven cases
- Language: C (Linux kernel)
- Target groups: 2–5
- Changed lines (Avg / Median): 283.6 / 115.0
- Files (Avg / Median): 4.0 / 3.0
- Difficulty distribution — Easy : Medium : Hard = 37 : 23 : 40
- Construction: Reviewer-requested patch splits confirmed by subsequent author-submitted revisions. This file is the gold-350 revision of the same 350 cases, with public ids 1–350.

Difficulty levels are defined as:
- **Easy**: No file overlap between any pair of target groups
- **Medium**: File overlap without function overlap
- **Hard**: Function-level overlap exists

### Java Dataset (Li et al., 2022)

- ~14K cases (13,500 total)
- Language: Java
- Evaluation: 80/20 train/test split (seed=42), test set = 2,700 cases
- Changed lines (Avg / Median): 192.8 / 91.0
- Files (Avg / Median): 9.2 / 5.0
- Difficulty E:M:H = 75:21:4

### C# Dataset (Partachi et al., 2020)

- 1,612 cases
- Language: C#
- Evaluation: 80/20 train/test split (seed=42), test set = 323 cases
- Changed lines (Avg / Median): 965.1 / 101.0
- Files (Avg / Median): 17.2 / 5.0
- Difficulty E:M:H = 39:52:10

### LKP-Synth (Training Data)

- ~1.07M cases (760K positive + 310K negative)
- Language: C (Linux kernel)
- Target groups: 2–5
- Changed lines (Avg / Median): 305.0 / 121.6
- Files (Avg / Median): 4.6 / 3.0
- Difficulty E:M:H = 5:22:73
- Fine-tuning subsets:
  - **2K subset**: 2,000 examples, positive:negative = 3:1, Easy:Medium:Hard = 1:1:1
  - **10K subset**: 10,000 examples, positive:negative = 3:1, Easy:Medium:Hard = 1:1:1

## Data Format

Each case in `lkp_untangle.json` follows this structure:

```json
{
  "id": 3,
  "source": {
    "diff": "diff --git a/... \n+added line\n-removed line\n...",
    "files": ["path/to/file.c"],
    "subject": "commit subject line",
    "message_id": "...",
    "lore_url": "https://lkml.kernel.org/r/..."
  },
  "targets": [
    {
      "diff": "diff --git a/... (group 0 changes)"
    },
    {
      "diff": "diff --git a/... (group 1 changes)"
    }
  ]
}
```

- `source.diff`: The tangled (combined) diff representing all changes.
- `targets`: A list of 2–5 atomic groups. Each target contains the diff lines belonging to that group.
- `ground_truth`: Structured line-level labels used for evaluation. Contains `files`, each with `lines` entries having `type` (add/del/context/meta) and `group_id` fields.
- Ground-truth labels for evaluation are extracted from `ground_truth.files[*].lines` where `type` is `"add"` or `"del"`. Only `exact` and `evolved` lines have a `group_id`. Dropped lines and diff lines without an annotation keep `group_id` null and are not scored. A shared exact or evolved line stores a list of group ids; a match to any of them counts as correct.

## Evaluation Protocol

### Metrics (Equation 1)

We use three case-level threshold metrics:

$$\mathrm{Accuracy}_C@\tau = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left[s_i \geq \frac{\tau}{100}\right]$$

where:
- $s_i$ = best changed-line agreement score for case $i$ after optimal group matching
- $\tau \in \{100, 90, 80\}$
- Optimal group matching uses exhaustive permutation search (practical for ≤6 groups)

| Metric | Interpretation |
|--------|---------------|
| Accuracy_C@100 | Exact case-level decomposition success |
| Accuracy_C@90 | Near-complete (may need minor correction) |
| Accuracy_C@80 | Relaxed (preserves most assignments) |

### Running Evaluation

```bash
# Compute dataset statistics
python scripts/dataset_stats.py --dataset data/lkp_untangle.json

# Evaluate predictions
python scripts/evaluate.py --dataset data/lkp_untangle.json \
                           --predictions your_predictions.json \
                           --by-difficulty

# Create Java/C# splits
python scripts/prepare_splits.py --java-dataset data/java_dataset.json \
                                  --csharp-dataset data/csharp_dataset.json \
                                  --output-dir data/
```

### Prediction Format

Predictions should be a JSON file containing a list of lists:
```json
[
  [0, 0, 1, 1, 0, 1],   // case 0: 6 changed lines assigned to groups 0 or 1
  [0, 1, 2, 0, 1, 2, 0], // case 1: 7 changed lines assigned to groups 0, 1, or 2
  ...
]
```

Each inner list has one integer per changed line (in order of appearance in the tangled diff), indicating the predicted group assignment.

## Results (Paper Tables)

### Table II: Overall Case-Level Success Rates

| Dataset | Method | Accuracy_C@100 | Accuracy_C@90 | Accuracy_C@80 |
|---------|--------|:-:|:-:|:-:|
| LKP-Untangle | Qwen3-32B ft-10K | 11.7% | **48.3%** | **58.3%** |
| LKP-Untangle | HD-GNN | **13.1%** | 43.1% | 56.9% |
| LKP-Untangle | Herzig | 12.9% | 40.9% | 53.4% |
| LKP-Untangle | UTANGO | 7.7% | 36.3% | 53.4% |
| LKP-Untangle | Barnett | 2.9% | 43.7% | 57.4% |
| LKP-Untangle | SmartCommit | 8.6% | 33.4% | 44.0% |
| LKP-Untangle | Flexeme | 0.0% | 5.1% | 12.9% |
| Java | Qwen3-32B | **37.7%** | **57.7%** | **70.5%** |
| Java | HD-GNN | 31.8% | 53.2% | 67.9% |
| Java | Herzig | 34.3% | 51.8% | 65.7% |
| Java | UTANGO | 26.2% | 45.1% | 59.6% |
| Java | Barnett | 18.1% | 42.5% | 57.9% |
| Java | SmartCommit | 12.1% | 20.5% | 28.0% |
| Java | Flexeme | 1.1% | 4.8% | 12.0% |
| C# | Qwen3-32B | **21.4%** | **36.2%** | **51.4%** |
| C# | HD-GNN | 16.4% | 35.0% | 47.7% |
| C# | Herzig | 18.0% | **36.2%** | 50.5% |
| C# | UTANGO | 10.2% | 24.5% | 42.1% |
| C# | Barnett | 9.6% | 35.0% | 51.1% |
| C# | SmartCommit | 10.2% | 26.0% | 41.8% |
| C# | Flexeme | 0.0% | 18.6% | 31.0% |

### Table III: LKP-Untangle Results by Difficulty Tier

| Method | Easy @100 | Med @100 | Hard @100 | Easy @90 | Med @90 | Hard @90 | Easy @80 | Med @80 | Hard @80 |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Qwen3-32B ft-10K | 24.2% | 7.5% | 3.6% | 47.5% | 53.8% | 45.3% | 53.3% | 62.4% | 59.9% |
| HD-GNN | 29.2% | 11.8% | 0.7% | 60.8% | 55.9% | 32.8% | 67.5% | 61.3% | 46.0% |
| Herzig | 26.7% | 12.9% | 0.7% | 57.5% | 43.0% | 24.8% | 66.7% | 59.1% | 38.0% |
| UTANGO | 15.0% | 7.5% | 1.5% | 41.7% | 45.2% | 25.5% | 61.7% | 58.1% | 43.1% |
| Barnett | 1.7% | 8.6% | 0.0% | 37.5% | 58.1% | 39.4% | 50.0% | 66.7% | 57.7% |
| SmartCommit | 18.3% | 8.6% | 0.0% | 39.2% | 39.8% | 24.1% | 49.2% | 48.4% | 36.5% |
| Flexeme | 0.0% | 0.0% | 0.0% | 5.0% | 8.6% | 2.9% | 12.5% | 16.1% | 10.9% |

### Table IV: LKP-Untangle Exact Success by Changed-Line Size

| Method | 1–20 | 21–50 | 51–100 | 101–200 | 201+ |
|--------|:-:|:-:|:-:|:-:|:-:|
| Qwen3-32B ft-10K | 53.8% | 17.7% | 9.5% | 1.4% | 2.8% |
| HD-GNN | 48.7% | 21.0% | 16.2% | 4.4% | 2.8% |
| Herzig | 46.2% | 24.2% | 6.8% | 2.9% | 4.7% |
| UTANGO | 41.0% | 11.3% | 2.7% | 1.4% | 0.9% |
| Barnett | 12.8% | 3.2% | 2.7% | 0.0% | 0.9% |
| SmartCommit | 33.3% | 14.5% | 5.4% | 2.9% | 1.9% |
| Flexeme | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Table V: Qwen3-32B Fine-Tuning Results on LKP-Untangle

| Model | Training Data | Accuracy_C@100 | Accuracy_C@90 | Accuracy_C@80 |
|-------|---------------|:-:|:-:|:-:|
| Qwen3-32B base | None | 4.6% | 28.3% | 38.3% |
| Qwen3-32B ft-2K | 2K examples (pos:neg=3:1) | 5.4% | 28.9% | 39.7% |
| Qwen3-32B ft-10K | 10K examples (pos:neg=3:1) | **11.7%** | **48.3%** | **58.3%** |

## Baselines

### Program-Analysis and Heuristic Methods
- **Barnett**: Dependency-based clustering (Barnett et al., ICSE 2015)
- **Herzig**: Confidence-voter framework (Herzig & Zeller, ESE 2016)

### Graph and Learning-Based Methods
- **Flexeme**: Delta-PDG based clustering (Partachi et al., FSE 2020)
- **SmartCommit**: Graph-based change partitioning (Shen et al., FSE 2021)
- **UTANGO**: Graph neural network approach (Li et al., FSE 2022)
- **HD-GNN**: Hierarchical dependency GNN (Fan et al., FSE 2024)

### Language-Model Baselines
- **Qwen3-32B base**: Zero-shot inference
- **Qwen3-32B ft-2K**: QLoRA fine-tuned on 2K LKP-Synth examples
- **Qwen3-32B ft-10K**: QLoRA fine-tuned on 10K LKP-Synth examples

All methods receive the same changed-line inputs and are evaluated with the same case-level metrics and group-matching procedure.

## Reproducing Results

### Requirements
- Python >= 3.8
- Evaluation scripts (`scripts/evaluate.py`, `scripts/dataset_stats.py`): standard library only
- Baseline methods (`scripts/reproduce_tables.py`): `tree-sitter`, `tree-sitter-c`, `numpy`, `scipy`
- HD-GNN GNN model: additionally requires `torch`
- Fine-tuning: `transformers`, `peft`, `torch`, `bitsandbytes`

```bash
# Minimal (evaluation only)
pip install -e .  # no deps

# Full baseline reproduction
pip install tree-sitter tree-sitter-c numpy scipy torch
```

### Step-by-step

1. **Compute dataset statistics** (reproduces Table I):
   ```bash
   python scripts/dataset_stats.py --dataset data/lkp_untangle.json
   ```

2. **Prepare cross-language splits** (for Java/C# evaluation):
   ```bash
   python scripts/prepare_splits.py \
       --java-dataset data/java_dataset.json \
       --csharp-dataset data/csharp_dataset.json \
       --output-dir data/ --seed 42
   ```

3. **Run evaluation on your predictions**:
   ```bash
   python scripts/evaluate.py \
       --dataset data/lkp_untangle.json \
       --predictions your_method_predictions.json \
       --by-difficulty
   ```

4. **Verify results against paper tables**:
   The `results/` directory contains the paper-reported numbers in CSV format.
   Running `reproduce_tables.py` regenerates these from scratch by executing
   each baseline's predict function against the dataset's `ground_truth` labels.

   ```bash
   # Full reproduction (runs all 6 baselines, ~1 min on a single core)
   python scripts/reproduce_tables.py

   # Run a subset of methods
   python scripts/reproduce_tables.py --methods barnett herzig smartcommit

   # Output is written to results/reproduced/ (does not overwrite reference CSVs)
   ```

   The script imports each baseline's predict function from `baselines/`,
   runs it on all 350 cases, and evaluates using the same permutation-invariant
   metrics defined in the paper. Output CSVs are written to `results/reproduced/`.


   **Note on Windows**: If you encounter an OMP error with numpy+torch, set:
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE
   ```

### Fine-Tuning Qwen3-32B

To reproduce the LLM fine-tuning experiments (requires 2x A800-80GB):
1. Use `data/lkp_synth_sample_2k.json` or `data/lkp_synth_sample_10k.json`
2. Fine-tune Qwen3-32B using QLoRA with the training data
3. Evaluate the fine-tuned model on all 350 LKP-Untangle cases
4. Fine-tuning hyperparameters: QLoRA with rank=16, alpha=32, batch_size=8, learning_rate=2e-4

See `scripts/llm_finetune/README.md` for detailed instructions.

### Running Baselines

To reproduce baseline method results:
```bash
cd baselines

# Evaluate all methods on LKP-Untangle
python barnett/run_barnett.py
python herzig/run_herzig.py
python flexeme/run_flexeme.py
python smartcommit/run_smartcommit.py

# Train HD-GNN per dataset (each model evaluates on its own test split)
python hdgnn/train_hdgnn_source.py --dataset java --epochs 100 --seed 42
python hdgnn/train_hdgnn_source.py --dataset csharp --epochs 100 --seed 42
python hdgnn/train_hdgnn_source.py --dataset lkp --epochs 100 --seed 42

# Cross-dataset evaluation (Java/C# val splits, uses per-dataset models)
python eval_all_datasets.py
```

See `baselines/README.md` for detailed instructions and dependencies.

### Constructing LKP-Synth from Scratch

To rebuild the full LKP-Synth dataset from raw LKML data:
```bash
cd scripts/data_construction

# Step 1-2: Clone and preprocess LKML archives
# (See data_construction/README.md for git clone commands)
python lkml_preprocess.py --mode full

# Step 3: Build series relations
python relation_builder.py

# Step 4-5: Build and filter tangled dataset (all 7 strategies)
python run_all_strategies.py

# Step 6: Build balanced dataset
python build_balanced_dataset.py \
    --filter-dir ../../data/lkp_synthesize/filter \
    --per-level 2000 --neg-ratio 1.0 --seed 42 \
    --out-dir ../../data/lkp_synthesize/balanced
```

See `scripts/data_construction/README.md` for the full pipeline documentation.
