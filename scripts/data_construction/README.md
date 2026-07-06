# LKP-Synth Data Construction Pipeline

This document describes the complete pipeline for constructing the LKP-Synth dataset from raw Linux kernel mailing list (LKML) data.

## Prerequisites

- Python >= 3.10
- Required packages: `orjson`, `tree-sitter`, `tree-sitter-c`
- Git (for cloning LKML archives)
- ~50GB disk space for raw data, ~20GB for processed data

```bash
pip install orjson tree-sitter tree-sitter-c
```

## Pipeline Overview

```
Step 1: Clone LKML archives (lore.kernel.org git epochs)
         ↓
Step 2: Preprocess emails → SQLite database
         ↓
Step 3: Build series relations (group patches into series)
         ↓
Step 4: Build tangled dataset (7 merging strategies)
         ↓
Step 5: Filter (atom/statement count bounds)
         ↓
Step 6: Build balanced dataset (difficulty-stratified, deduplicated)
         ↓
Step 7: Sample fine-tuning mixtures (2K, 10K with 3:1 pos:neg ratio)
```

## Step 1: Clone LKML Archives

The Linux kernel mailing list is archived on lore.kernel.org as git repositories split into "epochs". Each epoch contains emails as git commits.

```bash
# Create data directory
mkdir -p data/raw_data/lkml

# Clone LKML git epochs (example for epochs 0-9)
# Each epoch is a separate git repository
for i in $(seq 0 9); do
    git clone --bare https://lore.kernel.org/all/$i \
        data/raw_data/lkml/$i
done
```

Note: The full LKML archive contains 10+ epochs totaling ~40GB. Each epoch covers a time range of kernel development.

## Step 2: Preprocess Emails into SQLite

Parse the raw git-based mbox data into structured emails and store in SQLite.

```bash
python lkml_preprocess.py --mode full
```

This script:
1. Iterates over all git commits in each epoch (each commit = one email)
2. Parses email headers (Message-ID, Subject, From, In-Reply-To, References)
3. Extracts patch content and metadata (patch version, series position)
4. Stores structured data in `data/structure_data/lkml.sqlite`

Output: `data/structure_data/lkml.sqlite` (~5GB)

## Step 3: Build Series Relations

Group individual patches into patch series using threading (In-Reply-To/References) and subject line matching.

```bash
python relation_builder.py
```

This produces:
- Series-level metadata (series_id, version, patch count)
- Version links (v1 → v2 → v3 revision chains)
- `lkml_series_global.sqlite` — Global series index

## Step 4: Build Tangled Dataset (7 Strategies)

Merge patches from the same series to simulate tangled commits using 7 different strategies:

```bash
# Run all 7 strategies in parallel
python run_all_strategies.py
```

Or run individual strategies:
```bash
# Strategy: 2wayfile (merge 2 patches sharing a file)
python build_tangled_dataset.py --mode build \
    --strategy 1 0 0 0 0 0 0 \
    --out data/lkp_synthesize/origin/2wayfile.json

# Strategy: connected_func (merge via function-overlap connected components)
python build_tangled_dataset.py --mode build \
    --strategy 0 0 0 1 0 0 0 \
    --out data/lkp_synthesize/origin/connected_func.json
```

**Seven strategies:**
| Strategy | Description |
|----------|-------------|
| `2wayfile` | Greedy 2-patch pairs sharing files |
| `2wayfunction` | Greedy 2-patch pairs sharing functions |
| `connected_file` | Connected-component merge via shared files |
| `connected_func` | Connected-component merge via shared functions |
| `all_file` | Merge entire series if any pair shares files |
| `all_func` | Merge entire series if any pair shares functions |
| `all_direct` | Merge entire series directly (≥2 patches) |

Output: `data/lkp_synthesize/origin/{strategy}.json`

## Step 5: Filter Cases

Apply atom count and statement count bounds to remove trivial or excessively large cases.

```bash
python build_tangled_dataset.py --mode filter \
    --input data/lkp_synthesize/origin/2wayfile.json \
    --min-stmts 2 --max-stmts 4000 \
    --min-atoms 2 --max-atoms 10 \
    --out data/lkp_synthesize/filter/2wayfile.json
```

Filter parameters (from `run_all_strategies.py`):
- `min_atoms = 2` (at least 2 target groups)
- `max_atoms = 10`
- `min_stmts = 2` (at least 2 changed statements)
- `max_stmts = 4000`

Output: `data/lkp_synthesize/filter/{strategy}.json`

## Step 6: Build Balanced Dataset

Classify filtered cases by difficulty level and sample a balanced dataset with negative examples.

```bash
python build_balanced_dataset.py \
    --filter-dir data/lkp_synthesize/filter \
    --per-level 2000 \
    --neg-ratio 1.0 \
    --seed 42 \
    --out-dir data/lkp_synthesize/balanced
```

**Difficulty levels (5-level, mapped to 3-tier for evaluation):**
| Level | Overlap Condition | Paper Tier |
|-------|------------------|------------|
| L1 | No file overlap | Easy |
| L2 | Partial file overlap, no function overlap | Medium |
| L3 | Full file overlap, no function overlap | Medium |
| L4 | Partial function overlap | Hard |
| L5 | Full function overlap | Hard |

Parameters:
- `--per-level N`: Number of positive cases per difficulty level
- `--neg-ratio R`: Ratio of negative to total positive examples
- Negative examples are non-tangled patches (single-concern changes)

Output: `data/lkp_synthesize/balanced/balanced_N{per_level}.json`

## Step 7: Sample Fine-Tuning Mixtures

Sample the actual training mixtures used for Qwen3-32B fine-tuning:

```bash
# 2K mixture (1500 positive + 500 negative = 2000 total)
python sample_training_mixture.py \
    --pool data/lkp_synth_sample_2k.json \
    --size 2000 --pos-neg-ratio 3 --seed 42 \
    --output mixture_2k.json

# 10K mixture (7500 positive + 2500 negative = 10000 total)
python sample_training_mixture.py \
    --pool data/lkp_synth_sample_10k.json \
    --size 10000 --pos-neg-ratio 3 --seed 42 \
    --output mixture_10k.json
```

Mixture properties:
- Positive:Negative = 3:1
- Positive examples balanced across Easy:Medium:Hard = 1:1:1

## Full Dataset Scale

The complete LKP-Synth dataset contains:
- **~760K positive** synthetic tangled cases (across all strategies)
- **~310K negative** non-tangled cases
- **Total: ~1.07M** training cases

## Key Files

| File | Purpose |
|------|---------|
| `lkml_preprocess.py` | Parse LKML git epochs → SQLite |
| `core.py` | Shared constants and data structures |
| `relation_builder.py` | Build series/version relations |
| `build_tangled_dataset.py` | 7-strategy tangled case construction |
| `run_all_strategies.py` | Orchestrate all strategies in parallel |
| `build_balanced_dataset.py` | Difficulty classification and balancing |
| `diff_grouper.py` | Diff parsing and grouping utility |
| `processors/` | SQLite and JSONL I/O utilities |
