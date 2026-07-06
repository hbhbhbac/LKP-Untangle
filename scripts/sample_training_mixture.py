"""
sample_training_mixture.py — Sample fine-tuning mixtures from LKP-Synth balanced pools.

The paper uses two fine-tuning subsets:
  - 2K mixture: 2,000 examples total, pos:neg = 3:1, positive balanced E:M:H = 1:1:1
  - 10K mixture: 10,000 examples total, pos:neg = 3:1, positive balanced E:M:H = 1:1:1

Usage:
    python sample_training_mixture.py \
        --pool ../data/lkp_synth_sample_10k.json \
        --size 10000 \
        --pos-neg-ratio 3 \
        --seed 42 \
        --output mixture_10k.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def sample_mixture(pool: list, size: int, pos_neg_ratio: float, seed: int) -> list:
    """
    Sample a training mixture from the balanced pool.

    Args:
        pool: Full balanced dataset with 'label' and 'difficulty_level' fields
        size: Total number of examples to sample
        pos_neg_ratio: Ratio of positive to negative examples (3 means 3:1)
        seed: Random seed for reproducibility

    Returns:
        Sampled mixture list
    """
    random.seed(seed)

    n_pos = int(size * pos_neg_ratio / (pos_neg_ratio + 1))
    n_neg = size - n_pos

    positives = [e for e in pool if e["label"] == "tangled"]
    negatives = [e for e in pool if e["label"] == "no_split"]

    # Group positives by difficulty for balanced sampling
    # Map L1 -> Easy, L2/L3 -> Medium, L4/L5 -> Hard
    difficulty_map = {"L1": "Easy", "L2": "Medium", "L3": "Medium", "L4": "Hard", "L5": "Hard"}
    pos_by_diff = {"Easy": [], "Medium": [], "Hard": []}
    for e in positives:
        diff_level = difficulty_map.get(e.get("difficulty_level", ""), None)
        if diff_level:
            pos_by_diff[diff_level].append(e)

    # Sample equal numbers from each difficulty tier
    n_per_tier = n_pos // 3
    remainder = n_pos - 3 * n_per_tier

    sampled_pos = []
    for tier in ["Easy", "Medium", "Hard"]:
        available = pos_by_diff[tier]
        n_sample = n_per_tier + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        if len(available) >= n_sample:
            sampled_pos.extend(random.sample(available, n_sample))
        else:
            sampled_pos.extend(available)
            print(f"Warning: only {len(available)} {tier} examples available, needed {n_sample}")

    # Sample negatives
    if len(negatives) >= n_neg:
        sampled_neg = random.sample(negatives, n_neg)
    else:
        sampled_neg = negatives
        print(f"Warning: only {len(negatives)} negatives available, needed {n_neg}")

    mixture = sampled_pos + sampled_neg
    random.shuffle(mixture)

    print(f"Sampled mixture: {len(sampled_pos)} positive + {len(sampled_neg)} negative = {len(mixture)} total")
    return mixture


def main():
    parser = argparse.ArgumentParser(description="Sample fine-tuning mixture from LKP-Synth.")
    parser.add_argument("--pool", required=True, help="Path to balanced pool JSON")
    parser.add_argument("--size", type=int, required=True, help="Total mixture size (e.g., 2000 or 10000)")
    parser.add_argument("--pos-neg-ratio", type=float, default=3.0, help="Positive:negative ratio (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.pool, encoding="utf-8") as f:
        pool = json.load(f)

    print(f"Pool size: {len(pool)}")
    mixture = sample_mixture(pool, args.size, args.pos_neg_ratio, args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mixture, f, ensure_ascii=False)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
