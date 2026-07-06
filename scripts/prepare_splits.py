"""
prepare_splits.py — Create the 80/20 train/test splits for Java and C# datasets.

The paper uses seed=42 for reproducible 80/20 splits. This script generates
the validation (test) splits used in Tables II and III.

Usage:
    python prepare_splits.py --java-dataset ../data/java_dataset.json \
                             --csharp-dataset ../data/csharp_dataset.json \
                             --output-dir ../data/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_dataset(data: list, test_ratio: float = 0.2, seed: int = 42):
    """Split dataset into train and test with fixed seed."""
    indices = list(range(len(data)))
    random.seed(seed)
    random.shuffle(indices)

    split_point = int(len(data) * (1 - test_ratio))
    train_indices = sorted(indices[:split_point])
    test_indices = sorted(indices[split_point:])

    train = [data[i] for i in train_indices]
    test = [data[i] for i in test_indices]
    return train, test


def main():
    parser = argparse.ArgumentParser(description="Create train/test splits.")
    parser.add_argument("--java-dataset", help="Path to Java dataset JSON")
    parser.add_argument("--csharp-dataset", help="Path to C# dataset JSON")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.java_dataset:
        print(f"Loading Java dataset from {args.java_dataset}...")
        with open(args.java_dataset, encoding="utf-8") as f:
            java_data = json.load(f)
        train, test = split_dataset(java_data, 0.2, args.seed)
        print(f"  Total: {len(java_data)}, Train: {len(train)}, Test: {len(test)}")

        with open(output_dir / "java_train.json", "w", encoding="utf-8") as f:
            json.dump(train, f)
        with open(output_dir / "java_val.json", "w", encoding="utf-8") as f:
            json.dump(test, f)
        print(f"  Saved to {output_dir}/java_train.json and java_val.json")

    if args.csharp_dataset:
        print(f"Loading C# dataset from {args.csharp_dataset}...")
        with open(args.csharp_dataset, encoding="utf-8") as f:
            csharp_data = json.load(f)
        train, test = split_dataset(csharp_data, 0.2, args.seed)
        print(f"  Total: {len(csharp_data)}, Train: {len(train)}, Test: {len(test)}")

        with open(output_dir / "csharp_train.json", "w", encoding="utf-8") as f:
            json.dump(train, f)
        with open(output_dir / "csharp_val.json", "w", encoding="utf-8") as f:
            json.dump(test, f)
        print(f"  Saved to {output_dir}/csharp_train.json and csharp_val.json")


if __name__ == "__main__":
    main()
