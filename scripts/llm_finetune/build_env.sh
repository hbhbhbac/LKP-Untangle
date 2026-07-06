#!/bin/bash
# build_env.sh — Create Python environment for Qwen3-32B fine-tuning
set -e

echo "=== Creating environment $(date) ==="
conda create -y -n qwen3_ft python=3.12
conda activate qwen3_ft
python --version

echo "=== Installing PyTorch 2.6 (CUDA 12.4) ==="
pip install --no-cache-dir "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing transformers stack ==="
pip install --no-cache-dir \
    "transformers>=4.40" \
    "accelerate>=1.4" \
    "peft>=0.10" \
    "trl>=0.12" \
    "datasets>=3.0" \
    "bitsandbytes>=0.43" \
    sentencepiece

echo "=== Environment setup complete $(date) ==="
