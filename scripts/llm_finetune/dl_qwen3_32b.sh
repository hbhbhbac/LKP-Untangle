#!/bin/bash
# dl_qwen3_32b.sh — Download Qwen3-32B model weights from ModelScope
# Requires ~100GB disk space for the complete model (17 shards)
set -u

# Configuration
MODEL=Qwen/Qwen3-32B
OUTPUT_DIR=${1:-"./Qwen3-32B"}
mkdir -p "$OUTPUT_DIR"

echo "=== Downloading Qwen3-32B to $OUTPUT_DIR ==="
echo "=== Start: $(date) ==="

# Download config and tokenizer files
for f in config.json configuration.json generation_config.json \
         merges.txt vocab.json tokenizer.json tokenizer_config.json \
         model.safetensors.index.json LICENSE README.md; do
  if [ -e "$OUTPUT_DIR/$f" ]; then
    echo "skip $f (exists)"
    continue
  fi
  echo "Downloading $f..."
  modelscope download --model "$MODEL" "$f" --local_dir "$OUTPUT_DIR" 2>&1
done

# Download 17 model shards
for i in $(seq -w 1 17); do
  f="model-000${i}-of-00017.safetensors"
  if [ -e "$OUTPUT_DIR/$f" ]; then
    echo "skip $f (exists)"
    continue
  fi
  echo "--- $(date) downloading $f ---"
  modelscope download --model "$MODEL" "$f" --local_dir "$OUTPUT_DIR" 2>&1
  if [ -f "$OUTPUT_DIR/$f" ]; then
    echo "OK $f ($(du -h "$OUTPUT_DIR/$f" | cut -f1))"
  else
    echo "FAIL $f"
  fi
done

echo "=== Download complete: $(date) ==="
echo "Model path: $OUTPUT_DIR"
