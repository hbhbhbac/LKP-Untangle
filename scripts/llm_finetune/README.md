# LLM Fine-Tuning Pipeline (Qwen3-32B)

This directory contains scripts for fine-tuning Qwen3-32B using QLoRA on LKP-Synth data.

## Environment Setup

### Hardware Requirements
- 2x NVIDIA A800 80GB (model distributed via device_map="auto")
- ~100GB disk space for model weights

### Software Setup

```bash
# Create environment
bash build_env.sh

# Download Qwen3-32B weights
bash dl_qwen3_32b.sh
```

### Python Dependencies

```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
datasets
accelerate
trl
```

## Training

### QLoRA Fine-Tuning Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-32B |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 2e-4 |
| Batch size | 8 (with gradient accumulation) |
| Max sequence length | 7168 tokens |
| Training epochs | 3 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Optimizer | paged_adamw_8bit |

### Run Training

```bash
# Full pipeline: train → evaluate
bash train_eval_chain_n2000.sh

# Or train directly
python train_qlora_q3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --data_path ../../../data/lkp_synth_sample_10k.json \
    --output_dir ./output_n10000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

### Training Data Format

Each training example is formatted as a conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a commit untangling assistant. Given a tangled diff, assign each changed line to a group."
    },
    {
      "role": "user",
      "content": "Decompose the following tangled diff into {k} groups:\n\n{diff_content}"
    },
    {
      "role": "assistant",
      "content": "{line_assignments}"
    }
  ]
}
```

## Evaluation

### Evaluate Fine-Tuned Model

```bash
python eval_local_q3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --adapter_path ./output_n10000/checkpoint-best \
    --dataset_path ../../../data/lkp_manual/lkp_untangle.json \
    --output_path results_q3_32b_n10000.json
```

### Evaluate on Cross-Language Datasets

```bash
python eval_cross_dataset.py \
    --model_path /path/to/Qwen3-32B \
    --adapter_path ./output_n10000/checkpoint-best \
    --java_dataset ../../../data/java_data/dataset.json \
    --csharp_dataset ../../../data/csharp_data/dataset.json
```

## Results

### Fine-Tuning Scaling (Table V)

| Model | Training Data | Accuracy_C@100 | Accuracy_C@90 | Accuracy_C@80 |
|-------|---------------|:-:|:-:|:-:|
| Qwen3-32B base | None | 4.6% | 28.3% | 38.3% |
| Qwen3-32B ft-2K | 2K examples | 5.4% | 28.9% | 39.7% |
| Qwen3-32B ft-10K | 10K examples | **11.7%** | **48.3%** | **58.3%** |

### Cross-Language Performance (Table II)

| Dataset | Accuracy_C@100 | Accuracy_C@90 | Accuracy_C@80 |
|---------|:-:|:-:|:-:|
| LKP-Untangle | 11.7% | 48.3% | 58.3% |
| Java (2,700 test) | 37.7% | 57.7% | 70.5% |
| C# (323 test) | 21.4% | 36.2% | 51.4% |

## Key Scripts

| File | Purpose |
|------|---------|
| `train_qlora_q3_32b.py` | QLoRA training on Qwen3-32B (2x A800) |
| `eval_local_q3_32b.py` | Local inference evaluation |
| `eval_cross_dataset.py` | Cross-dataset (Java/C#) evaluation |
| `config.py` | Global configuration (paths, hyperparameters) |
| `llm_api.py` | Prompt construction and prediction parsing |
| `build_env.sh` | Environment setup script |
| `dl_qwen3_32b.sh` | Model weight download script |
| `train_eval_chain_n2000.sh` | End-to-end train + eval pipeline |
