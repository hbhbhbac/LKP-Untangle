"""
config.py — LLM Configuration

Model paths, parameters, and data paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent  # replication/
RESULTS_DIR = SCRIPT_DIR / "results"

# Data paths (within replication package)
MANUAL_DATASET = PROJECT_ROOT / "data" / "lkp_manual" / "lkp_untangle.json"
SYNTH_DATA_DIR = PROJECT_ROOT / "data"

# Training data output
TRAIN_DATA_DIR = SCRIPT_DIR / "data"
TRAIN_JSONL = TRAIN_DATA_DIR / "train.jsonl"
VAL_JSONL = TRAIN_DATA_DIR / "val.jsonl"
TEST_JSONL = TRAIN_DATA_DIR / "test.jsonl"

# Results output
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
FINETUNED_RESULTS_DIR = RESULTS_DIR / "finetuned"

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

# Base model (HuggingFace model ID or local path)
MODEL_NAME = "Qwen/Qwen3-32B"

# Checkpoint output
LORA_OUTPUT_DIR = SCRIPT_DIR / "checkpoints" / "lora"
MERGED_MODEL_DIR = SCRIPT_DIR / "checkpoints" / "merged"

# ---------------------------------------------------------------------------
# Inference Configuration
# ---------------------------------------------------------------------------

INFERENCE_CONFIG = {
    "max_new_tokens": 8192,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

# Max input tokens before skipping (to avoid OOM)
MAX_INPUT_TOKENS = 24000

# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

TRAIN_CONFIG = {
    # LoRA parameters
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],

    # Training parameters
    "num_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 7168,

    # Data limits
    "max_train_samples": 8000,
    "max_val_samples": 500,

    # Save strategy
    "save_steps": 200,
    "eval_steps": 200,
    "logging_steps": 10,
}

# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

EVAL_CONFIG = {
    "batch_size": 1,
    "num_workers": 0,
    "save_predictions": True,
}
