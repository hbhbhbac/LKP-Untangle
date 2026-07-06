"""
train_qlora_q3_32b.py — Qwen3-32B (dense, bf16) bitsandbytes 4-bit NF4 QLoRA

Hardware: 2x A800-80GB
Environment: torch >= 2.1, transformers >= 4.40, peft >= 0.10, bitsandbytes >= 0.43

Usage:
    python train_qlora_q3_32b.py --model_path /path/to/Qwen3-32B --dataset n2000 --epochs 3
    python train_qlora_q3_32b.py --model_path /path/to/Qwen3-32B --dataset n10000 --epochs 3

Data: data/train/train_{dataset}.jsonl (chat JSONL, messages: system/user/assistant)
Output: ./checkpoints/lora_q3_32b_{dataset}/
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = SCRIPT_DIR / "data" / "train"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "checkpoints"

# Qwen3-32B dense model: MLP uses nn.Linear, bnb-4bit -> ~18-20GB per GPU
# With 2x A800-80GB, model is split across GPUs via device_map="auto"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to Qwen3-32B model weights")
    ap.add_argument("--data_path", default=None, help="Path to training JSONL (overrides --dataset)")
    ap.add_argument("--dataset", default="n10000", choices=["n2000", "n10000"])
    ap.add_argument("--output_dir", default=None, help="Output directory for adapter")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=7168)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint in output_dir")
    args = ap.parse_args()

    if args.data_path:
        train_file = Path(args.data_path)
    else:
        train_file = DEFAULT_TRAIN_DIR / f"train_{args.dataset}.jsonl"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = DEFAULT_OUT_ROOT / f"lora_q3_32b_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] dataset={train_file}  out={out_dir}")

    model_path = args.model_path
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # bitsandbytes 4-bit NF4 quantization (QLoRA)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map="auto",  # distribute across 2x A800-80GB
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files=str(train_file), split="train")
    print(f"[data] {len(ds)} samples")

    # trl SFTConfig: max_length controls sequence length
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tok,
    )
    # Resume from checkpoint if requested
    resume_ckpt = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        resume_ckpt = get_last_checkpoint(str(out_dir))
        print(f"[resume] from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[done] adapter saved -> {out_dir}")


if __name__ == "__main__":
    main()
