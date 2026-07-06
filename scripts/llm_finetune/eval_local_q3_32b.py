"""
eval_local_q3_32b.py — Evaluate Qwen3-32B (QLoRA / base) via transformers

Reuses build_prompt / build_reference / build_prediction / _extract_json from llm_api.py
and evaluate_all from metrics.py.

Usage:
    python eval_local_q3_32b.py --model_path /path/to/Qwen3-32B \
        --adapter /path/to/checkpoints/lora_q3_32b_n10000 \
        --eval ../../data/lkp_manual/lkp_untangle.json \
        --out results_q3_32b_n10000.json

    # base (no adapter)
    python eval_local_q3_32b.py --model_path /path/to/Qwen3-32B \
        --eval ../../data/lkp_manual/lkp_untangle.json \
        --out results_q3_32b_base.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_api import (
    SYSTEM_PROMPT, build_prompt, build_reference,
    build_prediction, _extract_json, _get_changed_lines,
)
from metrics import evaluate_all

MODEL_PATH = None  # Set via --model_path argument


def load_model(adapter: str | None):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # 4-bit NF4 , ( 32B)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_cfg, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[model] loaded adapter: {adapter}")
    else:
        print("[model] base (no adapter)")
    model.eval()
    return model, tok


def make_chat_text(tok, prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    # Qwen3 hybrid reasoning , <think>; groups JSON
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def parse_raw(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    try:
        return json.loads(_extract_json(raw))
    except Exception:
        return {"groups": []}


@torch.no_grad()
def infer(model, tok, prompt: str, max_new_tokens: int = 1024,
          max_input_tokens: int = 24000) -> tuple[dict, str]:
    """ (dict, ). : ok / skip_long / oom."""
    text = make_chat_text(tok, prompt)
    inputs = tok(text, return_tensors="pt")
    ntok = inputs["input_ids"].shape[1]
    # 1) : , (), GPUOOM
    if ntok > max_input_tokens:
        return {"groups": []}, "skip_long"
    inputs = inputs.to(model.device)
    try:
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.1, top_p=0.9, repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_raw(raw), "ok"
    except torch.cuda.OutOfMemoryError:
        # 2) OOM: , ,
        torch.cuda.empty_cache()
        return {"groups": []}, "oom"
    finally:
        # 3)
        del inputs
        torch.cuda.empty_cache()


@torch.no_grad()
def infer_batch(model, tok, prompts: list[str], max_new_tokens: int = 1024,
                max_input_tokens: int = 24000) -> list[tuple[dict, str]]:
    """batch OOM OOM oom"""
    if not prompts:
        return []
    if len(prompts) == 1:
        return [infer(model, tok, prompts[0], max_new_tokens, max_input_tokens)]

    texts = [make_chat_text(tok, prompt) for prompt in prompts]
    lengths = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    keep = [i for i, ntok in enumerate(lengths) if ntok <= max_input_tokens]
    results: list[tuple[dict, str] | None] = [None] * len(prompts)
    for i, ntok in enumerate(lengths):
        if ntok > max_input_tokens:
            results[i] = ({"groups": []}, "skip_long")
    if not keep:
        return [r for r in results if r is not None]

    kept_texts = [texts[i] for i in keep]
    try:
        inputs = tok(kept_texts, return_tensors="pt", padding=True).to(model.device)
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.1, top_p=0.9, repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        for row_idx, original_idx in enumerate(keep):
            raw = tok.decode(out[row_idx][prompt_len:], skip_special_tokens=True)
            results[original_idx] = (parse_raw(raw), "ok")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = len(prompts) // 2
        left = infer_batch(model, tok, prompts[:mid], max_new_tokens, max_input_tokens)
        right = infer_batch(model, tok, prompts[mid:], max_new_tokens, max_input_tokens)
        return left + right
    finally:
        if "inputs" in locals():
            del inputs
        torch.cuda.empty_cache()

    return [r for r in results if r is not None]


def load_checkpoint(path: Path):
    if not path.exists():
        return [], [], [], [], 0, 0, []
    data = json.loads(path.read_text())
    records = data.get("records", [])
    preds = [r["pred"] for r in records]
    refs = [r["ref"] for r in records]
    levels = [r.get("level", "NA") for r in records]
    n_skip = sum(1 for r in records if r.get("status") == "skip_long")
    n_oom = sum(1 for r in records if r.get("status") == "oom")
    skipped_ids = [r.get("id") for r in records
                   if r.get("status") in {"skip_long", "oom"}]
    return preds, refs, levels, records, n_skip, n_oom, skipped_ids


def write_checkpoint(path: Path, records: list[dict], summary: dict | None = None):
    payload = {"records": records}
    if summary is not None:
        payload["summary"] = summary
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to Qwen3-32B model weights")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model")
    ap.add_argument("--eval", required=True, help="Evaluation dataset JSON path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint file")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint path (default: <out>.ckpt.json)")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_input_tokens", type=int, default=24000,
                    help="Max input tokens (skip longer prompts to avoid OOM)")
    ap.add_argument("--hint_groups", action="store_true", help="Include group count hint in prompt")
    args = ap.parse_args()

    global MODEL_PATH
    MODEL_PATH = args.model_path

    data = json.loads(Path(args.eval).read_text())
    print(f"[data] {len(data)} cases from {args.eval}")
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be >= 1")

    out_path = Path(args.out)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(args.out + ".ckpt.json")
    if args.resume:
        preds, refs, levels, records, n_skip, n_oom, skipped_ids = load_checkpoint(ckpt_path)
        print(f"[resume] {len(records)} records from {ckpt_path}")
    else:
        preds, refs, levels, records = [], [], [], []
        n_skip, n_oom = 0, 0
        skipped_ids = []

    model, tok = load_model(args.adapter)

    import time as _t
    start = len(records)
    for batch_start in range(start, len(data), args.batch_size):
        batch = data[batch_start: batch_start + args.batch_size]
        prompts = [build_prompt(entry, hint_groups=args.hint_groups) for entry in batch]
        display_ntoks = [len(tok(prompt)["input_ids"]) for prompt in prompts]
        _t0 = _t.time()
        outs = infer_batch(model, tok, prompts, args.max_new_tokens,
                           args.max_input_tokens)
        dt = _t.time() - _t0
        for j, (entry, ntok, (llm_out, status)) in enumerate(zip(batch, display_ntoks, outs)):
            i = batch_start + j
            changed = _get_changed_lines(entry)
            print(f"  [{i+1}/{len(data)}] id={entry.get('id')} in={ntok}tok "
                  f"status={status} batch={len(batch)} {dt:.1f}s", flush=True)
            if status == "skip_long":
                n_skip += 1; skipped_ids.append(entry.get("id"))
            elif status == "oom":
                n_oom += 1; skipped_ids.append(entry.get("id"))
            pred, _ = build_prediction(entry, llm_out, len(changed))
            ref = build_reference(entry)
            level = entry.get("difficulty", {}).get("level", "NA")
            preds.append(pred)
            refs.append(ref)
            levels.append(level)
            records.append({"id": entry.get("id"), "pred": pred, "ref": ref,
                            "status": status, "level": level})
        write_checkpoint(ckpt_path, records)

    result = evaluate_all(preds, refs, levels)
    result["n_skipped_long"] = n_skip
    result["n_oom"] = n_oom
    result["n_evaluated_full"] = len(data) - n_skip - n_oom
    result["skipped_ids"] = skipped_ids
    print(json.dumps({k: v for k, v in result.items() if k != "skipped_ids"},
                     indent=2, ensure_ascii=False))
    print(f"[={n_skip} OOM={n_oom} ={result['n_evaluated_full']}]")
    write_checkpoint(ckpt_path, records, result)
    out_path.write_text(json.dumps(
        {"summary": result, "records": records}, indent=2, ensure_ascii=False))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
