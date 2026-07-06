#!/bin/bash
# train_eval_chain_n2000.sh — End-to-end pipeline: train QLoRA -> evaluate
set -u

# Configuration (set these environment variables before running)
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH to the Qwen3-32B weights directory"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ADAPTER="$SCRIPT_DIR/checkpoints/lora_q3_32b_n2000"
EVAL="../../data/lkp_manual/lkp_untangle.json"
LOG="$SCRIPT_DIR/chain_n2000.log"
TRAIN_LOG="$SCRIPT_DIR/train_q3_32b_n2000.log"
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >>"$LOG"; }
: >"$LOG"
log "=== N2000 chain start ==="

# 1) Train (1 epoch)
log "=== Train N2000 (1 epoch) ==="
python train_qlora_q3_32b.py --model_path "$MODEL_PATH" --dataset n2000 --epochs 1 >"$TRAIN_LOG" 2>&1
rc=$?
log "Train exit=$rc"
if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  log "ERROR: adapter not found after training"; exit 1
fi
log "Training done, adapter saved"

check_ok(){ python3 -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if 'accuracy_a_mean' in d.get('summary',{}) else 1)" 2>/dev/null; }

# 2) Evaluate N2000
log "=== Evaluate N2000 ==="
python eval_local_q3_32b.py --model_path "$MODEL_PATH" --adapter "$ADAPTER" --eval "$EVAL" \
  --out results_q3_32b_n2000.json \
  --checkpoint results_q3_32b_n2000.json.ckpt.json \
  --resume --batch_size "$EVAL_BATCH_SIZE" >>"$LOG" 2>&1
if check_ok results_q3_32b_n2000.json; then log "N2000 eval success"; else log "ERROR: N2000 eval incomplete"; fi

# 3) Summary
log "=== Results ==="
python3 - >>"$LOG" 2>&1 <<'PY'
import json
for tag, fn in [("BASE","results_q3_32b_base.json"),("N2000","results_q3_32b_n2000.json")]:
    try:
        s = json.load(open(fn))["summary"]
        print(f"{tag}: acc_a_mean={s.get('accuracy_a_mean'):.4f} "
              f"c@100={s.get('accuracy_c_100'):.4f} c@90={s.get('accuracy_c_90'):.4f} "
              f"c@80={s.get('accuracy_c_80'):.4f} n={s.get('n_evaluated_full')}")
    except Exception as e:
        print(f"{tag}: {e}")
PY
log "=== CHAIN DONE ==="
