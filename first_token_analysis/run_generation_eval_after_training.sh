#!/usr/bin/env bash
set -u

ENV_PY=/data/chenyang2/conda_envs/verl/bin/python
MODEL=/data/chenyang2/models/Qwen3-4B-SFT
ROUTER=/data/chenyang2/verl/router_data_5000_qwen3_4b_sft_20260921/router_weights.pt
EVAL=/data/chenyang2/verl/examples/grpo_trainer/eval_router.py
OUT=/data/chenyang2/verl/router_data_5000_qwen3_4b_sft_20260921/generation_eval_20260922
TRAIN_PATTERN='train_router.py.*router_data_5000_qwen3_4b_sft_20260921/router_weights.pt'

mkdir -p "$OUT"
echo "[$(date '+%F %T')] watcher started" >> "$OUT/watcher.log"

# The router file is written at the end of train_router.py.  Also wait for the
# training process to exit so that the evaluation never competes for GPU 0.
while [ ! -s "$ROUTER" ] || pgrep -u chenyang2 -f "$TRAIN_PATTERN" >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] waiting for router training to finish" >> "$OUT/watcher.log"
    sleep 60
done

echo "[$(date '+%F %T')] router weights ready: $ROUTER" >> "$OUT/watcher.log"

run_one() {
    local name="$1"
    local gpu="$2"
    local data="$3"
    local tmp="$OUT/${name}_eval_tmp.pt"
    local select_log="$OUT/${name}_select.log"
    local generate_log="$OUT/${name}_generate.log"

    echo "[$(date '+%F %T')] ${name}: phase 1 select on GPU ${gpu}" >> "$OUT/watcher.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$ENV_PY" "$EVAL" \
        --model "$MODEL" \
        --router "$ROUTER" \
        --data "$data" \
        --generate --step select \
        --num-prompts 0 \
        --batch-size 8 \
        --tmp-file "$tmp" \
        > "$select_log" 2>&1

    echo "[$(date '+%F %T')] ${name}: phase 2 generation on GPU ${gpu}" >> "$OUT/watcher.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$ENV_PY" "$EVAL" \
        --model "$MODEL" \
        --router "$ROUTER" \
        --data "$data" \
        --generate --step generate \
        --num-prompts 0 \
        --tp 1 \
        --gpu-mem-util 0.9 \
        --max-tokens 8192 \
        --max-model-len 16384 \
        --tmp-file "$tmp" \
        > "$generate_log" 2>&1

    echo "[$(date '+%F %T')] ${name}: complete" >> "$OUT/watcher.log"
}

run_one test 1 /data/chenyang2/verl/router_data_5000_qwen3_4b_sft_20260921/split/test.pt &
test_pid=$!
run_one ood 2 /data/chenyang2/verl/router_data_5000/split/ood_eval.pt &
ood_pid=$!

wait "$test_pid"
test_status=$?
wait "$ood_pid"
ood_status=$?

printf 'test_status=%s\nood_status=%s\n' "$test_status" "$ood_status" > "$OUT/status.txt"
echo "[$(date '+%F %T')] all evaluations finished" >> "$OUT/watcher.log"
