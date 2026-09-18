#!/usr/bin/env bash
# ============================================================================ #
#  首 token 坍塌实验：model × benchmark 热力图
# ============================================================================
#
#  用法：
#    bash run_collapse_heatmap.sh
#
#  说明：
#    - Qwen3.5 系列用 swift 环境（transformers 版本更高），其余用 verl 环境
#    - 每个模型跑完后结果写入同一个 collapse_cache.pkl，可断点续跑
#    - 全部跑完后自动用 --plot-only 出图
#    - 如需单独重跑某个模型，删掉对应缓存条目或直接重跑即可（会覆盖）
# ============================================================================ #
set -euo pipefail
cd /data/chenyang2/verl

# ------------------------------------------------------------------
# 1. verl 环境：跑非 Qwen3.5 的模型
#    skip_tokens: DeepSeek-R1-Distill-Qwen-1.5B=2, Qwen3-8B=2,
#                glm-4-9b-chat=1, Mistral-7B-Instruct-v0.3=1,
#                Phi-3.5-mini-instruct=0, DeepSeek-R1-Distill-Llama-8B=2
# ------------------------------------------------------------------
echo "==========  [1/2] verl 环境 =========="
/data/chenyang2/conda_envs/verl/bin/python examples/grpo_trainer/plot_collapse_heatmap.py \
    --model /data/chenyang2/Qwen3-8B,1,2 \
    --model /data/chenyang2/models/DeepSeek-R1-Distill-Qwen-1.5B,1,2 \
    --model /data/chenyang2/models/glm-4-9b-chat,1,1 \
    --model /data/chenyang2/models/Mistral-7B-Instruct-v0.3,1,1 \
    --model /data/chenyang2/models/Phi-3.5-mini-instruct,1,0 \
    --model /data/chenyang2/models/DeepSeek-R1-Distill-Llama-8B,1,2 \
    --data-dir /data/chenyang2/verl/data \
    --num-prompts 300 \
    --gpu-mem-util 0.85 \
    --max-model-len 8192 \
    --metric top1 \
    --cache collapse_cache.pkl \
    --out collapse_heatmap_top1.png

# ------------------------------------------------------------------
# 2. swift 环境：跑 Qwen3.5 系列
#    skip_tokens: Qwen3.5-4B=0, Qwen3.5-35B-A3B=2
# ------------------------------------------------------------------
echo "==========  [2/2] swift 环境 =========="
/data/chenyang2/conda_envs/swift/bin/python examples/grpo_trainer/plot_collapse_heatmap.py \
    --model /data/chenyang2/models/Qwen3.5-4B,1,0 \
    --model /data/chenyang2/models/Qwen3.5-35B-A3B,4,2 \
    --data-dir /data/chenyang2/verl/data \
    --num-prompts 300 \
    --gpu-mem-util 0.85 \
    --max-model-len 8192 \
    --metric top1 \
    --cache collapse_cache.pkl \
    --out collapse_heatmap_top1.png

# ------------------------------------------------------------------
# 3. 出图（从 cache 读取所有已跑过的模型）
# ------------------------------------------------------------------
echo "==========  画图  =========="
/data/chenyang2/conda_envs/verl/bin/python examples/grpo_trainer/plot_collapse_heatmap.py \
    --plot-only \
    --metric top1 \
    --out collapse_heatmap_top1.png

echo "==========  完成！==========="
echo "热力图:  /data/chenyang2/verl/collapse_heatmap_top1.png"
echo "数据表:  /data/chenyang2/verl/collapse_heatmap_top1.csv"
