#!/usr/bin/env bash
# GRPO variant | Qwen3-8B | FSDP training | Forced-First-Token re-rollout
#
# Algorithm: standard GRPO, but for every prompt group whose n rollouts are ALL
# wrong, re-rollout the group with forced first tokens (read from
# FORCED_FIRST_TOKEN_LIST, one per trajectory in order).  If any re-rolled
# trajectory is correct, replace position 0 of the original group with the
# smallest-index correct one.  Everything else is identical to GRPO.
#
# Knobs (in addition to run_qwen3_8b_fsdp.sh):
#   FORCED_FIRST_TOKEN_LIST   comma-separated token ids, e.g. "151665,151666"
#                             (required; the algorithm is a no-op without it)
#
# Run from the verl repo root.

set -xeuo pipefail

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH=${MODEL_PATH:-/data/chenyang2/Qwen3-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-5}
sp_size=${SP_SIZE:-1}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

# ---- forced-first-token list (REQUIRED) ----
# Example Qwen3 token ids; replace with your own. Length should be >= rollout_n
# (wraps with modulo if shorter).
export FORCED_FIRST_TOKEN_LIST=${FORCED_FIRST_TOKEN_LIST:-32313,71486,93217,106287,35439,4416,3925,16910}

# HuggingFace mirror (needed if huggingface.co is unreachable)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# Disable NCCL P2P (needed when GPUs lack NVLink, e.g. cloud/PCIe setups)
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_forced_first_token_grpo_$(date +%Y%m%d_%H%M)}

# absolute path to the agent-loop config (safe for multi-node)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_LOOP_CONFIG="${SCRIPT_DIR}/forced_first_token_agent.yaml"
########################### end user-adjustable ###########################

actor_param_offload=False
actor_optimizer_offload=False

EXTRA=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
)

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$HOME/data/gsm8k/train.parquet', '$HOME/data/math/train.parquet']"
    data.val_files="['$HOME/data/gsm8k/test.parquet', '$HOME/data/math/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # flash_attention_2 (compiled from source on this machine, compatible with GLIBC 2.31)
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
    # Load model in bf16 so flash_attention_2 works (default fp32 crashes FA2)
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    # Force the legacy V0 path (RayPPOTrainer) which our subclass extends.
    trainer.use_v1=False
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): GPU vllm/sglang x fsdp run the driver
# and every Ray worker through `uv run` on the matching extras of the committed
# uv.lock; other backends fall back to ambient python.  Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ] && { [ "${INFER_BACKEND}" = vllm ] || [ "${INFER_BACKEND}" = sglang ]; }; then
    LAUNCH=(uv run --frozen --all-packages --extra "${INFER_BACKEND}" --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra ${INFER_BACKEND} --extra fsdp")
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

"${LAUNCH[@]}" -m examples.grpo_trainer.forced_first_token_grpo_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
