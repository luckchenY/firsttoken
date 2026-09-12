# Forced-First-Token GRPO 完整执行指南

## 概述

本指南涵盖从数据下载到 GRPO 训练的完整流程。

**算法**：标准 GRPO + 对全错 group 做 forced first token re-rollout。如果 re-rollout 有正确答案，替换原 group 的 position 0。可选使用训练好的 Router 自动选择最佳 forced token。

**环境**：
- 机器：`eez233` (HKUST), 4x GPU (无 NVLink)
- Conda 环境：`/data/chenyang2/conda_envs/verl`
- 模型：`/data/chenyang2/Qwen3-8B` (本地副本)
- verl 仓库：`/data/chenyang2/verl`

---

## Step 0: 环境准备

```bash
# 激活 conda 环境
conda activate /data/chenyang2/conda_envs/verl

# 进入 verl 仓库
cd /data/chenyang2/verl

# 确认关键依赖
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## Step 1: 下载模型和数据

### 1.1 下载 Qwen3-8B 模型（如果还没有）

```bash
# 使用 HF 镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

# 下载到本地
huggingface-cli download Qwen/Qwen3-8B --local-dir /data/chenyang2/Qwen3-8B
```

### 1.2 下载 GSM8K + MATH 数据集

verl 自带数据预处理脚本，在 `examples/data_preprocess/` 目录下：

```bash
cd /data/chenyang2/verl
export HF_ENDPOINT=https://hf-mirror.com

# GSM8K (1319 test, 7473 train)
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# MATH (5000 test, 7500 train)
python examples/data_preprocess/math_dataset.py --local_save_dir ~/data/math
```

### 1.3 验证数据

```bash
python -c "
import pandas as pd
for name, path in [('GSM8K test', '~/data/gsm8k/test.parquet'),
                   ('MATH test', '~/data/math/test.parquet')]:
    df = pd.read_parquet(path)
    print(f'{name}: {df.shape[0]} rows, columns={df.columns.tolist()}')
    print(f'  data_source: {df[\"data_source\"].unique()}')
    print(f'  ground_truth sample: {df[\"reward_model\"].iloc[0]}')
"
```

---

## Step 2: 统计首 Token 分布

用 vLLM logprobs 统计模型在第 3 个 token 位置（跳过 `imd` + `\n`）的概率分布，取概率质量最高的 K 个 token。

```bash
cd /data/chenyang2/verl

python examples/grpo_trainer/analyze_first_tokens.py \
    --model /data/chenyang2/Qwen3-8B \
    --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
    --num-prompts 1000 \
    --top-k 20 \
    --skip-tokens 2 \
    --tp 4 \
    --gpu-mem-util 0.9
```

**参数说明**：
- `--num-prompts 1000`：用 1000 条 prompt 做统计，样本越多越稳定
- `--top-k 20`：取概率质量最高的 20 个 token
- `--skip-tokens 2`：分析第 3 个 token 位置（跳过 chat template 前缀生成的 `imd` + `\n`）
- `--tp 4`：4 卡张量并行
- 只生成 3 个 token（`max_tokens=skip_tokens+1`），几分钟完成

**输出**：脚本最后会打印：
```
FORCED_FIRST_TOKEN_LIST=32313,71486,93217,...
```
**记下这个 token ID 列表**，后续步骤要用。

---

## Step 3: 收集 Router 训练数据

对每个 prompt，用 K 个 forced token 各做 1 次 rollout，记录每个 token 是否导致正确答案。一次 vLLM generate 搞定，直接得到 router 训练数据。

```bash
cd /data/chenyang2/verl

# 把下面 <TOKEN_LIST> 替换为 Step 2 输出的 token ID 列表
python examples/grpo_trainer/collect_router_data.py \
    --model /data/chenyang2/Qwen3-8B \
    --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
    --num-prompts 1000 \
    --forced-tokens <TOKEN_LIST> \
    --tp 4 \
    --gpu-mem-util 0.9 \
    --max-tokens 4096 \
    --max-model-len 8192 \
    --save-data /data/chenyang2/router_data.pt
```

**参数说明**：
- `--num-prompts 1000`：1000 个 prompt（打乱顺序，GSM8K/MATH 混合）
- `--forced-tokens`：Step 2 统计出的 K 个 token ID（逗号分隔）
- `--max-tokens 4096`：Qwen3 推理链很长，需要 4096
- `--max-model-len 8192`：prompt + 生成总长度上限

**总量**：1000 prompts × K tokens = K×1000 条轨迹（如 K=20 则 20,000 条），预计 2-3 小时。

**输出**：`/data/chenyang2/router_data.pt`，包含：
- `prompts_text`：1000 条 chat-templated prompt
- `rewards`：`[1000, K]` 矩阵，`rewards[i][k] = 1.0` 表示 prompt i 用 token k 做对了
- `forced_token_ids`：K 个候选 token ID

---

## Step 4: 训练 Router MLP

用收集到的数据训练一个小 MLP：Qwen3-8B hidden state [4096] → MLP → K 个 token 的 logits。

```bash
cd /data/chenyang2/verl

python examples/grpo_trainer/train_router.py \
    --model /data/chenyang2/Qwen3-8B \
    --data /data/chenyang2/router_data.pt \
    --output /data/chenyang2/router_weights.pt \
    --epochs 50 \
    --lr 1e-3 \
    --batch-size 64 \
    --hidden-dim 1024 \
    --bad-weight 2.0
```

**参数说明**：
- `--model`：用于提取 hidden state 的模型（Qwen3-8B）
- `--data`：Step 3 保存的 router 数据
- `--output`：训练好的 router 权重保存路径
- `--epochs 50`：训练 50 轮
- `--bad-weight 2.0`：bad group 样本权重 ×2（训练时 router 只对 bad group 调用）

**输出**：`/data/chenyang2/router_weights.pt`，包含：
- `router_state_dict`：MLP 权重
- `config`：模型配置（input_dim, hidden_dim, num_candidates）
- `forced_token_ids`：K 个候选 token ID

**预期输出**：
```
Final selection accuracy: 0.xxxx (N/total)
  Bad groups:  0.xxxx
  Good groups: 0.xxxx
  (Random baseline would be ~0.0500 for K=20)
```

---

## Step 5: GRPO 训练

用 forced-first-token GRPO 算法训练 Qwen3-8B。

### 5.1 不使用 Router（全量 re-rollout）

对每个全错 group，用 token list 中的所有 token 依次尝试 re-rollout。

```bash
cd /data/chenyang2/verl

# 设置 forced token list（与 Step 2 一致）
export FORCED_FIRST_TOKEN_LIST="<TOKEN_LIST>"

# 使用 conda python（不是 uv）
export VERL_USE_UV=0

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# NCCL 设置（无 NVLink 的机器需要）
export NCCL_P2P_DISABLE=1

# 运行训练
bash examples/grpo_trainer/run_qwen3_8b_forced_first_token_grpo.sh
```

### 5.2 使用 Router（智能选择 token）

设置 `ROUTER_WEIGHTS_PATH` 环境变量，训练时对每个全错 group 用 router 选择 1 个最佳 token（而不是试所有 K 个）。

```bash
cd /data/chenyang2/verl

# Forced token list + Router 权重
export FORCED_FIRST_TOKEN_LIST="<TOKEN_LIST>"
export ROUTER_WEIGHTS_PATH=/data/chenyang2/router_weights.pt

export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_P2P_DISABLE=1

# 运行训练
bash examples/grpo_trainer/run_qwen3_8b_forced_first_token_grpo.sh
```

### 5.3 可调参数

训练脚本支持以下环境变量覆盖默认值：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `NGPUS_PER_NODE` | 8 | GPU 数量（本机设为 4） |
| `ROLLOUT_N` | 5 | 每 prompt 的 rollout 数 |
| `ROLLOUT_TP` | 2 | rollout 张量并行大小 |
| `TRAIN_BATCH_SIZE` | 1024 | 训练 batch size |
| `ACTOR_LR` | 1e-6 | Actor 学习率 |
| `TOTAL_EPOCHS` | 15 | 总训练轮数 |
| `MAX_RESPONSE_LENGTH` | 2048 | 最大生成长度 |
| `ROLLOUT_GPU_MEM_UTIL` | 0.6 | rollout GPU 显存占用 |

例如用 4 卡训练：
```bash
NGPUS_PER_NODE=4 ROLLOUT_TP=2 bash examples/grpo_trainer/run_qwen3_8b_forced_first_token_grpo.sh
```

---

## 文件清单

| 文件 | 用途 |
|---|---|
| `analyze_first_tokens.py` | 统计首 token 概率分布，输出 top-K token ID |
| `collect_router_data.py` | 一次性收集 router 训练数据（单阶段） |
| `train_router.py` | 训练 Router MLP |
| `router_manager.py` | Router 推理（训练时调用） |
| `forced_first_token_agent_loop.py` | 自定义 agent loop（forced token 注入） |
| `forced_first_token_agent.yaml` | Agent loop 配置 |
| `forced_first_token_grpo_trainer.py` | GRPO 训练器（re-rollout 逻辑） |
| `run_qwen3_8b_forced_first_token_grpo.sh` | GRPO 训练启动脚本 |
| `test_forced_first_token.py` | 评估脚本（三阶段对比，可选） |

---

## 完整流程速查

```bash
# 0. 环境
conda activate /data/chenyang2/conda_envs/verl
cd /data/chenyang2/verl

# 1. 统计首 token（几分钟）
python examples/grpo_trainer/analyze_first_tokens.py \
    --model /data/chenyang2/Qwen3-8B \
    --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
    --num-prompts 1000 --top-k 20 --skip-tokens 2 \
    --tp 4 --gpu-mem-util 0.9
# → 记下输出的 FORCED_FIRST_TOKEN_LIST

# 2. 收集 router 数据（2-3 小时）
python examples/grpo_trainer/collect_router_data.py \
    --model /data/chenyang2/Qwen3-8B \
    --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
    --num-prompts 1000 \
    --forced-tokens <TOKEN_LIST> \
    --tp 4 --gpu-mem-util 0.9 --max-tokens 4096 \
    --save-data /data/chenyang2/router_data.pt

# 3. 训练 router（几分钟）
python examples/grpo_trainer/train_router.py \
    --model /data/chenyang2/Qwen3-8B \
    --data /data/chenyang2/router_data.pt \
    --output /data/chenyang2/router_weights.pt \
    --epochs 50 --lr 1e-3 --bad-weight 2.0

# 4. GRPO 训练
export FORCED_FIRST_TOKEN_LIST="<TOKEN_LIST>"
export ROUTER_WEIGHTS_PATH=/data/chenyang2/router_weights.pt
export VERL_USE_UV=0
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_P2P_DISABLE=1
NGPUS_PER_NODE=4 bash examples/grpo_trainer/run_qwen3_8b_forced_first_token_grpo.sh
```

---

## 常见问题

### Q: flash_attn 报 GLIBC_2.32 not found
本机 GLIBC 是 2.31，预编译 wheel 需要 2.32。需要从源码编译：
```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-binary flash-attn --no-build-isolation
```

### Q: NCCL P2P error (Cuda failure 217)
GPU 之间无 NVLink，需要禁用 P2P：
```bash
export NCCL_P2P_DISABLE=1
```
已在训练脚本和 Python 代码中设置。

### Q: vLLM 生成越来越慢
数据没有打乱（GSM8K 短题在前，MATH 长题在后）。`collect_router_data.py` 已修复：`--num-prompts -1` 时也会打乱顺序。

### Q: uv: command not found
本机没有 uv，用 conda python：
```bash
export VERL_USE_UV=0
```

### Q: HuggingFace 下载卡住
使用国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
