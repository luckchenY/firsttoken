# Forced-First-Token 完整执行指南

## 概述

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
    --data /data/chenyang2/router_data_batches/batch_0000.pt \
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

## Step 5: 合并数据与切分训练/测试集

`collect_router_data.py` 的 batch 模式会生成多个 `batch_*.pt` 文件，训练前需要合并。
`merge_and_split.py` 会合并所有 batch 并按数据源分层切分为 train/test。

```bash
cd /workspace/firsttoken

# 合并所有 batch 并切分（10% test，按数据源分层）
python examples/grpo_trainer/merge_and_split.py \
    --input-dir router_data_5000 \
    --output-dir router_data_5000/split \
    --test-ratio 0.1 --seed 42
```

**输出**：
- `router_data_5000/split/train.pt`：训练集（~90%）
- `router_data_5000/split/test.pt`：测试集（~10%，与训练集不重叠）
- `router_data_5000/split/all.pt`：全部数据

如果只需要合并（不切分），用 `merge_router_data.py`：

```bash
python examples/grpo_trainer/merge_router_data.py \
    --input-dir router_data_5000 \
    --output router_data_5000/merged.pt --shuffle
```

---

## Step 6: 评测 Router

`eval_router.py` 提供两种评测模式：

### 6.1 Fast 模式（快速，推荐先用）

直接用已收集的 reward 矩阵，router 预测 token，对照 reward 矩阵看选中的 token 是否正确。
**不需要生成**，只需 HF 模型抽 hidden state，几秒到几分钟完成。

```bash
cd /workspace/firsttoken

python examples/grpo_trainer/eval_router.py \
    --model /workspace/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data router_data_5000/split/test.pt
```

**输出对比**：
- Router accuracy：router 选中的 token 是否正确
- Random baseline：随机选 token 的平均正确率
- Majority baseline：总是选全局最优 token 的正确率
- Oracle：上界（如果总是选对 token）

### 6.2 Generation 模式（真实生成，更准确）

用 vLLM 真实生成，对比三种策略的实际正确率：
- **Normal**：不强制首 token（baseline）
- **Random forced**：随机强制首 token
- **Router forced**：router 选择首 token

分两阶段执行（避免 GPU 显存冲突）：

```bash
cd /workspace/firsttoken

# Phase 1: HF 模型抽 hidden state + router 选 token（~1 分钟）
python examples/grpo_trainer/eval_router.py \
    --model /workspace/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data router_data_5000/split/test.pt \
    --generate --step select \
    --num-prompts 0 \
    --batch-size 8 \
    --tmp-file router_data_5000/split/eval_tmp.pt

# Phase 2: vLLM 生成 + 评分（较慢，取决于数据量）
python examples/grpo_trainer/eval_router.py \
    --model /workspace/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data router_data_5000/split/test.pt \
    --generate --step generate \
    --tp 1 --gpu-mem-util 0.9 \
    --max-tokens 8192 --max-model-len 16384 \
    --tmp-file router_data_5000/split/eval_tmp.pt
```

**参数说明**：
- `--num-prompts 0`：使用全部测试数据（设为正数则采样）
- `--tp 1`：张量并行大小（单卡设为 1）
- `--max-tokens 8192`：最大生成长度
- `--max-model-len 16384`：prompt + 生成总长度上限
- `--batch-size 8`：Phase 1 hidden state 提取的 batch size

**输出**：每组策略的正确率 + 各数据源分项对比。

### 6.3 在外部 Benchmark 上评测

评测 router 在未见过的 benchmark 上的泛化能力。支持以下数据集：

| Benchmark | 数据集 | 预处理脚本 | Reward 类型 |
|---|---|---|---|
| AIME 2024/25/26 | `math-ai/aime24`, `aime25`, `aime26` | `aime.py` | 数值匹配 (math_dapo) |
| GPQA-Diamond | `Idavidrein/gpqa` (gated) | `gpqa_diamond.py` | 多选字母 (mmlu_pro) |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | `mmlu_pro.py` | 多选字母 (mmlu_pro) |
| MuSR | `TAUR-Lab/MuSR` | `musr.py` | 多选字母 (mmlu_pro) |
| BBEH | `BBEH/bbeh` | `bbeh.py` | 自由文本 EM |

#### 下载 Benchmark 数据

```bash
cd /workspace/firsttoken
export HF_ENDPOINT=https://hf-mirror.com

# AIME（90 题，合并 24/25/26）
python examples/data_preprocess/aime.py --local_save_dir data/aime

# MMLU-Pro（12032 test）
python examples/data_preprocess/mmlu_pro.py --local_save_dir data/mmlu_pro

# MuSR（756 题，3 个子集）
python examples/data_preprocess/musr.py --local_save_dir data/musr

# BBEH mini（460 题，每任务 20 题）
python examples/data_preprocess/bbeh.py --local_save_dir data/bbeh --mini-only

# GPQA-Diamond（需要 HF 认证 + 申请访问权限）
# 1. 去 https://huggingface.co/datasets/Idavidrein/gpqa 申请访问
# 2. huggingface-cli login  # 粘贴 HF token
# 3. python examples/data_preprocess/gpqa_diamond.py --local_save_dir data/gpqa_diamond
```

#### 创建组合测试集

从各 benchmark 采样，合并为一个 parquet 文件：

```bash
cd /workspace/firsttoken
python3 - <<'PY'
import pandas as pd, json, numpy as np

def to_jsonable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer, np.floating)): return v.item()
    if isinstance(v, dict): return {k: to_jsonable(vv) for k, vv in v.items()}
    if isinstance(v, list): return [to_jsonable(x) for x in v]
    return v

samples = [
    ("data/aime/test.parquet", None),        # 全部 90
    ("data/mmlu_pro/test.parquet", 200),      # 采样 200
    ("data/musr/test.parquet", 200),          # 采样 200
    ("data/bbeh/test_mini.parquet", 200),     # 采样 200
    # ("data/gpqa_diamond/test.parquet", 198), # GPQA 全部 198
]
dfs = []
for path, n in samples:
    df = pd.read_parquet(path)
    if n and len(df) > n: df = df.sample(n=n, random_state=42)
    for col in ["reward_model", "extra_info", "prompt"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(to_jsonable(x)) if isinstance(x, (dict, list)) else x)
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet("data/benchmark_eval_combined.parquet")
print(f"Combined: {len(combined)} rows")
PY
```

#### 运行 Benchmark 评测

```bash
cd /workspace/firsttoken

# Phase 1: router 选 token
python examples/grpo_trainer/eval_router.py \
    --model /workspace/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data data/benchmark_eval_combined.parquet \
    --generate --step select \
    --num-prompts 0 --batch-size 8 \
    --tmp-file router_data_5000/split/benchmark_eval_tmp.pt

# Phase 2: vLLM 生成 + 评分
python examples/grpo_trainer/eval_router.py \
    --model /workspace/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data data/benchmark_eval_combined.parquet \
    --generate --step generate \
    --tp 1 --gpu-mem-util 0.9 \
    --max-tokens 8192 --max-model-len 16384 \
    --tmp-file router_data_5000/split/benchmark_eval_tmp.pt
```

**注意**：`eval_router.py` 的 generation 模式支持 `.pt` 文件（已 chat-templated）和 parquet 文件（自动 apply chat template）。Reward 分发复用 `collect_router_data.py` 的 `compute_score`，确保所有数据集的评分逻辑一致。

---

## Step 7: GRPO 训练

用 forced-first-token GRPO 算法训练 Qwen3-8B。

### 7.1 不使用 Router（全量 re-rollout）

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

### 7.2 使用 Router（智能选择 token）

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

### 7.3 可调参数

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
| `collect_router_data.py` | 一次性收集 router 训练数据（单阶段，支持 batch） |
| `merge_router_data.py` | 合并多个 batch_*.pt 为单个 .pt 文件 |
| `merge_and_split.py` | 合并 batch 并按数据源分层切分 train/test |
| `train_router.py` | 训练 Router MLP |
| `eval_router.py` | 评测 Router（Fast 模式 + Generation 模式） |
| `router_manager.py` | Router 推理（训练时调用） |
| `forced_first_token_agent_loop.py` | 自定义 agent loop（forced token 注入） |
| `forced_first_token_agent.yaml` | Agent loop 配置 |
| `forced_first_token_grpo_trainer.py` | GRPO 训练器（re-rollout 逻辑） |
| `run_qwen3_8b_forced_first_token_grpo.sh` | GRPO 训练启动脚本 |
| `test_forced_first_token.py` | 评估脚本（三阶段对比，可选） |

### 数据预处理脚本（`examples/data_preprocess/`）

| 文件 | 用途 |
|---|---|
| `gsm8k.py` | GSM8K 数据集预处理 |
| `math_dataset.py` | MATH 数据集预处理 |
| `arc_challenge.py` | ARC-Challenge 数据集预处理 |
| `logiqa2.py` | LogiQA2.0 数据集预处理 |
| `drop.py` | DROP 数据集预处理 |
| `taco.py` | TACO 代码数据集预处理 |
| `aime.py` | AIME 2024/25/26 合并预处理 |
| `mmlu_pro.py` | MMLU-Pro 数据集预处理 |
| `gpqa_diamond.py` | GPQA-Diamond 数据集预处理 |
| `musr.py` | MuSR 数据集预处理 |
| `bbeh.py` | BBEH 数据集预处理 |

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

# 2. 收集 router 数据（batch 模式，2-3 小时）
python examples/grpo_trainer/collect_router_data.py \
    --model /data/chenyang2/Qwen3-8B \
    --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
    --num-prompts 5000 \
    --tp 4 --gpu-mem-util 0.9 --max-tokens 8192 --max-model-len 16384 \
    --save-dir /data/chenyang2/router_data_5000 --batch-size 200

# 3. 合并数据 + 切分 train/test
python examples/grpo_trainer/merge_and_split.py \
    --input-dir router_data_5000 \
    --output-dir router_data_5000/split \
    --test-ratio 0.1 --seed 42

# 4. 训练 router（几分钟）
python examples/grpo_trainer/train_router.py \
    --model /data/chenyang2/Qwen3-8B \
    --data router_data_5000/split/train.pt \
    --output router_data_5000/router_weights.pt \
    --epochs 50 --lr 1e-3 --bad-weight 2.0

# 5. 评测 router（Fast 模式，快速验证）
python examples/grpo_trainer/eval_router.py \
    --model /data/chenyang2/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data router_data_5000/split/test.pt

# 6. 评测 router（Generation 模式，真实生成）
# Phase 1
python examples/grpo_trainer/eval_router.py \
    --model /data/chenyang2/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data router_data_5000/split/test.pt \
    --generate --step select --num-prompts 0 --batch-size 8 \
    --tmp-file router_data_5000/split/eval_tmp.pt
# Phase 2
python examples/grpo_trainer/eval_router.py \
    --model /data/chenyang2/Qwen3-8B \
    --router router_data_5000/router_weights.pt \
    --data router_data_5000/split/test.pt \
    --generate --step generate --tp 1 --gpu-mem-util 0.9 \
    --max-tokens 8192 --max-model-len 16384 \
    --tmp-file router_data_5000/split/eval_tmp.pt

# # 7. GRPO 训练
# export FORCED_FIRST_TOKEN_LIST="<TOKEN_LIST>"
# export ROUTER_WEIGHTS_PATH=/data/chenyang2/router_weights.pt
# export VERL_USE_UV=0
# export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_P2P_DISABLE=1
# NGPUS_PER_NODE=4 bash examples/grpo_trainer/run_qwen3_8b_forced_first_token_grpo.sh
```

---
