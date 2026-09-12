#!/usr/bin/env python3
"""Collect router training data in a single phase, with batch support.

For each prompt, try K forced first tokens (1 rollout each), record reward vector.
Supports batch processing: split all prompts into chunks of --batch-size, save each
chunk to a separate file in --save-dir. Skips already-completed batches (resume).

Usage (batch mode, recommended):
  python collect_router_data.py \
      --model /data/chenyang2/Qwen3-8B \
      --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
      --num-prompts -1 \
      --forced-tokens 32313,71486,... \
      --tp 4 --gpu-mem-util 0.9 --max-tokens 4096 \
      --save-dir /data/chenyang2/router_data_batches \
      --batch-size 1000

Usage (single-file mode):
  python collect_router_data.py \
      --model /data/chenyang2/Qwen3-8B \
      --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
      --num-prompts 500 \
      --forced-tokens 32313,71486,... \
      --tp 4 --gpu-mem-util 0.9 --max-tokens 4096 \
      --save-data /data/chenyang2/router_data.pt

Usage (merge batches into one file):
  python collect_router_data.py --merge \
      --save-dir /data/chenyang2/router_data_batches \
      --save-data /data/chenyang2/router_data.pt
"""

import argparse
import os
import sys
import glob
import pandas as pd
import torch
from collections import Counter
from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from verl.utils.reward_score import default_compute_score


def compute_score(response, data_source, ground_truth):
    try:
        score = default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
        )
        return float(score) if score is not None else 0.0
    except Exception as e:
        print(f"[compute_score] error for data_source={data_source}: {e}")
        return 0.0


def parse_messages(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        import ast
        return ast.literal_eval(val)
    return list(val)


def load_and_prepare_data(data_paths, num_prompts, tokenizer):
    """Load parquet files, shuffle, build chat-templated prompts."""
    dfs = [pd.read_parquet(p) for p in data_paths]
    df = pd.concat(dfs, ignore_index=True)
    total = len(df)
    if num_prompts < 0:
        df = df.sample(n=total, random_state=42)
        print(f"Using all {total} prompts (shuffled)")
    elif total > num_prompts:
        df = df.sample(n=num_prompts, random_state=42)
        print(f"Sampled {num_prompts} prompts from {total}")
    else:
        print(f"Using all {total} prompts")

    prompts_text = []
    ground_truths = []
    data_sources = []
    for _, row in df.iterrows():
        messages = parse_messages(row["prompt"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts_text.append(text)
        rm = row.get("reward_model", {})
        ground_truths.append(rm.get("ground_truth", "") if isinstance(rm, dict) else "")
        data_sources.append(row.get("data_source", ""))

    return prompts_text, ground_truths, data_sources


def process_batch(llm, sp, tokenizer, forced_token_list,
                   prompts_text, ground_truths, data_sources, batch_idx):
    """Process one batch of prompts through vLLM, return reward matrix."""
    K = len(forced_token_list)
    n = len(prompts_text)

    # Build all prompt objects: each prompt x K forced tokens
    all_prompt_objs = []
    all_gts = []
    all_ds = []
    for p_idx in range(n):
        prompt_ids = tokenizer.encode(prompts_text[p_idx], add_special_tokens=False)
        for k in range(K):
            forced_tok = forced_token_list[k]
            ids = prompt_ids + [forced_tok]
            all_prompt_objs.append(TokensPrompt(prompt_token_ids=ids))
            all_gts.append(ground_truths[p_idx])
            all_ds.append(data_sources[p_idx])

    total_traj = len(all_prompt_objs)
    print(f"\n[Batch {batch_idx}] Generating {total_traj} trajectories "
          f"({n} prompts x {K} tokens) ...")

    outputs = llm.generate(all_prompt_objs, sampling_params=sp)

    # Score
    print(f"[Batch {batch_idx}] Scoring {total_traj} trajectories ...")
    scores = []
    for i, out in enumerate(outputs):
        resp = out.outputs[0].text
        score = compute_score(resp, all_ds[i], all_gts[i])
        scores.append(score)

    # Build reward matrix [n, K]
    reward_matrix = torch.zeros(n, K, dtype=torch.float32)
    for i in range(total_traj):
        p_idx = i // K
        k = i % K
        reward_matrix[p_idx, k] = scores[i]

    # Stats
    n_has_correct = (reward_matrix.sum(dim=1) > 0).sum().item()
    print(f"[Batch {batch_idx}] Prompts with >=1 correct token: "
          f"{n_has_correct}/{n} ({n_has_correct/n*100:.1f}%)")
    for k in range(K):
        tok = forced_token_list[k]
        n_correct = int(reward_matrix[:, k].sum().item())
        print(f"  token {tok:>8} ({repr(tokenizer.decode([tok]))}): "
              f"{n_correct}/{n} ({n_correct/n*100:.1f}%)")

    return reward_matrix


def save_batch(path, prompts_text, ground_truths, data_sources,
               forced_token_list, reward_matrix):
    """Save one batch to a .pt file."""
    data = {
        "prompts_text": prompts_text,
        "ground_truths": ground_truths,
        "data_sources": data_sources,
        "forced_token_ids": forced_token_list,
        "rewards": reward_matrix,
    }
    torch.save(data, path)
    print(f"  Saved to {path}  (shape: {reward_matrix.shape})")


def merge_batches(save_dir, save_data):
    """Merge all batch_*.pt files in save_dir into one file."""
    pattern = os.path.join(save_dir, "batch_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No batch files found in {save_dir}")
        return

    print(f"Merging {len(files)} batch files ...")
    all_prompts = []
    all_gts = []
    all_ds = []
    all_rewards = []
    forced_token_ids = None

    for f in files:
        data = torch.load(f, weights_only=False)
        all_prompts.extend(data["prompts_text"])
        all_gts.extend(data["ground_truths"])
        all_ds.extend(data["data_sources"])
        all_rewards.append(data["rewards"])
        forced_token_ids = data["forced_token_ids"]

    rewards = torch.cat(all_rewards, dim=0)
    merged = {
        "prompts_text": all_prompts,
        "ground_truths": all_gts,
        "data_sources": all_ds,
        "forced_token_ids": forced_token_ids,
        "rewards": rewards,
    }
    torch.save(merged, save_data)

    n_total = rewards.shape[0]
    K = rewards.shape[1]
    n_has_correct = (rewards.sum(dim=1) > 0).sum().item()
    print(f"Merged: {n_total} prompts, K={K}")
    print(f"  Prompts with >=1 correct token: {n_has_correct}/{n_total} "
          f"({n_has_correct/n_total*100:.1f}%)")
    print(f"  Saved to {save_data}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect router training data (single phase, batch support)")
    parser.add_argument("--model", default="/data/chenyang2/Qwen3-8B")
    parser.add_argument("--data", nargs="+", help="Parquet data files")
    parser.add_argument("--num-prompts", type=int, default=500,
                        help="Total prompts to process (-1 = all, shuffled)")
    parser.add_argument("--forced-tokens", type=str,
                        help="Comma-separated token ids")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)

    # Single-file mode
    parser.add_argument("--save-data", type=str, default=None,
                        help="Single-file output path (.pt)")

    # Batch mode
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory for batch files (batch mode)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Prompts per batch (0 = no batching, single file)")
    parser.add_argument("--batch-start", type=int, default=0,
                        help="Start from this batch index (for resume)")

    # Merge mode
    parser.add_argument("--merge", action="store_true",
                        help="Merge all batch files in --save-dir into --save-data")

    args = parser.parse_args()

    # ---- Merge mode ----
    if args.merge:
        if not args.save_dir or not args.save_data:
            print("--merge requires --save-dir and --save-data")
            sys.exit(1)
        merge_batches(args.save_dir, args.save_data)
        return

    # ---- Validate ----
    if not args.data:
        print("--data is required (unless --merge)")
        sys.exit(1)
    if not args.forced_tokens:
        print("--forced-tokens is required")
        sys.exit(1)
    if not args.save_data and not args.save_dir:
        print("Need --save-data (single file) or --save-dir (batch mode)")
        sys.exit(1)

    forced_token_list = [int(x.strip()) for x in args.forced_tokens.split(",")]
    K = len(forced_token_list)
    print(f"K = {K} forced tokens: {forced_token_list}")

    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  decoded: {[repr(tokenizer.decode([t])) for t in forced_token_list]}")

    # 2. Load and prepare all data
    prompts_text, ground_truths, data_sources = load_and_prepare_data(
        args.data, args.num_prompts, tokenizer)
    total_prompts = len(prompts_text)
    print(f"Total prompts to process: {total_prompts}")

    # 3. Determine batch mode or single mode
    use_batch = args.save_dir is not None and args.batch_size > 0

    if use_batch:
        os.makedirs(args.save_dir, exist_ok=True)
        num_batches = (total_prompts + args.batch_size - 1) // args.batch_size
        print(f"Batch mode: {num_batches} batches x {args.batch_size} prompts")
        print(f"  Save dir: {args.save_dir}")
        print(f"  Start from batch: {args.batch_start}")
    else:
        num_batches = 1
        args.batch_size = total_prompts
        print(f"Single-file mode: {total_prompts} prompts in one batch")

    # 4. Load vLLM model (once)
    print(f"\nLoading vLLM model {args.model} (tp={args.tp}) ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
    )

    # 5. Process each batch
    all_reward_matrices = []
    for batch_idx in range(args.batch_start, num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, total_prompts)
        batch_prompts = prompts_text[start:end]
        batch_gts = ground_truths[start:end]
        batch_ds = data_sources[start:end]

        print(f"\n{'='*70}")
        print(f"Batch {batch_idx}/{num_batches - 1}  (prompts {start}..{end-1})")
        print(f"{'='*70}")

        # Skip if already done (resume support)
        if use_batch:
            batch_path = os.path.join(args.save_dir, f"batch_{batch_idx:04d}.pt")
            if os.path.exists(batch_path):
                print(f"  Already exists, skipping: {batch_path}")
                data = torch.load(batch_path, weights_only=False)
                all_reward_matrices.append(data["rewards"])
                continue

        # Process
        reward_matrix = process_batch(
            llm, sp, tokenizer, forced_token_list,
            batch_prompts, batch_gts, batch_ds, batch_idx)

        # Save
        if use_batch:
            save_batch(batch_path, batch_prompts, batch_gts, batch_ds,
                        forced_token_list, reward_matrix)
        else:
            save_batch(args.save_data, batch_prompts, batch_gts, batch_ds,
                        forced_token_list, reward_matrix)

        all_reward_matrices.append(reward_matrix)

    # 6. Final summary
    print(f"\n{'='*70}")
    print(f"ALL DONE")
    print(f"{'='*70}")
    all_rewards = torch.cat(all_reward_matrices, dim=0)
    n_total = all_rewards.shape[0]
    n_has_correct = (all_rewards.sum(dim=1) > 0).sum().item()
    print(f"  Total prompts processed: {n_total}")
    print(f"  Prompts with >=1 correct token: {n_has_correct}/{n_total} "
          f"({n_has_correct/n_total*100:.1f}%)")
    print(f"  Correct per token:")
    for k in range(K):
        tok = forced_token_list[k]
        n_correct = int(all_rewards[:, k].sum().item())
        print(f"    token {tok:>8} ({repr(tokenizer.decode([tok]))}): "
              f"{n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")

    if use_batch:
        print(f"\n  Batch files saved in: {args.save_dir}")
        print(f"  To merge: python {sys.argv[0]} --merge "
              f"--save-dir {args.save_dir} --save-data /data/chenyang2/router_data.pt")
    else:
        print(f"\n  Single file: {args.save_data}")


if __name__ == "__main__":
    main()
