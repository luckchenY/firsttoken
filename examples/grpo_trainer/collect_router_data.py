#!/usr/bin/env python3
"""Collect router training data in a single phase.

For each prompt, try K forced first tokens (1 rollout each), record reward vector.
No Phase 1/2/3 -- just one efficient pass. Output is directly usable by train_router.py.

Usage:
  python collect_router_data.py \
      --model /data/chenyang2/Qwen3-8B \
      --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
      --num-prompts 500 \
      --forced-tokens 32313,71486,93217,106287,35439,4416,3925,16910 \
      --tp 4 --gpu-mem-util 0.9 --max-tokens 4096 \
      --save-data /data/chenyang2/router_data.pt
"""

import argparse
import os
import sys
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


def main():
    parser = argparse.ArgumentParser(description="Collect router training data (single phase)")
    parser.add_argument("--model", default="/data/chenyang2/Qwen3-8B")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, default=500,
                        help="Number of prompts to sample (-1 = all, shuffled)")
    parser.add_argument("--forced-tokens", type=str, required=True,
                        help="Comma-separated token ids, e.g. 32313,71486,...")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--save-data", type=str, required=True,
                        help="Save router training data to this file (.pt)")
    args = parser.parse_args()

    forced_token_list = [int(x.strip()) for x in args.forced_tokens.split(",")]
    K = len(forced_token_list)
    print(f"K = {K} forced tokens: {forced_token_list}")

    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  decoded: {[repr(tokenizer.decode([t])) for t in forced_token_list]}")

    # 2. Load data
    dfs = [pd.read_parquet(p) for p in args.data]
    df = pd.concat(dfs, ignore_index=True)
    total = len(df)
    if args.num_prompts < 0:
        df = df.sample(n=total, random_state=42)
        print(f"Using all {total} prompts (shuffled)")
    elif len(df) > args.num_prompts:
        df = df.sample(n=args.num_prompts, random_state=42)
        print(f"Sampled {args.num_prompts} prompts from {total}")
    else:
        print(f"Using all {total} prompts")

    num_prompts = len(df)

    # 3. Build chat-templated prompts
    #    Data format: "prompt" column = list of message dicts [{"role": ..., "content": ...}]
    def parse_messages(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            import ast
            return ast.literal_eval(val)
        return list(val)

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

    print(f"Built {num_prompts} chat-templated prompts")

    # 4. Build all rollout prompts: each prompt x K forced tokens
    #    Layout: [prompt0_tok0, prompt0_tok1, ..., prompt0_tokK-1,
    #             prompt1_tok0, prompt1_tok1, ..., prompt1_tokK-1, ...]
    #    Total = num_prompts * K trajectories
    all_prompt_objs = []
    all_gts = []
    all_ds = []
    all_token_ids = []  # which forced token for each trajectory
    for p_idx in range(num_prompts):
        prompt_ids = tokenizer.encode(prompts_text[p_idx], add_special_tokens=False)
        for k in range(K):
            forced_tok = forced_token_list[k]
            ids = prompt_ids + [forced_tok]
            all_prompt_objs.append(TokensPrompt(prompt_token_ids=ids))
            all_gts.append(ground_truths[p_idx])
            all_ds.append(data_sources[p_idx])
            all_token_ids.append(forced_tok)

    total_trajectories = len(all_prompt_objs)
    print(f"\nTotal trajectories: {total_trajectories} ({num_prompts} prompts x {K} tokens)")

    # 5. Generate all at once (vLLM batches internally)
    print(f"\nGenerating all {total_trajectories} trajectories ...")
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
    outputs = llm.generate(all_prompt_objs, sampling_params=sp)

    # 6. Score all trajectories
    print(f"\nScoring {total_trajectories} trajectories ...")
    scores = []
    for i, out in enumerate(outputs):
        resp = out.outputs[0].text
        score = compute_score(resp, all_ds[i], all_gts[i])
        scores.append(score)

    # 7. Build reward matrix [num_prompts, K]
    #    reward_matrix[p_idx, k] = score of prompt p_idx with forced token k
    reward_matrix = torch.zeros(num_prompts, K, dtype=torch.float32)
    for i in range(total_trajectories):
        p_idx = i // K
        k = i % K
        reward_matrix[p_idx, k] = scores[i]

    # 8. Report statistics
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Prompts: {num_prompts}")
    print(f"  Candidate tokens: {K}")
    print(f"  Total trajectories: {total_trajectories}")
    n_has_correct = (reward_matrix.sum(dim=1) > 0).sum().item()
    print(f"  Prompts with >=1 correct token: {n_has_correct}/{num_prompts} "
          f"({n_has_correct/num_prompts*100:.1f}%)")
    print(f"  Correct per token:")
    for k in range(K):
        tok = forced_token_list[k]
        n_correct = reward_matrix[:, k].sum().item()
        print(f"    token {tok:>8} ({repr(tokenizer.decode([tok]))}): "
              f"{n_correct}/{num_prompts} ({n_correct/num_prompts*100:.1f}%)")

    # 9. Save
    router_data = {
        "prompts_text": prompts_text,
        "ground_truths": ground_truths,
        "data_sources": data_sources,
        "forced_token_ids": forced_token_list,
        "rewards": reward_matrix,
    }
    torch.save(router_data, args.save_data)
    print(f"\nRouter training data saved to {args.save_data}")
    print(f"  Reward matrix shape: {reward_matrix.shape}  (N={num_prompts}, K={K})")
    if n_has_correct < 30:
        print(f"  WARNING: only {n_has_correct} usable training samples. "
              f"Consider increasing --num-prompts.")


if __name__ == "__main__":
    main()
