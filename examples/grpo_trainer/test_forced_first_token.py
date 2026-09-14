#!/usr/bin/env python3
"""Standalone test: does forcing the first token improve rollout accuracy?

This script does NOT run GRPO training.  It just:
  1. Generates n rollouts per prompt (like GRPO rollout).
  2. Checks correctness using the GSM8K/MATH reward functions.
  3. For groups with NO correct answer, re-generates with forced first tokens.
  4. Checks correctness again.
  5. Reports: original pass-rate vs pass-rate after forced-token replacement.

Usage:
    python3 examples/grpo_trainer/test_forced_first_token.py \
        --model Qwen/Qwen3-8B \
        --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
        --num-prompts 200 --rollout-n 5 \
        --forced-tokens 32313,71486,93217,106287,35439,4416,3925,16910 \
        --tp 1 --gpu-mem-util 0.5
"""

import argparse
import re
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer


def parse_messages(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        import ast
        return ast.literal_eval(val)
    return list(val)


# ---- reward functions (copied from verl/utils/reward_score) ------------------
def extract_gsm8k_solution(solution_str):
    match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if match is None:
        return None
    return match.group(0).split("#### ")[1].replace(",", "")


def extract_boxed_answer(response):
    """Extract the last \\boxed{} answer from a response (works for Qwen3 thinking mode)."""
    idx = response.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(response) and depth > 0:
        if response[end] == "{":
            depth += 1
        elif response[end] == "}":
            depth -= 1
        end += 1
    if depth != 0:
        return None
    return response[start:end - 1].strip()


def gsm8k_compute_score(response, ground_truth):
    # Try #### format first (standard GSM8K), then \boxed{} (Qwen3 thinking mode)
    answer = extract_gsm8k_solution(response)
    if answer is None:
        answer = extract_boxed_answer(response)
    if answer is None:
        # try to find the last number in the response
        numbers = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))
        answer = numbers[-1] if numbers else None
    if answer is None:
        return 0.0
    try:
        return 1.0 if float(answer) == float(ground_truth) else 0.0
    except (ValueError, TypeError):
        return 1.0 if str(answer) == str(ground_truth) else 0.0


def math_compute_score(response, ground_truth):
    # extract \boxed{} answer (Qwen3 thinking mode uses this)
    answer = extract_boxed_answer(response)
    if answer is None:
        return 0.0
    # simple normalization
    gt = str(ground_truth).strip()
    if answer == gt:
        return 1.0
    # try numeric comparison
    try:
        if float(answer) == float(gt):
            return 1.0
    except (ValueError, TypeError):
        pass
    return 0.0


def compute_score(response, data_source, ground_truth):
    """Use verl's official reward functions for exact consistency with training."""
    from verl.utils.reward_score import default_compute_score
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/Qwen3-8B")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--rollout-n", type=int, default=5, help="Rollouts per prompt")
    parser.add_argument("--forced-tokens", type=str, required=True,
                        help="Comma-separated token ids, e.g. 32313,71486,...")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    # max_tokens: 单次生成的最大 token 数（只管生成部分，不含 prompt）
    #   Qwen3 推理模式先生成很长的思维链，2048 会截断，4096 比较安全
    parser.add_argument("--max-tokens", type=int, default=4096)
    # max_model_len: vLLM 的最大序列长度 = prompt + 生成，必须 >= prompt长度 + max_tokens
    #   prompt 约 500-1000 token，加 max_tokens=4096，所以 8192 够用
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.5)
    parser.add_argument("--save-data", type=str, default=None,
                        help="Save router training data to this file (.pt)")
    parser.add_argument("--rerollout-all", action="store_true", default=True,
                        help="Re-rollout ALL groups with forced tokens (not just bad ones). "
                             "Default True. Use --no-rerollout-all to only re-rollout bad groups.")
    parser.add_argument("--no-rerollout-all", dest="rerollout_all", action="store_false",
                        help="Only re-rollout bad groups (legacy behavior)")
    args = parser.parse_args()

    forced_token_list = [int(x.strip()) for x in args.forced_tokens.split(",")]
    print(f"Forced first tokens: {forced_token_list}")

    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  decoded: {[repr(tokenizer.decode([t])) for t in forced_token_list]}")

    # 2. Load data
    dfs = [pd.read_parquet(p) for p in args.data]
    df = pd.concat(dfs, ignore_index=True)
    # -1 means use all prompts (no sampling, but still shuffle for consistent speed)
    total = len(df)
    if args.num_prompts < 0:
        df = df.sample(n=total, random_state=42)  # shuffle all
        print(f"Using all {total} prompts (shuffled)")
    elif len(df) > args.num_prompts:
        df = df.sample(n=args.num_prompts, random_state=42)
    # Keep original index as a column so we can trace back
    df = df.reset_index()  # original index becomes column "index"
    original_indices = df["index"].tolist()
    print(f"Loaded {len(df)} prompts (sampled from {len(pd.concat(dfs))} total)")
    print(f"  Original data indices: {original_indices}")

    prompts_text = []
    ground_truths = []
    data_sources = []
    for _, row in df.iterrows():
        messages = parse_messages(row["prompt"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_text.append(text)
        rm = row["reward_model"]
        if isinstance(rm, str):
            import ast
            rm = ast.literal_eval(rm)
        ground_truths.append(rm.get("ground_truth", ""))
        data_sources.append(row.get("data_source", ""))

    # 3. Load vLLM
    print(f"Loading vLLM model {args.model} (tp={args.tp}) ...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    # 4. Phase 1: Normal rollout (rollout_n per prompt)
    print(f"\n{'='*70}")
    print(f"Phase 1: Normal rollout ({args.rollout_n} per prompt, {len(prompts_text)} prompts)")
    print(f"{'='*70}")

    # Repeat each prompt rollout_n times (interleaved)
    expanded_prompts = []
    expanded_gts = []
    expanded_ds = []
    for i in range(len(prompts_text)):
        for _ in range(args.rollout_n):
            expanded_prompts.append(prompts_text[i])
            expanded_gts.append(ground_truths[i])
            expanded_ds.append(data_sources[i])

    sp_normal = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
    )
    outputs_normal = llm.generate(expanded_prompts, sp_normal)

    # Check correctness
    normal_scores = []
    for i, out in enumerate(outputs_normal):
        resp = out.outputs[0].text
        score = compute_score(resp, expanded_ds[i], expanded_gts[i])
        normal_scores.append(score)

    # Group results
    num_groups = len(prompts_text)
    normal_group_correct = []  # per group: True if any rollout correct
    bad_group_indices = []     # indices of groups with no correct rollout
    for g in range(num_groups):
        start = g * args.rollout_n
        group_scores = normal_scores[start: start + args.rollout_n]
        has_correct = any(s > 0 for s in group_scores)
        normal_group_correct.append(has_correct)
        if not has_correct:
            bad_group_indices.append(g)

    normal_pass_rate = sum(normal_group_correct) / num_groups
    print(f"\nNormal rollout pass-rate: {normal_pass_rate:.4f} "
          f"({sum(normal_group_correct)}/{num_groups} groups have >=1 correct)")
    print(f"Bad groups (no correct answer): {len(bad_group_indices)}/{num_groups}")
    if bad_group_indices:
        print(f"  Bad group indices (sampled): {bad_group_indices}")
        print(f"  Bad group indices (original data): {[original_indices[g] for g in bad_group_indices]}")

    if not bad_group_indices and not args.rerollout_all:
        print("\nAll groups already have a correct answer. No need for forced tokens.")
        return

    # 5. Phase 2: Re-rollout with forced first tokens
    #    --rerollout-all (default): re-rollout ALL groups → more router training data
    #    --no-rerollout-all: only re-rollout bad groups (legacy)
    rerollout_group_indices = list(range(num_groups)) if args.rerollout_all else bad_group_indices
    rerollout_group_labels = ["bad" if g in set(bad_group_indices) else "good"
                              for g in rerollout_group_indices]

    print(f"\n{'='*70}")
    print(f"Phase 2: Re-rollout {len(rerollout_group_indices)} groups with forced first tokens")
    if args.rerollout_all:
        n_good = sum(1 for l in rerollout_group_labels if l == "good")
        n_bad = sum(1 for l in rerollout_group_labels if l == "bad")
        print(f"  ({n_bad} bad + {n_good} good = {n_bad + n_good} total)")
    print(f"  Tokens: {forced_token_list}")
    print(f"  Decoded: {[repr(tokenizer.decode([t])) for t in forced_token_list]}")
    print(f"{'='*70}")

    # Build prompts for re-rollout: each group gets rollout_n prompts,
    # each with a different forced first token appended to the prompt
    rerollout_prompts = []
    rerollout_gts = []
    rerollout_ds = []
    rerollout_token_assignments = []  # which forced token for each rerollout
    for g in rerollout_group_indices:
        for i in range(args.rollout_n):
            forced_tok = forced_token_list[i % len(forced_token_list)]
            # Append forced token to the prompt text (as token ids)
            prompt_ids = tokenizer.encode(prompts_text[g], add_special_tokens=False)
            prompt_ids = prompt_ids + [forced_tok]
            rerollout_prompts.append(prompt_ids)
            rerollout_gts.append(ground_truths[g])
            rerollout_ds.append(data_sources[g])
            rerollout_token_assignments.append(forced_tok)

    print(f"Re-rolling out {len(rerollout_prompts)} trajectories ...")

    sp_forced = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
    )
    # Use TokensPrompt to pass token IDs (vLLM 0.11+ API)
    from vllm import TokensPrompt
    forced_prompt_objs = [TokensPrompt(prompt_token_ids=ids) for ids in rerollout_prompts]
    outputs_forced = llm.generate(forced_prompt_objs, sampling_params=sp_forced)

    # Check correctness of re-rolled trajectories
    forced_scores = []
    for i, out in enumerate(outputs_forced):
        resp = out.outputs[0].text
        score = compute_score(resp, rerollout_ds[i], rerollout_gts[i])
        forced_scores.append(score)

    # For each bad group, check if any re-rolled trajectory is correct
    rescued = 0
    rescued_by_token = Counter()
    # Only count rescued for bad groups (good groups already had correct answers)
    bad_idx_in_rerollout = [i for i, g in enumerate(rerollout_group_indices)
                            if g in set(bad_group_indices)]
    for idx in bad_idx_in_rerollout:
        r_start = idx * args.rollout_n
        r_scores = forced_scores[r_start: r_start + args.rollout_n]
        has_correct = any(s > 0 for s in r_scores)
        if has_correct:
            rescued += 1
            # which token rescued it?
            for j in range(args.rollout_n):
                if r_scores[j] > 0:
                    rescued_by_token[rerollout_token_assignments[r_start + j]] += 1

    after_pass_rate = (sum(normal_group_correct) + rescued) / num_groups

    # 6. Phase 3 (control): Re-rollout bad groups WITHOUT forced first tokens
    print(f"\n{'='*70}")
    print(f"Phase 3 (control): Re-rollout {len(bad_group_indices)} bad groups WITHOUT forced tokens")
    print(f"{'='*70}")

    control_prompts = []
    control_gts = []
    control_ds = []
    for g in bad_group_indices:
        for _ in range(args.rollout_n):
            control_prompts.append(prompts_text[g])
            control_gts.append(ground_truths[g])
            control_ds.append(data_sources[g])

    print(f"Re-rolling out {len(control_prompts)} trajectories (plain, no forced token) ...")
    outputs_control = llm.generate(control_prompts, sp_normal)

    control_scores = []
    for i, out in enumerate(outputs_control):
        resp = out.outputs[0].text
        score = compute_score(resp, control_ds[i], control_gts[i])
        control_scores.append(score)

    control_rescued = 0
    for idx, g in enumerate(bad_group_indices):
        r_start = idx * args.rollout_n
        r_scores = control_scores[r_start: r_start + args.rollout_n]
        if any(s > 0 for s in r_scores):
            control_rescued += 1

    control_pass_rate = (sum(normal_group_correct) + control_rescued) / num_groups

    # 7. Report
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Normal pass-rate:              {normal_pass_rate:.4f}  "
          f"({sum(normal_group_correct)}/{num_groups} groups)")
    print(f"  After plain re-rollout (ctrl):  {control_pass_rate:.4f}  "
          f"({sum(normal_group_correct) + control_rescued}/{num_groups} groups)")
    print(f"  After forced-token re-rollout:  {after_pass_rate:.4f}  "
          f"({sum(normal_group_correct) + rescued}/{num_groups} groups)")
    print(f"  ")
    print(f"  Improvement from re-rollout (no forced token): +{control_pass_rate - normal_pass_rate:.4f}")
    print(f"  Improvement from forced token:                  +{after_pass_rate - normal_pass_rate:.4f}")
    print(f"  Forced token vs plain re-rollout:                +{after_pass_rate - control_pass_rate:.4f}")
    print(f"  ")
    print(f"  Bad groups:                  {len(bad_group_indices)}/{num_groups}")
    print(f"  Rescued by plain re-rollout:  {control_rescued}/{len(bad_group_indices)}")
    print(f"  Rescued by forced tokens:    {rescued}/{len(bad_group_indices)}")
    if rescued_by_token:
        print(f"\n  Rescued by token:")
        for tid, count in rescued_by_token.most_common():
            print(f"    {tid:>8} ({repr(tokenizer.decode([tid]))}): rescued {count} groups")
    print(f"{'='*70}")

    # 7. Show some examples of rescued groups
    print(f"\nSample rescued responses (first 3):")
    shown = 0
    # Map group index -> position in rerollout_group_indices
    rerollout_pos = {g: i for i, g in enumerate(rerollout_group_indices)}
    for g in bad_group_indices:
        if shown >= 3:
            break
        idx = rerollout_pos[g]
        r_start = idx * args.rollout_n
        r_scores = forced_scores[r_start: r_start + args.rollout_n]
        if not any(s > 0 for s in r_scores):
            continue
        # find the first correct re-rolled trajectory
        for j in range(args.rollout_n):
            if r_scores[j] > 0:
                forced_tok = rerollout_token_assignments[r_start + j]
                resp = outputs_forced[r_start + j].outputs[0].text
                print(f"\n  Group {g} (orig_idx={original_indices[g]}, gt={ground_truths[g]!r}):")
                print(f"    Forced token: {forced_tok} ({repr(tokenizer.decode([forced_tok]))})")
                print(f"    Response (first 300 chars): {resp[:300]!r}")
                shown += 1
                break

    # 8. Save router training data
    if args.save_data:
        import torch
        K = len(forced_token_list)
        if args.rollout_n != K:
            print(f"\nWARNING: rollout_n={args.rollout_n} != len(forced_tokens)={K}. "
                  f"Router data requires rollout_n == number of forced tokens. "
                  f"Skipping save.")
        else:
            router_data = {
                "prompts_text": [],       # list[str], chat-templated prompt strings
                "ground_truths": [],       # list[str], ground truth answers
                "data_sources": [],        # list[str], data source identifiers
                "forced_token_ids": forced_token_list,  # list[int], K candidate token ids
                "rewards": [],             # list[list[float]], will become [N, K] tensor
                "group_was_bad": [],       # list[bool], True if normal rollout had no correct answer
            }
            # Save ALL re-rolled groups (both good and bad)
            for idx, g in enumerate(rerollout_group_indices):
                r_start = idx * args.rollout_n
                r_scores = forced_scores[r_start: r_start + args.rollout_n]
                # reward_vec[k] = 1.0 if forced_token_list[k] produces a correct answer
                reward_vec = [float(r_scores[k]) for k in range(K)]
                router_data["prompts_text"].append(prompts_text[g])
                router_data["ground_truths"].append(ground_truths[g])
                router_data["data_sources"].append(data_sources[g])
                router_data["rewards"].append(reward_vec)
                router_data["group_was_bad"].append(g in set(bad_group_indices))

            router_data["rewards"] = torch.tensor(router_data["rewards"], dtype=torch.float32)
            router_data["group_was_bad"] = torch.tensor(router_data["group_was_bad"],
                                                        dtype=torch.bool)
            torch.save(router_data, args.save_data)

            n_total = len(router_data["prompts_text"])
            n_bad = router_data["group_was_bad"].sum().item()
            n_good = n_total - n_bad
            n_has_correct = (router_data["rewards"].sum(dim=1) > 0).sum().item()
            n_bad_has_correct = ((router_data["rewards"].sum(dim=1) > 0) &
                                 router_data["group_was_bad"]).sum().item()
            n_good_has_correct = ((router_data["rewards"].sum(dim=1) > 0) &
                                  ~router_data["group_was_bad"]).sum().item()

            print(f"\nRouter training data saved to {args.save_data}")
            print(f"  Total groups: {n_total} ({n_bad} bad + {n_good} good)")
            print(f"  Reward matrix shape: {router_data['rewards'].shape}  (N, K={K})")
            print(f"  Prompts with >=1 correct token: {n_has_correct}/{n_total}")
            print(f"    Bad groups with >=1 correct token: {n_bad_has_correct}/{n_bad}")
            print(f"    Good groups with >=1 correct token: {n_good_has_correct}/{n_good}")
            print(f"  Correct per token: {router_data['rewards'].sum(dim=0).tolist()}")
            if n_has_correct < 30:
                print(f"  WARNING: only {n_has_correct} usable training samples. "
                      f"Consider increasing --num-prompts.")


if __name__ == "__main__":
    main()
