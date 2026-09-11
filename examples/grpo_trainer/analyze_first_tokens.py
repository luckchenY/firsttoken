#!/usr/bin/env python3
"""Analyze first-token distribution of Qwen3-8B on GSM8K + MATH prompts.

Generates full responses with vLLM, prints sample responses for verification,
then extracts the first response token and counts frequencies.

Usage:
    python3 examples/grpo_trainer/analyze_first_tokens.py \
        --model Qwen/Qwen3-8B \
        --data ~/data/gsm8k/train.parquet ~/data/math/train.parquet \
        --num-prompts 500 --top-k 8
"""

import argparse
from collections import Counter
from math import exp

import pandas as pd
from transformers import AutoTokenizer


def parse_messages(val):
    """Parse the prompt field from parquet into a list of message dicts."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        import ast
        return ast.literal_eval(val)
    # numpy array or other
    return list(val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--skip-tokens", type=int, default=2,
                        help="Skip this many leading tokens, analyze logprobs at this position "
                             "(0=imd, 1=second token, 2=third token, ...)")
    args = parser.parse_args()

    # 1. Load tokenizer and prompts
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  vocab size: {len(tokenizer)}")
    print(f"  eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # Load data
    dfs = [pd.read_parquet(p) for p in args.data]
    df = pd.concat(dfs, ignore_index=True)
    if len(df) > args.num_prompts:
        df = df.sample(n=args.num_prompts, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} prompts")

    # Build prompt text via chat template
    prompts_text = []
    for _, row in df.iterrows():
        messages = parse_messages(row["prompt"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts_text.append(text)

    # DEBUG: print first 2 prompts to verify format
    print(f"\n{'='*70}")
    print("Sample prompt (first one, last 300 chars):")
    print(repr(prompts_text[0][-300:]))
    print(f"\nSample prompt (second one, last 300 chars):")
    print(repr(prompts_text[1][-300:]))
    print(f"{'='*70}\n")

    # 2. Generate with vLLM
    print(f"Loading vLLM model {args.model} (tp={args.tp}) ...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.skip_tokens + 1,  # generate enough tokens to reach the target position
        logprobs=20,         # collect top-20 logprobs at each position for weighted aggregation
    )

    print(f"Generating for {len(prompts_text)} prompts "
          f"(max_tokens={args.skip_tokens + 1}, logprobs=20, "
          f"analyzing position {args.skip_tokens}) ...")
    outputs = llm.generate(prompts_text, sampling_params)

    # 3. Print sample responses for verification
    print(f"\n{'='*70}")
    print("Sample responses (first 5):")
    for i in range(min(5, len(outputs))):
        resp_tokens = outputs[i].outputs[0].token_ids
        resp_logprobs = outputs[i].outputs[0].logprobs
        print(f"\n--- Response {i} ---")
        print(f"Generated token ids: {resp_tokens}")
        print(f"Generated tokens:    {[repr(tokenizer.decode([t])) for t in resp_tokens]}")
        target_pos = args.skip_tokens
        if resp_logprobs and len(resp_logprobs) > target_pos:
            print(f"Logprobs at position {target_pos}, top 10:")
            sorted_lp = sorted(resp_logprobs[target_pos].items(), key=lambda x: -x[1].logprob)
            for tid, lp in sorted_lp[:10]:
                print(f"    {tid:>8}  logprob={lp.logprob:>8.3f}  prob={exp(lp.logprob):.4f}  {repr(tokenizer.decode([tid]))}")
    print(f"{'='*70}\n")

    # 4. Weighted aggregation of logprobs at the target position
    #
    #    Skip the first `skip_tokens` tokens (e.g. imd + second token),
    #    then look at the probability distribution at position `skip_tokens`.
    #    Aggregate per-token probabilities across all prompts.

    target_pos = args.skip_tokens
    prob_accum = {}   # token_id -> summed probability across prompts
    total_prompts = 0

    for out in outputs:
        toks = out.outputs[0].token_ids
        lps = out.outputs[0].logprobs
        if not lps or len(toks) == 0:
            continue
        if target_pos >= len(lps):
            continue  # response too short

        total_prompts += 1
        pos_logprobs = lps[target_pos]  # dict: token_id -> Logprob object

        for tid, lp_obj in pos_logprobs.items():
            p = exp(lp_obj.logprob)
            prob_accum[tid] = prob_accum.get(tid, 0.0) + p

    print(f"Aggregated logprobs from {total_prompts} prompts "
          f"(at position {target_pos}, skipping first {target_pos} tokens)")

    # 5. Rank tokens by total probability mass
    ranked = sorted(prob_accum.items(), key=lambda x: -x[1])

    print(f"\n{'='*70}")
    print(f"Top {args.top_k} tokens by weighted probability "
          f"(sum of per-prompt probs across {total_prompts} prompts):")
    print(f"{'Rank':<6} {'TokenID':<10} {'TotalProb':<12} {'AvgProb%':<10} {'Decoded'}")
    print(f"{'-'*70}")

    top_tokens = []
    for rank, (tid, total_prob) in enumerate(ranked[:args.top_k], 1):
        avg_prob = total_prob / total_prompts * 100
        token_str = tokenizer.decode([tid])
        print(f"{rank:<6} {tid:<10} {total_prob:<12.4f} {avg_prob:<10.2f} {repr(token_str)}")
        top_tokens.append(tid)

    # 6. Also show top-20 for reference
    print(f"\nTop 20 (for reference):")
    for rank, (tid, total_prob) in enumerate(ranked[:20], 1):
        avg_prob = total_prob / total_prompts * 100
        print(f"  {rank:>3}. {tid:>8}  avg_prob={avg_prob:>6.2f}%  {repr(tokenizer.decode([tid]))}")

    total_mass_top_k = sum(p for _, p in ranked[:args.top_k])
    total_mass_all = sum(prob_accum.values())
    print(f"\nTop-{args.top_k} probability mass: {total_mass_top_k:.2f} / {total_mass_all:.2f} "
          f"({total_mass_top_k / total_mass_all * 100:.1f}% of captured mass)")

    # 7. Output for FORCED_FIRST_TOKEN_LIST
    token_list_str = ",".join(str(t) for t in top_tokens)
    print(f"\n{'='*70}")
    print(f"FORCED_FIRST_TOKEN_LIST={token_list_str}")
    print(f"\nAdd to training script:")
    print(f'  export FORCED_FIRST_TOKEN_LIST="{token_list_str}"')


if __name__ == "__main__":
    main()
