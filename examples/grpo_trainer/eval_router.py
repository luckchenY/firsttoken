#!/usr/bin/env python3
"""Evaluate a trained router on test data.

Two modes:
  1. Fast mode (default): load test data saved by collect_router_data.py (has reward matrix),
     router predicts token, check against reward matrix. No generation needed.
  2. Generation mode (--generate): load raw prompts, router selects token, generate with
     forced token, check correctness. Slower but tests on unseen prompts.

Usage (fast mode, recommended):
  python eval_router.py \
      --model /data/chenyang2/Qwen3-8B \
      --router /data/chenyang2/router_weights.pt \
      --data /data/chenyang2/router_data_test.pt

Usage (generation mode):
  python eval_router.py \
      --model /data/chenyang2/Qwen3-8B \
      --router /data/chenyang2/router_weights.pt \
      --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
      --num-prompts 200 --generate \
      --tp 4 --gpu-mem-util 0.9 --max-tokens 4096
"""

import argparse
import os
import sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mmlu_pro_reward import compute_score_mmlu_pro


def load_router(router_path, device):
    """Load router MLP weights."""
    data = torch.load(router_path, weights_only=False, map_location="cpu")
    config = data["config"]
    forced_token_ids = data["forced_token_ids"]

    from train_router import RouterMLP
    router = RouterMLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_candidates=config["num_candidates"],
    )
    router.load_state_dict(data["router_state_dict"])
    router.to(device)
    router.eval()

    print(f"Router loaded from {router_path}")
    print(f"  input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, "
          f"num_candidates={config['num_candidates']}")
    print(f"  forced_token_ids: {forced_token_ids}")
    return router, forced_token_ids, config


def extract_hidden_states(model, tokenizer, prompts_text, device, batch_size=8):
    """Forward pass on prompts, return last-token hidden state for each."""
    all_hidden = []
    for i in range(0, len(prompts_text), batch_size):
        batch = prompts_text[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len, D]
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=device)
        gathered = last_hidden[batch_indices, seq_lengths]  # [B, D]
        all_hidden.append(gathered.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  Hidden states: {min(i + batch_size, len(prompts_text))}/{len(prompts_text)}")
    return torch.cat(all_hidden, dim=0)  # [N, D]


def eval_fast_mode(args, device):
    """Fast mode: use pre-collected reward matrix, no generation."""
    from train_router import RouterMLP

    print(f"\n=== Fast evaluation mode ===")
    print(f"Loading test data from {args.data} ...")
    data = torch.load(args.data if isinstance(args.data, str) else args.data[0],
                      weights_only=False)
    prompts_text = data["prompts_text"]
    rewards = data["rewards"]  # [N, K]
    forced_token_ids = data["forced_token_ids"]
    N, K = rewards.shape

    # Load router
    router, router_token_ids, config = load_router(args.router, device)
    assert router_token_ids == forced_token_ids, \
        f"Token mismatch! Router: {router_token_ids}, Data: {forced_token_ids}"

    # Extract hidden states
    print(f"\nLoading model {args.model} for hidden state extraction ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    print(f"Extracting hidden states for {N} prompts ...")
    hidden = extract_hidden_states(model, tokenizer, prompts_text, device,
                                    batch_size=args.batch_size)
    hidden = hidden.float()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Router predictions
    with torch.no_grad():
        logits = router(hidden.to(device))  # [N, K]
        selected = logits.argmax(dim=1).cpu()  # [N]

    # Evaluate
    # 1. Router accuracy: does router's selected token give correct answer?
    router_correct = rewards[torch.arange(N), selected] > 0
    router_acc = router_correct.float().mean().item()

    # 2. Random baseline: average accuracy if we pick a random token
    random_acc = rewards.mean().item()  # average reward across all tokens

    # 3. Oracle: if we always pick the best token (upper bound)
    oracle_correct = (rewards.sum(dim=1) > 0)  # any token works
    oracle_acc = oracle_correct.float().mean().item()

    # 4. Most frequent token baseline: always pick the token with highest overall accuracy
    token_acc = rewards.mean(dim=0)  # [K]
    best_token = token_acc.argmax().item()
    majority_acc = token_acc[best_token].item()

    # 5. Per-token accuracy
    print(f"\n{'='*70}")
    print(f"RESULTS (N={N} prompts, K={K} tokens)")
    print(f"{'='*70}")
    print(f"  Router accuracy:      {router_acc:.4f}  ({router_correct.sum()}/{N})")
    print(f"  Random baseline:      {random_acc:.4f}  (avg reward per token)")
    print(f"  Majority baseline:     {majority_acc:.4f}  (always pick token {best_token}: "
          f"{repr(tokenizer.decode([forced_token_ids[best_token]]))})")
    print(f"  Oracle (best token):  {oracle_acc:.4f}  (if always picked the correct token)")
    print(f"  Router vs random:      +{router_acc - random_acc:.4f}")
    print(f"  Router vs majority:    +{router_acc - majority_acc:.4f}")
    print(f"  Router vs oracle:      {router_acc - oracle_acc:.4f}  (gap to upper bound)")
    print(f"\n  Per-token accuracy:")
    for k in range(K):
        tok = forced_token_ids[k]
        acc = rewards[:, k].mean().item()
        n_sel = (selected == k).sum().item()
        marker = " ← router picks this" if k == selected.mode().values.item() else ""
        print(f"    token {tok:>8} ({repr(tokenizer.decode([tok]))}): "
              f"acc={acc:.4f}, router_picks={n_sel}")
    print(f"{'='*70}")


def eval_generate_mode(args, device):
    """Generation mode: two-phase to avoid GPU memory conflict.
    
    Phase 1 (--step select): HF model extracts hidden states, router selects tokens, save to file.
    Phase 2 (--step generate): vLLM generates with forced tokens, score, report.
    """
    step = args.step
    if step == "select":
        _phase1_select(args, device)
    elif step == "generate":
        _phase2_generate(args, device)
    else:
        # Run both phases in sequence (may fail due to GPU memory)
        _phase1_select(args, device)
        print("\n" + "="*70)
        print("Phase 1 done. Now starting Phase 2 (vLLM generation)...")
        print("="*70 + "\n")
        _phase2_generate(args, device)


def _phase1_select(args, device):
    """Phase 1: Extract hidden states, run router, save selected tokens."""
    from train_router import RouterMLP

    print(f"\n=== Phase 1: Router token selection ===")

    # Load router
    router, forced_token_ids, config = load_router(args.router, device)
    K = len(forced_token_ids)

    # Load data
    dfs = [pd.read_parquet(p) for p in args.data]
    df = pd.concat(dfs, ignore_index=True)
    if args.num_prompts > 0 and len(df) > args.num_prompts:
        df = df.sample(n=args.num_prompts, random_state=42)
    print(f"Test prompts: {len(df)}")

    def parse_messages(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            import ast
            return ast.literal_eval(val)
        return list(val)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts_text = []
    ground_truths = []
    data_sources = []
    for _, row in df.iterrows():
        messages = parse_messages(row["prompt"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompts_text.append(text)
        rm = row.get("reward_model", {})
        ground_truths.append(rm.get("ground_truth", "") if isinstance(rm, dict) else "")
        data_sources.append(row.get("data_source", ""))

    # Extract hidden states with HF model
    print(f"\nLoading HF model for hidden state extraction ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    print(f"Extracting hidden states for {len(prompts_text)} prompts ...")
    hidden = extract_hidden_states(hf_model, tokenizer, prompts_text, device,
                                    batch_size=args.batch_size)
    hidden = hidden.float()
    del hf_model
    import gc; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Router selects token for each prompt
    with torch.no_grad():
        logits = router(hidden.to(device))  # [N, K]
        selected = logits.argmax(dim=1).cpu()  # [N]
    selected_tokens = [forced_token_ids[selected[i].item()] for i in range(len(prompts_text))]
    print(f"\nRouter token selection distribution:")
    from collections import Counter
    tok_counts = Counter(selected_tokens)
    for tok, count in tok_counts.most_common():
        print(f"  {tok:>8} ({repr(tokenizer.decode([tok]))}): {count}")

    # Also randomly select tokens for comparison
    import random
    random.seed(42)
    random_tokens = [random.choice(forced_token_ids) for _ in range(len(prompts_text))]
    print(f"\nRandom token selection distribution:")
    rand_counts = Counter(random_tokens)
    for tok, count in rand_counts.most_common():
        print(f"  {tok:>8} ({repr(tokenizer.decode([tok]))}): {count}")

    # Save intermediate results
    tmp_file = args.tmp_file
    torch.save({
        "prompts_text": prompts_text,
        "ground_truths": ground_truths,
        "data_sources": data_sources,
        "selected_tokens": selected_tokens,
        "random_tokens": random_tokens,
        "forced_token_ids": forced_token_ids,
    }, tmp_file)
    print(f"\nPhase 1 done. Saved to {tmp_file}")
    print(f"Now run Phase 2:")
    print(f"  python {sys.argv[0]} --model {args.model} --router {args.router} "
          f"--data {args.data[0]} --generate --step generate "
          f"--tmp-file {tmp_file} --tp {args.tp} --gpu-mem-util {args.gpu_mem_util} "
          f"--max-tokens {args.max_tokens}")


def _phase2_generate(args, device):
    """Phase 2: Load vLLM, generate with forced tokens, score, report."""
    from verl.utils.reward_score import default_compute_score

    def compute_score(response, data_source, ground_truth):
        # MMLU-Pro uses custom reward
        if "MMLU-Pro" in data_source or "mmlu" in data_source.lower():
            return compute_score_mmlu_pro(response, ground_truth)
        try:
            score = default_compute_score(
                data_source=data_source, solution_str=response,
                ground_truth=ground_truth)
            return float(score) if score is not None else 0.0
        except Exception:
            return 0.0

    print(f"\n=== Phase 2: vLLM generation + scoring ===")

    # Load intermediate results
    data = torch.load(args.tmp_file, weights_only=False)
    prompts_text = data["prompts_text"]
    ground_truths = data["ground_truths"]
    data_sources = data["data_sources"]
    selected_tokens = data["selected_tokens"]
    random_tokens = data.get("random_tokens", None)
    forced_token_ids = data["forced_token_ids"]
    N = len(prompts_text)
    print(f"Loaded {N} prompts from {args.tmp_file}")
    if random_tokens:
        print(f"  Router tokens + Random tokens both available")
    else:
        print(f"  Only router tokens (no random baseline)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Generate with vLLM
    print(f"\nLoading vLLM for generation ...")
    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(
        model=args.model, tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len, trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    # --- Group 1: Router-selected forced tokens ---
    prompt_objs_router = []
    for i, text in enumerate(prompts_text):
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_ids = prompt_ids + [selected_tokens[i]]
        prompt_objs_router.append(TokensPrompt(prompt_token_ids=prompt_ids))

    print(f"Generating {len(prompt_objs_router)} trajectories with router-selected tokens ...")
    outputs_router = llm.generate(prompt_objs_router, sampling_params=sp)

    # --- Group 2: Random forced tokens ---
    outputs_random = None
    if random_tokens:
        prompt_objs_random = []
        for i, text in enumerate(prompts_text):
            prompt_ids = tokenizer.encode(text, add_special_tokens=False)
            prompt_ids = prompt_ids + [random_tokens[i]]
            prompt_objs_random.append(TokensPrompt(prompt_token_ids=prompt_ids))

        print(f"Generating {len(prompt_objs_random)} trajectories with random tokens ...")
        outputs_random = llm.generate(prompt_objs_random, sampling_params=sp)

    # --- Group 3: Normal (no forced token) ---
    print(f"Generating {len(prompts_text)} trajectories without forced token (baseline) ...")
    sp_normal = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    outputs_normal = llm.generate(prompts_text, sampling_params=sp_normal)

    # Score
    router_correct = 0
    random_correct = 0
    normal_correct = 0
    for i in range(N):
        # Router-forced
        resp = outputs_router[i].outputs[0].text
        if compute_score(resp, data_sources[i], ground_truths[i]) > 0:
            router_correct += 1
        # Random-forced
        if outputs_random is not None:
            resp_r = outputs_random[i].outputs[0].text
            if compute_score(resp_r, data_sources[i], ground_truths[i]) > 0:
                random_correct += 1
        # Normal
        resp_n = outputs_normal[i].outputs[0].text
        if compute_score(resp_n, data_sources[i], ground_truths[i]) > 0:
            normal_correct += 1

    print(f"\n{'='*70}")
    print(f"RESULTS (N={N} prompts)")
    print(f"{'='*70}")
    print(f"  Normal (no forced token):      {normal_correct}/{N} = {normal_correct/N:.4f}")
    if outputs_random is not None:
        print(f"  Random forced token:           {random_correct}/{N} = {random_correct/N:.4f}")
    print(f"  Router-selected forced token:  {router_correct}/{N} = {router_correct/N:.4f}")
    print(f"")
    print(f"  Router vs Normal:   +{(router_correct - normal_correct)/N:.4f}")
    if outputs_random is not None:
        print(f"  Router vs Random:   +{(router_correct - random_correct)/N:.4f}")
        print(f"  Random vs Normal:   +{(random_correct - normal_correct)/N:.4f}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained router")
    parser.add_argument("--model", default="/data/chenyang2/Qwen3-8B")
    parser.add_argument("--router", required=True, help="Path to router_weights.pt")
    parser.add_argument("--data", nargs="+", required=True,
                        help="Fast mode: path to .pt file. Generate mode: parquet files.")
    parser.add_argument("--generate", action="store_true",
                        help="Generation mode (slower, tests on raw prompts)")
    parser.add_argument("--step", type=str, default="both",
                        choices=["both", "select", "generate"],
                        help="generate mode: 'select' = phase 1 only, 'generate' = phase 2 only, "
                             "'both' = run both (may fail due to GPU memory)")
    parser.add_argument("--tmp-file", type=str, default="/tmp/router_eval_tmp.pt",
                        help="Temp file for intermediate results between phases")
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Number of test prompts (generate mode only)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for hidden state extraction")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if args.generate:
        eval_generate_mode(args, device)
    else:
        eval_fast_mode(args, device)


if __name__ == "__main__":
    main()
