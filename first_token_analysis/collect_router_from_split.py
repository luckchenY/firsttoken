#!/usr/bin/env python3
"""Re-collect router rewards for an existing router-question .pt file.

The question prompts, ground truths, and source labels are taken verbatim from
router_data_5000/split/all.pt.  Generation and reward scoring are delegated to
collect_router_data.py so this is only an adapter for the already-collected
question set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
COLLECTOR_DIR = Path("/data/chenyang2/verl/examples/grpo_trainer")
sys.path.insert(0, str(COLLECTOR_DIR))
from collect_router_data import score_trajectories_parallel  # noqa: E402


FORCED_TOKEN_IDS = [
    1654, 32313, 785, 46254, 1986, 50, 1205, 23657, 71486, 2679,
    896, 16910, 11578, 88854, 47866, 641, 3925, 9190, 22043, 1249,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-pt", required=True)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--score-workers", type=int, default=16)
    args = parser.parse_args()

    if args.start < 0 or args.end <= args.start:
        raise ValueError(f"invalid range [{args.start}, {args.end})")

    data = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    all_prompts = data["prompts_text"]
    all_ground_truths = data["ground_truths"]
    all_sources = data["data_sources"]
    n_total = len(all_prompts)
    if args.end > n_total:
        raise ValueError(f"range end {args.end} exceeds input size {n_total}")

    prompts = all_prompts[args.start:args.end]
    ground_truths = all_ground_truths[args.start:args.end]
    data_sources = all_sources[args.start:args.end]
    n = len(prompts)
    k = len(FORCED_TOKEN_IDS)

    print(f"Input: {args.input_pt}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Range: [{args.start}, {args.end}) -> {n} prompts", flush=True)
    print(f"Forced token ids: {FORCED_TOKEN_IDS}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(
        "Decoded forced tokens:",
        [repr(tokenizer.decode([token_id])) for token_id in FORCED_TOKEN_IDS],
        flush=True,
    )

    prompt_objects = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for token_id in FORCED_TOKEN_IDS:
            prompt_objects.append(TokensPrompt(prompt_token_ids=prompt_ids + [token_id]))

    print(f"Loading vLLM and generating {n * k} trajectories ...", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompt_objects, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    del llm

    extra_infos = [{} for _ in responses]
    print(f"Scoring {len(responses)} trajectories ...", flush=True)
    scores = score_trajectories_parallel(
        responses,
        [source for source in data_sources for _ in range(k)],
        [truth for truth in ground_truths for _ in range(k)],
        extra_infos,
        n_workers=args.score_workers,
        label=f"range-{args.start}-{args.end}",
    )
    rewards = torch.tensor(scores, dtype=torch.float32).reshape(n, k)

    output = {
        "prompts_text": prompts,
        "ground_truths": ground_truths,
        "data_sources": data_sources,
        "forced_token_ids": FORCED_TOKEN_IDS,
        "rewards": rewards,
        "source_input_pt": args.input_pt,
        "source_range": [args.start, args.end],
        "model": args.model,
    }
    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"Saved: {output_path}", flush=True)
    print(f"Reward shape: {tuple(rewards.shape)}", flush=True)
    print(
        f"Prompts with >=1 correct token: {(rewards.sum(dim=1) > 0).sum().item()}/{n}",
        flush=True,
    )
    print("Correct per token:", rewards.sum(dim=0).tolist(), flush=True)


if __name__ == "__main__":
    main()
