#!/usr/bin/env python3
"""Plot the "first-token collapse" effect of an LLM across benchmarks.

Produces a 2x2 figure that visualizes:
  (a) Only a few tokens dominate the first-response-token distribution.
  (b) Those few tokens capture almost all of the probability mass.
  (c) The per-prompt distributions look nearly identical regardless of the
      question -- the collapse is not driven by a particular prompt.
  (d) The same pattern shows up across different benchmarks -- it is not an
      artifact of any single dataset.

This script reuses the logprob-aggregation logic of
`analyze_first_tokens.py` but runs the generation *separately per benchmark*
so we can compare datasets in panel (d).

Usage:
    python3 examples/grpo_trainer/plot_first_token_collapse.py \
        --model /workspace/Qwen3-8B \
        --data gsm8k:~/data/gsm8k/train.parquet \
        --data math:~/data/math/train.parquet \
        --num-prompts 500 --top-k 8 --skip-tokens 2 \
        --out first_token_collapse.png

The `--data` argument can be repeated; each occurrence is `label:path`.
The label is used in the legend / panel titles.
"""

import argparse
import ast
from collections import defaultdict
from math import exp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from transformers import AutoTokenizer

# ---- nicer defaults ---------------------------------------------------------
rcParams["figure.dpi"] = 130
rcParams["savefig.dpi"] = 200
rcParams["font.size"] = 10
rcParams["axes.titlesize"] = 12
rcParams["axes.titleweight"] = "bold"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False


def parse_messages(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return ast.literal_eval(val)
    return list(val)


def build_prompts(df, tokenizer, num_prompts):
    if len(df) > num_prompts:
        df = df.sample(n=num_prompts, random_state=42).reset_index(drop=True)
    prompts = []
    for _, row in df.iterrows():
        msgs = parse_messages(row["prompt"])
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def run_benchmark(llm, sampling_params, tokenizer, parquet_path, num_prompts, skip_tokens):
    """Generate on one benchmark and return per-prompt logprob info.

    Returns
    -------
    per_prompt : list[dict[token_id -> prob]]
        One entry per valid prompt: the probability distribution over the
        top-K logprobs returned by vLLM at the target position.
    outputs : list
        Raw vLLM outputs (kept for debugging / decoding tokens).
    """
    from vllm import SamplingParams  # local import to avoid hard dep at import time

    df = pd.read_parquet(parquet_path)
    prompts = build_prompts(df, tokenizer, num_prompts)
    print(f"  [{parquet_path}] {len(prompts)} prompts, generating ...")
    outputs = llm.generate(prompts, sampling_params)

    per_prompt = []
    for out in outputs:
        toks = out.outputs[0].token_ids
        lps = out.outputs[0].logprobs
        if not lps or len(toks) == 0 or skip_tokens >= len(lps):
            continue
        pos = lps[skip_tokens]
        dist = {tid: exp(o.logprob) for tid, o in pos.items()}
        per_prompt.append(dist)
    print(f"    collected distributions for {len(per_prompt)} prompts")
    return per_prompt, outputs


def aggregate(per_prompt):
    """Aggregate per-prompt distributions into a single ranking.

    Returns (token_ids, avg_probs, total_mass) sorted by avg prob desc.
    """
    accum = defaultdict(float)
    n = len(per_prompt)
    for d in per_prompt:
        for tid, p in d.items():
            accum[tid] += p
    ranked = sorted(accum.items(), key=lambda x: -x[1])
    tids = [t for t, _ in ranked]
    avg = [v / n for _, v in ranked]
    return tids, avg, sum(avg)


# ---- plotting --------------------------------------------------------------


def plot_collapse(results, top_k, out_path):
    """results: dict label -> dict with keys:
       per_prompt, tids, avg, total_mass, tokenizer
    """
    labels = list(results.keys())
    primary = labels[0]
    tok = results[primary]["tokenizer"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ----- (a) Only a few words --------------------------------------------
    # Bar chart of top-K tokens by avg probability on the primary benchmark.
    tids = results[primary]["tids"][:top_k]
    probs = results[primary]["avg"][:top_k]
    decoded = [tok.decode([t]) for t in tids]
    # Clean labels: strip whitespace, keep repr-ish for control chars.
    labels_a = [repr(d) if (not d.strip() or "\\" in d) else d for d in decoded]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, top_k))
    bars = ax_a.bar(range(top_k), probs, color=colors, edgecolor="black", linewidth=0.4)
    ax_a.set_xticks(range(top_k))
    ax_a.set_xticklabels(labels_a, rotation=30, ha="right")
    ax_a.set_ylabel("Avg. probability")
    ax_a.set_title(
        f"(a) 只有几个词 — Top-{top_k} 首token占据分布"
    )
    ax_a.set_ylim(0, max(probs) * 1.15)
    for bar, p in zip(bars, probs):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(probs) * 0.02,
            f"{p*100:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )

    # ----- (b) They take almost all the mass ------------------------------
    # Cumulative probability mass curve over the token ranking.
    for label in labels:
        avg = np.array(results[label]["avg"])
        cum = np.cumsum(avg)
        x = np.arange(1, len(cum) + 1)
        ax_b.plot(x, cum, label=label, linewidth=2)
        # mark top-k
        k = min(top_k, len(cum))
        ax_b.scatter([k], [cum[k - 1]], zorder=5)
        ax_b.annotate(
            f"top-{top_k}: {cum[k-1]*100:.1f}%",
            xy=(k, cum[k - 1]),
            xytext=(k + max(1, len(cum) * 0.05), cum[k - 1] - 0.08),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.6),
        )
    ax_b.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_b.set_xlabel("Token rank")
    ax_b.set_ylabel("Cumulative probability mass")
    ax_b.set_title("(b) 而且它们占了几乎所有概率质量")
    ax_b.set_xscale("log")
    ax_b.set_xlim(1, max(len(results[l]["avg"]) for l in labels))
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(loc="lower right", frameon=False)

    # ----- (c) Same shape regardless of the question ---------------------
    # Overlay per-prompt distributions for a sample of prompts on the
    # primary benchmark. We align them by *global* token rank so curves are
    # comparable. Each curve = one prompt's CDF over the global ranking.
    per_prompt = results[primary]["per_prompt"]
    global_tids = results[primary]["tids"]
    tid_to_rank = {t: i for i, t in enumerate(global_tids)}
    n_show = min(15, len(per_prompt))
    sample_idx = np.linspace(0, len(per_prompt) - 1, n_show).astype(int)

    cmap = plt.cm.coolwarm(np.linspace(0, 1, n_show))
    for j, idx in enumerate(sample_idx):
        d = per_prompt[idx]
        # build a probability vector aligned to the global ranking
        vec = np.zeros(len(global_tids))
        for tid, p in d.items():
            r = tid_to_rank.get(tid)
            if r is not None:
                vec[r] = p
        # pad to the same length even if some tokens missing
        cum = np.cumsum(vec)
        ax_c.plot(
            np.arange(1, len(cum) + 1),
            cum,
            color=cmap[j],
            alpha=0.7,
            linewidth=1.0,
        )
    ax_c.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Token rank (global ranking)")
    ax_c.set_ylabel("Cumulative prob. (per prompt)")
    ax_c.set_title(f"(c) 不管题是什么，分布都长得差不多\n({n_show} 条 prompt 叠加, {primary})")
    ax_c.set_xlim(1, len(global_tids))
    ax_c.set_ylim(0, 1.05)

    # ----- (d) Not specific to one benchmark ------------------------------
    # Grouped bar chart comparing top-K avg-prob across benchmarks.
    # Use the union of top-K tokens across all benchmarks.
    union_tids = []
    for label in labels:
        for t in results[label]["tids"][:top_k]:
            if t not in union_tids:
                union_tids.append(t)
    union_decoded = [tok.decode([t]) for t in union_tids]
    union_labels = [repr(d) if (not d.strip() or "\\" in d) else d for d in union_decoded]

    x = np.arange(len(union_tids))
    width = 0.8 / len(labels)
    palette = plt.cm.Set2(np.linspace(0, 1, max(3, len(labels))))
    for i, label in enumerate(labels):
        avg_map = dict(zip(results[label]["tids"], results[label]["avg"]))
        vals = [avg_map.get(t, 0.0) for t in union_tids]
        ax_d.bar(
            x + (i - (len(labels) - 1) / 2) * width,
            vals,
            width,
            label=label,
            color=palette[i % len(palette)],
            edgecolor="black",
            linewidth=0.3,
        )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(union_labels, rotation=30, ha="right")
    ax_d.set_ylabel("Avg. probability")
    ax_d.set_title(f"(d) 这不是某一个 benchmark 的偶然现象\n(top-{top_k} 在各 benchmark 上的对比)")
    ax_d.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "首 token 坍塌效应 (first-token collapse)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/Qwen3-8B")
    p.add_argument(
        "--data", nargs="+", required=True,
        help="One or more 'label:path' entries, e.g. gsm8k:~/data/gsm8k/train.parquet",
    )
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--skip-tokens", type=int, default=2,
                   help="Analyze logprobs at this response position "
                        "(0=1st, 1=2nd, 2=3rd, ...)")
    p.add_argument("--out", default="first_token_collapse.png")
    args = p.parse_args()

    # parse data specs
    specs = []
    for spec in args.data:
        if ":" not in spec or not all(p.strip() for p in spec.split(":", 1)):
            raise SystemExit(f"--data must be 'label:path', got: {spec}")
        label, path = spec.split(":", 1)
        specs.append((label, path))

    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

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
        max_tokens=args.skip_tokens + 1,
        logprobs=20,
    )

    results = {}
    for label, path in specs:
        print(f"\n=== Benchmark: {label} ({path}) ===")
        per_prompt, outputs = run_benchmark(
            llm, sampling_params, tokenizer, path, args.num_prompts, args.skip_tokens
        )
        tids, avg, total = aggregate(per_prompt)
        print(f"  top-{args.top_k} mass: {sum(avg[:args.top_k]) / total * 100:.1f}% "
              f"of {total:.2f} captured mass")
        results[label] = dict(
            per_prompt=per_prompt,
            tids=tids,
            avg=avg,
            total_mass=total,
            tokenizer=tokenizer,
        )

    plot_collapse(results, args.top_k, args.out)


if __name__ == "__main__":
    main()
