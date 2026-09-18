#!/usr/bin/env python3
"""Plot a model x benchmark heatmap of the "first-token collapse" effect.

For every (model, benchmark) pair we generate a short continuation with vLLM,
read the top-20 logprobs at a target response position, and summarize the
distribution with several collapse metrics:

  * top_k_mass : share of probability mass on the top-K tokens (default K=8)
  * top1       : average probability of the single most likely token
  * entropy    : normalized entropy of the aggregated distribution
                 (0 = fully collapsed, 1 = uniform)

The result is rendered as a model (rows) x benchmark (columns) heatmap,
annotated with the numeric value. A pickle cache is written so you can
re-style / re-render the figure without re-running vLLM.

Memory strategy: models are loaded one at a time; after each model finishes
all its benchmarks the LLM is deleted and CUDA cache is cleared, so multiple
models can be scanned on a single GPU.

Usage:
    python3 examples/grpo_trainer/plot_collapse_heatmap.py \
        --model /data/chenyang2/Qwen3-8B,1,2 \
        --model /data/chenyang2/models/Qwen3.5-4B,1,0 \
        --model /data/chenyang2/models/Qwen2.5-7B-Instruct,1,0 \
        --model /data/chenyang2/models/Qwen3.5-35B-A3B,4,0 \
        --model /data/chenyang2/DeepSeek-R1-Distill-Qwen-1.5B,1,2 \
        --benchmark gsm8k:~/data/gsm8k/train.parquet \
        --benchmark math:~/data/math/train.parquet \
        --benchmark gpqa:~/data/gpqa_diamond/train.parquet \
        --benchmark mmlu_pro:~/data/mmlu_pro/test.parquet \
        --num-prompts 300 --top-k 8 \
        --out collapse_heatmap.png \
        --cache collapse_cache.pkl

Each --model is "path,tp,skip_tokens":
  tp           = tensor-parallel size (1, 2, 4, ...)
  skip_tokens  = response tokens to skip before analyzing (0 = first content token,
                 2 = skip thinking markers like IMD + newline)
"""

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse
import ast
import gc
import os
import pickle
from collections import defaultdict
from math import exp, log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams["figure.dpi"] = 130
rcParams["savefig.dpi"] = 200
rcParams["font.size"] = 11
rcParams["axes.titleweight"] = "bold"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def parse_messages(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return ast.literal_eval(val)
    return list(val)


def build_prompts(parquet_path, tokenizer, num_prompts):
    df = pd.read_parquet(parquet_path)
    if len(df) > num_prompts:
        df = df.sample(n=num_prompts, random_state=42).reset_index(drop=True)
    prompts = []
    for _, row in df.iterrows():
        msgs = parse_messages(row["prompt"])
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return prompts


# --------------------------------------------------------------------------- #
# Generation + aggregation
# --------------------------------------------------------------------------- #
def run_benchmark(llm, sampling_params, tokenizer, parquet_path, num_prompts, skip_tokens, max_model_len):
    """Return per-prompt top-logprob distributions at the target position.

    Prompts whose tokenized length exceeds (max_model_len - max_tokens) are
    skipped to avoid vLLM's "decoder prompt longer than max model length" error.
    """
    prompts = build_prompts(parquet_path, tokenizer, num_prompts)
    max_gen = skip_tokens + 1
    budget = max_model_len - max_gen

    # Filter out over-long prompts before sending to vLLM.
    kept = []
    skipped = 0
    for p in prompts:
        n = len(tokenizer(p, add_special_tokens=False).input_ids)
        if n > budget:
            skipped += 1
        else:
            kept.append(p)
    if skipped:
        print(f"    skipped {skipped}/{len(prompts)} prompts (>{budget} tokens)")
    print(f"    generating {len(kept)} prompts ...", flush=True)
    if not kept:
        return []
    outputs = llm.generate(kept, sampling_params)

    per_prompt = []
    fallback_count = 0
    for out in outputs:
        toks = out.outputs[0].token_ids
        lps = out.outputs[0].logprobs
        if not lps or len(toks) == 0:
            continue
        # Use the requested position; if the output is too short, fall back
        # to the LAST available token position so short-answer models (e.g.
        # Qwen2.5-7B-Instruct on multiple-choice) are not silently dropped.
        pos = skip_tokens
        if pos >= len(lps):
            pos = len(lps) - 1
            fallback_count += 1
        dist = {tid: exp(o.logprob) for tid, o in lps[pos].items()}
        per_prompt.append(dist)
    if fallback_count:
        print(f"    (fell back to last token for {fallback_count} short responses)")
    print(f"    -> {len(per_prompt)} valid distributions", flush=True)
    return per_prompt


def aggregate(per_prompt):
    """Aggregate per-prompt distributions into a single ranked distribution."""
    accum = defaultdict(float)
    n = len(per_prompt)
    for d in per_prompt:
        for tid, p in d.items():
            accum[tid] += p
    ranked = sorted(accum.items(), key=lambda x: -x[1])
    tids = [t for t, _ in ranked]
    avg = np.array([v / n for _, v in ranked]) if n else np.array([])
    return tids, avg


# --------------------------------------------------------------------------- #
# Collapse metrics
# --------------------------------------------------------------------------- #
def metrics_from_avg(avg, top_k):
    """Return dict of collapse metrics from an aggregated avg-prob vector."""
    if len(avg) == 0:
        return {"top_k_mass": float("nan"), "top1": float("nan"), "entropy": float("nan")}
    top_k_mass = float(avg[:top_k].sum())
    top1 = float(avg[0])
    p = avg[avg > 0]
    H = -float(np.sum(p * np.log(p)))
    Hmax = log(len(p)) if len(p) > 1 else 1.0
    entropy = H / Hmax if Hmax > 0 else 0.0
    return {"top_k_mass": top_k_mass, "top1": top1, "entropy": entropy}


# --------------------------------------------------------------------------- #
# Main scan
# --------------------------------------------------------------------------- #
def scan(args, cache_path):
    """Run all (model, benchmark) pairs, return results dict and cache to disk."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache with {len(cache)} cells from {cache_path}")

    results = {}
    from transformers import AutoTokenizer

    # Parse per-model tp and skip_tokens:
    #   "path,tp,skip"   -> tp, skip  (all three required)
    model_specs = []
    for spec in args.model:
        parts = spec.split(",")
        if len(parts) < 3 or not parts[1].strip() or not parts[2].strip():
            raise SystemExit(
                f"Each --model must be 'path,tp,skip_tokens', got: {spec!r}\n"
                f"  e.g. --model /path/Qwen3-8B,1,2  (skip 2 thinking tokens)\n"
                f"       --model /path/Qwen2.5-7B,1,0 (no skip)")
        path = parts[0]
        tp = int(parts[1])
        skip = int(parts[2])
        model_specs.append((path, tp, skip))

    for model_path, model_tp, model_skip in model_specs:
        model_label = os.path.basename(os.path.normpath(model_path))

        # Skip whole model if every cell is cached AND valid.
        def _cell_valid(key):
            v = cache.get(key)
            return v is not None and v.get("n_prompts", 0) > 0 and not np.isnan(v.get("top1", np.nan))

        all_cached = all(_cell_valid((model_label, b[0])) for b in args.benchmark)
        if all_cached:
            print(f"\n[{model_label}] fully cached, skipping generation.")
            for b_label, _ in args.benchmark:
                results[(model_label, b_label)] = cache[(model_label, b_label)]
            continue

        print(f"\n=== Loading model: {model_label} ({model_path}) "
              f"tp={model_tp} skip_tokens={model_skip} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            tensor_parallel_size=model_tp,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            dtype=args.dtype,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=model_skip + 1,
            logprobs=20,
        )

        for b_label, b_path in args.benchmark:
            key = (model_label, b_label)
            if _cell_valid(key):
                print(f"  [{b_label}] cached.")
                results[key] = cache[key]
                continue
            if key in cache:
                print(f"  [{b_label}] cached but invalid (n_prompts=0), re-running.")
            else:
                print(f"  [{b_label}] {b_path}")
            per_prompt = run_benchmark(
                llm, sampling_params, tokenizer, b_path,
                args.num_prompts, model_skip, args.max_model_len,
            )
            tids, avg = aggregate(per_prompt)
            m = metrics_from_avg(avg, args.top_k)
            m["n_prompts"] = len(per_prompt)
            m["top_tokens"] = [tokenizer.decode([t]) for t in tids[:args.top_k]]
            results[key] = m
            cache[key] = m
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
            print(f"    top-{args.top_k} mass={m['top_k_mass']*100:.1f}%  "
                  f"top1={m['top1']*100:.1f}%  entropy={m['entropy']:.3f}")

        print(f"  freeing {model_label} ...")
        del llm
        del tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _setup_cjk_font():
    """Pick a CJK-capable font so Chinese titles render instead of tofu boxes."""
    for fam in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
                "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"):
        try:
            from matplotlib.font_manager import findfont, FontProperties
            if findfont(FontProperties(family=fam), fallback_to_default=False):
                rcParams["font.family"] = fam
                rcParams["axes.unicode_minus"] = False
                return fam
        except Exception:
            continue
    return None


def _cell_text(v, fmt, vmin, vmax, cmap_invert):
    if np.isnan(v):
        return "—", "grey"
    txt = fmt(v)
    frac = (v - vmin) / (vmax - vmin + 1e-9)
    if cmap_invert:
        color = "white" if frac < 0.45 else "black"
    else:
        color = "white" if frac > 0.55 else "black"
    return txt, color


def plot_single_heatmap(results, model_labels, bench_labels, metric, top_k, out_path):
    """Render a benchmark (rows) x model (columns) heatmap for one metric.

    Models are placed on the x-axis so multiple models form a wide matrix;
    benchmarks stack on the y-axis.
    """
    _setup_cjk_font()
    meta = {
        "top_k_mass": dict(
            title=f"Top-{top_k} 概率质量占比 (越高 = 坍塌越严重)",
            fmt=lambda v: f"{v*100:.1f}%", vmin=0.0, vmax=1.0, cmap="YlOrRd", invert=False),
        "top1": dict(
            title="Top-1 token 平均概率 (越高 = 坍塌越严重)",
            fmt=lambda v: f"{v*100:.1f}%", vmin=0.0, vmax=1.0, cmap="YlOrRd", invert=False),
        "entropy": dict(
            title="归一化熵 (越低 = 坍塌越严重)",
            fmt=lambda v: f"{v:.3f}", vmin=0.0, vmax=1.0, cmap="YlOrRd_r", invert=True),
    }[metric]

    # matrix shape: M rows (models) x B cols (benchmarks)
    # models on y-axis, benchmarks on x-axis
    M, B = len(model_labels), len(bench_labels)
    mat = np.full((M, B), np.nan)
    for i, m in enumerate(model_labels):
        for j, b in enumerate(bench_labels):
            mat[i, j] = results.get((m, b), {}).get(metric, np.nan)

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * B + 2), max(4, 0.7 * M + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap=meta["cmap"],
                   vmin=meta["vmin"], vmax=meta["vmax"])
    ax.set_xticks(range(B)); ax.set_xticklabels(bench_labels, rotation=30, ha="right")
    ax.set_yticks(range(M)); ax.set_yticklabels(model_labels)
    ax.set_xlabel("Benchmark"); ax.set_ylabel("Model")
    for i in range(M):
        for j in range(B):
            txt, color = _cell_text(mat[i, j], meta["fmt"], meta["vmin"], meta["vmax"], meta["invert"])
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(meta["title"], rotation=90)
    ax.set_title(f"首 token 坍塌: {meta['title']}", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved heatmap to {out_path}")
    plt.close(fig)


def plot_combined_heatmap(results, model_labels, bench_labels, top_k, out_path):
    """Render top_k_mass / top1 / entropy side by side in one figure."""
    metrics = [
        ("top_k_mass", f"Top-{top_k} 概率质量", lambda v: f"{v*100:.1f}%", "YlOrRd", 0, 1, False),
        ("top1", "Top-1 概率", lambda v: f"{v*100:.1f}%", "YlOrRd", 0, 1, False),
        ("entropy", "归一化熵", lambda v: f"{v:.3f}", "YlOrRd_r", 0, 1, True),
    ]
    M, B = len(model_labels), len(bench_labels)
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * B + 3, max(3.5, 0.85 * M + 1.2)))
    for ax, (metric, title, fmt, cmap, vmin, vmax, invert) in zip(axes, metrics):
        mat = np.full((M, B), np.nan)
        for i, m in enumerate(model_labels):
            for j, b in enumerate(bench_labels):
                mat[i, j] = results.get((m, b), {}).get(metric, np.nan)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(B)); ax.set_xticklabels(bench_labels, rotation=20, ha="right")
        ax.set_yticks(range(M)); ax.set_yticklabels(model_labels)
        ax.set_title(title)
        for i in range(M):
            for j in range(B):
                txt, color = _cell_text(mat[i, j], fmt, vmin, vmax, invert)
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("首 token 坍塌: model × benchmark", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved combined heatmap to {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_spec(spec, kind):
    if ":" not in spec or not all(p.strip() for p in spec.split(":", 1)):
        raise SystemExit(f"--{kind} must be 'label:path', got: {spec}")
    label, path = spec.split(":", 1)
    return label, os.path.expanduser(path)


def discover_benchmarks(data_dir, split_priority=("train", "test", "validation")):
    """Auto-discover benchmark parquet files under data_dir.

    Each subdirectory of data_dir is treated as one benchmark, named after
    the directory. Within each subdirectory we pick the first parquet file
    matching the split priority order (train > test > validation). If none of
    the priority splits exist, fall back to any *.parquet in the subdir.

    Returns a list of (label, path) tuples sorted by label.
    """
    data_dir = os.path.expanduser(data_dir)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"--data-dir not found: {data_dir}")

    found = []
    for name in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, name)
        if not os.path.isdir(sub):
            continue
        chosen = None
        for split in split_priority:
            cand = os.path.join(sub, f"{split}.parquet")
            if os.path.isfile(cand):
                chosen = cand
                break
        if chosen is None:
            # fallback: any parquet in the subdir
            pq = [f for f in os.listdir(sub) if f.endswith(".parquet")]
            if pq:
                chosen = os.path.join(sub, sorted(pq)[0])
        if chosen is not None:
            found.append((name, chosen))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=[],
                   help="One model path per flag; repeat --model for each model. "
                        "Optionally append ',tp,skip' or ',tp,0' to set tensor-parallel "
                        "size and skip_tokens per model, e.g. "
                        "--model /path/Qwen3-8B,1,2 --model /path/Qwen3.5-35B-A3B,4,2. "
                        "In --plot-only mode, --model is optional (uses all models in --cache).")
    p.add_argument("--benchmark", nargs="+", default=None,
                   help="One or more 'label:path' entries, e.g. gsm8k:~/data/gsm8k/train.parquet. "
                        "If omitted, --data-dir is used to auto-discover benchmarks.")
    p.add_argument("--data-dir", default=None,
                   help="Auto-discover benchmarks under this directory. Each subdirectory "
                        "is one benchmark (named after the dir); picks train.parquet > "
                        "test.parquet > validation.parquet. e.g. --data-dir /data/chenyang2/verl/data")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Benchmark labels to skip when using --data-dir (e.g. --exclude math musr)")
    p.add_argument("--num-prompts", type=int, default=300)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM max model len; prompts longer than this are skipped")
    p.add_argument("--dtype", default="auto",
                   help="vLLM dtype: auto/bfloat16/float16/float32")
    p.add_argument("--out", default="collapse_heatmap.png")
    p.add_argument("--cache", default="collapse_cache.pkl",
                   help="Pickle cache for incremental save / re-plotting without re-running vLLM")
    p.add_argument("--metric", default="top1",
                   choices=["top1", "top_k_mass", "entropy"],
                   help="Metric for the heatmap (default: top1)")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip generation; just read --cache and render the heatmap. "
                        "Use this to draw after running each model in its own env.")
    p.add_argument("--combined", action="store_true",
                   help="Also save a 3-panel figure with all metrics side by side")
    args = p.parse_args()

    # Resolve benchmark list: explicit --benchmark wins; otherwise auto-discover.
    # (Skipped in --plot-only mode: use whatever is in the cache.)
    if not args.plot_only:
        if args.benchmark:
            args.benchmark = [_parse_spec(s, "benchmark") for s in args.benchmark]
        elif args.data_dir:
            args.benchmark = discover_benchmarks(args.data_dir)
            if not args.benchmark:
                raise SystemExit(f"No parquet files found under --data-dir {args.data_dir}")
        else:
            raise SystemExit("Provide either --benchmark label:path ... or --data-dir")

    if args.exclude and args.benchmark:
        excl = set(args.exclude)
        args.benchmark = [(l, p) for l, p in args.benchmark if l not in excl]

    if args.plot_only:
        # Just read the cache and plot — no vLLM, no GPU needed.
        if not os.path.exists(args.cache):
            raise SystemExit(f"--plot-only but cache not found: {args.cache}")
        with open(args.cache, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache with {len(cache)} cells from {args.cache}")
        results = cache
        # Derive the full set of models/benchmarks present in the cache so the
        # matrix contains every model that has been run, regardless of which
        # --model flags were passed on this invocation.
        cache_models = sorted({k[0] for k in cache})
        cache_benches = sorted({k[1] for k in cache})
        # Keep the ordering: prefer the order given on the CLI, then any extras
        # found in the cache.
        cli_models = [os.path.basename(os.path.normpath(m.split(",")[0])) for m in args.model]
        model_labels = [m for m in cli_models if m in cache_models] + \
                        [m for m in cache_models if m not in cli_models]
        cli_benches = [b[0] for b in (args.benchmark or [])]
        bench_labels = [b for b in cli_benches if b in cache_benches] + \
                       [b for b in cache_benches if b not in cli_benches]
        print(f"Models:     {model_labels}")
        print(f"Benchmarks: {bench_labels}")
        print(f"Metric:     {args.metric}  top-k={args.top_k}")
    else:
        model_labels = [os.path.basename(os.path.normpath(m.split(",")[0])) for m in args.model]
        bench_labels = [b[0] for b in args.benchmark]
        results = scan(args, args.cache)
        # Merge cached cells that belong to other models so the CSV/heatmap
        # reflects everything computed so far, not just this run's --model.
        if os.path.exists(args.cache):
            with open(args.cache, "rb") as f:
                full_cache = pickle.load(f)
            for k, v in full_cache.items():
                results.setdefault(k, v)
        print(f"Models:     {model_labels}")
        print(f"Benchmarks: {bench_labels}")
        print(f"Metric:     {args.metric}  top-k={args.top_k}")

    # always dump a CSV for easy inspection
    rows = []
    for m in model_labels:
        for b in bench_labels:
            r = results.get((m, b), {})
            rows.append({
                "model": m, "benchmark": b,
                "top_k_mass": r.get("top_k_mass", np.nan),
                "top1": r.get("top1", np.nan),
                "entropy": r.get("entropy", np.nan),
                "n_prompts": r.get("n_prompts", 0),
                "top_tokens": " | ".join(r.get("top_tokens", [])),
            })
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved raw metrics to {csv_path}")

    plot_single_heatmap(results, model_labels, bench_labels,
                       args.metric, args.top_k, args.out)
    if args.combined:
        combined_path = os.path.splitext(args.out)[0] + "_combined.png"
        plot_combined_heatmap(results, model_labels, bench_labels, args.top_k, combined_path)


if __name__ == "__main__":
    main()
