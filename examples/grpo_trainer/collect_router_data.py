#!/usr/bin/env python3
"""Collect router training data in a single phase, with batch support.

For each prompt, try K forced first tokens (1 rollout each), record reward vector.
Supports batch processing: split all prompts into chunks of --batch-size, save each
chunk to a separate file in --save-dir. Skips already-completed batches (resume).

Usage (batch mode, recommended):
  python collect_router_data.py \
      --model /workspace/Qwen3-8B \
      --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
      --num-prompts -1 \
      --forced-tokens 32313,71486,... \
      --tp 4 --gpu-mem-util 0.9 --max-tokens 4096 \
      --save-dir /data/chenyang2/router_data_batches \
      --batch-size 1000

Usage (single-file mode):
  python collect_router_data.py \
      --model /workspace/Qwen3-8B \
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
import time
import pandas as pd
import torch
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer, em_check, subem_check
from mmlu_pro_reward import compute_score_mmlu_pro


# Default forced first tokens: top-20 by probability mass on Qwen3-8B
# (analyzed over 2000 prompts pooled from all 6 datasets, skip-tokens=2).
# 'Okay' (32313) dominates at ~99.98% natural probability; the other 19 are
# off-policy exploration tokens that the router learns to pick when they help.
DEFAULT_FORCED_TOKENS = [
    32313,   # Okay      (dominant natural start)
    71486,   # Alright
    785,     # The
    4416,    # So
    40,      # I
    93217,   # okay
    1249,    # To
    10061,   # Let
    16141,   # Answer
    106287,  # 嗯
    35439,   #  Okay
    99692,   # 好的
    1986,    # This
    16910,   #  okay
    3925,    # OK
    2132,    # It
    641,     # In
    7039,    # Now
    151668,  # <think> special token
    5338,    # First
]


# Datasets whose ground_truth is a single multiple-choice letter (A-J).
_MC_DATA_SOURCES = {
    "allenai/ai2_arc/ARC-Challenge",
    "datatune/LogiQA2.0/mrc",
    "TAUR-Lab/MuSR",
}
# Datasets whose ground_truth is a short text span (exact-match / substring).
_SPAN_DATA_SOURCES = {
    "ucinlp/drop",
}
# Code-execution datasets: ground_truth is a JSON string with
# {"solutions": [...], "input_output": {...}, "fn_name": "...}.
_CODE_DATA_SOURCES = {
    "BAAI/TACO",
}
# Free-form exact-match datasets (normalized string comparison).
_EM_DATA_SOURCES = {
    "BBEH/bbeh",
}


def _extract_after_marker(text, marker="####"):
    """Extract the final answer after the last `####` marker (DROP style)."""
    idx = text.rfind(marker)
    if idx < 0:
        # fall back to last non-empty line
        tail = text.strip().splitlines()
        return tail[-1].strip() if tail else ""
    return text[idx + len(marker):].strip().splitlines()[0].strip() if text[idx + len(marker):].strip() else ""


def compute_score_span(response, ground_truth, accepted_answers=None):
    """EM / substring reward for span-extraction datasets (e.g. DROP).

    ground_truth is the canonical answer string; accepted_answers (optional)
    is a list of alternative acceptable spans (may be a numpy array).
    """
    pred = _extract_after_marker(response)
    if not pred:
        return 0.0
    # normalize accepted_answers (could be a numpy array / Series / None)
    if accepted_answers is None:
        accepted = []
    else:
        try:
            accepted = list(accepted_answers)
        except TypeError:
            accepted = []
    gold = [ground_truth] + accepted
    # strict EM first, then substring match as a fallback
    if em_check(pred, gold):
        return 1.0
    if subem_check(pred, gold):
        return 0.5
    return 0.0


def compute_score_code(response, ground_truth, n_cases=None, threshold=0.5):
    """Code-execution reward for TACO. ground_truth is a JSON string.

    Tests the first `n_cases` test cases INDIVIDUALLY (not the all-pairs
    combined check, which mismatches when solutions read one input() per
    case). Returns 1.0 if the pass rate >= `threshold`, else 0.0.

    n_cases defaults to env var TACO_N_CASES (or 10 if unset). Lower it
    (e.g. 3) for faster scoring during tests.
    """
    import json as _json
    from verl.utils.reward_score.prime_code.utils import check_correctness as _check
    if n_cases is None:
        n_cases = int(os.environ.get("TACO_N_CASES", "10"))
    try:
        gt = _json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except Exception:
        return 0.0
    if not isinstance(gt, dict):
        return 0.0
    io = gt.get("input_output", {})
    if not isinstance(io, dict) or not io.get("inputs"):
        return 0.0
    # extract code from the ```python ... ``` block
    try:
        solution = response.split("```python")[-1].split("```")[0]
    except Exception:
        return 0.0
    # strip leading/trailing whitespace: a leading blank line makes run_test
    # wrap the whole solution in `def code():` (breaking top-level imports).
    solution = solution.strip()
    inputs, outputs = io["inputs"], io["outputs"]
    n = min(n_cases, len(inputs), len(outputs))
    if n == 0:
        return 0.0
    n_pass = 0
    for c in range(n):
        tc = {"inputs": [inputs[c]], "outputs": [outputs[c]]}
        try:
            res, _ = _check(in_outs=tc, generation=solution, timeout=10, debug=False)
            if res and res[0] is True:
                n_pass += 1
        except Exception:
            pass
    pass_rate = n_pass / n
    return 1.0 if pass_rate >= threshold else 0.0


def compute_score_em(response, ground_truth):
    """Free-form exact-match reward (normalized string comparison).

    Used for BBEH where the target is a short text answer (e.g., "proved",
    "disproved", a number, etc.). Extracts the last non-empty line of the
    response, normalizes, and compares with the normalized ground truth.
    """
    import re
    # Extract the last non-empty line as the predicted answer
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    pred = lines[-1] if lines else response.strip()
    # Remove common answer prefixes like "Answer:", "The answer is", etc.
    pred = re.sub(r"^(answer|the answer is|final answer)\s*[:：]?\s*", "", pred, flags=re.IGNORECASE).strip()
    # Remove surrounding quotes, backticks, \boxed{}
    pred = re.sub(r"\\boxed\{([^}]*)\}", r"\1", pred)
    pred = pred.strip("\"'` ")
    # Normalize: lowercase, collapse whitespace, strip punctuation
    def norm(s):
        s = str(s).strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = s.strip(".,;:!?")
        return s
    if norm(pred) == norm(ground_truth):
        return 1.0
    return 0.0


def compute_score(response, data_source, ground_truth, extra_info=None):
    # Multiple-choice (letter) reward: MMLU-Pro / GPQA / ARC-Challenge / LogiQA2.0 / MuSR
    if ("MMLU-Pro" in data_source or "mmlu" in data_source.lower()
            or "gpqa" in data_source.lower() or data_source in _MC_DATA_SOURCES):
        return compute_score_mmlu_pro(response, ground_truth)
    # Span-extraction reward: DROP
    if data_source in _SPAN_DATA_SOURCES:
        accepted = None
        if isinstance(extra_info, dict):
            accepted = extra_info.get("answer_spans")
        return compute_score_span(response, ground_truth, accepted_answers=accepted)
    # Code-execution reward: TACO
    if data_source in _CODE_DATA_SOURCES:
        return compute_score_code(response, ground_truth)
    # Free-form exact-match reward: BBEH
    if data_source in _EM_DATA_SOURCES:
        return compute_score_em(response, ground_truth)
    try:
        score = default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
        )
        if score is None:
            return 0.0
        if isinstance(score, dict):
            # math_dapo returns {'score': float, 'acc': bool, 'pred': str}
            # Clamp to [0, 1] for binary router reward
            return max(0.0, min(1.0, float(score.get("score", 0.0))))
        return float(score)
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


# Difficulty weights per data_source for balanced sampling.
# Harder datasets have LOWER survival rate (more "all-wrong" groups filtered out
# during router-data collection), so they need MORE upfront samples to end up
# with a comparable number of usable prompts after filtering.
# weight = relative upfront quota multiplier (harder => larger).
DEFAULT_DIFFICULTY_WEIGHTS = {
    "openai/gsm8k":                 1.5,   # signal rate 80% (great)
    "DigitalLearningGmbH/MATH-lighteval": 2.0,  # signal rate 57%
    "allenai/ai2_arc/ARC-Challenge": 0.5,  # signal rate 43%, too easy (57% all-correct)
    "datatune/LogiQA2.0/mrc":       2.5,   # signal rate 82% (best)
    "ucinlp/drop":                  1.5,   # signal rate 57%
}

# TACO difficulty levels to KEEP when pre-filtering (skip EASY=too easy / VERY_HARD
# = almost always all-wrong). None = keep all.
TACO_KEEP_DIFFICULTIES = {"MEDIUM", "MEDIUM_HARD", "HARD"}


def _sample_per_source(df, num_prompts, per_source_n, difficulty_weighted, weights, base_n, taco_difficulty_filter):
    """Return a dataframe sampled with per-source quotas.

    - per_source_n > 0: fixed N per data_source (equal quota).
    - difficulty_weighted: N = base_n * weight[data_source] per source.
    - otherwise (num_prompts): legacy pool-and-random-sample behaviour.
    """
    if per_source_n <= 0 and not difficulty_weighted:
        # legacy mode: pool all and sample num_prompts
        total = len(df)
        if num_prompts < 0 or total <= num_prompts:
            df = df.sample(n=total, random_state=42)
            print(f"Using all {total} prompts (shuffled)")
        else:
            df = df.sample(n=num_prompts, random_state=42)
            print(f"Sampled {num_prompts} prompts from {total} (pooled random)")
        return df

    # TACO difficulty pre-filter (drop too-easy / too-hard to maximize boundary signal)
    if taco_difficulty_filter and "BAAI/TACO" in df["data_source"].unique():
        mask = ~((df["data_source"] == "BAAI/TACO"))
        if "extra_info" in df.columns:
            def _taco_diff_ok(row):
                if row["data_source"] != "BAAI/TACO":
                    return True
                ei = row.get("extra_info", {})
                if isinstance(ei, str):
                    import json as _json
                    try:
                        ei = _json.loads(ei)
                    except Exception:
                        ei = {}
                ei = ei if isinstance(ei, dict) else {}
                d = str(ei.get("difficulty", "")).upper()
                return d in TACO_KEEP_DIFFICULTIES
            taco_mask = df.apply(_taco_diff_ok, axis=1)
        else:
            taco_mask = mask
        n_before = int((df["data_source"] == "BAAI/TACO").sum())
        df = df[taco_mask]
        n_after = int((df["data_source"] == "BAAI/TACO").sum())
        print(f"TACO difficulty pre-filter: {n_before} -> {n_after} "
              f"(kept {sorted(TACO_KEEP_DIFFICULTIES)})")

    parts = []
    print(f"\nPer-source sampling:")
    print(f"  {'data_source':40} {'available':>10} {'quota':>7} {'sampled':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*7} {'-'*8}")
    for ds, sub in df.groupby("data_source"):
        avail = len(sub)
        if per_source_n > 0:
            quota = per_source_n
        else:  # difficulty_weighted
            w = weights.get(ds, 1.0)
            quota = max(1, int(round(base_n * w)))
        n_take = min(avail, quota)
        sampled = sub.sample(n=n_take, random_state=42)
        parts.append(sampled)
        print(f"  {ds:40} {avail:>10} {quota:>7} {n_take:>8}")
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42)  # shuffle
    print(f"  {'TOTAL':40} {len(df):>10} {'':>7} {len(out):>8}")
    return out


def load_and_prepare_data(data_paths, num_prompts, tokenizer,
                          per_source_n=0, difficulty_weighted=False,
                          weights=None, base_n=100, taco_difficulty_filter=True):
    """Load parquet files, sample (optionally per-source balanced), build prompts."""
    dfs = [pd.read_parquet(p) for p in data_paths]
    df = pd.concat(dfs, ignore_index=True)

    df = _sample_per_source(
        df, num_prompts, per_source_n, difficulty_weighted,
        weights or DEFAULT_DIFFICULTY_WEIGHTS, base_n, taco_difficulty_filter,
    )
    total = len(df)
    print(f"\nTotal prompts to use: {total}")

    prompts_text = []
    ground_truths = []
    data_sources = []
    extra_infos = []
    for _, row in df.iterrows():
        messages = parse_messages(row["prompt"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts_text.append(text)
        rm = row.get("reward_model", {})
        if isinstance(rm, str):
            import json as _json
            try:
                rm = _json.loads(rm)
            except Exception:
                rm = {}
        ground_truths.append(rm.get("ground_truth", "") if isinstance(rm, dict) else "")
        data_sources.append(row.get("data_source", ""))
        ei = row.get("extra_info", {})
        if isinstance(ei, str):
            import json as _json
            try:
                ei = _json.loads(ei)
            except Exception:
                ei = {}
        extra_infos.append(ei if isinstance(ei, dict) else {})

    return prompts_text, ground_truths, data_sources, extra_infos


def _score_one_trajectory(args):
    """Score a single trajectory. Returns (flat_index, score)."""
    flat_idx, resp, ds, gt, ei = args
    return flat_idx, compute_score(resp, ds, gt, extra_info=ei)


def score_trajectories_parallel(responses, data_sources, ground_truths,
                                  extra_infos, n_workers=64, label=""):
    """Score a list of trajectories in parallel using a thread pool.

    TACO code-execution scoring spawns subprocesses (releases the GIL while
    waiting), so threads give real parallelism on the many CPU cores.
    """
    n = len(responses)
    if n == 0:
        return []
    if n_workers <= 1 or n <= 1:
        print(f"  [{label}] Scoring {n} trajectories (sequential) ...")
        return [compute_score(responses[i], data_sources[i], ground_truths[i],
                              extra_infos[i]) for i in range(n)]
    print(f"  [{label}] Scoring {n} trajectories with {n_workers} workers ...")
    args_list = [(i, responses[i], data_sources[i], ground_truths[i], extra_infos[i])
                 for i in range(n)]
    scores = [0.0] * n
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_score_one_trajectory, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            flat_idx, score = fut.result()
            scores[flat_idx] = score
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                elapsed = time.time() - t0
                print(f"  [{label}] scored {done}/{n} "
                      f"({done/n*100:.0f}%, {elapsed:.0f}s)")
    return scores


def build_prompt_objects(tokenizer, prompts_text, forced_token_list, indices,
                          ground_truths, data_sources, extra_infos):
    """Build TokensPrompt objects for given prompt indices x K forced tokens.

    Returns (prompt_objs, flat_gts, flat_ds, flat_ei) where each flat list has
    len(indices) * K entries ordered as [prompt0_tok0, prompt0_tok1, ..., prompt1_tok0, ...].
    """
    K = len(forced_token_list)
    objs, flat_gts, flat_ds, flat_ei = [], [], [], []
    for p_idx in indices:
        prompt_ids = tokenizer.encode(prompts_text[p_idx], add_special_tokens=False)
        for k in range(K):
            objs.append(TokensPrompt(prompt_token_ids=prompt_ids + [forced_token_list[k]]))
            flat_gts.append(ground_truths[p_idx])
            flat_ds.append(data_sources[p_idx])
            flat_ei.append(extra_infos[p_idx] if p_idx < len(extra_infos) else {})
    return objs, flat_gts, flat_ds, flat_ei


def scores_to_reward_matrix(scores, n_prompts, K):
    """Convert flat scores [n_prompts * K] into a [n_prompts, K] reward matrix."""
    rm = torch.zeros(n_prompts, K, dtype=torch.float32)
    for i, s in enumerate(scores):
        rm[i // K, i % K] = s
    return rm


def process_batch(llm, sp, tokenizer, forced_token_list,
                   prompts_text, ground_truths, data_sources, extra_infos, batch_idx):
    """Process one batch of prompts through vLLM, return reward matrix."""
    K = len(forced_token_list)
    n = len(prompts_text)

    # Build all prompt objects: each prompt x K forced tokens
    all_prompt_objs = []
    all_gts = []
    all_ds = []
    all_ei = []
    for p_idx in range(n):
        prompt_ids = tokenizer.encode(prompts_text[p_idx], add_special_tokens=False)
        for k in range(K):
            forced_tok = forced_token_list[k]
            ids = prompt_ids + [forced_tok]
            all_prompt_objs.append(TokensPrompt(prompt_token_ids=ids))
            all_gts.append(ground_truths[p_idx])
            all_ds.append(data_sources[p_idx])
            all_ei.append(extra_infos[p_idx] if p_idx < len(extra_infos) else {})

    total_traj = len(all_prompt_objs)
    print(f"\n[Batch {batch_idx}] Generating {total_traj} trajectories "
          f"({n} prompts x {K} tokens) ...")

    outputs = llm.generate(all_prompt_objs, sampling_params=sp)

    # Score
    print(f"[Batch {batch_idx}] Scoring {total_traj} trajectories ...")
    scores = []
    for i, out in enumerate(outputs):
        resp = out.outputs[0].text
        score = compute_score(resp, all_ds[i], all_gts[i], extra_info=all_ei[i])
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
    parser.add_argument("--model", default="/workspace/Qwen3-8B")
    parser.add_argument("--data", nargs="+", help="Parquet data files")
    parser.add_argument("--num-prompts", type=int, default=500,
                        help="Total prompts to process (-1 = all, shuffled). "
                             "Only used when not in per-source / difficulty-weighted mode.")
    parser.add_argument("--per-source-n", type=int, default=0,
                        help="If >0, sample exactly N prompts per data_source (equal quota). "
                             "Overrides num-prompts.")
    parser.add_argument("--difficulty-weighted", action="store_true",
                        help="Sample per-source with quotas = base_n * difficulty_weight "
                             "(harder datasets get more, since more are filtered as all-wrong).")
    parser.add_argument("--base-n", type=int, default=100,
                        help="Base quota for --difficulty-weighted (quota = base_n * weight).")
    parser.add_argument("--no-taco-difficulty-filter", action="store_true",
                        help="Disable TACO difficulty pre-filter (keep EASY/VERY_HARD too).")
    parser.add_argument("--forced-tokens", type=str, default=None,
                        help="Comma-separated token ids. If omitted, uses the built-in "
                             "DEFAULT_FORCED_TOKENS (20 tokens analyzed on Qwen3-8B).")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max generation tokens for ALL datasets.")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)

    # Parallel TACO scoring
    parser.add_argument("--parallel-taco", action="store_true", default=True,
                        help="Generate TACO first, then score TACO in a background "
                             "thread pool WHILE generating+scoring other datasets on "
                             "the GPU. Overlaps CPU-bound TACO scoring with GPU-bound "
                             "generation. (default: on when TACO is present)")
    parser.add_argument("--no-parallel-taco", dest="parallel_taco", action="store_false",
                        help="Disable parallel TACO scoring (use legacy sequential mode).")
    parser.add_argument("--taco-score-workers", type=int, default=0,
                        help="Thread pool size for parallel TACO scoring (0 = auto, "
                             "uses min(nproc//2, 128)).")

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
    parser.add_argument("--batch-end", type=int, default=-1,
                        help="Stop after this batch index (-1 = run to end). "
                             "e.g. --batch-start 2 --batch-end 2 runs only batch 2")

    # Merge mode
    parser.add_argument("--merge", action="store_true",
                        help="Merge all batch files in --save-dir into --save-data")

    # Extract-only mode: sample a fixed dataset and save to parquet (no vLLM)
    parser.add_argument("--extract-only", type=str, default=None,
                        help="Sample a balanced dataset using --difficulty-weighted / "
                             "--base-n (or --per-source-n) and save to this parquet path. "
                             "Does NOT run vLLM. Use the output file with --data for "
                             "collection later.")

    args = parser.parse_args()

    # ---- Merge mode ----
    if args.merge:
        if not args.save_dir or not args.save_data:
            print("--merge requires --save-dir and --save-data")
            sys.exit(1)
        merge_batches(args.save_dir, args.save_data)
        return

    # ---- Extract-only mode: sample a fixed balanced dataset, save to parquet ----
    if args.extract_only:
        if not args.data:
            print("--extract-only requires --data")
            sys.exit(1)
        print(f"Extract-only mode: sampling balanced dataset -> {args.extract_only}")
        import pandas as pd
        dfs = [pd.read_parquet(p) for p in args.data]
        df_all = pd.concat(dfs, ignore_index=True)
        df_sampled = _sample_per_source(
            df_all, args.num_prompts, args.per_source_n, args.difficulty_weighted,
            DEFAULT_DIFFICULTY_WEIGHTS, args.base_n,
            not args.no_taco_difficulty_filter,
        )
        # Serialize dict columns to JSON strings (mixed schemas across datasets
        # break pyarrow). load_and_prepare_data handles both dicts and JSON strings.
        import json as _json
        import numpy as _np

        def _to_jsonable(x):
            if isinstance(x, _np.ndarray):
                return x.tolist()
            if isinstance(x, _np.generic):
                return x.item()
            if isinstance(x, dict):
                return {k: _to_jsonable(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_to_jsonable(v) for v in x]
            return x

        # Keep only columns needed for collection; drop mixed-schema metadata (id, etc.)
        keep_cols = [c for c in ("prompt", "reward_model", "extra_info", "data_source")
                     if c in df_sampled.columns]
        df_out = df_sampled[keep_cols].copy()
        for col in ("reward_model", "extra_info"):
            if col in df_out.columns:
                df_out[col] = df_out[col].apply(
                    lambda x: _json.dumps(_to_jsonable(x)) if isinstance(x, (dict, list)) else x)
        df_out.to_parquet(args.extract_only, index=False)
        print(f"\nSaved {len(df_sampled)} prompts to {args.extract_only}")
        print(f"Per-source breakdown:")
        for ds, sub in df_sampled.groupby("data_source"):
            print(f"  {ds:42} {len(sub):>5}")
        return

    # ---- Validate ----
    if not args.data:
        print("--data is required (unless --merge)")
        sys.exit(1)
    if not args.save_data and not args.save_dir:
        print("Need --save-data (single file) or --save-dir (batch mode)")
        sys.exit(1)

    # Use built-in default token list if none provided
    if args.forced_tokens:
        forced_token_list = [int(x.strip()) for x in args.forced_tokens.split(",")]
    else:
        forced_token_list = list(DEFAULT_FORCED_TOKENS)
        print(f"Using built-in DEFAULT_FORCED_TOKENS ({len(forced_token_list)} tokens)")
    K = len(forced_token_list)
    print(f"K = {K} forced tokens: {forced_token_list}")

    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  decoded: {[repr(tokenizer.decode([t])) for t in forced_token_list]}")

    # 2. Load and prepare all data
    prompts_text, ground_truths, data_sources, extra_infos = load_and_prepare_data(
        args.data, args.num_prompts, tokenizer,
        per_source_n=args.per_source_n,
        difficulty_weighted=args.difficulty_weighted,
        base_n=args.base_n,
        taco_difficulty_filter=not args.no_taco_difficulty_filter,
    )
    total_prompts = len(prompts_text)
    print(f"Total prompts to process: {total_prompts}")

    # 3. Determine batch mode or single mode
    use_batch = args.save_dir is not None and args.batch_size > 0

    if use_batch:
        os.makedirs(args.save_dir, exist_ok=True)
        num_batches = (total_prompts + args.batch_size - 1) // args.batch_size
        batch_end = num_batches if args.batch_end < 0 else args.batch_end + 1
        print(f"Batch mode: {num_batches} batches x {args.batch_size} prompts")
        print(f"  Save dir: {args.save_dir}")
        print(f"  Running batches: {args.batch_start}..{batch_end - 1}")
    else:
        num_batches = 1
        args.batch_size = total_prompts
        batch_end = 1
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

    # 5. Decide pipeline mode
    taco_indices = [i for i, ds in enumerate(data_sources) if ds in _CODE_DATA_SOURCES]
    other_indices = [i for i, ds in enumerate(data_sources) if ds not in _CODE_DATA_SOURCES]
    n_taco = len(taco_indices)
    n_other = len(other_indices)
    use_parallel = args.parallel_taco and n_taco > 0 and n_other > 0
    if args.parallel_taco and n_taco > 0:
        taco_workers = args.taco_score_workers
        if taco_workers <= 0:
            taco_workers = min(os.cpu_count() // 2 or 1, 128)
        print(f"\nTACO prompts: {n_taco}, other prompts: {n_other}")
        print(f"Parallel TACO scoring: {'ON' if use_parallel else 'OFF'} "
              f"(workers={taco_workers if n_taco > 0 else 'n/a'})")
    else:
        taco_workers = 1

    all_reward_matrices = []

    if use_parallel:
        # ================================================================
        # PARALLEL PIPELINE:
        #   Phase 1: Generate TACO trajectories (GPU)
        #   Phase 2: Score TACO in background thread pool (CPU)  ||  Phase 3: Generate+score other (GPU+CPU)
        #   Phase 4: Collect TACO scores, save TACO batches
        # ================================================================
        import threading

        # --- Phase 1: Generate TACO trajectories ---
        print(f"\n{'='*70}")
        print(f"PHASE 1: Generate TACO trajectories ({n_taco} prompts x {K} tokens)")
        print(f"{'='*70}")
        taco_objs, taco_flat_gts, taco_flat_ds, taco_flat_ei = build_prompt_objects(
            tokenizer, prompts_text, forced_token_list, taco_indices,
            ground_truths, data_sources, extra_infos)
        print(f"  Generating {len(taco_objs)} TACO trajectories ...")
        taco_outputs = llm.generate(taco_objs, sp)
        taco_responses = [out.outputs[0].text for out in taco_outputs]
        print(f"  TACO generation done ({len(taco_responses)} responses)")

        # --- Phase 2: Start TACO scoring in background ---
        taco_scores_result = {"scores": None}

        def _taco_score_worker():
            taco_scores_result["scores"] = score_trajectories_parallel(
                taco_responses, taco_flat_ds, taco_flat_gts, taco_flat_ei,
                n_workers=taco_workers, label="TACO-bg")

        taco_score_thread = threading.Thread(target=_taco_score_worker, daemon=True)
        taco_score_thread.start()
        print(f"  TACO scoring started in background ({taco_workers} threads)")

        # --- Phase 3: Generate + score OTHER datasets in foreground ---
        print(f"\n{'='*70}")
        print(f"PHASE 3: Generate+score other datasets ({n_other} prompts)")
        print(f"{'='*70}")
        other_batch_size = args.batch_size if use_batch else n_other
        other_num_batches = (n_other + other_batch_size - 1) // other_batch_size
        for ob_idx in range(other_num_batches):
            o_start = ob_idx * other_batch_size
            o_end = min(o_start + other_batch_size, n_other)
            ob_indices = other_indices[o_start:o_end]
            ob_n = len(ob_indices)
            print(f"\n  [Other batch {ob_idx}/{other_num_batches-1}] "
                  f"{ob_n} prompts x {K} tokens = {ob_n*K} trajectories")
            ob_objs, ob_flat_gts, ob_flat_ds, ob_flat_ei = build_prompt_objects(
                tokenizer, prompts_text, forced_token_list, ob_indices,
                ground_truths, data_sources, extra_infos)
            ob_outputs = llm.generate(ob_objs, sp)
            ob_responses = [out.outputs[0].text for out in ob_outputs]
            # Non-TACO scoring is fast (string match), use modest parallelism
            ob_scores = score_trajectories_parallel(
                ob_responses, ob_flat_ds, ob_flat_gts, ob_flat_ei,
                n_workers=min(8, ob_n * K), label=f"Other-{ob_idx}")
            ob_rm = scores_to_reward_matrix(ob_scores, ob_n, K)
            n_correct_ob = (ob_rm.sum(dim=1) > 0).sum().item()
            print(f"  [Other batch {ob_idx}] >=1 correct: {n_correct_ob}/{ob_n} "
                  f"({n_correct_ob/ob_n*100:.1f}%)")
            # Save other batch
            if use_batch:
                ob_path = os.path.join(args.save_dir, f"batch_other_{ob_idx:04d}.pt")
                save_batch(ob_path,
                           [prompts_text[i] for i in ob_indices],
                           [ground_truths[i] for i in ob_indices],
                           [data_sources[i] for i in ob_indices],
                           forced_token_list, ob_rm)
            all_reward_matrices.append(ob_rm)

        # --- Phase 4: Collect TACO scores, save TACO batches ---
        print(f"\n{'='*70}")
        print(f"PHASE 4: Collect TACO scores & save")
        print(f"{'='*70}")
        if taco_score_thread.is_alive():
            print(f"  Waiting for TACO background scoring to finish ...")
        taco_score_thread.join()
        taco_scores = taco_scores_result["scores"]
        taco_rm = scores_to_reward_matrix(taco_scores, n_taco, K)
        n_correct_taco = (taco_rm.sum(dim=1) > 0).sum().item()
        print(f"  TACO >=1 correct: {n_correct_taco}/{n_taco} "
              f"({n_correct_taco/n_taco*100:.1f}%)")
        # Save TACO in batches
        taco_batch_size = args.batch_size if use_batch else n_taco
        taco_num_batches = (n_taco + taco_batch_size - 1) // taco_batch_size
        for tb_idx in range(taco_num_batches):
            t_start = tb_idx * taco_batch_size
            t_end = min(t_start + taco_batch_size, n_taco)
            tb_indices = taco_indices[t_start:t_end]
            tb_rm = taco_rm[t_start:t_end]
            if use_batch:
                tb_path = os.path.join(args.save_dir, f"batch_taco_{tb_idx:04d}.pt")
                save_batch(tb_path,
                           [prompts_text[i] for i in tb_indices],
                           [ground_truths[i] for i in tb_indices],
                           [data_sources[i] for i in tb_indices],
                           forced_token_list, tb_rm)
            else:
                save_batch(args.save_data,
                           [prompts_text[i] for i in tb_indices],
                           [ground_truths[i] for i in tb_indices],
                           [data_sources[i] for i in tb_indices],
                           forced_token_list, tb_rm)
        all_reward_matrices.append(taco_rm)

    else:
        # ================================================================
        # LEGACY SEQUENTIAL PIPELINE (or single-dataset mode)
        # ================================================================
        for batch_idx in range(args.batch_start, batch_end):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, total_prompts)
            batch_prompts = prompts_text[start:end]
            batch_gts = ground_truths[start:end]
            batch_ds = data_sources[start:end]
            batch_ei = extra_infos[start:end]

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
                batch_prompts, batch_gts, batch_ds, batch_ei, batch_idx)

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
              f"--save-dir {args.save_dir} --save-data router_data.pt")
    else:
        print(f"\n  Single file: {args.save_data}")


if __name__ == "__main__":
    main()
