#!/usr/bin/env python3
"""Preprocess AIME 2024/25/26 datasets to parquet format for verl.

Combines three AIME datasets from HuggingFace:
  - math-ai/aime24 (30 problems, solution in \\boxed{XXX} format)
  - math-ai/aime25 (30 problems, answer as plain integer string)
  - math-ai/aime26 (30 problems, answer as plain integer string)

AIME answers are integers 0-999. We extract the numeric answer and store it
as ground_truth. The data_source is set to "aime" so that
default_compute_score dispatches to math_dapo.compute_score (which handles
\\boxed{} extraction and numeric comparison).

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/aime.py --local_save_dir ~/data/aime
"""

import argparse
import os
import re

import datasets


def extract_answer(solution_str):
    """Extract numeric answer from solution string.

    aime24 solution: '\\boxed{204}' -> '204'
    aime25/26 answer: '204' (already plain)
    """
    # Try \\boxed{} first
    m = re.search(r"\\boxed\{(\d+)\}", str(solution_str))
    if m:
        return m.group(1)
    # Try plain integer
    s = str(solution_str).strip()
    if s.isdigit():
        return s
    # Fallback: extract last number
    nums = re.findall(r"\d+", str(solution_str))
    return nums[-1] if nums else str(solution_str)


def format_problem(problem):
    """Format AIME problem as a prompt."""
    return problem.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/aime")
    args = parser.parse_args()

    aime_sources = [
        ("math-ai/aime24", "solution"),  # (dataset_name, answer_field)
        ("math-ai/aime25", "answer"),
        ("math-ai/aime26", "answer"),
    ]

    all_rows = []
    for ds_name, answer_field in aime_sources:
        print(f"Loading {ds_name} ...")
        ds = datasets.load_dataset(ds_name)
        split = list(ds.keys())[0]  # usually 'test'
        ds = ds[split]
        print(f"  {len(ds)} rows, answer field: {answer_field}")
        for i, row in enumerate(ds):
            problem = row["problem"]
            raw_answer = row[answer_field]
            answer = extract_answer(raw_answer)
            formatted = format_problem(problem)
            all_rows.append({
                "data_source": "aime",
                "prompt": [{"role": "user", "content": formatted}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "split": ds_name,
                    "index": i,
                    "raw_answer": str(raw_answer),
                },
            })

    print(f"\nTotal AIME problems: {len(all_rows)}")

    # Save as parquet
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    out_path = os.path.join(local_save_dir, "test.parquet")
    ds_out = datasets.Dataset.from_list(all_rows)
    ds_out.to_parquet(out_path)
    print(f"Saved {len(all_rows)} rows -> {out_path}")
