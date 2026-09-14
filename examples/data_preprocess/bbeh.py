#!/usr/bin/env python3
"""Preprocess BBEH (BIG-Bench Extra Hard) dataset to parquet format for verl.

BBEH has 23 reasoning tasks with 4,520 total examples.
Dataset: BBEH/bbeh on HuggingFace.
Columns: task (string), input (string), target (string), canary (string), mini (int64)

The target is a free-form text answer (e.g., "proved", "disproved", numbers, etc.).
We use exact-match (normalized) as the reward.

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/bbeh.py --local_save_dir ~/data/bbeh
  # Or only the mini subset (460 examples, 20 per task):
  python examples/data_preprocess/bbeh.py --local_save_dir ~/data/bbeh --mini-only
"""

import argparse
import os
import re

import datasets


def normalize_answer(text):
    """Normalize a text answer for exact-match comparison."""
    s = str(text).strip().lower()
    # Remove surrounding quotes
    s = s.strip("\"'`")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def format_question(task, input_text):
    """Format a BBEH instance as a prompt.

    We prepend the task name as context and ask for the answer.
    """
    lines = [
        f"Task: {task}",
        "",
        input_text.strip(),
        "",
        "Provide the answer concisely.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/bbeh")
    parser.add_argument("--mini-only", action="store_true",
                        help="Only use the mini subset (20 per task, 460 total)")
    args = parser.parse_args()

    data_source = "BBEH/bbeh"
    print(f"Loading {data_source} from HuggingFace ...")
    dataset = datasets.load_dataset(data_source)

    splits = list(dataset.keys())
    print(f"Available splits: {splits}")

    all_rows = []
    for split in splits:
        ds = dataset[split]
        for i, row in enumerate(ds):
            if args.mini_only and not row.get("mini", 0):
                continue
            task = row["task"]
            input_text = row["input"]
            target = row["target"]

            formatted = format_question(task, input_text)

            all_rows.append({
                "data_source": data_source,
                "prompt": [{"role": "user", "content": formatted}],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(target),
                },
                "extra_info": {
                    "split": split,
                    "index": i,
                    "task": task,
                    "target": str(target),
                    "mini": bool(row.get("mini", 0)),
                },
            })

    print(f"\nTotal BBEH problems: {len(all_rows)}")
    if args.mini_only:
        print("  (mini subset only)")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    suffix = "_mini" if args.mini_only else ""
    out_path = os.path.join(local_save_dir, f"test{suffix}.parquet")
    ds_out = datasets.Dataset.from_list(all_rows)
    ds_out.to_parquet(out_path)
    print(f"Saved {len(all_rows)} rows -> {out_path}")
