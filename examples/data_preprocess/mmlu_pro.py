#!/usr/bin/env python3
"""Preprocess MMLU-Pro dataset to parquet format for verl.

MMLU-Pro is a multiple-choice benchmark with 10 options (A-J).
Dataset: TIGER-Lab/MMLU-Pro on HuggingFace.

Usage:
  python examples/data_preprocess/mmlu_pro.py --local_save_dir ~/data/mmlu_pro
"""

import argparse
import os

import datasets


def format_question(question, options):
    """Format MMLU-Pro question with options into a prompt string."""
    letters = "ABCDEFGHIJ"
    lines = [question.strip(), ""]
    for i, opt in enumerate(options):
        lines.append(f"{letters[i]}. {opt}")
    lines.append("")
    lines.append("Answer with the letter of the correct option.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/mmlu_pro")
    args = parser.parse_args()

    data_source = "TIGER-Lab/MMLU-Pro"
    print(f"Loading {data_source} from HuggingFace ...")

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # MMLU-Pro only has test split
    splits = list(dataset.keys())
    print(f"Available splits: {splits}")

    def make_map_fn(split):
        def process_fn(doc, idx):
            question = doc["question"]
            options = doc["options"]
            # answer_index in MMLU-Pro is an integer (0-9), convert to letter (A-J)
            letters = "ABCDEFGHIJ"
            raw_answer = doc["answer_index"]
            if isinstance(raw_answer, int):
                answer_index = letters[raw_answer]
            else:
                answer_index = str(raw_answer).strip().upper()
                # If it's a digit string, convert to letter
                if answer_index.isdigit():
                    answer_index = letters[int(answer_index)]

            formatted = format_question(question, options)

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": formatted}],
                "ability": "knowledge",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_index,  # letter like "B"
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "category": doc.get("category", ""),
                    "options": options,
                    "answer_index": answer_index,
                },
            }
            return data

        return process_fn

    local_save_dir = args.local_dir or args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    for split in splits:
        ds = dataset[split]
        ds = ds.map(function=make_map_fn(split), with_indices=True)
        out_path = os.path.join(local_save_dir, f"{split}.parquet")
        ds.to_parquet(out_path)
        print(f"MMLU-Pro {split}: {len(ds)} rows -> {out_path}")

    if args.hdfs_dir is not None:
        print(f"HDFS upload not supported in standalone mode. Skipping: {args.hdfs_dir}")
