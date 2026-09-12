#!/usr/bin/env python3
"""Preprocess GPQA Diamond dataset to parquet format for verl.

GPQA Diamond: 198 graduate-level science questions, 4 options (A-D).
Dataset: Idavidrein/gpqa (config: gpqa_diamond)

The raw dataset has columns:
  - Question
  - Correct Answer
  - Incorrect Answer 1/2/3

We randomly shuffle the 4 answers into positions A-D, store the correct letter as ground_truth.

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/gpqa_diamond.py --local_save_dir ~/data/gpqa_diamond
"""

import argparse
import os
import random

import datasets


def format_question(question, options):
    """Format GPQA question with options A-D."""
    letters = "ABCD"
    lines = [question.strip(), ""]
    for i, opt in enumerate(options):
        lines.append(f"{letters[i]}. {opt}")
    lines.append("")
    lines.append("Answer with the letter of the correct option.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--local_save_dir", default="~/data/gpqa_diamond")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling options")
    args = parser.parse_args()

    data_source = "Idavidrein/gpqa"
    config_name = "gpqa_diamond"
    print(f"Loading {data_source} ({config_name}) from HuggingFace ...")

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, config_name)
    else:
        dataset = datasets.load_dataset(data_source, config_name)

    splits = list(dataset.keys())
    print(f"Available splits: {splits}")

    rng = random.Random(args.seed)
    letters = "ABCD"

    def make_map_fn(split):
        def process_fn(doc, idx):
            question = doc["Question"]
            correct = doc["Correct Answer"]
            wrong1 = doc["Incorrect Answer 1"]
            wrong2 = doc["Incorrect Answer 2"]
            wrong3 = doc["Incorrect Answer 3"]

            # Shuffle answers into positions A-D
            all_answers = [correct, wrong1, wrong2, wrong3]
            perm = list(range(4))
            rng.shuffle(perm)
            options = [all_answers[p] for p in perm]
            correct_pos = perm.index(0)  # where the correct answer ended up
            answer_letter = letters[correct_pos]

            formatted = format_question(question, options)

            data = {
                "data_source": data_source + "/" + config_name,
                "prompt": [{"role": "user", "content": formatted}],
                "ability": "knowledge",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_letter,  # "A", "B", "C", or "D"
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "options": options,
                    "correct_answer_text": correct,
                    "answer_letter": answer_letter,
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
        print(f"GPQA Diamond {split}: {len(ds)} rows -> {out_path}")
