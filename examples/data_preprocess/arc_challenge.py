# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the ARC-Challenge dataset to parquet format.

ARC (AI2 Reasoning Challenge) is a dataset of genuine grade-school level
multiple-choice science questions. The "Challenge" subset contains only
questions answered incorrectly by both a retrieval-based algorithm and a
word co-occurrence algorithm.

Splits: train (1,119) / validation (299) / test (1,172).
Dataset: allenai/ai2_arc (config: ARC-Challenge)

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/arc_challenge.py --local_save_dir ~/data/arc_challenge
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/arc_challenge", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "allenai/ai2_arc"
    config_name = "ARC-Challenge"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, config_name)
    else:
        dataset = datasets.load_dataset(data_source, config_name)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    instruction = (
        "Answer the following multiple-choice question. "
        'Reply with only the letter of the correct option.'
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            choices = example.pop("choices")
            answer_key = example.pop("answerKey")

            labels = choices["label"]  # e.g. ["A", "B", "C", "D"] (sometimes 1/2/3/4)
            texts = choices["text"]

            # Build a formatted prompt with lettered options.
            letters = "ABCDEFGH"
            # Use the dataset's own labels if they look like letters, otherwise fall
            # back to A, B, C, ... so the ground truth is always a single letter.
            use_native_labels = all(isinstance(lab, str) and len(lab) == 1 and lab.isalpha() for lab in labels)
            option_letters = labels if use_native_labels else [letters[i] for i in range(len(texts))]

            lines = [question_raw.strip(), ""]
            for letter, text in zip(option_letters, texts):
                lines.append(f"{letter}. {text}")
            lines.append("")
            lines.append(instruction)
            content = "\n".join(lines)

            # Normalize the ground-truth answer to a single letter.
            if use_native_labels:
                ground_truth = answer_key
            else:
                # answer_key may be an index (as string) or a label
                try:
                    gt_idx = int(answer_key)
                    ground_truth = letters[gt_idx]
                except (ValueError, TypeError):
                    ground_truth = answer_key

            data = {
                "data_source": f"{data_source}/{config_name}",
                "prompt": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "ability": "knowledge",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": example.get("id", ""),
                    "question": question_raw,
                    "options": texts,
                    "option_labels": option_letters,
                    "answer": answer_key,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"ARC-Challenge train:      {len(train_dataset)} rows -> {os.path.join(local_save_dir, 'train.parquet')}")
    print(f"ARC-Challenge validation: {len(val_dataset)} rows -> {os.path.join(local_save_dir, 'validation.parquet')}")
    print(f"ARC-Challenge test:       {len(test_dataset)} rows -> {os.path.join(local_save_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
