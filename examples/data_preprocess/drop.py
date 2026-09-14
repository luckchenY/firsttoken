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
Preprocess the DROP dataset to parquet format.

DROP (Discrete Reasoning Over Paragraphs) is a crowdsourced, adversarially
created QA benchmark of ~96k questions. A system must resolve references in a
question (perhaps to multiple input positions) and perform discrete
operations over them (addition, counting, sorting, ...). Answers are short
text spans extracted from the passage.

Splits: train (77,409) / validation (9,536). There is no public test split.
Dataset: ucinlp/drop

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/drop.py --local_save_dir ~/data/drop
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/drop", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "ucinlp/drop"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    instruction = (
        'Read the passage and answer the question. '
        'Output the final answer after "####".'
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            passage = example.pop("passage")
            question_raw = example.pop("question")
            answers_spans = example.pop("answers_spans")

            # answers_spans is a dict: {"spans": [...], "types": [...]}
            spans = list(answers_spans.get("spans", [])) if isinstance(answers_spans, dict) else []
            types = list(answers_spans.get("types", [])) if isinstance(answers_spans, dict) else []

            # The first span is the canonical reference answer; the rest are
            # alternative acceptable answers.
            ground_truth = spans[0] if spans else ""

            content = (
                f"Passage: {passage.strip()}\n\n"
                f"Question: {question_raw.strip()}\n\n"
                f"{instruction}"
            )

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                    "accepted_answers": spans,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "section_id": example.get("section_id", ""),
                    "query_id": example.get("query_id", ""),
                    "passage": passage,
                    "question": question_raw,
                    "answer_spans": spans,
                    "answer_types": types,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)

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
    print(f"DROP train:      {len(train_dataset)} rows -> {os.path.join(local_save_dir, 'train.parquet')}")
    print(f"DROP validation: {len(val_dataset)} rows -> {os.path.join(local_save_dir, 'validation.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
