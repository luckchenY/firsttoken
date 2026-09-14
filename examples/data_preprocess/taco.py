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
Preprocess the TACO (Topics in Algorithmic COde generation) dataset to parquet format.

TACO is a benchmark for Python code generation with 25,443 train and 1,000 test
competition-style problems, annotated with topics, algorithms, skills, and
difficulty levels. Each sample carries the problem statement, ground-truth
Python solutions, and test cases (inputs/outputs) used to verify generated code.

Dataset: BAAI/TACO
Repo:    https://github.com/FlagOpen/TACO

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/taco.py --local_save_dir ~/data/taco
  # optionally filter by difficulty / skill:
  python examples/data_preprocess/taco.py --difficulties EASY MEDIUM \
      --skills "Data structures" "Sorting"
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def safe_parse_list(raw):
    """Parse a string-encoded Python list (e.g. "['a', 'b']") into a real list.

    TACO stores `raw_tags`, `tags`, `skill_types` as Python-repr strings.
    Empty strings parse to an empty list.
    """
    if raw is None or raw == "":
        return []
    try:
        return eval(raw, {"__builtins__": {}}, {})
    except Exception:
        return []


def safe_parse_json(raw):
    """Parse a JSON-encoded string field (`solutions`, `input_output`)."""
    if raw is None or raw == "":
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/taco", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=None,
        help='Filter by difficulty: EASY, MEDIUM, MEDIUM_HARD, HARD, VERY_HARD. None = all.',
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        default=None,
        help='Filter by skill: e.g. "Data structures", "Sorting", "Range queries", '
        '"Complete search", "Amortized analysis", "Dynamic programming", '
        '"Bit manipulation", "Greedy algorithms". None = all.',
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "BAAI/TACO"

    # NOTE: `datasets>=4.0` no longer supports dataset loading scripts, and the
    # BAAI/TACO repo still ships a `TACO.py` script, so `load_dataset("BAAI/TACO")`
    # raises `RuntimeError: Dataset scripts are no longer supported`. The repo
    # has however been converted to Parquet under the `ALL/` config, so we load
    # the parquet files directly (bypassing the script) and apply the
    # difficulty / skill filters ourselves after loading.
    difficulties_filter = args.difficulties  # e.g. ["EASY", "MEDIUM"]
    skills_filter = args.skills            # e.g. ["Data structures", "Sorting"]

    def _load_split(split):
        if local_dataset_path is not None:
            # Local copy: point at a directory or a glob of parquet files.
            import glob
            if os.path.isdir(local_dataset_path):
                pattern = os.path.join(local_dataset_path, "**", f"{split}-*.parquet")
                files = sorted(glob.glob(pattern, recursive=True))
                if not files:
                    # fall back to <split>.parquet
                    single = os.path.join(local_dataset_path, f"{split}.parquet")
                    files = [single] if os.path.exists(single) else []
            else:
                files = [local_dataset_path]
            return datasets.load_dataset("parquet", data_files={split: files}, split=split)

        # Remote: load parquet files straight from the HF hub (ALL config).
        base = f"https://huggingface.co/datasets/{data_source}/resolve/main/ALL"
        if split == "train":
            data_files = [f"{base}/train-{i:05d}-of-00009.parquet" for i in range(9)]
        else:  # test
            data_files = [f"{base}/test-00000-of-00001.parquet"]
        return datasets.load_dataset("parquet", data_files={split: data_files}, split=split)

    train_dataset = _load_split("train")
    test_dataset = _load_split("test")

    # Apply difficulty / skill filters (previously done by the loading script).
    if difficulties_filter is not None:
        diff_set = set(d.upper() for d in difficulties_filter)
        train_dataset = train_dataset.filter(lambda x: x.get("difficulty", "").upper() in diff_set)
        test_dataset = test_dataset.filter(lambda x: x.get("difficulty", "").upper() in diff_set)
    if skills_filter is not None:
        skill_set = set(skills_filter)
        def _has_skill(example):
            return any(s in skill_set for s in safe_parse_list(example.get("skill_types", "")))
        train_dataset = train_dataset.filter(_has_skill)
        test_dataset = test_dataset.filter(_has_skill)

    instruction_following = (
        'Please reason step by step, and write a complete Python solution. '
        'Enclose your final code within a single ```python ... ``` block.'
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            starter_code = example.pop("starter_code") or ""
            solutions_raw = example.pop("solutions")
            input_output_raw = example.pop("input_output")

            solutions = safe_parse_json(solutions_raw)  # list[str] or {}
            input_output = safe_parse_json(input_output_raw)  # {"inputs": [...], "outputs": [...], "fn_name"?}

            # Build the prompt: question + optional starter code + instruction.
            question = question_raw.strip()
            if starter_code:
                question = question + "\n\n" + starter_code.strip()
            question = question + "\n\n" + instruction_following

            # Ground truth used by the rule-based code reward model: the test cases.
            # Reward functions can re-parse `input_output` to execute the
            # generated code against inputs/outputs.
            ground_truth = {
                "solutions": solutions if isinstance(solutions, list) else [],
                "input_output": input_output,
                "fn_name": input_output.get("fn_name", "") if isinstance(input_output, dict) else "",
            }

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "difficulty": example.get("difficulty", ""),
                    "raw_tags": safe_parse_list(example.get("raw_tags", "")),
                    "tags": safe_parse_list(example.get("tags", "")),
                    "skill_types": safe_parse_list(example.get("skill_types", "")),
                    "source": example.get("source", ""),
                    "url": example.get("url", ""),
                    "name": example.get("name", ""),
                    "time_limit": example.get("time_limit", ""),
                    "memory_limit": example.get("memory_limit", ""),
                    "expected_time_complexity": example.get("Expected Time Complexity", ""),
                    "expected_auxiliary_space": example.get("Expected Auxiliary Space", ""),
                    "date": example.get("date", ""),
                    "picture_num": example.get("picture_num", ""),
                    "question": question_raw,
                    "starter_code": starter_code,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
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
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"TACO train: {len(train_dataset)} rows -> {os.path.join(local_save_dir, 'train.parquet')}")
    print(f"TACO test:  {len(test_dataset)} rows -> {os.path.join(local_save_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
