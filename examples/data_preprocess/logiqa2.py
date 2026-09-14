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
Preprocess the LogiQA 2.0 (MRC) dataset to parquet format.

LogiQA 2.0 is an improved dataset for logical reasoning in question answering,
collected from the Chinese Civil Service Entrance Examination and translated
to English. The MRC task is multiple-choice reading comprehension: given a
passage, a question, and several options, pick the correct option.

Each record:
  - id      : int
  - answer  : int (0-based index into `options`)
  - text    : str (the passage / context)
  - question: str
  - options : list[str]
  - type    : dict of reasoning-type flags (optional)

Dataset: datatune/LogiQA2.0 (config: "mrc" -> train / dev / test)
Repo:    https://github.com/csitfun/LogiQA2.0

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/logiqa2.py --local_save_dir ~/data/logiqa2
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
        "--local_save_dir", default="~/data/logiqa2", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--config_name",
        default="mrc",
        help='Config to load from datatune/LogiQA2.0. Use "mrc" (default) or "nli".',
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "datatune/LogiQA2.0"
    config_name = args.config_name.lower()

    # NOTE: `datatune/LogiQA2.0` only exposes a `default` builder config and
    # stores its data as JSON-lines `.txt` files under `MRC/` and `NLI/`
    # folders (no parquet, no `mrc`/`nli` config). `datasets>=4.0` also rejects
    # dataset loading scripts, so we load the raw jsonl files directly with the
    # `json` builder, bypassing any repo-level script.
    if config_name not in ("mrc", "nli"):
        raise ValueError(f"config_name must be 'mrc' or 'nli', got {config_name!r}")

    if config_name == "nli":
        raise NotImplementedError(
            "NLI config has a different schema (label/major_premise/minor_premise/conclusion) "
            "and is not handled by this preprocessor yet. Use 'mrc'."
        )

    folder = "MRC"
    # Map our split names to the raw jsonl files in the repo.
    split_files = {
        "train": "train.txt",
        "dev": "dev.txt",      # will be saved as 'validation'
        "test": "test.txt",
    }

    def _load_split(split, filename):
        if local_dataset_path is not None:
            # Local copy: a directory containing the raw .txt files, or a single file.
            import os as _os
            if _os.path.isdir(local_dataset_path):
                candidate = _os.path.join(local_dataset_path, folder, filename)
                if not _os.path.exists(candidate):
                    candidate = _os.path.join(local_dataset_path, filename)
            else:
                candidate = local_dataset_path
            data_files = {split: candidate}
        else:
            base = f"https://huggingface.co/datasets/{data_source}/resolve/main/{folder}"
            data_files = {split: f"{base}/{filename}"}
        return datasets.load_dataset("json", data_files=data_files, split=split)

    raw_splits = {split: _load_split(split, fname) for split, fname in split_files.items()}

    # LogiQA2.0 uses dev / test (no validation split name); normalize split names.
    split_map = {"train": "train", "dev": "validation", "test": "test"}

    instruction = (
        "Read the passage and answer the multiple-choice question. "
        'Reply with only the letter of the correct option.'
    )

    letters = "ABCDEFGH"

    def make_map_fn(split):
        def process_fn(example, idx):
            passage = example.get("text", "")
            question_raw = example.get("question", "")
            options = example.get("options", [])
            answer_idx = example.get("answer", 0)

            # Build a formatted prompt: passage + question + lettered options.
            lines = []
            if passage:
                lines.append(passage.strip())
                lines.append("")
            lines.append(question_raw.strip())
            lines.append("")
            for i, opt in enumerate(options):
                lines.append(f"{letters[i]}. {opt}")
            lines.append("")
            lines.append(instruction)
            content = "\n".join(lines)

            # Normalize ground truth to a single letter.
            try:
                gt_idx = int(answer_idx)
                ground_truth = letters[gt_idx] if 0 <= gt_idx < len(letters) else str(answer_idx)
            except (ValueError, TypeError):
                ground_truth = str(answer_idx)

            data = {
                "data_source": f"{data_source}/{config_name}",
                "prompt": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "ability": "logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": example.get("id", ""),
                    "passage": passage,
                    "question": question_raw,
                    "options": options,
                    "answer_index": answer_idx,
                    "answer_letter": ground_truth,
                    "reasoning_types": example.get("type", {}),
                },
            }
            return data

        return process_fn

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    hdfs_dir = args.hdfs_dir

    for split in raw_splits.keys():
        ds = raw_splits[split].map(function=make_map_fn(split_map[split]), with_indices=True)
        out_name = split_map[split]
        out_path = os.path.join(local_save_dir, f"{out_name}.parquet")
        ds.to_parquet(out_path)
        print(f"LogiQA2.0 {out_name}: {len(ds)} rows -> {out_path}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
