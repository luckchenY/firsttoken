#!/usr/bin/env python3
"""Preprocess MuSR dataset to parquet format for verl.

MuSR (Multistep Soft Reasoning) has 3 domains:
  - murder_mysteries (250): who is the murderer? (2 choices)
  - object_placements (256): where is the object? (2 choices)
  - team_allocation (250): who is on which team? (2 choices)

Dataset: TAUR-Lab/MuSR on HuggingFace.
Each instance has: narrative, question, choices (string), answer_index (0/1),
answer_choice.

We format as 2-option multiple choice (A/B) and store the correct letter
as ground_truth. data_source = "TAUR-Lab/MuSR" (dispatched to compute_score_mmlu_pro).

Usage:
  export HF_ENDPOINT=https://hf-mirror.com
  python examples/data_preprocess/musr.py --local_save_dir ~/data/musr
"""

import argparse
import ast
import os

import datasets


def parse_choices(choices_field):
    """Parse the choices field into a list of strings.

    MuSR choices are stored as a Python list repr string, e.g.:
      "['Mackenzie', 'Ana']"
      "['piano', \"producer's desk\", 'recording booth']"
    Use ast.literal_eval for robust parsing (handles commas inside strings).
    """
    if isinstance(choices_field, list):
        return [str(x) for x in choices_field]
    s = str(choices_field).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    # Fallback: simple comma split (may break on commas inside choices)
    return [p.strip() for p in s.split(",") if p.strip()]


def format_question(narrative, question, choices):
    """Format MuSR instance as a multiple-choice prompt (2-4 options)."""
    letters = "ABCDEFGHIJ"
    lines = [narrative.strip(), "", question.strip(), ""]
    for i, opt in enumerate(choices):
        lines.append(f"{letters[i]}. {opt}")
    lines.append("")
    lines.append("Answer with the letter of the correct option.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/musr")
    args = parser.parse_args()

    data_source = "TAUR-Lab/MuSR"
    print(f"Loading {data_source} from HuggingFace ...")
    dataset = datasets.load_dataset(data_source)

    splits = list(dataset.keys())
    print(f"Available splits: {splits}")

    all_rows = []
    for split in splits:
        ds = dataset[split]
        print(f"  {split}: {len(ds)} rows")
        for i, row in enumerate(ds):
            narrative = row["narrative"]
            question = row["question"]
            choices = parse_choices(row["choices"])
            answer_index = int(row["answer_index"])

            letters = "ABCDEFGHIJ"
            if answer_index < 0 or answer_index >= len(choices):
                print(f"  WARNING: bad answer_index={answer_index} in {split}[{i}]")
                continue
            answer_letter = letters[answer_index]

            formatted = format_question(narrative, question, choices)

            all_rows.append({
                "data_source": data_source,
                "prompt": [{"role": "user", "content": formatted}],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_letter,
                },
                "extra_info": {
                    "split": split,
                    "index": i,
                    "choices": choices,
                    "answer_index": answer_index,
                    "answer_choice": str(row.get("answer_choice", "")),
                },
            })

    print(f"\nTotal MuSR problems: {len(all_rows)}")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    out_path = os.path.join(local_save_dir, "test.parquet")
    ds_out = datasets.Dataset.from_list(all_rows)
    ds_out.to_parquet(out_path)
    print(f"Saved {len(all_rows)} rows -> {out_path}")
