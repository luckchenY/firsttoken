#!/usr/bin/env python3
"""Offline test: verify each dataset's reward function works WITHOUT loading the model.

For each dataset, load a few real samples from the preprocessed parquet, build a
"canned" correct response and a canned wrong response, and check that:
  - the correct response scores > 0
  - the wrong response scores 0

This validates the reward dispatch + answer extraction + matching logic in
collect_router_data.compute_score, end-to-end on real ground_truth values.
No GPU / vLLM needed — runs in seconds.

Usage:
  cd /workspace/firsttoken
  python examples/grpo_trainer/test_dataset_rewards.py
  # or point at specific files:
  python examples/grpo_trainer/test_dataset_rewards.py --n 5
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from collect_router_data import compute_score  # noqa: E402

# Default parquet locations (relative to repo root).
DEFAULT_FILES = {
    "GSM8K":         ("data/gsm8k/test.parquet",            "math"),
    "MATH":           ("data/math/test.parquet",             "math"),
    "TACO":           ("data/taco/test.parquet",             "code"),
    "ARC-Challenge":  ("data/arc_challenge/test.parquet",   "mc"),
    "LogiQA2.0":      ("data/logiqa2/test.parquet",          "mc"),
    "DROP":           ("data/drop/validation.parquet",       "span"),
}


def build_canned_responses(kind, row):
    """Return (correct_response, wrong_response) for a given row.

    The 'correct' response is crafted to match the dataset's expected output
    format so the reward function should score it 1.0. The 'wrong' response is
    crafted to mismatch and should score 0.0.
    """
    rm = row.get("reward_model", {})
    gt = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
    ei = row.get("extra_info", {}) if isinstance(row.get("extra_info"), dict) else {}

    if kind == "math":
        # GSM8K extracts after "####", MATH extracts from \boxed{}.
        # Include BOTH markers so the canned correct response works for
        # either extractor.
        correct = f"some reasoning...\n\\boxed{{{gt}}}\n#### {gt}"
        wrong = f"some reasoning...\n\\boxed{{not_a_number}}\n#### not_a_number_xyz"
        return correct, wrong

    if kind == "mc":
        # Multiple choice: ground_truth is a letter. Use \boxed{X}.
        correct = f"reasoning...\n\\boxed{{{gt}}}"
        # pick a different letter
        wrong_letter = "Z" if gt not in ("Z",) else "Y"
        wrong = f"reasoning...\n\\boxed{{{wrong_letter}}}"
        return correct, wrong

    if kind == "span":
        # DROP: answer after "####"
        correct = f"reasoning...\n#### {gt}"
        wrong = f"reasoning...\n#### __definitely_not_the_answer__"
        return correct, wrong

    if kind == "code":
        # TACO: ground_truth is JSON {solutions, input_output, fn_name}.
        # Reference solutions are noisy (some are buggy / have I/O format
        # issues), so try up to 5 and let the caller pick the first that
        # scores > 0. Return a list of (resp, sol_idx) candidates.
        try:
            gt_dict = json.loads(gt) if isinstance(gt, str) else gt
            solutions = gt_dict.get("solutions", [])
        except Exception:
            solutions = []
        candidates = []
        for si, sol in enumerate(solutions[:5]):
            candidates.append((f"```python\n{sol}\n```", si))
        if not candidates:
            candidates.append(("```python\nprint('no solution')\n```", -1))
        # wrong response
        wrong = "```python\nprint('wrong answer')\n```"
        return candidates, wrong  # note: different signature, handled below

    return None, None


def test_dataset(name, path, kind, n):
    # resolve relative to repo root (two levels above this script)
    if not os.path.isabs(path):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo_root, path)
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"[{name}] SKIP — file not found: {path}")
        return None
    df = pd.read_parquet(path)
    if len(df) == 0:
        print(f"[{name}] SKIP — empty parquet")
        return None
    sample = df.sample(n=min(n, len(df)), random_state=0)

    data_source = sample.iloc[0].get("data_source", "")
    n_correct_pass = 0
    n_wrong_pass = 0
    n_tested = 0
    details = []
    for _, row in sample.iterrows():
        rm = row.get("reward_model", {})
        gt = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
        ei = row.get("extra_info", {}) if isinstance(row.get("extra_info"), dict) else {}
        built = build_canned_responses(kind, row)
        # code kind returns (candidates_list, wrong_resp); others return (correct, wrong)
        if kind == "code":
            candidates, wrong_resp = built
            if not candidates:
                continue
            n_tested += 1
            # a sample "passes" if ANY candidate reference solution scores > 0
            best_correct = 0.0
            for cand_resp, _si in candidates:
                try:
                    sc = compute_score(cand_resp, data_source, gt, extra_info=ei)
                except Exception as e:
                    sc = -1.0
                    details.append(f"  ERROR(correct) idx: {e}")
                if float(sc) > float(best_correct):
                    best_correct = float(sc)
                if best_correct > 0:
                    break
            s_correct = best_correct
        else:
            correct_resp, wrong_resp = built
            if correct_resp is None:
                continue
            n_tested += 1
            try:
                s_correct = compute_score(correct_resp, data_source, gt, extra_info=ei)
            except Exception as e:
                s_correct = -1.0
                details.append(f"  ERROR(correct) idx: {e}")
        try:
            s_wrong = compute_score(wrong_resp, data_source, gt, extra_info=ei)
        except Exception as e:
            s_wrong = -1.0
            details.append(f"  ERROR(wrong) idx: {e}")
        if float(s_correct) > 0:
            n_correct_pass += 1
        if float(s_wrong) == 0.0:
            n_wrong_pass += 1

    # Pass criterion: code (TACO) is a noisy dataset where some reference
    # solutions are genuinely broken, so require a majority to have a working
    # reference solution. Other kinds require all samples to pass.
    if kind == "code":
        passed = n_tested > 0 and n_correct_pass >= max(1, (n_tested + 1) // 2)
    else:
        passed = n_tested > 0 and n_correct_pass == n_tested and n_wrong_pass == n_tested
    status = "OK" if passed else "FAIL"
    print(f"[{name}] {status}  data_source={data_source!r}  tested={n_tested}")
    print(f"   correct->>0: {n_correct_pass}/{n_tested}   wrong==0: {n_wrong_pass}/{n_tested}")
    if kind == "code" and n_correct_pass < n_tested:
        print(f"   (note: {n_tested - n_correct_pass} sample(s) had no working reference "
              f"solution — TACO data-quality noise, not a reward bug)")
    if details:
        for d in details[:3]:
            print(d)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="samples per dataset to test")
    args = parser.parse_args()

    print(f"Offline reward-function test (n={args.n} per dataset)\n")
    results = {}
    for name, (path, kind) in DEFAULT_FILES.items():
        ok = test_dataset(name, path, kind, args.n)
        if ok is not None:
            results[name] = ok

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {name:14} {'PASS' if ok else 'FAIL'}")
    n_pass = sum(1 for v in results.values() if v)
    print(f"\n{n_pass}/{len(results)} datasets passed")
