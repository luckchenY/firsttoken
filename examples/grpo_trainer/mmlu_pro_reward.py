#!/usr/bin/env python3
"""Reward function for MMLU-Pro (multiple-choice A-J).

Extracts the answer letter from the model's response and compares with ground truth.
Supports Qwen3 thinking mode (reasoning chain + final answer).

Usage:
  from mmlu_pro_reward import compute_score_mmlu_pro
  score = compute_score_mmlu_pro(response, ground_truth="B")
"""

import re


def extract_answer_letter(text):
    """Extract the answer letter (A-J) from a model response.

    Tries multiple patterns:
    1. \\boxed{X}
    2. "The answer is X" / "answer is (X)"
    3. "X" at the end of the response (last standalone letter A-J)
    """
    # Pattern 1: \boxed{X}
    boxed = re.findall(r"\\boxed\{([A-J])\}", text)
    if boxed:
        return boxed[-1]

    # Pattern 2: "the answer is X" or "answer is (X)"
    answer_patterns = [
        r"(?:the\s+)?answer\s+is\s*\(?([A-J])\)?",
        r"(?:the\s+)?correct\s+answer\s+is\s*\(?([A-J])\)?",
        r"answer:\s*\(?([A-J])\)?",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Pattern 3: last standalone letter A-J in the response
    # Look for "X" at the end, possibly after whitespace/punctuation
    tail = text.strip()[-50:] if len(text) > 50 else text.strip()
    last_letter = re.findall(r"\b([A-J])\b", tail)
    if last_letter:
        return last_letter[-1]

    return None


def compute_score_mmlu_pro(solution_str, ground_truth):
    """Compute score for MMLU-Pro.

    Args:
        solution_str: Model's response text.
        ground_truth: Correct answer letter (e.g. "B").

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    if not ground_truth:
        return 0.0

    ground_truth = ground_truth.strip().upper()
    predicted = extract_answer_letter(solution_str)

    if predicted is None:
        return 0.0

    return 1.0 if predicted == ground_truth else 0.0


if __name__ == "__main__":
    # Quick test
    tests = [
        ("The answer is B.", "B", 1.0),
        ("The answer is (C).", "C", 1.0),
        ("\\boxed{D}", "D", 1.0),
        ("blah blah \\boxed{A} end", "A", 1.0),
        ("The answer is B.", "C", 0.0),
        ("some reasoning... the correct answer is J", "J", 1.0),
        ("answer: F", "F", 1.0),
        ("blah blah blah", "B", 0.0),  # no answer found
    ]
    for resp, gt, expected in tests:
        score = compute_score_mmlu_pro(resp, gt)
        status = "OK" if score == expected else "FAIL"
        print(f"  [{status}] resp={resp!r:40s} gt={gt} -> score={score} (expected {expected})")
