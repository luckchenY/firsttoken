#!/usr/bin/env python3
"""Streaming statistics for Okay-like openings in the M2RL SFT sources.

The script intentionally does not download the full datasets.  It takes a
shuffled sample from every published split, reports source-level rates, and
also computes a mixture-weighted estimate using the target counts in the
M2RL data-processing script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


SOURCES = {
    "math_proofs": {
        "dataset": "nvidia/Nemotron-Math-Proofs-v1",
        "splits": ["lean"],
        "target_rows": 335_122,
    },
    "math": {
        "dataset": "nvidia/Nemotron-Math-v2",
        "splits": ["high_part00", "high_part01", "high_part02", "medium", "low"],
        "target_rows": 2_950_525,
    },
    "science": {
        "dataset": "nvidia/Nemotron-Science-v1",
        "splits": ["MCQ", "RQA"],
        "target_rows": 2_263_340,
    },
    "code": {
        "dataset": "nvidia/Nemotron-Competitive-Programming-v1",
        "splits": [
            "competitive_coding_cpp_part00",
            "competitive_coding_cpp_part01",
            "competitive_coding_python_part00",
            "competitive_coding_python_part01",
            "infinibyte_part00",
            "infinibyte_part01",
        ],
        "target_rows": 3_927_984,
    },
    "chat": {
        "dataset": "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "splits": ["chat_if", "structured_outputs"],
        "target_rows": 4_309_780,
    },
    "agent": {
        "dataset": "nvidia/Nemotron-Agentic-v1",
        "splits": ["interactive_agent", "tool_calling"],
        "target_rows": 335_122,
    },
}


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
OKAY_RE = re.compile(r"(?<![A-Za-z])okay(?![A-Za-z])", re.IGNORECASE)
TAG_RE = re.compile(r"</?(?:think|answer)>", re.IGNORECASE)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def first_word(text: str) -> str:
    match = WORD_RE.search(text)
    return match.group(0) if match else ""


def word_position(text: str, char_position: int) -> int:
    """Return a one-based approximate word position for an Okay occurrence."""
    return len(WORD_RE.findall(text[:char_position])) + 1


def okay_bucket(position: int) -> str:
    if position <= 1:
        return "first_word"
    if position <= 5:
        return "word_2_to_5"
    if position <= 20:
        return "word_6_to_20"
    return "word_21_plus"


def message_fields(message: dict[str, Any]) -> tuple[str, str]:
    reasoning = as_text(message.get("reasoning_content", message.get("reasoning", "")))
    content = as_text(message.get("content", ""))
    return reasoning, content


def new_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "assistant_messages": 0,
        "assistant_messages_with_reasoning": 0,
        "assistant_messages_with_content": 0,
        "assistant_messages_with_any_text": 0,
        "reasoning_okay_messages": 0,
        "content_okay_messages": 0,
        "all_text_okay_messages": 0,
        "reasoning_okay_occurrences": 0,
        "content_okay_occurrences": 0,
        "all_text_okay_occurrences": 0,
        "reasoning_first_word_okay": 0,
        "content_first_word_okay": 0,
        "all_text_first_word_okay": 0,
        "okay_after_newline_or_sentence": 0,
        "think_tag_occurrences": 0,
        "answer_tag_occurrences": 0,
        "first_reasoning_words": Counter(),
        "first_content_words": Counter(),
        "okay_position_buckets": Counter(),
        "okay_case_variants": Counter(),
        "reasoning_characters": 0,
        "content_characters": 0,
    }


def update_text_stats(stats: dict[str, Any], text: str, field: str, examples: list[dict[str, Any]],
                      source: str, split: str, row_index: int, message_index: int) -> None:
    if not text.strip():
        return

    if field == "reasoning":
        stats["assistant_messages_with_reasoning"] += 1
        stats["reasoning_characters"] += len(text)
        stats["first_reasoning_words"][first_word(text)] += 1
    else:
        stats["assistant_messages_with_content"] += 1
        stats["content_characters"] += len(text)
        stats["first_content_words"][first_word(text)] += 1

    matches = list(OKAY_RE.finditer(text))
    if not matches:
        return

    occurrence_key = f"{field}_okay_occurrences"
    message_key = f"{field}_okay_messages"
    stats[occurrence_key] += len(matches)
    stats[message_key] += 1
    if first_word(text).lower() == "okay":
        stats[f"{field}_first_word_okay"] += 1

    for match in matches:
        position = word_position(text, match.start())
        stats["okay_position_buckets"][okay_bucket(position)] += 1
        left = text[max(0, match.start() - 1):match.start()]
        if match.start() == 0 or left in {"\n", ".", ":", "!", "?"}:
            stats["okay_after_newline_or_sentence"] += 1
        surface = text[match.start():match.end()]
        stats["okay_case_variants"][surface] += 1

    if len(examples) < 30:
        first = matches[0]
        lo = max(0, first.start() - 180)
        hi = min(len(text), first.end() + 260)
        examples.append({
            "source": source,
            "split": split,
            "row_index": row_index,
            "message_index": message_index,
            "field": field,
            "first_word": first_word(text),
            "word_position": word_position(text, first.start()),
            "snippet": text[lo:hi].replace("\n", "\\n"),
        })


def merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, Counter):
            dst[key].update(value)
        elif isinstance(value, (int, float)):
            dst[key] += value


def serialise_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            out[key] = dict(value.most_common())
        else:
            out[key] = value
    return out


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def source_rate_row(source: str, stats: dict[str, Any], target_rows: int) -> dict[str, Any]:
    n = stats["assistant_messages"]
    return {
        "source": source,
        "target_rows": target_rows,
        "sampled_rows": stats["rows"],
        "assistant_messages": n,
        "reasoning_nonempty_pct": pct(stats["assistant_messages_with_reasoning"], n),
        "content_nonempty_pct": pct(stats["assistant_messages_with_content"], n),
        "reasoning_okay_message_pct": pct(stats["reasoning_okay_messages"], n),
        "content_okay_message_pct": pct(stats["content_okay_messages"], n),
        "all_text_okay_message_pct": pct(stats["all_text_okay_messages"], n),
        "reasoning_okay_occurrences_per_1000_assistant": 1000.0 * stats["reasoning_okay_occurrences"] / n if n else 0.0,
        "content_okay_occurrences_per_1000_assistant": 1000.0 * stats["content_okay_occurrences"] / n if n else 0.0,
        "reasoning_first_word_okay_pct": pct(stats["reasoning_first_word_okay"], stats["assistant_messages_with_reasoning"]),
        "content_first_word_okay_pct": pct(stats["content_first_word_okay"], stats["assistant_messages_with_content"]),
        "all_text_first_word_okay_pct": pct(stats["all_text_first_word_okay"], n),
        "okay_after_newline_or_sentence": stats["okay_after_newline_or_sentence"],
        "think_tags": stats["think_tag_occurrences"],
        "answer_tags": stats["answer_tag_occurrences"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-per-split", type=int, default=1500)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_stats: dict[str, dict[str, Any]] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    started = time.time()

    print(f"Output: {output}", flush=True)
    print(f"Rows per split: {args.rows_per_split}", flush=True)

    for source, spec in SOURCES.items():
        source_acc = new_stats()
        print(f"\n[{source}] {spec['dataset']}", flush=True)
        for split in spec["splits"]:
            split_acc = new_stats()
            print(f"  loading {split} ...", flush=True)
            try:
                dataset = load_dataset(spec["dataset"], split=split, streaming=True)
                dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
                for row_index, row in enumerate(dataset):
                    if row_index >= args.rows_per_split:
                        break
                    split_acc["rows"] += 1
                    messages = row.get("messages", []) if isinstance(row, dict) else []
                    if not isinstance(messages, list):
                        continue
                    for message_index, message in enumerate(messages):
                        if not isinstance(message, dict) or message.get("role") != "assistant":
                            continue
                        split_acc["assistant_messages"] += 1
                        reasoning, content = message_fields(message)
                        if reasoning.strip() or content.strip():
                            split_acc["assistant_messages_with_any_text"] += 1
                        for field, text in (("reasoning", reasoning), ("content", content)):
                            update_text_stats(split_acc, text, field, examples, source, split, row_index, message_index)
                            if text:
                                split_acc["think_tag_occurrences"] += len(re.findall(r"<think>", text, flags=re.IGNORECASE))
                                split_acc["answer_tag_occurrences"] += len(re.findall(r"</?answer>", text, flags=re.IGNORECASE))
                        combined = "\n".join(part for part in (reasoning, content) if part.strip())
                        if OKAY_RE.search(combined):
                            split_acc["all_text_okay_messages"] += 1
                            split_acc["all_text_okay_occurrences"] += len(OKAY_RE.findall(combined))
                            if first_word(combined).lower() == "okay":
                                split_acc["all_text_first_word_okay"] += 1
                print(f"    rows={split_acc['rows']} assistants={split_acc['assistant_messages']} okay={split_acc['all_text_okay_messages']}", flush=True)
            except Exception as exc:  # keep other sources running if one split is unavailable
                split_acc["error"] = repr(exc)
                print(f"    ERROR: {exc!r}", file=sys.stderr, flush=True)
            split_stats[f"{source}/{split}"] = split_acc
            merge_stats(source_acc, split_acc)
        source_stats[source] = source_acc
        print(f"  total assistants={source_acc['assistant_messages']} all_text_okay={source_acc['all_text_okay_messages']}", flush=True)

    # Source-level rates and a target-mixture weighted estimate.
    rows = [source_rate_row(source, stats, SOURCES[source]["target_rows"])
            for source, stats in source_stats.items()]
    total_target = sum(SOURCES[name]["target_rows"] for name in SOURCES)
    weighted = {}
    rate_keys = [
        "reasoning_okay_message_pct",
        "content_okay_message_pct",
        "all_text_okay_message_pct",
        "reasoning_first_word_okay_pct",
        "content_first_word_okay_pct",
    ]
    for key in rate_keys:
        weighted[key] = sum(row[key] * row["target_rows"] for row in rows) / total_target

    pooled = new_stats()
    for stats in source_stats.values():
        merge_stats(pooled, stats)

    first_words = {
        source: {
            "reasoning": dict(stats["first_reasoning_words"].most_common(40)),
            "content": dict(stats["first_content_words"].most_common(40)),
            "okay_case_variants": dict(stats["okay_case_variants"].most_common()),
            "okay_position_buckets": dict(stats["okay_position_buckets"].most_common()),
        }
        for source, stats in source_stats.items()
    }

    summary = {
        "generated_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "rows_per_split": args.rows_per_split,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
        "sources": {name: {"dataset": spec["dataset"], "splits": spec["splits"], "target_rows": spec["target_rows"]}
                    for name, spec in SOURCES.items()},
        "source_rates": rows,
        "target_mixture_weighted_rates": weighted,
        "pooled_sample_rates": source_rate_row("pooled_sample", pooled, total_target),
        "pooled_sample_counts": serialise_stats(pooled),
    }

    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "first_words_and_okay_positions.json").write_text(json.dumps(first_words, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "split_stats.json").write_text(json.dumps({k: serialise_stats(v) for k, v in split_stats.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output / "source_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["source"])
        writer.writeheader()
        writer.writerows(rows)

    readme = "# SFT Okay statistics\n\n"
    readme += "This directory contains streaming statistics from the public NVIDIA Nemotron sources used by M2RL. It samples each published split after a deterministic shuffle; it does not download the full corpora or save raw examples.\n\n"
    readme += f"- Rows per split: `{args.rows_per_split}`\n- Shuffle buffer: `{args.shuffle_buffer}`\n- Seed: `{args.seed}`\n- Total target rows in M2RL mixture: `{total_target:,}`\n\n"
    readme += "`source_rates.csv` is the main table. `first_words_and_okay_positions.json` gives the most common first words and whether Okay tends to occur at the beginning or later in a response. `examples.json` contains short context snippets only.\n"
    (output / "README.md").write_text(readme, encoding="utf-8")

    print("\n=== SOURCE RATES ===", flush=True)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print("=== TARGET-MIXTURE WEIGHTED RATES ===", flush=True)
    print(json.dumps(weighted, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
