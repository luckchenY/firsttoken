#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


OKAY_RE = re.compile(r"(?<![A-Za-z])okay(?![A-Za-z])", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def text(value):
    return "" if value is None else (value if isinstance(value, str) else str(value))


def first_word(value):
    match = WORD_RE.search(value)
    return match.group(0) if match else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ["chat_if", "structured_outputs"]:
        stats = {
            "rows": 0,
            "assistant_messages": 0,
            "reasoning_nonempty": 0,
            "reasoning_any_okay": 0,
            "reasoning_first_word_okay": 0,
            "content_any_okay": 0,
            "content_first_word_okay": 0,
            "examples": [],
        }
        ds = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", split=split, streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=2000)
        for row_index, row in enumerate(ds):
            if row_index >= args.rows:
                break
            stats["rows"] += 1
            for message_index, message in enumerate(row.get("messages", [])):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                stats["assistant_messages"] += 1
                reasoning = text(message.get("reasoning_content", message.get("reasoning", "")))
                content = text(message.get("content", ""))
                if reasoning.strip():
                    stats["reasoning_nonempty"] += 1
                if OKAY_RE.search(reasoning):
                    stats["reasoning_any_okay"] += 1
                if first_word(reasoning).lower() == "okay":
                    stats["reasoning_first_word_okay"] += 1
                if OKAY_RE.search(content):
                    stats["content_any_okay"] += 1
                if first_word(content).lower() == "okay":
                    stats["content_first_word_okay"] += 1
                if len(stats["examples"]) < 12 and first_word(reasoning).lower() == "okay":
                    stats["examples"].append({
                        "row_index": row_index,
                        "message_index": message_index,
                        "reasoning": reasoning[:900],
                        "content": content[:500],
                    })
        report[split] = stats
        print(split, json.dumps(stats, ensure_ascii=False), flush=True)
    (out / "chat_okay_details.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
