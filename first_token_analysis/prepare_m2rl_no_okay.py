#!/usr/bin/env python3
"""Stream a proportionally sampled M2RL SFT set and remove leading Okay markers.

The source datasets and target proportions follow Mosi-AI/M2RL's public data
processing configuration.  The script writes JSONL incrementally so it does
not need to hold the complete SFT corpus in RAM.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from datasets import Features, Json, List, Value, load_dataset


SOURCES = [
    ("math_proofs", "nvidia/Nemotron-Math-Proofs-v1", ["lean"], 335_122),
    (
        "math",
        "nvidia/Nemotron-Math-v2",
        ["high_part00", "high_part01", "high_part02", "medium", "low"],
        2_950_525,
    ),
    ("science", "nvidia/Nemotron-Science-v1", ["MCQ", "RQA"], 2_263_340),
    (
        "code",
        "nvidia/Nemotron-Competitive-Programming-v1",
        [
            "competitive_coding_cpp_part00",
            "competitive_coding_cpp_part01",
            "competitive_coding_python_part00",
            "competitive_coding_python_part01",
            "infinibyte_part00",
            "infinibyte_part01",
        ],
        3_927_984,
    ),
    (
        "chat",
        "nvidia/Nemotron-Instruction-Following-Chat-v1",
        ["chat_if", "structured_outputs"],
        4_309_780,
    ),
    (
        "agent",
        "nvidia/Nemotron-Agentic-v1",
        ["interactive_agent", "tool_calling"],
        335_122,
    ),
]

# The Agentic dataset has a small schema inconsistency across JSON shards:
# `tools[].function.parameters.properties` is sometimes an object and
# sometimes a JSON-encoded string.  Treat each tool as opaque JSON because
# the SFT record only needs to preserve it, not cast its nested schema.
AGENTIC_FEATURES = Features(
    {
        "uuid": Value("string"),
        "messages": List(Json()),
        "license": Value("string"),
        "used_in": List(Value("string")),
        "tools": List(Json()),
        "reasoning": Value("string"),
    }
)

LEADING_OKAY = re.compile(r"^(?P<prefix>\s*(?:<think>\s*)?)(?P<word>okay)\b", re.I)


def remove_leading_okay(text: Any) -> tuple[Any, bool]:
    """Remove only leading Okay markers and their attached punctuation."""
    if not isinstance(text, str) or not text:
        return text, False
    current = text
    changed = False
    for _ in range(4):
        match = LEADING_OKAY.match(current)
        if not match:
            break
        rest = current[match.end() :].lstrip()
        # Remove punctuation attached to the marker (ASCII and Unicode), so
        # `Okay, we...`, `Okay. We...`, and `Okay：我们...` all become
        # `we...`/`我们...` rather than starting with punctuation.
        while rest and unicodedata.category(rest[0]).startswith("P"):
            rest = rest[1:].lstrip()
        current = match.group("prefix") + rest
        changed = True
    return current, changed


def as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_messages(raw_messages: Any) -> tuple[list[dict[str, str]], int]:
    if not isinstance(raw_messages, list):
        return [], 0
    messages: list[dict[str, str]] = []
    changed_fields = 0
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        content = as_string(raw.get("content", ""))
        reasoning = as_string(raw.get("reasoning_content", ""))
        role = as_string(raw.get("role", ""))
        # Only assistant generations are model outputs.  Do not rewrite a
        # user prompt that happens to begin with the word "Okay".
        if role == "assistant":
            content, content_changed = remove_leading_okay(content)
            reasoning, reasoning_changed = remove_leading_okay(reasoning)
        else:
            content_changed = reasoning_changed = False
        changed_fields += int(content_changed) + int(reasoning_changed)
        tool_calls = raw.get("tool_calls", "")
        if not isinstance(tool_calls, str):
            tool_calls = json.dumps(tool_calls, ensure_ascii=False)
        messages.append(
            {
                "content": content,
                "reasoning_content": reasoning,
                "role": role,
                "tool_calls": tool_calls,
            }
        )
    return messages, changed_fields


def usable(messages: list[dict[str, str]]) -> bool:
    if not messages:
        return False
    # Keep only actual SFT conversations with an assistant target.  This also
    # drops a degenerate sample whose only assistant text was `Okay` and
    # became empty after marker removal.
    for message in messages:
        if message.get("role") != "assistant":
            continue
        if message.get("content", "").strip() or message.get("reasoning_content", "").strip():
            return True
        tool_calls = message.get("tool_calls", "").strip()
        if tool_calls not in {"", "[]", "{}", "null"}:
            return True
    return False


def allocate_targets(total: int) -> dict[str, int]:
    denominator = sum(item[3] for item in SOURCES)
    exact = {name: total * count / denominator for name, _, _, count in SOURCES}
    result = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(exact, key=lambda name: exact[name] - result[name], reverse=True)
    for name in order[:remaining]:
        result[name] += 1
    return result


def iter_source_rows(repo: str, splits: list[str], seed: int, buffer_size: int) -> Iterable[dict[str, Any]]:
    for split_index, split in enumerate(splits):
        print(f"  loading {repo}:{split}", flush=True)
        load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if repo == "nvidia/Nemotron-Agentic-v1":
            load_kwargs["features"] = AGENTIC_FEATURES
        stream = load_dataset(repo, **load_kwargs)
        stream = stream.shuffle(seed=seed + split_index, buffer_size=buffer_size)
        yield from stream


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-total", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse source files that already contain exactly their target rows",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = allocate_targets(args.target_total)
    combined_path = args.output_dir / "sft_train_no_okay.jsonl"
    manifest: dict[str, Any] = {
        "target_total": args.target_total,
        "seed": args.seed,
        "operation": "remove leading Okay/okay/OKAY and adjacent punctuation",
        "sources": {},
    }
    examples: list[dict[str, Any]] = []

    combined_mode = "a" if args.resume and combined_path.exists() else "w"
    with combined_path.open(combined_mode, encoding="utf-8") as combined:
        for source_index, (name, repo, splits, official_count) in enumerate(SOURCES):
            target = targets[name]
            selected = 0
            seen = 0
            changed_rows = 0
            skipped_rows = 0
            source_path = args.output_dir / f"{name}.jsonl"
            if args.resume and source_path.exists():
                existing_rows = count_lines(source_path)
                if existing_rows == target:
                    manifest["sources"][name] = {
                        "repo": repo,
                        "official_count": official_count,
                        "target": target,
                        "selected": existing_rows,
                        "scanned": None,
                        "skipped_empty": None,
                        "rows_with_removed_okay": None,
                        "resumed": True,
                        "output": str(source_path),
                    }
                    print(f"{name}: resumed existing {existing_rows}/{target}", flush=True)
                    continue
                if existing_rows > 0:
                    raise RuntimeError(
                        f"Cannot resume partial source {source_path}: "
                        f"found {existing_rows} rows, expected {target}"
                    )
            source_mode = "a" if args.resume and source_path.exists() else "w"
            with source_path.open(source_mode, encoding="utf-8") as source_out:
                for row in iter_source_rows(repo, splits, args.seed + source_index * 1000, args.buffer_size):
                    seen += 1
                    messages, changed_fields = normalize_messages(row.get("messages"))
                    if not usable(messages):
                        skipped_rows += 1
                        continue
                    record = {
                        "messages": messages,
                        "tools": as_string(row.get("tools", "[]")) or "[]",
                    }
                    line = json.dumps(record, ensure_ascii=False)
                    source_out.write(line + "\n")
                    combined.write(line + "\n")
                    selected += 1
                    changed_rows += int(changed_fields > 0)
                    if changed_fields > 0 and len(examples) < 20:
                        examples.append({"source": name, "record": record})
                    if selected >= target:
                        break
            manifest["sources"][name] = {
                "repo": repo,
                "official_count": official_count,
                "target": target,
                "selected": selected,
                "scanned": seen,
                "skipped_empty": skipped_rows,
                "rows_with_removed_okay": changed_rows,
                "output": str(source_path),
            }
            print(
                f"{name}: selected={selected}/{target}, scanned={seen}, "
                f"rows_with_removed_okay={changed_rows}, skipped_empty={skipped_rows}",
                flush=True,
            )

    manifest["combined_output"] = str(combined_path)
    manifest["examples"] = examples
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {combined_path}", flush=True)
    print(f"Saved {args.output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
