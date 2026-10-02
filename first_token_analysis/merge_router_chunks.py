#!/usr/bin/env python3
"""Merge four router-data chunks and recreate the original train/test split.

The script can wait for the four background collection jobs to finish.  It
never writes to the source directory; all merged files go below --output-dir.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def load_chunks(paths: list[Path]) -> list[dict]:
    loaded = []
    for path in paths:
        loaded.append(torch.load(path, map_location="cpu", weights_only=False))
    return loaded


def wait_for_chunks(paths: list[Path], interval: int) -> list[dict]:
    while True:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            log(f"waiting for {len(missing)} chunk(s)")
            time.sleep(interval)
            continue
        try:
            loaded = load_chunks(paths)
        except Exception as exc:
            log(f"chunks exist but are not readable yet ({exc!r}); retrying")
            time.sleep(interval)
            continue
        return loaded


def row_count(data: dict) -> int:
    return len(data["prompts_text"])


def select_rows(data: dict, indices: list[int]) -> dict:
    result = {}
    for key, value in data.items():
        if key in {"source_range", "source_input_pt"}:
            continue
        if key == "rewards":
            result[key] = value[indices]
        elif key in {"prompts_text", "ground_truths", "data_sources"}:
            result[key] = [value[index] for index in indices]
        elif key == "forced_token_ids":
            result[key] = value
        elif not isinstance(value, (str, int, float, bool, type(None))):
            # Keep simple metadata only; tensors/lists here are not expected
            # in the generated chunk apart from the fields handled above.
            result[key] = value
        else:
            result[key] = value
    return result


def split_indices(merged: dict, source_split: Path, split_name: str) -> list[int]:
    old = torch.load(source_split / f"{split_name}.pt", map_location="cpu", weights_only=False)
    positions: dict[tuple[object, object, object], list[int]] = {}
    for index, key in enumerate(
        zip(merged["prompts_text"], merged["ground_truths"], merged["data_sources"])
    ):
        positions.setdefault(tuple(key), []).append(index)

    selected = []
    for row in zip(old["prompts_text"], old["ground_truths"], old["data_sources"]):
        key = tuple(row)
        candidates = positions.get(key)
        if not candidates:
            raise KeyError(f"could not find split row in merged data: {key!r}")
        selected.append(candidates.pop(0))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-split", required=True, type=Path)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    chunk_paths = [
        args.output_dir / "chunks" / f"chunk_{index:03d}.pt"
        for index in range(args.num_chunks)
    ]
    if args.wait:
        chunks = wait_for_chunks(chunk_paths, args.poll_seconds)
    else:
        chunks = load_chunks(chunk_paths)

    chunks.sort(key=lambda item: item["source_range"][0])
    expected_start = 0
    for chunk in chunks:
        start, end = chunk["source_range"]
        if start != expected_start or end - start != row_count(chunk):
            raise ValueError(
                f"non-contiguous chunk range {chunk['source_range']} "
                f"(expected start {expected_start}, rows {row_count(chunk)})"
            )
        expected_start = end

    merged = {
        "prompts_text": [row for chunk in chunks for row in chunk["prompts_text"]],
        "ground_truths": [row for chunk in chunks for row in chunk["ground_truths"]],
        "data_sources": [row for chunk in chunks for row in chunk["data_sources"]],
        "forced_token_ids": chunks[0]["forced_token_ids"],
        "rewards": torch.cat([chunk["rewards"] for chunk in chunks], dim=0),
        "model": chunks[0].get("model"),
        "source_input_pt": chunks[0].get("source_input_pt"),
        "num_chunks": len(chunks),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output_dir / "all.pt")
    log(f"saved {args.output_dir / 'all.pt'} with {len(merged['prompts_text'])} prompts")

    split_dir = args.output_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "test"):
        indices = split_indices(merged, args.source_split, split_name)
        split_data = select_rows(merged, indices)
        split_data["source_split"] = str(args.source_split / f"{split_name}.pt")
        torch.save(split_data, split_dir / f"{split_name}.pt")
        log(f"saved {split_dir / (split_name + '.pt')} with {len(indices)} prompts")


if __name__ == "__main__":
    main()
