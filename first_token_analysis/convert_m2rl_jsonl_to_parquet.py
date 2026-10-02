#!/usr/bin/env python3
"""Convert the cleaned M2RL SFT JSONL into the Parquet shape used by verl.

The JSONL intentionally keeps ``tools`` and assistant ``tool_calls`` as JSON
strings so that it remains easy to inspect.  The custom verl dataset loader
parses those fields back into Python objects immediately before applying the
Qwen chat template.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


MESSAGE_TYPE = pa.struct(
    [
        pa.field("content", pa.string()),
        pa.field("reasoning_content", pa.string()),
        pa.field("role", pa.string()),
        pa.field("tool_calls", pa.string()),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("messages", pa.list_(MESSAGE_TYPE)),
        pa.field("tools", pa.string()),
    ]
)


def json_string(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_message(message: dict) -> dict:
    return {
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "role": message.get("role"),
        "tool_calls": json_string(message.get("tool_calls")),
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT.jsonl OUTPUT.parquet")

    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(target, SCHEMA, compression="zstd")
    batch = []
    rows = 0
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = [normalize_message(m) for m in record["messages"]]
                tools = record.get("tools", "[]")
                batch.append({"messages": messages, "tools": json_string(tools) or "[]"})
                if len(batch) >= 256:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    rows += len(batch)
                    batch.clear()
                    if rows % 8192 == 0:
                        print(f"converted {rows} rows", flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            rows += len(batch)
    finally:
        writer.close()

    print(f"wrote {rows} rows to {target}")


if __name__ == "__main__":
    main()
