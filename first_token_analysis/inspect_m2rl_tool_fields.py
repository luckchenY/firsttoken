#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

import pyarrow.parquet as pq


def decode(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def main() -> None:
    table = pq.read_table(sys.argv[1], columns=["messages", "tools"])
    rows = table.to_pylist()
    tool_call_rows = 0
    tools_rows = 0
    for row_index, row in enumerate(rows):
        tools = decode(row.get("tools"))
        if tools:
            tools_rows += 1
            if tools_rows <= 5:
                print(f"tools_row={row_index} tools={repr(tools)[:2000]}")
        for message_index, message in enumerate(row["messages"]):
            raw = message.get("tool_calls")
            calls = decode(raw)
            if calls:
                tool_call_rows += 1
                if tool_call_rows <= 20:
                    print(
                        f"tool_call_row={row_index} message={message_index} "
                        f"calls_type={type(calls).__name__} calls={repr(calls)[:4000]}"
                    )
    print(f"rows={len(rows)} rows_with_tools={tools_rows} rows_with_tool_calls={tool_call_rows}")


if __name__ == "__main__":
    main()
