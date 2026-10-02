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
    rows = pq.read_table(sys.argv[1], columns=["messages", "tools"]).to_pylist()
    invalid = 0
    for row_index, row in enumerate(rows):
        tools = decode(row.get("tools"))
        if tools is not None and not isinstance(tools, list):
            print(f"invalid_tools row={row_index} type={type(tools).__name__} value={repr(tools)[:1000]}")
            invalid += 1
        for message_index, message in enumerate(row["messages"]):
            calls = decode(message.get("tool_calls"))
            if not calls:
                continue
            if not isinstance(calls, list):
                print(f"invalid_calls_type row={row_index} message={message_index} value={repr(calls)[:2000]}")
                invalid += 1
                continue
            for call_index, call in enumerate(calls):
                normalized = call.get("function") if isinstance(call, dict) and call.get("function") else call
                if not isinstance(normalized, dict) or "name" not in normalized or "arguments" not in normalized:
                    print(
                        f"invalid_call row={row_index} message={message_index} call={call_index} "
                        f"value={repr(call)[:3000]}"
                    )
                    invalid += 1
                    if invalid >= 20:
                        print("stopping_after_20_invalid")
                        print(f"invalid_count_at_least={invalid}")
                        return
    print(f"invalid_count={invalid}")


if __name__ == "__main__":
    main()
