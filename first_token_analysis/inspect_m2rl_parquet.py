#!/usr/bin/env python3
from __future__ import annotations

import sys

import pyarrow.parquet as pq


def main() -> None:
    path = sys.argv[1]
    table = pq.read_table(path, columns=["messages", "tools"])
    print(table.schema)
    messages_column = table.column("messages").to_pylist()
    tools_column = table.column("tools").to_pylist()
    print(f"rows={len(messages_column)}")
    for row_index, messages in enumerate(messages_column):
        for message_index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, str) and content is not None:
                print(
                    f"non_string_content row={row_index} message={message_index} "
                    f"type={type(content).__name__} value={repr(content)[:500]}"
                )
                return
        if row_index < 3:
            print(f"row={row_index} messages={messages}")
            print(f"row={row_index} tools_type={type(tools_column[row_index]).__name__}")
    print("all_content_values_are_strings_or_none")


if __name__ == "__main__":
    main()
