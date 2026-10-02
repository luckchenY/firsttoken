#!/usr/bin/env python3
from __future__ import annotations

import json
import copy
import sys

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from m2rl_json_tools_sft_dataset import M2RLJSONToolsSFTDataset


def main() -> None:
    model_path, parquet_path, index_text = sys.argv[1:4]
    index = int(index_text)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": None,
            "pad_mode": "no_padding",
            "max_length": 32768,
            "truncation": "right",
            "ignore_input_ids_mismatch": True,
        }
    )
    dataset = M2RLJSONToolsSFTDataset(parquet_path, tokenizer, config)
    messages = dataset._build_messages(dataset.dataframe.iloc[index].to_dict())
    tools = dataset.tools[index]
    print(f"messages={len(messages)} tools_type={type(tools).__name__}")
    for i, message in enumerate(messages):
        print(f"message={i} role={message.get('role')} content_type={type(message.get('content')).__name__} tool_calls={repr(message.get('tool_calls'))[:500]}")
        try:
            out = tokenizer.apply_chat_template(
                [message],
                tools=tools if i == 0 else None,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            print(f"  single_ok={out['input_ids'].shape}")
        except Exception as exc:
            print(f"  single_error={type(exc).__name__}: {exc}")

    for i, message in enumerate(messages):
        calls = message.get("tool_calls")
        if not calls:
            continue
        native = copy.deepcopy(message)
        native["tool_calls"] = [
            {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
            for call in calls
        ]
        try:
            out = tokenizer.apply_chat_template(
                [native],
                tools=tools if i == 0 else None,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            print(f"native_tool_call_ok={out['input_ids'].shape}")
        except Exception as exc:
            print(f"native_tool_call_error={type(exc).__name__}: {exc}")
    try:
        out = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        print(f"full_ok={out['input_ids'].shape}")
    except Exception as exc:
        print(f"full_error={type(exc).__name__}: {exc}")
    try:
        item = dataset[index]
        print(f"dataset_ok={item['input_ids'].shape}")
    except Exception as exc:
        print(f"dataset_error={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
