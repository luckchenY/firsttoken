#!/usr/bin/env python3
from __future__ import annotations

import sys

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from m2rl_json_tools_sft_dataset import M2RLJSONToolsSFTDataset


def main() -> None:
    model_path, parquet_path = sys.argv[1:3]
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
    for index in range(len(dataset)):
        messages = dataset._build_messages(dataset.dataframe.iloc[index].to_dict())
        tools = dataset.tools[index] if dataset.tools is not None else None
        for message_index, message in enumerate(messages):
            try:
                tokenizer.apply_chat_template(
                    [message],
                    tools=tools if message_index == 0 else None,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as exc:
                print(f"bad_index={index} message={message_index}")
                print(f"error={type(exc).__name__}: {exc}")
                print(f"tools_type={type(tools).__name__} tools_head={repr(tools)[:500]}")
                print(f"message={repr(message)[:2000]}")
                raise
        if index and index % 1000 == 0:
            print(f"checked={index}", flush=True)
    print("all_message_templates_ok")


if __name__ == "__main__":
    main()
