#!/usr/bin/env python3
"""Smoke-test the cleaned M2RL Parquet through the verl SFT dataset path."""

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
            "num_workers": 0,
        }
    )
    dataset = M2RLJSONToolsSFTDataset(parquet_path, tokenizer, config)
    print(f"dataset_len={len(dataset)}")
    print(f"tools0_type={type(dataset.tools[0]).__name__}")
    print(f"messages0={len(dataset.messages[0])}")

    for index in range(min(5, len(dataset))):
        item = dataset[index]
        print(
            f"sample={index} input_len={item['input_ids'].numel()} "
            f"loss_tokens={int(item['loss_mask'].sum())} "
            f"has_tools={bool(dataset.tools[index])}"
        )

    for index, messages in enumerate(dataset.messages[:100]):
        if any(message.get("tool_calls") for message in messages if isinstance(message, dict)):
            print(f"tool_call_sample={index}")
            item = dataset[index]
            print(
                f"tool_call_input_len={item['input_ids'].numel()} "
                f"tool_call_loss_tokens={int(item['loss_mask'].sum())}"
            )
            break
    else:
        print("tool_call_sample=none_in_first_100")


if __name__ == "__main__":
    main()
