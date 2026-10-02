#!/usr/bin/env python3
"""Find the first Parquet sample that cannot pass Qwen3 tokenization."""

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
        try:
            dataset[index]
        except Exception as exc:
            messages = dataset._build_messages(dataset.dataframe.iloc[index].to_dict())
            print(f"bad_index={index}")
            print(f"error={type(exc).__name__}: {exc}")
            for message_index, message in enumerate(messages):
                content = message.get("content")
                print(
                    f"message={message_index} role={message.get('role')} "
                    f"content_type={type(content).__name__} "
                    f"content_head={repr(content)[:300]}"
                )
            raise
        if index and index % 1000 == 0:
            print(f"checked={index}", flush=True)
    print("all_samples_ok")


if __name__ == "__main__":
    main()
