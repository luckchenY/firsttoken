"""verl SFT dataset loader for the cleaned M2RL JSONL->Parquet conversion.

The conversion keeps tool-bearing fields as JSON strings to avoid a highly
irregular Parquet nested schema.  Qwen's chat template needs the decoded
Python lists, so decode them after the stock MultiTurnSFTDataset has loaded
the Parquet file.
"""

from __future__ import annotations

import json

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def decode_json(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class M2RLJSONToolsSFTDataset(MultiTurnSFTDataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        # This pilot contains text-only conversations.  Some verl model
        # configurations expose an auxiliary AutoProcessor even for a text
        # checkpoint; forcing the tokenizer path avoids multimodal content
        # wrapping and keeps Qwen3's text chat template type-stable.
        super().__init__(parquet_files, tokenizer, config, processor=None, max_samples=max_samples)

    def _read_files_and_process(self):
        super()._read_files_and_process()

        if self.tools is not None:
            self.tools = [decode_json(value) for value in self.tools]

        for messages in self.messages:
            for message in messages:
                if not isinstance(message, dict):
                    continue
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, str):
                    decoded = decode_json(tool_calls)
                    if isinstance(decoded, list):
                        message["tool_calls"] = decoded

    def _build_messages(self, example: dict):
        messages = super()._build_messages(example)

        for message in messages:
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, str):
                decoded = decode_json(tool_calls)
                if isinstance(decoded, list):
                    message["tool_calls"] = decoded

        # HFModelConfig may expose an AutoProcessor even for a text-only Qwen
        # checkpoint.  The stock dataset then wraps every string as
        # [{"type": "text", "text": ...}], but Qwen3's text template calls
        # startswith()/endswith() on user content.  There are no images or
        # videos in this M2RL SFT pilot, so unwrap text-only blocks.
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            text_parts = []
            can_unwrap = True
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") in {"text", "input_text"}:
                    text_parts.append(block.get("text", ""))
                else:
                    can_unwrap = False
                    break
            if can_unwrap:
                message["content"] = "".join(text_parts)
        return messages
