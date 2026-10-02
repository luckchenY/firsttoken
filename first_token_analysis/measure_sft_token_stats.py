#!/usr/bin/env python3
import json
from pathlib import Path

from transformers import AutoTokenizer


DATA_DIR = Path("/data/chenyang2/verl/sft_data/m2rl_no_okay_100k_20260922")
MODEL_DIR = "/data/chenyang2/models/Qwen3-4B-Base"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True
    )
    for name in ["math_proofs", "math", "science", "code", "chat", "agent"]:
        path = DATA_DIR / f"{name}.jsonl"
        texts = []
        rows = 0
        bytes_total = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if rows >= 1000:
                    break
                record = json.loads(line)
                parts = []
                for message in record.get("messages", []):
                    parts.append(str(message.get("content") or ""))
                    parts.append(str(message.get("reasoning_content") or ""))
                texts.append("\n".join(parts))
                bytes_total += len(line.encode("utf-8"))
                rows += 1
        encoded = tokenizer(texts, add_special_tokens=False, padding=False)
        lengths = sorted(len(ids) for ids in encoded["input_ids"])
        p95_index = max(0, int(rows * 0.95) - 1)
        print(
            name,
            "rows=", rows,
            "avg_tokens=", round(sum(lengths) / rows, 1),
            "p50=", lengths[rows // 2],
            "p95=", lengths[p95_index],
            "max=", max(lengths),
            "avg_bytes=", round(bytes_total / rows),
            flush=True,
        )


if __name__ == "__main__":
    main()
