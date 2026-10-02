from datasets import get_dataset_split_names, load_dataset


DATASETS = [
    "nvidia/Nemotron-Math-Proofs-v1",
    "nvidia/Nemotron-Math-v2",
    "nvidia/Nemotron-Science-v1",
    "nvidia/Nemotron-Competitive-Programming-v1",
    "nvidia/Nemotron-Instruction-Following-Chat-v1",
    "nvidia/Nemotron-Agentic-v1",
]


for name in DATASETS:
    try:
        splits = get_dataset_split_names(name)
        print(f"{name}: splits={splits}", flush=True)
        split = splits[0]
        row = next(iter(load_dataset(name, split=split, streaming=True)))
        print(f"  sample split={split}; keys={list(row.keys())}", flush=True)
        print(f"  sample={str(row)[:1200]}", flush=True)
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
