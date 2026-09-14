#!/usr/bin/env python3
"""Merge all batch_*.pt and split into train/test sets (stratified by data_source).

Usage:
    python3 examples/grpo_trainer/merge_and_split.py \
        --input-dir router_data_5000 \
        --output-dir router_data_5000/split \
        --test-ratio 0.1 --seed 42
"""
import argparse
import glob
import os
import random
import torch
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of data reserved for test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "batch_*.pt")))
    if not files:
        print(f"ERROR: no batch_*.pt in {args.input_dir}")
        return
    print(f"Found {len(files)} batch files")

    all_prompts, all_gt, all_ds, all_rewards = [], [], [], []
    forced_token_ids = None
    for f in files:
        d = torch.load(f, weights_only=False)
        all_prompts.extend(d["prompts_text"])
        all_gt.extend(d["ground_truths"])
        all_ds.extend(d["data_sources"])
        all_rewards.append(d["rewards"])
        if forced_token_ids is None:
            forced_token_ids = d["forced_token_ids"]

    rewards = torch.cat(all_rewards, dim=0)
    N, K = rewards.shape
    print(f"Merged: {N} prompts x {K} tokens")

    # Stratified split by data_source
    rng = random.Random(args.seed)
    by_ds = defaultdict(list)
    for i, ds in enumerate(all_ds):
        by_ds[ds].append(i)

    train_idx, test_idx = [], []
    print("\nPer-source split:")
    for ds in sorted(by_ds.keys()):
        idxs = by_ds[ds]
        rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * args.test_ratio))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
        n_surv_train = int((rewards[idxs[n_test:]].sum(dim=1) > 0).sum().item())
        n_surv_test = int((rewards[idxs[:n_test]].sum(dim=1) > 0).sum().item())
        print(f"  {ds:42} train={len(idxs)-n_test:>4} (存活{n_surv_train})  test={n_test:>3} (存活{n_surv_test})")

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    print(f"\nTotal: train={len(train_idx)}, test={len(test_idx)}")

    os.makedirs(args.output_dir, exist_ok=True)

    def save(name, idxs):
        out = {
            "prompts_text": [all_prompts[i] for i in idxs],
            "rewards": rewards[idxs],
            "forced_token_ids": forced_token_ids,
            "ground_truths": [all_gt[i] for i in idxs],
            "data_sources": [all_ds[i] for i in idxs],
        }
        path = os.path.join(args.output_dir, name)
        torch.save(out, path)
        n_surv = int((out["rewards"].sum(dim=1) > 0).sum().item())
        print(f"  Saved {path}: {len(idxs)} prompts, 存活 {n_surv} ({n_surv/len(idxs)*100:.1f}%)")

    save("train.pt", train_idx)
    save("test.pt", test_idx)

    # Also save full merged for reference
    save("all.pt", list(range(N)))


if __name__ == "__main__":
    main()
