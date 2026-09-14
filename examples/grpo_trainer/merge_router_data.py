#!/usr/bin/env python3
"""Merge multiple batch_*.pt files (from collect_router_data.py) into a single
router_data.pt file that train_router.py can consume.

Usage:
    python3 examples/grpo_trainer/merge_router_data.py \
        --input-dir router_data_5000 \
        --output router_data_5000/merged.pt
"""
import argparse
import glob
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing batch_*.pt files")
    parser.add_argument("--output", required=True,
                        help="Output merged .pt file path")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the merged data")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "batch_*.pt")))
    if not files:
        print(f"ERROR: no batch_*.pt found in {args.input_dir}")
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
        elif forced_token_ids != d["forced_token_ids"]:
            print(f"WARNING: forced_token_ids mismatch in {f}")

    rewards = torch.cat(all_rewards, dim=0)
    N, K = rewards.shape
    print(f"Merged: {N} prompts x {K} tokens = {N*K} trajectories")

    # stats
    n_surv = int((rewards.sum(dim=1) > 0).sum().item())
    print(f"  Survival: {n_surv}/{N} ({n_surv/N*100:.1f}%)")

    if args.shuffle:
        perm = torch.randperm(N)
        rewards = rewards[perm]
        all_prompts = [all_prompts[i] for i in perm.tolist()]
        all_gt = [all_gt[i] for i in perm.tolist()]
        all_ds = [all_ds[i] for i in perm.tolist()]
        print("  (shuffled)")

    out = {
        "prompts_text": all_prompts,
        "rewards": rewards,
        "forced_token_ids": forced_token_ids,
        # train_router.py uses group_was_bad for weighting; we don't have it,
        # so omit it and let train_router default to all-bad (no weighting).
        # Keep extra fields for reference / future use.
        "ground_truths": all_gt,
        "data_sources": all_ds,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
