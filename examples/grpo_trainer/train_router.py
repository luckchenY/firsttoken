#!/usr/bin/env python3
"""Train a router MLP that selects the best forced first token for a given prompt.

Pipeline:
  1. Load router training data saved by test_forced_first_token.py (--save-data)
  2. Load Qwen3-8B with HF transformers, forward pass on each prompt to get hidden states
  3. Train a small MLP: hidden_state [4096] -> logits [K]
  4. Save router weights

Usage:
  # Step 1: collect data (run test script with --save-data)
  python3 examples/grpo_trainer/test_forced_first_token.py \
      --model /data/chenyang2/Qwen3-8B \
      --data ~/data/gsm8k/test.parquet ~/data/math/test.parquet \
      --num-prompts 500 --rollout-n 8 \
      --forced-tokens 32313,71486,93217,106287,35439,4416,3925,16910 \
      --tp 4 --gpu-mem-util 0.9 \
      --save-data /data/chenyang2/router_data.pt

  # Step 2: train router
  python3 examples/grpo_trainer/train_router.py \
      --model /data/chenyang2/Qwen3-8B \
      --data /data/chenyang2/router_data.pt \
      --output /data/chenyang2/router_weights.pt \
      --epochs 50 --lr 1e-3 --batch-size 64
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class RouterMLP(nn.Module):
    """Small MLP that maps prompt hidden state -> token selection logits.

    Input:  hidden_state [B, D]  (D = model hidden size, e.g. 4096)
    Output: logits [B, K]        (K = number of candidate first tokens)
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024, num_candidates: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_candidates),
        )

    def forward(self, x):
        return self.net(x)  # [B, K] logits


def extract_hidden_states(model, tokenizer, prompts_text, device, batch_size=8):
    """Forward pass on prompts, return last-token hidden state for each prompt.

    Args:
        model: HF causal LM model
        tokenizer: tokenizer
        prompts_text: list of prompt strings (already chat-templated)
        device: torch device
        batch_size: forward pass batch size

    Returns:
        hidden_states: [N, D] tensor
    """
    model.eval()
    all_hidden = []

    with torch.no_grad():
        for i in range(0, len(prompts_text), batch_size):
            batch = prompts_text[i:i + batch_size]
            # Tokenize with padding
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            # Forward pass, get hidden states
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

            # last_hidden_state: [B, seq_len, D]
            last_hidden = outputs.hidden_states[-1]

            # Get the hidden state at the last non-padding token for each prompt
            # attention_mask: [B, seq_len], sum gives the actual length
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # [B], index of last token
            # Gather hidden state at last token position
            batch_indices = torch.arange(last_hidden.size(0), device=device)
            gathered = last_hidden[batch_indices, seq_lengths]  # [B, D]

            all_hidden.append(gathered.cpu())

    return torch.cat(all_hidden, dim=0)  # [N, D]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model path for hidden state extraction")
    parser.add_argument("--data", required=True, help="Path to router_data.pt from test script")
    parser.add_argument("--output", required=True, help="Path to save router weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--bad-weight", type=float, default=2.0,
                        help="Weight multiplier for bad groups (router is only called for "
                             "bad groups during training). 1.0 = no weighting.")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda or cpu or auto")
    args = parser.parse_args()

    # ---- device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- load data ----
    print(f"Loading router data from {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    prompts_text = data["prompts_text"]       # list[str], chat-templated
    rewards = data["rewards"]                   # tensor [N, K]
    forced_token_ids = data["forced_token_ids"] # list[int], K token ids
    K = len(forced_token_ids)
    N = len(prompts_text)

    # group_was_bad: True if normal rollout had no correct answer (new format)
    # If not present (old format), assume all are bad groups
    if "group_was_bad" in data:
        group_was_bad = data["group_was_bad"]  # tensor [N] bool
    else:
        group_was_bad = torch.ones(N, dtype=torch.bool)

    # Sanity checks
    assert rewards.shape[0] == N, f"rewards rows ({rewards.shape[0]}) != num prompts ({N})"
    assert rewards.shape[1] == K, f"rewards cols ({rewards.shape[1]}) != num tokens ({K})"

    n_bad = group_was_bad.sum().item()
    n_good = N - n_bad
    print(f"  {N} prompts ({n_bad} bad + {n_good} good), {K} candidate tokens")
    print(f"  Token ids: {forced_token_ids}")
    print(f"  Reward matrix: {rewards.shape}  (N, K)")
    print(f"  Prompts with >=1 correct token: {(rewards.sum(dim=1) > 0).sum().item()}/{N}")
    print(f"    Bad groups with >=1 correct: {((rewards.sum(dim=1) > 0) & group_was_bad).sum().item()}/{n_bad}")
    print(f"    Good groups with >=1 correct: {((rewards.sum(dim=1) > 0) & ~group_was_bad).sum().item()}/{n_good}")
    print(f"  Correct per token: {rewards.sum(dim=0).tolist()}")

    # ---- extract hidden states ----
    print(f"\nLoading model {args.model} for hidden state extraction ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    print(f"Extracting hidden states for {N} prompts ...")
    hidden_states = extract_hidden_states(model, tokenizer, prompts_text, device)
    # Convert to float32 for training (hidden states are bfloat16 from the model)
    hidden_states = hidden_states.float()
    print(f"  Hidden states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")

    # Free model memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- train router ----
    D = hidden_states.shape[1]
    router = RouterMLP(input_dim=D, hidden_dim=args.hidden_dim, num_candidates=K)
    print(f"\nRouter architecture: {router}")

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=0.01)

    # Dataset: only use prompts where at least one token works
    has_correct = rewards.sum(dim=1) > 0  # [N]
    train_hidden = hidden_states[has_correct]  # [N', D]
    train_rewards = rewards[has_correct]  # [N', K]
    train_was_bad = group_was_bad[has_correct]  # [N'] bool
    train_n = train_hidden.shape[0]
    n_train_bad = train_was_bad.sum().item()
    n_train_good = train_n - n_train_bad
    print(f"\nTraining on {train_n} prompts (with >=1 correct token)")
    print(f"  Bad groups: {n_train_bad}, Good groups: {n_train_good}")

    if train_n == 0:
        print("ERROR: No prompts with any correct token. Cannot train router.")
        return

    # Per-sample weights: upweight bad groups since router is only called for bad groups
    # during GRPO training. Bad groups get weight = args.bad_weight, good groups get 1.0.
    sample_weights = torch.where(train_was_bad, args.bad_weight, 1.0)
    # Normalize so mean weight = 1
    sample_weights = sample_weights / sample_weights.mean()
    print(f"  Sample weights: bad={args.bad_weight}, good=1.0 (normalized)")

    dataset = TensorDataset(train_hidden, train_rewards, sample_weights)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Weighted multi-label BCE loss: predict which tokens will work
    best_loss = float("inf")
    for epoch in range(args.epochs):
        router.train()
        total_loss = 0
        for batch_hidden, batch_rewards, batch_weights in dataloader:
            logits = router(batch_hidden)  # [B, K]
            # Per-sample weighted BCE
            loss_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch_rewards, reduction="none")  # [B, K]
            loss_per_sample = loss_per_sample.mean(dim=1)  # [B] average over K tokens
            loss = (loss_per_sample * batch_weights).mean()  # weighted average over batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in router.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Evaluate: for each prompt, does argmax select a correct token?
            router.eval()
            with torch.no_grad():
                all_logits = router(train_hidden)  # [N', K]
                selected = all_logits.argmax(dim=1)  # [N']
                # Check if selected token is correct
                correct = train_rewards[torch.arange(train_n), selected] > 0
                acc = correct.float().mean().item()
                # Per-group-type accuracy
                bad_mask = train_was_bad
                good_mask = ~train_was_bad
                bad_acc = correct[bad_mask].float().mean().item() if bad_mask.any() else 0.0
                good_acc = correct[good_mask].float().mean().item() if good_mask.any() else 0.0
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: loss={avg_loss:.4f}, "
                  f"acc={acc:.4f} ({correct.sum()}/{train_n}), "
                  f"bad_acc={bad_acc:.4f}, good_acc={good_acc:.4f}")

    # ---- save ----
    router.load_state_dict(best_state)
    save_dict = {
        "router_state_dict": router.state_dict(),
        "config": {
            "input_dim": D,
            "hidden_dim": args.hidden_dim,
            "num_candidates": K,
        },
        "forced_token_ids": forced_token_ids,
    }
    torch.save(save_dict, args.output)
    print(f"\nRouter weights saved to {args.output}")

    # ---- final evaluation ----
    router.eval()
    with torch.no_grad():
        all_logits = router(train_hidden)
        selected = all_logits.argmax(dim=1)
        correct = train_rewards[torch.arange(train_n), selected] > 0
        acc = correct.float().mean().item()
        bad_mask = train_was_bad
        good_mask = ~train_was_bad
        bad_acc = correct[bad_mask].float().mean().item() if bad_mask.any() else 0.0
        good_acc = correct[good_mask].float().mean().item() if good_mask.any() else 0.0
    print(f"\nFinal selection accuracy: {acc:.4f} ({correct.sum()}/{train_n})")
    print(f"  Bad groups:  {bad_acc:.4f} ({correct[bad_mask].sum()}/{bad_mask.sum().item()})")
    print(f"  Good groups: {good_acc:.4f} ({correct[good_mask].sum()}/{good_mask.sum().item()})")
    print(f"  (If router always picked the best token, it would be 1.0)")
    print(f"  (Random baseline would be ~{1/K:.4f})")


if __name__ == "__main__":
    main()
