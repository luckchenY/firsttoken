#!/usr/bin/env python3
"""Router manager: loads trained MLP router + frozen HF model for hidden state extraction.

During GRPO training, for each bad group (all rollouts wrong), the router
selects the best forced first token instead of trying all K tokens.

Usage:
  Set env var ROUTER_WEIGHTS_PATH=/path/to/router_weights.pt before training.
  If not set, the trainer falls back to trying all tokens (original behavior).
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class RouterMLP(nn.Module):
    """Small MLP: hidden_state [D] -> logits [K]."""

    def __init__(self, input_dim=4096, hidden_dim=1024, num_candidates=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_candidates),
        )

    def forward(self, x):
        return self.net(x)


class RouterManager:
    """Manages router MLP + frozen HF model for hidden state extraction."""

    def __init__(self, router_path, model_path, device="cuda:0"):
        # Load router weights
        checkpoint = torch.load(router_path, weights_only=False)
        cfg = checkpoint["config"]
        self.router = RouterMLP(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_candidates=cfg["num_candidates"],
        ).to(device)
        self.router.load_state_dict(checkpoint["router_state_dict"])
        self.router.eval()
        self.forced_token_ids = checkpoint["forced_token_ids"]
        self.K = len(self.forced_token_ids)

        # Load frozen HF model for hidden state extraction
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def select_tokens(self, prompts_text):
        """For each prompt, return the router-selected forced token id.

        Args:
            prompts_text: list of chat-templated prompt strings

        Returns:
            list of token ids (one per prompt)
        """
        inputs = self.tokenizer(
            prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len, D]

        # Get hidden state at last non-padding token
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.size(0), device=self.device)
        hidden = last_hidden[batch_indices, seq_lengths]  # [B, D]

        # Router selects token
        logits = self.router(hidden)  # [B, K]
        selected = logits.argmax(dim=-1)  # [B]
        token_ids = [self.forced_token_ids[i] for i in selected.tolist()]
        return token_ids
