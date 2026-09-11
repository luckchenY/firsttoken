#!/usr/bin/env python3
# Copyright 2025
# Licensed under the Apache License, Version 2.0
"""GRPO variant: re-rollout groups with no correct answer using forced first tokens.

Algorithm (per training step):
  1. Normal rollout produces ``n`` trajectories per prompt group.
  2. Reward is computed (rule-based, during rollout).
  3. For every group that has **no** correct trajectory (reward <= 0):
       a. Assign a forced first token to each of the ``n`` trajectories in the
          group, taken in order from a token list (``FORCED_FIRST_TOKEN_LIST``).
       b. Re-rollout the whole group in batch (model generates a continuation
          conditioned on the forced first token).
       c. If any re-rolled trajectory is correct, take the one with the smallest
          index and **replace position 0** of the original group with it.
  4. Everything else (advantage computation, actor/critic updates) is standard
     GRPO.

This file does NOT modify any existing verl code.  It only:
  - subclasses ``RayPPOTrainer`` to wrap ``generate_sequences`` with the
    re-rollout logic, and
  - provides a ``main`` entry point (a copy of ``main_ppo_v0``'s TaskRunner that
    instantiates the custom trainer instead of ``RayPPOTrainer``).

The token list is read from the environment variable ``FORCED_FIRST_TOKEN_LIST``
as a comma-separated list of token ids, e.g. ``FORCED_FIRST_TOKEN_LIST=151665,151666``.
If the list has fewer entries than ``rollout.n``, it wraps around with modulo.
"""

import os
import socket
from typing import Any

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import (
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
)
from verl.utils.config import validate_config


def _load_forced_token_list() -> list[int]:
    """Read forced first-token ids from the FORCED_FIRST_TOKEN_LIST env var."""
    raw = os.environ.get("FORCED_FIRST_TOKEN_LIST", "")
    raw = raw.strip()
    if not raw:
        return []
    tokens = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            tokens.append(int(part))
    return tokens


# Tensor keys that describe a full trajectory and must be replaced together.
_REPLACE_KEYS = [
    "prompts",
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "response_mask",
    "rm_scores",
    "rollout_log_probs",
]


class ForcedFirstTokenGRPOTrainer(RayPPOTrainer):
    """RayPPOTrainer subclass that re-rollouts all-wrong groups with forced first tokens.

    If ROUTER_WEIGHTS_PATH env var is set, uses a trained router to select
    the best forced token per bad group (1 generation each). Otherwise,
    falls back to trying all K tokens (K generations each).
    """

    def init_workers(self):
        super().init_workers()
        self._last_rerollout_stats: dict = {}

        # Load router if available
        router_path = os.environ.get("ROUTER_WEIGHTS_PATH", "")
        self._router_manager = None
        if router_path and os.path.exists(router_path):
            from examples.grpo_trainer.router_manager import RouterManager
            model_path = self.config.actor_rollout_ref.model.path
            self._router_manager = RouterManager(
                router_path=router_path,
                model_path=model_path,
                device="cuda:0",
            )
            print(f"[forced_first_token_grpo] Router loaded from {router_path}")
        else:
            print(f"[forced_first_token_grpo] No router loaded (ROUTER_WEIGHTS_PATH not set). "
                  f"Using full re-rollout (all tokens).")

        # Wrap the rollout manager's generate_sequences with our re-rollout logic.
        self._original_generate_sequences = self.async_rollout_manager.generate_sequences
        self.async_rollout_manager.generate_sequences = self._wrapped_generate_sequences

    # ------------------------------------------------------------------ wrapper
    def _wrapped_generate_sequences(self, gen_batch: DataProto) -> DataProto:
        gen_output = self._original_generate_sequences(gen_batch)

        # Skip during validation or when no rule-based reward is available yet.
        if gen_batch.meta_info.get("validate", False):
            return gen_output
        if "rm_scores" not in gen_output.batch:
            return gen_output

        return self._rerollout_bad_groups(gen_batch, gen_output)

    # ----------------------------------------------------------- re-rollout core
    def _rerollout_bad_groups(self, gen_batch: DataProto, gen_output: DataProto) -> DataProto:
        rollout_n: int = self.config.actor_rollout_ref.rollout.n
        token_list = _load_forced_token_list()
        if not token_list:
            return gen_output

        rm_scores = gen_output.batch["rm_scores"]  # (N, resp_len)
        rewards = rm_scores.sum(dim=-1)  # (N,)
        uids = gen_output.non_tensor_batch["uid"]
        N = len(gen_output)
        if N == 0 or N % rollout_n != 0:
            return gen_output
        num_groups = N // rollout_n

        # ---- compute original accuracy (before re-rollout) ------------------
        original_correct_groups = 0
        bad_group_starts: list[int] = []
        for g in range(num_groups):
            start = g * rollout_n
            group_rewards = rewards[start: start + rollout_n]
            has_correct = bool((group_rewards > 0).any())
            if has_correct:
                original_correct_groups += 1
            else:
                bad_group_starts.append(start)

        original_acc = original_correct_groups / num_groups

        if not bad_group_starts:
            # all groups correct — still log accuracy
            step = gen_batch.meta_info.get("global_steps", -1)
            print(f"[forced_first_token_grpo] step={step} "
                  f"original_acc={original_acc:.4f} ({original_correct_groups}/{num_groups}) "
                  f"bad_groups=0 — no re-rollout needed")
            self._last_rerollout_stats = {
                "fft/original_acc": original_acc,
                "fft/after_acc": original_acc,
                "fft/rescued_groups": 0,
                "fft/bad_groups": 0,
                "fft/total_groups": num_groups,
            }
            return gen_output

        step = gen_batch.meta_info.get("global_steps", -1)
        print(
            f"[forced_first_token_grpo] step={step} "
            f"original_acc={original_acc:.4f} ({original_correct_groups}/{num_groups}) "
            f"bad_groups={len(bad_group_starts)}/{num_groups}, re-rolling out..."
        )

        # Build the re-rollout batch from the bad groups' rows.
        # If router is available, select 1 token per bad group.
        # Otherwise, try all K tokens (rollout_n per bad group).
        rerollout_indices: list[int] = []
        forced_tokens: list[int] = []

        if self._router_manager is not None:
            # --- Router mode: 1 generation per bad group ---
            # Extract prompt text for bad groups
            bad_prompts_text = []
            for start in bad_group_starts:
                # Get the prompt text from the original gen_batch
                # The prompt is stored in non_tensor_batch
                prompt_text = gen_output.non_tensor_batch["prompts"][start]
                if isinstance(prompt_text, list):
                    # Apply chat template if needed
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_text, tokenize=False, add_generation_prompt=True
                    )
                bad_prompts_text.append(prompt_text)

            # Router selects best token for each bad group
            selected_tokens = self._router_manager.select_tokens(bad_prompts_text)
            print(f"[forced_first_token_grpo] Router selected tokens: "
                  f"{[repr(self.tokenizer.decode([t])) for t in selected_tokens]}")

            for idx, start in enumerate(bad_group_starts):
                rerollout_indices.append(start)
                forced_tokens.append(selected_tokens[idx])

            rerollout_n_per_group = 1
        else:
            # --- Full mode: try all K tokens (original behavior) ---
            for start in bad_group_starts:
                for i in range(rollout_n):
                    rerollout_indices.append(start + i)
                    forced_tokens.append(token_list[i % len(token_list)])
            rerollout_n_per_group = rollout_n

        rerollout_non_tensor: dict[str, np.ndarray] = {}
        for key, val in gen_output.non_tensor_batch.items():
            rerollout_non_tensor[key] = val[np.array(rerollout_indices, dtype=np.int64)]
        rerollout_non_tensor["agent_name"] = np.array(
            ["forced_first_token_agent"] * len(rerollout_indices), dtype=object
        )
        rerollout_non_tensor["forced_first_token"] = np.array(forced_tokens, dtype=np.int64)

        rerollout_batch = DataProto(
            batch=None,
            non_tensor_batch=rerollout_non_tensor,
            meta_info={"global_steps": gen_batch.meta_info.get("global_steps", -1)},
        )

        # Pad so the batch is divisible by the number of agent-loop workers.
        size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
        rerollout_batch_padded, pad_size = pad_dataproto_to_divisor(rerollout_batch, size_divisor)

        # Re-rollout (replicas are still awake here).
        rerollout_output_padded = self._original_generate_sequences(rerollout_batch_padded)

        if pad_size > 0:
            rerollout_output = unpad_dataproto(rerollout_output_padded, pad_size)
        else:
            rerollout_output = rerollout_output_padded

        if "rm_scores" not in rerollout_output.batch:
            print("[forced_first_token_grpo] re-rollout produced no rm_scores; skipping replacement")
            self._last_rerollout_stats = {
                "fft/original_acc": original_acc,
                "fft/after_acc": original_acc,
                "fft/rescued_groups": 0,
                "fft/bad_groups": len(bad_group_starts),
                "fft/total_groups": num_groups,
            }
            return gen_output

        rerollout_rewards = rerollout_output.batch["rm_scores"].sum(dim=-1)
        replaced = 0
        rescued_by_token: dict[int, int] = {}  # token_id -> how many groups it rescued
        for idx, start in enumerate(bad_group_starts):
            r_start = idx * rerollout_n_per_group
            r_rewards = rerollout_rewards[r_start: r_start + rerollout_n_per_group]
            correct_mask = r_rewards > 0
            if not bool(correct_mask.any()):
                continue  # still no correct answer after re-rollout
            j = int(torch.argmax(correct_mask).item())  # smallest correct index
            self._replace_trajectory(gen_output, start, rerollout_output, r_start + j)
            replaced += 1
            # which forced token rescued this group?
            used_token = forced_tokens[r_start + j]
            rescued_by_token[used_token] = rescued_by_token.get(used_token, 0) + 1

        after_correct_groups = original_correct_groups + replaced
        after_acc = after_correct_groups / num_groups

        print(
            f"[forced_first_token_grpo] step={step} "
            f"after_acc={after_acc:.4f} ({after_correct_groups}/{num_groups}) "
            f"rescued={replaced}/{len(bad_group_starts)} bad groups "
            f"(+{after_acc - original_acc:.4f} improvement)"
        )
        if rescued_by_token:
            token_strs = {tid: repr(self.tokenizer.decode([tid])) for tid in rescued_by_token}
            print(f"[forced_first_token_grpo]   rescued by token: "
                  f"{ {token_strs.get(t, t): c for t, c in rescued_by_token.items()} }")

        self._last_rerollout_stats = {
            "fft/original_acc": original_acc,
            "fft/after_acc": after_acc,
            "fft/acc_improvement": after_acc - original_acc,
            "fft/rescued_groups": replaced,
            "fft/bad_groups": len(bad_group_starts),
            "fft/total_groups": num_groups,
        }
        return gen_output

    @staticmethod
    def _replace_trajectory(
        gen_output: DataProto, dst_idx: int, rerollout_output: DataProto, src_idx: int
    ) -> None:
        """Overwrite row ``dst_idx`` of ``gen_output`` with row ``src_idx`` of ``rerollout_output``."""
        for key in _REPLACE_KEYS:
            if key not in gen_output.batch or key not in rerollout_output.batch:
                continue
            dst = gen_output.batch[key]
            src = rerollout_output.batch[key]
            gen_output.batch[key][dst_idx] = src[src_idx].to(dst.device, dtype=dst.dtype)


# ===========================================================================
#  Main entry point — a copy of main_ppo_v0.TaskRunner that uses the custom
#  trainer.  (We cannot reuse main_ppo_v0 directly because it hardcodes
#  ``RayPPOTrainer``.)
# ===========================================================================
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role  # noqa: E402


class _BaseTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = (
            lora_rank > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        from verl.workers.engine_workers import TrainingWorker

        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
            reward_pool = (
                [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            )
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")
            teacher_pool = (
                [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            )
            resource_pool_spec["teacher_pool"] = teacher_pool

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping
        )
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        if config.reward.reward_model.enable:
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_teacher_model_resource_pool(self, config):
        if is_distillation_enabled(config.get("distillation")):
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        return

    def run(self, config):
        pass


@ray.remote
class ForcedFirstTokenTaskRunner(_BaseTaskRunner):
    """Ray remote task runner that instantiates ForcedFirstTokenGRPOTrainer."""

    def __init__(self):
        super().__init__()

    def run(self, config):
        from pprint import pprint

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer = model_config.tokenizer
        processor = model_config.processor

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = ForcedFirstTokenGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def run_ppo(config, task_runner_class):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.utils.device import is_cuda_available

    rollout_cfg = config.actor_rollout_ref.rollout
    rm_rollout_cfg = config.reward.reward_model.rollout
    if rollout_cfg.full_determinism or (
        config.reward.reward_model.enable and rm_rollout_cfg.full_determinism
    ):
        os.environ["VERL_FULL_DETERMINISM"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["PYTHONHASHSEED"] = str(rollout_cfg.seed)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env(config)
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        if config.transfer_queue.enable:
            default_runtime_env.setdefault("env_vars", {})["TRANSFER_QUEUE_ENABLE"] = "1"
        # Forward NCCL env vars to Ray workers (shell exports don't propagate
        # to Ray worker processes automatically).
        for _nccl_key in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE"):
            _val = os.environ.get(_nccl_key)
            if _val is not None:
                default_runtime_env.setdefault("env_vars", {})[_nccl_key] = _val
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform."
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "verl", "trainer", "config")
)


@hydra.main(config_path=_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    """Hydra entry point for forced-first-token GRPO training."""
    from verl.trainer.ppo.utils import need_reference_policy, need_critic
    from verl.utils.config import validate_config as _vc
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    _vc(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    # This algorithm requires the legacy V0 path (RayPPOTrainer-based) which our
    # subclass extends.  V1 is the default in config; we force V0 here.
    if config.trainer.get("use_v1", True):
        print(
            "[forced_first_token_grpo] trainer.use_v1 is True; forcing V0 path "
            "(RayPPOTrainer) for forced-first-token re-rollout support."
        )
    run_ppo(config, ForcedFirstTokenTaskRunner)


if __name__ == "__main__":
    main()
