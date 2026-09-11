#!/usr/bin/env python3
# Copyright 2025
# Licensed under the Apache License, Version 2.0
"""Custom single-turn agent loop that can force the first response token.

When ``forced_first_token`` (a token id >= 0) is supplied via the trajectory's
non-tensor batch, the loop appends that token to the prompt before generation so
the model produces a *continuation* conditioned on it.  The forced token is then
moved to the response side (``response_mask = 1``) so it participates in the
training loss, exactly as if the model had chosen it.

If ``forced_first_token`` is absent / None / < 0, behaviour is identical to the
standard ``single_turn_agent``.
"""

import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = os.getenv("VERL_LOGGING_LEVEL", "WARN")  # noqa: F841


@register("forced_first_token_agent")
class ForcedFirstTokenAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop with optional forced first response token."""

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. build the initial prompt with Continuous Token
        self._assert_mm_supported(bool(multi_modal_data))
        prompt_ids = await self.ct_build_initial_tokens(
            messages,
            images=images,
            videos=videos,
            audios=audios,
        )

        # ---- forced first token -------------------------------------------------
        forced_token = kwargs.get("forced_first_token", None)
        if forced_token is not None:
            try:
                forced_token = int(forced_token)
            except (TypeError, ValueError):
                forced_token = None
        has_forced = forced_token is not None and forced_token >= 0
        # Prompt fed to the engine: original prompt + [forced_token] so the
        # model generates a continuation conditioned on the forced first token.
        gen_prompt_ids = list(prompt_ids) + [forced_token] if has_forced else list(prompt_ids)

        # 3. generate sequences
        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            request_id = (
                f"det-{priority}"
                if getattr(self.rollout_config, "full_determinism", False)
                else uuid4().hex
            )
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=gen_prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = (
                output.num_preempted if output.num_preempted is not None else -1
            )

        # ---- build assistant token ids / logprobs for merge --------------------
        if has_forced:
            # Prepend the forced token to the response; it gets mask=1 (model
            # "chose" it).  A dummy 0.0 logprob is inserted for the forced token;
            # real logprobs are recomputed by the actor during training.
            assistant_token_ids = [forced_token] + list(output.token_ids)
            if output.log_probs:
                assistant_logprobs = [0.0] + list(output.log_probs)
                response_logprob_seed: list[float] | None = []
            else:
                assistant_logprobs = None
                response_logprob_seed = None
        else:
            assistant_token_ids = list(output.token_ids)
            assistant_logprobs = output.log_probs if output.log_probs else None
            response_logprob_seed = [] if output.log_probs else None

        # 4. merge (handles any chat-template boundary tokens)
        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            assistant_token_ids,
            [],
            response_logprob_seed,
            assistant_logprobs=assistant_logprobs,
        )
        response_ids = (
            merge_result.token_ids[-len(response_mask):] if response_mask else []
        )
        prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]

        out = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                response_logprobs[: self.response_length]
                if response_logprobs is not None
                else None
            ),
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        # keep schema consistent with tool_agent_loop
        out.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return out
