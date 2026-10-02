#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporary nuVLA E2E inference + cross-attention dump.

Goal
----
1. 3-camera 448x448 input
2. Qwen3-VL-2B non-quantized BF16
3. VLM reasoning generation
4. Full VLM sequence:
      image/prefill + generated reasoning
5. Extract layer-wise KV:
      4, 9, 14, 19, 24, 28
6. Action Expert trajectory inference
7. Capture Action/State Query -> full VLM KV cross-attention
8. Save trajectory / reasoning / raw attention / heatmaps

IMPORTANT
---------
This is inference only.
No optimizer.
No backward.
No LoRA training.
No Q-LoRA.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nureasoning.nuvla import train as train_mod


# ======================================================================
# CONFIG
# ======================================================================

KV_LAYERS_1BASED = [4, 9, 14, 19, 24, 28]
CAMERAS = ["front", "front_left", "front_right"]

DEFAULT_DATA_ROOT = "/media/HDD/nuR_ds/data/train/part_1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

DEFAULT_RESULT_ROOT = (
    Path("~/lab/nuReasoning/outputs/e2e_attention_tmp")
    .expanduser()
)


# ======================================================================
# UTIL
# ======================================================================

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[VRAM] {tag:<28} "
        f"allocated={allocated:.3f} GiB  "
        f"reserved={reserved:.3f} GiB  "
        f"peak={peak:.3f} GiB"
    )


def tensor_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return x


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ======================================================================
# INFERENCE-ONLY TRAINER
# ======================================================================

class InferenceVLATrainer(train_mod.VLATrainer):
    """
    Reuse the official/local nuVLA model/data construction,
    but NEVER build optimizers/schedulers.
    """

    def _build_optimizers(self):
        self.use_bf16 = True

        # Deliberately no:
        #   AdamW
        #   scheduler
        #   gradients
        self.vlm_optimizer = None
        self.action_optimizer = None
        self.vlm_scheduler = None
        self.action_scheduler = None


# ======================================================================
# CHECKPOINT
# ======================================================================

def find_latest_action_checkpoint() -> Optional[Path]:
    root = Path("~/lab/nuReasoning/outputs").expanduser()

    if not root.exists():
        return None

    candidates = list(root.rglob("action_expert.pt"))

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return candidates[0]


def load_action_checkpoint(
    model: torch.nn.Module,
    path: Optional[Path],
    device: torch.device,
) -> Optional[Path]:

    if path is None:
        path = find_latest_action_checkpoint()

    if path is None:
        print()
        print("=" * 80)
        print("[WARNING] ACTION CHECKPOINT NOT FOUND")
        print("=" * 80)
        print(
            "Action Expert remains randomly initialized.\n"
            "E2E plumbing / attention extraction can be tested,\n"
            "but predicted trajectory and attention are NOT meaningful."
        )
        print()
        return None

    path = path.expanduser().resolve()

    if path.is_dir():
        candidate = path / "action_expert.pt"

        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(
                f"action_expert.pt not found in {path}"
            )

    if not path.exists():
        raise FileNotFoundError(path)

    state = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print("[ACTION CHECKPOINT]")
    print(" path      :", path)
    print(" missing   :", len(missing))
    print(" unexpected:", len(unexpected))

    if missing:
        print(" first missing:", missing[:10])

    if unexpected:
        print(" first unexpected:", unexpected[:10])

    return path


# ======================================================================
# BATCH HELPERS
# ======================================================================

def get_sample_id(
    batch: Dict[str, Any],
    batch_index: int,
    fallback: str,
) -> str:

    for key in (
        "sample_ids",
        "sample_id",
        "tokens",
        "token",
        "ids",
        "id",
        "frame_tokens",
    ):
        if key not in batch:
            continue

        value = batch[key]

        if isinstance(value, (list, tuple)):
            if len(value) > batch_index:
                return str(value[batch_index])

        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] > batch_index:
                return str(value[batch_index].item())

        if isinstance(value, str):
            return value

    return fallback


def build_ego_inputs(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    ego_history = batch[
        "ego_history_trajectories"
    ].to(
        device=device,
        non_blocking=True,
    )

    velocity = batch[
        "ego_velocities"
    ].to(
        device=device,
        non_blocking=True,
    )

    acceleration = batch[
        "ego_accelerations"
    ].to(
        device=device,
        non_blocking=True,
    )

    ego_state = torch.cat(
        [
            velocity,
            acceleration,
        ],
        dim=-1,
    )

    return ego_state, ego_history


# ======================================================================
# VLM PROMPT
# ======================================================================

def prepare_prompt_only_input(
    trainer: InferenceVLATrainer,
    batch: Dict[str, Any],
    batch_idx: int = 0,
) -> Dict[str, Any]:

    vlm = trainer.vlm_unwrapped

    images = batch["images"][batch_idx]
    images = [
        img for img in images
        if img is not None
    ]

    image_contexts = []

    # Current config:
    # history=1 means historical + current images can exist.
    for t in range(
        -trainer.args.num_history_steps,
        1,
    ):
        t_label = (
            f"t={t}"
            if t < 0
            else "t=0 (current)"
        )

        for cam in CAMERAS:
            image_contexts.append(
                f"{t_label}, {cam} camera."
            )

    image_contexts = image_contexts[:len(images)]

    prompts = batch.get("user_prompts")

    prompt = None

    if prompts is not None and len(prompts) > batch_idx:
        prompt = prompts[batch_idx]

    if not prompt:
        prompt = vlm.build_multiview_prompt(
            num_history_steps=trainer.args.num_history_steps,
            num_current_cameras=len(CAMERAS),
            mission_command=batch[
                "mission_commands"
            ][batch_idx],
        )

    # IMPORTANT:
    # assistant_response=None
    # because this stage must GENERATE reasoning.
    prepared = vlm.prepare_inputs(
        images=images,
        text_prompt=prompt,
        image_contexts=image_contexts,
        assistant_response=None,
    )

    result = {}

    for key, value in prepared.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(trainer.device)
        else:
            result[key] = value

    return result


# ======================================================================
# GENERATE REASONING
# ======================================================================

def get_underlying_generation_model(vlm):
    """
    VLMBackbone stores HF model in .model in current nuVLA code.
    """
    if hasattr(vlm, "model"):
        return vlm.model

    raise AttributeError(
        "Could not find underlying HF model at vlm.model"
    )


@torch.inference_mode()
def generate_reasoning(
    trainer: InferenceVLATrainer,
    prompt_inputs: Dict[str, Any],
    max_new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:

    vlm = trainer.vlm_unwrapped
    model = get_underlying_generation_model(vlm)

    model.eval()

    input_ids = prompt_inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    generate_kwargs = {
        "input_ids":
            input_ids,

        "attention_mask":
            prompt_inputs["attention_mask"],

        "max_new_tokens":
            max_new_tokens,

        "do_sample":
            False,

        "use_cache":
            True,
    }

    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        if (
            key in prompt_inputs
            and prompt_inputs[key] is not None
        ):
            generate_kwargs[key] = prompt_inputs[key]

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        generated = model.generate(
            **generate_kwargs
        )

    reasoning_ids = generated[:, prompt_len:]

    tokenizer = getattr(
        vlm,
        "tokenizer",
        None,
    )

    if tokenizer is None:
        processor = getattr(
            vlm,
            "processor",
            None,
        )

        if processor is not None:
            tokenizer = getattr(
                processor,
                "tokenizer",
                None,
            )

    if tokenizer is None:
        raise RuntimeError(
            "Could not locate tokenizer "
            "from VLMBackbone."
        )

    reasoning_text = tokenizer.decode(
        reasoning_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return (
        generated,
        reasoning_ids,
        reasoning_text,
    )


# ======================================================================
# FULL SEQUENCE INPUT FOR KV EXTRACTION
# ======================================================================

def build_full_sequence_inputs(
    prompt_inputs: Dict[str, Any],
    full_input_ids: torch.Tensor,
) -> Dict[str, Any]:

    result = {}

    prompt_len = int(
        prompt_inputs["input_ids"].shape[1]
    )

    full_len = int(
        full_input_ids.shape[1]
    )

    extra = full_len - prompt_len

    result["input_ids"] = full_input_ids

    result["attention_mask"] = torch.ones(
        full_input_ids.shape,
        dtype=prompt_inputs[
            "attention_mask"
        ].dtype,
        device=full_input_ids.device,
    )

    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        if key in prompt_inputs:
            result[key] = prompt_inputs[key]

    # Qwen3-VL mm token type ids:
    # generated reasoning tokens are normal text tokens.
    if (
        "mm_token_type_ids"
        in prompt_inputs
        and prompt_inputs[
            "mm_token_type_ids"
        ] is not None
    ):
        mm = prompt_inputs[
            "mm_token_type_ids"
        ]

        if extra > 0:
            extra_types = torch.zeros(
                (
                    mm.shape[0],
                    extra,
                ),
                dtype=mm.dtype,
                device=mm.device,
            )

            mm = torch.cat(
                [
                    mm,
                    extra_types,
                ],
                dim=1,
            )

        result[
            "mm_token_type_ids"
        ] = mm

    # Wrapper uses this to know the prompt/reasoning boundary.
    result["prompt_lengths"] = torch.tensor(
        [prompt_len],
        dtype=torch.long,
        device=full_input_ids.device,
    )

    return result


# ======================================================================
# EXTRACT CURRENT LOCAL VLM OUTPUT
# ======================================================================

@torch.inference_mode()
def extract_vlm_output(
    trainer: InferenceVLATrainer,
    full_inputs: Dict[str, Any],
) -> Dict[str, Any]:

    vlm = trainer.vlm_unwrapped
    vlm.eval()

    kwargs = {
        "input_ids":
            full_inputs["input_ids"],

        "attention_mask":
            full_inputs["attention_mask"],

        "pixel_values":
            full_inputs.get(
                "pixel_values"
            ),

        "image_grid_thw":
            full_inputs.get(
                "image_grid_thw"
            ),

        "mm_token_type_ids":
            full_inputs.get(
                "mm_token_type_ids"
            ),

        "labels":
            None,

        "prompt_lengths":
            full_inputs[
                "prompt_lengths"
            ],

        "return_features":
            True,
    }

    # Current modified version may support these.
    sig = inspect.signature(
        vlm.forward
    )

    if "return_kv" in sig.parameters:
        kwargs["return_kv"] = True

    if "output_kv" in sig.parameters:
        kwargs["output_kv"] = True

    if "kv_layers" in sig.parameters:
        kwargs["kv_layers"] = (
            KV_LAYERS_1BASED
        )

    if "selected_kv_layers" in sig.parameters:
        kwargs["selected_kv_layers"] = [
            layer - 1
            for layer in KV_LAYERS_1BASED
        ]


    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in sig.parameters
    }

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        outputs = vlm(**kwargs)

    if isinstance(outputs, dict):
        return outputs

    # transformers ModelOutput
    if hasattr(outputs, "items"):
        return dict(outputs.items())

    raise RuntimeError(
        f"Unexpected VLM output type: "
        f"{type(outputs)}"
    )


def find_layer_kv(
    outputs: Dict[str, Any],
) -> Any:
    """
    Support several local names because the working tree
    was modified during the current experiment.
    """

    candidates = (
        "layer_kv",
        "layerwise_kv",
        "vlm_kv",
        "reasoning_kv",
        "kv_cache",
        "past_key_values",
    )

    for key in candidates:
        if key in outputs:
            print(
                f"[VLM] using KV output key: {key}"
            )
            return outputs[key]

    print(
        "[VLM] available output keys:",
        list(outputs.keys()),
    )

    raise KeyError(
        "No KV output found from VLMBackbone."
    )


# ======================================================================
# CROSS ATTENTION RECORDER
# ======================================================================

class CrossAttentionRecorder:

    def __init__(
        self,
        action_expert: torch.nn.Module,
        num_waypoints: int,
    ):
        self.model = action_expert
        self.num_waypoints = int(
            num_waypoints
        )

        self.records = []
        self.handles = []

    @staticmethod
    def _looks_like_cross_attention(
        name: str,
        module: torch.nn.Module,
    ) -> bool:

        n = name.lower()
        c = module.__class__.__name__.lower()

        has_q = hasattr(
            module,
            "q_proj",
        )

        text_hit = any(
            token in n or token in c
            for token in (
                "cross",
                "fusion",
                "rawkv",
                "kvattention",
                "kv_attention",
            )
        )

        return bool(
            has_q and text_hit
        )

    @staticmethod
    def _reshape_q(
        module,
        q: torch.Tensor,
    ) -> torch.Tensor:

        if q.ndim == 4:
            return q

        if q.ndim != 3:
            raise ValueError(
                f"Unexpected Q shape {q.shape}"
            )

        b, nq, total = q.shape

        num_heads = getattr(
            module,
            "num_heads",
            None,
        )

        if num_heads is None:
            num_heads = getattr(
                module,
                "num_attention_heads",
                None,
            )

        if num_heads is None:
            num_heads = getattr(
                module,
                "kv_num_heads",
                None,
            )

        head_dim = getattr(
            module,
            "head_dim",
            None,
        )

        if head_dim is None:
            head_dim = getattr(
                module,
                "kv_head_dim",
                None,
            )

        if (
            num_heads is None
            and head_dim is not None
        ):
            num_heads = (
                total // int(head_dim)
            )

        if (
            head_dim is None
            and num_heads is not None
        ):
            head_dim = (
                total // int(num_heads)
            )

        if (
            num_heads is None
            or head_dim is None
        ):
            raise RuntimeError(
                "Cannot infer query heads from "
                f"{module.__class__.__name__}"
            )

        q = q.view(
            b,
            nq,
            int(num_heads),
            int(head_dim),
        )

        return q.transpose(
            1,
            2,
        ).contiguous()

    @staticmethod
    def _reshape_k(
        k: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:

        if k.ndim == 4:
            # Expected:
            # [B, H, S, D]
            if k.shape[-1] == head_dim:
                return k

            # possible [B,S,H,D]
            if k.shape[-1] == head_dim:
                return k.transpose(
                    1,
                    2,
                )

        if k.ndim == 3:
            b, seq, total = k.shape

            if total % head_dim != 0:
                raise ValueError(
                    f"K hidden dim {total} "
                    f"not divisible by {head_dim}"
                )

            heads = total // head_dim

            return (
                k.view(
                    b,
                    seq,
                    heads,
                    head_dim,
                )
                .transpose(1, 2)
                .contiguous()
            )

        raise ValueError(
            f"Unexpected K shape {tuple(k.shape)}"
        )

    @staticmethod
    def _repeat_kv(
        k: torch.Tensor,
        target_heads: int,
    ) -> torch.Tensor:

        kv_heads = int(
            k.shape[1]
        )

        if kv_heads == target_heads:
            return k

        if target_heads % kv_heads != 0:
            raise ValueError(
                f"Q heads={target_heads}, "
                f"KV heads={kv_heads}"
            )

        groups = (
            target_heads // kv_heads
        )

        return k.repeat_interleave(
            groups,
            dim=1,
        )

    @staticmethod
    def _find_external_key(
        args,
        kwargs,
    ) -> Optional[torch.Tensor]:

        for name in (
            "vlm_key",
            "vlm_k",
            "key",
            "k",
            "memory_key",
        ):
            value = kwargs.get(name)

            if (
                isinstance(
                    value,
                    torch.Tensor,
                )
                and value.ndim >= 3
            ):
                return value

        # Raw-KV style module normally:
        #   forward(tokens, vlm_key, vlm_value, mask)
        if len(args) >= 2:
            candidate = args[1]

            if (
                isinstance(
                    candidate,
                    torch.Tensor,
                )
                and candidate.ndim >= 3
            ):
                return candidate

        return None

    def _make_hook(
        self,
        module_name: str,
    ):

        def hook(
            module,
            args,
            kwargs,
        ):
            try:
                if len(args) == 0:
                    return

                query_input = args[0]

                if not isinstance(
                    query_input,
                    torch.Tensor,
                ):
                    return

                vlm_key = (
                    self._find_external_key(
                        args,
                        kwargs,
                    )
                )

                if vlm_key is None:
                    return

                with torch.no_grad():

                    q_linear = module.q_proj(
                        query_input
                    )

                    q = self._reshape_q(
                        module,
                        q_linear,
                    )

                    head_dim = int(
                        q.shape[-1]
                    )

                    k = self._reshape_k(
                        vlm_key,
                        head_dim,
                    )

                    k = self._repeat_kv(
                        k,
                        int(q.shape[1]),
                    )

                    scores = torch.matmul(
                        q.float(),
                        k.float().transpose(
                            -2,
                            -1,
                        ),
                    )

                    scores = scores / math.sqrt(
                        head_dim
                    )

                    attn = torch.softmax(
                        scores,
                        dim=-1,
                    )

                    # average across attention heads
                    mean_attn = attn.mean(
                        dim=1
                    )

                    q_len = int(
                        mean_attn.shape[-2]
                    )

                    # nuVLA query convention:
                    # final N positions correspond to
                    # trajectory/action tokens.
                    action_count = min(
                        self.num_waypoints,
                        q_len,
                    )

                    state_count = (
                        q_len - action_count
                    )

                    record = {
                        "module":
                            module_name,

                        "q_len":
                            q_len,

                        "kv_len":
                            int(
                                mean_attn.shape[-1]
                            ),

                        "state_query_count":
                            state_count,

                        "action_query_count":
                            action_count,

                        "full_attention":
                            mean_attn[
                                0
                            ].detach().cpu(),

                        "state_attention":
                            (
                                mean_attn[
                                    0,
                                    :state_count,
                                ]
                                .detach()
                                .cpu()
                                if state_count > 0
                                else None
                            ),

                        "action_attention":
                            mean_attn[
                                0,
                                state_count:,
                            ].detach().cpu(),
                    }

                    self.records.append(
                        record
                    )

            except Exception as exc:
                print(
                    "[ATTN HOOK WARNING]",
                    module_name,
                    repr(exc),
                )

        return hook

    def register(self) -> None:

        print()
        print("=" * 80)
        print("REGISTER CROSS-ATTENTION HOOKS")
        print("=" * 80)

        for name, module in (
            self.model.named_modules()
        ):
            if not self._looks_like_cross_attention(
                name,
                module,
            ):
                continue

            print(
                "[HOOK]",
                name,
                module.__class__.__name__,
            )

            handle = (
                module.register_forward_pre_hook(
                    self._make_hook(name),
                    with_kwargs=True,
                )
            )

            self.handles.append(
                handle
            )

        if not self.handles:
            print(
                "[WARNING] No cross-attention "
                "module matched automatically."
            )

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

        self.handles.clear()


# ======================================================================
# ACTION SAMPLE - SIGNATURE ADAPTER
# ======================================================================

@torch.inference_mode()
def run_action_sample(
    action_expert: torch.nn.Module,
    layer_kv: Any,
    vlm_outputs: Dict[str, Any],
    ego_state: torch.Tensor,
    ego_history: torch.Tensor,
) -> torch.Tensor:

    action_expert.eval()

    sample_fn = action_expert.sample

    sig = inspect.signature(
        sample_fn
    )

    params = sig.parameters

    kwargs = {}

    # --------------------------------------------------------------
    # KV-aware current implementation
    # --------------------------------------------------------------

    for name in (
        "layer_kv",
        "layerwise_kv",
        "vlm_kv",
        "reasoning_kv",
        "kv_cache",
    ):
        if name in params:
            kwargs[name] = layer_kv
            break

    # --------------------------------------------------------------
    # Older feature-based implementation fallback
    # --------------------------------------------------------------

    if "vlm_features" in params:
        if "vlm_features" not in vlm_outputs:
            raise KeyError(
                "sample() requires vlm_features "
                "but VLM output has none."
            )

        kwargs[
            "vlm_features"
        ] = vlm_outputs[
            "vlm_features"
        ]

    # --------------------------------------------------------------
    # State / history
    # --------------------------------------------------------------

    for name in (
        "ego_state",
        "state",
    ):
        if name in params:
            kwargs[name] = ego_state
            break

    for name in (
        "history_trajectory",
        "ego_history_trajectory",
        "ego_history",
    ):
        if name in params:
            kwargs[name] = ego_history
            break

    print()
    print("[ACTION sample signature]")
    print(sig)
    print(
        "[ACTION kwargs]",
        list(kwargs.keys()),
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        try:
            pred = sample_fn(
                **kwargs
            )

        except TypeError:
            # Known older nuVLA signature:
            # sample(vlm_features, ego_state, history)
            if "vlm_features" in vlm_outputs:
                pred = sample_fn(
                    vlm_outputs[
                        "vlm_features"
                    ],
                    ego_state,
                    ego_history,
                )
            else:
                raise

    return pred


# ======================================================================
# TOKEN REGIONS
# ======================================================================

def get_vlm_tokenizer(
    vlm,
):
    """
    Resolve the tokenizer from the local nuVLA VLM wrapper.
    """

    tokenizer = getattr(
        vlm,
        "tokenizer",
        None,
    )

    if tokenizer is not None:
        return tokenizer

    processor = getattr(
        vlm,
        "processor",
        None,
    )

    if processor is not None:
        tokenizer = getattr(
            processor,
            "tokenizer",
            None,
        )

    if tokenizer is None:
        raise RuntimeError(
            "Could not locate tokenizer from VLMBackbone."
        )

    return tokenizer


def _valid_token_id(
    token_id: Any,
) -> bool:
    """
    Hugging Face tokenizers may return None or unk_token_id for a token
    that does not exist. Treat those cases as invalid.
    """

    if token_id is None:
        return False

    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return False

    return token_id >= 0


def _find_contiguous_runs(
    ids: List[int],
    target_id: int,
) -> List[Tuple[int, int]]:
    """
    Return [start, end) runs where ids == target_id.
    """

    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for idx, token_id in enumerate(ids):
        if token_id == target_id:
            if start is None:
                start = idx
        else:
            if start is not None:
                runs.append(
                    (start, idx)
                )
                start = None

    if start is not None:
        runs.append(
            (start, len(ids))
        )

    return runs


def _find_vision_marker_spans(
    ids: List[int],
    vision_start_id: Optional[int],
    vision_end_id: Optional[int],
) -> List[Tuple[int, int]]:
    """
    Fallback image-span detector based on
    <|vision_start|> ... <|vision_end|>.
    """

    if (
        not _valid_token_id(vision_start_id)
        or not _valid_token_id(vision_end_id)
    ):
        return []

    spans: List[Tuple[int, int]] = []
    open_start: Optional[int] = None

    for idx, token_id in enumerate(ids):
        if (
            token_id == int(vision_start_id)
            and open_start is None
        ):
            open_start = idx

        elif (
            token_id == int(vision_end_id)
            and open_start is not None
        ):
            spans.append(
                (open_start, idx + 1)
            )
            open_start = None

    return spans


def _complement_spans(
    start: int,
    end: int,
    excluded: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Complement of excluded [start, end) intervals.
    Used to identify prompt-text spans after image spans are removed.
    """

    clipped: List[Tuple[int, int]] = []

    for s, e in excluded:
        s = max(start, min(int(s), end))
        e = max(start, min(int(e), end))

        if e > s:
            clipped.append(
                (s, e)
            )

    clipped.sort()

    merged: List[Tuple[int, int]] = []

    for s, e in clipped:
        if (
            not merged
            or s > merged[-1][1]
        ):
            merged.append(
                (s, e)
            )
        else:
            merged[-1] = (
                merged[-1][0],
                max(
                    merged[-1][1],
                    e,
                ),
            )

    result: List[Tuple[int, int]] = []
    cursor = start

    for s, e in merged:
        if s > cursor:
            result.append(
                (cursor, s)
            )

        cursor = max(
            cursor,
            e,
        )

    if cursor < end:
        result.append(
            (cursor, end)
        )

    return result


def infer_token_regions(
    prompt_inputs: Dict[str, Any],
    reasoning_ids: torch.Tensor,
    tokenizer,
    image_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Recover the actual token layout used by the VLM KV sequence.

    Important:
      - Images are part of the prompt/prefill sequence.
      - Reasoning starts exactly at prompt_len.
      - Qwen3-VL image spans are detected from <|image_pad|> runs.
      - If image_pad cannot be found, vision_start/vision_end is used
        as a fallback.

    Returned spans use [start, end) indexing.
    """

    prompt_ids_tensor = prompt_inputs[
        "input_ids"
    ]

    prompt_len = int(
        prompt_ids_tensor.shape[1]
    )

    reasoning_len = int(
        reasoning_ids.shape[1]
    )

    total_len = (
        prompt_len + reasoning_len
    )

    prompt_ids = (
        prompt_ids_tensor[
            0
        ]
        .detach()
        .cpu()
        .tolist()
    )

    if image_labels is None:
        image_labels = [
            "t=-1s FRONT",
            "t=-1s FRONT_LEFT",
            "t=-1s FRONT_RIGHT",
            "t=0s FRONT",
            "t=0s FRONT_LEFT",
            "t=0s FRONT_RIGHT",
        ]

    image_pad_id = (
        tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"
        )
    )

    vision_start_id = (
        tokenizer.convert_tokens_to_ids(
            "<|vision_start|>"
        )
    )

    vision_end_id = (
        tokenizer.convert_tokens_to_ids(
            "<|vision_end|>"
        )
    )

    image_spans: List[Tuple[int, int]] = []

    # --------------------------------------------------------------
    # Preferred Qwen3-VL path:
    # contiguous <|image_pad|> runs correspond to visual-token blocks.
    # Expand each block by adjacent vision_start / vision_end markers
    # so the annotation covers the complete visual segment.
    # --------------------------------------------------------------
    if _valid_token_id(
        image_pad_id
    ):
        pad_runs = (
            _find_contiguous_runs(
                prompt_ids,
                int(image_pad_id),
            )
        )

        for s, e in pad_runs:
            expanded_s = s
            expanded_e = e

            if (
                s > 0
                and _valid_token_id(
                    vision_start_id
                )
                and prompt_ids[
                    s - 1
                ] == int(
                    vision_start_id
                )
            ):
                expanded_s = s - 1

            if (
                e < prompt_len
                and _valid_token_id(
                    vision_end_id
                )
                and prompt_ids[
                    e
                ] == int(
                    vision_end_id
                )
            ):
                expanded_e = e + 1

            image_spans.append(
                (
                    expanded_s,
                    expanded_e,
                )
            )

    # --------------------------------------------------------------
    # Fallback:
    # explicit vision_start ... vision_end pairs.
    # --------------------------------------------------------------
    if not image_spans:
        image_spans = (
            _find_vision_marker_spans(
                prompt_ids,
                vision_start_id,
                vision_end_id,
            )
        )

    image_regions: List[Dict[str, Any]] = []

    for idx, (s, e) in enumerate(
        image_spans
    ):
        if idx < len(
            image_labels
        ):
            label = image_labels[idx]
        else:
            label = (
                f"IMAGE_{idx + 1}"
            )

        image_regions.append(
            {
                "index": idx,
                "label": label,
                "start": int(s),
                "end": int(e),
                "num_tokens": int(
                    e - s
                ),
            }
        )

    prompt_text_spans = (
        _complement_spans(
            0,
            prompt_len,
            image_spans,
        )
    )

    info: Dict[str, Any] = {
        # Entire multimodal prefill.
        "prefill": [
            0,
            prompt_len,
        ],

        # Text-only pieces inside the prefill after image blocks
        # are excluded.
        "prompt_text_spans": [
            [
                int(s),
                int(e),
            ]
            for s, e
            in prompt_text_spans
        ],

        # Individual visual blocks.
        "images":
            image_regions,

        # Generated reasoning appended after the prompt.
        "reasoning": [
            prompt_len,
            total_len,
        ],

        "prompt_tokens":
            prompt_len,

        "reasoning_tokens":
            reasoning_len,

        "total_tokens":
            total_len,

        "detected_image_count":
            len(
                image_regions
            ),

        "expected_image_count":
            len(
                image_labels
            ),

        "special_token_ids": {
            "image_pad":
                int(image_pad_id)
                if _valid_token_id(
                    image_pad_id
                )
                else None,

            "vision_start":
                int(vision_start_id)
                if _valid_token_id(
                    vision_start_id
                )
                else None,

            "vision_end":
                int(vision_end_id)
                if _valid_token_id(
                    vision_end_id
                )
                else None,
        },
    }

    if (
        "image_grid_thw"
        in prompt_inputs
        and prompt_inputs[
            "image_grid_thw"
        ] is not None
    ):
        info[
            "image_grid_thw"
        ] = (
            prompt_inputs[
                "image_grid_thw"
            ]
            .detach()
            .cpu()
            .tolist()
        )

    print()
    print(
        "[TOKEN REGIONS]"
    )
    print(
        "  prefill      :",
        info["prefill"],
    )
    print(
        "  prompt text  :",
        info[
            "prompt_text_spans"
        ],
    )
    print(
        "  reasoning    :",
        info["reasoning"],
    )
    print(
        "  images       :",
        len(
            image_regions
        ),
        "/ expected",
        len(
            image_labels
        ),
    )

    for region in image_regions:
        print(
            "   -",
            region["label"],
            f"[{region['start']}, "
            f"{region['end']})",
            f"{region['num_tokens']} tokens",
        )

    if (
        len(image_regions)
        != len(image_labels)
    ):
        print(
            "[TOKEN REGION WARNING] "
            "Detected image-span count does not match "
            "the expected 3 cameras x 2 timestamps."
        )

    return info


# ======================================================================
# SAVE ATTENTION
# ======================================================================

def _clip_span(
    span: Tuple[int, int],
    kv_len: int,
) -> Optional[Tuple[int, int]]:
    """
    Clip a [start, end) token span to the actual attention KV length.
    """

    s = max(
        0,
        min(
            int(span[0]),
            kv_len,
        ),
    )

    e = max(
        0,
        min(
            int(span[1]),
            kv_len,
        ),
    )

    if e <= s:
        return None

    return (
        s,
        e,
    )


def _draw_region_bar(
    ax,
    token_regions: Dict[str, Any],
    kv_len: int,
) -> None:
    """
    Draw a token-layout bar directly below the attention heatmap.

    Top row:
        PREFILL | REASONING

    Bottom row:
        PROMPT TEXT fragments | six IMAGE blocks | REASONING
    """

    from matplotlib.patches import Rectangle

    # Standard, visually distinct colors.
    prompt_color = "#4C78A8"
    image_color = "#F58518"
    reasoning_color = "#54A24B"
    prefill_color = "#9ECAE9"

    ax.set_xlim(
        -0.5,
        max(
            kv_len - 0.5,
            0.5,
        ),
    )

    ax.set_ylim(
        -0.05,
        2.0,
    )

    ax.set_yticks(
        [
            0.45,
            1.35,
        ]
    )

    ax.set_yticklabels(
        [
            "Detail",
            "Major",
        ]
    )

    ax.set_xlabel(
        "VLM KV token index"
    )

    ax.tick_params(
        axis="y",
        length=0,
    )

    # --------------------------------------------------------------
    # Major row: whole prefill vs generated reasoning
    # --------------------------------------------------------------
    prefill = token_regions.get(
        "prefill",
        [
            0,
            0,
        ],
    )

    clipped = _clip_span(
        (
            prefill[0],
            prefill[1],
        ),
        kv_len,
    )

    if clipped is not None:
        s, e = clipped

        ax.add_patch(
            Rectangle(
                (
                    s - 0.5,
                    1.05,
                ),
                e - s,
                0.55,
                facecolor=prefill_color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )
        )

        if (
            e - s
        ) >= 12:
            ax.text(
                (
                    s + e - 1
                ) / 2,
                1.325,
                f"PREFILL / PROMPT  [{s}:{e}]",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    reasoning = token_regions.get(
        "reasoning",
        [
            0,
            0,
        ],
    )

    clipped_reasoning = _clip_span(
        (
            reasoning[0],
            reasoning[1],
        ),
        kv_len,
    )

    if clipped_reasoning is not None:
        s, e = clipped_reasoning

        ax.add_patch(
            Rectangle(
                (
                    s - 0.5,
                    1.05,
                ),
                e - s,
                0.55,
                facecolor=reasoning_color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )
        )

        if (
            e - s
        ) >= 8:
            ax.text(
                (
                    s + e - 1
                ) / 2,
                1.325,
                f"REASONING  [{s}:{e}]",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    # --------------------------------------------------------------
    # Detail row: prompt-text pieces
    # --------------------------------------------------------------
    for span in token_regions.get(
        "prompt_text_spans",
        [],
    ):
        clipped = _clip_span(
            (
                span[0],
                span[1],
            ),
            kv_len,
        )

        if clipped is None:
            continue

        s, e = clipped

        ax.add_patch(
            Rectangle(
                (
                    s - 0.5,
                    0.15,
                ),
                e - s,
                0.55,
                facecolor=prompt_color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
        )

        if (
            e - s
        ) >= 35:
            ax.text(
                (
                    s + e - 1
                ) / 2,
                0.425,
                "PROMPT TEXT",
                ha="center",
                va="center",
                fontsize=7,
            )

    # --------------------------------------------------------------
    # Detail row: each image block
    # --------------------------------------------------------------
    for image_region in token_regions.get(
        "images",
        [],
    ):
        clipped = _clip_span(
            (
                image_region[
                    "start"
                ],
                image_region[
                    "end"
                ],
            ),
            kv_len,
        )

        if clipped is None:
            continue

        s, e = clipped

        ax.add_patch(
            Rectangle(
                (
                    s - 0.5,
                    0.15,
                ),
                e - s,
                0.55,
                facecolor=image_color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
        )

        label = image_region.get(
            "label",
            "IMAGE",
        )

        if (
            e - s
        ) >= 25:
            ax.text(
                (
                    s + e - 1
                ) / 2,
                0.425,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
            )

    # --------------------------------------------------------------
    # Detail row: reasoning
    # --------------------------------------------------------------
    if clipped_reasoning is not None:
        s, e = clipped_reasoning

        ax.add_patch(
            Rectangle(
                (
                    s - 0.5,
                    0.15,
                ),
                e - s,
                0.55,
                facecolor=reasoning_color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
        )

        if (
            e - s
        ) >= 8:
            ax.text(
                (
                    s + e - 1
                ) / 2,
                0.425,
                "REASONING",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )

    # --------------------------------------------------------------
    # Vertical boundaries for readability.
    # --------------------------------------------------------------
    reasoning_start = int(
        reasoning[0]
    )

    if (
        0 < reasoning_start < kv_len
    ):
        ax.axvline(
            reasoning_start - 0.5,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )

    ax.grid(
        axis="x",
        alpha=0.15,
    )


def plot_attention(
    arr: np.ndarray,
    path: Path,
    title: str,
    token_regions: Dict[str, Any],
) -> None:

    if arr.size == 0:
        return

    if arr.ndim != 2:
        raise ValueError(
            "Attention plot expects a 2D array, "
            f"got shape={arr.shape}"
        )

    kv_len = int(
        arr.shape[-1]
    )

    fig = plt.figure(
        figsize=(
            18,
            7.5,
        )
    )

    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[
            5.5,
            1.5,
        ],
        hspace=0.12,
    )

    ax = fig.add_subplot(
        grid[0]
    )

    region_ax = fig.add_subplot(
        grid[1],
        sharex=ax,
    )

    im = ax.imshow(
        arr,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_ylabel(
        "Action/State query index"
    )

    ax.set_title(
        title
    )

    # x labels live on the token-region bar below.
    ax.tick_params(
        axis="x",
        labelbottom=False,
    )

    # --------------------------------------------------------------
    # Show reasoning boundary on the heatmap itself.
    # --------------------------------------------------------------
    reasoning = token_regions.get(
        "reasoning",
        [
            kv_len,
            kv_len,
        ],
    )

    reasoning_start = min(
        int(
            reasoning[0]
        ),
        kv_len,
    )

    if (
        0 < reasoning_start < kv_len
    ):
        ax.axvline(
            reasoning_start - 0.5,
            color="white",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )

        ax.text(
            reasoning_start,
            1.01,
            f"reasoning start = {reasoning_start}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    _draw_region_bar(
        region_ax,
        token_regions,
        kv_len,
    )

    fig.colorbar(
        im,
        ax=[
            ax,
            region_ax,
        ],
        fraction=0.022,
        pad=0.015,
        label="Attention weight",
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.93,
        top=0.92,
        bottom=0.10,
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_attention_records(
    records: List[Dict[str, Any]],
    sample_dir: Path,
    token_regions: Dict[str, Any],
) -> None:

    attn_root = (
        sample_dir
        / "attention"
    )

    attn_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    reasoning_start = (
        token_regions[
            "reasoning"
        ][0]
    )

    reasoning_end = (
        token_regions[
            "reasoning"
        ][1]
    )

    summary = []

    for idx, rec in enumerate(records):

        module_dir = (
            attn_root
            / f"{idx:03d}_{rec['module'].replace('.', '_')}"
        )

        module_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        full = rec[
            "full_attention"
        ].numpy()

        np.save(
            module_dir
            / "full_query_to_vlm.npy",
            full,
        )

        plot_attention(
            full,
            module_dir
            / "full_query_to_vlm.png",
            rec["module"],
            token_regions,
        )

        action = rec[
            "action_attention"
        ]

        state = rec[
            "state_attention"
        ]

        action_reasoning_ratio = None
        state_reasoning_ratio = None

        # Matrix token dimension can differ slightly from input token
        # dimension for some KV-cache implementations.
        kv_len = full.shape[-1]

        rs = min(
            reasoning_start,
            kv_len,
        )

        re = min(
            reasoning_end,
            kv_len,
        )

        if action is not None:
            action_np = action.numpy()

            np.save(
                module_dir
                / "action_query_to_vlm.npy",
                action_np,
            )

            plot_attention(
                action_np,
                module_dir
                / "action_query_to_vlm.png",
                rec["module"]
                + " | Action Query",
                token_regions,
            )

            if re > rs:
                action_reasoning_ratio = float(
                    action[
                        :,
                        rs:re,
                    ]
                    .sum(dim=-1)
                    .mean()
                    .item()
                )

        if state is not None:
            state_np = state.numpy()

            np.save(
                module_dir
                / "state_query_to_vlm.npy",
                state_np,
            )

            plot_attention(
                state_np,
                module_dir
                / "state_query_to_vlm.png",
                rec["module"]
                + " | State Query",
                token_regions,
            )

            if re > rs:
                state_reasoning_ratio = float(
                    state[
                        :,
                        rs:re,
                    ]
                    .sum(dim=-1)
                    .mean()
                    .item()
                )

        summary.append(
            {
                "module":
                    rec["module"],

                "q_len":
                    rec["q_len"],

                "kv_len":
                    rec["kv_len"],

                "state_query_count":
                    rec[
                        "state_query_count"
                    ],

                "action_query_count":
                    rec[
                        "action_query_count"
                    ],

                "state_to_reasoning_attention":
                    state_reasoning_ratio,

                "action_to_reasoning_attention":
                    action_reasoning_ratio,
            }
        )

    save_json(
        sample_dir
        / "attention_summary.json",
        summary,
    )


# ======================================================================
# CLI
# ======================================================================

def parse_args():

    p = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    p.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
    )

    p.add_argument(
        "--vlm-model",
        default=DEFAULT_MODEL,
    )

    p.add_argument(
        "--action-checkpoint",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
    )

    p.add_argument(
        "--num-samples",
        type=int,
        default=1,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


# ======================================================================
# BUILD nuVLA ARGS
# ======================================================================

def make_nuvla_args(args):

    # Use the original parser defaults safely without
    # parsing this temporary script's CLI.
    p = argparse.Namespace()

    p.data_root = args.data_root
    p.test_data_root = ""

    p.vlm_model_path = args.vlm_model

    p.output_dir = str(
        args.result_root
    )

    p.batch_size = 1
    p.gradient_accumulation_steps = 1
    p.epochs = 1
    p.num_workers = 0

    p.vlm_lr = 0.0
    p.action_lr = 0.0
    p.weight_decay = 0.0
    p.max_grad_norm = 0.0
    p.warmup_ratio = 0.0
    p.action_loss_weight = 1.0

    # ----------------------------------------------------------
    # VLM
    # ----------------------------------------------------------

    p.freeze_vision_encoder = True

    # No LoRA adapter creation.
    # Pure pretrained BF16 inference.
    p.lora_rank = 0
    p.lora_alpha = 32
    p.lora_dropout = 0.0

    # Current modified VLM config supports this.
    # FALSE = NO 4-bit quantization.
    p.load_in_4bit = False

    p.reasoning_mode = "structured"

    p.reasoning_format = (
        "spatial_driving_counterfactual"
    )

    p.reasoning_max_items_per_list = 10
    p.vqa_root = None
    p.qa_seed = 42

    # exactly three cameras
    p.cameras = (
        "front,front_left,front_right"
    )

    # ----------------------------------------------------------
    # Action Expert
    # ----------------------------------------------------------

    p.action_hidden_dim = 512
    p.num_dit_layers = 12
    p.num_dit_heads = 8

    p.kv_layers = "4,9,14,19,24,28"

    p.dropout = 0.1
    p.mlp_ratio = 4.0

    p.interleave_self_attention = True

    p.num_inference_steps = 5
    p.num_timestep_buckets = 1000

    p.noise_beta_alpha = 2.5
    p.noise_beta_beta = 1.5

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------

    p.num_history_steps = 1
    p.history_stride = 10

    p.current_res_w = 448
    p.current_res_h = 448

    p.history_res_w = 448
    p.history_res_h = 448

    p.trajectory_future_seconds = 5.0
    p.frame_rate_hz = 10.0

    p.num_trajectory_points = 10
    p.max_history_traj_points = 6

    p.log_interval = 1
    p.eval_interval = 999
    p.save_interval = 999
    p.resume_from = None

    return p


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required."
        )

    torch.cuda.set_device(
        args.gpu_id
    )

    device = torch.device(
        f"cuda:{args.gpu_id}"
    )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    args.result_root = (
        args.result_root
        .expanduser()
        .resolve()
    )

    args.result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("NUVLA E2E INFERENCE + ATTENTION")
    print("=" * 80)

    print(
        "GPU:",
        torch.cuda.get_device_name(
            args.gpu_id
        ),
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )

    print(
        "Quantization: OFF"
    )

    print(
        "Cameras:",
        CAMERAS,
    )

    print(
        "Resolution: 448 x 448"
    )

    print(
        "KV layers:",
        KV_LAYERS_1BASED,
    )

    torch.cuda.reset_peak_memory_stats()

    nuvla_args = make_nuvla_args(
        args
    )

    # ----------------------------------------------------------
    # Build original nuVLA
    # ----------------------------------------------------------

    trainer = InferenceVLATrainer(
        nuvla_args,
        local_rank=args.gpu_id,
        global_rank=0,
        world_size=1,
    )

    vlm = trainer.vlm_unwrapped
    action = trainer.action_expert_unwrapped

    # absolutely no gradients
    vlm.eval()
    action.eval()

    for p in vlm.parameters():
        p.requires_grad_(False)

    for p in action.parameters():
        p.requires_grad_(False)

    cuda_mem(
        "models loaded"
    )

    # ----------------------------------------------------------
    # Load Action Expert checkpoint
    # ----------------------------------------------------------

    loaded_action_path = (
        load_action_checkpoint(
            action,
            args.action_checkpoint,
            device,
        )
    )

    # ----------------------------------------------------------
    # Attention hooks
    # ----------------------------------------------------------

    recorder = CrossAttentionRecorder(
        action,
        num_waypoints=(
            nuvla_args.num_trajectory_points
        ),
    )

    recorder.register()

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------

    loader = trainer.train_loader

    processed = 0

    with torch.inference_mode():

        for batch_index, batch in enumerate(
            loader
        ):

            if (
                processed
                >= args.num_samples
            ):
                break

            recorder.records.clear()

            sample_id = get_sample_id(
                batch,
                0,
                f"sample_{processed:06d}",
            )

            safe_id = (
                sample_id
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )

            sample_dir = (
                args.result_root
                / safe_id
            )

            sample_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print()
            print("=" * 80)
            print(
                f"SAMPLE {processed}:",
                sample_id,
            )
            print("=" * 80)

            # ==================================================
            # 1. Prepare multimodal prompt
            # ==================================================

            prompt_inputs = (
                prepare_prompt_only_input(
                    trainer,
                    batch,
                    batch_idx=0,
                )
            )

            print(
                "[INPUT] prompt tokens:",
                tuple(
                    prompt_inputs[
                        "input_ids"
                    ].shape
                ),
            )

            if (
                "image_grid_thw"
                in prompt_inputs
            ):
                print(
                    "[INPUT] image_grid_thw:",
                    prompt_inputs[
                        "image_grid_thw"
                    ],
                )

            cuda_mem(
                "after preprocessing"
            )

            # ==================================================
            # 2. Generate reasoning
            # ==================================================

            (
                full_ids,
                reasoning_ids,
                reasoning_text,
            ) = generate_reasoning(
                trainer,
                prompt_inputs,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

            print()
            print("[REASONING]")
            print(reasoning_text)
            print()

            print(
                "[TOKENS] prompt=",
                prompt_inputs[
                    "input_ids"
                ].shape[1],
                " reasoning=",
                reasoning_ids.shape[1],
                " total=",
                full_ids.shape[1],
            )

            (
                sample_dir
                / "reasoning.txt"
            ).write_text(
                reasoning_text,
                encoding="utf-8",
            )

            torch.save(
                reasoning_ids.cpu(),
                sample_dir
                / "reasoning_token_ids.pt",
            )

            cuda_mem(
                "after reasoning generation"
            )

            # ==================================================
            # 3. Full sequence → VLM KV
            # ==================================================

            full_inputs = (
                build_full_sequence_inputs(
                    prompt_inputs,
                    full_ids,
                )
            )

            vlm_outputs = (
                extract_vlm_output(
                    trainer,
                    full_inputs,
                )
            )

            layer_kv = find_layer_kv(
                vlm_outputs
            )

            tokenizer = get_vlm_tokenizer(
                vlm
            )

            token_regions = (
                infer_token_regions(
                    prompt_inputs,
                    reasoning_ids,
                    tokenizer,
                )
            )

            save_json(
                sample_dir
                / "token_regions.json",
                token_regions,
            )

            cuda_mem(
                "after full VLM KV"
            )

            # ==================================================
            # 4. Ego conditioning
            # ==================================================

            (
                ego_state,
                ego_history,
            ) = build_ego_inputs(
                batch,
                device,
            )

            # ==================================================
            # 5. Action Expert inference
            # ==================================================

            pred_traj = run_action_sample(
                action,
                layer_kv,
                vlm_outputs,
                ego_state,
                ego_history,
            )

            pred_cpu = (
                pred_traj
                .detach()
                .float()
                .cpu()
            )

            np.save(
                sample_dir
                / "trajectory.npy",
                pred_cpu.numpy(),
            )

            save_json(
                sample_dir
                / "trajectory.json",
                pred_cpu.tolist(),
            )

            print()
            print("[TRAJECTORY]")
            print(pred_cpu)
            print()

            cuda_mem(
                "after action inference"
            )

            # ==================================================
            # 6. Save attention
            # ==================================================

            save_attention_records(
                recorder.records,
                sample_dir,
                token_regions,
            )

            metadata = {
                "sample_id":
                    sample_id,

                "vlm_model":
                    args.vlm_model,

                "quantized":
                    False,

                "dtype":
                    "bfloat16",

                "cameras":
                    CAMERAS,

                "resolution":
                    [448, 448],

                "kv_layers":
                    KV_LAYERS_1BASED,

                "num_inference_steps":
                    5,

                "action_checkpoint":
                    (
                        str(
                            loaded_action_path
                        )
                        if loaded_action_path
                        is not None
                        else None
                    ),

                "num_attention_records":
                    len(
                        recorder.records
                    ),
            }

            save_json(
                sample_dir
                / "metadata.json",
                metadata,
            )

            print(
                "[SAVED]",
                sample_dir,
            )

            processed += 1

            del (
                prompt_inputs,
                full_inputs,
                full_ids,
                reasoning_ids,
                vlm_outputs,
                layer_kv,
                ego_state,
                ego_history,
                pred_traj,
                pred_cpu,
            )

            cleanup_cuda()

    recorder.remove()

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "Samples:",
        processed,
    )

    print(
        "Result root:",
        args.result_root,
    )

    cuda_mem(
        "final"
    )


if __name__ == "__main__":
    main()
