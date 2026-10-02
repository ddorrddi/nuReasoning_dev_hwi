#!/usr/bin/env python3
"""
Multi-checkpoint E2E Action Expert validation on nuReasoning validation split.

Pipeline per evaluated epoch
----------------------------
validation images + validation VQA question
    -> frozen VLM E2E generation
    -> SAME live generation past_key_values
    -> Qwen3-VL MRoPE continuation for Action Expert Q
    -> selected epoch Action Expert
    -> flow-matching validation loss + predicted trajectory
    -> ADE / FDE / heading error

Default checkpoints
-------------------
./outputs/action_expert_stage2/epoch_1/action_expert.pt
./outputs/action_expert_stage2/epoch_2/action_expert.pt
./outputs/action_expert_stage2/epoch_3/action_expert.pt

Important
---------
- Epoch checkpoints are evaluated sequentially to keep VRAM near single-model evaluation.
- VLM generation is deterministic (do_sample=False), and evaluation seeds are matched across epochs.
- Validation Loss uses the Action Expert's own training forward objective.
- ADE/FDE use E2E trajectory sampling from each checkpoint.
- The same random seeds are reused per batch for all checkpoints so stochastic
  flow-matching loss/sampling noise is comparable across epochs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from nureasoning.common.pretrained import resolve_pretrained_path

from nureasoning.nuvla.models.vlm_backbone import (
    VLMBackbone,
    VLMBackboneConfig,
)

from nureasoning.nuvla.models.data_loader import (
    VLADataConfig,
    NuReasoningVLADataset as BaseNuReasoningVLADataset,
    vla_collate_fn,
)

from nureasoning.nuvla.models.action_expert_28layer import (
    ActionExpertConfig,
    FlowMatchingDiTActionExpert,
    compute_trajectory_metrics,
)

from nureasoning.reasoning.modules.prompt_format import (
    format_question_prompt,
)


logger = logging.getLogger(__name__)


# ============================================================
# Utilities
# ============================================================


def dataclass_field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


def parse_camera_list(raw: str) -> list[str]:
    cameras = [x.strip() for x in raw.split(",") if x.strip()]

    if not cameras:
        raise ValueError("--cameras must contain at least one camera")

    return cameras


def resolve_adapter_dir(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))

    nested = os.path.join(path, "vlm_adapter")

    if os.path.isdir(nested):
        return nested

    if os.path.isdir(path):
        return path

    raise FileNotFoundError(
        f"VLM checkpoint/adapter directory not found: {path}"
    )


# ============================================================
# Test dataset
# ============================================================


class NuReasoningTestVQADataset(BaseNuReasoningVLADataset):
    """
    Test-only VQA dataset.

    Difference from training data_loader_vqa.py:
        training:
            question + GT answer required

        testing:
            only question required

    The VLM generates the answer itself.
    """

    def __init__(
        self,
        config: VLADataConfig,
        split: str = "test",
    ):
        super().__init__(
            config=config,
            split=split,
        )

        # Keep only raw frames that actually have a test-question JSON.
        before = len(self.samples)

        self.samples = [
            sample
            for sample in self.samples
            if (
                sample["clip_name"],
                str(sample["timestamp_us"]),
            )
            in self._vqa_index
        ]

        logger.info("=" * 72)
        logger.info("TEST VQA DATASET")
        logger.info("Raw candidate samples : %d", before)
        logger.info("Matched test samples  : %d", len(self.samples))
        logger.info("=" * 72)

        if len(self.samples) == 0:
            raise RuntimeError(
                "No test samples matched the test question JSON files.\n"
                f"data_root = {config.data_root}\n"
                f"vqa_root  = {config.vqa_root}"
            )

    def _index_vqa(self):
        """
        Robust test JSON indexing.

        Supports:
            *_vqa.json

        and also arbitrary .json files containing:

            {
                "clip": "...",
                "frame_timestamp": 123456789,
                "questions": [...]
            }
        """

        if not self.config.vqa_root:
            return

        root = os.path.expanduser(self.config.vqa_root)

        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Test VQA root does not exist: {root}"
            )

        count = 0
        skipped = 0

        for dirpath, _, filenames in os.walk(root):

            for filename in filenames:

                if not filename.endswith(".json"):
                    continue

                path = os.path.join(
                    dirpath,
                    filename,
                )

                try:
                    with open(
                        path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        payload = json.load(f)

                except Exception:
                    skipped += 1
                    continue

                if not isinstance(payload, dict):
                    continue

                questions = payload.get("questions")

                if not isinstance(questions, list):
                    continue

                if len(questions) == 0:
                    continue

                # ------------------------------------------------
                # Preferred:
                #
                # {
                #   "clip": "...",
                #   "frame_timestamp": ...
                # }
                # ------------------------------------------------

                clip_name = payload.get("clip")

                timestamp = payload.get(
                    "frame_timestamp"
                )

                # ------------------------------------------------
                # Fallback for normal *_vqa.json structure
                # ------------------------------------------------

                if not clip_name:
                    clip_name = os.path.basename(
                        dirpath
                    )

                if timestamp is None:

                    suffix = "_vqa.json"

                    if filename.endswith(suffix):
                        timestamp = filename[
                            : -len(suffix)
                        ]

                if (
                    clip_name is None
                    or timestamp is None
                ):
                    continue

                key = (
                    str(clip_name),
                    str(timestamp),
                )

                self._vqa_index[key] = path

                count += 1

        logger.info(
            "[test] Indexed %d test VQA JSON files under %s",
            count,
            root,
        )

        if skipped:
            logger.info(
                "[test] Skipped unreadable JSON files: %d",
                skipped,
            )

    def _select_qa_target(
        self,
        clip_name: str,
        timestamp_us: int,
    ) -> Optional[tuple[str, str]]:
        """
        Pick one question deterministically.

        There is intentionally NO ground-truth VQA answer.
        """

        path = self._vqa_index.get(
            (
                clip_name,
                str(timestamp_us),
            )
        )

        if path is None:
            return None

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                payload = json.load(f)

        except Exception:
            return None

        questions = payload.get("questions") or []

        valid_questions = [
            q
            for q in questions
            if isinstance(q, dict)
            and q.get("question")
        ]

        if not valid_questions:
            return None

        # --------------------------------------------------------
        # Deterministic.
        #
        # Prefer driving question because Action Expert prediction
        # should be conditioned on driving-oriented reasoning.
        # --------------------------------------------------------

        driving_questions = [
            q
            for q in valid_questions
            if str(
                q.get("category", "")
            ).lower() == "driving"
        ]

        if driving_questions:
            question = driving_questions[0]
        else:
            question = valid_questions[0]

        prompt = format_question_prompt(
            question
        )

        # reasoning_text is intentionally empty.
        # Generated KV mode does not use GT reasoning text.
        return prompt, ""


# ============================================================
# Evaluator
# ============================================================


class ActionExpertEvaluator:

    def __init__(
        self,
        args: argparse.Namespace,
    ):
        self.args = args

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required."
            )

        self.device = torch.device(
            args.device
        )

        torch.cuda.set_device(
            self.device
        )

        self.use_bf16 = True

        self.cameras = parse_camera_list(
            args.cameras
        )

        self._build_vlm()
        self._build_action_expert()
        self._build_dataset()

    # ========================================================
    # VLM
    # ========================================================

    def _build_vlm(self):

        args = self.args

        model_path = resolve_pretrained_path(
            args.vlm_model_path
        )

        config_kwargs = dict(
            model_name_or_path=model_path,

            freeze_vision_encoder=False,

            lora_rank=args.vlm_lora_rank,
            lora_alpha=args.vlm_lora_alpha,
            lora_dropout=args.vlm_lora_dropout,

            current_resolution=(
                args.current_res_w,
                args.current_res_h,
            ),

            history_resolution=(
                args.history_res_w,
                args.history_res_h,
            ),

            reasoning_format=args.reasoning_format,
        )

        vlm_fields = dataclass_field_names(
            VLMBackboneConfig
        )

        if "load_in_4bit" in vlm_fields:
            config_kwargs[
                "load_in_4bit"
            ] = args.load_in_4bit

        self.vlm = VLMBackbone(
            VLMBackboneConfig(
                **config_kwargs
            )
        )

        if not args.load_in_4bit:
            self.vlm = self.vlm.to(
                self.device
            )

        adapter_dir = resolve_adapter_dir(
            args.vlm_checkpoint
        )

        logger.info(
            "Loading VLM adapter: %s",
            adapter_dir,
        )

        self.vlm.load_adapter(
            adapter_dir,
            device=self.device,
        )

        for parameter in self.vlm.parameters():
            parameter.requires_grad_(False)

        self.vlm.eval()

        base_model = self.vlm.model

        if hasattr(
            base_model,
            "gradient_checkpointing_disable",
        ):
            base_model.gradient_checkpointing_disable()

        if hasattr(
            base_model.config,
            "use_cache",
        ):
            base_model.config.use_cache = False

        (
            self.vlm_num_layers,
            self.kv_num_heads,
            self.kv_head_dim,
        ) = self.vlm.get_kv_spec()

        if args.kv_layers.lower() == "all":

            self.selected_kv_layers = list(
                range(
                    self.vlm_num_layers
                )
            )

        else:

            requested = [
                int(x.strip())
                for x in args.kv_layers.split(",")
                if x.strip()
            ]

            self.selected_kv_layers = [
                x - 1
                for x in requested
            ]

        logger.info(
            "VLM layers          : %d",
            self.vlm_num_layers,
        )

        logger.info(
            "Selected KV layers  : %s",
            [
                x + 1
                for x in self.selected_kv_layers
            ],
        )

        logger.info(
            "KV heads / head dim : %d / %d",
            self.kv_num_heads,
            self.kv_head_dim,
        )

    # ========================================================
    # Action Expert / checkpoints
    # ========================================================

    def _make_action_expert_config(self):
        args = self.args

        return ActionExpertConfig(
            vlm_feature_dim=self.vlm.feature_dim,
            ego_state_dim=4,
            max_history_traj_points=args.max_history_traj_points,
            num_waypoints=args.num_trajectory_points,
            trajectory_dim=3,
            hidden_dim=args.action_hidden_dim,
            num_heads=args.num_dit_heads,
            num_dit_layers=len(self.selected_kv_layers),
            dropout=args.dropout,
            mlp_ratio=args.mlp_ratio,
            self_attention_every=args.self_attention_every,
            kv_layer_indices=self.selected_kv_layers,
            kv_num_heads=self.kv_num_heads,
            kv_head_dim=self.kv_head_dim,
            qk_norm_eps=args.qk_norm_eps,
            num_inference_steps=args.num_inference_steps,
            num_timestep_buckets=args.num_timestep_buckets,
            noise_beta_alpha=args.noise_beta_alpha,
            noise_beta_beta=args.noise_beta_beta,
        )

    def _parse_epochs(self) -> list[int]:
        epochs = []
        for token in self.args.epochs.split(","):
            token = token.strip()
            if not token:
                continue
            epoch = int(token)
            if epoch <= 0:
                raise ValueError(f"Invalid epoch: {epoch}")
            epochs.append(epoch)

        if not epochs:
            raise ValueError("--epochs must contain at least one epoch")

        return list(dict.fromkeys(epochs))

    def _build_action_expert(self):
        self.epoch_ids = self._parse_epochs()
        config = self._make_action_expert_config()
        self.action_expert = FlowMatchingDiTActionExpert(config).to(self.device)
        self.action_expert.eval()
        self.num_expert_tokens = self.action_expert.num_expert_tokens

        total = sum(p.numel() for p in self.action_expert.parameters())
        logger.info("Action Expert params: %s", f"{total:,}")
        logger.info("Epochs to evaluate: %s", self.epoch_ids)

    def _load_epoch_checkpoint(self, epoch: int) -> Path:
        checkpoint = (
            Path(os.path.expanduser(self.args.checkpoint_root))
            / f"epoch_{epoch}"
            / "action_expert.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))

        logger.info("Loading E%d checkpoint: %s", epoch, checkpoint)
        state_dict = torch.load(
            checkpoint,
            map_location=self.device,
        )
        incompatible = self.action_expert.load_state_dict(
            state_dict,
            strict=True,
        )
        self.action_expert.eval()

        logger.info(
            "E%d loaded | missing=%s unexpected=%s",
            epoch,
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
        return checkpoint

    @staticmethod
    def _extract_loss_value(output: Any) -> torch.Tensor:
        """Accept the common return shapes used by Action Expert training."""
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in (
                "loss",
                "flow_matching_loss",
                "fm_loss",
                "trajectory_loss",
            ):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value

        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value) and value.ndim == 0:
                    return value
                if isinstance(value, dict):
                    try:
                        return ActionExpertEvaluator._extract_loss_value(value)
                    except RuntimeError:
                        pass

        raise RuntimeError(
            "Could not extract validation loss from Action Expert forward output. "
            f"type={type(output)!r}"
        )

    def _compute_flow_matching_loss(
        self,
        *,
        target: torch.Tensor,
        layer_kv: Dict[int, Dict[str, torch.Tensor]],
        ego_state: torch.Tensor,
        ego_history: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same training objective as train_action_expert:
        GT future trajectory is passed as x_1; VLM conditioning stays frozen.
        """
        output = self.action_expert(
            x_1=target,
            layer_kv=layer_kv,
            ego_state=ego_state,
            history_trajectory=ego_history,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        return self._extract_loss_value(output).float()

    # ========================================================
    # Dataset
    # ========================================================

    def _build_dataset(self):

        args = self.args

        cfg = VLADataConfig(

            data_root=args.test_data_root,

            num_history_steps=(
                args.num_history_steps
            ),

            history_stride=(
                args.history_stride
            ),

            current_resolution=(
                args.current_res_w,
                args.current_res_h,
            ),

            history_resolution=(
                args.history_res_w,
                args.history_res_h,
            ),

            trajectory_future_seconds=(
                args.trajectory_future_seconds
            ),

            frame_rate_hz=(
                args.frame_rate_hz
            ),

            num_waypoints=(
                args.num_trajectory_points
            ),

            max_history_traj_points=(
                args.max_history_traj_points
            ),

            cameras=list(
                self.cameras
            ),

            reasoning_format=(
                args.reasoning_format
            ),

            reasoning_max_items_per_list=10,

            reasoning_mode="vqa",

            vqa_root=args.test_vqa_root,

            qa_seed=42,
        )

        dataset = NuReasoningTestVQADataset(
            cfg,
            split="test",
        )

        self.loader = DataLoader(

            dataset,

            batch_size=args.batch_size,

            shuffle=False,

            num_workers=args.num_workers,

            collate_fn=vla_collate_fn,

            pin_memory=True,

            drop_last=False,
        )

        logger.info(
            "Test samples: %d",
            len(dataset),
        )

    # ========================================================
    # VLM prompt
    # ========================================================

    def _sample_prompt_and_context(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ):

        images = [
            img
            for img in batch["images"][batch_idx]
            if img is not None
        ]

        expected = (
            len(self.cameras)
            * (
                self.args.num_history_steps
                + 1
            )
        )

        if len(images) != expected:
            raise RuntimeError(
                f"Expected {expected} images, "
                f"got {len(images)}"
            )

        image_contexts = []

        for t in range(
            -self.args.num_history_steps,
            1,
        ):

            t_label = (
                f"t={t}"
                if t < 0
                else "t=0 (current)"
            )

            for cam in self.cameras:

                image_contexts.append(
                    f"{t_label}, "
                    f"{cam} camera."
                )

        prompts = batch.get(
            "user_prompts"
        )

        prompt = None

        if (
            prompts is not None
            and batch_idx < len(prompts)
        ):
            prompt = prompts[batch_idx]

        if not prompt:

            prompt = self.vlm.build_multiview_prompt(

                num_history_steps=(
                    self.args.num_history_steps
                ),

                num_current_cameras=len(
                    self.cameras
                ),

                mission_command=(
                    batch[
                        "mission_commands"
                    ][batch_idx]
                ),
            )

        return (
            images,
            image_contexts,
            prompt,
        )

    def _prepare_vlm_input(
        self,
        batch,
        batch_idx,
        assistant_response=None,
    ):

        (
            images,
            image_contexts,
            prompt,
        ) = self._sample_prompt_and_context(
            batch,
            batch_idx,
        )

        inputs = self.vlm.prepare_inputs(

            images=images,

            text_prompt=prompt,

            image_contexts=(
                image_contexts
            ),

            assistant_response=(
                assistant_response
            ),
        )

        result = {}

        for key, value in inputs.items():

            if isinstance(
                value,
                torch.Tensor,
            ):
                result[key] = value.to(
                    self.device
                )
            else:
                result[key] = value

        if "prompt_length" in result:

            result[
                "prompt_lengths"
            ] = result.pop(
                "prompt_length"
            ).to(self.device)

        return result

    # ========================================================
    # GENERATED-LIVE KV + Qwen3-VL MRoPE
    # ========================================================

    @staticmethod
    def _cache_seq_len(
        past_key_values: Any,
        layer_idx: int = 0,
    ) -> int:
        """Read the current autoregressive KV-cache length."""
        layers = getattr(past_key_values, "layers", None)
        if layers is not None:
            layer = layers[layer_idx]
            key = getattr(layer, "keys", None)
            if key is not None:
                return int(key.shape[-2])

        key_cache = getattr(past_key_values, "key_cache", None)
        if key_cache is not None:
            return int(key_cache[layer_idx].shape[-2])

        if isinstance(past_key_values, (tuple, list)):
            return int(past_key_values[layer_idx][0].shape[-2])

        to_legacy = getattr(past_key_values, "to_legacy_cache", None)
        if callable(to_legacy):
            return int(to_legacy()[layer_idx][0].shape[-2])

        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return int(get_seq_length(layer_idx))
            except TypeError:
                return int(get_seq_length())

        raise TypeError(
            f"Unsupported generation cache type: {type(past_key_values)!r}"
        )

    @torch.no_grad()
    def _generate_live_cache_for_sample(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ):
        """
        Same generated-only bridge used during Action Expert training:

        image + VQA prompt
            -> frozen Qwen3-VL generate(use_cache=True)
            -> reuse the SAME generation past_key_values
            -> no generated-text re-tokenization
            -> no second VLM forward
        """
        prompt_input = self._prepare_vlm_input(
            batch,
            batch_idx,
            assistant_response=None,
        )

        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input["input_ids"],
            "attention_mask": prompt_input["attention_mask"],
            "max_new_tokens": self.args.generation_max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": True,
        }

        for key in (
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        ):
            value = prompt_input.get(key)
            if value is not None:
                generation_kwargs[key] = value

        with torch.amp.autocast(
            "cuda",
            enabled=self.use_bf16,
            dtype=torch.bfloat16,
        ):
            generation_output = self.vlm.model.generate(
                **generation_kwargs
            )

        generated_sequences = generation_output.sequences
        past_key_values = getattr(
            generation_output,
            "past_key_values",
            None,
        )

        if past_key_values is None:
            raise RuntimeError(
                "generate() returned no past_key_values. "
                "Need return_dict_in_generate=True and use_cache=True."
            )

        prompt_len = int(
            prompt_input["input_ids"].shape[1]
        )
        total_len = int(
            generated_sequences.shape[1]
        )
        generated_len = total_len - prompt_len

        if generated_len <= 0:
            raise RuntimeError(
                "Frozen VLM generated zero reasoning tokens."
            )

        cache_len = self._cache_seq_len(
            past_key_values,
            self.selected_kv_layers[0],
        )

        if cache_len <= 0 or cache_len > total_len:
            raise RuntimeError(
                "Invalid generation cache length: "
                f"cache={cache_len}, sequence={total_len}"
            )

        generated_attention_mask = torch.ones(
            generated_sequences.shape,
            device=generated_sequences.device,
            dtype=prompt_input["attention_mask"].dtype,
        )
        generated_attention_mask[:, :prompt_len] = (
            prompt_input["attention_mask"]
        )

        # Patched VLMBackbone helper used by the training pipeline.
        layer_kv = self.vlm._extract_full_context_kv(
            past_key_values=past_key_values,
            selected_kv_layers=self.selected_kv_layers,
            attention_mask=generated_attention_mask[:, :cache_len],
        )

        for layer_idx, entry in layer_kv.items():
            key = entry["key"]
            value = entry["value"]

            if key.ndim != 4 or value.ndim != 4:
                raise RuntimeError(
                    f"Layer {layer_idx}: expected KV [B,H,T,D], "
                    f"K={tuple(key.shape)} V={tuple(value.shape)}"
                )

            layer_cache_len = int(key.shape[-2])

            entry["mask"] = (
                generated_attention_mask[
                    :, :layer_cache_len
                ].bool().detach()
            )
            entry["key"] = key.detach()
            entry["value"] = value.detach()

        new_ids = generated_sequences[:, prompt_len:]
        generated_text = (
            self.vlm.processor.batch_decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        )

        return (
            layer_kv,
            prompt_input,
            generated_text,
            cache_len,
        )

    def _resolve_qwen_text_model(self):
        """Resolve the loaded Qwen3-VL module that owns get_rope_index()."""
        queue = [getattr(self.vlm, "model", None)]
        seen = set()

        while queue:
            obj = queue.pop(0)

            if obj is None or id(obj) in seen:
                continue

            seen.add(id(obj))

            if hasattr(obj, "get_rope_index"):
                return obj

            get_base_model = getattr(
                obj,
                "get_base_model",
                None,
            )
            if callable(get_base_model):
                try:
                    queue.append(
                        get_base_model()
                    )
                except Exception:
                    pass

            for name in (
                "model",
                "base_model",
                "module",
            ):
                child = getattr(
                    obj,
                    name,
                    None,
                )
                if child is not None:
                    queue.append(child)

        raise RuntimeError(
            "Could not resolve the underlying Qwen3-VL model "
            "exposing get_rope_index()."
        )

    def _resolve_qwen_rotary_emb(self):
        """Resolve Qwen3-VL language_model.rotary_emb."""
        qwen = self._resolve_qwen_text_model()

        candidates = [
            getattr(qwen, "language_model", None),
            qwen,
        ]

        model_child = getattr(
            qwen,
            "model",
            None,
        )

        if model_child is not None:
            candidates.extend(
                [
                    getattr(
                        model_child,
                        "language_model",
                        None,
                    ),
                    model_child,
                ]
            )

        for module in candidates:
            if module is None:
                continue

            rotary_emb = getattr(
                module,
                "rotary_emb",
                None,
            )

            if rotary_emb is not None:
                return rotary_emb

        raise RuntimeError(
            "Could not resolve Qwen3-VL "
            "language_model.rotary_emb."
        )

    @torch.no_grad()
    def _compute_qwen_rope_delta(
        self,
        sample_input: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute multimodal rope_delta from the same prompt context."""
        qwen = self._resolve_qwen_text_model()

        mm_token_type_ids = sample_input.get(
            "mm_token_type_ids"
        )

        if mm_token_type_ids is None:
            raise RuntimeError(
                "Qwen3-VL MRoPE alignment requires "
                "mm_token_type_ids."
            )

        _, rope_delta = qwen.get_rope_index(
            input_ids=sample_input["input_ids"],
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=sample_input.get(
                "image_grid_thw"
            ),
            video_grid_thw=sample_input.get(
                "video_grid_thw"
            ),
            attention_mask=sample_input[
                "attention_mask"
            ],
        )

        rope_delta = rope_delta.to(
            device=sample_input["input_ids"].device,
            dtype=torch.long,
        ).reshape(-1)

        if rope_delta.numel() != 1:
            raise RuntimeError(
                "Expected one rope_delta for a single sample, "
                f"got {tuple(rope_delta.shape)}"
            )

        return rope_delta

    @torch.no_grad()
    def _build_expert_rope_for_sample(
        self,
        sample_input: Dict[str, Any],
        *,
        rope_delta: torch.Tensor,
        num_expert_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Build Qwen3-VL MRoPE continuation positions for Expert tokens.
        """
        attention_mask = sample_input[
            "attention_mask"
        ]

        if (
            attention_mask.ndim != 2
            or attention_mask.shape[0] != 1
        ):
            raise RuntimeError(
                "Expected single-sample attention_mask [1,T], "
                f"got {tuple(attention_mask.shape)}"
            )

        valid_len = attention_mask.long().sum(
            dim=-1
        )

        local = torch.arange(
            num_expert_tokens,
            device=attention_mask.device,
            dtype=torch.long,
        )

        base = (
            valid_len[:, None]
            + rope_delta[:, None]
            + local[None, :]
        )

        position_ids = (
            base.unsqueeze(0)
            .expand(3, -1, -1)
            .contiguous()
        )

        rotary_emb = (
            self._resolve_qwen_rotary_emb()
        )

        dummy = torch.empty(
            (1, num_expert_tokens, 1),
            device=device,
            dtype=dtype,
        )

        pos = position_ids.to(device)

        rope_error = None

        try:
            cos, sin = rotary_emb(
                dummy,
                pos,
            )
        except Exception as exc:
            rope_error = exc
            text_pos = pos[:1]
            pos4 = torch.cat(
                [text_pos, pos],
                dim=0,
            )

            try:
                cos, sin = rotary_emb(
                    dummy,
                    pos4,
                )
            except Exception as exc4:
                raise RuntimeError(
                    "Failed to generate Expert RoPE with "
                    "the loaded Qwen3-VL rotary_emb. "
                    f"3-axis error={rope_error!r}; "
                    f"4-axis error={exc4!r}"
                ) from exc4

        if cos.ndim != 3 or sin.ndim != 3:
            raise RuntimeError(
                "Unexpected Qwen rotary output: "
                f"cos={tuple(cos.shape)}, "
                f"sin={tuple(sin.shape)}"
            )

        if cos.shape[-1] != self.kv_head_dim:
            raise RuntimeError(
                "Qwen RoPE dimension does not match "
                "VLM K/V head dim: "
                f"rope={cos.shape[-1]}, "
                f"kv_head_dim={self.kv_head_dim}"
            )

        return (
            cos.detach().to(
                device=device,
                dtype=dtype,
            ),
            sin.detach().to(
                device=device,
                dtype=dtype,
            ),
            position_ids.detach().to(device),
        )

    # ========================================================
    # Batch KV
    # ========================================================

    @staticmethod
    def _stack_layer_kv(
        samples,
    ):
        if not samples:
            raise ValueError(
                "samples is empty."
            )

        layer_ids = sorted(
            samples[0].keys()
        )

        for sample in samples[1:]:
            if sorted(sample.keys()) != layer_ids:
                raise RuntimeError(
                    "Samples expose different VLM KV layers."
                )

        batched = {}

        for layer_idx in layer_ids:
            max_len = max(
                sample[layer_idx]["key"].shape[2]
                for sample in samples
            )

            keys = []
            values = []
            masks = []

            for sample in samples:
                entry = sample[layer_idx]

                key = entry["key"]
                value = entry["value"]

                if (
                    key.ndim != 4
                    or value.ndim != 4
                ):
                    raise RuntimeError(
                        "Expected KV [B,H,T,D], "
                        f"K={tuple(key.shape)}, "
                        f"V={tuple(value.shape)}"
                    )

                if key.shape != value.shape:
                    raise RuntimeError(
                        f"K/V shape mismatch at layer {layer_idx}: "
                        f"K={tuple(key.shape)}, "
                        f"V={tuple(value.shape)}"
                    )

                current_len = key.shape[2]

                mask = entry.get("mask")

                if mask is None:
                    mask = torch.ones(
                        key.shape[0],
                        current_len,
                        device=key.device,
                        dtype=torch.bool,
                    )
                else:
                    mask = mask.to(
                        device=key.device,
                        dtype=torch.bool,
                    )

                pad = max_len - current_len

                if pad > 0:
                    key = torch.cat(
                        [
                            key,
                            key.new_zeros(
                                key.shape[0],
                                key.shape[1],
                                pad,
                                key.shape[3],
                            ),
                        ],
                        dim=2,
                    )

                    value = torch.cat(
                        [
                            value,
                            value.new_zeros(
                                value.shape[0],
                                value.shape[1],
                                pad,
                                value.shape[3],
                            ),
                        ],
                        dim=2,
                    )

                    mask = torch.cat(
                        [
                            mask,
                            torch.zeros(
                                mask.shape[0],
                                pad,
                                device=mask.device,
                                dtype=torch.bool,
                            ),
                        ],
                        dim=1,
                    )

                keys.append(key)
                values.append(value)
                masks.append(mask)

            batched[layer_idx] = {
                "key": torch.cat(
                    keys,
                    dim=0,
                ).detach(),
                "value": torch.cat(
                    values,
                    dim=0,
                ).detach(),
                "mask": torch.cat(
                    masks,
                    dim=0,
                ).detach(),
            }

        return batched

    @torch.no_grad()
    def _extract_generated_live_context(
        self,
        batch,
    ):
        """
        Build the exact inference conditioning used in training:
        live generation KV + Qwen MRoPE continuation.
        """
        layer_kv_list = []
        rope_cos_list = []
        rope_sin_list = []
        generated_texts = []

        num_expert_tokens = self.num_expert_tokens

        for b in range(
            len(batch["images"])
        ):
            (
                sample_kv,
                prompt_input,
                generated_text,
                cache_len,
            ) = self._generate_live_cache_for_sample(
                batch,
                b,
            )

            first_idx = (
                self.selected_kv_layers[0]
            )

            first_k = sample_kv[
                first_idx
            ]["key"]

            if first_k.shape[1] != self.kv_num_heads:
                raise RuntimeError(
                    "KV head mismatch: "
                    f"expected {self.kv_num_heads}, "
                    f"got {first_k.shape[1]}"
                )

            if first_k.shape[-1] != self.kv_head_dim:
                raise RuntimeError(
                    "KV head_dim mismatch: "
                    f"expected {self.kv_head_dim}, "
                    f"got {first_k.shape[-1]}"
                )

            # Same multimodal prompt -> same Qwen rope_delta.
            rope_delta = self._compute_qwen_rope_delta(
                prompt_input
            )

            # Expert positions continue after the live generation cache.
            rope_input = dict(
                prompt_input
            )

            rope_input["attention_mask"] = (
                torch.ones(
                    (1, cache_len),
                    device=first_k.device,
                    dtype=prompt_input[
                        "attention_mask"
                    ].dtype,
                )
            )

            (
                rope_cos,
                rope_sin,
                _,
            ) = self._build_expert_rope_for_sample(
                rope_input,
                rope_delta=rope_delta,
                num_expert_tokens=num_expert_tokens,
                dtype=first_k.dtype,
                device=first_k.device,
            )

            layer_kv_list.append(
                sample_kv
            )
            rope_cos_list.append(
                rope_cos
            )
            rope_sin_list.append(
                rope_sin
            )
            generated_texts.append(
                generated_text
            )

        layer_kv = self._stack_layer_kv(
            layer_kv_list
        )

        rope_cos = torch.cat(
            rope_cos_list,
            dim=0,
        ).detach()

        rope_sin = torch.cat(
            rope_sin_list,
            dim=0,
        ).detach()

        return (
            layer_kv,
            generated_texts,
            rope_cos,
            rope_sin,
        )

    # ========================================================
    # Evaluation
    # ========================================================

    @staticmethod
    def _set_eval_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def _evaluate_one_epoch(self, epoch: int) -> Dict[str, Any]:
        checkpoint = self._load_epoch_checkpoint(epoch)
        self.vlm.eval()
        self.action_expert.eval()

        loss_sum = 0.0
        metric_sums: Dict[str, float] = {}
        num_samples = 0
        predictions = []
        start_time = time.time()

        logger.info("=" * 96)
        logger.info("E%d E2E VALIDATION START", epoch)
        logger.info("Checkpoint: %s", checkpoint)
        logger.info("=" * 96)

        for step, batch in enumerate(self.loader):
            # Full E2E path for this checkpoint:
            # images/question -> frozen VLM generation -> live KV -> AE.
            (
                layer_kv,
                generated_texts,
                rope_cos,
                rope_sin,
            ) = self._extract_generated_live_context(batch)

            ego_history = batch["ego_history_trajectories"].to(self.device)
            ego_velocities = batch["ego_velocities"].to(self.device)
            ego_accelerations = batch["ego_accelerations"].to(self.device)
            ego_state = torch.cat(
                [ego_velocities, ego_accelerations],
                dim=-1,
            )
            target = batch["ego_trajectories"].to(self.device).float()
            bs = int(target.shape[0])

            # Same seeds at the same validation step for every epoch.
            loss_seed = self.args.eval_seed + step * 2
            sample_seed = self.args.eval_seed + step * 2 + 1

            self._set_eval_seed(loss_seed)
            with torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                val_loss = self._compute_flow_matching_loss(
                    target=target,
                    layer_kv=layer_kv,
                    ego_state=ego_state,
                    ego_history=ego_history,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                )

            self._set_eval_seed(sample_seed)
            with torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                prediction = self.action_expert.sample(
                    layer_kv=layer_kv,
                    ego_state=ego_state,
                    history_trajectory=ego_history,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    num_steps=self.args.num_inference_steps,
                )

            prediction = prediction.float()
            metrics = compute_trajectory_metrics(
                prediction,
                target,
            )

            loss_sum += float(val_loss.item()) * bs
            num_samples += bs

            for key, value in metrics.items():
                metric_sums[key] = (
                    metric_sums.get(key, 0.0)
                    + float(value) * bs
                )

            pred_cpu = prediction.detach().cpu().tolist()
            target_cpu = target.detach().cpu().tolist()

            for i in range(bs):
                predictions.append(
                    {
                        "clip": batch["clip_names"][i],
                        "timestamp_us": int(batch["timestamps"][i]),
                        "generated_reasoning": generated_texts[i],
                        "predicted_trajectory": pred_cpu[i],
                        "gt_trajectory": target_cpu[i],
                    }
                )

            if (
                (step + 1) % self.args.log_interval == 0
                or step + 1 == len(self.loader)
            ):
                n = max(num_samples, 1)
                current_loss = loss_sum / n
                current_ade = metric_sums.get("ADE_m", 0.0) / n
                current_fde = metric_sums.get("FDE_m", 0.0) / n
                current_heading = (
                    metric_sums.get("heading_error_deg", 0.0) / n
                )
                elapsed_min = (time.time() - start_time) / 60.0

                logger.info(
                    "E%d | Step %d/%d | samples=%d | Loss=%.6f | "
                    "ADE=%.4f m | FDE=%.4f m | Heading=%.4f deg | "
                    "elapsed=%.1f min",
                    epoch,
                    step + 1,
                    len(self.loader),
                    num_samples,
                    current_loss,
                    current_ade,
                    current_fde,
                    current_heading,
                    elapsed_min,
                )

        n = max(num_samples, 1)
        result = {
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "validation_loss": loss_sum / n,
            **{
                key: value / n
                for key, value in metric_sums.items()
            },
            "num_validation_samples": num_samples,
            "elapsed_seconds": time.time() - start_time,
        }

        epoch_dir = Path(self.args.output_dir) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        with open(
            epoch_dir / "validation_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result, f, indent=2)

        with open(
            epoch_dir / "validation_predictions.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                predictions,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("-" * 96)
        logger.info(
            "E%d COMPLETE | Loss=%.6f | ADE=%.4f m | FDE=%.4f m | "
            "Heading=%.4f deg | samples=%d",
            epoch,
            result["validation_loss"],
            result.get("ADE_m", float("nan")),
            result.get("FDE_m", float("nan")),
            result.get("heading_error_deg", float("nan")),
            num_samples,
        )
        logger.info("-" * 96)

        return result

    @torch.no_grad()
    def evaluate(self):
        output_root = Path(self.args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        summary = []
        total_start = time.time()

        for epoch in self.epoch_ids:
            result = self._evaluate_one_epoch(epoch)
            summary.append(result)
            torch.cuda.empty_cache()

        with open(
            output_root / "all_epochs_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        csv_path = output_root / "all_epochs_summary.csv"
        metric_keys = [
            "validation_loss",
            "ADE_m",
            "FDE_m",
            "heading_error_deg",
            "num_validation_samples",
            "elapsed_seconds",
        ]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch," + ",".join(metric_keys) + "\n")
            for row in summary:
                values = [str(row.get(key, "")) for key in metric_keys]
                f.write(str(row["epoch"]) + "," + ",".join(values) + "\n")

        logger.info("=" * 96)
        logger.info("ALL EPOCHS E2E VALIDATION COMPLETE")
        logger.info(
            "%-8s %-14s %-14s %-14s %-18s",
            "Epoch",
            "Val Loss",
            "ADE (m)",
            "FDE (m)",
            "Heading (deg)",
        )
        logger.info("-" * 96)

        for row in summary:
            logger.info(
                "E%-7d %-14.6f %-14.4f %-14.4f %-18.4f",
                row["epoch"],
                row.get("validation_loss", float("nan")),
                row.get("ADE_m", float("nan")),
                row.get("FDE_m", float("nan")),
                row.get("heading_error_deg", float("nan")),
            )

        logger.info("-" * 96)
        logger.info(
            "Total elapsed : %.1f min",
            (time.time() - total_start) / 60.0,
        )
        logger.info("Summary JSON : %s", output_root / "all_epochs_summary.json")
        logger.info("Summary CSV  : %s", csv_path)
        logger.info("=" * 96)

        return summary


# ============================================================
# Args
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    parser.add_argument(
        "--test_data_root",
        default="/media/HDD/nuR_ds/data/validation",
    )

    parser.add_argument(
        "--test_vqa_root",
        default="/home/lhh/lab/dataset/nuReasoning/vqa_validation_filtered",
    )

    # --------------------------------------------------------
    # VLM
    # --------------------------------------------------------

    parser.add_argument(
        "--vlm_model_path",
        default=(
            "/home/lhh/lab/models/vlm/"
            "Qwen3-VL-2B-Instruct"
        ),
    )

    parser.add_argument(
        "--vlm_checkpoint",
        default=(
            "./outputs/vlm_pretrain/final"
        ),
    )

    # --------------------------------------------------------
    # Action Expert checkpoints
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint_root",
        default="./outputs/action_expert_stage2",
        help="Root containing epoch_N/action_expert.pt directories.",
    )

    parser.add_argument(
        "--epochs",
        default="1,2,3",
        help="Comma-separated Action Expert epochs to evaluate.",
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "./outputs/action_expert_stage2/"
            "validation_e2e_multi_epoch"
        ),
    )

    # --------------------------------------------------------
    # Must match training
    # --------------------------------------------------------

    parser.add_argument(
        "--vlm_lora_rank",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--vlm_lora_alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--vlm_lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
    )

    parser.add_argument(
        "--cameras",
        default=(
            "front,"
            "front_left,"
            "front_right"
        ),
    )

    parser.add_argument(
        "--num_history_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--history_stride",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--current_res_w",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--current_res_h",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--history_res_w",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--history_res_h",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--reasoning_format",
        default=(
            "spatial_driving_counterfactual"
        ),
    )

    parser.add_argument(
        "--trajectory_future_seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--frame_rate_hz",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--num_trajectory_points",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_history_traj_points",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--kv_layers",
        default="all",
    )

    # --------------------------------------------------------
    # Action Expert architecture
    # --------------------------------------------------------

    parser.add_argument(
        "--action_hidden_dim",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--num_dit_heads",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--self_attention_every",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--qk_norm_eps",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num_timestep_buckets",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--noise_beta_alpha",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--noise_beta_beta",
        type=float,
        default=2.5,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Shared seed for fair flow-loss/sampling comparison across epochs.",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    logging.basicConfig(

        level=logging.INFO,

        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s: "
            "%(message)s"
        ),

        handlers=[

            logging.StreamHandler(),

            logging.FileHandler(
                os.path.join(
                    args.output_dir,
                    "validation_multi_epoch.log",
                ),
                mode="w",
            ),
        ],
    )

    logger.info(
        "Arguments:\n%s",
        json.dumps(
            vars(args),
            indent=2,
        ),
    )

    evaluator = (
        ActionExpertEvaluator(
            args
        )
    )

    evaluator.evaluate()


if __name__ == "__main__":
    main()