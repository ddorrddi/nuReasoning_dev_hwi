#!/usr/bin/env python3
"""
Action Expert evaluation on nuReasoning test split.

Pipeline
--------
test images + test VQA question
    -> frozen VLM
    -> autoregressive generated reasoning / answer
    -> frozen VLM full-context K/V
    -> Action Expert
    -> predicted trajectory
    -> ADE / FDE / heading error

Checkpoint
----------
outputs/action_expert_stage2/epoch_1/action_expert.pt

Test data
---------
/media/HDD/nuR_ds/data/test

Important
---------
- Test VQA JSON does NOT need an answer field.
- Only the question + choices are used as VLM prompt.
- VLM weights are frozen.
- Action Expert weights are loaded from action_expert.pt.
- No optimizer / training_state.pt is needed for evaluation.
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

from nureasoning.nuvla.models.action_expert import (
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
        self._load_action_checkpoint()
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
    # Action Expert
    # ========================================================

    def _build_action_expert(self):

        args = self.args

        config = ActionExpertConfig(

            vlm_feature_dim=self.vlm.feature_dim,

            ego_state_dim=4,

            max_history_traj_points=(
                args.max_history_traj_points
            ),

            num_waypoints=(
                args.num_trajectory_points
            ),

            trajectory_dim=3,

            hidden_dim=(
                args.action_hidden_dim
            ),

            num_heads=(
                args.num_dit_heads
            ),

            num_dit_layers=len(
                self.selected_kv_layers
            ),

            dropout=args.dropout,

            mlp_ratio=args.mlp_ratio,

            self_attention_every=(
                args.self_attention_every
            ),

            kv_layer_indices=(
                self.selected_kv_layers
            ),

            kv_num_heads=(
                self.kv_num_heads
            ),

            kv_head_dim=(
                self.kv_head_dim
            ),

            num_inference_steps=(
                args.num_inference_steps
            ),

            num_timestep_buckets=(
                args.num_timestep_buckets
            ),

            noise_beta_alpha=(
                args.noise_beta_alpha
            ),

            noise_beta_beta=(
                args.noise_beta_beta
            ),
        )

        self.action_expert = (
            FlowMatchingDiTActionExpert(
                config
            ).to(self.device)
        )

        self.action_expert.eval()

        total = sum(
            p.numel()
            for p in self.action_expert.parameters()
        )

        logger.info(
            "Action Expert params: %s",
            f"{total:,}",
        )

    # ========================================================
    # Checkpoint
    # ========================================================

    def _load_action_checkpoint(self):

        checkpoint = os.path.expanduser(
            self.args.action_checkpoint
        )

        if os.path.isdir(checkpoint):

            checkpoint = os.path.join(
                checkpoint,
                "action_expert.pt",
            )

        if not os.path.isfile(checkpoint):

            raise FileNotFoundError(
                checkpoint
            )

        logger.info(
            "Loading Action Expert checkpoint: %s",
            checkpoint,
        )

        state_dict = torch.load(
            checkpoint,
            map_location=self.device,
        )

        incompatible = (
            self.action_expert.load_state_dict(
                state_dict,
                strict=True,
            )
        )

        logger.info(
            "Checkpoint loaded."
        )

        logger.info(
            "missing_keys=%s unexpected_keys=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )

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
    # Generate VLM reasoning
    # ========================================================

    @torch.no_grad()
    def _generate_reasoning(
        self,
        prompt_inputs,
    ):

        kwargs = {

            "input_ids":
                prompt_inputs[
                    "input_ids"
                ],

            "attention_mask":
                prompt_inputs[
                    "attention_mask"
                ],

            "max_new_tokens":
                self.args.generation_max_new_tokens,

            "do_sample":
                False,

            "use_cache":
                True,
        }

        for key in (
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        ):

            value = prompt_inputs.get(
                key
            )

            if value is not None:
                kwargs[key] = value

        with torch.amp.autocast(
            "cuda",
            enabled=True,
            dtype=torch.bfloat16,
        ):

            output_ids = (
                self.vlm.model.generate(
                    **kwargs
                )
            )

        prompt_len = int(
            prompt_inputs[
                "input_ids"
            ].shape[1]
        )

        new_ids = output_ids[
            :,
            prompt_len:
        ]

        generated = (
            self.vlm.processor.batch_decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        )

        return generated

    # ========================================================
    # KV
    # ========================================================

    @staticmethod
    def _stack_layer_kv(
        samples,
    ):

        layer_ids = sorted(
            samples[0].keys()
        )

        batched = {}

        for layer_idx in layer_ids:

            max_len = max(
                sample[
                    layer_idx
                ]["key"].shape[2]
                for sample in samples
            )

            keys = []
            values = []
            masks = []

            for sample in samples:

                key = sample[
                    layer_idx
                ]["key"]

                value = sample[
                    layer_idx
                ]["value"]

                mask = sample[
                    layer_idx
                ]["mask"]

                pad = (
                    max_len
                    - key.shape[2]
                )

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
                                dtype=torch.bool,
                                device=mask.device,
                            ),
                        ],
                        dim=1,
                    )

                keys.append(key)
                values.append(value)
                masks.append(mask)

            batched[layer_idx] = {

                "key":
                    torch.cat(
                        keys,
                        dim=0,
                    ).detach(),

                "value":
                    torch.cat(
                        values,
                        dim=0,
                    ).detach(),

                "mask":
                    torch.cat(
                        masks,
                        dim=0,
                    ).detach(),
            }

        return batched

    @torch.no_grad()
    def _extract_generated_kv(
        self,
        batch,
    ):

        sample_kv = []
        generated_texts = []

        batch_size = len(
            batch["images"]
        )

        for b in range(
            batch_size
        ):

            # ----------------------------------------------
            # 1. image + test question
            # ----------------------------------------------

            prompt_inputs = (
                self._prepare_vlm_input(
                    batch,
                    b,
                    assistant_response=None,
                )
            )

            # ----------------------------------------------
            # 2. VLM autoregressive answer/reasoning
            # ----------------------------------------------

            generated = (
                self._generate_reasoning(
                    prompt_inputs
                )
            )

            generated_texts.append(
                generated
            )

            # ----------------------------------------------
            # 3. Re-feed generated reasoning
            # ----------------------------------------------

            full_inputs = (
                self._prepare_vlm_input(
                    batch,
                    b,
                    assistant_response=generated,
                )
            )

            # ----------------------------------------------
            # 4. Extract full-context KV
            # ----------------------------------------------

            with torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):

                outputs = self.vlm(

                    input_ids=(
                        full_inputs[
                            "input_ids"
                        ]
                    ),

                    attention_mask=(
                        full_inputs[
                            "attention_mask"
                        ]
                    ),

                    pixel_values=(
                        full_inputs.get(
                            "pixel_values"
                        )
                    ),

                    image_grid_thw=(
                        full_inputs.get(
                            "image_grid_thw"
                        )
                    ),

                    mm_token_type_ids=(
                        full_inputs.get(
                            "mm_token_type_ids"
                        )
                    ),

                    labels=None,

                    prompt_lengths=(
                        full_inputs.get(
                            "prompt_lengths"
                        )
                    ),

                    reasoning_token_mask=(
                        full_inputs.get(
                            "reasoning_token_mask"
                        )
                    ),

                    selected_kv_layers=(
                        self.selected_kv_layers
                    ),

                    return_features=True,
                )

            if "layer_kv" not in outputs:

                raise RuntimeError(
                    "VLM did not return layer_kv. "
                    "Use your patched KV-extraction "
                    "VLM backbone."
                )

            sample_kv.append(
                outputs["layer_kv"]
            )

        return (
            self._stack_layer_kv(
                sample_kv
            ),
            generated_texts,
        )

    # ========================================================
    # Evaluation
    # ========================================================

    @torch.no_grad()
    def evaluate(self):

        self.vlm.eval()
        self.action_expert.eval()

        sums: Dict[str, float] = {}

        num_samples = 0

        sample_results = []

        start_time = time.time()

        for step, batch in enumerate(
            self.loader
        ):

            # ----------------------------------------------
            # VLM -> generated reasoning -> KV
            # ----------------------------------------------

            (
                layer_kv,
                generated_texts,
            ) = self._extract_generated_kv(
                batch
            )

            # ----------------------------------------------
            # Ego state
            # ----------------------------------------------

            ego_history = batch[
                "ego_history_trajectories"
            ].to(
                self.device
            )

            ego_velocities = batch[
                "ego_velocities"
            ].to(
                self.device
            )

            ego_accelerations = batch[
                "ego_accelerations"
            ].to(
                self.device
            )

            ego_state = torch.cat(
                [
                    ego_velocities,
                    ego_accelerations,
                ],
                dim=-1,
            )

            # ----------------------------------------------
            # Action Expert trajectory generation
            # ----------------------------------------------

            with torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):

                prediction = (
                    self.action_expert.sample(

                        layer_kv=layer_kv,

                        ego_state=ego_state,

                        history_trajectory=(
                            ego_history
                        ),

                        num_steps=(
                            self.args.num_inference_steps
                        ),
                    )
                )

            prediction = prediction.float()

            target = batch[
                "ego_trajectories"
            ].to(
                self.device
            ).float()

            # ----------------------------------------------
            # Metrics
            # ----------------------------------------------

            metrics = (
                compute_trajectory_metrics(
                    prediction,
                    target,
                )
            )

            bs = int(
                target.shape[0]
            )

            num_samples += bs

            for key, value in metrics.items():

                sums[key] = (
                    sums.get(
                        key,
                        0.0,
                    )
                    + float(value) * bs
                )

            # ----------------------------------------------
            # Save sample results
            # ----------------------------------------------

            pred_cpu = (
                prediction.detach()
                .cpu()
                .tolist()
            )

            target_cpu = (
                target.detach()
                .cpu()
                .tolist()
            )

            for i in range(bs):

                result = {

                    "clip":
                        batch[
                            "clip_names"
                        ][i],

                    "timestamp_us":
                        int(
                            batch[
                                "timestamps_us"
                            ][i]
                        ),

                    "generated_reasoning":
                        generated_texts[i],

                    "predicted_trajectory":
                        pred_cpu[i],

                    "gt_trajectory":
                        target_cpu[i],
                }

                sample_results.append(
                    result
                )

            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            if (
                (step + 1)
                % self.args.log_interval
                == 0
                or step + 1
                == len(self.loader)
            ):

                current_ade = (
                    sums.get(
                        "ADE_m",
                        0.0,
                    )
                    / max(
                        num_samples,
                        1,
                    )
                )

                current_fde = (
                    sums.get(
                        "FDE_m",
                        0.0,
                    )
                    / max(
                        num_samples,
                        1,
                    )
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                logger.info(
                    "Step %d/%d | "
                    "samples=%d | "
                    "ADE=%.4f m | "
                    "FDE=%.4f m | "
                    "elapsed=%.1f min",
                    step + 1,
                    len(self.loader),
                    num_samples,
                    current_ade,
                    current_fde,
                    elapsed / 60.0,
                )

        # ====================================================
        # Final metrics
        # ====================================================

        metrics = {
            key:
                value
                / max(
                    num_samples,
                    1,
                )
            for key, value
            in sums.items()
        }

        metrics[
            "num_test_samples"
        ] = num_samples

        metrics[
            "elapsed_seconds"
        ] = (
            time.time()
            - start_time
        )

        logger.info("=" * 72)

        logger.info(
            "TEST COMPLETE"
        )

        logger.info(
            "Samples       : %d",
            num_samples,
        )

        logger.info(
            "ADE           : %.4f m",
            metrics.get(
                "ADE_m",
                float("nan"),
            ),
        )

        logger.info(
            "FDE           : %.4f m",
            metrics.get(
                "FDE_m",
                float("nan"),
            ),
        )

        logger.info(
            "Heading error : %.4f deg",
            metrics.get(
                "heading_error_deg",
                float("nan"),
            ),
        )

        logger.info("=" * 72)

        # ====================================================
        # Save result
        # ====================================================

        output_dir = Path(
            self.args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_dir
            / "test_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=2,
            )

        with open(
            output_dir
            / "test_predictions.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                sample_results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Metrics saved: %s",
            output_dir
            / "test_metrics.json",
        )

        logger.info(
            "Predictions saved: %s",
            output_dir
            / "test_predictions.json",
        )


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
        default="/media/HDD/nuR_ds/data/test",
    )

    parser.add_argument(
        "--test_vqa_root",
        default="/media/HDD/nuR_ds/data/test",
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
    # Action Expert checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "--action_checkpoint",
        default=(
            "./outputs/"
            "action_expert_stage2/"
            "epoch_1/"
            "action_expert.pt"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "./outputs/"
            "action_expert_stage2/"
            "epoch_1/"
            "test_results"
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
                    "test_action_expert.log",
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