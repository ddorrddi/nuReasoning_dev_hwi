#!/usr/bin/env python3
"""
Stage-2 Action Expert training for nuVLA.

Pipeline:
    images + VQA prompt/answer
        -> pretrained VLM adapter
        -> frozen VLM, torch.no_grad()
        -> layer-wise reasoning K/V
        -> layer-matched Flow-Matching DiT
        -> trajectory loss
        -> backward ONLY through Action Expert

Important:
- The VLM optimizer does not exist in this script.
- Every VLM parameter has requires_grad=False.
- VLM remains in eval() mode during Action Expert training.
- K/V tensors are detached before entering the Action Expert.
- By default ALL VLM language layers are used, giving exact depth matching.
- KV source is GENERATED-ONLY.
- The frozen VLM generates reasoning autoregressively with use_cache=True.
- The exact past_key_values accumulated during generation are reused directly.
- Generated reasoning is NEVER re-tokenized and re-forwarded just to rebuild K/V.
- The current patched VLMBackbone returns full-context K/V for selected layers.
- Qwen3-VL rope_delta is computed from the exact same multimodal context.
- Expert cross-attention Q uses Qwen3-VL MRoPE to align with cached VLM K.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import math
import os
import time
from dataclasses import fields
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nureasoning.common.pretrained import resolve_pretrained_path
from nureasoning.nuvla.models.vlm_backbone import VLMBackbone, VLMBackboneConfig
from nureasoning.nuvla.models.data_loader_vqa import (
    VLADataConfig,
    NuReasoningVLADataset,
    build_dataloader,
    vla_collate_fn,
)

# Alpamayo-referenced layer-matched Action Expert.
# RoPE-aware 28-layer Action Expert.
from nureasoning.nuvla.models.action_expert_28layer import (
    ActionExpertConfig,
    FlowMatchingDiTActionExpert,
    compute_trajectory_metrics,
)

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


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
        # Allow directly passing .../final/vlm_adapter
        return path
    raise FileNotFoundError(f"VLM checkpoint/adapter directory not found: {path}")


class ActionExpertTrainer:
    def __init__(
        self,
        args: argparse.Namespace,
        local_rank: int,
        global_rank: int,
        world_size: int,
    ):
        self.args = args
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main = global_rank == 0
        self.device = torch.device("cuda", local_rank)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")

        self.use_bf16 = True
        self.cameras = parse_camera_list(args.cameras)

        self._build_frozen_vlm()
        self._build_action_expert()
        self._build_data()
        self._build_optimizer()

        self.global_step = 0
        self.best_ade = float("inf")

        if self.is_main:
            os.makedirs(args.output_dir, exist_ok=True)
            self._save_config()

        if self.is_distributed:
            dist.barrier()

    @property
    def action_unwrapped(self) -> FlowMatchingDiTActionExpert:
        return self.action_expert.module if self.is_distributed else self.action_expert

    def _build_frozen_vlm(self):
        args = self.args
        args.vlm_model_path = resolve_pretrained_path(args.vlm_model_path)

        config_kwargs = dict(
            model_name_or_path=args.vlm_model_path,
            freeze_vision_encoder=False,
            lora_rank=args.vlm_lora_rank,
            lora_alpha=args.vlm_lora_alpha,
            lora_dropout=args.vlm_lora_dropout,
            current_resolution=(args.current_res_w, args.current_res_h),
            history_resolution=(args.history_res_w, args.history_res_h),
            reasoning_format=args.reasoning_format,
        )

        vlm_fields = dataclass_field_names(VLMBackboneConfig)
        if "load_in_4bit" in vlm_fields:
            config_kwargs["load_in_4bit"] = args.load_in_4bit
        elif args.load_in_4bit:
            raise RuntimeError(
                "VLMBackboneConfig has no load_in_4bit field. "
                "Use the patched VLM backbone."
            )

        self.vlm = VLMBackbone(VLMBackboneConfig(**config_kwargs))
        if not args.load_in_4bit:
            self.vlm = self.vlm.to(self.device)

        adapter_dir = resolve_adapter_dir(args.vlm_checkpoint)
        logger.info("Loading pretrained VLM adapter: %s", adapter_dir)
        self.vlm.load_adapter(adapter_dir, device=self.device)

        # Freeze AFTER loading the adapter.
        for p in self.vlm.parameters():
            p.requires_grad_(False)
        self.vlm.eval()

        # No activation checkpointing is needed for a frozen no-grad VLM.
        base_model = self.vlm.model
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        if hasattr(base_model.config, "use_cache"):
            # The custom selected_kv_layers path explicitly returns layer K/V.
            # Keep HF generation cache off unless the backbone requires it.
            base_model.config.use_cache = False

        trainable = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
        if trainable != 0:
            raise RuntimeError(f"Frozen VLM still has {trainable:,} trainable parameters.")

        (
            self.vlm_num_layers,
            self.kv_num_heads,
            self.kv_head_dim,
        ) = self.vlm.get_kv_spec()

        if args.kv_layers.strip().lower() == "all":
            self.selected_kv_layers = list(range(self.vlm_num_layers))
        else:
            requested_1based = [
                int(x.strip()) for x in args.kv_layers.split(",") if x.strip()
            ]
            if not requested_1based:
                raise ValueError("--kv_layers must be 'all' or a comma-separated list.")
            if min(requested_1based) < 1 or max(requested_1based) > self.vlm_num_layers:
                raise ValueError(
                    f"Invalid --kv_layers={requested_1based}; "
                    f"VLM has {self.vlm_num_layers} layers."
                )
            self.selected_kv_layers = [x - 1 for x in requested_1based]

        if self.is_main:
            logger.info("Frozen VLM total params: %s", f"{sum(p.numel() for p in self.vlm.parameters()):,}")
            logger.info("Frozen VLM trainable params: 0")
            logger.info(
                "VLM KV spec: layers=%d, kv_heads=%d, head_dim=%d",
                self.vlm_num_layers,
                self.kv_num_heads,
                self.kv_head_dim,
            )
            logger.info(
                "Selected KV layers (1-based): %s",
                [x + 1 for x in self.selected_kv_layers],
            )

    def _build_action_expert(self):
        args = self.args

        # In exact layer-matched mode the planner depth is defined by the
        # number of selected VLM layers. With --kv_layers all this is 28.
        num_layers = len(self.selected_kv_layers)

        config = ActionExpertConfig(
            vlm_feature_dim=self.vlm.feature_dim,
            ego_state_dim=4,
            max_history_traj_points=args.max_history_traj_points,
            num_waypoints=args.num_trajectory_points,
            trajectory_dim=3,
            hidden_dim=args.action_hidden_dim,
            num_heads=args.num_dit_heads,
            num_dit_layers=num_layers,
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

        self.action_expert = FlowMatchingDiTActionExpert(config).to(self.device)

        if self.is_distributed:
            self.action_expert = DDP(
                self.action_expert,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        if self.is_main:
            total = sum(p.numel() for p in self.action_unwrapped.parameters())
            trainable = sum(
                p.numel() for p in self.action_unwrapped.parameters()
                if p.requires_grad
            )
            logger.info("Action Expert layers: %d", num_layers)
            logger.info("Action Expert params: %s", f"{total:,}")
            logger.info("Action Expert trainable: %s", f"{trainable:,}")

    def _data_config(
        self,
        data_root: str,
        vqa_root: str,
    ) -> VLADataConfig:
        args = self.args
        kwargs = dict(
            data_root=data_root,
            num_history_steps=args.num_history_steps,
            history_stride=args.history_stride,
            current_resolution=(args.current_res_w, args.current_res_h),
            history_resolution=(args.history_res_w, args.history_res_h),
            trajectory_future_seconds=args.trajectory_future_seconds,
            frame_rate_hz=args.frame_rate_hz,
            num_waypoints=args.num_trajectory_points,
            max_history_traj_points=args.max_history_traj_points,
            reasoning_format=args.reasoning_format,
            reasoning_max_items_per_list=args.reasoning_max_items_per_list,
            reasoning_mode="vqa",
            vqa_root=vqa_root,
            qa_seed=args.qa_seed,
        )

        data_fields = dataclass_field_names(VLADataConfig)
        if "cameras" not in data_fields:
            raise RuntimeError("Current VLADataConfig must support camera selection.")
        kwargs["cameras"] = list(self.cameras)
        return VLADataConfig(**kwargs)

    def _build_data(self):
        args = self.args

        train_cfg = self._data_config(args.data_root, args.vqa_root)
        train_dataset = NuReasoningVLADataset(train_cfg, split="train")

        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=self.train_sampler,
                num_workers=args.num_workers,
                collate_fn=vla_collate_fn,
                pin_memory=True,
                drop_last=False,
                persistent_workers=args.num_workers > 0,
            )
        else:
            self.train_sampler = None
            self.train_loader = build_dataloader(
                train_cfg,
                split="train",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )

        self.val_loader = None
        self.val_sampler = None
        if args.test_data_root:
            val_cfg = self._data_config(args.test_data_root, args.test_vqa_root)
            val_dataset = NuReasoningVLADataset(val_cfg, split="test")

            if self.is_distributed:
                self.val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    shuffle=False,
                    drop_last=False,
                )
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    sampler=self.val_sampler,
                    num_workers=args.num_workers,
                    collate_fn=vla_collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=args.num_workers > 0,
                )
            else:
                self.val_loader = build_dataloader(
                    val_cfg,
                    split="test",
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                )

        if self.is_main:
            logger.info("Training samples: %d", len(self.train_loader.dataset))
            if self.val_loader is not None:
                logger.info("Validation samples: %d", len(self.val_loader.dataset))

    def _build_optimizer(self):
        args = self.args

        self.optimizer = AdamW(
            self.action_unwrapped.parameters(),
            lr=args.action_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

        steps_per_epoch = math.ceil(
            len(self.train_loader) / args.gradient_accumulation_steps
        )
        total_steps = max(steps_per_epoch * args.epochs, 1)
        warmup_steps = int(total_steps * args.warmup_ratio)

        min_ratio = min(args.min_lr / max(args.action_lr, 1e-12), 1.0)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if total_steps <= warmup_steps:
                return 1.0

            progress = (
                (step - warmup_steps)
                / float(max(total_steps - warmup_steps, 1))
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        if self.is_main:
            logger.info(
                "Scheduler warmup+cosine: warmup=%d total=%d",
                warmup_steps,
                total_steps,
            )

    def _save_config(self):
        payload = vars(self.args).copy()
        payload.update(
            {
                "resolved_cameras": list(self.cameras),
                "vlm_num_layers": self.vlm_num_layers,
                "selected_kv_layers_1based": [
                    x + 1 for x in self.selected_kv_layers
                ],
                "kv_num_heads": self.kv_num_heads,
                "kv_head_dim": self.kv_head_dim,
                "vlm_frozen": True,
                "lora_rank": self.args.vlm_lora_rank,
                "lora_alpha": self.args.vlm_lora_alpha,
                "lora_dropout": self.args.vlm_lora_dropout,
                "kv_source": "generated_live_cache",
                "generated_kv_reuse": True,
                "generated_reasoning_reforward": False,
                "expert_rope_alignment": "qwen3vl_mrope_alpamayo_style",
                "qk_norm_eps": self.args.qk_norm_eps,
            }
        )
        with open(
            os.path.join(self.args.output_dir, "training_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, indent=2, default=str)

    def _sample_prompt_and_context(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> tuple[list[Any], list[str], str]:
        images = [img for img in batch["images"][batch_idx] if img is not None]

        expected = len(self.cameras) * (self.args.num_history_steps + 1)
        if len(images) != expected:
            raise RuntimeError(
                f"Expected {expected} images/sample, got {len(images)}."
            )

        image_contexts = []
        for t in range(-self.args.num_history_steps, 1):
            t_label = f"t={t}" if t < 0 else "t=0 (current)"
            for cam in self.cameras:
                image_contexts.append(f"{t_label}, {cam} camera.")

        prompts = batch.get("user_prompts")
        prompt = None
        if prompts is not None and batch_idx < len(prompts):
            prompt = prompts[batch_idx]

        if not prompt:
            prompt = self.vlm.build_multiview_prompt(
                num_history_steps=self.args.num_history_steps,
                num_current_cameras=len(self.cameras),
                mission_command=batch["mission_commands"][batch_idx],
            )

        return images, image_contexts, prompt

    def _prepare_single_vlm_input(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        *,
        assistant_response: Optional[str],
    ) -> Dict[str, Any]:
        images, image_contexts, prompt = self._sample_prompt_and_context(
            batch,
            batch_idx,
        )

        inputs = self.vlm.prepare_inputs(
            images=images,
            text_prompt=prompt,
            image_contexts=image_contexts,
            assistant_response=assistant_response,
        )

        result: Dict[str, Any] = {}
        for key, value in inputs.items():
            result[key] = (
                value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
            )

        if "prompt_length" in result:
            result["prompt_lengths"] = result.pop(
                "prompt_length"
            ).to(self.device)

        return result

    @staticmethod
    def _cache_seq_len(
        past_key_values: Any,
        layer_idx: int = 0,
    ) -> int:
        """Read current autoregressive KV-cache length."""
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
        raise TypeError(f"Unsupported generation cache type: {type(past_key_values)!r}")

    @torch.no_grad()
    def _generate_live_cache_for_sample(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> tuple[
        Dict[int, Dict[str, torch.Tensor]],
        Dict[str, Any],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        GENERATED-ONLY Alpamayo-style bridge.

        image + prompt
            -> frozen Qwen3-VL autoregressive generation
            -> K/V cache grows inside generate()
            -> return the SAME generation past_key_values
            -> NO decode/re-tokenize/re-forward step
        """
        prompt_input = self._prepare_single_vlm_input(
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
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            value = prompt_input.get(key)
            if value is not None:
                generation_kwargs[key] = value

        with torch.amp.autocast(
            "cuda",
            enabled=self.use_bf16,
            dtype=torch.bfloat16,
        ):
            generation_output = self.vlm.model.generate(**generation_kwargs)

        generated_sequences = generation_output.sequences
        past_key_values = getattr(generation_output, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError(
                "generate() returned no past_key_values. "
                "Need return_dict_in_generate=True and use_cache=True support."
            )

        prompt_len = int(prompt_input["input_ids"].shape[1])
        total_len = int(generated_sequences.shape[1])
        generated_len = total_len - prompt_len
        if generated_len <= 0:
            raise RuntimeError("Frozen VLM generated zero reasoning tokens.")

        cache_len = self._cache_seq_len(
            past_key_values,
            self.selected_kv_layers[0],
        )
        if cache_len <= 0 or cache_len > total_len:
            raise RuntimeError(
                f"Invalid generation cache length: cache={cache_len}, sequence={total_len}"
            )

        generated_attention_mask = torch.ones(
            generated_sequences.shape,
            device=generated_sequences.device,
            dtype=prompt_input["attention_mask"].dtype,
        )
        generated_attention_mask[:, :prompt_len] = prompt_input["attention_mask"]

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
            entry["mask"] = generated_attention_mask[:, :layer_cache_len].bool().detach()
            entry["key"] = key.detach()
            entry["value"] = value.detach()

        if self.is_main and self.global_step == 0:
            logger.info(
                "Generated live KV | prompt=%d generated=%d sequence=%d cache=%d",
                prompt_len,
                generated_len,
                total_len,
                cache_len,
            )

        return layer_kv, prompt_input, generated_sequences, generated_attention_mask

    def _resolve_qwen_text_model(self):
        """
        Resolve the Qwen3-VL multimodal base model that owns get_rope_index().

        This deliberately unwraps PEFT/DDP-style wrappers at runtime so the
        already-trained VLM checkpoint and vlm_backbone.py do not need changes.
        """
        queue = [getattr(self.vlm, "model", None)]
        seen = set()

        while queue:
            obj = queue.pop(0)
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))

            if hasattr(obj, "get_rope_index"):
                return obj

            get_base_model = getattr(obj, "get_base_model", None)
            if callable(get_base_model):
                try:
                    queue.append(get_base_model())
                except Exception:
                    pass

            for name in ("model", "base_model", "module"):
                child = getattr(obj, name, None)
                if child is not None:
                    queue.append(child)

        raise RuntimeError(
            "Could not resolve the underlying Qwen3-VL model exposing "
            "get_rope_index(). The frozen checkpoint itself is not the issue; "
            "inspect the installed PEFT/transformers wrapper layout."
        )

    def _resolve_qwen_rotary_emb(self):
        """
        Resolve the loaded Qwen3-VL language-model rotary embedding.

        Uses the already-loaded frozen VLM; no VLM parameter or checkpoint is
        changed. Qwen remains the source of truth for RoPE/MRoPE settings.
        """
        qwen = self._resolve_qwen_text_model()

        candidates = [
            getattr(qwen, "language_model", None),
            qwen,
        ]

        model_child = getattr(qwen, "model", None)
        if model_child is not None:
            candidates.extend([
                getattr(model_child, "language_model", None),
                model_child,
            ])

        for module in candidates:
            if module is None:
                continue
            rotary_emb = getattr(module, "rotary_emb", None)
            if rotary_emb is not None:
                return rotary_emb

        raise RuntimeError(
            "Could not resolve Qwen3-VL language_model.rotary_emb from the "
            "already-loaded frozen VLM."
        )

    @torch.no_grad()
    def _compute_qwen_rope_delta(
        self,
        sample_input: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute Qwen3-VL multimodal rope_delta for one prepared sample.

        Qwen3-VL get_rope_index() returns the multimodal position ids plus the
        delta used to continue later tokens in the same MRoPE coordinate frame.
        """
        qwen = self._resolve_qwen_text_model()

        mm_token_type_ids = sample_input.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            raise RuntimeError(
                "Qwen3-VL MRoPE alignment requires mm_token_type_ids, "
                "but prepare_inputs() did not return it."
            )

        _, rope_delta = qwen.get_rope_index(
            input_ids=sample_input["input_ids"],
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=sample_input.get("image_grid_thw"),
            video_grid_thw=sample_input.get("video_grid_thw"),
            attention_mask=sample_input["attention_mask"],
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Alpamayo-style Expert continuation positions.

        Expert tokens begin directly after all valid VLM context tokens:

            expert_position
              = valid_context_length + rope_delta + local_expert_index

        For these new text-like Expert tokens, the same scalar position is used
        on Qwen3-VL's temporal/height/width RoPE axes.

        Returns
        -------
        cos          : [1, S_expert, Dh]
        sin          : [1, S_expert, Dh]
        position_ids : [3, 1, S_expert]
        """
        attention_mask = sample_input["attention_mask"]
        if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
            raise RuntimeError(
                "Expected single-sample attention_mask [1,T], got "
                f"{tuple(attention_mask.shape)}"
            )

        valid_len = attention_mask.long().sum(dim=-1)
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

        rotary_emb = self._resolve_qwen_rotary_emb()
        dummy = torch.empty(
            (1, num_expert_tokens, 1),
            device=device,
            dtype=dtype,
        )

        pos = position_ids.to(device)

        # Qwen3-VL releases differ in whether the language rotary module sees
        # 3 MRoPE axes directly or a leading text-position axis as well.
        # Try the native 3-axis continuation first, then a 4-axis compatible
        # form without changing the underlying VLM implementation.
        rope_error = None
        try:
            cos, sin = rotary_emb(dummy, pos)
        except Exception as exc:
            rope_error = exc
            text_pos = pos[:1]
            pos4 = torch.cat([text_pos, pos], dim=0)
            try:
                cos, sin = rotary_emb(dummy, pos4)
            except Exception as exc4:
                raise RuntimeError(
                    "Failed to generate Expert RoPE with the loaded Qwen3-VL "
                    f"rotary_emb. 3-axis error={rope_error!r}; "
                    f"4-axis error={exc4!r}"
                ) from exc4

        if cos.ndim != 3 or sin.ndim != 3:
            raise RuntimeError(
                "Unexpected Qwen rotary output: "
                f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            )

        if cos.shape[-1] != self.kv_head_dim:
            raise RuntimeError(
                "Qwen RoPE dimension does not match VLM K/V head dim: "
                f"rope={cos.shape[-1]}, kv_head_dim={self.kv_head_dim}"
            )

        return (
            cos.detach().to(device=device, dtype=dtype),
            sin.detach().to(device=device, dtype=dtype),
            position_ids.detach().to(device),
        )

    @staticmethod
    def _stack_layer_kv(
        layer_kv_list: list[Dict[int, Dict[str, torch.Tensor]]],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Pad variable-length VLM KV caches and combine samples into a batch.

        Input per sample:
            layer_idx -> {
                "key":   [1, Hkv, T, Dh],
                "value": [1, Hkv, T, Dh],
                "mask":  [1, T]   # optional
            }

        Current full-context VLM KV extraction may return only key/value.
        In that case every original KV token is valid, so an all-True mask is
        generated here. Padding added while batching is marked False.

        Output:
            layer_idx -> {
                "key":   [B, Hkv, Tmax, Dh],
                "value": [B, Hkv, Tmax, Dh],
                "mask":  [B, Tmax]
            }
        """
        if not layer_kv_list:
            raise ValueError("layer_kv_list is empty.")

        layer_ids = sorted(layer_kv_list[0].keys())

        for sample in layer_kv_list[1:]:
            if sorted(sample.keys()) != layer_ids:
                raise RuntimeError(
                    "Samples expose different VLM KV layers."
                )

        batched: Dict[int, Dict[str, torch.Tensor]] = {}

        for layer_idx in layer_ids:
            # -------------------------------------------------------------
            # Find longest KV sequence for this VLM layer in the batch
            # -------------------------------------------------------------
            max_len = max(
                sample[layer_idx]["key"].shape[2]
                for sample in layer_kv_list
            )

            keys = []
            values = []
            masks = []

            for sample in layer_kv_list:
                entry = sample[layer_idx]

                key = entry["key"]
                value = entry["value"]

                if key.ndim != 4 or value.ndim != 4:
                    raise RuntimeError(
                        f"Expected KV [B,H,T,D], "
                        f"got K={tuple(key.shape)}, "
                        f"V={tuple(value.shape)} "
                        f"at VLM layer {layer_idx}"
                    )

                if key.shape != value.shape:
                    raise RuntimeError(
                        f"K/V shape mismatch at layer {layer_idx}: "
                        f"K={tuple(key.shape)}, "
                        f"V={tuple(value.shape)}"
                    )

                current_len = key.shape[2]

                # ---------------------------------------------------------
                # Full-context KV path does not necessarily return a mask.
                # Before batching, every token in this sample is valid.
                # ---------------------------------------------------------
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

                    if mask.shape != (
                        key.shape[0],
                        current_len,
                    ):
                        raise RuntimeError(
                            f"Invalid KV mask shape at layer {layer_idx}: "
                            f"mask={tuple(mask.shape)}, "
                            f"expected={(key.shape[0], current_len)}"
                        )

                # ---------------------------------------------------------
                # Pad this sample to Tmax
                # ---------------------------------------------------------
                pad_len = max_len - current_len

                if pad_len > 0:
                    key = torch.cat(
                        [
                            key,
                            key.new_zeros(
                                key.shape[0],
                                key.shape[1],
                                pad_len,
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
                                pad_len,
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
                                pad_len,
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
                "key": torch.cat(keys, dim=0).detach(),
                "value": torch.cat(values, dim=0).detach(),
                "mask": torch.cat(masks, dim=0).detach(),
            }

        return batched

    def _extract_frozen_vlm_context(
        self,
        batch: Dict[str, Any],
    ) -> tuple[
        Dict[int, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        GENERATED-ONLY context extraction.

        Frozen VLM reasoning is generated autoregressively once, and the exact
        generation past_key_values are handed to the Action Expert. There is no
        teacher forcing and no second full-context VLM forward.
        """
        self.vlm.eval()
        layer_kv_list = []
        rope_cos_list = []
        rope_sin_list = []
        num_expert_tokens = self.action_unwrapped.num_expert_tokens

        with torch.no_grad():
            for b in range(len(batch["images"])):
                sample_kv, prompt_input, _, _ = self._generate_live_cache_for_sample(batch, b)

                first_idx = self.selected_kv_layers[0]
                first_k = sample_kv[first_idx]["key"]
                if first_k.shape[1] != self.kv_num_heads:
                    raise RuntimeError(
                        f"KV head mismatch: expected {self.kv_num_heads}, got {first_k.shape[1]}"
                    )
                if first_k.shape[-1] != self.kv_head_dim:
                    raise RuntimeError(
                        f"KV head_dim mismatch: expected {self.kv_head_dim}, got {first_k.shape[-1]}"
                    )

                # The generation K cache is already post-RoPE. Build Expert Q
                # positions as a continuation of the actual live cache length.
                rope_delta = self._compute_qwen_rope_delta(prompt_input)
                cache_len = int(first_k.shape[-2])
                rope_input = dict(prompt_input)
                rope_input["attention_mask"] = torch.ones(
                    (1, cache_len),
                    device=first_k.device,
                    dtype=prompt_input["attention_mask"].dtype,
                )

                rope_cos, rope_sin, _ = self._build_expert_rope_for_sample(
                    rope_input,
                    rope_delta=rope_delta,
                    num_expert_tokens=num_expert_tokens,
                    dtype=first_k.dtype,
                    device=first_k.device,
                )

                layer_kv_list.append(sample_kv)
                rope_cos_list.append(rope_cos)
                rope_sin_list.append(rope_sin)

        layer_kv = self._stack_layer_kv(layer_kv_list)
        rope_cos = torch.cat(rope_cos_list, dim=0).detach()
        rope_sin = torch.cat(rope_sin_list, dim=0).detach()
        return layer_kv, rope_cos, rope_sin

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.action_expert.train()
        self.vlm.eval()

        layer_kv, rope_cos, rope_sin = self._extract_frozen_vlm_context(batch)

        ego_trajectories = batch["ego_trajectories"].to(
            self.device, non_blocking=True
        )
        ego_history = batch["ego_history_trajectories"].to(
            self.device, non_blocking=True
        )
        ego_velocities = batch["ego_velocities"].to(
            self.device, non_blocking=True
        )
        ego_accelerations = batch["ego_accelerations"].to(
            self.device, non_blocking=True
        )
        ego_state = torch.cat(
            [ego_velocities, ego_accelerations],
            dim=-1,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=self.use_bf16,
            dtype=torch.bfloat16,
        ):
            outputs = self.action_expert(
                x_1=ego_trajectories,
                layer_kv=layer_kv,
                ego_state=ego_state,
                history_trajectory=ego_history,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            loss = outputs["loss"]

        # Correct gradient accumulation scaling.
        (
            loss / self.args.gradient_accumulation_steps
        ).backward()

        return {
            "action_loss": float(loss.detach().item()),
            "mse_x": float(outputs["mse_x"].detach().item()),
            "mse_y": float(outputs["mse_y"].detach().item()),
            "mse_theta": float(outputs["mse_theta"].detach().item()),
        }

    def optimizer_step(self):
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.action_unwrapped.parameters(),
                self.args.max_grad_norm,
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1

    @torch.no_grad()
    def evaluate(self) -> Optional[Dict[str, float]]:
        if self.val_loader is None:
            return None

        self.vlm.eval()
        self.action_unwrapped.eval()

        sums: Dict[str, float] = {}
        n = 0

        for batch in self.val_loader:
            layer_kv, rope_cos, rope_sin = self._extract_frozen_vlm_context(batch)

            ego_history = batch["ego_history_trajectories"].to(self.device)
            ego_velocities = batch["ego_velocities"].to(self.device)
            ego_accelerations = batch["ego_accelerations"].to(self.device)
            ego_state = torch.cat(
                [ego_velocities, ego_accelerations],
                dim=-1,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=self.use_bf16,
                dtype=torch.bfloat16,
            ):
                pred = self.action_unwrapped.sample(
                    layer_kv=layer_kv,
                    ego_state=ego_state,
                    history_trajectory=ego_history,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    num_steps=self.args.num_inference_steps,
                )

            target = batch["ego_trajectories"].to(self.device).float()
            metrics = compute_trajectory_metrics(pred.float(), target)
            bs = int(target.shape[0])
            n += bs

            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * bs

        if self.is_distributed:
            keys = sorted(sums.keys())
            packed = torch.tensor(
                [sums[k] for k in keys] + [float(n)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            vals = packed.tolist()
            sums = {k: vals[i] for i, k in enumerate(keys)}
            n = int(round(vals[-1]))

        if n == 0:
            return None

        result = {k: v / n for k, v in sums.items()}
        result["num_eval_samples"] = float(n)
        return result

    def save_checkpoint(
        self,
        epoch: int,
        *,
        is_final: bool = False,
        is_best: bool = False,
    ):
        if not self.is_main:
            if self.is_distributed:
                dist.barrier()
            return

        if is_best:
            name = "best"
        elif is_final:
            name = "final"
        else:
            name = f"epoch_{epoch}"

        save_dir = os.path.join(self.args.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {k: v.cpu() for k, v in self.action_unwrapped.state_dict().items()},
            os.path.join(save_dir, "action_expert.pt"),
        )
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": epoch,
                "best_ade": self.best_ade,
            },
            os.path.join(save_dir, "training_state.pt"),
        )

        logger.info("Saved Action Expert checkpoint: %s", save_dir)

        if self.is_distributed:
            dist.barrier()

    def load_checkpoint(self, checkpoint_dir: str):
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        action_path = os.path.join(checkpoint_dir, "action_expert.pt")
        state_path = os.path.join(checkpoint_dir, "training_state.pt")

        if not os.path.isfile(action_path):
            raise FileNotFoundError(action_path)

        self.action_unwrapped.load_state_dict(
            torch.load(action_path, map_location=self.device)
        )

        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = int(state.get("global_step", 0))
            self.best_ade = float(state.get("best_ade", float("inf")))

        logger.info(
            "Resumed Action Expert from %s, global_step=%d",
            checkpoint_dir,
            self.global_step,
        )

        if self.is_distributed:
            dist.barrier()

    def train(self):
        args = self.args
        self.optimizer.zero_grad(set_to_none=True)

        if self.is_main:
            logger.info("=" * 76)
            logger.info("Stage-2 Action Expert training")
            logger.info("VLM: FROZEN")
            logger.info("VLM checkpoint: %s", args.vlm_checkpoint)
            logger.info("KV layers: %s", [x + 1 for x in self.selected_kv_layers])
            logger.info("KV source: generated live autoregressive cache ONLY")
            logger.info(
                "Expert RoPE: Qwen3-VL MRoPE continuation "
                "(Alpamayo-style positional alignment)"
            )
            logger.info(
                "VLM weights/backbone: unchanged; RoPE is read at runtime "
                "from the already-trained frozen VLM"
            )
            logger.info("Action depth: %d", len(self.selected_kv_layers))
            logger.info(
                "Action width=%d heads=%d mlp_ratio=%.2f self_attn_every=%d",
                args.action_hidden_dim,
                args.num_dit_heads,
                args.mlp_ratio,
                args.self_attention_every,
            )
            logger.info(
                "Effective batch size: %d",
                args.batch_size
                * self.world_size
                * args.gradient_accumulation_steps,
            )
            logger.info("=" * 76)

        for epoch in range(args.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            running_loss = 0.0
            batches = 0
            log_start = time.time()

            for step, batch in enumerate(self.train_loader):
                is_last = step + 1 == len(self.train_loader)
                is_sync = (
                    (step + 1) % args.gradient_accumulation_steps == 0
                ) or is_last

                if self.is_distributed and not is_sync:
                    sync_ctx = self.action_expert.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()

                with sync_ctx:
                    metrics = self.train_step(batch)

                running_loss += metrics["action_loss"]
                batches += 1

                if is_sync:
                    self.optimizer_step()

                if self.is_main and (
                    (step + 1) % args.log_interval == 0 or is_last
                ):
                    elapsed = time.time() - epoch_start
                    rate = (step + 1) / max(elapsed, 1e-6)
                    remaining = len(self.train_loader) - (step + 1)
                    eta_sec = remaining / max(rate, 1e-6)

                    logger.info(
                        "Epoch %d/%d | Step %d/%d | "
                        "Action Loss %.4f | x_mse %.4f | y_mse %.4f | "
                        "LR %.2e | ETA %.1f min",
                        epoch + 1,
                        args.epochs,
                        step + 1,
                        len(self.train_loader),
                        running_loss / max(batches, 1),
                        metrics["mse_x"],
                        metrics["mse_y"],
                        self.scheduler.get_last_lr()[0],
                        eta_sec / 60.0,
                    )
                    log_start = time.time()

            if self.is_main:
                logger.info(
                    "Epoch %d complete in %.1f min | avg action loss %.4f",
                    epoch + 1,
                    (time.time() - epoch_start) / 60.0,
                    running_loss / max(batches, 1),
                )

            eval_metrics = None
            if (
                self.val_loader is not None
                and args.eval_interval > 0
                and (epoch + 1) % args.eval_interval == 0
            ):
                eval_metrics = self.evaluate()

                if self.is_main and eval_metrics is not None:
                    logger.info(
                        "[Validation] ADE %.3f m | FDE %.3f m | Heading %.2f deg | n=%d",
                        eval_metrics["ADE_m"],
                        eval_metrics["FDE_m"],
                        eval_metrics["heading_error_deg"],
                        int(eval_metrics["num_eval_samples"]),
                    )

                    if eval_metrics["ADE_m"] < self.best_ade:
                        self.best_ade = eval_metrics["ADE_m"]
                        self.save_checkpoint(epoch + 1, is_best=True)

            if (
                args.save_interval > 0
                and (epoch + 1) % args.save_interval == 0
            ):
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(args.epochs, is_final=True)

        if self.is_main:
            logger.info("Action Expert training complete.")


def validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    for path_name in ("data_root", "vqa_root"):
        path = getattr(args, path_name)
        if not path or not os.path.isdir(os.path.expanduser(path)):
            parser.error(f"--{path_name} is not a directory: {path}")

    if args.test_data_root:
        if not os.path.isdir(os.path.expanduser(args.test_data_root)):
            parser.error(
                f"--test_data_root is not a directory: {args.test_data_root}"
            )
        if not args.test_vqa_root:
            parser.error(
                "--test_data_root requires --test_vqa_root for the current VQA dataset loader."
            )
        if not os.path.isdir(os.path.expanduser(args.test_vqa_root)):
            parser.error(
                f"--test_vqa_root is not a directory: {args.test_vqa_root}"
            )

    try:
        resolve_adapter_dir(args.vlm_checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if args.action_hidden_dim % args.num_dit_heads != 0:
        parser.error("--action_hidden_dim must be divisible by --num_dit_heads.")

    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be >= 1.")

    if args.generation_max_new_tokens < 1:
        parser.error("--generation_max_new_tokens must be >= 1.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen-VLM Stage-2 Action Expert training for nuVLA"
    )

    # Current server defaults from the VLM-VQA stage.
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/HDD/nuR_ds/data/train",
    )
    parser.add_argument(
        "--vqa_root",
        type=str,
        default="/home/lhh/lab/dataset/nuReasoning/vqa_filtered",
    )
    parser.add_argument("--test_data_root", type=str, default=None)
    parser.add_argument("--test_vqa_root", type=str, default=None)

    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        required=True,
        help="VLM-only pretraining checkpoint, e.g. outputs/vlm_pretrain/final",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/action_expert_stage2",
    )

    # Must match the VLM pretraining adapter config.
    parser.add_argument("--vlm_lora_rank", type=int, default=16)
    parser.add_argument("--vlm_lora_alpha", type=int, default=32)
    parser.add_argument("--vlm_lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--cameras",
        type=str,
        default="front,front_left,front_right",
    )
    parser.add_argument("--num_history_steps", type=int, default=1)
    parser.add_argument("--history_stride", type=int, default=10)
    parser.add_argument("--current_res_w", type=int, default=448)
    parser.add_argument("--current_res_h", type=int, default=448)
    parser.add_argument("--history_res_w", type=int, default=448)
    parser.add_argument("--history_res_h", type=int, default=448)

    parser.add_argument(
        "--reasoning_format",
        type=str,
        default="spatial_driving_counterfactual",
    )
    parser.add_argument("--reasoning_max_items_per_list", type=int, default=10)
    parser.add_argument("--qa_seed", type=int, default=42)

    parser.add_argument("--trajectory_future_seconds", type=float, default=5.0)
    parser.add_argument("--frame_rate_hz", type=float, default=10.0)
    parser.add_argument("--num_trajectory_points", type=int, default=10)
    parser.add_argument("--max_history_traj_points", type=int, default=6)

    # Exact 28<->28 by default on a 28-layer VLM.
    parser.add_argument(
        "--kv_layers",
        type=str,
        default="all",
        help="'all' for exact VLM-depth matching, or 1-based comma-separated layers.",
    )

    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=256,
        help="Maximum reasoning tokens generated before handing the live KV cache to the Action Expert.",
    )

    # Compact 28-layer planner defaults.
    parser.add_argument("--action_hidden_dim", type=int, default=384)
    parser.add_argument("--num_dit_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--qk_norm_eps",
        type=float,
        default=1e-6,
        help="RMSNorm epsilon for planner Q before Qwen3-VL MRoPE.",
    )
    parser.add_argument(
        "--self_attention_every",
        type=int,
        default=4,
        help="Extra planner self-attention every N matched blocks; 0 disables.",
    )

    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--num_timestep_buckets", type=int, default=1000)
    parser.add_argument("--noise_beta_alpha", type=float, default=1.5)
    parser.add_argument("--noise_beta_beta", type=float, default=2.5)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--action_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)
    args.vqa_root = os.path.expanduser(args.vqa_root)
    args.test_data_root = (
        os.path.expanduser(args.test_data_root)
        if args.test_data_root
        else None
    )
    args.test_vqa_root = (
        os.path.expanduser(args.test_vqa_root)
        if args.test_vqa_root
        else None
    )
    args.vlm_checkpoint = os.path.expanduser(args.vlm_checkpoint)
    args.output_dir = os.path.expanduser(args.output_dir)

    validate_args(args, parser)
    return args


def main():
    local_rank, global_rank, world_size = setup_distributed()
    args = parse_args()

    if args.load_in_4bit and world_size > 1:
        raise RuntimeError(
            "Use single-GPU for the current 4-bit frozen VLM configuration."
        )

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO if global_rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            *(
                [
                    logging.FileHandler(
                        os.path.join(args.output_dir, "train_action_expert.log"),
                        mode="a",
                    )
                ]
                if global_rank == 0
                else []
            ),
        ],
    )

    if global_rank == 0:
        logger.info(
            "Arguments: %s",
            json.dumps(vars(args), indent=2, default=str),
        )

    trainer = ActionExpertTrainer(
        args=args,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
    )

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.train()
    cleanup_distributed()


if __name__ == "__main__":
    main()
