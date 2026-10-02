#!/usr/bin/env python3
"""
VLM-only pretraining for nuReasoning / nuVLA.

Purpose
-------
Train only the Qwen3-VL backbone with reasoning-text supervision.
No Action Expert is constructed, no trajectory loss is computed, and no
layer-wise KV cache is extracted during this stage.

Default setup for the current experiment:
- Base: Qwen/Qwen3-VL-2B-Instruct
- Cameras: front, front_left, front_right
- num_history_steps=1 -> 3 current + 3 history = 6 images/sample
- Current/history resolution: 448x448
- 4-bit Q-LoRA when --load_in_4bit is passed
- LoRA: rank=16, alpha=32, dropout=0.05 by default below
"""

import argparse
import contextlib
import datetime
import inspect
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

logger = logging.getLogger(__name__)

REASONING_MODE_STRUCTURED = "structured"
REASONING_MODE_VQA = "vqa"
REASONING_MODES = (REASONING_MODE_STRUCTURED, REASONING_MODE_VQA)


def setup_distributed() -> tuple:
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


def parse_camera_list(raw: str) -> list[str]:
    cameras = [x.strip() for x in raw.split(",") if x.strip()]
    if not cameras:
        raise ValueError("--cameras must contain at least one camera")

    valid = {
        "front",
        "front_left",
        "front_right",
        "left",
        "right",
        "back",
        "back_left",
        "back_right",
    }
    unknown = [x for x in cameras if x not in valid]
    if unknown:
        raise ValueError(
            f"Unknown camera(s): {unknown}. Valid cameras: {sorted(valid)}"
        )
    return cameras


def dataclass_field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


class VLMPretrainTrainer:
    def __init__(
        self,
        args: argparse.Namespace,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
    ):
        self.args = args
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.is_distributed = world_size > 1
        self.is_main = global_rank == 0
        self.device = torch.device("cuda", local_rank)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this training script.")

        self.use_bf16 = True
        self.cameras = parse_camera_list(args.cameras)

        self._build_model()
        self._build_data()
        self._build_optimizer()

        self.global_step = 0

        if self.is_main:
            os.makedirs(args.output_dir, exist_ok=True)
            self._save_config()

        if self.is_distributed:
            dist.barrier()

    @property
    def vlm_unwrapped(self) -> VLMBackbone:
        return self.vlm.module if self.is_distributed else self.vlm

    def _build_model(self):
        args = self.args

        args.vlm_model_path = resolve_pretrained_path(args.vlm_model_path)
        logger.info(
            "VLM weights: %s  (HF_HOME=%s)",
            args.vlm_model_path,
            os.environ.get("HF_HOME"),
        )

        config_kwargs = dict(
            model_name_or_path=args.vlm_model_path,
            freeze_vision_encoder=args.freeze_vision_encoder,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            current_resolution=(args.current_res_w, args.current_res_h),
            history_resolution=(args.history_res_w, args.history_res_h),
            reasoning_format=args.reasoning_format,
        )

        # Current patched VLMBackboneConfig supports load_in_4bit.
        # Keep compatibility with older public versions by checking the dataclass.
        vlm_config_fields = dataclass_field_names(VLMBackboneConfig)
        if "load_in_4bit" in vlm_config_fields:
            config_kwargs["load_in_4bit"] = args.load_in_4bit
        elif args.load_in_4bit:
            raise RuntimeError(
                "This VLMBackboneConfig does not expose load_in_4bit. "
                "Apply the existing 4-bit Q-LoRA VLM backbone patch first."
            )

        self.vlm = VLMBackbone(VLMBackboneConfig(**config_kwargs)).to(self.device)

        # ---------------------------------------------------------
        # Gradient checkpointing
        # 12GB GPU� Qwen3-VL Q-LoRA activation memory 
        # ---------------------------------------------------------
        base_model = self.vlm.model

        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

        logger.info(
            "Gradient checkpointing enabled; use_cache=False"
        )

        if self.is_main:
            total_params = sum(p.numel() for p in self.vlm.parameters())
            trainable_params = sum(
                p.numel() for p in self.vlm.parameters() if p.requires_grad
            )
            logger.info("VLM parameters: %s", f"{total_params:,}")
            logger.info(
                "VLM trainable parameters: %s (LoRA rank=%d)",
                f"{trainable_params:,}",
                args.lora_rank,
            )
            logger.info("Cameras: %s", self.cameras)
            logger.info(
                "Images/sample: %d cameras x %d timesteps = %d",
                len(self.cameras),
                args.num_history_steps + 1,
                len(self.cameras) * (args.num_history_steps + 1),
            )
            logger.info(
                "Resolution: current=%dx%d, history=%dx%d",
                args.current_res_w,
                args.current_res_h,
                args.history_res_w,
                args.history_res_h,
            )
            logger.info("4-bit Q-LoRA base: %s", args.load_in_4bit)

        if self.is_distributed:
            self.vlm = DDP(
                self.vlm,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

    def _data_config(
        self,
        data_root: str,
        vqa_root: str,
    ) -> VLADataConfig:
        args = self.args

        config_kwargs = dict(
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
            # VQA ONLY
            reasoning_mode=REASONING_MODE_VQA,
            vqa_root=vqa_root,
            qa_seed=args.qa_seed,
        )

        data_config_fields = dataclass_field_names(VLADataConfig)
        if "cameras" in data_config_fields:
            config_kwargs["cameras"] = list(self.cameras)
        else:
            raise RuntimeError(
                "VLADataConfig has no 'cameras' field. "
                "The current data_loader_vqa.py must support camera selection."
            )

        return VLADataConfig(**config_kwargs)

    def _build_data(self):
        args = self.args
        train_config = self._data_config(args.data_root, args.vqa_root)
        train_dataset = NuReasoningVLADataset(train_config, split="train")

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
            )
        else:
            self.train_sampler = None
            self.train_loader = build_dataloader(
                train_config,
                split="train",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )

        self.val_loader = None
        if args.test_data_root:
            val_config = self._data_config(args.test_data_root, args.test_vqa_root)
            val_dataset = NuReasoningVLADataset(val_config, split="test")

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
                )
            else:
                self.val_sampler = None
                self.val_loader = build_dataloader(
                    val_config,
                    split="test",
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                )
        else:
            self.val_sampler = None

        if self.is_main:
            logger.info("Training samples: %d", len(self.train_loader.dataset))
            if self.val_loader is not None:
                logger.info("Validation samples: %d", len(self.val_loader.dataset))

    def _build_optimizer(self):
        args = self.args

        trainable = [
            p for p in self.vlm_unwrapped.parameters()
            if p.requires_grad
        ]
        if not trainable:
            raise RuntimeError("VLM has no trainable parameters.")

        self.optimizer = AdamW(
            trainable,
            lr=args.vlm_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

        steps_per_epoch = math.ceil(
            len(self.train_loader) / args.gradient_accumulation_steps
        )
        total_steps = max(steps_per_epoch * args.epochs, 1)
        warmup_steps = int(total_steps * args.warmup_ratio)
        min_lr = 1e-6
        min_ratio = min(min_lr / max(args.vlm_lr, 1e-12), 1.0)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)

            if total_steps <= warmup_steps:
                return 1.0

            progress = (
                (current_step - warmup_steps)
                / float(max(total_steps - warmup_steps, 1))
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        if self.is_main:
            logger.info(
                "Scheduler: warmup+cosine, warmup=%d, total_steps=%d",
                warmup_steps,
                total_steps,
            )

    def _save_config(self):
        path = os.path.join(self.args.output_dir, "training_config.json")
        payload = vars(self.args).copy()
        payload["resolved_cameras"] = list(self.cameras)
        payload["images_per_sample"] = (
            len(self.cameras) * (self.args.num_history_steps + 1)
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def _prepare_single_vlm_input(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        vlm = self.vlm_unwrapped

        images = batch["images"][batch_idx]
        images = [img for img in images if img is not None]

        expected_images = len(self.cameras) * (
            self.args.num_history_steps + 1
        )
        if len(images) != expected_images:
            raise RuntimeError(
                f"Expected {expected_images} images/sample "
                f"({len(self.cameras)} cameras x "
                f"{self.args.num_history_steps + 1} timesteps), "
                f"but got {len(images)}."
            )

        image_contexts = []
        for t in range(-self.args.num_history_steps, 1):
            t_label = f"t={t}" if t < 0 else "t=0 (current)"
            for cam in self.cameras:
                image_contexts.append(
                    f"{t_label}, {cam} camera."
                )

        prompts = batch.get("user_prompts")
        prompt = None
        if prompts is not None and batch_idx < len(prompts):
            prompt = prompts[batch_idx]

        if not prompt:
            prompt = vlm.build_multiview_prompt(
                num_history_steps=self.args.num_history_steps,
                num_current_cameras=len(self.cameras),
                mission_command=batch["mission_commands"][batch_idx],
            )

        reasoning_text = batch["reasoning_texts"][batch_idx]

        inputs = vlm.prepare_inputs(
            images=images,
            text_prompt=prompt,
            image_contexts=image_contexts,
            assistant_response=reasoning_text,
        )

        result: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value

        if "prompt_length" in result:
            result["prompt_lengths"] = result.pop(
                "prompt_length"
            ).to(self.device)

        return result

    def _prepare_vlm_inputs(
        self,
        batch: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        return [
            self._prepare_single_vlm_input(batch, b)
            for b in range(len(batch["images"]))
        ]

    def _compute_reasoning_loss(
        self,
        batch: Dict[str, Any],
        *,
        train: bool,
    ) -> torch.Tensor:
        sample_inputs = self._prepare_vlm_inputs(batch)
        losses = []

        model = self.vlm if train else self.vlm_unwrapped

        for sample_input in sample_inputs:
            with torch.amp.autocast(
                "cuda",
                enabled=self.use_bf16,
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    input_ids=sample_input["input_ids"],
                    attention_mask=sample_input["attention_mask"],
                    pixel_values=sample_input.get("pixel_values"),
                    image_grid_thw=sample_input.get("image_grid_thw"),
                    mm_token_type_ids=sample_input.get(
                        "mm_token_type_ids"
                    ),
                    labels=sample_input.get("labels"),
                    prompt_lengths=sample_input.get(
                        "prompt_lengths"
                    ),
                    return_features=False,
                )

            print(
                "pixel_values",
                sample_input["pixel_values"].shape
                if sample_input.get("pixel_values") is not None
                else None,
                "input_ids",
                sample_input["input_ids"].shape,
            )

            loss = outputs.get("loss")

            if loss is None:
                raise RuntimeError(
                    "VLM forward returned no reasoning loss."
                )
            losses.append(loss)

        return torch.stack(losses).mean()

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.vlm.train()

        torch.cuda.reset_peak_memory_stats()

        alloc_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3

        logger.info(
            "[VRAM before forward] allocated=%.2f GiB reserved=%.2f GiB",
            alloc_before,
            reserved_before,
        )

        reasoning_loss = self._compute_reasoning_loss(
            batch,
            train=True,
        )

        alloc_forward = torch.cuda.memory_allocated() / 1024**3
        reserved_forward = torch.cuda.memory_reserved() / 1024**3

        logger.info(
            "[VRAM after forward] allocated=%.2f GiB reserved=%.2f GiB",
            alloc_forward,
            reserved_forward,
        )

        scaled_loss = (
            reasoning_loss
            / self.args.gradient_accumulation_steps
        )

        scaled_loss.backward()

        alloc_backward = torch.cuda.memory_allocated() / 1024**3
        reserved_backward = torch.cuda.memory_reserved() / 1024**3
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

        logger.info(
            "[VRAM after backward] allocated=%.2f GiB reserved=%.2f GiB "
            "peak_allocated=%.2f GiB peak_reserved=%.2f GiB",
            alloc_backward,
            reserved_backward,
            peak_alloc,
            peak_reserved,
        )

        return {
            "reasoning_loss": float(reasoning_loss.detach().item())
        }

    def optimizer_step(self):
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in self.vlm_unwrapped.parameters()
                    if p.requires_grad
                ],
                self.args.max_grad_norm,
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1

    @torch.no_grad()
    def evaluate_reasoning_loss(self) -> Optional[float]:
        if self.val_loader is None:
            return None

        self.vlm_unwrapped.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in self.val_loader:
            loss = self._compute_reasoning_loss(
                batch,
                train=False,
            )
            batch_size = len(batch["images"])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        if self.is_distributed:
            values = torch.tensor(
                [total_loss, float(total_samples)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            total_loss = float(values[0].item())
            total_samples = int(values[1].item())

        if total_samples == 0:
            return None

        return total_loss / total_samples

    def train(self):
        args = self.args

        if self.is_main:
            logger.info("=" * 72)
            logger.info("Starting VLM-only pretraining")
            logger.info("Base model: %s", args.vlm_model_path)
            logger.info("Cameras: %s", ",".join(self.cameras))
            logger.info(
                "Images/sample: %d",
                len(self.cameras) * (args.num_history_steps + 1),
            )
            logger.info(
                "Resolution: current=%dx%d, history=%dx%d",
                args.current_res_w,
                args.current_res_h,
                args.history_res_w,
                args.history_res_h,
            )
            logger.info("4-bit: %s", args.load_in_4bit)
            logger.info(
                "LoRA: r=%d alpha=%d dropout=%.3f",
                args.lora_rank,
                args.lora_alpha,
                args.lora_dropout,
            )
            logger.info("Reasoning mode: vqa (FIXED)")
            logger.info("VLM LR: %.3e", args.vlm_lr)
            logger.info(
                "Gradient accumulation: %d",
                args.gradient_accumulation_steps,
            )
            logger.info("=" * 72)

        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(args.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            running_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(self.train_loader):
                is_last_batch = (
                    step + 1 == len(self.train_loader)
                )
                is_sync_step = (
                    (step + 1)
                    % args.gradient_accumulation_steps
                    == 0
                ) or is_last_batch

                if self.is_distributed and not is_sync_step:
                    sync_ctx = self.vlm.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()

                with sync_ctx:
                    metrics = self.train_step(batch)

                running_loss += metrics["reasoning_loss"]
                num_batches += 1

                if is_sync_step:
                    self.optimizer_step()

                if (
                    self.is_main
                    and (step + 1) % args.log_interval == 0
                ):
                    avg_loss = running_loss / max(num_batches, 1)
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        "Epoch %d/%d | Step %d/%d | "
                        "Reasoning Loss %.4f | LR %.2e",
                        epoch + 1,
                        args.epochs,
                        step + 1,
                        len(self.train_loader),
                        avg_loss,
                        lr,
                    )

            epoch_time = time.time() - epoch_start
            avg_train_loss = running_loss / max(num_batches, 1)

            if self.is_main:
                logger.info(
                    "Epoch %d complete in %.1fs | "
                    "Train reasoning loss %.4f",
                    epoch + 1,
                    epoch_time,
                    avg_train_loss,
                )

            if (
                self.val_loader is not None
                and args.eval_interval > 0
                and (epoch + 1) % args.eval_interval == 0
            ):
                val_loss = self.evaluate_reasoning_loss()
                if self.is_main and val_loss is not None:
                    logger.info(
                        "[Validation] reasoning loss %.4f",
                        val_loss,
                    )

            if (
                args.save_interval > 0
                and (epoch + 1) % args.save_interval == 0
            ):
                self.save_checkpoint(epoch + 1)

        self.save_checkpoint(args.epochs, is_final=True)

        if self.is_main:
            logger.info("VLM-only pretraining complete.")

    def save_checkpoint(
        self,
        epoch: int,
        is_final: bool = False,
    ):
        if self.is_main:
            suffix = "final" if is_final else f"epoch_{epoch}"
            save_dir = os.path.join(
                self.args.output_dir,
                suffix,
            )
            os.makedirs(save_dir, exist_ok=True)

            adapter_dir = os.path.join(
                save_dir,
                "vlm_adapter",
            )
            self.vlm_unwrapped.save_adapter(adapter_dir)

            torch.save(
                {
                    "vlm_optimizer": self.optimizer.state_dict(),
                    "vlm_scheduler": self.scheduler.state_dict(),
                    "global_step": self.global_step,
                    "epoch": epoch,
                },
                os.path.join(
                    save_dir,
                    "training_state.pt",
                ),
            )

            logger.info(
                "Saved VLM adapter checkpoint: %s",
                save_dir,
            )

        if self.is_distributed:
            dist.barrier()

    def load_checkpoint(self, checkpoint_dir: str):
        adapter_dir = os.path.join(
            checkpoint_dir,
            "vlm_adapter",
        )
        if os.path.isdir(adapter_dir):
            self.vlm_unwrapped.load_adapter(
                adapter_dir,
                device=self.device,
            )
        else:
            raise FileNotFoundError(
                f"Missing VLM adapter directory: {adapter_dir}"
            )

        state_path = os.path.join(
            checkpoint_dir,
            "training_state.pt",
        )
        if os.path.isfile(state_path):
            state = torch.load(
                state_path,
                map_location=self.device,
            )
            self.optimizer.load_state_dict(
                state["vlm_optimizer"]
            )
            self.scheduler.load_state_dict(
                state["vlm_scheduler"]
            )
            self.global_step = int(
                state.get("global_step", 0)
            )
            logger.info(
                "Resumed from global step %d",
                self.global_step,
            )

        if self.is_distributed:
            dist.barrier()


def resolve_vqa_training_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate strict VQA-only training paths."""

    if not os.path.isdir(args.data_root):
        parser.error(
            f"--data_root is not a directory: {args.data_root}"
        )

    if not args.vqa_root:
        parser.error("--vqa_root is required.")
    if not os.path.isdir(args.vqa_root):
        parser.error(
            f"--vqa_root is not a directory: {args.vqa_root}"
        )

    if args.test_data_root:
        if not os.path.isdir(args.test_data_root):
            parser.error(
                f"--test_data_root is not a directory: "
                f"{args.test_data_root}"
            )

        if not args.test_vqa_root:
            parser.error(
                "--test_data_root is set, therefore "
                "--test_vqa_root is also required."
            )
        if not os.path.isdir(args.test_vqa_root):
            parser.error(
                f"--test_vqa_root is not a directory: "
                f"{args.test_vqa_root}"
            )
    else:
        args.test_vqa_root = None

    # Saved config/log compatibility.
    args.reasoning_mode = REASONING_MODE_VQA


def parse_args():
    parser = argparse.ArgumentParser(
        description="nuReasoning VLM-only Q-LoRA pretraining"
    )

    # =========================================================================
    # ★★★ 현재 서버 기준 데이터 경로 ★★★
    #
    # 원본 train:
    #   /media/HDD/nuR_ds/data/train
    #
    # 생성된 train VQA:
    #   /home/lhh/lab/dataset/nuReasoning/vqa_filtered
    #
    # generate_vqa.py를 --history-frames 1로 생성했으므로
    # 아래 --num_history_steps 역시 반드시 1이어야 합니다.
    #
    # 입력 이미지:
    #   t=-1s : front / front_left / front_right
    #   t= 0s : front / front_left / front_right
    #   => 총 6장/sample
    # =========================================================================
    # [중요]
    # train_filtered 경로는 이 학습 스크립트에 직접 넣지 않습니다.
    # 학습 입력의 센서/이미지는 원본 data_root에서 읽고,
    # 필터링된 reasoning을 바탕으로 이미 생성된 VQA는 vqa_root에서 읽습니다.
    #
    # ORIGINAL dataset root:
    # metadata / cameras / ego_state are loaded from here.
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/HDD/nuR_ds/data/train",
    )
    # Validation ORIGINAL dataset root.
    parser.add_argument(
        "--test_data_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/vlm_pretrain",
    )

    parser.add_argument(
        "--cameras",
        type=str,
        default="front,front_left,front_right",
        help=(
            "Comma-separated cameras. Default: "
            "front,front_left,front_right"
        ),
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--freeze_vision_encoder",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--vlm_lr",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--reasoning_format",
        type=str,
        default="spatial_driving_counterfactual",
        choices=[
            "driving",
            "spatial_driving",
            "spatial_driving_counterfactual",
            "spatial",
            "driving_counterfactual",
        ],
    )
    parser.add_argument(
        "--reasoning_max_items_per_list",
        type=int,
        default=10,
    )
    # GENERATED VQA roots.
    # These VQA files must be generated from *_filtered reasoning.
    parser.add_argument(
        "--vqa_root",
        type=str,
        default="/home/lhh/lab/dataset/nuReasoning/vqa_filtered",
    )
    # NOTE:
    # Validation을 사용할 경우 validation_filtered로 VQA를 별도로 생성한 뒤
    # 실제 저장 위치를 --test_vqa_root로 지정해야 합니다.
    # 현재 값은 기존 코드의 validation VQA 기본 경로를 유지합니다.
    parser.add_argument(
        "--test_vqa_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--qa_seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_history_steps",
        type=int,
        default=1,
        help=(
            "1 means previous timestep + current timestep. "
            "With 3 cameras this is 6 images/sample."
        ),
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

    # These remain because the shared VLADataConfig / Dataset currently
    # constructs trajectory metadata even though this trainer does not use it.
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
        "--log_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    resolve_vqa_training_args(args, parser)
    return args


def main():
    local_rank, global_rank, world_size = setup_distributed()
    args = parse_args()

    if global_rank == 0:
        os.makedirs(
            os.path.expanduser(args.output_dir),
            exist_ok=True,
        )

    args.output_dir = os.path.expanduser(args.output_dir)

    logging.basicConfig(
        level=(
            logging.INFO
            if global_rank == 0
            else logging.WARNING
        ),
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s: %(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
            *(
                [
                    logging.FileHandler(
                        os.path.join(
                            args.output_dir,
                            "train_vlm.log",
                        ),
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
            json.dumps(
                vars(args),
                indent=2,
                default=str,
            ),
        )

    trainer = VLMPretrainTrainer(
        args,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
    )

    if args.resume_from:
        trainer.load_checkpoint(
            os.path.expanduser(args.resume_from)
        )

    trainer.train()
    cleanup_distributed()


if __name__ == "__main__":
    main()