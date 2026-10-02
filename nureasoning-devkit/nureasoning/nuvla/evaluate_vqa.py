#!/usr/bin/env python3
"""
VQA-only evaluation for the nuReasoning / nuVLA Qwen3-VL LoRA model.

This evaluator is intentionally matched to train_vlm_vqa.py:
  - VLM only (no Action Expert / no planning benchmark)
  - data_loader_vqa.py
  - strict VQA mode
  - same camera selection / history setting / image resolution
  - same user_prompts -> reasoning_texts supervision
  - loads <checkpoint_dir>/vlm_adapter

Evaluation modes
----------------
1) loss
   Teacher-forced reasoning/VQA loss over the validation/test VQA dataset.

2) generate
   Autoregressive generation using the VQA question as the user prompt.
   Saves GT/prediction pairs and reports Exact Match, token F1, ROUGE-L.

3) both
   Runs loss evaluation first, then generation evaluation.

Typical usage
-------------
python -m nureasoning.nuvla.evaluate_vqa \
    --checkpoint_dir ./outputs/vlm_pretrain/final \
    --data_root /media/HDD/nuR_ds/data/validation \
    --vqa_root /home/lhh/lab/dataset/nuReasoning/vqa_validation_filtered \
    --mode both \
    --num_generation_samples 500 \
    --load_in_4bit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import fields
from typing import Any, Dict, List, Optional

import torch

from nureasoning.common.pretrained import resolve_pretrained_path
from nureasoning.nuvla.models.vlm_backbone import VLMBackbone, VLMBackboneConfig
from nureasoning.nuvla.models.data_loader_vqa import (
    VLADataConfig,
    NuReasoningVLADataset,
    build_dataloader,
)

logger = logging.getLogger(__name__)

REASONING_MODE_VQA = "vqa"


def dataclass_field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


def parse_camera_list(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple)):
        cameras = [str(x).strip() for x in raw if str(x).strip()]
    else:
        cameras = [x.strip() for x in str(raw).split(",") if x.strip()]

    if not cameras:
        raise ValueError("Camera list is empty.")
    return cameras


def load_training_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Find the training_config.json saved by train_vlm_vqa.py.

    train_vlm_vqa.py stores:
        <output_dir>/training_config.json
        <output_dir>/final/vlm_adapter/...

    Therefore when checkpoint_dir=<output_dir>/final, the config normally lives
    one directory above checkpoint_dir.
    """
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    candidates = [
        os.path.join(checkpoint_dir, "training_config.json"),
        os.path.join(os.path.dirname(checkpoint_dir), "training_config.json"),
        os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), "training_config.json"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            logger.info("Loaded training config: %s", path)
            return cfg

    raise FileNotFoundError(
        "Could not find training_config.json. Checked:\n  - "
        + "\n  - ".join(candidates)
    )


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction).split()
    ref = normalize_text(reference).split()

    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    # O(len(pred) * len(ref)) LCS; VQA answers are normally short enough.
    prev = [0] * (len(ref) + 1)
    for p in pred:
        cur = [0]
        for j, r in enumerate(ref, start=1):
            if p == r:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur

    lcs = prev[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def format_eta(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class VQAEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this evaluation script.")

        self.device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(self.device)

        self.train_cfg = load_training_config(args.checkpoint_dir)
        self.cameras = self._resolve_cameras()

        self._build_model()
        self._load_checkpoint()
        self._build_data()

    def _resolve_cameras(self) -> List[str]:
        if self.args.cameras:
            cameras = parse_camera_list(self.args.cameras)
        elif self.train_cfg.get("resolved_cameras"):
            cameras = parse_camera_list(self.train_cfg["resolved_cameras"])
        else:
            cameras = parse_camera_list(
                self.train_cfg.get("cameras", "front,front_left,front_right")
            )

        logger.info("Evaluation cameras: %s", ",".join(cameras))
        return cameras

    def _cfg(self, key: str, default: Any = None) -> Any:
        value = self.train_cfg.get(key, default)
        return default if value is None else value

    def _build_model(self) -> None:
        model_path = self.args.vlm_model_path or self._cfg(
            "vlm_model_path", "Qwen/Qwen3-VL-2B-Instruct"
        )
        model_path = resolve_pretrained_path(model_path)

        load_in_4bit = (
            self.args.load_in_4bit
            if self.args.load_in_4bit is not None
            else bool(self._cfg("load_in_4bit", False))
        )

        config_kwargs = dict(
            model_name_or_path=model_path,
            freeze_vision_encoder=bool(self._cfg("freeze_vision_encoder", False)),
            lora_rank=int(self._cfg("lora_rank", 16)),
            lora_alpha=int(self._cfg("lora_alpha", 32)),
            lora_dropout=float(self._cfg("lora_dropout", 0.05)),
            current_resolution=(
                int(self._cfg("current_res_w", 448)),
                int(self._cfg("current_res_h", 448)),
            ),
            history_resolution=(
                int(self._cfg("history_res_w", 448)),
                int(self._cfg("history_res_h", 448)),
            ),
            reasoning_format=str(
                self._cfg("reasoning_format", "spatial_driving_counterfactual")
            ),
        )

        vlm_fields = dataclass_field_names(VLMBackboneConfig)
        if "load_in_4bit" in vlm_fields:
            config_kwargs["load_in_4bit"] = load_in_4bit
        elif load_in_4bit:
            raise RuntimeError(
                "VLMBackboneConfig does not expose load_in_4bit, but this checkpoint "
                "was requested in 4-bit mode. Apply the same VLM backbone patch used "
                "for training."
            )

        logger.info("Building VLM from %s", model_path)
        logger.info("4-bit base: %s", load_in_4bit)

        self.vlm = VLMBackbone(VLMBackboneConfig(**config_kwargs)).to(self.device)
        self.vlm.eval()

        # Evaluation/generation does not need gradient checkpointing.
        base_model = self.vlm.model
        if hasattr(base_model, "gradient_checkpointing_disable"):
            try:
                base_model.gradient_checkpointing_disable()
            except Exception:
                pass
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = True

    def _load_checkpoint(self) -> None:
        adapter_dir = os.path.join(
            os.path.abspath(os.path.expanduser(self.args.checkpoint_dir)),
            "vlm_adapter",
        )
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Missing VLM adapter directory: {adapter_dir}")

        self.vlm.load_adapter(adapter_dir, device=self.device)
        self.vlm.eval()
        logger.info("Loaded VLM adapter: %s", adapter_dir)

        self._verify_lora_load(adapter_dir)

    def _verify_lora_load(self, adapter_dir: str) -> None:
        """Audit that saved LoRA tensors are actually present in the live model."""
        sft_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.isfile(sft_path):
            logger.warning("LoRA audit skipped: %s not found", sft_path)
            return

        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.warning("LoRA audit skipped: safetensors is not installed")
            return

        saved = load_file(sft_path)
        live = {
            n: p for n, p in self.vlm.model.named_parameters()
            if "lora_" in n
        }

        def norm_key(key: str) -> str:
            k = key.replace(".default.weight", ".weight")
            if k.startswith("base_model.model.model."):
                k = "base_model.model." + k[len("base_model.model.model."):]
            return k

        live_norm = {norm_key(n): p for n, p in live.items()}
        matches = 0
        total = 0
        not_found = 0
        mismatches = 0

        for key, file_tensor in saved.items():
            total += 1
            live_param = live_norm.get(norm_key(key))
            if live_param is None:
                not_found += 1
                continue

            live_tensor = live_param.detach().to("cpu", dtype=file_tensor.dtype)
            if live_tensor.shape == file_tensor.shape and torch.allclose(
                live_tensor, file_tensor, atol=1e-6, rtol=1e-5
            ):
                matches += 1
            else:
                mismatches += 1

        logger.info(
            "LoRA load audit: %d/%d match (mismatch=%d, not_found=%d)",
            matches, total, mismatches, not_found,
        )

        if total > 0 and matches < int(0.95 * total):
            raise RuntimeError(
                f"LoRA adapter audit failed: only {matches}/{total} tensors match. "
                f"mismatch={mismatches}, not_found={not_found}"
            )

    def _build_data(self) -> None:
        data_root = os.path.abspath(os.path.expanduser(self.args.data_root))
        vqa_root = os.path.abspath(os.path.expanduser(self.args.vqa_root))

        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"data_root is not a directory: {data_root}")
        if not os.path.isdir(vqa_root):
            raise FileNotFoundError(f"vqa_root is not a directory: {vqa_root}")

        config_kwargs = dict(
            data_root=data_root,
            num_history_steps=int(self._cfg("num_history_steps", 1)),
            history_stride=int(self._cfg("history_stride", 10)),
            current_resolution=(
                int(self._cfg("current_res_w", 448)),
                int(self._cfg("current_res_h", 448)),
            ),
            history_resolution=(
                int(self._cfg("history_res_w", 448)),
                int(self._cfg("history_res_h", 448)),
            ),
            trajectory_future_seconds=float(
                self._cfg("trajectory_future_seconds", 5.0)
            ),
            frame_rate_hz=float(self._cfg("frame_rate_hz", 10.0)),
            num_waypoints=int(self._cfg("num_trajectory_points", 10)),
            max_history_traj_points=int(
                self._cfg("max_history_traj_points", 6)
            ),
            reasoning_format=str(
                self._cfg("reasoning_format", "spatial_driving_counterfactual")
            ),
            reasoning_max_items_per_list=int(
                self._cfg("reasoning_max_items_per_list", 10)
            ),
            reasoning_mode=REASONING_MODE_VQA,
            vqa_root=vqa_root,
            qa_seed=int(self.args.qa_seed),
        )

        data_fields = dataclass_field_names(VLADataConfig)
        if "cameras" not in data_fields:
            raise RuntimeError(
                "VLADataConfig has no 'cameras' field. Use the same patched "
                "data_loader_vqa.py that was used for train_vlm_vqa.py."
            )
        config_kwargs["cameras"] = list(self.cameras)

        data_config = VLADataConfig(**config_kwargs)

        # Keep split='test' exactly consistent with train_vlm_vqa.py validation.
        dataset = NuReasoningVLADataset(data_config, split="test")
        self.loader = build_dataloader(
            data_config,
            split="test",
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

        logger.info("Evaluation samples: %d", len(dataset))
        logger.info(
            "Images/sample: %d cameras x %d timesteps = %d",
            len(self.cameras),
            int(self._cfg("num_history_steps", 1)) + 1,
            len(self.cameras) * (int(self._cfg("num_history_steps", 1)) + 1),
        )
        logger.info(
            "Resolution: current=%sx%s history=%sx%s",
            self._cfg("current_res_w", 448),
            self._cfg("current_res_h", 448),
            self._cfg("history_res_w", 448),
            self._cfg("history_res_h", 448),
        )

    def _get_prompt(self, batch: Dict[str, Any], batch_idx: int) -> str:
        prompts = batch.get("user_prompts")
        prompt = None
        if prompts is not None and batch_idx < len(prompts):
            prompt = prompts[batch_idx]

        if prompt:
            return prompt

        return self.vlm.build_multiview_prompt(
            num_history_steps=int(self._cfg("num_history_steps", 1)),
            num_current_cameras=len(self.cameras),
            mission_command=batch["mission_commands"][batch_idx],
        )

    def _prepare_single_input(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        *,
        include_answer: bool,
    ) -> Dict[str, Any]:
        images = batch["images"][batch_idx]
        images = [img for img in images if img is not None]

        num_history_steps = int(self._cfg("num_history_steps", 1))
        expected_images = len(self.cameras) * (num_history_steps + 1)
        if len(images) != expected_images:
            raise RuntimeError(
                f"Expected {expected_images} images/sample "
                f"({len(self.cameras)} cameras x {num_history_steps + 1} timesteps), "
                f"but got {len(images)}."
            )

        image_contexts: List[str] = []
        for t in range(-num_history_steps, 1):
            t_label = f"t={t}" if t < 0 else "t=0 (current)"
            for cam in self.cameras:
                image_contexts.append(f"{t_label}, {cam} camera.")

        prompt = self._get_prompt(batch, batch_idx)
        assistant_response = (
            batch["reasoning_texts"][batch_idx] if include_answer else None
        )

        inputs = self.vlm.prepare_inputs(
            images=images,
            text_prompt=prompt,
            image_contexts=image_contexts,
            assistant_response=assistant_response,
        )

        result: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value

        if "prompt_length" in result:
            result["prompt_lengths"] = result.pop("prompt_length").to(self.device)

        result["raw_prompt"] = prompt
        return result

    @torch.no_grad()
    def _loss_for_batch(self, batch: Dict[str, Any]) -> tuple[float, int]:
        losses: List[torch.Tensor] = []

        for b in range(len(batch["images"])):
            inputs = self._prepare_single_input(batch, b, include_answer=True)

            with torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                outputs = self.vlm(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    mm_token_type_ids=inputs.get("mm_token_type_ids"),
                    labels=inputs.get("labels"),
                    prompt_lengths=inputs.get("prompt_lengths"),
                    return_features=False,
                )

            loss = outputs.get("loss")
            if loss is None:
                raise RuntimeError("VLM forward returned no reasoning loss.")
            losses.append(loss.float())

        if not losses:
            return 0.0, 0

        mean_loss = torch.stack(losses).mean().item()
        return float(mean_loss), len(losses)

    @torch.no_grad()
    def evaluate_loss(self) -> Dict[str, Any]:
        self.vlm.eval()
        logger.info("=" * 72)
        logger.info("VQA teacher-forced loss evaluation")
        logger.info("=" * 72)

        total_loss = 0.0
        total_samples = 0
        start = time.time()
        max_samples = self.args.max_loss_samples

        for step, batch in enumerate(self.loader, start=1):
            if max_samples > 0 and total_samples >= max_samples:
                break

            loss, n = self._loss_for_batch(batch)
            if max_samples > 0 and total_samples + n > max_samples:
                n = max_samples - total_samples
                if n <= 0:
                    break

            total_loss += loss * n
            total_samples += n

            if step % self.args.log_interval == 0:
                elapsed = time.time() - start
                avg = total_loss / max(total_samples, 1)

                # Estimate ETA from sample progress when full dataset is used.
                if max_samples > 0:
                    target = min(max_samples, len(self.loader.dataset))
                else:
                    target = len(self.loader.dataset)
                rate = total_samples / max(elapsed, 1e-6)
                eta = (target - total_samples) / max(rate, 1e-6)

                logger.info(
                    "[LOSS] %d/%d samples | loss=%.4f | %.2f samples/s | ETA %s",
                    total_samples,
                    target,
                    avg,
                    rate,
                    format_eta(eta),
                )

        elapsed = time.time() - start
        avg_loss = total_loss / max(total_samples, 1)
        perplexity = float(torch.exp(torch.tensor(min(avg_loss, 20.0))).item())

        result = {
            "num_samples": total_samples,
            "reasoning_loss": avg_loss,
            "perplexity": perplexity,
            "total_time_s": elapsed,
        }

        logger.info("=" * 72)
        logger.info("Loss evaluation complete")
        logger.info("Samples:    %d", total_samples)
        logger.info("Loss:       %.6f", avg_loss)
        logger.info("Perplexity: %.4f", perplexity)
        logger.info("Time:       %.1fs", elapsed)
        logger.info("=" * 72)

        path = os.path.join(self.args.output_dir, "vqa_loss_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Saved loss results: %s", path)

        return result

    @torch.no_grad()
    def generate_one(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        inputs = self._prepare_single_input(batch, batch_idx, include_answer=False)

        generate_kwargs: Dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
        }

        if self.args.temperature > 0:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p

        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in inputs:
                generate_kwargs[key] = inputs[key]

        output_ids = self.vlm.model.generate(**generate_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, prompt_len:]
        prediction = self.vlm.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        gt = str(batch["reasoning_texts"][batch_idx]).strip()
        prompt = str(inputs["raw_prompt"])

        return {
            "prompt": prompt,
            "ground_truth": gt,
            "prediction": prediction,
            "exact_match": float(normalize_text(prediction) == normalize_text(gt)),
            "token_f1": token_f1_score(prediction, gt),
            "rouge_l_f1": rouge_l_f1(prediction, gt),
        }

    @torch.no_grad()
    def evaluate_generation(self) -> Dict[str, Any]:
        self.vlm.eval()
        logger.info("=" * 72)
        logger.info("VQA autoregressive generation evaluation")
        logger.info("=" * 72)

        requested = self.args.num_generation_samples
        target = len(self.loader.dataset) if requested <= 0 else min(
            requested, len(self.loader.dataset)
        )

        records: List[Dict[str, Any]] = []
        start = time.time()

        for batch in self.loader:
            for b in range(len(batch["images"])):
                if len(records) >= target:
                    break

                t0 = time.time()
                rec = self.generate_one(batch, b)
                rec["generation_time_s"] = time.time() - t0

                # Add metadata when available without depending on it.
                if "clip_names" in batch and b < len(batch["clip_names"]):
                    rec["clip_name"] = batch["clip_names"][b]
                if "frame_indices" in batch and b < len(batch["frame_indices"]):
                    value = batch["frame_indices"][b]
                    rec["frame_index"] = int(value.item()) if torch.is_tensor(value) else int(value)

                records.append(rec)
                idx = len(records)

                if idx <= self.args.print_first_n:
                    logger.info("\n--- VQA sample %d/%d ---", idx, target)
                    logger.info("Q: %s", rec["prompt"])
                    logger.info("GT: %s", rec["ground_truth"])
                    logger.info("PR: %s", rec["prediction"])
                    logger.info(
                        "EM=%.3f | F1=%.3f | ROUGE-L=%.3f | %.2fs",
                        rec["exact_match"],
                        rec["token_f1"],
                        rec["rouge_l_f1"],
                        rec["generation_time_s"],
                    )

                if idx % self.args.log_interval == 0:
                    elapsed = time.time() - start
                    rate = idx / max(elapsed, 1e-6)
                    eta = (target - idx) / max(rate, 1e-6)
                    mean_em = sum(x["exact_match"] for x in records) / idx
                    mean_f1 = sum(x["token_f1"] for x in records) / idx
                    mean_rouge = sum(x["rouge_l_f1"] for x in records) / idx
                    logger.info(
                        "[GEN] %d/%d | EM=%.4f | F1=%.4f | ROUGE-L=%.4f | ETA %s",
                        idx, target, mean_em, mean_f1, mean_rouge, format_eta(eta)
                    )

            if len(records) >= target:
                break

        n = len(records)
        elapsed = time.time() - start
        summary = {
            "num_samples": n,
            "exact_match": (
                sum(x["exact_match"] for x in records) / n if n else 0.0
            ),
            "token_f1": (
                sum(x["token_f1"] for x in records) / n if n else 0.0
            ),
            "rouge_l_f1": (
                sum(x["rouge_l_f1"] for x in records) / n if n else 0.0
            ),
            "avg_generation_time_s": (
                sum(x["generation_time_s"] for x in records) / n if n else 0.0
            ),
            "total_time_s": elapsed,
        }

        logger.info("=" * 72)
        logger.info("Generation evaluation complete")
        logger.info("Samples:       %d", n)
        logger.info("Exact Match:   %.4f", summary["exact_match"])
        logger.info("Token F1:      %.4f", summary["token_f1"])
        logger.info("ROUGE-L F1:    %.4f", summary["rouge_l_f1"])
        logger.info("Avg gen time:  %.3fs/sample", summary["avg_generation_time_s"])
        logger.info("=" * 72)

        result_path = os.path.join(self.args.output_dir, "vqa_generation_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "samples": records},
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Saved generation results: %s", result_path)

        return summary

    def run(self) -> None:
        logger.info("Checkpoint: %s", self.args.checkpoint_dir)
        logger.info("Data root:  %s", self.args.data_root)
        logger.info("VQA root:   %s", self.args.vqa_root)
        logger.info("Mode:       %s", self.args.mode)

        all_results: Dict[str, Any] = {}

        if self.args.mode in ("loss", "both"):
            all_results["loss"] = self.evaluate_loss()

        if self.args.mode in ("generate", "both"):
            all_results["generation"] = self.evaluate_generation()

        summary_path = os.path.join(self.args.output_dir, "vqa_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved evaluation summary: %s", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQA-only evaluation for train_vlm_vqa.py checkpoints"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory containing vlm_adapter, e.g. ./outputs/vlm_pretrain/final",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Original validation/test nuReasoning dataset root (images/metadata).",
    )
    parser.add_argument(
        "--vqa_root",
        type=str,
        required=True,
        help="Generated validation/test VQA root matching data_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Default: <checkpoint workspace>/eval_vqa",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["loss", "generate", "both"],
    )

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument(
        "--max_loss_samples",
        type=int,
        default=0,
        help="0 = evaluate loss on the full VQA dataset.",
    )
    parser.add_argument(
        "--num_generation_samples",
        type=int,
        default=100,
        help="0 = generate on the full VQA dataset.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy decoding (recommended for reproducible evaluation).",
    )
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--print_first_n", type=int, default=10)
    parser.add_argument("--qa_seed", type=int, default=42)

    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Override training cameras. Normally leave unset.",
    )
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default=None,
        help="Override base model path. Normally leave unset.",
    )

    # Tri-state: if omitted, use the value saved in training_config.json.
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true")
    quant.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=None)

    args = parser.parse_args()
    args.checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    args.data_root = os.path.abspath(os.path.expanduser(args.data_root))
    args.vqa_root = os.path.abspath(os.path.expanduser(args.vqa_root))

    if args.output_dir is None:
        workspace = os.path.dirname(args.checkpoint_dir)
        args.output_dir = os.path.join(workspace, "eval_vqa")
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "evaluate_vqa.log"),
                mode="a",
                encoding="utf-8",
            ),
        ],
    )

    logger.info("Evaluation args:\n%s", json.dumps(vars(args), indent=2, ensure_ascii=False))

    evaluator = VQAEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
