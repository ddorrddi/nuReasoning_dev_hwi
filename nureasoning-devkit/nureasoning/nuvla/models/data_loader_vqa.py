#!/usr/bin/env python3
"""
Strict VQA-only data loader for nuReasoning / nuVLA.

This module reuses the existing data_loader.py implementation and adds
VQA-only filtering/validation.

Important behavior
------------------
1. reasoning_mode must be "vqa".
2. vqa_root must exist.
3. Only raw frames with a matching *_vqa.json are candidates.
4. BEFORE training starts, every matched VQA file is checked.
5. Frames whose VQA file is unreadable, empty, or has no formatter-valid QA
   are removed from the dataset.
6. During training, a VQA target must always exist. No silent fallback to
   structured reasoning is allowed.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from nureasoning.nuvla.models.data_loader import (
    VLADataConfig,
    NuReasoningVLADataset as _BaseNuReasoningVLADataset,
    vla_collate_fn,
)

logger = logging.getLogger(__name__)


class NuReasoningVLADataset(_BaseNuReasoningVLADataset):
    """Strict VQA-only dataset built on top of the existing raw-data loader."""

    def __init__(self, config: VLADataConfig, split: str = "train"):
        mode = str(config.reasoning_mode).lower().strip()

        if mode != "vqa":
            raise ValueError(
                "data_loader_vqa.py is VQA-only. "
                f"Expected reasoning_mode='vqa', got {config.reasoning_mode!r}."
            )

        if not config.vqa_root:
            raise ValueError("VQA mode requires config.vqa_root.")

        if not os.path.isdir(config.vqa_root):
            raise FileNotFoundError(
                f"VQA root does not exist: {config.vqa_root}"
            )

        # Base loader discovers raw samples and indexes *_vqa.json files.
        super().__init__(config=config, split=split)

        if not self._vqa_index:
            raise RuntimeError(
                f"No *_vqa.json files were indexed under {config.vqa_root}"
            )

        raw_sample_count = len(self.samples)

        # Import exactly the same formatter used by the base loader.
        from nureasoning.reasoning.modules.prompt_format import (
            format_assistant_answer,
        )

        usable_samples: List[Dict[str, Any]] = []

        no_vqa_file = 0
        unreadable_vqa = 0
        empty_questions = 0
        no_valid_answers = 0

        for sample in self.samples:
            key = (
                sample["clip_name"],
                str(sample["timestamp_us"]),
            )
            path = self._vqa_index.get(key)

            if not path:
                no_vqa_file += 1
                continue

            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                unreadable_vqa += 1
                continue

            questions = payload.get("questions") or []
            if not isinstance(questions, list) or len(questions) == 0:
                empty_questions += 1
                continue

            has_valid_answer = False
            for question in questions:
                if not isinstance(question, dict):
                    continue
                try:
                    answer = format_assistant_answer(question)
                except Exception:
                    answer = None
                if answer:
                    has_valid_answer = True
                    break

            if not has_valid_answer:
                no_valid_answers += 1
                continue

            usable_samples.append(sample)

        self.samples = usable_samples

        if not self.samples:
            raise RuntimeError(
                "No usable VQA training samples remain after validation.\n"
                f"  data_root = {config.data_root}\n"
                f"  vqa_root  = {config.vqa_root}"
            )

        logger.info("=" * 72)
        logger.info("[%s] STRICT VQA PRECHECK", split)
        logger.info("Raw candidate samples        : %d", raw_sample_count)
        logger.info("Usable VQA samples           : %d", len(self.samples))
        logger.info("Excluded - no VQA file       : %d", no_vqa_file)
        logger.info("Excluded - unreadable JSON   : %d", unreadable_vqa)
        logger.info("Excluded - zero questions    : %d", empty_questions)
        logger.info("Excluded - no valid answers  : %d", no_valid_answers)
        logger.info("=" * 72)

    def _select_qa_target(
        self,
        clip_name: str,
        timestamp_us: int,
    ) -> Optional[Tuple[str, str]]:
        """
        Select one deterministic QA pair.

        The initialization precheck guarantees that every remaining sample has
        at least one formatter-valid QA. If this still fails, the file changed
        after dataset construction or there is an unexpected runtime issue.
        """
        qa_pair = super()._select_qa_target(
            clip_name=clip_name,
            timestamp_us=timestamp_us,
        )

        if qa_pair is None:
            vqa_path = self._vqa_index.get(
                (clip_name, str(timestamp_us)),
                "<not indexed>",
            )
            raise RuntimeError(
                "A prevalidated VQA sample became invalid during training.\n"
                f"  clip      = {clip_name}\n"
                f"  timestamp = {timestamp_us}\n"
                f"  vqa_file  = {vqa_path}"
            )

        return qa_pair


def build_dataloader(
    config: VLADataConfig,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = NuReasoningVLADataset(config, split=split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=vla_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


__all__ = [
    "VLADataConfig",
    "NuReasoningVLADataset",
    "build_dataloader",
    "vla_collate_fn",
]
