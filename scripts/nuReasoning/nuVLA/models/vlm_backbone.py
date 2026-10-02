"""
VLM Backbone: Qwen3-VL / Qwen3.5 wrapper for nuReasoning VLA training.

Modified from the public nuReasoning implementation.
Main change:
  - instead of returning only the final hidden state for the planner,
    return selected layer-wise *actual attention KV cache* tensors.
  - only the supervised assistant/reasoning token span is kept for planner conditioning.

Everything else (prompt construction, LoRA, processor, text reasoning loss) is kept.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from nureasoning.common.pretrained import from_pretrained
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)

CAMERA_NAMES = [
    "front", "front_left", "front_right",
    "left", "right",
    "back", "back_left", "back_right",
]

CAMERA_TOKENS = {cam: f"<camera_{cam}>" for cam in CAMERA_NAMES}
TIMESTEP_TOKEN_FN = lambda t: f"<t={t}>"


@dataclass
class VLMBackboneConfig:
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    freeze_vision_encoder: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_dropout: float = 0.05
    current_resolution: Tuple[int, int] = (448, 448)
    history_resolution: Tuple[int, int] = (448, 448)
    reasoning_format: str = "spatial_driving_counterfactual"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    torch_dtype: torch.dtype = torch.bfloat16


class VLMBackbone(nn.Module):
    """Qwen VLM wrapper with layer-wise reasoning KV extraction."""

    def __init__(self, config: VLMBackboneConfig):
        super().__init__()
        self.config = config
        self.model_is_vl = self._is_vl_backbone(config.model_name_or_path)

        self.model = self._load_backbone(config)
        self.feature_dim = self._infer_feature_dim()
        self.num_hidden_layers = self._infer_num_hidden_layers()
        self.kv_num_heads, self.kv_head_dim = self._infer_kv_spec()

        processor_kwargs: Dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
        }
        if self.model_is_vl:
            processor_kwargs.update({
                "min_pixels": 64 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            })
        self.processor = from_pretrained(
            AutoProcessor.from_pretrained,
            config.model_name_or_path,
            **processor_kwargs,
        )

        if config.freeze_vision_encoder:
            self._freeze_vision_encoder()

        if config.lora_rank > 0:
            self._apply_lora(config)

    # ------------------------------------------------------------------
    # Config / shape inference
    # ------------------------------------------------------------------

    def _text_config(self):
        cfg = getattr(self.model, "config", None)
        return getattr(cfg, "text_config", None) or cfg

    def _infer_feature_dim(self) -> int:
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        for source in (text_cfg, cfg):
            hidden_size = getattr(source, "hidden_size", None) if source is not None else None
            if isinstance(hidden_size, int) and hidden_size > 0:
                return hidden_size
        raise RuntimeError("Could not infer VLM hidden size from model config.")

    def _infer_num_hidden_layers(self) -> int:
        text_cfg = self._text_config()
        value = getattr(text_cfg, "num_hidden_layers", None)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError("Could not infer num_hidden_layers from VLM config.")
        return value

    def _infer_kv_spec(self) -> Tuple[int, int]:
        text_cfg = self._text_config()
        num_attention_heads = getattr(text_cfg, "num_attention_heads", None)
        num_key_value_heads = getattr(text_cfg, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(text_cfg, "head_dim", None)
        hidden_size = getattr(text_cfg, "hidden_size", None)

        if not isinstance(num_attention_heads, int) or num_attention_heads <= 0:
            raise RuntimeError("Could not infer num_attention_heads from VLM config.")
        if not isinstance(num_key_value_heads, int) or num_key_value_heads <= 0:
            raise RuntimeError("Could not infer num_key_value_heads from VLM config.")
        if not isinstance(head_dim, int) or head_dim <= 0:
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise RuntimeError("Could not infer VLM attention head_dim.")
            head_dim = hidden_size // num_attention_heads

        logger.info(
            "VLM KV spec: layers=%d, kv_heads=%d, head_dim=%d",
            self.num_hidden_layers,
            num_key_value_heads,
            head_dim,
        )
        return num_key_value_heads, head_dim

    def get_kv_spec(self) -> Tuple[int, int, int]:
        """Return (num_hidden_layers, num_key_value_heads, head_dim)."""
        return self.num_hidden_layers, self.kv_num_heads, self.kv_head_dim

    # ------------------------------------------------------------------
    # Backbone loading helpers
    # ------------------------------------------------------------------

    def _from_pretrained_with_fallback(
        self,
        model_cls: Any,
        model_name_or_path: str,
        kwargs: Dict[str, Any],
    ) -> nn.Module:
        try:
            return from_pretrained(model_cls.from_pretrained, model_name_or_path, **kwargs)
        except Exception:
            reduced_kwargs = dict(kwargs)
            for key in ("attn_implementation", "dtype", "device_map"):
                reduced_kwargs.pop(key, None)
            return from_pretrained(model_cls.from_pretrained, model_name_or_path, **reduced_kwargs)

    def _is_vl_backbone(self, model_name_or_path: str) -> bool:
        try:
            cfg = from_pretrained(
                AutoConfig.from_pretrained,
                model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            model_type = str(getattr(cfg, "model_type", "")).lower()
            if "vl" in model_type or "vision" in model_type:
                return True
        except Exception:
            pass
        lowered = model_name_or_path.lower()
        return ("-vl" in lowered) or ("vision" in lowered)

    def _load_backbone(self, config: VLMBackboneConfig) -> nn.Module:
        lowered = config.model_name_or_path.lower()
        is_qwen35 = any(t in lowered for t in ("qwen3.5", "qwen3_5", "qwen35"))

        common_kwargs: Dict[str, Any] = {
            "torch_dtype": config.torch_dtype,
            "attn_implementation": config.attn_implementation,
            "trust_remote_code": config.trust_remote_code,
        }

        if self.model_is_vl:
            is_qwen3_vl = ("qwen3-vl" in lowered) or ("qwen3_vl" in lowered)
            if is_qwen3_vl:
                if Qwen3VLForConditionalGeneration is None:
                    raise ImportError(
                        "Qwen3-VL requires Qwen3VLForConditionalGeneration from transformers."
                    )
                qwen3_vl_kwargs: Dict[str, Any] = {
                    "torch_dtype": config.torch_dtype,
                    "attn_implementation": config.attn_implementation,
                    "trust_remote_code": config.trust_remote_code,
                }
                return self._from_pretrained_with_fallback(
                    Qwen3VLForConditionalGeneration,
                    config.model_name_or_path,
                    qwen3_vl_kwargs,
                )
            return self._from_pretrained_with_fallback(
                AutoModelForCausalLM,
                config.model_name_or_path,
                common_kwargs,
            )

        if is_qwen35:
            if Qwen3_5ForConditionalGeneration is None:
                raise ImportError(
                    "Qwen3.5 requires Qwen3_5ForConditionalGeneration from transformers."
                )
            qwen35_kwargs: Dict[str, Any] = {
                "torch_dtype": config.torch_dtype,
                "attn_implementation": config.attn_implementation,
                "trust_remote_code": config.trust_remote_code,
            }
            return self._from_pretrained_with_fallback(
                Qwen3_5ForConditionalGeneration,
                config.model_name_or_path,
                qwen35_kwargs,
            )

        return self._from_pretrained_with_fallback(
            AutoModelForCausalLM,
            config.model_name_or_path,
            common_kwargs,
        )

    # ------------------------------------------------------------------
    # Freezing & LoRA via PEFT
    # ------------------------------------------------------------------

    def _freeze_vision_encoder(self):
        vision_modules = ["visual", "vision_tower", "vision_model"]
        for name in vision_modules:
            module = getattr(self.model, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
                logger.info("Froze vision encoder module: %s", name)
                return
        logger.info("No vision encoder module found to freeze.")

    def _apply_lora(self, config: VLMBackboneConfig):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise ModuleNotFoundError(
                "LoRA is enabled (lora_rank > 0), but `peft` is not installed. "
                "Install it with `pip install peft`."
            ) from exc

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # PEFT adapter save / load
    # ------------------------------------------------------------------

    def save_adapter(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        if self._is_peft_model():
            self.model.save_pretrained(save_dir)
            logger.info("PEFT adapter saved to %s", save_dir)
        else:
            torch.save(
                {k: v.cpu() for k, v in self.state_dict().items()},
                os.path.join(save_dir, "vlm_backbone.pt"),
            )
            logger.info("Full VLM state saved to %s", save_dir)

    def load_adapter(self, load_dir: str, device: Optional[torch.device] = None) -> None:
        if self._is_peft_model():
            sft_path = os.path.join(load_dir, "adapter_model.safetensors")
            bin_path = os.path.join(load_dir, "adapter_model.bin")

            if os.path.isfile(sft_path):
                from safetensors.torch import load_file
                state = load_file(sft_path)
            elif os.path.isfile(bin_path):
                state = torch.load(bin_path, map_location="cpu")
            else:
                logger.warning("No adapter checkpoint found in %s", load_dir)
                return

            from peft import set_peft_model_state_dict
            load_result = set_peft_model_state_dict(
                self.model, state, adapter_name="default",
            )
            missing = getattr(load_result, "missing_keys", []) or []
            unexpected = getattr(load_result, "unexpected_keys", []) or []

            if hasattr(self.model, "set_adapter"):
                self.model.set_adapter("default")

            lora_params = [
                (n, p) for n, p in self.model.named_parameters() if "lora_" in n
            ]
            num_lora = len(lora_params)
            if num_lora == 0:
                logger.error("No LoRA parameters found after load_adapter.")
                return

            total_abs = sum(p.detach().float().abs().sum().item() for _, p in lora_params)
            mean_abs = total_abs / max(1, sum(p.numel() for _, p in lora_params))
            sample_name, sample_param = lora_params[0]

            logger.info(
                "PEFT adapter loaded from %s (%d tensors -> %d LoRA params; "
                "missing=%d, unexpected=%d; active=%s; mean|w|=%.4e)",
                load_dir,
                len(state),
                num_lora,
                len(missing),
                len(unexpected),
                getattr(self.model, "active_adapters", "?"),
                mean_abs,
            )
            logger.info(
                "  audit sample: %s mean|w|=%.4e max|w|=%.4e",
                sample_name,
                sample_param.detach().float().abs().mean().item(),
                sample_param.detach().float().abs().max().item(),
            )
        else:
            pt_path = os.path.join(load_dir, "vlm_backbone.pt")
            if os.path.isfile(pt_path):
                state = torch.load(pt_path, map_location=device or "cpu")
                self.load_state_dict(state)
                logger.info("Full VLM state loaded from %s", pt_path)

    def _is_peft_model(self) -> bool:
        try:
            from peft import PeftModel
            return isinstance(self.model, PeftModel)
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_multiview_prompt(
        self,
        num_history_steps: int,
        num_current_cameras: int = 8,
        mission_command: str = "LANE_FOLLOW",
        include_reasoning_prefix: bool = True,
        reasoning_format: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        analysis = (
            "Analyze the scene and provide the structured reasoning requested below."
            if include_reasoning_prefix
            else "Analyze the scene to support the driving mission."
        )
        parts.append(
            "You are an autonomous driving assistant. "
            "You are given multi-view camera images from a self-driving vehicle "
            "at multiple timesteps. Each image is paired with a text label that "
            f"identifies its timestep and camera view. {analysis}\n\n"
        )
        parts.append(
            f"You are given {num_history_steps} history timesteps plus the current "
            f"timestep across {num_current_cameras} cameras.\n"
        )
        parts.append(f"Mission command: {mission_command}\n\n")

        if include_reasoning_prefix:
            fmt = str(reasoning_format or self.config.reasoning_format).lower().strip()
            if fmt == "driving":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Driving] scene description, critical components, decision, and trace\n\n"
                )
            elif fmt == "spatial_driving":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Spatial] scene/layout, object relations, and map context\n"
                    "2. [Driving] scene description, critical components, decision, and trace\n\n"
                )
            elif fmt == "spatial":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Spatial] scene/layout, object relations, and map context\n\n"
                )
            elif fmt == "driving_counterfactual":
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Driving] scene description, critical components, decision, and trace\n"
                    "2. [Counterfactual] alternative and unsafe actions with risk rationale\n\n"
                )
            else:
                parts.append(
                    "Based on the multi-view multi-frame observations, provide:\n"
                    "1. [Spatial] scene/layout, object relations, and map context\n"
                    "2. [Driving] scene description, critical components, decision, and trace\n"
                    "3. [Counterfactual] alternative and unsafe actions with risk rationale\n\n"
                )

        return "".join(parts)

    # ------------------------------------------------------------------
    # Processor / input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        images: List[Any],
        text_prompt: str,
        image_contexts: Optional[List[str]] = None,
        assistant_response: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not images:
            raise ValueError("prepare_inputs requires at least one image.")

        content: List[Dict[str, Any]] = []
        if image_contexts is not None and len(image_contexts) != len(images):
            raise ValueError(
                f"image_contexts length ({len(image_contexts)}) must match "
                f"images length ({len(images)})"
            )
        if image_contexts is None:
            content.extend([{"type": "image", "image": img} for img in images])
        else:
            for img, context in zip(images, image_contexts):
                content.append({"type": "text", "text": context})
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text_prompt})

        user_msg = {"role": "user", "content": content}
        prefix_text = self.processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True,
        )

        if assistant_response is None:
            full_text = prefix_text
        else:
            assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            }
            full_text = self.processor.apply_chat_template(
                [user_msg, assistant_msg],
                tokenize=False,
                add_generation_prompt=False,
            )

        inputs = self.processor(
            text=[full_text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        tok = self.processor.tokenizer
        prefix_tok = tok(prefix_text, add_special_tokens=False)["input_ids"]
        full_tok = tok(full_text, add_special_tokens=False)["input_ids"]
        assistant_token_count = max(0, len(full_tok) - len(prefix_tok))
        total_len = int(inputs["input_ids"].shape[-1])
        prompt_length = total_len - assistant_token_count

        inputs["prompt_length"] = torch.tensor([prompt_length], dtype=torch.long)

        if assistant_response is not None:
            built_labels = inputs["input_ids"].clone()
            built_labels[:, :prompt_length] = -100
            if "attention_mask" in inputs:
                built_labels[inputs["attention_mask"] == 0] = -100
            inputs["labels"] = built_labels
            # Explicit mask used for selecting the reasoning KV span even when
            # reasoning loss is disabled (e.g. teacher-forced trajectory eval).
            inputs["reasoning_token_mask"] = built_labels.ne(-100)
        elif labels is not None:
            inputs["labels"] = labels
            inputs["reasoning_token_mask"] = labels.ne(-100)

        return inputs

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_layer(past_key_values: Any, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read one layer from multiple Transformers Cache API versions."""
        # New Cache API: cache.layers[i].keys / .values
        layers = getattr(past_key_values, "layers", None)
        if layers is not None:
            layer = layers[layer_idx]
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is not None and value is not None:
                return key, value

        # Older DynamicCache API: key_cache / value_cache lists
        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            return key_cache[layer_idx], value_cache[layer_idx]

        # Legacy tuple cache
        if isinstance(past_key_values, (tuple, list)):
            layer = past_key_values[layer_idx]
            return layer[0], layer[1]

        # Last-resort compatibility path
        to_legacy = getattr(past_key_values, "to_legacy_cache", None)
        if callable(to_legacy):
            legacy = to_legacy()
            return legacy[layer_idx][0], legacy[layer_idx][1]

        raise TypeError(
            f"Unsupported past_key_values type: {type(past_key_values)!r}"
        )

    @staticmethod
    def _normalize_cache_layout(
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize cache tensors to [B, Hkv, T, Dh]."""
        if key.ndim != 4 or value.ndim != 4:
            raise RuntimeError(
                f"Expected 4-D KV cache, got K={tuple(key.shape)}, V={tuple(value.shape)}"
            )
        if key.shape[2] == seq_len:
            return key, value
        if key.shape[1] == seq_len:
            return key.transpose(1, 2), value.transpose(1, 2)
        raise RuntimeError(
            "Cannot identify cache sequence dimension: "
            f"K={tuple(key.shape)}, input_seq_len={seq_len}"
        )

    def _extract_reasoning_kv(
        self,
        past_key_values: Any,
        selected_kv_layers: List[int],
        reasoning_token_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Extract selected layer K/V only for assistant reasoning tokens.

        Returned tensors are per-sample and still keep the original VLM KV-head
        structure: [B, Hkv, Tr, Dh]. No planner-side K/V reprojection is applied.
        """
        if reasoning_token_mask is None:
            raise ValueError(
                "reasoning_token_mask is required for reasoning-only KV injection."
            )

        token_mask = reasoning_token_mask.bool() & attention_mask.bool()
        seq_len = int(attention_mask.shape[-1])
        result: Dict[int, Dict[str, torch.Tensor]] = {}

        for layer_idx in selected_kv_layers:
            if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
                raise ValueError(
                    f"KV layer index {layer_idx} out of range [0, {self.num_hidden_layers - 1}]"
                )

            key, value = self._cache_layer(past_key_values, layer_idx)
            key, value = self._normalize_cache_layout(key, value, seq_len)

            # Training code feeds one sample at a time to the VLM, but this is
            # implemented for general B as well by requiring equal selected span
            # lengths inside this local forward. The trainer later pads samples.
            selected_keys = []
            selected_values = []
            lengths = []
            for b in range(key.shape[0]):
                idx = token_mask[b].nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    raise RuntimeError(
                        "Reasoning KV selection produced zero tokens. Check prompt/labels."
                    )
                selected_keys.append(key[b:b+1, :, idx, :])
                selected_values.append(value[b:b+1, :, idx, :])
                lengths.append(int(idx.numel()))

            if len(set(lengths)) != 1:
                raise RuntimeError(
                    "VLMBackbone.forward expects equal reasoning span lengths inside one "
                    "local call. The official trainer calls it sample-by-sample."
                )

            result[layer_idx] = {
                "key": torch.cat(selected_keys, dim=0),
                "value": torch.cat(selected_values, dim=0),
                "mask": torch.ones(
                    key.shape[0], lengths[0], device=key.device, dtype=torch.bool
                ),
            }

        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        reasoning_token_mask: Optional[torch.Tensor] = None,
        selected_kv_layers: Optional[List[int]] = None,
        return_features: bool = True,
    ) -> Dict[str, Any]:
        # prompt_lengths is intentionally retained in the public API so the rest
        # of the nuVLA training pipeline stays compatible, although planner
        # conditioning now uses reasoning KV rather than prompt-side final hidden.
        del prompt_lengths

        want_kv = return_features and bool(selected_kv_layers)
        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "output_hidden_states": False,
            "use_cache": want_kv,
            "return_dict": True,
        }
        if self.model_is_vl and pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw
            if mm_token_type_ids is not None:
                model_inputs["mm_token_type_ids"] = mm_token_type_ids

        outputs = self.model(**model_inputs)

        result: Dict[str, Any] = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss

        if want_kv:
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError(
                    "VLM did not return past_key_values with use_cache=True. "
                    "Check the installed transformers/Qwen3-VL implementation."
                )
            if reasoning_token_mask is None:
                if labels is None:
                    raise ValueError(
                        "Need reasoning_token_mask or labels for reasoning-only KV extraction."
                    )
                reasoning_token_mask = labels.ne(-100)

            result["layer_kv"] = self._extract_reasoning_kv(
                past_key_values=past_key_values,
                selected_kv_layers=list(selected_kv_layers or []),
                reasoning_token_mask=reasoning_token_mask,
                attention_mask=attention_mask,
            )

        return result

    def get_feature_dim(self) -> int:
        return self.feature_dim
