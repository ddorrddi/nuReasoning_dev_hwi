#!/usr/bin/env python3
"""Patch the public nureasoning/nuvla/train.py for layer-wise reasoning KV injection.

Target: current public nureasoning-devkit train.py structure.
The script creates train.py.bak before changing anything and aborts if expected
anchors are missing, so unrelated public training behavior stays untouched.
"""

from pathlib import Path
import re
import sys

if len(sys.argv) != 2:
    print("Usage: python patch_train_layerwise_kv.py /path/to/nureasoning/nuvla/train.py")
    raise SystemExit(2)

path = Path(sys.argv[1]).expanduser().resolve()
if not path.is_file():
    raise SystemExit(f"train.py not found: {path}")

text = path.read_text(encoding="utf-8")
original = text

# -----------------------------------------------------------------------------
# 1) Build model: infer VLM KV geometry, parse selected 1-based layers, and
#    pass them to the modified action expert.
# -----------------------------------------------------------------------------
old = '''        self.vlm = VLMBackbone(vlm_config).to(self.device)\n\n        vlm_feature_dim = self.vlm.feature_dim\n\n        action_config = ActionExpertConfig(\n            vlm_feature_dim=vlm_feature_dim,\n'''
new = '''        self.vlm = VLMBackbone(vlm_config).to(self.device)\n\n        vlm_feature_dim = self.vlm.feature_dim\n        vlm_num_layers, kv_num_heads, kv_head_dim = self.vlm.get_kv_spec()\n\n        # --kv_layers uses human-readable 1-based VLM layer numbers.\n        kv_layers_1based = [\n            int(x.strip()) for x in args.kv_layers.split(",") if x.strip()\n        ]\n        if not kv_layers_1based:\n            raise ValueError("--kv_layers must contain at least one layer number")\n        if min(kv_layers_1based) < 1 or max(kv_layers_1based) > vlm_num_layers:\n            raise ValueError(\n                f"--kv_layers={kv_layers_1based} invalid for a {vlm_num_layers}-layer VLM"\n            )\n        self.selected_kv_layers = [x - 1 for x in kv_layers_1based]\n\n        action_config = ActionExpertConfig(\n            vlm_feature_dim=vlm_feature_dim,\n            kv_layer_indices=self.selected_kv_layers,\n            kv_num_heads=kv_num_heads,\n            kv_head_dim=kv_head_dim,\n'''
if old not in text:
    raise SystemExit("Anchor 1 not found: public _build_models block changed; aborting safely.")
text = text.replace(old, new, 1)

old = '''            logger.info(f"VLM feature dim (inferred): {vlm_feature_dim}")\n            logger.info(\n'''
new = '''            logger.info(f"VLM feature dim (inferred): {vlm_feature_dim}")\n            logger.info(\n                "Layer-wise reasoning KV injection: VLM layers %s (1-based), "\n                "KV heads=%d, head_dim=%d",\n                [x + 1 for x in self.selected_kv_layers],\n                kv_num_heads,\n                kv_head_dim,\n            )\n            logger.info(\n'''
if old not in text:
    raise SystemExit("Anchor 2 not found: logging block changed; aborting safely.")
text = text.replace(old, new, 1)

# -----------------------------------------------------------------------------
# 2) Replace final-hidden-state batching/extraction with layer-wise reasoning KV.
# -----------------------------------------------------------------------------
pattern = re.compile(
    r'''    @staticmethod\n    def _stack_vlm_features\([\s\S]*?\n    # ------------------------------------------------------------------\n    # Train step\n    # ------------------------------------------------------------------''',
    re.MULTILINE,
)
replacement = '''    @staticmethod
    def _stack_layer_kv(
        layer_kv_list: list[Dict[int, Dict[str, torch.Tensor]]],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Pad variable reasoning-token lengths and re-batch layer-wise KV.

        Each per-sample entry is:
          layer -> key/value [1, Hkv, Tr, Dh], mask [1, Tr]
        Returned batch entry is:
          layer -> key/value [B, Hkv, Tmax, Dh], mask [B, Tmax]
        """
        if not layer_kv_list:
            raise ValueError("layer_kv_list must not be empty")

        layer_ids = sorted(layer_kv_list[0].keys())
        for sample in layer_kv_list[1:]:
            if sorted(sample.keys()) != layer_ids:
                raise RuntimeError("All samples must expose the same selected VLM KV layers")

        batched: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in layer_ids:
            max_len = max(sample[layer_idx]["key"].shape[2] for sample in layer_kv_list)
            keys = []
            values = []
            masks = []

            for sample in layer_kv_list:
                key = sample[layer_idx]["key"]
                value = sample[layer_idx]["value"]
                mask = sample[layer_idx]["mask"]
                pad_len = max_len - key.shape[2]

                if pad_len > 0:
                    key = torch.cat(
                        [key, key.new_zeros(key.shape[0], key.shape[1], pad_len, key.shape[3])],
                        dim=2,
                    )
                    value = torch.cat(
                        [value, value.new_zeros(value.shape[0], value.shape[1], pad_len, value.shape[3])],
                        dim=2,
                    )
                    mask = torch.cat(
                        [mask, torch.zeros(mask.shape[0], pad_len, device=mask.device, dtype=torch.bool)],
                        dim=1,
                    )

                keys.append(key)
                values.append(value)
                masks.append(mask)

            batched[layer_idx] = {
                "key": torch.cat(keys, dim=0),
                "value": torch.cat(values, dim=0),
                "mask": torch.cat(masks, dim=0),
            }

        return batched

    def _extract_vlm_kv(
        self,
        batch: Dict[str, Any],
        *,
        compute_reasoning_loss: bool,
    ) -> tuple[Dict[int, Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """Teacher-force reasoning and extract selected reasoning-token KV caches."""
        sample_inputs = self._prepare_vlm_inputs(batch)
        layer_kv_list = []
        reasoning_losses = []

        vlm_module = self.vlm if compute_reasoning_loss else self.vlm_unwrapped

        for sample_input in sample_inputs:
            with torch.amp.autocast("cuda", enabled=self.use_bf16, dtype=torch.bfloat16):
                vlm_outputs = vlm_module(
                    input_ids=sample_input["input_ids"],
                    attention_mask=sample_input["attention_mask"],
                    pixel_values=sample_input.get("pixel_values"),
                    image_grid_thw=sample_input.get("image_grid_thw"),
                    mm_token_type_ids=sample_input.get("mm_token_type_ids"),
                    labels=sample_input.get("labels") if compute_reasoning_loss else None,
                    prompt_lengths=sample_input.get("prompt_lengths"),
                    reasoning_token_mask=sample_input.get("reasoning_token_mask"),
                    selected_kv_layers=self.selected_kv_layers,
                    return_features=True,
                )

            layer_kv_list.append(vlm_outputs["layer_kv"])
            if compute_reasoning_loss:
                reasoning_losses.append(
                    vlm_outputs.get("loss", torch.tensor(0.0, device=self.device))
                )

        layer_kv = self._stack_layer_kv(layer_kv_list)
        reasoning_loss = None
        if compute_reasoning_loss:
            reasoning_loss = torch.stack(reasoning_losses).mean()

        return layer_kv, reasoning_loss

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------'''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit("Anchor 3 not found: feature extraction section changed; aborting safely.")

# -----------------------------------------------------------------------------
# 3) Training/evaluation calls: vlm_features -> layer_kv.
# -----------------------------------------------------------------------------
repls = [
    (
'''        vlm_features, reasoning_loss = self._extract_vlm_features(\n            batch,\n            compute_reasoning_loss=True,\n        )''',
'''        layer_kv, reasoning_loss = self._extract_vlm_kv(\n            batch,\n            compute_reasoning_loss=True,\n        )'''
    ),
    (
'''                vlm_features=vlm_features,\n                ego_state=ego_state,''',
'''                layer_kv=layer_kv,\n                ego_state=ego_state,'''
    ),
    (
'''        vlm_features, _ = self._extract_vlm_features(\n            batch,\n            compute_reasoning_loss=False,\n        )''',
'''        layer_kv, _ = self._extract_vlm_kv(\n            batch,\n            compute_reasoning_loss=False,\n        )'''
    ),
    (
'''            pred_traj = action_expert.sample(\n                vlm_features, ego_state, ego_history_trajectories,\n            )''',
'''            pred_traj = action_expert.sample(\n                layer_kv, ego_state, ego_history_trajectories,\n            )'''
    ),
]
for idx, (src, dst) in enumerate(repls, start=4):
    if src not in text:
        raise SystemExit(f"Anchor {idx} not found; aborting safely.")
    text = text.replace(src, dst, 1)

# Update the train-step comments without changing behavior.
text = text.replace(
    "Step 1: Forward VLM, compute reasoning loss, extract features",
    "Step 1: Forward VLM, compute reasoning loss, extract layer-wise reasoning KV",
    1,
)
text = text.replace(
    "Step 2: Forward action expert with VLM features, compute action loss",
    "Step 2: Forward action expert with injected VLM reasoning KV, compute action loss",
    1,
)

# -----------------------------------------------------------------------------
# 4) Add CLI argument. 1-based layer numbering intentionally matches diagram.
# -----------------------------------------------------------------------------
old = '''    parser.add_argument("--num_dit_heads", type=int, default=8)\n    parser.add_argument("--dropout", type=float, default=0.1)\n'''
new = '''    parser.add_argument("--num_dit_heads", type=int, default=8)\n    parser.add_argument(\n        "--kv_layers",\n        type=str,\n        default="4,9,14,19,24,28",\n        help=(\n            "Comma-separated 1-based VLM language-layer numbers whose reasoning "\n            "K/V caches are injected into planner cross-attention blocks. "\n            "For Qwen3-VL-2B (28 layers), default selects six depths for the six "\n            "cross-attention blocks of the default 12-layer interleaved DiT."\n        ),\n    )\n    parser.add_argument("--dropout", type=float, default=0.1)\n'''
if old not in text:
    raise SystemExit("Anchor 8 not found: argparse action-expert block changed; aborting safely.")
text = text.replace(old, new, 1)

# Optional log line.
old = '''            logger.info(f"  Action loss weight: {args.action_loss_weight}")\n            logger.info("  Precision: bf16-only (autocast enabled on CUDA)")\n'''
new = '''            logger.info(f"  Action loss weight: {args.action_loss_weight}")\n            logger.info(f"  Reasoning KV layers (1-based): {args.kv_layers}")\n            logger.info("  Precision: bf16-only (autocast enabled on CUDA)")\n'''
if old not in text:
    raise SystemExit("Anchor 9 not found: training log block changed; aborting safely.")
text = text.replace(old, new, 1)

if text == original:
    raise SystemExit("No changes made.")

backup = path.with_suffix(path.suffix + ".bak")
if not backup.exists():
    backup.write_text(original, encoding="utf-8")
path.write_text(text, encoding="utf-8")

print(f"Patched: {path}")
print(f"Backup : {backup}")
print("Added --kv_layers (1-based; default 4,9,14,19,24,28)")
