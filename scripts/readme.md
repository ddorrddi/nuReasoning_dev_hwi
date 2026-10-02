nuVLA Layer-wise Reasoning KV Injection
=======================================

Target repository:
https://github.com/nureasoning/nureasoning-devkit

Dataset used in this setup:
/media/HDD/nuR_ds/data/train/part_1

Changed files only:
1) nureasoning/nuvla/models/vlm_backbone.py
   - keeps original prompt/processor/LoRA/reasoning SFT behavior
   - runs VLM with use_cache=True
   - extracts actual selected layer KV cache
   - selects only assistant/reasoning-token span

2) nureasoning/nuvla/models/action_expert.py
   - keeps StateEncoder / ActionEncoder / flow matching / decoder
   - planner hidden states create Q
   - VLM K/V are injected directly; no planner-side K/V reprojection
   - planner Q projection is expanded into VLM KV-head space

3) train.py is minimally patched by patch_train_layerwise_kv.py
   - no data-loader changes
   - no optimizer/scheduler/checkpoint changes
   - replaces vlm_features path with layer_kv path
   - adds --kv_layers (1-based)

Install
-------
From the nureasoning-devkit repository root:

cp nureasoning/nuvla/models/vlm_backbone.py nureasoning/nuvla/models/vlm_backbone.py.bak
cp nureasoning/nuvla/models/action_expert.py nureasoning/nuvla/models/action_expert.py.bak

cp /path/to/this/package/nureasoning/nuvla/models/vlm_backbone.py nureasoning/nuvla/models/vlm_backbone.py
cp /path/to/this/package/nureasoning/nuvla/models/action_expert.py nureasoning/nuvla/models/action_expert.py
python /path/to/this/package/patch_train_layerwise_kv.py nureasoning/nuvla/train.py

Compile check
-------------
python -m py_compile \
  nureasoning/nuvla/models/vlm_backbone.py \
  nureasoning/nuvla/models/action_expert.py \
  nureasoning/nuvla/train.py

Recommended first run (structured, train only)
----------------------------------------------
CUDA_VISIBLE_DEVICES=0 python -m nureasoning.nuvla.train \
  --data_root /media/HDD/nuR_ds/data/train/part_1 \
  --test_data_root "" \
  --reasoning_mode structured \
  --reasoning_format spatial_driving_counterfactual \
  --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
  --kv_layers 4,9,14,19,24,28 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --output_dir ./nureasoning_vla_layerwise_reasoning_kv

Notes
-----
- --kv_layers is 1-based to match diagrams/paper notation.
- Internally Qwen layers are converted to 0-based indices.
- Default public DiT has 12 blocks with interleaved cross/self blocks, hence 6 cross-attention blocks.
  The default selected VLM layers are also six depths: 4,9,14,19,24,28.
- If the number of selected VLM layers differs from the number of planner cross-attention blocks,
  the code maps planner cross-attention depth monotonically to the nearest selected VLM depth.
- Training is teacher-forced: reasoning text from the dataset is placed in the assistant turn,
  and only that reasoning-token span is used as planner K/V memory.
- Free-running inference requires the same KV extraction after generated reasoning; this package changes
  the training path and the train.py teacher-forced trajectory evaluation path.
