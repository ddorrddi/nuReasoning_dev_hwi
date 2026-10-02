#!/usr/bin/env python3

import argparse
import copy
import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
from PIL import Image

from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


# =============================================================================
# 기본 설정
# =============================================================================

HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

DEFAULT_DATA_ROOT = Path(
    "/media/HDD/nuR_ds/data/train/part_1"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/home/lhh/lab/dataset/nuReasoning/"
    "train_filtered/part_1/dr_cf_reasoning"
)

DEFAULT_MODEL_DIR = Path(
    "/home/lhh/lab/models/vlm/"
    "Qwen3-VL-2B-Instruct"
)


# -----------------------------------------------------------------------------
# Qwen이 실제로 보는 카메라
# -----------------------------------------------------------------------------
# 현재 frame의 전방 3개만 사용한다.
# history frame은 사용하지 않는다.
# left/right/back/back_left/back_right는 사용하지 않는다.
# -----------------------------------------------------------------------------

ALLOWED_CAMERAS = [
    "front",
    "front_left",
    "front_right",
]


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320


# =============================================================================
# Argument
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Filter nuReasoning Driving and Counterfactual "
            "using Qwen3-VL with FRONT / FRONT_LEFT / FRONT_RIGHT."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Number of actual Qwen inference JSONs. "
            "0 = all."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip JSONs whose filtered output already exists."
        ),
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Basic utility
# =============================================================================

def has_content(value):

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, dict):
        return len(value) > 0

    if isinstance(value, list):
        return len(value) > 0

    return True


# =============================================================================
# Qwen model
# =============================================================================

def ensure_model(
    model_dir: Path,
    allow_download: bool,
):

    model_dir = (
        model_dir
        .expanduser()
        .resolve()
    )

    config_path = (
        model_dir
        / "config.json"
    )

    if config_path.exists():

        print()
        print(
            "[MODEL] Existing local model:"
        )
        print(
            f"        {model_dir}"
        )

        return model_dir

    if not allow_download:

        raise FileNotFoundError(
            f"Qwen model not found:\n"
            f"{model_dir}"
        )

    print()
    print("=" * 100)
    print("[MODEL DOWNLOAD]")
    print("=" * 100)

    print(
        f"MODEL : {HF_MODEL_ID}"
    )

    print(
        f"DEST  : {model_dir}"
    )

    model_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=str(
            model_dir
        ),
    )

    if not config_path.exists():

        raise RuntimeError(
            "Model download finished, "
            "but config.json does not exist."
        )

    print()
    print(
        "[MODEL] Download complete."
    )

    return model_dir


def load_qwen(
    model_dir: Path,
):

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    # -------------------------------------------------------------------------
    # CUDA_VISIBLE_DEVICES=0 로 실행하므로
    # 여기서 cuda:0 = 실제 RTX 3080 Ti
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda:0"
    )

    print()
    print("=" * 100)
    print("[MODEL LOAD]")
    print("=" * 100)

    print(
        f"CUDA device : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Model path  : "
        f"{model_dir}"
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
    )

    # -------------------------------------------------------------------------
    # 중요:
    #
    # FlashAttention 사용하지 않는다.
    #
    # 이전 오류:
    # RuntimeError:
    # FlashAttention only supports Ampere GPUs or newer.
    #
    # 따라서 SDPA를 명시적으로 사용한다.
    # -------------------------------------------------------------------------

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            str(model_dir),

            torch_dtype=torch.bfloat16,

            attn_implementation="sdpa",

            low_cpu_mem_usage=True,

            trust_remote_code=True,
        )
    )

    model = model.to(
        device
    )

    model.eval()

    print()
    print(
        "[MODEL] Loaded successfully"
    )

    print(
        f"[MODEL] Device: "
        f"{next(model.parameters()).device}"
    )

    return (
        processor,
        model,
    )


# =============================================================================
# Original image paths
# =============================================================================

def get_clip_root(
    json_path: Path,
):

    # 구조:
    #
    # part_1/
    #   CLIP/
    #     reasoning/
    #       TIMESTAMP.json
    #
    # json_path.parent        = reasoning
    # json_path.parent.parent = CLIP

    return (
        json_path
        .parent
        .parent
    )


def extract_three_camera_paths(
    data,
    json_path: Path,
):

    spatial = data.get(
        "Spatial"
    )

    if not isinstance(
        spatial,
        dict,
    ):

        raise RuntimeError(
            "Spatial missing."
        )

    per_camera = spatial.get(
        "per_camera_results"
    )

    if not isinstance(
        per_camera,
        dict,
    ):

        raise RuntimeError(
            "Spatial.per_camera_results missing."
        )

    clip_root = (
        get_clip_root(
            json_path
        )
    )

    paths = {}

    for camera in ALLOWED_CAMERAS:

        camera_data = (
            per_camera.get(
                camera
            )
        )

        if not isinstance(
            camera_data,
            dict,
        ):

            raise RuntimeError(
                f"Camera missing: {camera}"
            )

        relative_path = (
            camera_data.get(
                "image_path"
            )
        )

        if not relative_path:

            raise RuntimeError(
                f"image_path missing: "
                f"{camera}"
            )

        absolute_path = (
            clip_root
            / relative_path
        )

        if not absolute_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n"
                f"camera={camera}\n"
                f"path={absolute_path}"
            )

        paths[
            camera
        ] = absolute_path

    return paths


def load_three_images(
    image_paths,
):

    images = []

    for camera in ALLOWED_CAMERAS:

        image_path = (
            image_paths[
                camera
            ]
        )

        with Image.open(
            image_path
        ) as img:

            img = img.convert(
                "RGB"
            )

            # -----------------------------------------------------------------
            # 현재 VLM 입력 조건과 동일:
            #
            # current image only
            # front/front_left/front_right
            # 320 x 320
            # -----------------------------------------------------------------

            img = img.resize(
                (
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                ),
                Image.Resampling.LANCZOS,
            )

            images.append(
                img.copy()
            )

    if len(images) != 3:

        raise RuntimeError(
            f"Expected 3 images, "
            f"got {len(images)}"
        )

    return images


# =============================================================================
# Counterfactual item extraction
# =============================================================================

def build_counterfactual_units(
    counterfactual,
):
    """
    Counterfactual 내부 항목을 개별 Qwen 판정 단위로 만든다.


    지원 구조 1:

    Counterfactual = {
        "Alternative actions": [
            item0,
            item1,
            ...
        ],
        "Top safety-critical actions": [
            item0,
            item1,
            ...
        ]
    }


    이 경우 각 list element가 하나의 unit.


    지원 구조 2:

    Counterfactual = {
        "some_key": value,
        "another_key": value
    }

    value가 list가 아니면
    해당 key:value 전체를 하나의 unit으로 취급.


    지원 구조 3:

    Counterfactual 자체가 list라면
    각 list element를 하나의 unit으로 취급.


    반환 예:

    [
        {
            "id": "CF_000",
            "kind": "dict_list_item",
            "key": "Alternative actions",
            "index": 0,
            "label": "Alternative actions[0]",
            "value": ...
        },
        ...
    ]
    """

    units = []

    counter = 0

    # -------------------------------------------------------------------------
    # Counterfactual이 dict
    # -------------------------------------------------------------------------

    if isinstance(
        counterfactual,
        dict,
    ):

        for key, value in (
            counterfactual.items()
        ):

            # -----------------------------------------------------------------
            # list이면 element별 판정
            # -----------------------------------------------------------------

            if isinstance(
                value,
                list,
            ):

                for index, item in enumerate(
                    value
                ):

                    unit_id = (
                        f"CF_{counter:03d}"
                    )

                    units.append(
                        {
                            "id": unit_id,
                            "kind": (
                                "dict_list_item"
                            ),
                            "key": key,
                            "index": index,
                            "label": (
                                f"{key}[{index}]"
                            ),
                            "value": item,
                        }
                    )

                    counter += 1

            # -----------------------------------------------------------------
            # list가 아니면
            # 해당 top-level field 전체를 하나의 항목으로 판정
            # -----------------------------------------------------------------

            else:

                unit_id = (
                    f"CF_{counter:03d}"
                )

                units.append(
                    {
                        "id": unit_id,
                        "kind": (
                            "dict_field"
                        ),
                        "key": key,
                        "index": None,
                        "label": key,
                        "value": value,
                    }
                )

                counter += 1

    # -------------------------------------------------------------------------
    # Counterfactual 자체가 list
    # -------------------------------------------------------------------------

    elif isinstance(
        counterfactual,
        list,
    ):

        for index, item in enumerate(
            counterfactual
        ):

            unit_id = (
                f"CF_{counter:03d}"
            )

            units.append(
                {
                    "id": unit_id,
                    "kind": (
                        "root_list_item"
                    ),
                    "key": None,
                    "index": index,
                    "label": (
                        f"Counterfactual[{index}]"
                    ),
                    "value": item,
                }
            )

            counter += 1

    # -------------------------------------------------------------------------
    # 그 외 scalar
    # -------------------------------------------------------------------------

    else:

        units.append(
            {
                "id": "CF_000",
                "kind": "root_value",
                "key": None,
                "index": None,
                "label": "Counterfactual",
                "value": counterfactual,
            }
        )

    return units


# =============================================================================
# Counterfactual reconstruction
# =============================================================================

def filter_counterfactual(
    original_counterfactual,
    units,
    verdicts,
):
    """
    원본 Counterfactual 형태를 최대한 그대로 유지하면서
    DROP된 unit만 제거한다.

    verdicts:

    {
        "CF_000": True,
        "CF_001": False,
        ...
    }

    return:
        filtered_counterfactual,
        keep_count,
        drop_count
    """

    keep_count = 0
    drop_count = 0

    # -------------------------------------------------------------------------
    # 원본이 dict
    # -------------------------------------------------------------------------

    if isinstance(
        original_counterfactual,
        dict,
    ):

        filtered = {}

        # key별 unit들 묶기
        units_by_key = {}

        for unit in units:

            key = unit[
                "key"
            ]

            units_by_key.setdefault(
                key,
                [],
            ).append(
                unit
            )

        for key, original_value in (
            original_counterfactual.items()
        ):

            key_units = (
                units_by_key.get(
                    key,
                    [],
                )
            )

            # -----------------------------------------------------------------
            # list field
            # -----------------------------------------------------------------

            if isinstance(
                original_value,
                list,
            ):

                filtered_list = []

                # 원본 index 순서 유지
                key_units = sorted(
                    key_units,
                    key=lambda x: (
                        x["index"]
                    ),
                )

                for unit in key_units:

                    unit_id = (
                        unit["id"]
                    )

                    keep = verdicts[
                        unit_id
                    ]

                    if keep:

                        filtered_list.append(
                            copy.deepcopy(
                                unit["value"]
                            )
                        )

                        keep_count += 1

                    else:

                        drop_count += 1

                # 하나라도 남은 경우만 key 유지
                if len(
                    filtered_list
                ) > 0:

                    filtered[
                        key
                    ] = filtered_list

            # -----------------------------------------------------------------
            # non-list field
            # -----------------------------------------------------------------

            else:

                if len(
                    key_units
                ) != 1:

                    raise RuntimeError(
                        f"Unexpected CF unit count "
                        f"for key={key}: "
                        f"{len(key_units)}"
                    )

                unit = (
                    key_units[0]
                )

                keep = verdicts[
                    unit["id"]
                ]

                if keep:

                    filtered[
                        key
                    ] = copy.deepcopy(
                        original_value
                    )

                    keep_count += 1

                else:

                    drop_count += 1

        return (
            filtered,
            keep_count,
            drop_count,
        )

    # -------------------------------------------------------------------------
    # 원본이 list
    # -------------------------------------------------------------------------

    elif isinstance(
        original_counterfactual,
        list,
    ):

        filtered = []

        units_sorted = sorted(
            units,
            key=lambda x: x[
                "index"
            ],
        )

        for unit in units_sorted:

            keep = verdicts[
                unit["id"]
            ]

            if keep:

                filtered.append(
                    copy.deepcopy(
                        unit["value"]
                    )
                )

                keep_count += 1

            else:

                drop_count += 1

        return (
            filtered,
            keep_count,
            drop_count,
        )

    # -------------------------------------------------------------------------
    # scalar
    # -------------------------------------------------------------------------

    else:

        if len(units) != 1:

            raise RuntimeError(
                "Unexpected scalar "
                "Counterfactual units."
            )

        keep = verdicts[
            units[0]["id"]
        ]

        if keep:

            return (
                copy.deepcopy(
                    original_counterfactual
                ),
                1,
                0,
            )

        return (
            None,
            0,
            1,
        )


# =============================================================================
# Prompt formatting
# =============================================================================

def serialize_value(
    value,
):

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    )


def build_prompt(
    driving,
    counterfactual_units,
):

    if has_content(
        driving
    ):

        driving_text = (
            serialize_value(
                driving
            )
        )

    else:

        driving_text = (
            "NOT_PRESENT"
        )

    # -------------------------------------------------------------------------
    # Counterfactual units 출력
    # -------------------------------------------------------------------------

    cf_blocks = []

    for unit in (
        counterfactual_units
    ):

        block = (
            f'ID: {unit["id"]}\n'
            f'FIELD: {unit["label"]}\n'
            f'CONTENT:\n'
            f'{serialize_value(unit["value"])}'
        )

        cf_blocks.append(
            block
        )

    if cf_blocks:

        cf_text = (
            "\n\n"
            "----------------------------------------"
            "\n\n"
        ).join(
            cf_blocks
        )

    else:

        cf_text = (
            "NOT_PRESENT"
        )

    # -------------------------------------------------------------------------
    # Counterfactual output example 생성
    # -------------------------------------------------------------------------

    if counterfactual_units:

        example_cf = {
            unit["id"]: "KEEP"
            for unit in (
                counterfactual_units
            )
        }

    else:

        example_cf = {}

    example_output = {
        "Driving": (
            "KEEP"
            if has_content(
                driving
            )
            else "DROP"
        ),
        "Counterfactual": (
            example_cf
        ),
    }

    example_output_text = (
        json.dumps(
            example_output,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    prompt = f"""
You are filtering an autonomous-driving visual reasoning dataset.

You receive exactly THREE synchronized CURRENT-FRAME camera images:

IMAGE 1 = FRONT
IMAGE 2 = FRONT_LEFT
IMAGE 3 = FRONT_RIGHT

No other visual information is available.

There are:
- NO left-side camera
- NO right-side camera
- NO rear camera
- NO rear-left camera
- NO rear-right camera
- NO historical frames


==================================================
YOUR TASK
==================================================

Judge VISUAL OBSERVABILITY only.

Do NOT solve the driving task.
Do NOT rewrite the reasoning.
Do NOT correct the reasoning.

VERY IMPORTANT:

The reasoning text is NOT visual evidence.

You are allowed to read the reasoning only to identify WHICH visual facts,
objects, agents, road features, and spatial relationships the reasoning depends on.

You must then verify whether those required facts are actually observable
from the supplied FRONT, FRONT_LEFT, and FRONT_RIGHT images.

Never assume that an object exists simply because the reasoning text says it exists.


==================================================
STRICT DRIVING EVALUATION
==================================================

Driving is evaluated as ONE COMPLETE BLOCK.

KEEP the Driving block ONLY when every essential visual fact used by the reasoning
is clearly and directly observable in the supplied images.

The supplied images are:

- FRONT
- FRONT_LEFT
- FRONT_RIGHT

Do not assume anything outside these images.

Return KEEP only if:

1. Every essential object or agent is clearly visible.
2. Every essential road feature is clearly visible.
3. Every essential spatial relationship can be directly verified.
4. The reasoning can be supported without guessing about unseen areas.

Return DROP if:

- any essential object is not clearly visible
- any essential object is only partially visible or heavily occluded
- an essential fact depends on an unseen side or rear region
- the reasoning requires guessing or inference beyond what is directly shown
- only part of the reasoning is supported
- you are uncertain whether the required evidence is visible

IMPORTANT:

When uncertain, return DROP.

Do not return KEEP merely because the driving decision sounds reasonable.

Do not infer that an object exists because the reasoning text mentions it.

KEEP = clearly and directly verifiable from the supplied images

DROP = ambiguous, partially supported, inferred, occluded, or outside the supplied views


==================================================
COUNTERFACTUAL
==================================================

Counterfactual is NOT evaluated as one complete block.

Each Counterfactual item has a unique ID such as:

CF_000
CF_001
CF_002

Evaluate EACH ID independently.

One Counterfactual item may be KEEP while another may be DROP.

Do NOT let one invisible Counterfactual item cause all other
Counterfactual items to be dropped.

For each Counterfactual item:

Return KEEP only if ALL essential visual evidence required by THAT SPECIFIC ITEM
is clearly and directly visible in at least one of:

- FRONT
- FRONT_LEFT
- FRONT_RIGHT

Return DROP if ANY essential visual evidence is:

- not clearly visible
- ambiguous
- heavily occluded
- only partially supported
- outside the supplied field of view
- inferred rather than directly observable

Do NOT infer invisible objects from the Counterfactual text.

Do NOT return KEEP merely because the Counterfactual action sounds plausible.

When uncertain, return DROP.

KEEP = clearly and directly verifiable from the supplied images

DROP = ambiguous, partially supported, inferred, occluded,
or outside the supplied views


==================================================
GENERAL VISUAL RULES
==================================================

An object does not need to appear in all three cameras.

If it is clearly visible in at least one supplied camera,
it is considered visually available.

However, merely being theoretically inside the front hemisphere is not enough.

The required object or fact must actually be observable in the supplied images.


==================================================
NUMERICAL VALUES
==================================================

Exact numerical values such as:

- exact distance
- exact velocity
- TTC
- metric coordinates

do NOT need to be exactly measurable from pixels.

However, the underlying object or interaction that the numerical reasoning refers to
MUST be visually observable.

Example:

If reasoning says:

"The front vehicle is approximately 12 meters away"

you do not need to verify exactly 12 meters.

But the front vehicle itself must actually be visible.

If the vehicle is not visible, DROP.


==================================================
IMPORTANT OUTPUT RULES
==================================================

Do not provide explanations.
Do not provide confidence.
Do not provide markdown.
Do not return additional keys.

Allowed verdict strings are ONLY:

"KEEP"
"DROP"


==================================================
REQUIRED OUTPUT
==================================================

Return exactly one JSON object following this structure:

{example_output_text}

You MUST return a verdict for EVERY Counterfactual ID shown below.


==================================================
DRIVING CONTENT
==================================================

{driving_text}


==================================================
COUNTERFACTUAL ITEMS
==================================================

{cf_text}


Return JSON only.
""".strip()
    
    return prompt


# =============================================================================
# Qwen inference
# =============================================================================

def run_qwen(
    processor,
    model,
    images,
    prompt,
    num_counterfactual_units,
):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "CAMERA: FRONT",
                },
                {
                    "type": "image",
                    "image": images[0],
                },

                {
                    "type": "text",
                    "text": "CAMERA: FRONT_LEFT",
                },
                {
                    "type": "image",
                    "image": images[1],
                },

                {
                    "type": "text",
                    "text": "CAMERA: FRONT_RIGHT",
                },
                {
                    "type": "image",
                    "image": images[2],
                },

                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    inputs = (
        processor.apply_chat_template(
            messages,

            add_generation_prompt=True,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",
        )
    )

    device = (
        next(
            model.parameters()
        ).device
    )

    for key, value in list(
        inputs.items()
    ):

        if torch.is_tensor(
            value
        ):

            inputs[
                key
            ] = value.to(
                device
            )

    # -------------------------------------------------------------------------
    # 출력은 KEEP/DROP만 있으므로 매우 짧다.
    #
    # Counterfactual 항목 수가 많아도 JSON이 잘리지 않게
    # 항목 수 기반으로 여유 있게 설정한다.
    # -------------------------------------------------------------------------

    max_new_tokens = max(
        64,
        20
        + num_counterfactual_units * 12,
    )

    max_new_tokens = min(
        max_new_tokens,
        512,
    )

    with torch.inference_mode():

        generated = (
            model.generate(
                **inputs,

                max_new_tokens=(
                    max_new_tokens
                ),

                do_sample=False,

                use_cache=True,
            )
        )

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    generated_only = (
        generated[
            0,
            input_length:
        ]
    )

    output_text = (
        processor.decode(
            generated_only,

            skip_special_tokens=True,
        )
        .strip()
    )

    return output_text


# =============================================================================
# Qwen output parser
# =============================================================================

def normalize_keep_drop(
    value,
):

    value = str(
        value
    ).strip().upper()

    if value == "KEEP":
        return True

    if value == "DROP":
        return False

    raise ValueError(
        f"Invalid KEEP/DROP value: "
        f"{value}"
    )


def clean_json_text(
    text,
):

    cleaned = (
        text.strip()
    )

    cleaned = re.sub(
        r"^\s*```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^\s*```\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    )

    return cleaned.strip()


def parse_qwen_output(
    raw_text,
    counterfactual_units,
):

    cleaned = (
        clean_json_text(
            raw_text
        )
    )

    # -------------------------------------------------------------------------
    # 첫 번째 방법:
    # 정상 JSON parse
    # -------------------------------------------------------------------------

    parsed = None

    try:

        parsed = json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        # 혹시 앞뒤에 잡문이 있으면
        # 첫 { ~ 마지막 } 추출
        start = cleaned.find(
            "{"
        )

        end = cleaned.rfind(
            "}"
        )

        if (
            start >= 0
            and end > start
        ):

            candidate = (
                cleaned[
                    start:
                    end + 1
                ]
            )

            try:

                parsed = json.loads(
                    candidate
                )

            except json.JSONDecodeError:

                parsed = None

    if not isinstance(
        parsed,
        dict,
    ):

        raise ValueError(
            "Qwen output is not a valid JSON object."
        )

    # -------------------------------------------------------------------------
    # Driving
    # -------------------------------------------------------------------------

    if "Driving" not in parsed:

        raise ValueError(
            "Qwen output missing Driving."
        )

    driving_keep = (
        normalize_keep_drop(
            parsed[
                "Driving"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Counterfactual
    # -------------------------------------------------------------------------

    parsed_cf = (
        parsed.get(
            "Counterfactual"
        )
    )

    if not isinstance(
        parsed_cf,
        dict,
    ):

        raise ValueError(
            "Qwen output Counterfactual "
            "must be a dictionary."
        )

    required_ids = [
        unit["id"]
        for unit in (
            counterfactual_units
        )
    ]

    cf_verdicts = {}

    missing_ids = []

    for unit_id in required_ids:

        if unit_id not in parsed_cf:

            missing_ids.append(
                unit_id
            )

            continue

        cf_verdicts[
            unit_id
        ] = normalize_keep_drop(
            parsed_cf[
                unit_id
            ]
        )

    # -------------------------------------------------------------------------
    # 매우 중요:
    #
    # 일부 ID가 빠진 상태로 조용히 DROP 처리하면 안 된다.
    # 데이터가 잘못 삭제될 수 있기 때문에 parse failure로 처리한다.
    # -------------------------------------------------------------------------

    if missing_ids:

        raise ValueError(
            "Qwen output missing "
            f"Counterfactual IDs: "
            f"{missing_ids}"
        )

    return (
        driving_keep,
        cf_verdicts,
    )


# =============================================================================
# Output path
# =============================================================================

def make_output_path(
    json_path,
    data_root,
    output_root,
):

    relative_path = (
        json_path.relative_to(
            data_root
        )
    )

    return (
        output_root
        / relative_path
    )


# =============================================================================
# Single JSON
# =============================================================================

def process_one_json(
    json_path,
    data_root,
    output_root,
    processor,
    model,
    resume,
):

    relative_path = (
        json_path.relative_to(
            data_root
        )
    )

    output_path = (
        make_output_path(
            json_path,
            data_root,
            output_root,
        )
    )

    # -------------------------------------------------------------------------
    # Read original
    # -------------------------------------------------------------------------

    try:

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(
                f
            )

    except Exception as e:

        print()
        print(
            f"[READ ERROR] "
            f"{relative_path}"
        )

        print(
            f"    "
            f"{type(e).__name__}: "
            f"{e}"
        )

        return {
            "status": "read_error",
        }

    driving = (
        data.get(
            "Driving"
        )
    )

    counterfactual = (
        data.get(
            "Counterfactual"
        )
    )

    has_driving = (
        has_content(
            driving
        )
    )

    has_cf = (
        has_content(
            counterfactual
        )
    )

    # -------------------------------------------------------------------------
    # 둘 다 없으면 Qwen 호출 X
    # -------------------------------------------------------------------------

    if (
        not has_driving
        and not has_cf
    ):

        return {
            "status": "no_target",
        }

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    if (
        resume
        and output_path.exists()
    ):

        print(
            f"[SKIP RESUME] "
            f"{relative_path}"
        )

        return {
            "status": "exists",
        }

    # -------------------------------------------------------------------------
    # Counterfactual 내부 unit 생성
    # -------------------------------------------------------------------------

    if has_cf:

        cf_units = (
            build_counterfactual_units(
                counterfactual
            )
        )

    else:

        cf_units = []

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------

    try:

        image_paths = (
            extract_three_camera_paths(
                data=data,
                json_path=json_path,
            )
        )

        images = (
            load_three_images(
                image_paths
            )
        )

    except Exception as e:

        print()
        print(
            f"[IMAGE ERROR] "
            f"{relative_path}"
        )

        print(
            f"    "
            f"{type(e).__name__}: "
            f"{e}"
        )

        return {
            "status": "image_error",
        }

    # -------------------------------------------------------------------------
    # Prompt
    # -------------------------------------------------------------------------

    prompt = (
        build_prompt(
            driving=driving,
            counterfactual_units=(
                cf_units
            ),
        )
    )

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Qwen
    # -------------------------------------------------------------------------

    try:

        raw_output = (
            run_qwen(
                processor=processor,
                model=model,
                images=images,
                prompt=prompt,
                num_counterfactual_units=(
                    len(
                        cf_units
                    )
                ),
            )
        )

    except torch.cuda.OutOfMemoryError:

        print()
        print(
            f"[CUDA OOM] "
            f"{relative_path}"
        )

        torch.cuda.empty_cache()

        return {
            "status": "oom",
        }

    except Exception as e:

        print()
        print(
            f"[INFERENCE ERROR] "
            f"{relative_path}"
        )

        print(
            f"    "
            f"{type(e).__name__}: "
            f"{e}"
        )

        return {
            "status": (
                "inference_error"
            ),
        }

    elapsed = (
        time.time()
        - start_time
    )

    # -------------------------------------------------------------------------
    # Parse
    # -------------------------------------------------------------------------

    try:

        (
            qwen_driving_keep,
            cf_verdicts,
        ) = parse_qwen_output(
            raw_text=raw_output,
            counterfactual_units=(
                cf_units
            ),
        )

    except Exception as e:

        print()
        print(
            f"[PARSE ERROR] "
            f"{relative_path}"
        )

        print()
        print(
            "----- RAW QWEN OUTPUT -----"
        )

        print(
            raw_output
        )

        print(
            "---------------------------"
        )

        print(
            f"{type(e).__name__}: "
            f"{e}"
        )

        return {
            "status": "parse_error",
        }

    # -------------------------------------------------------------------------
    # Driving
    # -------------------------------------------------------------------------

    if has_driving:

        driving_keep = (
            qwen_driving_keep
        )

    else:

        driving_keep = False

    # -------------------------------------------------------------------------
    # Counterfactual
    # -------------------------------------------------------------------------

    if has_cf:

        (
            filtered_cf,
            cf_keep_count,
            cf_drop_count,
        ) = filter_counterfactual(
            original_counterfactual=(
                counterfactual
            ),
            units=cf_units,
            verdicts=cf_verdicts,
        )

        cf_has_remaining = (
            has_content(
                filtered_cf
            )
        )

    else:

        filtered_cf = None

        cf_keep_count = 0
        cf_drop_count = 0

        cf_has_remaining = False

    # =========================================================================
    # Final output
    #
    # 절대 Qwen metadata 추가하지 않는다.
    # =========================================================================

    output_data = {}

    if "frame_index" in data:

        output_data[
            "frame_index"
        ] = data[
            "frame_index"
        ]

    # -------------------------------------------------------------------------
    # Driving
    # -------------------------------------------------------------------------

    if (
        has_driving
        and driving_keep
    ):

        output_data[
            "Driving"
        ] = copy.deepcopy(
            driving
        )

    # -------------------------------------------------------------------------
    # Counterfactual
    #
    # 일부 항목만 남더라도 Counterfactual의 원래 container 구조 유지.
    # -------------------------------------------------------------------------

    if (
        has_cf
        and cf_has_remaining
    ):

        output_data[
            "Counterfactual"
        ] = filtered_cf

    # -------------------------------------------------------------------------
    # 둘 다 결과 없음
    # -------------------------------------------------------------------------

    has_output_driving = (
        "Driving"
        in output_data
    )

    has_output_cf = (
        "Counterfactual"
        in output_data
    )

    if (
        not has_output_driving
        and not has_output_cf
    ):

        # 이전 실행 결과가 있다면 삭제
        if output_path.exists():

            output_path.unlink()

        print()
        print(
            f"[DROP FILE] "
            f"{relative_path}"
        )

        print(
            f"    Driving        : "
            f"{'DROP' if has_driving else 'NOT_PRESENT'}"
        )

        print(
            f"    CF items KEEP  : "
            f"{cf_keep_count}"
        )

        print(
            f"    CF items DROP  : "
            f"{cf_drop_count}"
        )

        print(
            f"    raw            : "
            f"{raw_output}"
        )

        print(
            f"    time           : "
            f"{elapsed:.2f}s"
        )

        del images

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return {
            "status": "dropped_file",

            "driving_present": (
                int(
                    has_driving
                )
            ),

            "driving_keep": 0,

            "driving_drop": (
                int(
                    has_driving
                )
            ),

            "cf_present": (
                int(
                    has_cf
                )
            ),

            "cf_items_total": (
                len(
                    cf_units
                )
            ),

            "cf_items_keep": (
                cf_keep_count
            ),

            "cf_items_drop": (
                cf_drop_count
            ),

            "output_saved": 0,
        }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output_data,
                f,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as e:

        print()
        print(
            f"[WRITE ERROR] "
            f"{relative_path}"
        )

        print(
            f"    "
            f"{type(e).__name__}: "
            f"{e}"
        )

        return {
            "status": "write_error",
        }

    # -------------------------------------------------------------------------
    # VRAM
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():

        peak_vram = (
            torch.cuda
            .max_memory_allocated()
            / (
                1024 ** 3
            )
        )

    else:

        peak_vram = 0.0

    # -------------------------------------------------------------------------
    # Terminal log
    # -------------------------------------------------------------------------

    print()
    print(
        f"[QWEN RESULT] "
        f"{relative_path}"
    )

    print(
        f"    Driving:"
    )

    if has_driving:

        print(
            f"      "
            f"{'KEEP' if driving_keep else 'DROP'}"
        )

    else:

        print(
            "      NOT_PRESENT"
        )

    print(
        f"    Counterfactual:"
    )

    print(
        f"      total items = "
        f"{len(cf_units)}"
    )

    print(
        f"      KEEP        = "
        f"{cf_keep_count}"
    )

    print(
        f"      DROP        = "
        f"{cf_drop_count}"
    )

    # 각 CF 항목 판정도 터미널에서 확인
    for unit in cf_units:

        unit_keep = (
            cf_verdicts[
                unit["id"]
            ]
        )

        print(
            f"      {unit['id']} "
            f"{unit['label']} "
            f"-> "
            f"{'KEEP' if unit_keep else 'DROP'}"
        )

    print(
        f"    raw Qwen      = "
        f"{raw_output}"
    )

    print(
        f"    time          = "
        f"{elapsed:.2f}s"
    )

    print(
        f"    peak VRAM     = "
        f"{peak_vram:.2f} GiB"
    )

    print(
        f"    saved         = "
        f"{output_path}"
    )

    del images

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return {
        "status": "processed",

        "driving_present": (
            int(
                has_driving
            )
        ),

        "driving_keep": (
            int(
                has_driving
                and driving_keep
            )
        ),

        "driving_drop": (
            int(
                has_driving
                and not driving_keep
            )
        ),

        "cf_present": (
            int(
                has_cf
            )
        ),

        "cf_items_total": (
            len(
                cf_units
            )
        ),

        "cf_items_keep": (
            cf_keep_count
        ),

        "cf_items_drop": (
            cf_drop_count
        ),

        "output_saved": 1,
    }


# =============================================================================
# Ratio helper
# =============================================================================

def percentage(
    numerator,
    denominator,
):

    if denominator == 0:

        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    model_dir = (
        args.model_dir
        .expanduser()
        .resolve()
    )

    print()
    print("=" * 100)
    print(
        "nuReasoning Driving / Counterfactual "
        "Qwen3-VL Filter"
    )
    print("=" * 100)

    print(
        f"DATA ROOT   : "
        f"{data_root}"
    )

    print(
        f"OUTPUT ROOT : "
        f"{output_root}"
    )

    print(
        f"MODEL DIR   : "
        f"{model_dir}"
    )

    print(
        f"CAMERAS     : "
        f"{ALLOWED_CAMERAS}"
    )

    print(
        f"IMAGE SIZE  : "
        f"{IMAGE_WIDTH}x"
        f"{IMAGE_HEIGHT}"
    )

    print(
        f"HISTORY     : 0"
    )

    print(
        f"LIMIT       : "
        f"{args.limit}"
    )

    print(
        f"RESUME      : "
        f"{args.resume}"
    )

    print("=" * 100)

    if not data_root.exists():

        raise FileNotFoundError(
            f"Data root not found:\n"
            f"{data_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model_dir = (
        ensure_model(
            model_dir=model_dir,
            allow_download=(
                not args.no_download
            ),
        )
    )

    processor, model = (
        load_qwen(
            model_dir
        )
    )

    # -------------------------------------------------------------------------
    # Original reasoning JSON
    # -------------------------------------------------------------------------

    json_files = sorted(
        data_root.glob(
            "**/reasoning/*.json"
        )
    )

    print()
    print(
        f"[DATA] Total reasoning JSON: "
        f"{len(json_files)}"
    )

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    error_stats = Counter()

    qwen_runs = 0

    target_jsons = 0

    output_files = 0

    driving_present_total = 0
    driving_keep_total = 0
    driving_drop_total = 0

    cf_present_total = 0

    cf_items_total = 0
    cf_items_keep_total = 0
    cf_items_drop_total = 0

    # -------------------------------------------------------------------------
    # Loop
    # -------------------------------------------------------------------------

    for json_path in json_files:

        # 실제 Qwen run 기준 limit
        if (
            args.limit > 0
            and qwen_runs
            >= args.limit
        ):

            break

        # ---------------------------------------------------------------------
        # 먼저 빠르게 target 여부 확인
        # ---------------------------------------------------------------------

        try:

            with json_path.open(
                "r",
                encoding="utf-8",
            ) as f:

                quick_data = (
                    json.load(
                        f
                    )
                )

        except Exception:

            error_stats[
                "read_error"
            ] += 1

            continue

        quick_has_driving = (
            has_content(
                quick_data.get(
                    "Driving"
                )
            )
        )

        quick_has_cf = (
            has_content(
                quick_data.get(
                    "Counterfactual"
                )
            )
        )

        if (
            not quick_has_driving
            and not quick_has_cf
        ):

            error_stats[
                "no_target"
            ] += 1

            continue

        target_jsons += 1

        # ---------------------------------------------------------------------
        # resume
        # ---------------------------------------------------------------------

        output_path = (
            make_output_path(
                json_path=(
                    json_path
                ),
                data_root=(
                    data_root
                ),
                output_root=(
                    output_root
                ),
            )
        )

        if (
            args.resume
            and output_path.exists()
        ):

            error_stats[
                "resume_skip"
            ] += 1

            continue

        # ---------------------------------------------------------------------
        # 실제 Qwen run
        # ---------------------------------------------------------------------

        result = (
            process_one_json(
                json_path=(
                    json_path
                ),
                data_root=(
                    data_root
                ),
                output_root=(
                    output_root
                ),
                processor=(
                    processor
                ),
                model=model,
                resume=(
                    args.resume
                ),
            )
        )

        status = (
            result.get(
                "status"
            )
        )

        # ---------------------------------------------------------------------
        # inference 시도된 경우
        # ---------------------------------------------------------------------

        if status not in {
            "read_error",
            "image_error",
            "no_target",
            "exists",
        }:

            qwen_runs += 1

        # ---------------------------------------------------------------------
        # 성공적으로 verdict가 나온 경우만
        # 비율 통계에 포함
        # ---------------------------------------------------------------------

        if status in {
            "processed",
            "dropped_file",
        }:

            driving_present_total += (
                result.get(
                    "driving_present",
                    0,
                )
            )

            driving_keep_total += (
                result.get(
                    "driving_keep",
                    0,
                )
            )

            driving_drop_total += (
                result.get(
                    "driving_drop",
                    0,
                )
            )

            cf_present_total += (
                result.get(
                    "cf_present",
                    0,
                )
            )

            cf_items_total += (
                result.get(
                    "cf_items_total",
                    0,
                )
            )

            cf_items_keep_total += (
                result.get(
                    "cf_items_keep",
                    0,
                )
            )

            cf_items_drop_total += (
                result.get(
                    "cf_items_drop",
                    0,
                )
            )

            output_files += (
                result.get(
                    "output_saved",
                    0,
                )
            )

        else:

            error_stats[
                status
            ] += 1

        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        if (
            qwen_runs > 0
            and qwen_runs % 10 == 0
        ):

            print()
            print(
                "=" * 80
            )

            print(
                f"[PROGRESS] "
                f"Qwen runs: "
                f"{qwen_runs}"
            )

            print(
                f"Driving KEEP/DROP: "
                f"{driving_keep_total}/"
                f"{driving_drop_total}"
            )

            print(
                f"CF item KEEP/DROP: "
                f"{cf_items_keep_total}/"
                f"{cf_items_drop_total}"
            )

            print(
                "=" * 80
            )

    # =========================================================================
    # Final summary
    # =========================================================================

    driving_keep_ratio = (
        percentage(
            driving_keep_total,
            driving_present_total,
        )
    )

    driving_drop_ratio = (
        percentage(
            driving_drop_total,
            driving_present_total,
        )
    )

    cf_keep_ratio = (
        percentage(
            cf_items_keep_total,
            cf_items_total,
        )
    )

    cf_drop_ratio = (
        percentage(
            cf_items_drop_total,
            cf_items_total,
        )
    )

    print()
    print()
    print("=" * 100)
    print("FILTER SUMMARY")
    print("=" * 100)

    print(
        f"Qwen inference JSONs        : "
        f"{qwen_runs}"
    )

    print(
        f"Target JSONs encountered    : "
        f"{target_jsons}"
    )

    print()

    print(
        "-------------------- "
        "DRIVING "
        "--------------------"
    )

    print(
        f"Driving present             : "
        f"{driving_present_total}"
    )

    print(
        f"Driving KEEP                : "
        f"{driving_keep_total}"
    )

    print(
        f"Driving DROP                : "
        f"{driving_drop_total}"
    )

    print(
        f"Driving KEEP ratio          : "
        f"{driving_keep_ratio:.2f}%"
    )

    print(
        f"Driving DROP ratio          : "
        f"{driving_drop_ratio:.2f}%"
    )

    print()

    print(
        "--------------- "
        "COUNTERFACTUAL "
        "---------------"
    )

    print(
        f"Counterfactual JSONs        : "
        f"{cf_present_total}"
    )

    print(
        f"Counterfactual items total  : "
        f"{cf_items_total}"
    )

    print(
        f"Counterfactual items KEEP   : "
        f"{cf_items_keep_total}"
    )

    print(
        f"Counterfactual items DROP   : "
        f"{cf_items_drop_total}"
    )

    print(
        f"Counterfactual KEEP ratio   : "
        f"{cf_keep_ratio:.2f}%"
    )

    print(
        f"Counterfactual DROP ratio   : "
        f"{cf_drop_ratio:.2f}%"
    )

    print()

    print(
        "-------------------- "
        "OUTPUT "
        "---------------------"
    )

    print(
        f"Output JSON files           : "
        f"{output_files}"
    )

    print(
        f"Output root                 : "
        f"{output_root}"
    )

    print()

    print(
        "-------------------- "
        "ERRORS "
        "---------------------"
    )

    for key in [
        "no_target",
        "resume_skip",
        "read_error",
        "image_error",
        "inference_error",
        "parse_error",
        "write_error",
        "oom",
    ]:

        print(
            f"{key:28s}: "
            f"{error_stats[key]}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
