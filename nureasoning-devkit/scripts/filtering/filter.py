#!/usr/bin/env python3

import argparse
import copy
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


# =============================================================================
# =============================================================================
# ★★★ PATH SETTINGS - 서버가 바뀌면 이 3곳만 우선 확인/수정 ★★★
# =============================================================================
# =============================================================================
#
# [1] DATA_ROOT
#     part_1, part_2, ... part_10 폴더를 "직접 포함하는 상위 train 폴더"
#
#     예:
#       /media/HDD/nuR_ds/data/train/
#       ├── part_1/
#       ├── part_2/
#       ├── ...
#       └── part_10/
#
# [2] OUTPUT_ROOT
#     필터링 결과의 상위 폴더.
#     원본과 동일하게 part_1 ~ part_10 구조가 자동으로 생성됨.
#
# [3] MODEL_DIR
#     Qwen3-VL-2B-Instruct 로컬 가중치 경로
#
# =============================================================================

HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# ★ INPUT: part_1 ~ part_10을 포함하는 상위 폴더
DATA_ROOT = Path(
    "/media/HDD/nuR_ds/data/test"
)

# ★ OUTPUT: 이 아래에 part_1 ~ part_10이 그대로 생성됨
OUTPUT_ROOT = Path(
    "/home/lhh/lab/dataset/nuReasoning/"
    "test_filtered"
)

# ★ Qwen3-VL 로컬 모델 경로
MODEL_DIR = Path(
    "/home/lhh/lab/models/vlm/"
    "Qwen3-VL-2B-Instruct"
)


# =============================================================================
# CAMERA
# =============================================================================

ALLOWED_CAMERAS = {
    "front",
    "front_left",
    "front_right",
}

CAMERA_ORDER = [
    "front",
    "front_left",
    "front_right",
]

IMAGE_WIDTH = 448
IMAGE_HEIGHT = 448


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=(
            "part_1 ~ part_10 폴더를 포함하는 상위 test 폴더."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=(
            "결과 상위 폴더. part_1 ~ part_10 구조가 그대로 생성됨."
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
    )

    parser.add_argument(
        "--limit-qwen",
        type=int,
        default=0,
        help=(
            "테스트용. 실제 Qwen inference JSON 수 기준. "
            "0이면 전체."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "이미 결과 JSON이 있으면 건너뜀."
        ),
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# COMMON
# =============================================================================

def has_content(value):

    if value is None:
        return False

    if isinstance(value, str):
        return bool(
            value.strip()
        )

    if isinstance(value, dict):
        return len(value) > 0

    if isinstance(value, list):
        return len(value) > 0

    return True


def percentage(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100.0
    )


def make_output_path(
    src_path,
    data_root,
    output_root,
):

    relative_path = (
        src_path.relative_to(
            data_root
        )
    )

    return (
        output_root
        / relative_path
    )


# =============================================================================
# MODEL
# =============================================================================

def ensure_model(
    model_dir,
    allow_download,
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

        print(
            f"[MODEL] Found: "
            f"{model_dir}"
        )

        return model_dir

    if not allow_download:

        raise FileNotFoundError(
            f"Model not found:\n"
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

    return model_dir


def load_qwen(
    model_dir,
):

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda:0"
    )

    print()
    print("=" * 100)
    print("[MODEL LOAD]")
    print("=" * 100)

    print(
        f"CUDA : "
        f"{torch.cuda.get_device_name(0)}"
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            str(model_dir),

            torch_dtype=(
                torch.bfloat16
            ),

            # FlashAttention 사용 X
            attn_implementation=(
                "sdpa"
            ),

            low_cpu_mem_usage=True,

            trust_remote_code=True,
        )
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"[MODEL] device = "
        f"{next(model.parameters()).device}"
    )

    return (
        processor,
        model,
    )


# =============================================================================
# SPATIAL
# =============================================================================

def filter_per_camera_results(
    spatial,
):

    per_camera = spatial.get(
        "per_camera_results"
    )

    if not isinstance(
        per_camera,
        dict,
    ):
        return

    spatial[
        "per_camera_results"
    ] = {
        camera: value

        for camera, value
        in per_camera.items()

        if camera
        in ALLOWED_CAMERAS
    }


def filter_cross_view_correspondence(
    spatial,
):

    correspondence = spatial.get(
        "cross_view_correspondence"
    )

    if not isinstance(
        correspondence,
        dict,
    ):
        return

    new_correspondence = {}

    for track_token, item in (
        correspondence.items()
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        views = item.get(
            "views",
            [],
        )

        if not isinstance(
            views,
            list,
        ):
            views = []

        filtered_views = [
            camera

            for camera
            in views

            if camera
            in ALLOWED_CAMERAS
        ]

        observations = item.get(
            "per_view_observations",
            [],
        )

        if not isinstance(
            observations,
            list,
        ):
            observations = []

        filtered_observations = [
            obs

            for obs
            in observations

            if (
                isinstance(
                    obs,
                    dict,
                )

                and obs.get(
                    "camera"
                )
                in ALLOWED_CAMERAS
            )
        ]

        # front 3개 어디에서도
        # 관측되지 않으면 제거
        if (
            not filtered_views
            and not filtered_observations
        ):
            continue

        new_item = (
            copy.deepcopy(
                item
            )
        )

        new_item[
            "views"
        ] = filtered_views

        new_item[
            "per_view_observations"
        ] = filtered_observations

        new_item[
            "num_views"
        ] = len(
            filtered_views
        )

        new_item[
            "is_multiview"
        ] = (
            len(
                filtered_views
            )
            > 1
        )

        new_correspondence[
            track_token
        ] = new_item

    spatial[
        "cross_view_correspondence"
    ] = new_correspondence


def filter_object_relations(
    spatial,
):

    relations = spatial.get(
        "object_relations"
    )

    if not isinstance(
        relations,
        list,
    ):
        return

    new_relations = []

    for relation in relations:

        if not isinstance(
            relation,
            dict,
        ):
            continue

        observations = (
            relation.get(
                "camera_observations",
                {},
            )
        )

        if isinstance(
            observations,
            dict,
        ):

            filtered_observations = {
                camera: value

                for camera, value
                in observations.items()

                if camera
                in ALLOWED_CAMERAS
            }

        else:

            filtered_observations = {}

        if not filtered_observations:
            continue

        new_relation = (
            copy.deepcopy(
                relation
            )
        )

        new_relation[
            "camera_observations"
        ] = filtered_observations

        new_relations.append(
            new_relation
        )

    spatial[
        "object_relations"
    ] = new_relations


def get_front3_tokens(
    spatial,
):

    tokens = set()

    per_camera = spatial.get(
        "per_camera_results",
        {},
    )

    if not isinstance(
        per_camera,
        dict,
    ):
        return tokens

    for camera, camera_data in (
        per_camera.items()
    ):

        if camera not in (
            ALLOWED_CAMERAS
        ):
            continue

        if not isinstance(
            camera_data,
            dict,
        ):
            continue

        objects = camera_data.get(
            "objects",
            [],
        )

        if not isinstance(
            objects,
            list,
        ):
            continue

        for obj in objects:

            if not isinstance(
                obj,
                dict,
            ):
                continue

            token = obj.get(
                "track_token"
            )

            if token:

                tokens.add(
                    token
                )

    return tokens


def enforce_token_consistency(
    spatial,
):

    valid_tokens = (
        get_front3_tokens(
            spatial
        )
    )

    correspondence = spatial.get(
        "cross_view_correspondence"
    )

    if isinstance(
        correspondence,
        dict,
    ):

        spatial[
            "cross_view_correspondence"
        ] = {
            token: item

            for token, item
            in correspondence.items()

            if token
            in valid_tokens
        }

    relations = spatial.get(
        "object_relations"
    )

    if isinstance(
        relations,
        list,
    ):

        spatial[
            "object_relations"
        ] = [
            relation

            for relation
            in relations

            if (
                isinstance(
                    relation,
                    dict,
                )

                and relation.get(
                    "track_token"
                )
                in valid_tokens
            )
        ]


def validate_spatial(
    spatial,
):

    errors = []

    per_camera = spatial.get(
        "per_camera_results",
        {},
    )

    if isinstance(
        per_camera,
        dict,
    ):

        for camera in per_camera:

            if camera not in (
                ALLOWED_CAMERAS
            ):

                errors.append(
                    f"per_camera:"
                    f"{camera}"
                )

    correspondence = spatial.get(
        "cross_view_correspondence",
        {},
    )

    if isinstance(
        correspondence,
        dict,
    ):

        for token, item in (
            correspondence.items()
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            for camera in item.get(
                "views",
                [],
            ):

                if camera not in (
                    ALLOWED_CAMERAS
                ):

                    errors.append(
                        f"cross:"
                        f"{token}:"
                        f"{camera}"
                    )

            for obs in item.get(
                "per_view_observations",
                [],
            ):

                if not isinstance(
                    obs,
                    dict,
                ):
                    continue

                camera = obs.get(
                    "camera"
                )

                if camera not in (
                    ALLOWED_CAMERAS
                ):

                    errors.append(
                        f"cross_obs:"
                        f"{token}:"
                        f"{camera}"
                    )

    relations = spatial.get(
        "object_relations",
        [],
    )

    if isinstance(
        relations,
        list,
    ):

        for relation in relations:

            if not isinstance(
                relation,
                dict,
            ):
                continue

            observations = (
                relation.get(
                    "camera_observations",
                    {},
                )
            )

            if isinstance(
                observations,
                dict,
            ):

                for camera in (
                    observations.keys()
                ):

                    if camera not in (
                        ALLOWED_CAMERAS
                    ):

                        errors.append(
                            f"object_rel:"
                            f"{camera}"
                        )

    return errors


def build_filtered_spatial(
    original_spatial,
):

    spatial = copy.deepcopy(
        original_spatial
    )

    filter_per_camera_results(
        spatial
    )

    filter_cross_view_correspondence(
        spatial
    )

    filter_object_relations(
        spatial
    )

    enforce_token_consistency(
        spatial
    )

    errors = validate_spatial(
        spatial
    )

    if errors:

        raise RuntimeError(
            "Spatial validation failed: "
            + "; ".join(
                errors[:20]
            )
        )

    return spatial


# =============================================================================
# IMAGE
# =============================================================================

def get_clip_root(
    json_path,
):

    # CLIP/reasoning/file.json
    return (
        json_path
        .parent
        .parent
    )


def get_three_camera_paths(
    original_data,
    json_path,
):

    # 반드시 ORIGINAL Spatial에서
    # image_path를 읽는다.
    spatial = original_data.get(
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
            "per_camera_results missing."
        )

    clip_root = get_clip_root(
        json_path
    )

    result = {}

    for camera in CAMERA_ORDER:

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
                f"Missing camera: "
                f"{camera}"
            )

        relative_path = (
            camera_data.get(
                "image_path"
            )
        )

        if not relative_path:

            raise RuntimeError(
                f"Missing image_path: "
                f"{camera}"
            )

        absolute_path = (
            clip_root
            / relative_path
        )

        if not absolute_path.exists():

            raise FileNotFoundError(
                f"{camera}: "
                f"{absolute_path}"
            )

        result[
            camera
        ] = absolute_path

    return result


def load_three_images(
    paths,
):

    images = []

    for camera in CAMERA_ORDER:

        path = paths[
            camera
        ]

        with Image.open(
            path
        ) as img:

            img = img.convert(
                "RGB"
            )

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

    return images


# =============================================================================
# COUNTERFACTUAL UNIT
# =============================================================================

def build_counterfactual_units(
    counterfactual,
):

    units = []

    counter = 0

    if isinstance(
        counterfactual,
        dict,
    ):

        for key, value in (
            counterfactual.items()
        ):

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
                            "key": key,
                            "index": index,
                            "label": (
                                f"{key}[{index}]"
                            ),
                            "value": item,
                        }
                    )

                    counter += 1

            else:

                unit_id = (
                    f"CF_{counter:03d}"
                )

                units.append(
                    {
                        "id": unit_id,
                        "key": key,
                        "index": None,
                        "label": key,
                        "value": value,
                    }
                )

                counter += 1

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
                    "key": None,
                    "index": index,
                    "label": (
                        f"Counterfactual"
                        f"[{index}]"
                    ),
                    "value": item,
                }
            )

            counter += 1

    else:

        units.append(
            {
                "id": "CF_000",
                "key": None,
                "index": None,
                "label": (
                    "Counterfactual"
                ),
                "value": counterfactual,
            }
        )

    return units


def filter_counterfactual(
    original,
    units,
    verdicts,
):

    keep_count = 0
    drop_count = 0

    if isinstance(
        original,
        dict,
    ):

        result = {}

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

        for key, value in (
            original.items()
        ):

            key_units = (
                units_by_key.get(
                    key,
                    [],
                )
            )

            if isinstance(
                value,
                list,
            ):

                filtered_list = []

                key_units = sorted(
                    key_units,
                    key=lambda x: (
                        x["index"]
                    ),
                )

                for unit in key_units:

                    if verdicts[
                        unit["id"]
                    ]:

                        filtered_list.append(
                            copy.deepcopy(
                                unit[
                                    "value"
                                ]
                            )
                        )

                        keep_count += 1

                    else:

                        drop_count += 1

                if filtered_list:

                    result[
                        key
                    ] = filtered_list

            else:

                if len(
                    key_units
                ) != 1:

                    raise RuntimeError(
                        f"Unexpected CF "
                        f"units for {key}"
                    )

                unit = (
                    key_units[0]
                )

                if verdicts[
                    unit["id"]
                ]:

                    result[
                        key
                    ] = copy.deepcopy(
                        value
                    )

                    keep_count += 1

                else:

                    drop_count += 1

        return (
            result,
            keep_count,
            drop_count,
        )

    if isinstance(
        original,
        list,
    ):

        result = []

        units = sorted(
            units,
            key=lambda x: (
                x["index"]
            ),
        )

        for unit in units:

            if verdicts[
                unit["id"]
            ]:

                result.append(
                    copy.deepcopy(
                        unit[
                            "value"
                        ]
                    )
                )

                keep_count += 1

            else:

                drop_count += 1

        return (
            result,
            keep_count,
            drop_count,
        )

    keep = verdicts[
        units[0]["id"]
    ]

    if keep:

        return (
            copy.deepcopy(
                original
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
# PROMPT
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
    cf_units,
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

    cf_blocks = []

    for unit in cf_units:

        cf_blocks.append(
            f'ID: {unit["id"]}\n'
            f'FIELD: {unit["label"]}\n'
            f'CONTENT:\n'
            f'{serialize_value(unit["value"])}'
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

    example_cf = {
        unit["id"]: "KEEP"
        for unit in cf_units
    }

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
            separators=(
                ",",
                ":",
            ),
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

When uncertain, return DROP.

Do not return KEEP merely because the driving decision sounds reasonable.

Do not infer that an object exists because the reasoning text mentions it.

KEEP = clearly and directly verifiable from the supplied images

DROP = ambiguous, partially supported, inferred, occluded,
or outside the supplied views


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


==================================================
OUTPUT
==================================================

Do not provide explanations.
Do not provide confidence.
Do not provide markdown.
Do not return additional keys.

Allowed values:

"KEEP"
"DROP"

Return exactly:

{example_output_text}

You MUST return a verdict for EVERY Counterfactual ID.


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
# QWEN
# =============================================================================

def run_qwen(
    processor,
    model,
    images,
    prompt,
    num_cf_units,
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

    max_new_tokens = max(
        64,
        20
        + num_cf_units * 12,
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

    return (
        processor.decode(
            generated_only,

            skip_special_tokens=True,
        )
        .strip()
    )


# =============================================================================
# PARSER
# =============================================================================

def normalize_keep_drop(
    value,
):

    value = (
        str(
            value
        )
        .strip()
        .upper()
    )

    if value == "KEEP":
        return True

    if value == "DROP":
        return False

    raise ValueError(
        f"Invalid verdict: "
        f"{value}"
    )


def parse_qwen_output(
    raw_text,
    cf_units,
):

    cleaned = (
        raw_text.strip()
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

    parsed = None

    try:

        parsed = json.loads(
            cleaned
        )

    except json.JSONDecodeError:

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

            try:

                parsed = json.loads(
                    cleaned[
                        start:
                        end + 1
                    ]
                )

            except Exception:

                parsed = None

    if not isinstance(
        parsed,
        dict,
    ):

        raise ValueError(
            "Invalid Qwen JSON."
        )

    if "Driving" not in parsed:

        raise ValueError(
            "Driving verdict missing."
        )

    driving_keep = (
        normalize_keep_drop(
            parsed[
                "Driving"
            ]
        )
    )

    parsed_cf = parsed.get(
        "Counterfactual"
    )

    if not isinstance(
        parsed_cf,
        dict,
    ):

        raise ValueError(
            "Counterfactual verdict "
            "must be dict."
        )

    cf_verdicts = {}

    for unit in cf_units:

        unit_id = unit[
            "id"
        ]

        if unit_id not in (
            parsed_cf
        ):

            raise ValueError(
                f"Missing verdict: "
                f"{unit_id}"
            )

        cf_verdicts[
            unit_id
        ] = normalize_keep_drop(
            parsed_cf[
                unit_id
            ]
        )

    return (
        driving_keep,
        cf_verdicts,
    )


# =============================================================================
# ONE JSON
# =============================================================================

def process_one(
    json_path,
    data_root,
    output_root,
    processor,
    model,
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
    # 원본 읽기
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

        return {
            "status": "read_error",
            "error": str(e),
        }

    # -------------------------------------------------------------------------
    # Spatial
    # -------------------------------------------------------------------------

    original_spatial = (
        data.get(
            "Spatial"
        )
    )

    if not isinstance(
        original_spatial,
        dict,
    ):

        return {
            "status": "no_spatial",
        }

    try:

        filtered_spatial = (
            build_filtered_spatial(
                original_spatial
            )
        )

    except Exception as e:

        return {
            "status": (
                "spatial_error"
            ),
            "error": str(e),
        }

    # -------------------------------------------------------------------------
    # 기본 출력
    # -------------------------------------------------------------------------

    output = {}

    if "frame_index" in data:
        output["frame_index"] = data["frame_index"]

    output["Spatial"] = filtered_spatial

    # 원본 nuReasoning 형식 유지
    output["Driving"] = ""
    output["Counterfactual"] = ""

    # -------------------------------------------------------------------------
    # Driving / Counterfactual
    # -------------------------------------------------------------------------

    driving = data.get(
        "Driving"
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
    # DR/CF가 없는 파일
    # Spatial만 저장
    # -------------------------------------------------------------------------

    if (
        not has_driving
        and not has_cf
    ):

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "status": (
                "saved_spatial_only"
            ),

            "qwen_used": 0,

            "output_saved": 1,

            "driving_present": 0,
            "driving_keep": 0,
            "driving_drop": 0,

            "cf_present": 0,
            "cf_total": 0,
            "cf_keep": 0,
            "cf_drop": 0,
        }

    # -------------------------------------------------------------------------
    # 이미지
    # -------------------------------------------------------------------------

    try:

        image_paths = (
            get_three_camera_paths(
                data,
                json_path,
            )
        )

        images = (
            load_three_images(
                image_paths
            )
        )

    except Exception as e:

        return {
            "status": (
                "image_error"
            ),
            "error": str(e),
        }

    # -------------------------------------------------------------------------
    # CF units
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
    # Qwen
    # -------------------------------------------------------------------------

    prompt = build_prompt(
        driving,
        cf_units,
    )

    start_time = (
        time.time()
    )

    try:

        raw_output = (
            run_qwen(
                processor,
                model,
                images,
                prompt,
                len(
                    cf_units
                ),
            )
        )

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        return {
            "status": "oom",
            "qwen_used": 1,
        }

    except Exception as e:

        return {
            "status": (
                "inference_error"
            ),
            "error": str(e),
            "qwen_used": 1,
        }

    try:

        (
            driving_keep,
            cf_verdicts,
        ) = parse_qwen_output(
            raw_output,
            cf_units,
        )

    except Exception as e:

        return {
            "status": (
                "parse_error"
            ),
            "error": str(e),
            "raw": raw_output,
            "qwen_used": 1,
        }

    # -------------------------------------------------------------------------
    # Driving
    # -------------------------------------------------------------------------

    if (
        has_driving
        and driving_keep
    ):

        output[
            "Driving"
        ] = copy.deepcopy(
            driving
        )

    # -------------------------------------------------------------------------
    # Counterfactual
    # -------------------------------------------------------------------------

    if has_cf:

        (
            filtered_cf,
            cf_keep_count,
            cf_drop_count,
        ) = filter_counterfactual(
            counterfactual,
            cf_units,
            cf_verdicts,
        )

        if has_content(
            filtered_cf
        ):
            output["Counterfactual"] = filtered_cf

    else:

        cf_keep_count = 0
        cf_drop_count = 0

    # -------------------------------------------------------------------------
    # IMPORTANT
    #
    # Driving DROP + Counterfactual 모두 DROP이어도
    # Spatial은 있으므로 파일은 저장한다.
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
                output,
                f,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as e:

        return {
            "status": (
                "write_error"
            ),
            "error": str(e),
            "qwen_used": 1,
        }

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"[QWEN] "
        f"{relative_path}"
    )

    print(
        f"    Driving : "
        f"{'KEEP' if has_driving and driving_keep else ('DROP' if has_driving else 'NOT_PRESENT')}"
    )

    print(
        f"    CF      : "
        f"KEEP={cf_keep_count}, "
        f"DROP={cf_drop_count}"
    )

    print(
        f"    time    : "
        f"{elapsed:.2f}s"
    )

    print(
        f"    saved   : "
        f"{output_path}"
    )

    del images

    torch.cuda.empty_cache()

    return {
        "status": (
            "saved_qwen"
        ),

        "qwen_used": 1,

        "output_saved": 1,

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

        "cf_total": (
            len(
                cf_units
            )
        ),

        "cf_keep": (
            cf_keep_count
        ),

        "cf_drop": (
            cf_drop_count
        ),
    }


# =============================================================================
# PART DISCOVERY
# =============================================================================

def discover_part_dirs(
    data_root,
):
    """
    DATA_ROOT 바로 아래의 part_1 ~ part_10만 사용한다.

    중요:
    - DATA_ROOT 아래에 다른 clip 폴더나 임시 폴더가 있어도 처리하지 않는다.
    - part 번호 순서대로 part_1, part_2, ... part_10 순으로 반환한다.
    - 존재하지 않는 part는 건너뛰되 시작 시 목록을 출력한다.
    """

    part_dirs = []

    for part_index in range(
        1,
        11,
    ):

        part_dir = (
            data_root
            / f"part_{part_index}"
        )

        if part_dir.is_dir():

            part_dirs.append(
                part_dir
            )

    return part_dirs


def collect_reasoning_jsons(
    data_root,
    part_dirs,
):
    """
    각 part의 **/reasoning/*.json만 수집한다.

    반환 path는 원본 절대경로이며,
    make_output_path()가 DATA_ROOT 기준 상대경로를 사용하므로:

      input:
        DATA_ROOT/part_3/CLIP/reasoning/A.json

      output:
        OUTPUT_ROOT/part_3/CLIP/reasoning/A.json

    처럼 part 구조까지 그대로 보존된다.
    """

    json_files = []

    for part_dir in part_dirs:

        json_files.extend(
            sorted(
                part_dir.glob(
                    "**/reasoning/*.json"
                )
            )
        )

    return json_files


# =============================================================================
# MAIN
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
        "nuReasoning part_1 ~ part_10 "
        "ALL-IN-ONE FILTER"
    )
    print("=" * 100)

    print(
        f"INPUT  : "
        f"{data_root}"
    )

    print(
        f"OUTPUT : "
        f"{output_root}"
    )

    print(
        "PATH MAP: "
        "INPUT/part_N/.../reasoning/X.json "
        "-> OUTPUT/part_N/.../reasoning/X.json"
    )

    print(
        "Spatial: "
        "front/front_left/front_right"
    )

    print(
        "Driving: "
        "Qwen block KEEP/DROP"
    )

    print(
        "Counterfactual: "
        "Qwen item KEEP/DROP"
    )

    print("=" * 100)

    if not data_root.exists():

        raise FileNotFoundError(
            data_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 모델
    # -------------------------------------------------------------------------

    model_dir = (
        ensure_model(
            model_dir,
            not args.no_download,
        )
    )

    processor, model = (
        load_qwen(
            model_dir
        )
    )

    # -------------------------------------------------------------------------
    # part_1 ~ part_10 자동 탐색
    # -------------------------------------------------------------------------

    part_dirs = discover_part_dirs(
        data_root
    )

    if not part_dirs:

        raise FileNotFoundError(
            "DATA_ROOT 아래에서 part_1 ~ part_10 폴더를 하나도 찾지 못했습니다.\n"
            f"DATA_ROOT = {data_root}"
        )

    print()
    print(
        "[DATA] detected parts:"
    )

    for part_dir in part_dirs:

        print(
            f"    - {part_dir.name}"
        )

    missing_parts = [
        f"part_{i}"
        for i in range(1, 11)
        if not (
            data_root
            / f"part_{i}"
        ).is_dir()
    ]

    if missing_parts:

        print()
        print(
            "[WARNING] missing parts:"
        )

        for name in missing_parts:

            print(
                f"    - {name}"
            )

    json_files = collect_reasoning_jsons(
        data_root=data_root,
        part_dirs=part_dirs,
    )

    print()
    print(
        f"[DATA] JSON count = "
        f"{len(json_files)}"
    )

    # part별 원본 JSON 개수도 시작 전에 확인
    print(
        "[DATA] JSON count by part:"
    )

    for part_dir in part_dirs:

        part_count = sum(
            1
            for _ in part_dir.glob(
                "**/reasoning/*.json"
            )
        )

        print(
            f"    {part_dir.name:8s}: "
            f"{part_count}"
        )

    stats = Counter()

    processed = 0
    qwen_runs = 0
    output_files = 0

    driving_present = 0
    driving_keep = 0
    driving_drop = 0

    cf_jsons = 0
    cf_total = 0
    cf_keep = 0
    cf_drop = 0

    for json_path in json_files:

        # ---------------------------------------------------------
        # 테스트 limit
        # ---------------------------------------------------------

        if (
            args.limit_qwen > 0
            and qwen_runs
            >= args.limit_qwen
        ):

            print(
                f"[STOP] "
                f"limit-qwen="
                f"{args.limit_qwen}"
            )

            break

        output_path = (
            make_output_path(
                json_path,
                data_root,
                output_root,
            )
        )

        if (
            args.resume
            and output_path.exists()
        ):

            stats[
                "resume_skip"
            ] += 1

            processed += 1

            continue

        result = process_one(
            json_path,
            data_root,
            output_root,
            processor,
            model,
        )

        processed += 1

        status = result.get(
            "status"
        )

        stats[
            status
        ] += 1

        qwen_runs += (
            result.get(
                "qwen_used",
                0,
            )
        )

        output_files += (
            result.get(
                "output_saved",
                0,
            )
        )

        if status in {
            "saved_spatial_only",
            "saved_qwen",
        }:

            driving_present += (
                result.get(
                    "driving_present",
                    0,
                )
            )

            driving_keep += (
                result.get(
                    "driving_keep",
                    0,
                )
            )

            driving_drop += (
                result.get(
                    "driving_drop",
                    0,
                )
            )

            cf_jsons += (
                result.get(
                    "cf_present",
                    0,
                )
            )

            cf_total += (
                result.get(
                    "cf_total",
                    0,
                )
            )

            cf_keep += (
                result.get(
                    "cf_keep",
                    0,
                )
            )

            cf_drop += (
                result.get(
                    "cf_drop",
                    0,
                )
            )

        else:

            print()
            print(
                f"[ERROR] "
                f"{json_path.relative_to(data_root)}"
            )

            print(
                f"    status = "
                f"{status}"
            )

            if result.get(
                "error"
            ):

                print(
                    f"    error  = "
                    f"{result['error']}"
                )

            if result.get(
                "raw"
            ):

                print(
                    f"    raw    = "
                    f"{result['raw']}"
                )

        if (
            processed % 500
            == 0
        ):

            print()
            print(
                f"[PROGRESS] "
                f"{processed}/"
                f"{len(json_files)} "
                f"| output="
                f"{output_files} "
                f"| qwen="
                f"{qwen_runs}"
            )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        f"Original JSONs             : "
        f"{len(json_files)}"
    )

    print(
        f"Processed                  : "
        f"{processed}"
    )

    print(
        f"Output JSONs               : "
        f"{output_files}"
    )

    print(
        f"Spatial-only JSONs         : "
        f"{stats['saved_spatial_only']}"
    )

    print(
        f"Qwen JSONs                 : "
        f"{stats['saved_qwen']}"
    )

    print(
        f"Qwen inference runs        : "
        f"{qwen_runs}"
    )

    print()
    print(
        "---------------- DRIVING ----------------"
    )

    print(
        f"Driving present            : "
        f"{driving_present}"
    )

    print(
        f"Driving KEEP               : "
        f"{driving_keep}"
    )

    print(
        f"Driving DROP               : "
        f"{driving_drop}"
    )

    print(
        f"Driving KEEP ratio         : "
        f"{percentage(driving_keep, driving_present):.2f}%"
    )

    print(
        f"Driving DROP ratio         : "
        f"{percentage(driving_drop, driving_present):.2f}%"
    )

    print()
    print(
        "------------- COUNTERFACTUAL -------------"
    )

    print(
        f"Counterfactual JSONs       : "
        f"{cf_jsons}"
    )

    print(
        f"CF items total             : "
        f"{cf_total}"
    )

    print(
        f"CF items KEEP              : "
        f"{cf_keep}"
    )

    print(
        f"CF items DROP              : "
        f"{cf_drop}"
    )

    print(
        f"CF KEEP ratio              : "
        f"{percentage(cf_keep, cf_total):.2f}%"
    )

    print(
        f"CF DROP ratio              : "
        f"{percentage(cf_drop, cf_total):.2f}%"
    )

    print()
    print(
        "---------------- ERRORS -----------------"
    )

    for key in [
        "resume_skip",
        "read_error",
        "no_spatial",
        "spatial_error",
        "image_error",
        "inference_error",
        "parse_error",
        "write_error",
        "oom",
    ]:

        print(
            f"{key:28s}: "
            f"{stats[key]}"
        )

    print()
    print(
        f"OUTPUT ROOT: "
        f"{output_root}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
