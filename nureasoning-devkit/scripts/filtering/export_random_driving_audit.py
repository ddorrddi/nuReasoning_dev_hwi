#!/usr/bin/env python3

import json
import random
import shutil
from pathlib import Path


# =============================================================================
# 경로
# =============================================================================

ORIGINAL_ROOT = Path(
    "/media/HDD/nuR_ds/data/train/part_1"
)

HARD_AUDIT_FILE = Path(
    "/home/lhh/lab/dataset/nuReasoning/"
    "train_filtered/part_1/dr_cf_reasoning/"
    "_driving_keep_hard_audit.jsonl"
)

OUTPUT_ROOT = Path(
    "/home/lhh/lab/dataset/nuReasoning/"
    "train_filtered/tmp"
)


# =============================================================================
# 설정
# =============================================================================

NUM_SAMPLES = 50

# 재현 가능한 random sampling
RANDOM_SEED = 42

CAMERAS = [
    "front",
    "front_left",
    "front_right",
]


# =============================================================================
# Audit JSONL 읽기
# =============================================================================

def load_audit_rows():

    if not HARD_AUDIT_FILE.exists():
        raise FileNotFoundError(
            f"Audit file not found:\n"
            f"{HARD_AUDIT_FILE}"
        )

    rows = []

    with HARD_AUDIT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError as e:
                print(
                    f"[JSONL ERROR] "
                    f"line={line_number} | {e}"
                )
                continue

            rows.append(
                row
            )

    return rows


# =============================================================================
# 원본 reasoning JSON 읽기
# =============================================================================

def load_original_json(
    relative_path: str,
):

    original_path = (
        ORIGINAL_ROOT
        / relative_path
    )

    if not original_path.exists():

        raise FileNotFoundError(
            f"Original reasoning JSON not found:\n"
            f"{original_path}"
        )

    with original_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    return (
        original_path,
        data,
    )


# =============================================================================
# 이미지 경로 추출
# =============================================================================

def get_three_camera_paths(
    original_json_path: Path,
    data: dict,
):

    spatial = data.get(
        "Spatial"
    )

    if not isinstance(
        spatial,
        dict,
    ):
        raise RuntimeError(
            "Spatial missing"
        )

    per_camera = spatial.get(
        "per_camera_results"
    )

    if not isinstance(
        per_camera,
        dict,
    ):
        raise RuntimeError(
            "Spatial.per_camera_results missing"
        )

    # 구조:
    #
    # CLIP/
    # ├── cameras/
    # └── reasoning/
    #     └── timestamp.json
    #
    # 따라서 clip root:
    clip_root = (
        original_json_path
        .parent
        .parent
    )

    result = {}

    for camera in CAMERAS:

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
                f"Camera data missing: "
                f"{camera}"
            )

        relative_image_path = (
            camera_data.get(
                "image_path"
            )
        )

        if not relative_image_path:

            raise RuntimeError(
                f"image_path missing: "
                f"{camera}"
            )

        absolute_image_path = (
            clip_root
            / relative_image_path
        )

        if not absolute_image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n"
                f"camera={camera}\n"
                f"path={absolute_image_path}"
            )

        result[
            camera
        ] = absolute_image_path

    return result


# =============================================================================
# 이미지 복사
# =============================================================================

def copy_images(
    image_paths,
    sample_dir: Path,
):

    for camera in CAMERAS:

        source = (
            image_paths[
                camera
            ]
        )

        # 원본 확장자 유지
        suffix = (
            source.suffix.lower()
        )

        if not suffix:
            suffix = ".jpg"

        destination = (
            sample_dir
            / f"{camera}{suffix}"
        )

        shutil.copy2(
            source,
            destination,
        )


# =============================================================================
# Driving 저장
# =============================================================================

def save_driving(
    row,
    original_data,
    original_path,
    sample_dir,
):

    driving = row.get(
        "Driving"
    )

    # audit row에 없으면 원본에서 가져오기
    if not driving:
        driving = original_data.get(
            "Driving"
        )

    output = {
        "source_relative_path": (
            row.get(
                "relative_path"
            )
        ),

        "source_original_path": str(
            original_path
        ),

        "hard_matches": (
            row.get(
                "hard_matches",
                [],
            )
        ),

        "soft_matches": (
            row.get(
                "soft_matches",
                [],
            )
        ),

        "frame_index": (
            original_data.get(
                "frame_index"
            )
        ),

        "Driving": driving,
    }

    output_path = (
        sample_dir
        / "driving.json"
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


# =============================================================================
# Main
# =============================================================================

def main():

    print()
    print("=" * 100)
    print("Random Driving HARD Audit Export")
    print("=" * 100)

    print(
        f"Audit file : "
        f"{HARD_AUDIT_FILE}"
    )

    print(
        f"Output     : "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Samples    : "
        f"{NUM_SAMPLES}"
    )

    print(
        f"Seed       : "
        f"{RANDOM_SEED}"
    )

    print("=" * 100)

    # -------------------------------------------------------------------------
    # HARD audit rows
    # -------------------------------------------------------------------------

    rows = load_audit_rows()

    print(
        f"[AUDIT] HARD suspicious rows: "
        f"{len(rows)}"
    )

    if len(rows) == 0:

        raise RuntimeError(
            "No HARD audit samples."
        )

    num_samples = min(
        NUM_SAMPLES,
        len(rows),
    )

    # -------------------------------------------------------------------------
    # Random sampling
    # -------------------------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    selected = rng.sample(
        rows,
        num_samples,
    )

    # -------------------------------------------------------------------------
    # 기존 tmp 초기화
    #
    # 이 스크립트 전용 결과만 깔끔하게 보기 위해
    # OUTPUT_ROOT 전체를 지운다.
    # -------------------------------------------------------------------------

    if OUTPUT_ROOT.exists():

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 선택 목록도 따로 저장
    # -------------------------------------------------------------------------

    selection_manifest = []

    success_count = 0
    error_count = 0

    for index, row in enumerate(
        selected,
        start=1,
    ):

        relative_path = (
            row.get(
                "relative_path"
            )
        )

        print()
        print(
            f"[{index:02d}/{num_samples}] "
            f"{relative_path}"
        )

        try:

            # -----------------------------------------------------------------
            # 원본 JSON
            # -----------------------------------------------------------------

            (
                original_path,
                original_data,
            ) = load_original_json(
                relative_path
            )

            # -----------------------------------------------------------------
            # 이미지
            # -----------------------------------------------------------------

            image_paths = (
                get_three_camera_paths(
                    original_json_path=(
                        original_path
                    ),
                    data=(
                        original_data
                    ),
                )
            )

            # -----------------------------------------------------------------
            # sample folder
            #
            # sample_001
            # sample_002
            # ...
            # -----------------------------------------------------------------

            sample_dir = (
                OUTPUT_ROOT
                / f"sample_{index:03d}"
            )

            sample_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # -----------------------------------------------------------------
            # 이미지 3장 복사
            # -----------------------------------------------------------------

            copy_images(
                image_paths=(
                    image_paths
                ),
                sample_dir=(
                    sample_dir
                ),
            )

            # -----------------------------------------------------------------
            # Driving 저장
            # -----------------------------------------------------------------

            save_driving(
                row=row,
                original_data=(
                    original_data
                ),
                original_path=(
                    original_path
                ),
                sample_dir=(
                    sample_dir
                ),
            )

            # -----------------------------------------------------------------
            # manifest
            # -----------------------------------------------------------------

            selection_manifest.append(
                {
                    "sample": (
                        f"sample_{index:03d}"
                    ),

                    "relative_path": (
                        relative_path
                    ),

                    "hard_matches": (
                        row.get(
                            "hard_matches",
                            [],
                        )
                    ),

                    "soft_matches": (
                        row.get(
                            "soft_matches",
                            [],
                        )
                    ),
                }
            )

            success_count += 1

            print(
                f"    HARD = "
                f"{row.get('hard_matches', [])}"
            )

            print(
                f"    saved -> "
                f"{sample_dir}"
            )

        except Exception as e:

            error_count += 1

            print(
                f"    [ERROR] "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # -------------------------------------------------------------------------
    # manifest 저장
    # -------------------------------------------------------------------------

    manifest_path = (
        OUTPUT_ROOT
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selection_manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # =========================================================================
    # Summary
    # =========================================================================

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"Selected : "
        f"{num_samples}"
    )

    print(
        f"Success  : "
        f"{success_count}"
    )

    print(
        f"Errors   : "
        f"{error_count}"
    )

    print(
        f"Output   : "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Manifest : "
        f"{manifest_path}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
