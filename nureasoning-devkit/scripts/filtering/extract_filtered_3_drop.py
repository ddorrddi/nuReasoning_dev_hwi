#!/usr/bin/env python3

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch


# =============================================================================
# ★★★ PATH SETTINGS ★★★
# =============================================================================
#
# 이 스크립트는 "Driving DROP 3개가 나왔던 진짜 첫 번째 코드"를 그대로 import해서
# 그 코드의:
#
#   - extract_three_camera_paths()
#   - load_three_images()
#   - build_counterfactual_units()
#   - build_prompt()
#   - run_qwen()
#   - parse_qwen_output()
#
# 을 그대로 사용합니다.
#
# 따라서 DROP 판정 기준을 재구현하지 않고, 당시 첫 번째 코드 그대로 재실행합니다.
#
# -----------------------------------------------------------------------------
# [1] FIRST_FILTER_SCRIPT
#     DROP 3개가 나왔던 첫 번째 .py 파일 경로로 반드시 수정
#
# [2] DATA_ROOT
#     당시 사용한 raw part_1
#
# [3] OUTPUT_ROOT
#     DROP 검사용 출력
# =============================================================================

FIRST_FILTER_SCRIPT = Path(
    "/home/lhh/lab/nuReasoning/nureasoning-devkit/scripts/filtering/filter_dr_cf_with_qwen.py"
)

DATA_ROOT = Path(
    "/media/HDD/nuR_ds/data/train/part_1"
)

OUTPUT_ROOT = Path(
    "/home/lhh/lab/nuReasoning/debug/first_filter_driving_drop"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filter-script",
        type=Path,
        default=FIRST_FILTER_SCRIPT,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )

    parser.add_argument(
        "--max-drops",
        type=int,
        default=3,
        help="몇 개의 Driving DROP을 찾으면 멈출지. 0이면 전체.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Driving이 존재하는 JSON 중 최대 몇 개를 검사할지. 0이면 전체.",
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
    )

    return parser.parse_args()


def load_first_filter(script_path: Path):
    script_path = script_path.expanduser().resolve()

    if not script_path.is_file():
        raise FileNotFoundError(
            f"첫 번째 필터 코드가 없습니다:\n{script_path}\n\n"
            "FIRST_FILTER_SCRIPT 또는 --filter-script를 수정하세요."
        )

    spec = importlib.util.spec_from_file_location(
        "first_dr_cf_filter",
        str(script_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"모듈 로드 실패: {script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "has_content",
        "ensure_model",
        "load_qwen",
        "extract_three_camera_paths",
        "load_three_images",
        "build_counterfactual_units",
        "build_prompt",
        "run_qwen",
        "parse_qwen_output",
        "ALLOWED_CAMERAS",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "첫 번째 필터 코드에서 필요한 항목이 없습니다:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    return module


def get_model_dir(module):
    if hasattr(module, "DEFAULT_MODEL_DIR"):
        return Path(module.DEFAULT_MODEL_DIR)

    if hasattr(module, "MODEL_DIR"):
        return Path(module.MODEL_DIR)

    raise RuntimeError(
        "첫 번째 코드에서 DEFAULT_MODEL_DIR/MODEL_DIR을 찾지 못했습니다."
    )


def save_drop(
    *,
    module,
    json_path,
    data_root,
    output_root,
    data,
    driving,
    counterfactual,
    image_paths,
    images,
    raw_output,
    drop_index,
):
    relative_json = json_path.relative_to(data_root)

    sample_dir = (
        output_root
        / f"drop_{drop_index:03d}"
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 원본 reasoning JSON 전체
    # -------------------------------------------------------------------------
    with (sample_dir / "original_reasoning.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Driving만 따로
    # -------------------------------------------------------------------------
    with (sample_dir / "driving.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            driving,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Counterfactual도 같이 저장
    # Qwen prompt에 함께 들어가므로 재현 확인용
    # -------------------------------------------------------------------------
    with (sample_dir / "counterfactual.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            counterfactual,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # 실제 Qwen 입력 이미지: 첫 번째 코드에서 resize된 320x320
    # -------------------------------------------------------------------------
    for camera, image in zip(
        module.ALLOWED_CAMERAS,
        images,
    ):
        image.save(
            sample_dir / f"{camera}_320x320.jpg",
            quality=95,
        )

    # -------------------------------------------------------------------------
    # 원본 이미지 3장
    # -------------------------------------------------------------------------
    originals_dir = (
        sample_dir
        / "original_images"
    )

    originals_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_records = {}

    for camera in module.ALLOWED_CAMERAS:
        src = Path(
            image_paths[camera]
        )

        suffix = (
            src.suffix
            if src.suffix
            else ".jpg"
        )

        dst = (
            originals_dir
            / f"{camera}{suffix}"
        )

        shutil.copy2(
            src,
            dst,
        )

        original_records[camera] = {
            "source": str(src),
            "saved": str(dst),
        }

    # -------------------------------------------------------------------------
    # Qwen 실제 raw verdict
    # -------------------------------------------------------------------------
    with (sample_dir / "qwen_raw_output.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(raw_output)
        f.write("\n")

    # -------------------------------------------------------------------------
    # 메타데이터
    # -------------------------------------------------------------------------
    metadata = {
        "drop_index": drop_index,
        "verdict": "DROP",
        "source_json": str(json_path),
        "relative_json": str(relative_json),
        "original_images": original_records,
    }

    with (sample_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return sample_dir


def main():
    args = parse_args()

    filter_script = (
        args.filter_script
        .expanduser()
        .resolve()
    )

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

    print()
    print("=" * 100)
    print("FIRST FILTER - DRIVING DROP REPRODUCER")
    print("=" * 100)
    print(f"FILTER SCRIPT : {filter_script}")
    print(f"DATA ROOT     : {data_root}")
    print(f"OUTPUT ROOT   : {output_root}")
    print(f"MAX DROPS     : {args.max_drops}")
    print("=" * 100)

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"DATA_ROOT 없음: {data_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    module = load_first_filter(
        filter_script
    )

    print()
    print("[OK] 첫 번째 필터 코드 import 완료")

    # =========================================================================
    # 첫 번째 코드의 동일 모델
    # =========================================================================
    model_dir = (
        get_model_dir(module)
        .expanduser()
        .resolve()
    )

    model_dir = module.ensure_model(
        model_dir=model_dir,
        allow_download=(
            not args.no_download
        ),
    )

    processor, model = module.load_qwen(
        model_dir
    )

    # =========================================================================
    # 첫 번째 코드와 동일한 JSON 검색
    # =========================================================================
    json_files = sorted(
        data_root.glob(
            "**/reasoning/*.json"
        )
    )

    print()
    print(
        f"[DATA] reasoning JSON 전체 = "
        f"{len(json_files)}"
    )

    # Driving이 실제 존재하는 JSON만 검사
    driving_files = []

    for json_path in json_files:
        try:
            with json_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

        except Exception as e:
            print(
                f"[READ ERROR] {json_path}: {e}"
            )
            continue

        if module.has_content(
            data.get("Driving")
        ):
            driving_files.append(
                json_path
            )

    print(
        f"[DATA] Driving 존재 JSON = "
        f"{len(driving_files)}"
    )

    if args.limit > 0:
        driving_files = (
            driving_files[:args.limit]
        )

        print(
            f"[LIMIT] {len(driving_files)}개만 검사"
        )

    # =========================================================================
    # 재실행
    # =========================================================================
    tested = 0
    keep_count = 0
    drop_count = 0
    error_count = 0

    manifest = []

    for idx, json_path in enumerate(
        driving_files,
        start=1,
    ):
        relative_path = (
            json_path.relative_to(
                data_root
            )
        )

        try:
            with json_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            driving = data.get(
                "Driving"
            )

            counterfactual = data.get(
                "Counterfactual"
            )

            # ================================================================
            # ★ 첫 번째 코드 그대로: 이미지 경로
            # ================================================================
            image_paths = (
                module.extract_three_camera_paths(
                    data=data,
                    json_path=json_path,
                )
            )

            # ================================================================
            # ★ 첫 번째 코드 그대로: 320x320 세 이미지
            # ================================================================
            images = (
                module.load_three_images(
                    image_paths
                )
            )

            # ================================================================
            # ★ 첫 번째 코드 그대로: Counterfactual units
            # ================================================================
            if module.has_content(
                counterfactual
            ):
                cf_units = (
                    module.build_counterfactual_units(
                        counterfactual
                    )
                )
            else:
                cf_units = []

            # ================================================================
            # ★ 첫 번째 코드 그대로: Prompt
            # ================================================================
            prompt = module.build_prompt(
                driving=driving,
                counterfactual_units=(
                    cf_units
                ),
            )

            # ================================================================
            # ★ 첫 번째 코드 그대로: Qwen
            # ================================================================
            started = time.time()

            raw_output = module.run_qwen(
                processor=processor,
                model=model,
                images=images,
                prompt=prompt,
                num_counterfactual_units=(
                    len(cf_units)
                ),
            )

            # ================================================================
            # ★ 첫 번째 코드 그대로: parser
            # ================================================================
            (
                driving_keep,
                _cf_verdicts,
            ) = module.parse_qwen_output(
                raw_text=raw_output,
                counterfactual_units=(
                    cf_units
                ),
            )

            elapsed = (
                time.time()
                - started
            )

            tested += 1

            if driving_keep:
                keep_count += 1

                print(
                    f"[{idx:4d}/{len(driving_files):4d}] "
                    f"KEEP | {relative_path} | "
                    f"{elapsed:.2f}s"
                )

            else:
                drop_count += 1

                sample_dir = save_drop(
                    module=module,
                    json_path=json_path,
                    data_root=data_root,
                    output_root=output_root,
                    data=data,
                    driving=driving,
                    counterfactual=counterfactual,
                    image_paths=image_paths,
                    images=images,
                    raw_output=raw_output,
                    drop_index=drop_count,
                )

                manifest.append(
                    {
                        "drop_index": drop_count,
                        "source_json": str(
                            json_path
                        ),
                        "relative_json": str(
                            relative_path
                        ),
                        "saved_dir": str(
                            sample_dir
                        ),
                        "raw_output": raw_output,
                    }
                )

                print()
                print("#" * 100)
                print(
                    f"[DRIVING DROP #{drop_count}]"
                )
                print(
                    f"JSON  : {relative_path}"
                )
                print(
                    f"RAW   : {raw_output}"
                )
                print(
                    f"SAVED : {sample_dir}"
                )
                print("#" * 100)
                print()

                if (
                    args.max_drops > 0
                    and drop_count
                    >= args.max_drops
                ):
                    del images
                    torch.cuda.empty_cache()
                    break

            del images
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            error_count += 1

            print(
                f"[OOM] {relative_path}"
            )

            torch.cuda.empty_cache()

        except Exception as e:
            error_count += 1

            print(
                f"[ERROR] {relative_path}"
            )
            print(
                f"        {type(e).__name__}: {e}"
            )

            torch.cuda.empty_cache()

    manifest_path = (
        output_root
        / "drop_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Driving tested : {tested}")
    print(f"KEEP           : {keep_count}")
    print(f"DROP           : {drop_count}")
    print(f"ERROR          : {error_count}")
    print(f"Manifest       : {manifest_path}")
    print(f"Output root    : {output_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()
