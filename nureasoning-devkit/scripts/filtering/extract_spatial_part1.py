#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import Counter


DEFAULT_SOURCE_ROOT = Path(
    "/media/HDD/nuR_ds/data/train/part_1"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/home/lhh/lab/dataset/nuReasoning/train_filtered/part_1"
)


def extract_spatial_from_file(
    source_json: Path,
    output_json: Path,
) -> str:
    """
    Read one nuReasoning reasoning JSON and save only:

        {
            "frame_index": ...,
            "Spatial": ...
        }

    Returns
    -------
    str
        "saved"
        "no_spatial"
        "invalid_json"
        "error"
    """

    try:
        with source_json.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except json.JSONDecodeError as e:
        print(
            f"[INVALID JSON] {source_json}\n"
            f"    {e}"
        )
        return "invalid_json"

    except Exception as e:
        print(
            f"[READ ERROR] {source_json}\n"
            f"    {type(e).__name__}: {e}"
        )
        return "error"

    # Spatial이 없는 파일은 저장하지 않음
    if "Spatial" not in data:
        print(
            f"[NO SPATIAL] {source_json}"
        )
        return "no_spatial"

    output_data = {}

    # 원래 frame_index가 있으면 그대로 유지
    if "frame_index" in data:
        output_data["frame_index"] = data["frame_index"]

    # Spatial은 내부 구조를 수정하지 않고 그대로 저장
    output_data["Spatial"] = data["Spatial"]

    try:
        output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_json.open(
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
        print(
            f"[WRITE ERROR] {output_json}\n"
            f"    {type(e).__name__}: {e}"
        )
        return "error"

    return "saved"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract Spatial sections from "
            "nuReasoning reasoning JSON files."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=(
            "Source part root. "
            "Default: /media/HDD/nuR_ds/data/train/part_1"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output part root. "
            "Default: "
            "/home/lhh/lab/dataset/nuReasoning/"
            "train_filtered/part_1"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSON files.",
    )

    args = parser.parse_args()

    source_root: Path = args.source_root.resolve()
    output_root: Path = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source root does not exist: {source_root}"
        )

    print("=" * 80)
    print("nuReasoning Spatial Extraction")
    print("=" * 80)
    print(f"Source : {source_root}")
    print(f"Output : {output_root}")
    print(f"Overwrite: {args.overwrite}")
    print("=" * 80)

    #
    # 예상 구조:
    #
    # part_1/
    #   <clip_name>/
    #       reasoning/
    #           <timestamp>.json
    #
    # part_1 아래를 재귀적으로 탐색하므로
    # clip 구조가 조금 달라도 reasoning/*.json이면 발견됨.
    #
    json_files = sorted(
        source_root.glob("**/reasoning/*.json")
    )

    print(
        f"[FOUND] reasoning JSON files: "
        f"{len(json_files)}"
    )

    if not json_files:
        print(
            "[ERROR] No reasoning JSON files found."
        )
        return

    stats = Counter()

    for idx, source_json in enumerate(
        json_files,
        start=1,
    ):
        #
        # 예:
        #
        # source:
        # part_1/
        #   CLIP_A/
        #     reasoning/
        #       123.json
        #
        # relative:
        # CLIP_A/reasoning/123.json
        #
        relative_path = source_json.relative_to(
            source_root
        )

        output_json = (
            output_root / relative_path
        )

        if (
            output_json.exists()
            and not args.overwrite
        ):
            stats["skipped_existing"] += 1

            if (
                idx <= 10
                or idx % 500 == 0
            ):
                print(
                    f"[{idx:6d}/{len(json_files)}] "
                    f"SKIP exists: "
                    f"{relative_path}"
                )

            continue

        result = extract_spatial_from_file(
            source_json=source_json,
            output_json=output_json,
        )

        stats[result] += 1

        if (
            idx <= 10
            or idx % 100 == 0
            or result != "saved"
        ):
            print(
                f"[{idx:6d}/{len(json_files)}] "
                f"{result.upper():12s} "
                f"{relative_path}"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        f"Total discovered : "
        f"{len(json_files)}"
    )
    print(
        f"Saved            : "
        f"{stats['saved']}"
    )
    print(
        f"Skipped existing : "
        f"{stats['skipped_existing']}"
    )
    print(
        f"No Spatial       : "
        f"{stats['no_spatial']}"
    )
    print(
        f"Invalid JSON     : "
        f"{stats['invalid_json']}"
    )
    print(
        f"Other errors     : "
        f"{stats['error']}"
    )
    print()
    print(
        f"Output root: {output_root}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
