#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import Counter


DEFAULT_SRC_ROOT = Path(
    "/media/HDD/nuR_ds/data/train/part_1"
)

DEFAULT_DST_ROOT = Path(
    "/home/lhh/lab/dataset/nuReasoning/"
    "train_filtered/part_1/dr_cf_reasoning"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract Driving and Counterfactual "
            "from nuReasoning Part_1 reasoning JSONs."
        )
    )

    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help="Original Part_1 root",
    )

    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help="Output root",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    return parser.parse_args()


def has_meaningful_content(value):
    """
    Driving / Counterfactual이 실제로 내용이 있는지 판정.

    None
    {}
    []
    ""
    만 있는 경우는 비어 있다고 처리.
    """

    if value is None:
        return False

    if isinstance(value, dict):
        return len(value) > 0

    if isinstance(value, list):
        return len(value) > 0

    if isinstance(value, str):
        return bool(value.strip())

    return True


def process_file(
    src_path: Path,
    src_root: Path,
    dst_root: Path,
    overwrite: bool,
):
    try:
        with src_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except Exception as e:
        print(
            f"[READ ERROR] {src_path} | "
            f"{type(e).__name__}: {e}"
        )
        return "read_error"

    driving = data.get("Driving")
    counterfactual = data.get("Counterfactual")

    has_driving = has_meaningful_content(
        driving
    )

    has_counterfactual = has_meaningful_content(
        counterfactual
    )

    # Driving / Counterfactual 둘 다 없으면 저장하지 않음
    if not has_driving and not has_counterfactual:
        return "no_target"

    # -----------------------------------------
    # 원본 상대경로 유지
    # 예:
    # clip_x/reasoning/123.json
    # -----------------------------------------
    rel_path = src_path.relative_to(
        src_root
    )

    dst_path = dst_root / rel_path

    dst_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dst_path.exists() and not overwrite:
        print(
            f"[SKIP EXISTS] {dst_path}"
        )
        return "exists"

    # -----------------------------------------
    # 저장할 JSON 구성
    # -----------------------------------------
    output = {}

    # frame_index는 원본 프레임 연결용으로 유지
    if "frame_index" in data:
        output["frame_index"] = data[
            "frame_index"
        ]

    if has_driving:
        output["Driving"] = driving

    if has_counterfactual:
        output[
            "Counterfactual"
        ] = counterfactual

    try:
        with dst_path.open(
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
        print(
            f"[WRITE ERROR] {dst_path} | "
            f"{type(e).__name__}: {e}"
        )
        return "write_error"

    if has_driving and has_counterfactual:
        target_type = "Driving+Counterfactual"

    elif has_driving:
        target_type = "Driving"

    else:
        target_type = "Counterfactual"

    print(
        f"[SAVE] {rel_path} | "
        f"{target_type}"
    )

    return (
        "both"
        if has_driving and has_counterfactual
        else "driving_only"
        if has_driving
        else "counterfactual_only"
    )


def main():
    args = parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.expanduser().resolve()

    print("=" * 100)
    print(
        "nuReasoning Part_1 "
        "Driving / Counterfactual Extractor"
    )
    print("=" * 100)
    print(f"SRC: {src_root}")
    print(f"DST: {dst_root}")
    print(
        f"Overwrite: {args.overwrite}"
    )
    print("=" * 100)

    if not src_root.exists():
        raise FileNotFoundError(
            f"Source root not found: "
            f"{src_root}"
        )

    dst_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # reasoning 폴더 아래 JSON만 탐색
    json_files = sorted(
        src_root.glob(
            "**/reasoning/*.json"
        )
    )

    print(
        f"Found reasoning JSON files: "
        f"{len(json_files)}"
    )
    print()

    stats = Counter()

    for idx, src_path in enumerate(
        json_files,
        start=1,
    ):
        result = process_file(
            src_path=src_path,
            src_root=src_root,
            dst_root=dst_root,
            overwrite=args.overwrite,
        )

        stats[result] += 1

        if idx % 100 == 0:
            print(
                f"[PROGRESS] "
                f"{idx}/{len(json_files)}"
            )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"total_json           : "
        f"{len(json_files)}"
    )

    print(
        f"both                 : "
        f"{stats['both']}"
    )

    print(
        f"driving_only         : "
        f"{stats['driving_only']}"
    )

    print(
        f"counterfactual_only  : "
        f"{stats['counterfactual_only']}"
    )

    print(
        f"no_target            : "
        f"{stats['no_target']}"
    )

    print(
        f"exists               : "
        f"{stats['exists']}"
    )

    print(
        f"read_error           : "
        f"{stats['read_error']}"
    )

    print(
        f"write_error          : "
        f"{stats['write_error']}"
    )

    saved = (
        stats["both"]
        + stats["driving_only"]
        + stats["counterfactual_only"]
    )

    print(
        f"saved_total          : "
        f"{saved}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
