#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any


# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = Path(
    "/media/HDD/nuR_ds/data/train/part_1"
)

OUTPUT_DIR = Path(
    "/home/lhh/lab/nuReasoning/outputs/extract"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "driving_counterfactual_part1.jsonl"
)


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path):

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception as e:

        print(
            f"[WARN] JSON load failed: {path}\n"
            f"       {type(e).__name__}: {e}"
        )

        return None


def is_nonempty(value: Any) -> bool:
    """
    실제 annotation 값이 존재하는지 검사.

    None
    ""
    {}
    []

    는 모두 empty 처리.
    """

    if value is None:
        return False

    if isinstance(value, str):
        return bool(
            value.strip()
        )

    if isinstance(
        value,
        (dict, list, tuple),
    ):
        return len(value) > 0

    return True


def find_annotation_dict(
    obj: Any,
):
    """
    JSON 내부에서 Driving / Counterfactual이
    실제로 존재하는 dict를 찾음.

    단순히 KEY 존재 여부가 아니라
    값이 non-empty인지 검사한다.
    """

    if isinstance(obj, dict):

        driving = obj.get(
            "Driving"
        )

        counterfactual = obj.get(
            "Counterfactual"
        )

        if (
            is_nonempty(driving)
            or is_nonempty(counterfactual)
        ):
            return obj

        for value in obj.values():

            found = find_annotation_dict(
                value
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:

            found = find_annotation_dict(
                value
            )

            if found is not None:
                return found

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: "
            f"{DATA_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_dirs = sorted(
        [
            p
            for p in DATA_ROOT.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )

    print(
        f"[INFO] DATA_ROOT   : {DATA_ROOT}"
    )

    print(
        f"[INFO] Sample dirs : {len(sample_dirs)}"
    )

    print(
        f"[INFO] Output      : {OUTPUT_PATH}"
    )

    # =========================================================================
    # STATS
    # =========================================================================

    total_json = 0
    total_reasoning_json = 0

    extracted = 0

    both_present = 0
    driving_only = 0
    counterfactual_only = 0

    empty_json = 0
    parse_fail = 0

    samples_with_annotation = 0
    samples_without_annotation = 0

    per_sample_counts = {}


    # =========================================================================
    # OUTPUT
    # =========================================================================

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for sample_idx, sample_dir in enumerate(
            sample_dirs,
            start=1,
        ):

            # -----------------------------------------------------------------
            # reasoning 디렉터리 우선
            # -----------------------------------------------------------------

            reasoning_dir = (
                sample_dir
                / "reasoning"
            )

            if reasoning_dir.exists():

                json_paths = sorted(
                    reasoning_dir.glob(
                        "*.json"
                    )
                )

            else:

                # fallback
                json_paths = sorted(
                    sample_dir.rglob(
                        "*.json"
                    )
                )

            sample_extracted = 0

            total_json += len(
                json_paths
            )

            total_reasoning_json += len(
                json_paths
            )

            # -----------------------------------------------------------------
            # 중요:
            # 샘플 안 JSON 하나만 보는 게 아니라
            # 전부 순회한다.
            # -----------------------------------------------------------------

            for json_path in json_paths:

                data = load_json(
                    json_path
                )

                if data is None:

                    parse_fail += 1
                    continue

                target = find_annotation_dict(
                    data
                )

                if target is None:

                    empty_json += 1
                    continue

                driving = target.get(
                    "Driving"
                )

                counterfactual = target.get(
                    "Counterfactual"
                )

                has_driving = is_nonempty(
                    driving
                )

                has_counterfactual = is_nonempty(
                    counterfactual
                )

                # 둘 다 비었다면 저장하지 않음
                if (
                    not has_driving
                    and not has_counterfactual
                ):
                    empty_json += 1
                    continue

                # -------------------------------------------------------------
                # timestamp = JSON filename stem
                # -------------------------------------------------------------

                frame_id = (
                    json_path.stem
                )

                record = {
                    "sample_id": sample_dir.name,
                    "frame_id": frame_id,
                    "source_json": str(
                        json_path
                    ),
                    "Driving": (
                        driving
                        if has_driving
                        else None
                    ),
                    "Counterfactual": (
                        counterfactual
                        if has_counterfactual
                        else None
                    ),
                }

                fout.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                extracted += 1
                sample_extracted += 1

                if (
                    has_driving
                    and has_counterfactual
                ):
                    both_present += 1

                elif has_driving:
                    driving_only += 1

                else:
                    counterfactual_only += 1

            # -----------------------------------------------------------------
            # SAMPLE STATUS
            # -----------------------------------------------------------------

            per_sample_counts[
                sample_dir.name
            ] = sample_extracted

            if sample_extracted > 0:

                samples_with_annotation += 1

                print(
                    f"[{sample_idx:04d}/"
                    f"{len(sample_dirs):04d}] "
                    f"[OK] "
                    f"{sample_dir.name} "
                    f"-> {sample_extracted} "
                    f"annotation JSON(s)"
                )

            else:

                samples_without_annotation += 1

                print(
                    f"[{sample_idx:04d}/"
                    f"{len(sample_dirs):04d}] "
                    f"[NONE] "
                    f"{sample_dir.name}"
                )


    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "Extraction complete"
    )

    print(
        "=" * 80
    )

    print(
        f"Sample directories       : "
        f"{len(sample_dirs)}"
    )

    print(
        f"Samples with annotation  : "
        f"{samples_with_annotation}"
    )

    print(
        f"Samples without annotation: "
        f"{samples_without_annotation}"
    )

    print(
        f"Reasoning JSON checked   : "
        f"{total_reasoning_json}"
    )

    print(
        f"Extracted annotation sets: "
        f"{extracted}"
    )

    print(
        f"  Driving + Counterfactual: "
        f"{both_present}"
    )

    print(
        f"  Driving only            : "
        f"{driving_only}"
    )

    print(
        f"  Counterfactual only     : "
        f"{counterfactual_only}"
    )

    print(
        f"Empty/non-target JSON     : "
        f"{empty_json}"
    )

    print(
        f"Parse failures            : "
        f"{parse_fail}"
    )

    print(
        f"Output                    : "
        f"{OUTPUT_PATH}"
    )


    # =========================================================================
    # DISTRIBUTION
    # =========================================================================

    counts = list(
        per_sample_counts.values()
    )

    if counts:

        print()
        print(
            "[Per-sample annotation count]"
        )

        print(
            f"min : {min(counts)}"
        )

        print(
            f"max : {max(counts)}"
        )

        print(
            f"mean: "
            f"{sum(counts) / len(counts):.2f}"
        )


if __name__ == "__main__":
    main()