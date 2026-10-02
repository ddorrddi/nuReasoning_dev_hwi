#!/usr/bin/env python3

import json
import re
from pathlib import Path
from collections import Counter


# =============================================================================
# 경로
# =============================================================================

ORIGINAL_ROOT = Path(
    "/media/HDD/nuR_ds/data/train/part_1"
)

FILTERED_ROOT = Path(
    "/home/lhh/lab/dataset/nuReasoning/"
    "train_filtered/part_1/dr_cf_reasoning"
)


# =============================================================================
# HARD suspicious patterns
#
# 전방 3개:
#   front
#   front_left
#   front_right
#
# 만으로는 직접 관측이 어려울 가능성이 높은 표현
# =============================================================================

HARD_PATTERNS = {

    "behind": re.compile(
        r"\bbehind\b",
        re.IGNORECASE,
    ),

    "rear": re.compile(
        r"\brear\b",
        re.IGNORECASE,
    ),

    "rear_left": re.compile(
        r"\brear[\s_-]*left\b",
        re.IGNORECASE,
    ),

    "rear_right": re.compile(
        r"\brear[\s_-]*right\b",
        re.IGNORECASE,
    ),

    "from_behind": re.compile(
        r"\bfrom\s+behind\b",
        re.IGNORECASE,
    ),

    "blind_spot": re.compile(
        r"\bblind[\s_-]*spot\b",
        re.IGNORECASE,
    ),

    "following_vehicle": re.compile(
        r"\bfollowing\s+vehicle\b",
        re.IGNORECASE,
    ),

    # 조금 더 직접적인 rear 표현도 추가
    "vehicle_behind": re.compile(
        r"\bvehicle\b.{0,40}\bbehind\b"
        r"|\bbehind\b.{0,40}\bvehicle\b",
        re.IGNORECASE,
    ),

    "car_behind": re.compile(
        r"\bcar\b.{0,40}\bbehind\b"
        r"|\bbehind\b.{0,40}\bcar\b",
        re.IGNORECASE,
    ),

    "approaching_from_behind": re.compile(
        r"\bapproach(?:ing|es|ed)?\b"
        r".{0,50}"
        r"\bfrom\s+behind\b",
        re.IGNORECASE,
    ),
}


# =============================================================================
# SOFT suspicious patterns
#
# 이것만 있다고 DROP 후보로 볼 수는 없음.
#
# front_left / front_right로
# adjacent lane의 전방 영역은 보일 수 있기 때문.
# =============================================================================

SOFT_PATTERNS = {

    "left_adjacent_lane": re.compile(
        r"\bleft\s+adjacent\s+lane\b",
        re.IGNORECASE,
    ),

    "right_adjacent_lane": re.compile(
        r"\bright\s+adjacent\s+lane\b",
        re.IGNORECASE,
    ),

    "adjacent_lane": re.compile(
        r"\badjacent\s+lane\b",
        re.IGNORECASE,
    ),
}


# 전체 pattern
ALL_PATTERNS = {
    **HARD_PATTERNS,
    **SOFT_PATTERNS,
}


# =============================================================================
# Utility
# =============================================================================

def flatten_text(value):
    """
    Driving JSON 전체를 검색 가능한 문자열로 변환.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def find_matches(
    text,
    patterns,
):
    """
    주어진 pattern dictionary에서
    실제 매칭된 이름 목록 반환.
    """

    matches = []

    for name, pattern in patterns.items():

        if pattern.search(
            text
        ):
            matches.append(
                name
            )

    return matches


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


# =============================================================================
# Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # filtered JSON 검색
    #
    # Driving이 남아 있으면
    # 현재 Qwen filter에서 KEEP된 것
    # -------------------------------------------------------------------------

    filtered_files = sorted(
        FILTERED_ROOT.glob(
            "**/reasoning/*.json"
        )
    )

    print()
    print("=" * 100)
    print("Driving KEEP audit")
    print("=" * 100)

    print(
        f"Filtered JSON files        : "
        f"{len(filtered_files)}"
    )

    # -------------------------------------------------------------------------
    # counters
    # -------------------------------------------------------------------------

    driving_keep_count = 0

    suspicious_count = 0

    hard_suspicious_count = 0

    soft_suspicious_count = 0

    soft_only_count = 0

    both_hard_soft_count = 0

    hard_only_count = 0

    # pattern별 unique sample count
    all_pattern_counts = Counter()

    hard_pattern_counts = Counter()

    soft_pattern_counts = Counter()

    # 결과 row
    suspicious_rows = []

    hard_rows = []

    soft_rows = []

    soft_only_rows = []

    # -------------------------------------------------------------------------
    # loop
    # -------------------------------------------------------------------------

    for filtered_path in filtered_files:

        # ---------------------------------------------------------------------
        # filtered JSON 읽기
        # ---------------------------------------------------------------------

        try:

            with filtered_path.open(
                "r",
                encoding="utf-8",
            ) as f:

                filtered = json.load(
                    f
                )

        except Exception as e:

            print(
                f"[READ ERROR] "
                f"{filtered_path}: "
                f"{type(e).__name__}: {e}"
            )

            continue

        # ---------------------------------------------------------------------
        # Driving 존재 여부
        #
        # filtered 결과에서 Driving이 존재하면
        # Qwen이 KEEP했던 sample.
        # ---------------------------------------------------------------------

        driving = filtered.get(
            "Driving"
        )

        if not driving:
            continue

        driving_keep_count += 1

        # ---------------------------------------------------------------------
        # text flatten
        # ---------------------------------------------------------------------

        text = flatten_text(
            driving
        )

        # ---------------------------------------------------------------------
        # 모든 suspicious pattern
        # ---------------------------------------------------------------------

        all_matches = find_matches(
            text,
            ALL_PATTERNS,
        )

        hard_matches = find_matches(
            text,
            HARD_PATTERNS,
        )

        soft_matches = find_matches(
            text,
            SOFT_PATTERNS,
        )

        # 아무것도 안 걸림
        if not all_matches:
            continue

        suspicious_count += 1

        # ---------------------------------------------------------------------
        # relative / original path
        # ---------------------------------------------------------------------

        relative_path = (
            filtered_path.relative_to(
                FILTERED_ROOT
            )
        )

        original_path = (
            ORIGINAL_ROOT
            / relative_path
        )

        # ---------------------------------------------------------------------
        # pattern count
        #
        # 한 sample에 같은 단어가 여러 번 나와도
        # pattern별 sample count는 1 증가.
        # ---------------------------------------------------------------------

        for name in all_matches:

            all_pattern_counts[
                name
            ] += 1

        for name in hard_matches:

            hard_pattern_counts[
                name
            ] += 1

        for name in soft_matches:

            soft_pattern_counts[
                name
            ] += 1

        # ---------------------------------------------------------------------
        # 공통 row
        # ---------------------------------------------------------------------

        row = {
            "relative_path": str(
                relative_path
            ),

            "original_path": str(
                original_path
            ),

            "hard_matches": (
                hard_matches
            ),

            "soft_matches": (
                soft_matches
            ),

            "all_matches": (
                all_matches
            ),

            "Driving": driving,
        }

        suspicious_rows.append(
            row
        )

        # ---------------------------------------------------------------------
        # HARD
        # ---------------------------------------------------------------------

        if hard_matches:

            hard_suspicious_count += 1

            hard_rows.append(
                row
            )

        # ---------------------------------------------------------------------
        # SOFT
        # ---------------------------------------------------------------------

        if soft_matches:

            soft_suspicious_count += 1

            soft_rows.append(
                row
            )

        # ---------------------------------------------------------------------
        # hard only / soft only / both
        # ---------------------------------------------------------------------

        if (
            hard_matches
            and soft_matches
        ):

            both_hard_soft_count += 1

        elif hard_matches:

            hard_only_count += 1

        elif soft_matches:

            soft_only_count += 1

            soft_only_rows.append(
                row
            )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Driving KEEP total         : "
        f"{driving_keep_count}"
    )

    print()

    print(
        f"Any suspicious KEEP        : "
        f"{suspicious_count}"
    )

    print(
        f"Any suspicious ratio       : "
        f"{percentage(suspicious_count, driving_keep_count):.2f}%"
    )

    print()

    print(
        f"HARD suspicious KEEP       : "
        f"{hard_suspicious_count}"
    )

    print(
        f"HARD suspicious ratio      : "
        f"{percentage(hard_suspicious_count, driving_keep_count):.2f}%"
    )

    print()

    print(
        f"SOFT suspicious KEEP       : "
        f"{soft_suspicious_count}"
    )

    print(
        f"SOFT suspicious ratio      : "
        f"{percentage(soft_suspicious_count, driving_keep_count):.2f}%"
    )

    print()

    print(
        f"HARD only                  : "
        f"{hard_only_count}"
    )

    print(
        f"SOFT only                  : "
        f"{soft_only_count}"
    )

    print(
        f"HARD + SOFT                : "
        f"{both_hard_soft_count}"
    )

    # =========================================================================
    # HARD patterns
    # =========================================================================

    print()
    print("-" * 100)
    print("HARD PATTERN COUNTS")
    print("-" * 100)

    if hard_pattern_counts:

        for name, count in (
            hard_pattern_counts
            .most_common()
        ):

            print(
                f"{name:30s}: "
                f"{count}"
            )

    else:

        print(
            "No HARD patterns found."
        )

    # =========================================================================
    # SOFT patterns
    # =========================================================================

    print()
    print("-" * 100)
    print("SOFT PATTERN COUNTS")
    print("-" * 100)

    if soft_pattern_counts:

        for name, count in (
            soft_pattern_counts
            .most_common()
        ):

            print(
                f"{name:30s}: "
                f"{count}"
            )

    else:

        print(
            "No SOFT patterns found."
        )

    # =========================================================================
    # ALL pattern counts
    # =========================================================================

    print()
    print("-" * 100)
    print("ALL PATTERN COUNTS")
    print("-" * 100)

    for name, count in (
        all_pattern_counts
        .most_common()
    ):

        print(
            f"{name:30s}: "
            f"{count}"
        )

    # =========================================================================
    # JSONL 저장
    # =========================================================================

    all_output = (
        FILTERED_ROOT
        / "_driving_keep_audit.jsonl"
    )

    hard_output = (
        FILTERED_ROOT
        / "_driving_keep_hard_audit.jsonl"
    )

    soft_output = (
        FILTERED_ROOT
        / "_driving_keep_soft_audit.jsonl"
    )

    soft_only_output = (
        FILTERED_ROOT
        / "_driving_keep_soft_only_audit.jsonl"
    )

    # -------------------------------------------------------------------------
    # 전체 suspicious
    # -------------------------------------------------------------------------

    with all_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in suspicious_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -------------------------------------------------------------------------
    # HARD
    # -------------------------------------------------------------------------

    with hard_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in hard_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -------------------------------------------------------------------------
    # SOFT
    # -------------------------------------------------------------------------

    with soft_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in soft_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -------------------------------------------------------------------------
    # SOFT only
    # -------------------------------------------------------------------------

    with soft_only_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in soft_only_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 100)
    print("AUDIT FILES")
    print("=" * 100)

    print(
        f"ALL suspicious : "
        f"{all_output}"
    )

    print(
        f"HARD suspicious: "
        f"{hard_output}"
    )

    print(
        f"SOFT suspicious: "
        f"{soft_output}"
    )

    print(
        f"SOFT only      : "
        f"{soft_only_output}"
    )


    # =========================================================================
    # 최종 한 줄
    # =========================================================================

    print()
    print("=" * 100)

    print(
        f"RESULT: "
        f"{hard_suspicious_count} / "
        f"{driving_keep_count} "
        f"Driving KEEP samples contain "
        f"at least one HARD rear-related pattern "
        f"("
        f"{percentage(hard_suspicious_count, driving_keep_count):.2f}%"
        f")"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
