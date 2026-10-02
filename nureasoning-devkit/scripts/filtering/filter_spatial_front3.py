#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter


ROOT = Path(
    "/home/lhh/lab/dataset/nuReasoning/train_filtered/part_1"
)

ALLOWED_CAMERAS = {
    "front",
    "front_left",
    "front_right",
}


def filter_per_camera_results(spatial):
    per_camera = spatial.get("per_camera_results")

    if not isinstance(per_camera, dict):
        return

    spatial["per_camera_results"] = {
        camera: value
        for camera, value in per_camera.items()
        if camera in ALLOWED_CAMERAS
    }


def filter_cross_view_correspondence(spatial):
    correspondence = spatial.get(
        "cross_view_correspondence"
    )

    if not isinstance(correspondence, dict):
        return

    new_correspondence = {}

    for track_token, item in correspondence.items():
        if not isinstance(item, dict):
            continue

        # ----------------------------------------
        # 1. views 필터링
        # ----------------------------------------
        views = item.get("views", [])

        filtered_views = [
            camera
            for camera in views
            if camera in ALLOWED_CAMERAS
        ]

        # ----------------------------------------
        # 2. per_view_observations 필터링
        # ----------------------------------------
        observations = item.get(
            "per_view_observations",
            [],
        )

        filtered_observations = [
            obs
            for obs in observations
            if (
                isinstance(obs, dict)
                and obs.get("camera")
                in ALLOWED_CAMERAS
            )
        ]

        # 세 전방 카메라 어디에서도
        # 관측되지 않은 객체면 제거
        if (
            not filtered_views
            and not filtered_observations
        ):
            continue

        new_item = dict(item)

        new_item["views"] = filtered_views

        new_item[
            "per_view_observations"
        ] = filtered_observations

        # num_views를 현재 필터링 결과에 맞춰 갱신
        new_item["num_views"] = len(
            filtered_views
        )

        # is_multiview도 다시 계산
        new_item["is_multiview"] = (
            len(filtered_views) > 1
        )

        new_correspondence[
            track_token
        ] = new_item

    spatial[
        "cross_view_correspondence"
    ] = new_correspondence


def filter_object_relations(spatial):
    relations = spatial.get(
        "object_relations"
    )

    if not isinstance(relations, list):
        return

    new_relations = []

    for relation in relations:
        if not isinstance(relation, dict):
            continue

        camera_observations = relation.get(
            "camera_observations",
            {},
        )

        if isinstance(
            camera_observations,
            dict,
        ):
            filtered_camera_observations = {
                camera: obs
                for camera, obs
                in camera_observations.items()
                if camera in ALLOWED_CAMERAS
            }
        else:
            filtered_camera_observations = {}

        # 이 객체가 front 3-view 어디에도
        # 없으면 object_relations에서도 제거
        if not filtered_camera_observations:
            continue

        new_relation = dict(relation)

        new_relation[
            "camera_observations"
        ] = filtered_camera_observations

        new_relations.append(
            new_relation
        )

    spatial["object_relations"] = (
        new_relations
    )


def get_front3_tokens(spatial):
    """
    최종 검증용.

    per_camera_results의 front 3-view에
    실제 존재하는 모든 track_token 집합 반환.
    """

    tokens = set()

    per_camera = spatial.get(
        "per_camera_results",
        {},
    )

    if not isinstance(per_camera, dict):
        return tokens

    for camera, camera_data in (
        per_camera.items()
    ):
        if camera not in ALLOWED_CAMERAS:
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

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            token = obj.get(
                "track_token"
            )

            if token:
                tokens.add(token)

    return tokens


def enforce_token_consistency(spatial):
    """
    per_camera_results의 실제 front3 객체를
    기준으로 correspondence와 relations를
    한 번 더 정리.

    즉:
      front3 카메라에 실제 존재하지 않는 token은
      최종적으로 제거.
    """

    valid_tokens = get_front3_tokens(
        spatial
    )

    # ----------------------------------------
    # cross_view_correspondence
    # ----------------------------------------
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
            if token in valid_tokens
        }

    # ----------------------------------------
    # object_relations
    # ----------------------------------------
    relations = spatial.get(
        "object_relations"
    )

    if isinstance(relations, list):
        spatial[
            "object_relations"
        ] = [
            relation
            for relation in relations
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


def validate_no_forbidden_camera(
    spatial,
    file_path,
):
    """
    필터 후 left/right/back 계열 카메라가
    핵심 camera 참조 필드에 남아있는지 검사.
    """

    errors = []

    # ----------------------------------------
    # per_camera_results
    # ----------------------------------------
    per_camera = spatial.get(
        "per_camera_results",
        {},
    )

    if isinstance(per_camera, dict):
        for camera in per_camera:
            if camera not in ALLOWED_CAMERAS:
                errors.append(
                    f"per_camera_results:"
                    f"{camera}"
                )

    # ----------------------------------------
    # cross_view_correspondence
    # ----------------------------------------
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
                if (
                    camera
                    not in ALLOWED_CAMERAS
                ):
                    errors.append(
                        f"cross_view:"
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

                if (
                    camera
                    not in ALLOWED_CAMERAS
                ):
                    errors.append(
                        f"cross_view_obs:"
                        f"{token}:"
                        f"{camera}"
                    )

    # ----------------------------------------
    # object_relations
    # ----------------------------------------
    relations = spatial.get(
        "object_relations",
        [],
    )

    if isinstance(relations, list):
        for relation in relations:
            if not isinstance(
                relation,
                dict,
            ):
                continue

            token = relation.get(
                "track_token",
                "unknown",
            )

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
                    if (
                        camera
                        not in
                        ALLOWED_CAMERAS
                    ):
                        errors.append(
                            f"object_rel:"
                            f"{token}:"
                            f"{camera}"
                        )

    if errors:
        print(
            f"[VALIDATION FAIL] "
            f"{file_path}"
        )

        for error in errors[:20]:
            print(
                f"    {error}"
            )

        return False

    return True


def process_file(path):
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except Exception as e:
        print(
            f"[READ ERROR] "
            f"{path}: "
            f"{type(e).__name__}: {e}"
        )
        return "error"

    spatial = data.get("Spatial")

    if not isinstance(spatial, dict):
        print(
            f"[NO SPATIAL] {path}"
        )
        return "no_spatial"

    # 원본 통계
    old_cameras = set(
        spatial.get(
            "per_camera_results",
            {},
        ).keys()
    )

    old_tokens = get_front3_tokens(
        spatial
    )

    old_cross = len(
        spatial.get(
            "cross_view_correspondence",
            {},
        )
    )

    old_rel = len(
        spatial.get(
            "object_relations",
            [],
        )
    )

    # ========================================
    # 실제 필터링
    # ========================================
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

    # ========================================
    # 검증
    # ========================================
    if not validate_no_forbidden_camera(
        spatial,
        path,
    ):
        return "validation_error"

    new_cameras = set(
        spatial.get(
            "per_camera_results",
            {},
        ).keys()
    )

    new_tokens = get_front3_tokens(
        spatial
    )

    new_cross = len(
        spatial.get(
            "cross_view_correspondence",
            {},
        )
    )

    new_rel = len(
        spatial.get(
            "object_relations",
            [],
        )
    )

    # ========================================
    # 같은 파일에 덮어쓰기
    # ========================================
    try:
        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as e:
        print(
            f"[WRITE ERROR] "
            f"{path}: "
            f"{type(e).__name__}: {e}"
        )
        return "error"

    removed_cameras = (
        old_cameras - new_cameras
    )

    print(
        f"[OK] {path.name} | "
        f"cameras "
        f"{sorted(old_cameras)} "
        f"-> "
        f"{sorted(new_cameras)} | "
        f"removed="
        f"{sorted(removed_cameras)} | "
        f"front3_tokens="
        f"{len(old_tokens)}"
        f"->{len(new_tokens)} | "
        f"cross="
        f"{old_cross}->{new_cross} | "
        f"relations="
        f"{old_rel}->{new_rel}"
    )

    return "saved"


def main():
    print("=" * 90)
    print(
        "nuReasoning Spatial "
        "Front-3 Camera Filter"
    )
    print("=" * 90)
    print(f"Root: {ROOT}")
    print(
        "Allowed cameras: "
        "front, front_left, front_right"
    )
    print(
        "Mode: IN-PLACE OVERWRITE"
    )
    print("=" * 90)

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Root not found: {ROOT}"
        )

    json_files = sorted(
        ROOT.glob(
            "**/reasoning/*.json"
        )
    )

    print(
        f"Found JSON files: "
        f"{len(json_files)}"
    )
    print()

    stats = Counter()

    for idx, path in enumerate(
        json_files,
        start=1,
    ):
        result = process_file(path)
        stats[result] += 1

        if idx % 100 == 0:
            print(
                f"[PROGRESS] "
                f"{idx}/"
                f"{len(json_files)}"
            )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)

    for key in [
        "saved",
        "no_spatial",
        "validation_error",
        "error",
    ]:
        print(
            f"{key:18s}: "
            f"{stats[key]}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
