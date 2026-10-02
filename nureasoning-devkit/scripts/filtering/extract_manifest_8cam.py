#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

# ============================================================================
# ★ PATH SETTINGS
# ============================================================================

MANIFEST_PATH = Path(
    "/home/lhh/lab/nuReasoning/debug/"
    "first_filter_driving_drop/drop_manifest.json"
)

OUTPUT_ROOT = Path(
    "/home/lhh/lab/nuReasoning/debug/"
    "first_filter_driving_drop_8cam"
)

# nuReasoning 8-camera order
CAMERA_ORDER = [
    "front",
    "front_left",
    "front_right",
    "left",
    "right",
    "back_left",
    "back",
    "back_right",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_clip_root(json_path: Path):
    # .../CLIP/reasoning/TIMESTAMP.json -> .../CLIP
    return json_path.parent.parent


def get_camera_paths(data, json_path: Path):
    spatial = data.get("Spatial")
    if not isinstance(spatial, dict):
        raise RuntimeError("Spatial missing")

    per_camera = spatial.get("per_camera_results")
    if not isinstance(per_camera, dict):
        raise RuntimeError("Spatial.per_camera_results missing")

    clip_root = get_clip_root(json_path)
    paths = {}

    for camera, camera_data in per_camera.items():
        if not isinstance(camera_data, dict):
            continue

        rel = camera_data.get("image_path")
        if not rel:
            continue

        paths[camera] = (clip_root / rel).resolve()

    return paths


def main():
    args = parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path}"
        )

    manifest = load_json(manifest_path)

    if not isinstance(manifest, list):
        raise RuntimeError("Manifest must be a JSON list")

    output_root.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 100)
    print("DROP MANIFEST -> 8 CAMERA IMAGE EXTRACTOR")
    print(f"MANIFEST : {manifest_path}")
    print(f"OUTPUT   : {output_root}")
    print("=" * 100)

    for item in manifest:
        drop_index = int(item["drop_index"])
        source_json = Path(item["source_json"]).expanduser().resolve()

        print()
        print(f"[DROP {drop_index:03d}]")
        print(f"JSON: {source_json}")

        try:
            data = load_json(source_json)
            camera_paths = get_camera_paths(data, source_json)

            print("Available camera keys:")
            for key in camera_paths.keys():
                print(f"  - {key}")

            sample_dir = output_root / f"drop_{drop_index:03d}"
            images_dir = sample_dir / "images_8cam"
            images_dir.mkdir(parents=True, exist_ok=True)

            saved = {}
            missing = []

            for camera in CAMERA_ORDER:
                src = camera_paths.get(camera)

                if src is None or not src.is_file():
                    missing.append(camera)
                    print(f"[MISSING] {camera}")
                    continue

                suffix = src.suffix if src.suffix else ".jpg"
                dst = images_dir / f"{camera}{suffix}"

                shutil.copy2(src, dst)

                saved[camera] = {
                    "source": str(src),
                    "saved": str(dst),
                }

                print(f"[COPY] {camera:12s} -> {dst}")

            # reasoning JSON도 같이 보관
            shutil.copy2(
                source_json,
                sample_dir / "reasoning.json",
            )

            metadata = {
                "drop_index": drop_index,
                "source_json": str(source_json),
                "raw_output": item.get("raw_output"),
                "available_camera_keys": list(camera_paths.keys()),
                "saved_cameras": saved,
                "missing_cameras": missing,
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

            results.append(
                {
                    "drop_index": drop_index,
                    "saved_count": len(saved),
                    "missing": missing,
                    "output_dir": str(sample_dir),
                }
            )

        except Exception as e:
            print(
                f"[ERROR] {type(e).__name__}: {e}"
            )

            results.append(
                {
                    "drop_index": drop_index,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    summary_path = output_root / "summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for result in results:
        if "error" in result:
            print(
                f"DROP {result['drop_index']:03d}: ERROR "
                f"{result['error']}"
            )
        else:
            print(
                f"DROP {result['drop_index']:03d}: "
                f"saved={result['saved_count']} "
                f"missing={result['missing']}"
            )

    print(f"SUMMARY JSON: {summary_path}")
    print(f"OUTPUT ROOT : {output_root}")
    print("=" * 100)


if __name__ == "__main__":
    main()
