#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_map(path: Path, key_field: str) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row[key_field]: row for row in rows}


def compare_rows(old: dict[str, dict], new: dict[str, dict], sha_key: str, extra_keys: list[str]) -> dict[str, list]:
    result = {
        "identical": [],
        "missing_in_new": [],
        "missing_in_old": [],
        "sha256_different": [],
        "metadata_different": [],
    }
    for key in sorted(set(old) | set(new)):
        if key not in new:
            result["missing_in_new"].append(key)
            continue
        if key not in old:
            result["missing_in_old"].append(key)
            continue
        old_row = old[key]
        new_row = new[key]
        if old_row.get(sha_key) != new_row.get(sha_key):
            result["sha256_different"].append(key)
            continue
        diffs = [field for field in extra_keys if old_row.get(field) != new_row.get(field)]
        if diffs:
            result["metadata_different"].append({"key": key, "fields": diffs})
        else:
            result["identical"].append(key)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", required=True, type=Path)
    parser.add_argument("--new-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    old_env = load_json(args.old_dir / "current_environment_manifest.json")
    new_env = load_json(args.new_dir / "current_environment_manifest.json")

    old_assets = load_csv_map(args.old_dir / "current_assets_manifest.csv", "relative_path")
    new_assets = load_csv_map(args.new_dir / "current_assets_manifest.csv", "relative_path")
    old_h5_5x = load_csv_map(args.old_dir / "current_h5_manifest_5x.csv", "file_name")
    new_h5_5x = load_csv_map(args.new_dir / "current_h5_manifest_5x.csv", "file_name")
    old_h5_20x = load_csv_map(args.old_dir / "current_h5_manifest_20x.csv", "file_name")
    new_h5_20x = load_csv_map(args.new_dir / "current_h5_manifest_20x.csv", "file_name")

    report = {
        "environment_version_differences": {
            key: {"old": old_env.get(key), "new": new_env.get(key)}
            for key in [
                "python_version",
                "torch_version",
                "torch_cuda_version",
                "cudnn_version",
                "numpy",
                "scipy",
                "sklearn",
                "h5py",
                "pandas",
                "open_clip",
                "transformers",
                "huggingface_hub",
                "tokenizers",
                "ml_collections",
                "git_commit",
                "git_branch",
            ]
            if old_env.get(key) != new_env.get(key)
        },
        "asset_compare": compare_rows(old_assets, new_assets, "sha256", ["size_bytes", "line_count"]),
        "h5_compare_5x": compare_rows(old_h5_5x, new_h5_5x, "sha256", ["feature_shape", "feature_dtype", "feature_hash", "coords_hash"]),
        "h5_compare_20x": compare_rows(old_h5_20x, new_h5_20x, "sha256", ["feature_shape", "feature_dtype", "feature_hash", "coords_hash"]),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
