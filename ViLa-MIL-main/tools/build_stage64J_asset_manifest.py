#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


NORMAL_FILES = [
    "dataset_csv/all_data.csv",
    "text_prompt/adenocarcinoma_dual_scale_prompt.csv",
    "dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json",
]

SPLIT_DIR = "splits/adenocarcinoma/task_adenocarcinoma_strictcv_100"
FEATURE_DIRS = {
    "5x": "features_biomedclip_5x",
    "20x": "features_biomedclip_20x",
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def line_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def inspect_h5(path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "dataset_keys": [],
        "dataset_summary": {},
        "feature_dataset": None,
        "feature_shape": None,
        "feature_dtype": None,
        "feature_has_nan": None,
        "feature_has_inf": None,
        "feature_hash": None,
        "coords_shape": None,
        "coords_hash": None,
    }
    with h5py.File(path, "r") as h5:
        keys = sorted(h5.keys())
        info["dataset_keys"] = keys
        preferred_feature_key = None
        for candidate in ("features", "feature", "imgs", "embeddings"):
            if candidate in h5:
                preferred_feature_key = candidate
                break
        coords_key = "coords" if "coords" in h5 else None
        for key in keys:
            dataset = h5[key]
            if not hasattr(dataset, "shape"):
                continue
            arr = dataset[()]
            arr_np = np.asarray(arr)
            is_numeric = np.issubdtype(arr_np.dtype, np.number)
            has_nan = bool(np.isnan(arr_np).any()) if is_numeric else False
            has_inf = bool(np.isinf(arr_np).any()) if is_numeric else False
            summary = {
                "shape": list(arr_np.shape),
                "dtype": str(arr_np.dtype),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "summary_hash": array_hash(arr_np),
            }
            info["dataset_summary"][key] = summary
            if preferred_feature_key is None and key != coords_key and arr_np.ndim >= 1 and is_numeric:
                preferred_feature_key = key
            if key == coords_key:
                info["coords_shape"] = list(arr_np.shape)
                info["coords_hash"] = summary["summary_hash"]

        if preferred_feature_key is not None:
            feature_summary = info["dataset_summary"][preferred_feature_key]
            info["feature_dataset"] = preferred_feature_key
            info["feature_shape"] = feature_summary["shape"]
            info["feature_dtype"] = feature_summary["dtype"]
            info["feature_has_nan"] = feature_summary["has_nan"]
            info["feature_has_inf"] = feature_summary["has_inf"]
            info["feature_hash"] = feature_summary["summary_hash"]
    return info


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["full", "fast"], default="full")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()

    normal_rows: list[dict[str, object]] = []
    split_root = project_root / SPLIT_DIR
    split_files = sorted(p for p in split_root.glob("*") if p.is_file())
    all_normal_paths = [project_root / rel for rel in NORMAL_FILES] + split_files

    for path in all_normal_paths:
        rel = path.relative_to(project_root).as_posix()
        print(f"[normal] {rel}", flush=True)
        normal_rows.append(
            {
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "line_count": line_count(path),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )

    feature_rows: dict[str, list[dict[str, object]]] = {"5x": [], "20x": []}
    aggregate_entries: list[str] = []
    for scale, rel_dir in FEATURE_DIRS.items():
        feature_dir = data_root / rel_dir
        files = sorted(feature_dir.glob("*.h5"))
        total = len(files)
        for idx, path in enumerate(files, start=1):
            print(f"[{scale}] {idx}/{total} {path.name}", flush=True)
            row = inspect_h5(path)
            row["relative_path"] = path.relative_to(project_root).as_posix() if str(path).startswith(str(project_root)) else path.relative_to(data_root).as_posix()
            row["dataset_keys"] = json.dumps(row["dataset_keys"], ensure_ascii=False)
            row["dataset_summary"] = json.dumps(row["dataset_summary"], ensure_ascii=False, sort_keys=True)
            feature_rows[scale].append(row)
            aggregate_entries.append(f"{scale}:{path.name}:{row['sha256']}:{row['feature_hash']}:{row['coords_hash']}")

    normal_csv = args.output_dir / "current_assets_manifest.csv"
    write_csv(normal_csv, normal_rows, ["relative_path", "size_bytes", "sha256", "line_count", "mtime"])

    h5_fields = [
        "file_name",
        "relative_path",
        "file_size",
        "sha256",
        "mtime",
        "dataset_keys",
        "dataset_summary",
        "feature_dataset",
        "feature_shape",
        "feature_dtype",
        "feature_has_nan",
        "feature_has_inf",
        "feature_hash",
        "coords_shape",
        "coords_hash",
    ]
    write_csv(args.output_dir / "current_h5_manifest_5x.csv", feature_rows["5x"], h5_fields)
    write_csv(args.output_dir / "current_h5_manifest_20x.csv", feature_rows["20x"], h5_fields)

    assets_json = {
        "mode": args.mode,
        "project_root": str(project_root),
        "data_root": str(data_root),
        "normal_files": normal_rows,
        "h5_counts": {scale: len(rows) for scale, rows in feature_rows.items()},
    }
    (args.output_dir / "current_assets_manifest.json").write_text(
        json.dumps(assets_json, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    aggregate_hasher = hashlib.sha256()
    for row in normal_rows:
        aggregate_hasher.update(f"normal:{row['relative_path']}:{row['sha256']}\n".encode("utf-8"))
    for entry in aggregate_entries:
        aggregate_hasher.update((entry + "\n").encode("utf-8"))
    (args.output_dir / "current_assets_aggregate_sha256.txt").write_text(
        aggregate_hasher.hexdigest() + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
