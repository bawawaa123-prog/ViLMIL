from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COORD_KEYS = ("coords", "coord", "coordinates", "patch_coords")
EMPTY_MATCH_RATIO_ALERT = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step41: audit whether low/high raw patch coordinates can support real spatial correspondence."
    )
    parser.add_argument("--data_root_dir", default="/xiangmu/data/VILMIL")
    parser.add_argument("--data_folder_s", default="features_biomedclip_5x")
    parser.add_argument("--data_folder_l", default="features_biomedclip_20x")
    parser.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    parser.add_argument("--split_dir", default="splits/adenocarcinoma/task_adenocarcinoma_strictcv_100")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output_dir", default="results_stage41/low_high_coordinate_audit")
    parser.add_argument("--max_slides", type=int, default=0)
    parser.add_argument("--low_patch_size", type=float, default=256.0)
    parser.add_argument("--high_patch_size", type=float, default=256.0)
    parser.add_argument("--bbox_expand", type=float, default=1.0)
    parser.add_argument("--try_scale_ratios", default="1,2,4,0.5,0.25")
    parser.add_argument("--coord_mode", choices=["auto", "top_left", "center"], default="auto")
    parser.add_argument("--num_examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def warn(message: str, warning_log: list[str]) -> None:
    print(f"[Warning] {message}")
    warning_log.append(message)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def parse_scale_ratios(raw: str) -> list[float]:
    ratios: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        ratio = float(token)
        if ratio <= 0:
            raise ValueError(f"Scale ratio must be positive, got {ratio}")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("No valid scale ratios were provided.")
    return ratios


def format_float(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("N/A").astype(str)
    header = "| " + " | ".join(safe_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(safe_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in safe_df.columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def serialise_attr_value(value: object) -> object:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [serialise_attr_value(item) for item in value]
    return value


def attrs_to_dict(attrs: h5py.AttributeManager) -> dict[str, object]:
    return {str(key): serialise_attr_value(value) for key, value in attrs.items()}


def load_slide_ids(
    csv_path: Path,
    split_dir: Path,
    fold: int,
    split: str,
    max_slides: int,
    warning_log: list[str],
) -> tuple[list[str], dict[str, object]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV path does not exist: {csv_path}")

    df_csv = pd.read_csv(csv_path)
    if "slide_id" not in df_csv.columns:
        raise ValueError(f"Expected 'slide_id' column in {csv_path}")

    csv_slide_ids = dedupe_preserve_order(df_csv["slide_id"].dropna().astype(str).tolist())
    selected_slide_ids = list(csv_slide_ids)
    source = "csv"
    source_path = csv_path
    fallback_to_csv = False

    if split != "all":
        split_path = split_dir / f"splits_{fold}.csv"
        if split_path.is_file():
            df_split = pd.read_csv(split_path)
            if split not in df_split.columns:
                warn(
                    f"Split file {split_path} does not contain column '{split}'; falling back to CSV slide list.",
                    warning_log,
                )
                fallback_to_csv = True
            else:
                selected_slide_ids = dedupe_preserve_order(df_split[split].dropna().astype(str).tolist())
                source = "split"
                source_path = split_path
        else:
            warn(
                f"Split file {split_path} does not exist; falling back to CSV slide list from {csv_path}.",
                warning_log,
            )
            fallback_to_csv = True

    if max_slides > 0:
        selected_slide_ids = selected_slide_ids[:max_slides]

    return selected_slide_ids, {
        "source": source,
        "source_path": str(source_path),
        "csv_slide_count": len(csv_slide_ids),
        "selected_slide_count": len(selected_slide_ids),
        "split": split,
        "fold": fold,
        "fallback_to_csv": fallback_to_csv,
    }


def read_h5_coords_summary(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        coord_key = next((key for key in COORD_KEYS if key in handle), None)
        if coord_key is None:
            raise KeyError(f"No coordinate dataset found in {path}; tried keys {COORD_KEYS}")

        coord_dataset = handle[coord_key]
        coords = np.asarray(coord_dataset)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"Coordinate dataset must have shape [N, >=2], got {coords.shape} in {path}")
        coords = coords[:, :2]

        feature_shape = None
        feature_dtype = None
        if "features" in handle:
            feature_shape = list(handle["features"].shape)
            feature_dtype = str(handle["features"].dtype)

        finite_mask = np.isfinite(coords).all(axis=1)
        finite_coords = coords[finite_mask]
        has_nan_or_inf = not bool(finite_mask.all())
        duplicate_count = 0
        duplicate_ratio = 0.0
        if len(coords) > 0:
            unique_count = len(np.unique(coords, axis=0))
            duplicate_count = int(len(coords) - unique_count)
            duplicate_ratio = float(duplicate_count / len(coords))

        min_xy = finite_coords.min(axis=0).astype(float) if len(finite_coords) else np.array([np.nan, np.nan])
        max_xy = finite_coords.max(axis=0).astype(float) if len(finite_coords) else np.array([np.nan, np.nan])
        span_xy = max_xy - min_xy

        coord_attrs = attrs_to_dict(coord_dataset.attrs)
        file_attrs = attrs_to_dict(handle.attrs)

        patch_level = coord_attrs.get("patch_level", file_attrs.get("patch_level"))
        patch_size = coord_attrs.get("patch_size", file_attrs.get("patch_size"))

        return {
            "path": str(path),
            "keys": keys,
            "coord_key": coord_key,
            "coords": coords,
            "coords_shape": list(coords.shape),
            "coords_dtype": str(coord_dataset.dtype),
            "features_shape": feature_shape,
            "features_dtype": feature_dtype,
            "patch_count": int(coords.shape[0]),
            "has_nan_or_inf": has_nan_or_inf,
            "duplicate_count": duplicate_count,
            "duplicate_ratio": duplicate_ratio,
            "min_x": float(min_xy[0]) if np.isfinite(min_xy[0]) else None,
            "min_y": float(min_xy[1]) if np.isfinite(min_xy[1]) else None,
            "max_x": float(max_xy[0]) if np.isfinite(max_xy[0]) else None,
            "max_y": float(max_xy[1]) if np.isfinite(max_xy[1]) else None,
            "span_x": float(span_xy[0]) if np.isfinite(span_xy[0]) else None,
            "span_y": float(span_xy[1]) if np.isfinite(span_xy[1]) else None,
            "patch_level": patch_level,
            "patch_size": patch_size,
            "coord_attrs": coord_attrs,
            "file_attrs": file_attrs,
        }


def compute_range_relationship(low_summary: dict[str, object], high_summary: dict[str, object]) -> dict[str, object]:
    low_min = np.array([low_summary["min_x"], low_summary["min_y"]], dtype=float)
    low_max = np.array([low_summary["max_x"], low_summary["max_y"]], dtype=float)
    high_min = np.array([high_summary["min_x"], high_summary["min_y"]], dtype=float)
    high_max = np.array([high_summary["max_x"], high_summary["max_y"]], dtype=float)

    if not np.isfinite(np.concatenate([low_min, low_max, high_min, high_max])).all():
        return {
            "range_iou": None,
            "span_ratio_x": None,
            "span_ratio_y": None,
            "min_delta_x": None,
            "min_delta_y": None,
            "same_coord_system_heuristic": False,
        }

    low_span = np.maximum(low_max - low_min, 1e-6)
    high_span = np.maximum(high_max - high_min, 1e-6)
    inter_min = np.maximum(low_min, high_min)
    inter_max = np.minimum(low_max, high_max)
    inter_span = np.maximum(inter_max - inter_min, 0.0)
    inter_area = float(inter_span[0] * inter_span[1])
    low_area = float((low_max[0] - low_min[0]) * (low_max[1] - low_min[1]))
    high_area = float((high_max[0] - high_min[0]) * (high_max[1] - high_min[1]))
    union_area = max(low_area + high_area - inter_area, 1e-6)
    range_iou = inter_area / union_area
    span_ratio = high_span / low_span
    min_delta = high_min - low_min
    same_coord_system = bool(
        range_iou >= 0.5
        and 0.5 <= span_ratio[0] <= 2.0
        and 0.5 <= span_ratio[1] <= 2.0
    )
    return {
        "range_iou": range_iou,
        "span_ratio_x": float(span_ratio[0]),
        "span_ratio_y": float(span_ratio[1]),
        "min_delta_x": float(min_delta[0]),
        "min_delta_y": float(min_delta[1]),
        "same_coord_system_heuristic": same_coord_system,
    }


def build_low_bboxes(
    low_coords: np.ndarray,
    low_patch_size: float,
    coord_mode: str,
    bbox_expand: float,
    scale_ratio: float,
) -> np.ndarray:
    coords = np.asarray(low_coords, dtype=np.float64)[:, :2]
    if coord_mode == "top_left":
        x0 = coords[:, 0]
        y0 = coords[:, 1]
        x1 = x0 + low_patch_size
        y1 = y0 + low_patch_size
    elif coord_mode == "center":
        half = low_patch_size / 2.0
        x0 = coords[:, 0] - half
        y0 = coords[:, 1] - half
        x1 = coords[:, 0] + half
        y1 = coords[:, 1] + half
    else:
        raise ValueError(f"Unsupported coord_mode: {coord_mode}")

    bboxes = np.stack([x0, y0, x1, y1], axis=1) * float(scale_ratio)
    if bbox_expand != 1.0:
        centers = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2.0
        half_sizes = (bboxes[:, 2:4] - bboxes[:, 0:2]) * (float(bbox_expand) / 2.0)
        bboxes[:, 0:2] = centers - half_sizes
        bboxes[:, 2:4] = centers + half_sizes
    return bboxes


def compute_high_centers(high_coords: np.ndarray, high_patch_size: float, coord_mode: str) -> np.ndarray:
    coords = np.asarray(high_coords, dtype=np.float64)[:, :2]
    if coord_mode == "top_left":
        return coords + (float(high_patch_size) / 2.0)
    if coord_mode == "center":
        return coords
    raise ValueError(f"Unsupported coord_mode: {coord_mode}")


def compute_child_counts(
    low_bboxes: np.ndarray,
    high_centers: np.ndarray,
    selected_low_indices: set[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    low_bboxes = np.asarray(low_bboxes, dtype=np.float64)
    high_centers = np.asarray(high_centers, dtype=np.float64)
    selected_low_indices = selected_low_indices or set()

    if low_bboxes.size == 0:
        return np.zeros((0,), dtype=np.int64), {}
    if high_centers.size == 0:
        empty_matches = {idx: np.zeros((0,), dtype=np.int64) for idx in selected_low_indices}
        return np.zeros((len(low_bboxes),), dtype=np.int64), empty_matches

    order = np.argsort(high_centers[:, 0], kind="mergesort")
    sorted_x = high_centers[order, 0]
    sorted_y = high_centers[order, 1]

    counts = np.zeros((len(low_bboxes),), dtype=np.int64)
    matches: dict[int, np.ndarray] = {}
    for low_index, (x0, y0, x1, y1) in enumerate(low_bboxes):
        left = int(np.searchsorted(sorted_x, x0, side="left"))
        right = int(np.searchsorted(sorted_x, x1, side="right"))
        if right <= left:
            if low_index in selected_low_indices:
                matches[low_index] = np.zeros((0,), dtype=np.int64)
            continue
        candidate_sorted_indices = np.arange(left, right, dtype=np.int64)
        inside_y = (sorted_y[candidate_sorted_indices] >= y0) & (sorted_y[candidate_sorted_indices] <= y1)
        matched_sorted_indices = candidate_sorted_indices[inside_y]
        matched_indices = order[matched_sorted_indices]
        counts[low_index] = int(len(matched_indices))
        if low_index in selected_low_indices:
            matches[low_index] = matched_indices
    return counts, matches


def summarize_distribution(counts: np.ndarray) -> dict[str, float | int | None]:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return {
            "child_count_mean": None,
            "child_count_std": None,
            "child_count_min": None,
            "child_count_max": None,
            "child_count_median": None,
            "child_count_p25": None,
            "child_count_p75": None,
            "child_count_p90": None,
            "child_count_p95": None,
            "child_count_nonzero_mean": None,
            "empty_match_ratio": None,
        }

    nonzero = counts[counts > 0]
    return {
        "child_count_mean": float(counts.mean()),
        "child_count_std": float(counts.std(ddof=0)),
        "child_count_min": int(counts.min()),
        "child_count_max": int(counts.max()),
        "child_count_median": float(np.median(counts)),
        "child_count_p25": float(np.percentile(counts, 25)),
        "child_count_p75": float(np.percentile(counts, 75)),
        "child_count_p90": float(np.percentile(counts, 90)),
        "child_count_p95": float(np.percentile(counts, 95)),
        "child_count_nonzero_mean": float(nonzero.mean()) if nonzero.size else 0.0,
        "empty_match_ratio": float(np.mean(counts == 0)),
    }


def expected_child_count(low_patch_size: float, high_patch_size: float, scale_ratio: float) -> float:
    return max(1.0, float((low_patch_size * scale_ratio / max(high_patch_size, 1e-6)) ** 2))


def choose_recommended_combo(
    match_df: pd.DataFrame,
    low_patch_size: float,
    high_patch_size: float,
) -> dict[str, object] | None:
    if match_df.empty:
        return None

    rows: list[dict[str, object]] = []
    for (coord_mode, scale_ratio), group in match_df.groupby(["coord_mode", "scale_ratio"], sort=False):
        total_low_patches = int(group["low_patch_count"].sum())
        total_high_patches = int(group["high_patch_count"].sum())
        total_empty_matches = float(group["empty_match_count"].sum())
        weighted_empty_ratio = total_empty_matches / max(total_low_patches, 1)

        child_counts = group["child_count_mean"].dropna().astype(float)
        median_values = group["child_count_median"].dropna().astype(float)
        p95_values = group["child_count_p95"].dropna().astype(float)
        slide_count = int(group["status"].eq("ok").sum())
        same_coord_rate = float(group["same_coord_system_heuristic"].astype(float).mean())
        expected = expected_child_count(low_patch_size, high_patch_size, float(scale_ratio))
        median_child = float(median_values.median()) if not median_values.empty else None
        median_error = abs(math.log1p(median_child or 0.0) - math.log1p(expected))
        p95_penalty = abs(math.log1p(float(p95_values.median()) if not p95_values.empty else 0.0) - math.log1p(expected))
        rows.append(
            {
                "coord_mode": coord_mode,
                "scale_ratio": float(scale_ratio),
                "slide_count": slide_count,
                "total_low_patches": total_low_patches,
                "total_high_patches": total_high_patches,
                "weighted_empty_ratio": weighted_empty_ratio,
                "median_child_count": median_child,
                "mean_child_count": float(child_counts.mean()) if not child_counts.empty else None,
                "median_p95": float(p95_values.median()) if not p95_values.empty else None,
                "same_coord_system_rate": same_coord_rate,
                "expected_child_count": expected,
                "score_tuple": (
                    round(weighted_empty_ratio, 8),
                    round(median_error, 8),
                    round(p95_penalty, 8),
                    round(abs(float(scale_ratio) - 1.0), 8),
                    0 if coord_mode == "top_left" else 1,
                ),
            }
        )

    if not rows:
        return None

    ranked = pd.DataFrame(rows).sort_values(by="score_tuple", kind="mergesort").reset_index(drop=True)
    best = ranked.iloc[0].to_dict()
    best.pop("score_tuple", None)
    best["ranking_table"] = ranked.drop(columns=["score_tuple"])
    return best


def format_key_counter(counter: Counter[str]) -> str:
    if not counter:
        return "N/A"
    return ", ".join(f"{key} ({count})" for key, count in sorted(counter.items()))


def format_value_counter(counter: Counter[str]) -> str:
    if not counter:
        return "N/A"
    return ", ".join(f"{key} ({count})" for key, count in sorted(counter.items()))


def build_examples(
    args: argparse.Namespace,
    candidate_slide_ids: list[str],
    recommended_combo: dict[str, object] | None,
    low_dir: Path,
    high_dir: Path,
    rng: random.Random,
) -> pd.DataFrame:
    if recommended_combo is None or args.num_examples <= 0:
        return pd.DataFrame(
            columns=[
                "slide_id",
                "coord_mode",
                "scale_ratio",
                "low_patch_index",
                "low_coord_x",
                "low_coord_y",
                "low_bbox_x0",
                "low_bbox_y0",
                "low_bbox_x1",
                "low_bbox_y1",
                "child_count",
                "high_child_rank",
                "high_child_coord_x",
                "high_child_coord_y",
                "high_child_center_x",
                "high_child_center_y",
            ]
        )

    coord_mode = str(recommended_combo["coord_mode"])
    scale_ratio = float(recommended_combo["scale_ratio"])
    rows: list[dict[str, object]] = []

    slide_order = list(candidate_slide_ids)
    rng.shuffle(slide_order)
    for slide_id in slide_order:
        if len(rows) >= args.num_examples:
            break
        low_path = low_dir / f"{slide_id}.h5"
        high_path = high_dir / f"{slide_id}.h5"
        if not low_path.is_file() or not high_path.is_file():
            continue

        try:
            low_summary = read_h5_coords_summary(low_path)
            high_summary = read_h5_coords_summary(high_path)
        except Exception:
            continue

        low_coords = np.asarray(low_summary["coords"])
        high_coords = np.asarray(high_summary["coords"])
        if len(low_coords) == 0:
            continue

        low_bboxes = build_low_bboxes(
            low_coords=low_coords,
            low_patch_size=args.low_patch_size,
            coord_mode=coord_mode,
            bbox_expand=args.bbox_expand,
            scale_ratio=scale_ratio,
        )
        high_centers = compute_high_centers(
            high_coords=high_coords,
            high_patch_size=args.high_patch_size,
            coord_mode=coord_mode,
        )

        candidate_indices = list(range(len(low_coords)))
        rng.shuffle(candidate_indices)
        selected_low_indices = set(candidate_indices[: min(len(candidate_indices), max(args.num_examples * 2, 8))])
        counts, matches = compute_child_counts(low_bboxes, high_centers, selected_low_indices=selected_low_indices)

        ranked_low_indices = sorted(
            selected_low_indices,
            key=lambda idx: (
                0 if counts[idx] > 0 else 1,
                abs(int(counts[idx]) - expected_child_count(args.low_patch_size, args.high_patch_size, scale_ratio)),
                idx,
            ),
        )

        for low_index in ranked_low_indices:
            if len(rows) >= args.num_examples:
                break
            matched_indices = matches.get(low_index, np.zeros((0,), dtype=np.int64))
            child_count = int(counts[low_index])
            low_bbox = low_bboxes[low_index]
            if child_count == 0:
                rows.append(
                    {
                        "slide_id": slide_id,
                        "coord_mode": coord_mode,
                        "scale_ratio": scale_ratio,
                        "low_patch_index": int(low_index),
                        "low_coord_x": float(low_coords[low_index, 0]),
                        "low_coord_y": float(low_coords[low_index, 1]),
                        "low_bbox_x0": float(low_bbox[0]),
                        "low_bbox_y0": float(low_bbox[1]),
                        "low_bbox_x1": float(low_bbox[2]),
                        "low_bbox_y1": float(low_bbox[3]),
                        "child_count": 0,
                        "high_child_rank": None,
                        "high_child_coord_x": None,
                        "high_child_coord_y": None,
                        "high_child_center_x": None,
                        "high_child_center_y": None,
                    }
                )
                continue

            for rank, high_index in enumerate(matched_indices[: min(child_count, 8)], start=1):
                rows.append(
                    {
                        "slide_id": slide_id,
                        "coord_mode": coord_mode,
                        "scale_ratio": scale_ratio,
                        "low_patch_index": int(low_index),
                        "low_coord_x": float(low_coords[low_index, 0]),
                        "low_coord_y": float(low_coords[low_index, 1]),
                        "low_bbox_x0": float(low_bbox[0]),
                        "low_bbox_y0": float(low_bbox[1]),
                        "low_bbox_x1": float(low_bbox[2]),
                        "low_bbox_y1": float(low_bbox[3]),
                        "child_count": child_count,
                        "high_child_rank": rank,
                        "high_child_coord_x": float(high_coords[high_index, 0]),
                        "high_child_coord_y": float(high_coords[high_index, 1]),
                        "high_child_center_x": float(high_centers[high_index, 0]),
                        "high_child_center_y": float(high_centers[high_index, 1]),
                    }
                )
            if len(rows) >= args.num_examples:
                break

    return pd.DataFrame(rows)


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    low_dir: Path,
    high_dir: Path,
    slide_source_info: dict[str, object],
    manifest: dict[str, object],
    match_df: pd.DataFrame,
    child_df: pd.DataFrame,
    empty_df: pd.DataFrame,
    h5_summary: dict[str, object],
    recommended_combo: dict[str, object] | None,
    warning_log: list[str],
) -> None:
    combo_summary = manifest.get("combo_summary", [])
    combo_df = pd.DataFrame(combo_summary)
    combo_report_df = pd.DataFrame(
        [
            {
                "coord_mode": row.get("coord_mode"),
                "scale_ratio": row.get("scale_ratio"),
                "slides": row.get("slide_count"),
                "total_low_patches": row.get("total_low_patches"),
                "weighted_empty_ratio": format_float(row.get("weighted_empty_ratio")),
                "mean_child_count": format_float(row.get("mean_child_count")),
                "median_child_count": format_float(row.get("median_child_count")),
                "median_p95": format_float(row.get("median_p95")),
                "same_coord_system_rate": format_float(row.get("same_coord_system_rate")),
                "expected_child_count": format_float(row.get("expected_child_count")),
            }
            for row in combo_summary
        ]
    )

    low_span_x_mean = match_df["low_span_x"].dropna().mean() if not match_df.empty else None
    low_span_y_mean = match_df["low_span_y"].dropna().mean() if not match_df.empty else None
    high_span_x_mean = match_df["high_span_x"].dropna().mean() if not match_df.empty else None
    high_span_y_mean = match_df["high_span_y"].dropna().mean() if not match_df.empty else None
    same_coord_rate = (
        float(match_df["same_coord_system_heuristic"].astype(float).mean()) if not match_df.empty else None
    )

    empty_summary_df = pd.DataFrame(
        [
            {
                "coord_mode": row.get("coord_mode"),
                "scale_ratio": row.get("scale_ratio"),
                "high_empty_slides": row.get("high_empty_slides"),
                "mean_slide_empty_ratio": format_float(row.get("mean_slide_empty_ratio")),
                "max_slide_empty_ratio": format_float(row.get("max_slide_empty_ratio")),
            }
            for row in manifest.get("empty_summary", [])
        ]
    )

    h5_rows = pd.DataFrame(
        [
            {
                "scale": "low",
                "coord_keys": format_key_counter(Counter(h5_summary["low_coord_keys"])),
                "patch_level_values": format_value_counter(Counter(h5_summary["low_patch_levels"])),
                "patch_size_values": format_value_counter(Counter(h5_summary["low_patch_sizes"])),
            },
            {
                "scale": "high",
                "coord_keys": format_key_counter(Counter(h5_summary["high_coord_keys"])),
                "patch_level_values": format_value_counter(Counter(h5_summary["high_patch_levels"])),
                "patch_size_values": format_value_counter(Counter(h5_summary["high_patch_sizes"])),
            },
        ]
    )

    recommended_text = "No reliable coord_mode / scale_ratio recommendation could be produced."
    step42_judgement = "暂缓 Step42。"
    risk_text = "No processed slides were available, so coordinate reliability remains unresolved."
    if recommended_combo is not None:
        recommended_text = (
            f"Recommended coord_mode: `{recommended_combo['coord_mode']}`; "
            f"recommended scale_ratio: `{recommended_combo['scale_ratio']}`."
        )
        weighted_empty_ratio = float(recommended_combo["weighted_empty_ratio"])
        median_child = recommended_combo.get("median_child_count")
        if weighted_empty_ratio < 0.5:
            step42_judgement = (
                "可以进入 Step42（CGSP Anchor Selection Audit），"
                "因为至少存在一组 coord_mode + scale_ratio 在大多数 low patches 上给出非空匹配。"
            )
            risk_text = (
                f"当前推荐组合的加权 empty match ratio 为 {weighted_empty_ratio:.4f}；"
                f"仍需在 Step42 使用真实 coords 而不是 attention centroid，并继续检查 anchor 选择是否稳定。"
            )
        else:
            step42_judgement = (
                "建议暂缓 Step42，优先复核坐标读取或低高倍坐标系转换。"
            )
            risk_text = (
                f"当前最优组合的加权 empty match ratio 仍为 {weighted_empty_ratio:.4f}，"
                "说明 low-high 对应关系不足够可靠。"
            )
        if median_child is not None:
            risk_text += f" 推荐组合的 slide-level child count median 约为 {float(median_child):.4f}。"

    lines = [
        "# Step41 Low-High Coordinate Correspondence Audit",
        "",
        "## Step41 目的",
        "- 审计 `features_biomedclip_5x` 与 `features_biomedclip_20x` 原始 patch coords 是否支持真实 low-high 空间对应。",
        "- 不训练模型，不修改模型前向逻辑，只为后续 HCRC / Step42 提供坐标可靠性结论。",
        "",
        "## 输入路径",
        f"- low directory: `{relative_path(low_dir)}`",
        f"- high directory: `{relative_path(high_dir)}`",
        f"- csv_path: `{relative_path(resolve_path(args.csv_path))}`",
        f"- split_dir: `{relative_path(resolve_path(args.split_dir))}`",
        f"- fold: `{args.fold}`",
        f"- split: `{args.split}`",
        f"- slide source: `{slide_source_info['source']}` from `{slide_source_info['source_path']}`",
        "",
        "## 处理概况",
        f"- requested slides: `{slide_source_info['selected_slide_count']}`",
        f"- processed slides: `{manifest['processed_slide_count']}`",
        f"- missing slides: `{manifest['missing_slide_count']}`",
        f"- failed slides: `{manifest['failed_slide_count']}`",
        f"- warning count: `{len(warning_log)}`",
        "",
        "## low/high h5 key 与 attrs 总结",
        markdown_table(h5_rows),
        "",
        "## low/high 坐标范围统计",
        f"- mean low span: `x={format_float(low_span_x_mean)}`, `y={format_float(low_span_y_mean)}`",
        f"- mean high span: `x={format_float(high_span_x_mean)}`, `y={format_float(high_span_y_mean)}`",
        f"- same-coordinate-system heuristic rate: `{format_float(same_coord_rate)}`",
        "",
        "## 不同 coord_mode 与 scale_ratio 的匹配结果",
        markdown_table(combo_report_df),
        "",
        "## empty match ratio 总结",
        markdown_table(empty_summary_df),
        "",
        "## child count 分布总结",
        markdown_table(
            child_df.sort_values(["coord_mode", "scale_ratio", "slide_id"])
            .head(12)[
                [
                    "slide_id",
                    "coord_mode",
                    "scale_ratio",
                    "child_count_mean",
                    "child_count_median",
                    "child_count_p90",
                    "child_count_p95",
                    "empty_match_ratio",
                ]
            ]
            .assign(
                child_count_mean=lambda df: df["child_count_mean"].map(format_float),
                child_count_median=lambda df: df["child_count_median"].map(format_float),
                child_count_p90=lambda df: df["child_count_p90"].map(format_float),
                child_count_p95=lambda df: df["child_count_p95"].map(format_float),
                empty_match_ratio=lambda df: df["empty_match_ratio"].map(format_float),
            )
        ),
        "",
        "## 推荐结论",
        f"- {recommended_text}",
        f"- {step42_judgement}",
        "",
        "## 风险与建议",
        f"- {risk_text}",
    ]

    if not empty_df.empty:
        lines.extend(
            [
                "",
                "## High Empty-Match Cases",
                markdown_table(
                    empty_df.sort_values(["empty_match_ratio", "slide_id"], ascending=[False, True])
                    .head(20)[["slide_id", "coord_mode", "scale_ratio", "empty_match_ratio", "child_count_mean", "child_count_p95"]]
                    .assign(
                        empty_match_ratio=lambda df: df["empty_match_ratio"].map(format_float),
                        child_count_mean=lambda df: df["child_count_mean"].map(format_float),
                        child_count_p95=lambda df: df["child_count_p95"].map(format_float),
                    )
                ),
            ]
        )

    if warning_log:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {message}" for message in warning_log[:50]])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    low_dir = resolve_path(os.path.join(args.data_root_dir, args.data_folder_s))
    high_dir = resolve_path(os.path.join(args.data_root_dir, args.data_folder_l))
    csv_path = resolve_path(args.csv_path)
    split_dir = resolve_path(args.split_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not low_dir.is_dir():
        raise FileNotFoundError(f"Low feature directory does not exist: {low_dir}")
    if not high_dir.is_dir():
        raise FileNotFoundError(f"High feature directory does not exist: {high_dir}")

    warning_log: list[str] = []
    slide_ids, slide_source_info = load_slide_ids(
        csv_path=csv_path,
        split_dir=split_dir,
        fold=args.fold,
        split=args.split,
        max_slides=args.max_slides,
        warning_log=warning_log,
    )

    coord_modes = ["top_left", "center"] if args.coord_mode == "auto" else [args.coord_mode]
    scale_ratios = parse_scale_ratios(args.try_scale_ratios)

    match_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    h5_summary: dict[str, list[str]] = {
        "low_coord_keys": [],
        "high_coord_keys": [],
        "low_patch_levels": [],
        "high_patch_levels": [],
        "low_patch_sizes": [],
        "high_patch_sizes": [],
    }

    processed_slide_ids: list[str] = []
    for slide_id in slide_ids:
        low_path = low_dir / f"{slide_id}.h5"
        high_path = high_dir / f"{slide_id}.h5"
        if not low_path.is_file() or not high_path.is_file():
            missing_rows.append(
                {
                    "slide_id": slide_id,
                    "low_exists": low_path.is_file(),
                    "high_exists": high_path.is_file(),
                    "low_path": str(low_path),
                    "high_path": str(high_path),
                }
            )
            continue

        slide_warning_log: list[str] = []
        try:
            low_summary = read_h5_coords_summary(low_path)
            high_summary = read_h5_coords_summary(high_path)
        except Exception as exc:
            warn(f"Failed to read slide {slide_id}: {exc}", warning_log)
            failed_rows.append({"slide_id": slide_id, "error": str(exc)})
            continue

        if low_summary["has_nan_or_inf"]:
            slide_warning_log.append("low coords contain NaN/Inf")
        if high_summary["has_nan_or_inf"]:
            slide_warning_log.append("high coords contain NaN/Inf")

        h5_summary["low_coord_keys"].append(str(low_summary["coord_key"]))
        h5_summary["high_coord_keys"].append(str(high_summary["coord_key"]))
        h5_summary["low_patch_levels"].append(str(low_summary["patch_level"]))
        h5_summary["high_patch_levels"].append(str(high_summary["patch_level"]))
        h5_summary["low_patch_sizes"].append(str(low_summary["patch_size"]))
        h5_summary["high_patch_sizes"].append(str(high_summary["patch_size"]))

        range_relationship = compute_range_relationship(low_summary, high_summary)
        low_coords = np.asarray(low_summary["coords"])
        high_coords = np.asarray(high_summary["coords"])
        if len(low_coords) == 0 or len(high_coords) == 0:
            slide_warning_log.append("low or high coords are empty")
        for slide_warning in slide_warning_log:
            warning_log.append(f"{slide_id}: {slide_warning}")

        for coord_mode in coord_modes:
            high_centers = compute_high_centers(high_coords, args.high_patch_size, coord_mode)
            for scale_ratio in scale_ratios:
                low_bboxes = build_low_bboxes(
                    low_coords=low_coords,
                    low_patch_size=args.low_patch_size,
                    coord_mode=coord_mode,
                    bbox_expand=args.bbox_expand,
                    scale_ratio=scale_ratio,
                )
                counts, _ = compute_child_counts(low_bboxes, high_centers)
                distribution = summarize_distribution(counts)
                empty_match_count = int(np.sum(counts == 0)) if len(counts) else 0
                match_row = {
                    "slide_id": slide_id,
                    "status": "ok",
                    "coord_mode": coord_mode,
                    "scale_ratio": float(scale_ratio),
                    "low_coord_key": low_summary["coord_key"],
                    "high_coord_key": high_summary["coord_key"],
                    "low_patch_count": int(low_summary["patch_count"]),
                    "high_patch_count": int(high_summary["patch_count"]),
                    "low_coords_shape": json.dumps(low_summary["coords_shape"]),
                    "high_coords_shape": json.dumps(high_summary["coords_shape"]),
                    "low_features_shape": json.dumps(low_summary["features_shape"]),
                    "high_features_shape": json.dumps(high_summary["features_shape"]),
                    "low_min_x": low_summary["min_x"],
                    "low_min_y": low_summary["min_y"],
                    "low_max_x": low_summary["max_x"],
                    "low_max_y": low_summary["max_y"],
                    "low_span_x": low_summary["span_x"],
                    "low_span_y": low_summary["span_y"],
                    "high_min_x": high_summary["min_x"],
                    "high_min_y": high_summary["min_y"],
                    "high_max_x": high_summary["max_x"],
                    "high_max_y": high_summary["max_y"],
                    "high_span_x": high_summary["span_x"],
                    "high_span_y": high_summary["span_y"],
                    "low_patch_level": low_summary["patch_level"],
                    "high_patch_level": high_summary["patch_level"],
                    "low_patch_size_attr": low_summary["patch_size"],
                    "high_patch_size_attr": high_summary["patch_size"],
                    "low_has_nan_or_inf": bool(low_summary["has_nan_or_inf"]),
                    "high_has_nan_or_inf": bool(high_summary["has_nan_or_inf"]),
                    "low_duplicate_count": int(low_summary["duplicate_count"]),
                    "high_duplicate_count": int(high_summary["duplicate_count"]),
                    "low_duplicate_ratio": float(low_summary["duplicate_ratio"]),
                    "high_duplicate_ratio": float(high_summary["duplicate_ratio"]),
                    "range_iou": range_relationship["range_iou"],
                    "span_ratio_x": range_relationship["span_ratio_x"],
                    "span_ratio_y": range_relationship["span_ratio_y"],
                    "min_delta_x": range_relationship["min_delta_x"],
                    "min_delta_y": range_relationship["min_delta_y"],
                    "same_coord_system_heuristic": bool(range_relationship["same_coord_system_heuristic"]),
                    "empty_match_count": empty_match_count,
                    "warning_count": len(slide_warning_log),
                    "warnings": " | ".join(slide_warning_log),
                }
                match_row.update(distribution)
                child_row = {
                    "slide_id": slide_id,
                    "coord_mode": coord_mode,
                    "scale_ratio": float(scale_ratio),
                    "low_patch_count": int(low_summary["patch_count"]),
                    "high_patch_count": int(high_summary["patch_count"]),
                    **distribution,
                }
                match_rows.append(match_row)
                child_rows.append(child_row)

        processed_slide_ids.append(slide_id)

    match_df = pd.DataFrame(match_rows)
    child_df = pd.DataFrame(child_rows)
    empty_df = pd.DataFrame(
        [
            row
            for row in child_rows
            if row.get("empty_match_ratio") is not None and float(row["empty_match_ratio"]) >= EMPTY_MATCH_RATIO_ALERT
        ]
    )

    recommended_combo = choose_recommended_combo(
        match_df=match_df,
        low_patch_size=args.low_patch_size,
        high_patch_size=args.high_patch_size,
    )

    example_df = build_examples(
        args=args,
        candidate_slide_ids=processed_slide_ids,
        recommended_combo=recommended_combo,
        low_dir=low_dir,
        high_dir=high_dir,
        rng=rng,
    )

    combo_summary: list[dict[str, object]] = []
    empty_summary: list[dict[str, object]] = []
    if not match_df.empty:
        for (coord_mode, scale_ratio), group in match_df.groupby(["coord_mode", "scale_ratio"], sort=False):
            combo_summary.append(
                {
                    "coord_mode": coord_mode,
                    "scale_ratio": float(scale_ratio),
                    "slide_count": int(group["status"].eq("ok").sum()),
                    "total_low_patches": int(group["low_patch_count"].sum()),
                    "total_high_patches": int(group["high_patch_count"].sum()),
                    "weighted_empty_ratio": float(group["empty_match_count"].sum() / max(group["low_patch_count"].sum(), 1)),
                    "mean_child_count": float(group["child_count_mean"].dropna().mean()) if not group["child_count_mean"].dropna().empty else None,
                    "median_child_count": float(group["child_count_median"].dropna().median()) if not group["child_count_median"].dropna().empty else None,
                    "median_p95": float(group["child_count_p95"].dropna().median()) if not group["child_count_p95"].dropna().empty else None,
                    "same_coord_system_rate": float(group["same_coord_system_heuristic"].astype(float).mean()),
                    "expected_child_count": expected_child_count(args.low_patch_size, args.high_patch_size, float(scale_ratio)),
                }
            )
        for (coord_mode, scale_ratio), group in child_df.groupby(["coord_mode", "scale_ratio"], sort=False):
            valid_empty = group["empty_match_ratio"].dropna().astype(float)
            if valid_empty.empty:
                continue
            empty_summary.append(
                {
                    "coord_mode": coord_mode,
                    "scale_ratio": float(scale_ratio),
                    "high_empty_slides": int((valid_empty >= EMPTY_MATCH_RATIO_ALERT).sum()),
                    "mean_slide_empty_ratio": float(valid_empty.mean()),
                    "max_slide_empty_ratio": float(valid_empty.max()),
                }
            )

    manifest = {
        "step": "Step41 Low-High Coordinate Correspondence Audit",
        "parameters": {
            "data_root_dir": args.data_root_dir,
            "data_folder_s": args.data_folder_s,
            "data_folder_l": args.data_folder_l,
            "csv_path": str(csv_path),
            "split_dir": str(split_dir),
            "fold": args.fold,
            "split": args.split,
            "output_dir": str(output_dir),
            "max_slides": args.max_slides,
            "low_patch_size": args.low_patch_size,
            "high_patch_size": args.high_patch_size,
            "bbox_expand": args.bbox_expand,
            "try_scale_ratios": scale_ratios,
            "coord_mode": args.coord_mode,
            "num_examples": args.num_examples,
            "seed": args.seed,
        },
        "input_paths": {
            "low_dir": str(low_dir),
            "high_dir": str(high_dir),
            "csv_path": str(csv_path),
            "split_dir": str(split_dir),
        },
        "output_paths": {
            "low_high_coordinate_match_stats_csv": str(output_dir / "low_high_coordinate_match_stats.csv"),
            "child_count_distribution_csv": str(output_dir / "child_count_distribution.csv"),
            "empty_match_cases_csv": str(output_dir / "empty_match_cases.csv"),
            "example_low_high_pairs_csv": str(output_dir / "example_low_high_pairs.csv"),
            "coordinate_match_report_md": str(output_dir / "coordinate_match_report.md"),
            "stage41_manifest_json": str(output_dir / "stage41_manifest.json"),
        },
        "slide_source": slide_source_info,
        "requested_slide_count": len(slide_ids),
        "processed_slide_count": len(processed_slide_ids),
        "missing_slide_count": len(missing_rows),
        "failed_slide_count": len(failed_rows),
        "processed_slide_ids": processed_slide_ids,
        "missing_slides": missing_rows,
        "failed_slides": failed_rows,
        "warning_count": len(warning_log),
        "warnings": warning_log,
        "recommended_coord_mode": recommended_combo["coord_mode"] if recommended_combo else None,
        "recommended_scale_ratio": recommended_combo["scale_ratio"] if recommended_combo else None,
        "recommended_combo": {
            key: value
            for key, value in (recommended_combo or {}).items()
            if key != "ranking_table"
        },
        "combo_summary": combo_summary,
        "empty_summary": empty_summary,
    }

    match_df.to_csv(output_dir / "low_high_coordinate_match_stats.csv", index=False)
    child_df.to_csv(output_dir / "child_count_distribution.csv", index=False)
    empty_df.to_csv(output_dir / "empty_match_cases.csv", index=False)
    example_df.to_csv(output_dir / "example_low_high_pairs.csv", index=False)
    write_report(
        report_path=output_dir / "coordinate_match_report.md",
        args=args,
        low_dir=low_dir,
        high_dir=high_dir,
        slide_source_info=slide_source_info,
        manifest=manifest,
        match_df=match_df,
        child_df=child_df,
        empty_df=empty_df,
        h5_summary=h5_summary,
        recommended_combo=recommended_combo,
        warning_log=warning_log,
    )

    manifest_path = output_dir / "stage41_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Step41 coordinate audit complete.")
    print(f"Processed slides: {len(processed_slide_ids)}")
    print(f"Missing slides: {len(missing_rows)}")
    print(f"Failed slides: {len(failed_rows)}")
    if recommended_combo is not None:
        print(
            "Recommended combo: "
            f"coord_mode={recommended_combo['coord_mode']}, scale_ratio={recommended_combo['scale_ratio']}, "
            f"weighted_empty_ratio={recommended_combo['weighted_empty_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
