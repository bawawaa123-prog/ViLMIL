from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_stage42_cgsp_anchor_selection_audit import (  # noqa: E402
    build_candidate_indices,
    build_candidate_proposals,
    compute_patch_scores,
    coords_to_centers,
    decide_recommendation,
    expand_bbox,
    get_class_name,
    load_prompt_features,
    load_prompt_metadata,
    load_slide_ids,
    load_stage41_settings,
    parse_float_list,
    read_h5_features_coords,
    resolve_path,
    select_anchors_nms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step42b: CGSP child coverage sweep.")
    parser.add_argument("--data_root_dir", default="/xiangmu/data/VILMIL")
    parser.add_argument("--data_folder_s", default="features_biomedclip_5x")
    parser.add_argument("--data_folder_l", default="features_biomedclip_20x")
    parser.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    parser.add_argument("--split_dir", default="splits/adenocarcinoma/task_adenocarcinoma_strictcv_100")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--stage41_manifest", default="results_stage41/low_high_coordinate_audit/stage41_manifest.json")
    parser.add_argument("--stage42_manifest", default="results_stage42/cgsp_anchor_selection_audit/stage42_manifest.json")
    parser.add_argument("--stage42_anchor_csv", default="results_stage42/cgsp_anchor_selection_audit/cgsp_selected_anchors.csv")
    parser.add_argument("--concept_prompt_path", default="dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json")
    parser.add_argument("--checkpoint_path", default="results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt")
    parser.add_argument("--output_dir", default="results_stage42b/cgsp_child_coverage_sweep")
    parser.add_argument("--max_slides", type=int, default=0)
    parser.add_argument("--low_patch_size", type=float, default=256.0)
    parser.add_argument("--high_patch_size", type=float, default=256.0)
    parser.add_argument("--coord_mode", default="", choices=["", "top_left", "center"])
    parser.add_argument("--scale_ratio", type=float, default=None)
    parser.add_argument("--proposal_radius_values", default="512,1024,2048,4096")
    parser.add_argument("--nms_radius_values", default="512,1024")
    parser.add_argument("--bbox_expand_values", default="2,3,4,6,8,10")
    parser.add_argument("--high_radius_values", default="512,1024,2048,4096,8192")
    parser.add_argument("--high_child_top_m", type=int, default=16)
    parser.add_argument("--num_anchors", type=int, default=16)
    parser.add_argument("--candidate_top_l", type=int, default=64)
    parser.add_argument("--top_g_concepts", type=int, default=8)
    parser.add_argument("--per_concept_top_m", type=int, default=4)
    parser.add_argument("--prompt_topk", type=int, default=3)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def warn(message: str, warning_log: list[str]) -> None:
    print(f"[Warning] {message}")
    warning_log.append(message)


def format_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("N/A").astype(str)
    header = "| " + " | ".join(safe_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(safe_df.columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in safe_df.columns) + " |" for _, row in safe_df.iterrows()]
    return "\n".join([header, sep] + rows)


def load_json_optional(path: Path, warning_log: list[str]) -> dict[str, object]:
    if not path.is_file():
        warn(f"Optional JSON not found: {path}", warning_log)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn(f"Failed to read JSON {path}: {exc}", warning_log)
        return {}


def bbox_area(bbox: np.ndarray) -> float:
    width = max(float(bbox[2] - bbox[0]), 0.0)
    height = max(float(bbox[3] - bbox[1]), 0.0)
    return width * height


def class_distribution_text(anchors: list[dict[str, object]]) -> str:
    return json.dumps(dict(Counter(str(anchor["proposal_class_name"]) for anchor in anchors)), ensure_ascii=False)


def concept_distribution_text(anchors: list[dict[str, object]]) -> str:
    return json.dumps(dict(Counter(str(anchor["proposal_concept"]) for anchor in anchors)), ensure_ascii=False)


def pair_distance_stats(centers: np.ndarray) -> dict[str, object]:
    if len(centers) <= 1:
        return {
            "anchor_pair_distance_mean": None,
            "anchor_pair_distance_min": None,
            "anchor_pair_distance_max": None,
        }
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distances.append(float(np.linalg.norm(centers[i] - centers[j])))
    arr = np.asarray(distances, dtype=np.float64)
    return {
        "anchor_pair_distance_mean": float(arr.mean()),
        "anchor_pair_distance_min": float(arr.min()),
        "anchor_pair_distance_max": float(arr.max()),
    }


def get_child_stats_for_anchor(
    anchor: dict[str, object],
    high_coords: np.ndarray,
    high_centers: np.ndarray,
    bbox_expand_values: list[float],
    high_radius_values: list[float],
    high_child_top_m: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    anchor_center = np.asarray(anchor["proposal_center"], dtype=np.float64)
    anchor_bbox = np.asarray(anchor["proposal_bbox"], dtype=np.float64)
    distances = np.linalg.norm(high_centers - anchor_center.reshape(1, 2), axis=1)
    nearest_order = np.argsort(distances)
    nearest_top = nearest_order[: min(high_child_top_m, len(nearest_order))]
    nearest_distance_values = distances[nearest_top]

    base_area = bbox_area(anchor_bbox)
    base_width = max(float(anchor_bbox[2] - anchor_bbox[0]), 0.0)
    base_height = max(float(anchor_bbox[3] - anchor_bbox[1]), 0.0)
    nearest_min = float(distances[nearest_order[0]]) if len(nearest_order) else None
    nearest_median = float(np.median(nearest_distance_values)) if len(nearest_distance_values) else None
    nearest_mean = float(nearest_distance_values.mean()) if len(nearest_distance_values) else None

    rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []

    def common(strategy: str, param_name: str, param_value: float, expanded_bbox: np.ndarray | None) -> dict[str, object]:
        if expanded_bbox is None:
            area = None
            width = None
            height = None
            density = None
        else:
            area = bbox_area(expanded_bbox)
            width = max(float(expanded_bbox[2] - expanded_bbox[0]), 0.0)
            height = max(float(expanded_bbox[3] - expanded_bbox[1]), 0.0)
            density = None
        return {
            "slide_id": anchor["slide_id"],
            "proposal_radius": float(anchor["proposal_radius"]),
            "nms_radius": float(anchor["nms_radius"]),
            "anchor_rank": int(anchor["anchor_rank"]),
            "strategy": strategy,
            "param_name": param_name,
            "param_value": float(param_value),
            "bbox_expand": float(param_value) if "bbox" in strategy else None,
            "high_radius": float(param_value) if strategy == "center_radius" else None,
            "anchor_score": float(anchor["proposal_score"]),
            "anchor_center_x": float(anchor_center[0]),
            "anchor_center_y": float(anchor_center[1]),
            "original_anchor_bbox_area": base_area,
            "original_bbox_width": base_width,
            "original_bbox_height": base_height,
            "expanded_bbox_area": area,
            "bbox_width": width,
            "bbox_height": height,
            "neighbor_count": int(anchor["neighbor_count"]),
            "top_class_name": str(anchor["proposal_class_name"]),
            "top_concept": str(anchor["proposal_concept"]),
            "nearest_high_min_distance": nearest_min,
            "nearest_high_median_topm_distance": nearest_median,
            "nearest_high_mean_topm_distance": nearest_mean,
            "high_patch_density_in_bbox": density,
        }

    for bbox_expand in bbox_expand_values:
        expanded = expand_bbox(anchor_bbox, bbox_expand)
        inside = (
            (high_centers[:, 0] >= expanded[0])
            & (high_centers[:, 0] <= expanded[2])
            & (high_centers[:, 1] >= expanded[1])
            & (high_centers[:, 1] <= expanded[3])
        )
        raw_child_indices = np.where(inside)[0]
        raw_distances = distances[raw_child_indices]
        raw_order = np.argsort(raw_distances) if len(raw_distances) else np.array([], dtype=np.int64)
        bbox_used = raw_child_indices[raw_order[:high_child_top_m]] if len(raw_order) else np.array([], dtype=np.int64)
        bbox_used_distances = distances[bbox_used] if len(bbox_used) else np.array([], dtype=np.float64)

        row = common("bbox_containment", "bbox_expand", bbox_expand, expanded)
        row.update(
            {
                "raw_child_count": int(len(raw_child_indices)),
                "used_child_count": int(len(bbox_used)),
                "empty_child_flag": bool(len(raw_child_indices) == 0),
                "fallback_child_count": 0,
                "fallback_ratio": 0.0,
                "used_child_distance_mean": float(bbox_used_distances.mean()) if len(bbox_used_distances) else None,
                "used_child_distance_min": float(bbox_used_distances.min()) if len(bbox_used_distances) else None,
                "used_child_distance_max": float(bbox_used_distances.max()) if len(bbox_used_distances) else None,
                "raw_child_distance_mean": float(raw_distances.mean()) if len(raw_distances) else None,
                "raw_child_distance_median": float(np.median(raw_distances)) if len(raw_distances) else None,
                "high_patch_density_in_bbox": float(len(raw_child_indices) / max(bbox_area(expanded), 1.0)),
            }
        )
        rows.append(row)

        fallback_needed = max(high_child_top_m - len(bbox_used), 0)
        bbox_set = set(int(idx) for idx in bbox_used)
        fallback_indices = [int(idx) for idx in nearest_order if int(idx) not in bbox_set][:fallback_needed]
        combined = list(map(int, bbox_used)) + fallback_indices
        combined_distances = distances[combined] if combined else np.array([], dtype=np.float64)
        row = common("bbox_then_nearest", "bbox_expand", bbox_expand, expanded)
        row.update(
            {
                "raw_child_count": int(len(raw_child_indices)),
                "used_child_count": int(len(combined)),
                "empty_child_flag": bool(len(combined) == 0),
                "fallback_child_count": int(len(fallback_indices)),
                "fallback_ratio": float(len(fallback_indices) / max(len(combined), 1)),
                "used_child_distance_mean": float(combined_distances.mean()) if len(combined_distances) else None,
                "used_child_distance_min": float(combined_distances.min()) if len(combined_distances) else None,
                "used_child_distance_max": float(combined_distances.max()) if len(combined_distances) else None,
                "raw_child_distance_mean": float(raw_distances.mean()) if len(raw_distances) else None,
                "raw_child_distance_median": float(np.median(raw_distances)) if len(raw_distances) else None,
                "high_patch_density_in_bbox": float(len(raw_child_indices) / max(bbox_area(expanded), 1.0)),
            }
        )
        rows.append(row)

        for rank, child_idx in enumerate(combined[: min(3, len(combined))], start=1):
            example_rows.append(
                {
                    "slide_id": anchor["slide_id"],
                    "proposal_radius": float(anchor["proposal_radius"]),
                    "nms_radius": float(anchor["nms_radius"]),
                    "strategy": "bbox_then_nearest",
                    "bbox_expand": float(bbox_expand),
                    "high_radius": None,
                    "anchor_rank": int(anchor["anchor_rank"]),
                    "anchor_center_x": float(anchor_center[0]),
                    "anchor_center_y": float(anchor_center[1]),
                    "child_rank": rank,
                    "child_source": "bbox" if child_idx in bbox_set else "nearest_fallback",
                    "high_child_index": int(child_idx),
                    "high_child_coord_x": float(high_coords[child_idx, 0]),
                    "high_child_coord_y": float(high_coords[child_idx, 1]),
                    "high_child_center_x": float(high_centers[child_idx, 0]),
                    "high_child_center_y": float(high_centers[child_idx, 1]),
                    "distance_to_anchor": float(distances[child_idx]),
                }
            )

    for high_radius in high_radius_values:
        inside_radius = distances <= high_radius
        raw_child_indices = np.where(inside_radius)[0]
        raw_distances = distances[raw_child_indices]
        raw_order = np.argsort(raw_distances) if len(raw_distances) else np.array([], dtype=np.int64)
        used = raw_child_indices[raw_order[:high_child_top_m]] if len(raw_order) else np.array([], dtype=np.int64)
        used_distances = distances[used] if len(used) else np.array([], dtype=np.float64)
        row = common("center_radius", "high_radius", high_radius, None)
        row.update(
            {
                "raw_child_count": int(len(raw_child_indices)),
                "used_child_count": int(len(used)),
                "empty_child_flag": bool(len(raw_child_indices) == 0),
                "fallback_child_count": 0,
                "fallback_ratio": 0.0,
                "used_child_distance_mean": float(used_distances.mean()) if len(used_distances) else None,
                "used_child_distance_min": float(used_distances.min()) if len(used_distances) else None,
                "used_child_distance_max": float(used_distances.max()) if len(used_distances) else None,
                "raw_child_distance_mean": float(raw_distances.mean()) if len(raw_distances) else None,
                "raw_child_distance_median": float(np.median(raw_distances)) if len(raw_distances) else None,
            }
        )
        rows.append(row)
        for rank, child_idx in enumerate(used[:3], start=1):
            example_rows.append(
                {
                    "slide_id": anchor["slide_id"],
                    "proposal_radius": float(anchor["proposal_radius"]),
                    "nms_radius": float(anchor["nms_radius"]),
                    "strategy": "center_radius",
                    "bbox_expand": None,
                    "high_radius": float(high_radius),
                    "anchor_rank": int(anchor["anchor_rank"]),
                    "anchor_center_x": float(anchor_center[0]),
                    "anchor_center_y": float(anchor_center[1]),
                    "child_rank": rank,
                    "child_source": "radius",
                    "high_child_index": int(child_idx),
                    "high_child_coord_x": float(high_coords[child_idx, 0]),
                    "high_child_coord_y": float(high_coords[child_idx, 1]),
                    "high_child_center_x": float(high_centers[child_idx, 0]),
                    "high_child_center_y": float(high_centers[child_idx, 1]),
                    "distance_to_anchor": float(distances[child_idx]),
                }
            )

    return rows, example_rows


def summarize_sweep(anchor_stats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["proposal_radius", "nms_radius", "strategy", "param_name", "param_value"]
    for keys, group in anchor_stats_df.groupby(group_cols, dropna=False, sort=False):
        proposal_radius, nms_radius, strategy, param_name, param_value = keys
        used = pd.to_numeric(group["used_child_count"], errors="coerce")
        raw = pd.to_numeric(group["raw_child_count"], errors="coerce")
        fallback = pd.to_numeric(group["fallback_ratio"], errors="coerce")
        nearest = pd.to_numeric(group["nearest_high_median_topm_distance"], errors="coerce")
        used_dist = pd.to_numeric(group["used_child_distance_mean"], errors="coerce")
        rows.append(
            {
                "proposal_radius": proposal_radius,
                "nms_radius": nms_radius,
                "strategy": strategy,
                "param_name": param_name,
                "param_value": param_value,
                "empty_anchor_ratio": float(group["empty_child_flag"].astype(float).mean()),
                "median_raw_child_count": float(raw.median()),
                "median_used_child_count": float(used.median()),
                "mean_used_child_count": float(used.mean()),
                "p75_used_child_count": float(used.quantile(0.75)),
                "p90_used_child_count": float(used.quantile(0.90)),
                "mean_fallback_ratio": float(fallback.mean()),
                "median_fallback_ratio": float(fallback.median()),
                "nearest_topm_distance_median": float(nearest.median()) if not nearest.dropna().empty else None,
                "used_child_distance_mean": float(used_dist.mean()) if not used_dist.dropna().empty else None,
                "anchor_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarize_anchor_stability(anchor_records: list[dict[str, object]], num_anchors: int) -> pd.DataFrame:
    df = pd.DataFrame(anchor_records)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (proposal_radius, nms_radius), group in df.groupby(["proposal_radius", "nms_radius"], sort=False):
        centers = group[["anchor_center_x", "anchor_center_y"]].to_numpy(dtype=np.float64)
        by_slide = group.groupby("slide_id").size()
        class_counts = Counter(group["top_class_name"].astype(str).tolist())
        concept_counts = Counter(group["top_concept"].astype(str).tolist())
        pair_stats = []
        for _, slide_group in group.groupby("slide_id"):
            slide_centers = slide_group[["anchor_center_x", "anchor_center_y"]].to_numpy(dtype=np.float64)
            pair_stats.append(pair_distance_stats(slide_centers))
        pair_df = pd.DataFrame(pair_stats)
        rows.append(
            {
                "proposal_radius": float(proposal_radius),
                "nms_radius": float(nms_radius),
                "slide_count": int(by_slide.size),
                "mean_selected_anchors": float(by_slide.mean()),
                "median_selected_anchors": float(by_slide.median()),
                "min_selected_anchors": int(by_slide.min()),
                "full_anchor_slide_ratio": float((by_slide >= num_anchors).mean()),
                "unique_classes": int(len(class_counts)),
                "unique_concepts": int(len(concept_counts)),
                "class_distribution": json.dumps(dict(class_counts), ensure_ascii=False),
                "top_concepts": json.dumps(dict(concept_counts.most_common(10)), ensure_ascii=False),
                "anchor_pair_distance_mean_median": float(pair_df["anchor_pair_distance_mean"].median()),
                "anchor_pair_distance_min_median": float(pair_df["anchor_pair_distance_min"].median()),
            }
        )

    out = pd.DataFrame(rows)
    base_rows = out[out["proposal_radius"] == out["proposal_radius"].min()]
    base_centers = {}
    if not base_rows.empty:
        base_radius = float(out["proposal_radius"].min())
        for (slide_id, nms_radius), group in df[df["proposal_radius"] == base_radius].groupby(["slide_id", "nms_radius"]):
            base_centers[(slide_id, nms_radius)] = group.sort_values("anchor_rank")[["anchor_center_x", "anchor_center_y"]].to_numpy(dtype=np.float64)
    drift_by_key = {}
    for (proposal_radius, nms_radius), group in df.groupby(["proposal_radius", "nms_radius"], sort=False):
        drifts = []
        for slide_id, slide_group in group.groupby("slide_id"):
            base = base_centers.get((slide_id, nms_radius))
            if base is None:
                continue
            current = slide_group.sort_values("anchor_rank")[["anchor_center_x", "anchor_center_y"]].to_numpy(dtype=np.float64)
            k = min(len(base), len(current))
            if k:
                drifts.extend(np.linalg.norm(current[:k] - base[:k], axis=1).tolist())
        drift_by_key[(proposal_radius, nms_radius)] = {
            "rankwise_anchor_drift_mean_vs_min_radius": float(np.mean(drifts)) if drifts else None,
            "rankwise_anchor_drift_median_vs_min_radius": float(np.median(drifts)) if drifts else None,
        }
    out["rankwise_anchor_drift_mean_vs_min_radius"] = [
        drift_by_key[(row.proposal_radius, row.nms_radius)]["rankwise_anchor_drift_mean_vs_min_radius"]
        for row in out.itertuples(index=False)
    ]
    out["rankwise_anchor_drift_median_vs_min_radius"] = [
        drift_by_key[(row.proposal_radius, row.nms_radius)]["rankwise_anchor_drift_median_vs_min_radius"]
        for row in out.itertuples(index=False)
    ]
    return out


def choose_strategy(summary_df: pd.DataFrame, stability_df: pd.DataFrame, high_child_top_m: int, num_anchors: int) -> dict[str, object]:
    if summary_df.empty:
        return {"recommended_strategy": None, "enter_step43": False, "reason": "No sweep rows were generated."}
    stability_ok = stability_df.set_index(["proposal_radius", "nms_radius"]).to_dict("index") if not stability_df.empty else {}
    candidates = []
    for _, row in summary_df.iterrows():
        stab = stability_ok.get((row["proposal_radius"], row["nms_radius"]), {})
        full_anchor_ratio = float(stab.get("full_anchor_slide_ratio", 0.0))
        unique_concepts = float(stab.get("unique_concepts", 0.0))
        pass_child = float(row["median_used_child_count"]) >= 4 and float(row["empty_anchor_ratio"]) <= 0.15
        pass_anchor = full_anchor_ratio >= 0.95 and unique_concepts >= 4
        strategy_priority = {"bbox_containment": 0, "center_radius": 1, "bbox_then_nearest": 2}.get(row["strategy"], 9)
        score = (
            0 if (pass_child and pass_anchor) else 1,
            strategy_priority,
            float(row["empty_anchor_ratio"]),
            -float(row["median_used_child_count"]),
            float(row["mean_fallback_ratio"]),
            float(row["proposal_radius"]),
            float(row["nms_radius"]),
        )
        candidates.append({**row.to_dict(), "pass_child": pass_child, "pass_anchor": pass_anchor, "score": score})
    ranked = sorted(candidates, key=lambda item: item["score"])
    best = ranked[0]
    enter_step43 = bool(best["pass_child"] and best["pass_anchor"])
    if best["strategy"] == "bbox_then_nearest" and best["median_used_child_count"] >= high_child_top_m:
        enter_step43 = bool(best["pass_anchor"])
    reason = (
        f"strategy={best['strategy']}, proposal_radius={best['proposal_radius']}, nms_radius={best['nms_radius']}, "
        f"{best['param_name']}={best['param_value']}, empty_anchor_ratio={best['empty_anchor_ratio']:.4f}, "
        f"median_used_child_count={best['median_used_child_count']:.2f}, "
        f"mean_fallback_ratio={best['mean_fallback_ratio']:.4f}, pass_anchor={best['pass_anchor']}."
    )
    return {
        "recommended_strategy": best["strategy"],
        "recommended_proposal_radius": float(best["proposal_radius"]),
        "recommended_nms_radius": float(best["nms_radius"]),
        "recommended_param_name": best["param_name"],
        "recommended_param_value": float(best["param_value"]),
        "recommended_bbox_expand": float(best["param_value"]) if "bbox" in best["strategy"] else None,
        "recommended_high_radius": float(best["param_value"]) if best["strategy"] == "center_radius" else None,
        "recommended_num_anchors": int(num_anchors),
        "recommended_num_high_children": int(high_child_top_m),
        "enter_step43": enter_step43,
        "reason": reason,
    }


def build_strategy_comparison(summary_df: pd.DataFrame, recommendation: dict[str, object]) -> pd.DataFrame:
    rows = []
    for strategy, group in summary_df.groupby("strategy", sort=False):
        pass_rows = group[(group["median_used_child_count"] >= 4) & (group["empty_anchor_ratio"] <= 0.15)]
        if not pass_rows.empty:
            best = pass_rows.sort_values(["empty_anchor_ratio", "mean_fallback_ratio", "param_value"], kind="mergesort").iloc[0]
        else:
            best = group.sort_values(["empty_anchor_ratio", "mean_fallback_ratio", "param_value"], kind="mergesort").iloc[0]
        rows.append(
            {
                "strategy": strategy,
                "best_proposal_radius": best["proposal_radius"],
                "best_nms_radius": best["nms_radius"],
                "best_param_name": best["param_name"],
                "best_param_value": best["param_value"],
                "best_empty_anchor_ratio": best["empty_anchor_ratio"],
                "best_median_used_child_count": best["median_used_child_count"],
                "best_mean_used_child_count": best["mean_used_child_count"],
                "best_mean_fallback_ratio": best["mean_fallback_ratio"],
                "meets_child_threshold": bool(
                    best["median_used_child_count"] >= 4 and best["empty_anchor_ratio"] <= 0.15
                ),
                "is_recommended": strategy == recommendation.get("recommended_strategy"),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    stage41_settings: dict[str, object],
    stage42_manifest: dict[str, object],
    prompt_source_info: dict[str, object],
    slide_source_info: dict[str, object],
    summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    recommendation: dict[str, object],
    processed_count: int,
    failed_rows: list[dict[str, object]],
    warning_log: list[str],
    anchor_level_write_info: dict[str, object],
) -> None:
    def prep(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = df[columns].copy() if not df.empty else pd.DataFrame(columns=columns)
        for col in out.columns:
            if out[col].dtype.kind in "fc":
                out[col] = out[col].map(format_float)
        return out

    bbox_df = summary_df[summary_df["strategy"] == "bbox_containment"]
    radius_df = summary_df[summary_df["strategy"] == "center_radius"]
    fallback_df = summary_df[summary_df["strategy"] == "bbox_then_nearest"]

    # Diagnose whether expanded bbox area changed while containment child counts stayed flat.
    bbox_diag = ""
    if not bbox_df.empty:
        by_param = bbox_df.groupby("param_value").agg(
            median_raw_child_count=("median_raw_child_count", "median"),
            median_used_child_count=("median_used_child_count", "median"),
            empty_anchor_ratio=("empty_anchor_ratio", "median"),
        )
        if by_param["median_raw_child_count"].max() <= max(by_param["median_raw_child_count"].min() + 1e-6, 1.0):
            bbox_diag = (
                "bbox_expand 增大后 child count 几乎不变；结合 Step41/Step42 结果，"
                "更可能是 high coords 在当前 anchor bbox 覆盖内较稀疏，或 low anchor bbox 的空间语义不足以覆盖多个 high patches。"
            )
        else:
            bbox_diag = "bbox_expand 增大后 child count 有改善，说明 bbox 构造本身可调。"

    enter_text = "建议进入 Step43 HCRC-Light。" if recommendation.get("enter_step43") else "不建议直接进入 Step43。"
    risks = []
    if recommendation.get("recommended_strategy") == "bbox_then_nearest":
        risks.append("推荐策略依赖 nearest fallback，Step43 不能解释为严格 bbox containment 的空间子 patch。")
    rec_summary = strategy_df[strategy_df["is_recommended"] == True] if not strategy_df.empty else pd.DataFrame()
    if not rec_summary.empty:
        nearest = summary_df[
            (summary_df["strategy"] == recommendation.get("recommended_strategy"))
            & (summary_df["proposal_radius"] == recommendation.get("recommended_proposal_radius"))
            & (summary_df["nms_radius"] == recommendation.get("recommended_nms_radius"))
            & (summary_df["param_value"] == recommendation.get("recommended_param_value"))
        ]
        if not nearest.empty and nearest.iloc[0].get("used_child_distance_mean", 0) and nearest.iloc[0]["used_child_distance_mean"] > 8192:
            risks.append("推荐方案的平均 child 距离偏大，可能更接近 weak correspondence。")
    if not risks:
        risks.append("仍需在 Step43 smoke 中验证 child evidence 是否带来稳定增益，而不是只满足数量约束。")

    lines = [
        "# Step42b CGSP Child Coverage Sweep",
        "",
        "## Step42b 目的",
        "- 基于 Step42 的 CGSP anchor selection，系统 sweep `bbox_expand / proposal_radius / high child selection strategy`。",
        "- 判断是否能得到足够稳定的 high child patches；如果 bbox containment 不足，评估 nearest top-M fallback 是否可用于 Step43 HCRC-Light。",
        "- 不训练模型，不修改模型前向逻辑。",
        "",
        "## Step41/Step42 继承设置",
        f"- Step41 coord_mode: `{stage41_settings.get('coord_mode')}`",
        f"- Step41 scale_ratio: `{stage41_settings.get('scale_ratio')}`",
        f"- Step42 enter_step43: `{stage42_manifest.get('enter_step43')}`",
        f"- Step42 recommendation: `{stage42_manifest.get('recommendation_reason')}`",
        "",
        "## 输入 checkpoint / prompt source",
        f"- checkpoint source: `{prompt_source_info.get('source')}`",
        f"- checkpoint path: `{prompt_source_info.get('resolved_checkpoint_path')}`",
        f"- prompt path: `{prompt_source_info.get('prompt_path')}`",
        "",
        "## 处理概况",
        f"- requested slides: `{slide_source_info.get('selected_slide_count')}`",
        f"- processed slides: `{processed_count}`",
        f"- failed slides: `{len(failed_rows)}`",
        f"- warning count: `{len(warning_log)}`",
        f"- anchor-level CSV split: `{anchor_level_write_info.get('split')}`",
        f"- anchor-level CSV parts: `{len(anchor_level_write_info.get('parts', []))}`",
        "",
        "## proposal_radius / nms_radius sweep 结果",
        markdown_table(
            prep(
                stability_df,
                [
                    "proposal_radius",
                    "nms_radius",
                    "median_selected_anchors",
                    "full_anchor_slide_ratio",
                    "unique_classes",
                    "unique_concepts",
                    "anchor_pair_distance_mean_median",
                    "rankwise_anchor_drift_median_vs_min_radius",
                ],
            )
        ),
        "",
        "## bbox_expand sweep 结果",
        markdown_table(
            prep(
                bbox_df.sort_values(["proposal_radius", "nms_radius", "param_value"]).head(24),
                [
                    "proposal_radius",
                    "nms_radius",
                    "param_value",
                    "empty_anchor_ratio",
                    "median_raw_child_count",
                    "median_used_child_count",
                    "mean_used_child_count",
                ],
            )
        ),
        "",
        "## center_radius sweep 结果",
        markdown_table(
            prep(
                radius_df.sort_values(["proposal_radius", "nms_radius", "param_value"]).head(24),
                [
                    "proposal_radius",
                    "nms_radius",
                    "param_value",
                    "empty_anchor_ratio",
                    "median_raw_child_count",
                    "median_used_child_count",
                    "mean_used_child_count",
                    "used_child_distance_mean",
                ],
            )
        ),
        "",
        "## bbox_then_nearest fallback 结果",
        markdown_table(
            prep(
                fallback_df.sort_values(["proposal_radius", "nms_radius", "param_value"]).head(24),
                [
                    "proposal_radius",
                    "nms_radius",
                    "param_value",
                    "empty_anchor_ratio",
                    "median_raw_child_count",
                    "median_used_child_count",
                    "mean_fallback_ratio",
                    "used_child_distance_mean",
                ],
            )
        ),
        "",
        "## 为什么 Step42 的 bbox_expand 没有明显改善 child count",
        f"- {bbox_diag}",
        "",
        "## strategy 综合对比",
        markdown_table(
            prep(
                strategy_df,
                [
                    "strategy",
                    "best_proposal_radius",
                    "best_nms_radius",
                    "best_param_name",
                    "best_param_value",
                    "best_empty_anchor_ratio",
                    "best_median_used_child_count",
                    "best_mean_fallback_ratio",
                    "meets_child_threshold",
                    "is_recommended",
                ],
            )
        ),
        "",
        "## 推荐给 Step43 的参数",
        f"- strategy: `{recommendation.get('recommended_strategy')}`",
        f"- proposal_radius: `{recommendation.get('recommended_proposal_radius')}`",
        f"- nms_radius: `{recommendation.get('recommended_nms_radius')}`",
        f"- {recommendation.get('recommended_param_name')}: `{recommendation.get('recommended_param_value')}`",
        f"- num_anchors: `{recommendation.get('recommended_num_anchors')}`",
        f"- num_high_children: `{recommendation.get('recommended_num_high_children')}`",
        f"- use bbox_then_nearest fallback: `{recommendation.get('recommended_strategy') == 'bbox_then_nearest'}`",
        f"- decision: {enter_text}",
        f"- reason: {recommendation.get('reason')}",
        "",
        "## 风险说明",
    ]
    lines.extend([f"- {risk}" for risk in risks])
    if warning_log:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {msg}" for msg in warning_log[:80]])
    (output_dir / "stage42b_child_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_with_github_safe_split(
    df: pd.DataFrame,
    path: Path,
    *,
    max_bytes: int = 95 * 1024 * 1024,
) -> dict[str, object]:
    """Write a CSV, splitting it into GitHub-safe part files if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_bytes = path.stat().st_size
    if size_bytes <= max_bytes:
        return {
            "path": str(path),
            "size_bytes": int(size_bytes),
            "split": False,
            "parts": [],
            "row_count": int(len(df)),
            "max_bytes": int(max_bytes),
        }

    rows_per_part = max(1, int(len(df) * max_bytes / max(size_bytes, 1) * 0.90))
    parts = []
    part_index = 1
    for start in range(0, len(df), rows_per_part):
        part_path = path.with_name(f"{path.stem}_part_{part_index:03d}{path.suffix}")
        df.iloc[start : start + rows_per_part].to_csv(part_path, index=False)
        parts.append(
            {
                "path": str(part_path),
                "size_bytes": int(part_path.stat().st_size),
                "row_start": int(start),
                "row_end_exclusive": int(min(start + rows_per_part, len(df))),
            }
        )
        part_index += 1
    path.unlink()
    return {
        "path": str(path),
        "original_size_bytes": int(size_bytes),
        "split": True,
        "parts": parts,
        "row_count": int(len(df)),
        "rows_per_part": int(rows_per_part),
        "max_bytes": int(max_bytes),
        "note": "Original CSV was removed because it exceeded the GitHub 100MB file limit. Use parts in order.",
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warning_log: list[str] = []

    proposal_radius_values = parse_float_list(args.proposal_radius_values)
    nms_radius_values = parse_float_list(args.nms_radius_values)
    bbox_expand_values = parse_float_list(args.bbox_expand_values)
    high_radius_values = parse_float_list(args.high_radius_values)

    stage41_manifest = resolve_path(args.stage41_manifest)
    stage42_manifest_path = resolve_path(args.stage42_manifest)
    stage42_anchor_csv = resolve_path(args.stage42_anchor_csv)
    stage42_manifest = load_json_optional(stage42_manifest_path, warning_log)
    if not stage42_anchor_csv.is_file():
        warn(f"Step42 anchor CSV not found: {stage42_anchor_csv}; continuing with regenerated anchors only.", warning_log)

    stage41_settings = load_stage41_settings(stage41_manifest, args, warning_log)
    coord_mode = str(stage41_settings["coord_mode"])
    scale_ratio = float(stage41_settings["scale_ratio"])

    low_dir = resolve_path(Path(args.data_root_dir) / args.data_folder_s)
    high_dir = resolve_path(Path(args.data_root_dir) / args.data_folder_l)
    csv_path = resolve_path(args.csv_path)
    split_dir = resolve_path(args.split_dir)
    concept_prompt_path = resolve_path(args.concept_prompt_path)

    prompt_metadata = load_prompt_metadata(concept_prompt_path, warning_log)
    low_prompt_features, high_prompt_features, prompt_source_info = load_prompt_features(
        args.checkpoint_path,
        concept_prompt_path,
        warning_log,
    )
    if low_prompt_features is None:
        raise RuntimeError("Prompt features unavailable; Step42b requires checkpoint prompt buffers.")

    slide_ids, slide_source_info = load_slide_ids(csv_path, split_dir, args.fold, args.split, args.max_slides, warning_log)

    anchor_level_rows: list[dict[str, object]] = []
    anchor_records: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    processed_slide_ids: list[str] = []

    for slide_id in slide_ids:
        low_path = low_dir / f"{slide_id}.h5"
        high_path = high_dir / f"{slide_id}.h5"
        if not low_path.is_file() or not high_path.is_file():
            failed_rows.append({"slide_id": slide_id, "error": "missing low/high h5"})
            warn(f"Missing low/high h5 for slide {slide_id}", warning_log)
            continue
        try:
            low_features, low_coords, _ = read_h5_features_coords(low_path)
            _, high_coords, _ = read_h5_features_coords(high_path)
            score_info = compute_patch_scores(low_features, low_prompt_features, args.prompt_topk, args.margin_weight)
            low_centers = coords_to_centers(low_coords, args.low_patch_size, coord_mode) * scale_ratio
            high_centers = coords_to_centers(high_coords, args.high_patch_size, coord_mode)
            candidates = build_candidate_indices(
                sim=score_info["sim"],
                patch_score=score_info["patch_score"],
                candidate_top_l=args.candidate_top_l,
                top_g_concepts=args.top_g_concepts,
                per_concept_top_m=args.per_concept_top_m,
                prompt_topk=args.prompt_topk,
            )
            for proposal_radius in proposal_radius_values:
                proposal_args = argparse.Namespace(**vars(args))
                proposal_args.proposal_radius = proposal_radius
                proposal_rows, proposals = build_candidate_proposals(
                    slide_id=slide_id,
                    candidates=candidates,
                    low_coords=low_coords,
                    low_centers=low_centers,
                    low_features_norm=score_info["low_features_norm"],
                    score_info=score_info,
                    prompt_metadata=prompt_metadata,
                    args=proposal_args,
                    coord_mode=coord_mode,
                    scale_ratio=scale_ratio,
                )
                for proposal, proposal_row in zip(proposals, proposal_rows):
                    proposal["proposal_radius"] = float(proposal_radius)
                    proposal["proposal_class_name"] = get_class_name(prompt_metadata, int(proposal["proposal_class"]))
                for nms_radius in nms_radius_values:
                    selected = select_anchors_nms(proposals, args.num_anchors, nms_radius)
                    for anchor in selected:
                        anchor["proposal_radius"] = float(proposal_radius)
                        anchor["nms_radius"] = float(nms_radius)
                        record = {
                            "slide_id": slide_id,
                            "proposal_radius": float(proposal_radius),
                            "nms_radius": float(nms_radius),
                            "anchor_rank": int(anchor["anchor_rank"]),
                            "anchor_score": float(anchor["proposal_score"]),
                            "anchor_center_x": float(anchor["proposal_center"][0]),
                            "anchor_center_y": float(anchor["proposal_center"][1]),
                            "neighbor_count": int(anchor["neighbor_count"]),
                            "top_class_name": str(anchor["proposal_class_name"]),
                            "top_concept": str(anchor["proposal_concept"]),
                        }
                        anchor_records.append(record)
                        rows, pairs = get_child_stats_for_anchor(
                            anchor=anchor,
                            high_coords=high_coords,
                            high_centers=high_centers,
                            bbox_expand_values=bbox_expand_values,
                            high_radius_values=high_radius_values,
                            high_child_top_m=args.high_child_top_m,
                        )
                        anchor_level_rows.extend(rows)
                        if len(example_rows) < args.num_examples:
                            example_rows.extend(pairs[: max(0, args.num_examples - len(example_rows))])
            processed_slide_ids.append(slide_id)
        except Exception as exc:
            failed_rows.append({"slide_id": slide_id, "error": str(exc)})
            warn(f"Failed to process slide {slide_id}: {exc}", warning_log)

    anchor_stats_df = pd.DataFrame(anchor_level_rows)
    summary_df = summarize_sweep(anchor_stats_df)
    stability_df = summarize_anchor_stability(anchor_records, args.num_anchors)
    recommendation = choose_strategy(summary_df, stability_df, args.high_child_top_m, args.num_anchors)
    strategy_df = build_strategy_comparison(summary_df, recommendation)
    example_df = pd.DataFrame(example_rows)

    anchor_level_write_info = write_csv_with_github_safe_split(
        anchor_stats_df,
        output_dir / "stage42b_anchor_level_child_stats.csv",
    )
    summary_df.to_csv(output_dir / "stage42b_child_coverage_sweep_summary.csv", index=False)
    stability_df.to_csv(output_dir / "stage42b_anchor_stability_summary.csv", index=False)
    strategy_df.to_csv(output_dir / "stage42b_strategy_comparison.csv", index=False)
    example_df.to_csv(output_dir / "stage42b_example_anchor_child_pairs.csv", index=False)

    manifest = {
        "step": "Step42b CGSP Child Coverage Sweep",
        "status": "ok",
        "parameters": vars(args),
        "input_paths": {
            "low_dir": str(low_dir),
            "high_dir": str(high_dir),
            "csv_path": str(csv_path),
            "split_dir": str(split_dir),
            "stage41_manifest": str(stage41_manifest),
            "stage42_manifest": str(stage42_manifest_path),
            "stage42_anchor_csv": str(stage42_anchor_csv),
            "concept_prompt_path": str(concept_prompt_path),
            "checkpoint_path": str(resolve_path(args.checkpoint_path)),
        },
        "output_paths": {
            "summary_csv": str(output_dir / "stage42b_child_coverage_sweep_summary.csv"),
            "anchor_level_csv": str(output_dir / "stage42b_anchor_level_child_stats.csv"),
            "anchor_level_csv_parts": [part["path"] for part in anchor_level_write_info.get("parts", [])],
            "anchor_stability_csv": str(output_dir / "stage42b_anchor_stability_summary.csv"),
            "strategy_comparison_csv": str(output_dir / "stage42b_strategy_comparison.csv"),
            "example_pairs_csv": str(output_dir / "stage42b_example_anchor_child_pairs.csv"),
            "report_md": str(output_dir / "stage42b_child_coverage_report.md"),
            "manifest_json": str(output_dir / "stage42b_manifest.json"),
        },
        "stage41_settings": stage41_settings,
        "stage42_enter_step43": stage42_manifest.get("enter_step43"),
        "anchor_level_write_info": anchor_level_write_info,
        "prompt_source": prompt_source_info,
        "slide_source": slide_source_info,
        "processed_slide_count": len(processed_slide_ids),
        "failed_slide_count": len(failed_rows),
        "processed_slide_ids": processed_slide_ids,
        "failed_slides": failed_rows,
        "warning_count": len(warning_log),
        "warnings": warning_log,
        **recommendation,
    }
    write_report(
        output_dir=output_dir,
        args=args,
        stage41_settings=stage41_settings,
        stage42_manifest=stage42_manifest,
        prompt_source_info=prompt_source_info,
        slide_source_info=slide_source_info,
        summary_df=summary_df,
        stability_df=stability_df,
        strategy_df=strategy_df,
        recommendation=recommendation,
        processed_count=len(processed_slide_ids),
        failed_rows=failed_rows,
        warning_log=warning_log,
        anchor_level_write_info=anchor_level_write_info,
    )
    (output_dir / "stage42b_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Step42b CGSP child coverage sweep complete.")
    print(f"Processed slides: {len(processed_slide_ids)}")
    print(f"Failed slides: {len(failed_rows)}")
    print(f"Recommended strategy: {recommendation.get('recommended_strategy')}")
    print(f"Enter Step43: {recommendation.get('enter_step43')}")


if __name__ == "__main__":
    main()
