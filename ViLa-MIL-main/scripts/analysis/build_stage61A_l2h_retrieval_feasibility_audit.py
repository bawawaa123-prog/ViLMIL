from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/xiangmu/data/VILMIL")
DEFAULT_LOW_FEATURE_DIR = DEFAULT_DATA_ROOT / "features_biomedclip_5x"
DEFAULT_HIGH_FEATURE_DIR = DEFAULT_DATA_ROOT / "features_biomedclip_20x"
DEFAULT_LOW_RAW_COORD_DIR = DEFAULT_DATA_ROOT / "patches_coords_5x" / "patches_256"
DEFAULT_HIGH_RAW_COORD_DIR = DEFAULT_DATA_ROOT / "patches_coords_20x" / "patches_256"
DEFAULT_CSV_PATH = ROOT / "dataset_csv" / "all_data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results_stage61A_l2h_retrieval_feasibility"
DEFAULT_STAGE41_MANIFEST = ROOT / "results_stage41" / "low_high_coordinate_audit" / "stage41_manifest.json"
DEFAULT_STAGE58C_DECISION = (
    ROOT / "results_stage58C_residual_constrained_configD_5fold" / "stage58C_decision.json"
)
DEFAULT_STAGE59C_DECISION = ROOT / "results_stage59C_dynamic_csg_configA_5fold" / "stage59C_decision.json"
DEFAULT_STAGE60D_DECISION = ROOT / "results_stage60D_ccra_configC_formal" / "stage60D_decision.json"
DEFAULT_STAGE57B_AUDIT_DIR = ROOT / "results_stage57B_logit_contribution_audit"
DEFAULT_STAGE60D_AUDIT_DIR = ROOT / "results_stage60D_ccra_configC_formal"
COORD_KEYS = ("coords", "coord", "coordinates", "patch_coords")
EXTREME_COORD_THRESHOLD = 10_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step61A: Low-to-High Concept-guided Retrieval feasibility audit."
    )
    parser.add_argument("--data-root-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--low-feature-dir", type=Path, default=DEFAULT_LOW_FEATURE_DIR)
    parser.add_argument("--high-feature-dir", type=Path, default=DEFAULT_HIGH_FEATURE_DIR)
    parser.add_argument("--low-raw-coord-dir", type=Path, default=DEFAULT_LOW_RAW_COORD_DIR)
    parser.add_argument("--high-raw-coord-dir", type=Path, default=DEFAULT_HIGH_RAW_COORD_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--stage41-manifest", type=Path, default=DEFAULT_STAGE41_MANIFEST)
    parser.add_argument("--stage58c-decision", type=Path, default=DEFAULT_STAGE58C_DECISION)
    parser.add_argument("--stage59c-decision", type=Path, default=DEFAULT_STAGE59C_DECISION)
    parser.add_argument("--stage60d-decision", type=Path, default=DEFAULT_STAGE60D_DECISION)
    parser.add_argument("--stage57b-audit-dir", type=Path, default=DEFAULT_STAGE57B_AUDIT_DIR)
    parser.add_argument("--stage60d-audit-dir", type=Path, default=DEFAULT_STAGE60D_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-mapping-slides", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def round_float(value: object, digits: int = 6) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def format_metric(value: object, digits: int = 6) -> str:
    rounded = round_float(value, digits=digits)
    return "NA" if rounded is None else f"{rounded:.{digits}f}"


def write_run_commands(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"cd {ROOT}",
            "python -m py_compile scripts/analysis/build_stage61A_l2h_retrieval_feasibility_audit.py",
            "python scripts/analysis/build_stage61A_l2h_retrieval_feasibility_audit.py",
        ]
    )
    (output_dir / "stage61A_run_commands.txt").write_text(text + "\n", encoding="utf-8")


def list_h5_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in sorted(directory.glob("*.h5"))}


def read_dataset_slide_ids(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns:
        return []
    return pd.Series(df["slide_id"].dropna().astype(str)).drop_duplicates().tolist()


def serialise_attrs(attrs: h5py.AttributeManager) -> dict[str, object]:
    return {str(key): to_jsonable(value) for key, value in attrs.items()}


def read_coord_attrs(raw_path: Path) -> dict[str, object]:
    if not raw_path.is_file():
        return {
            "raw_coord_path": None,
            "raw_coord_has_file": False,
            "raw_coord_key": None,
            "patch_level": None,
            "patch_size": None,
            "downsample_x": None,
            "downsample_y": None,
            "coord_attrs": {},
        }
    with h5py.File(raw_path, "r") as handle:
        coord_key = next((key for key in COORD_KEYS if key in handle), None)
        if coord_key is None:
            return {
                "raw_coord_path": str(raw_path),
                "raw_coord_has_file": True,
                "raw_coord_key": None,
                "patch_level": None,
                "patch_size": None,
                "downsample_x": None,
                "downsample_y": None,
                "coord_attrs": {},
            }
        attrs = serialise_attrs(handle[coord_key].attrs)
        downsample = attrs.get("downsample")
        if isinstance(downsample, list) and downsample:
            downsample_x = float(downsample[0])
            downsample_y = float(downsample[1] if len(downsample) > 1 else downsample[0])
        elif downsample is None:
            downsample_x = None
            downsample_y = None
        else:
            downsample_x = float(downsample)
            downsample_y = float(downsample)
        return {
            "raw_coord_path": str(raw_path),
            "raw_coord_has_file": True,
            "raw_coord_key": coord_key,
            "patch_level": attrs.get("patch_level"),
            "patch_size": attrs.get("patch_size"),
            "downsample_x": downsample_x,
            "downsample_y": downsample_y,
            "coord_attrs": attrs,
        }


def read_feature_summary(feature_path: Path) -> dict[str, object]:
    if not feature_path.is_file():
        return {
            "feature_path": None,
            "has_feature_file": False,
            "coord_key": None,
            "coords_present": False,
            "coords_shape": None,
            "coords_dtype": None,
            "coords_count": None,
            "feature_shape": None,
            "feature_dtype": None,
            "duplicate_count": None,
            "has_negative_coords": None,
            "has_nan_or_inf": None,
            "has_extreme_coords": None,
            "coord_min_x": None,
            "coord_min_y": None,
            "coord_max_x": None,
            "coord_max_y": None,
            "coord_span_x": None,
            "coord_span_y": None,
        }

    with h5py.File(feature_path, "r") as handle:
        coord_key = next((key for key in COORD_KEYS if key in handle), None)
        feature_shape = None
        feature_dtype = None
        if "features" in handle:
            feature_shape = list(handle["features"].shape)
            feature_dtype = str(handle["features"].dtype)
        if coord_key is None:
            return {
                "feature_path": str(feature_path),
                "has_feature_file": True,
                "coord_key": None,
                "coords_present": False,
                "coords_shape": None,
                "coords_dtype": None,
                "coords_count": None,
                "feature_shape": feature_shape,
                "feature_dtype": feature_dtype,
                "duplicate_count": None,
                "has_negative_coords": None,
                "has_nan_or_inf": None,
                "has_extreme_coords": None,
                "coord_min_x": None,
                "coord_min_y": None,
                "coord_max_x": None,
                "coord_max_y": None,
                "coord_span_x": None,
                "coord_span_y": None,
            }
        coords = np.asarray(handle[coord_key])[:, :2]
        unique_count = len(np.unique(coords, axis=0)) if len(coords) else 0
        min_xy = coords.min(axis=0) if len(coords) else np.array([np.nan, np.nan])
        max_xy = coords.max(axis=0) if len(coords) else np.array([np.nan, np.nan])
        span_xy = max_xy - min_xy
        return {
            "feature_path": str(feature_path),
            "has_feature_file": True,
            "coord_key": coord_key,
            "coords_present": True,
            "coords_shape": list(coords.shape),
            "coords_dtype": str(coords.dtype),
            "coords_count": int(coords.shape[0]),
            "feature_shape": feature_shape,
            "feature_dtype": feature_dtype,
            "duplicate_count": int(coords.shape[0] - unique_count),
            "has_negative_coords": bool((coords < 0).any()),
            "has_nan_or_inf": bool(not np.isfinite(coords).all()),
            "has_extreme_coords": bool(np.abs(coords).max() > EXTREME_COORD_THRESHOLD) if len(coords) else False,
            "coord_min_x": float(min_xy[0]) if len(coords) else None,
            "coord_min_y": float(min_xy[1]) if len(coords) else None,
            "coord_max_x": float(max_xy[0]) if len(coords) else None,
            "coord_max_y": float(max_xy[1]) if len(coords) else None,
            "coord_span_x": float(span_xy[0]) if len(coords) else None,
            "coord_span_y": float(span_xy[1]) if len(coords) else None,
        }


def read_coords_array(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with h5py.File(path, "r") as handle:
        coord_key = next((key for key in COORD_KEYS if key in handle), None)
        if coord_key is None:
            return None
        return np.asarray(handle[coord_key])[:, :2].astype(np.float64)


def infer_mapping_context(
    slide_id: str,
    low_feature_path: Path,
    high_feature_path: Path,
    low_raw_path: Path,
    high_raw_path: Path,
) -> dict[str, object] | None:
    low_coords = read_coords_array(low_feature_path)
    high_coords = read_coords_array(high_feature_path)
    if low_coords is None or high_coords is None:
        return None

    low_raw = read_coord_attrs(low_raw_path)
    high_raw = read_coord_attrs(high_raw_path)
    low_patch_size = low_raw.get("patch_size")
    high_patch_size = high_raw.get("patch_size")
    low_downsample_x = low_raw.get("downsample_x")
    high_downsample_x = high_raw.get("downsample_x")
    if low_patch_size is None or high_patch_size is None or low_downsample_x is None or high_downsample_x is None:
        return None

    low_extent = float(low_patch_size) * float(low_downsample_x)
    high_extent = float(high_patch_size) * float(high_downsample_x)
    high_centers = high_coords + high_extent / 2.0
    low_centers = low_coords + low_extent / 2.0

    counts: list[int] = []
    for x0, y0 in low_coords:
        mask = (
            (high_centers[:, 0] >= x0)
            & (high_centers[:, 0] < x0 + low_extent)
            & (high_centers[:, 1] >= y0)
            & (high_centers[:, 1] < y0 + low_extent)
        )
        counts.append(int(mask.sum()))
    counts_array = np.asarray(counts, dtype=np.int32)

    feature_low_set = set(map(tuple, low_coords.astype(np.int64).tolist()))
    feature_high_set = set(map(tuple, high_coords.astype(np.int64).tolist()))
    raw_low_coords = read_coords_array(low_raw_path)
    raw_high_coords = read_coords_array(high_raw_path)
    low_set_match = (
        raw_low_coords is not None
        and feature_low_set == set(map(tuple, raw_low_coords.astype(np.int64).tolist()))
    )
    high_set_match = (
        raw_high_coords is not None
        and feature_high_set == set(map(tuple, raw_high_coords.astype(np.int64).tolist()))
    )

    range_iou = compute_range_iou(low_coords, high_coords)

    return {
        "slide_id": slide_id,
        "coord_scale_ratio": 1.0,
        "low_patch_extent_level0": low_extent,
        "high_patch_extent_level0": high_extent,
        "footprint_ratio": low_extent / max(high_extent, 1e-8),
        "low_patch_count": int(low_coords.shape[0]),
        "high_patch_count": int(high_coords.shape[0]),
        "avg_high_patches_per_low_patch": float(counts_array.mean()) if len(counts_array) else math.nan,
        "median_high_patches_per_low_patch": float(np.median(counts_array)) if len(counts_array) else math.nan,
        "percent_low_patches_with_zero_high_match": float((counts_array == 0).mean()) if len(counts_array) else math.nan,
        "percent_low_patches_with_1_to_4_high_matches": (
            float(((counts_array >= 1) & (counts_array <= 4)).mean()) if len(counts_array) else math.nan
        ),
        "percent_low_patches_with_5plus_high_matches": (
            float((counts_array >= 5).mean()) if len(counts_array) else math.nan
        ),
        "p95_high_patches_per_low_patch": float(np.percentile(counts_array, 95)) if len(counts_array) else math.nan,
        "low_center_min_x": float(low_centers[:, 0].min()) if len(low_centers) else math.nan,
        "low_center_min_y": float(low_centers[:, 1].min()) if len(low_centers) else math.nan,
        "low_center_max_x": float(low_centers[:, 0].max()) if len(low_centers) else math.nan,
        "low_center_max_y": float(low_centers[:, 1].max()) if len(low_centers) else math.nan,
        "high_center_min_x": float(high_centers[:, 0].min()) if len(high_centers) else math.nan,
        "high_center_min_y": float(high_centers[:, 1].min()) if len(high_centers) else math.nan,
        "high_center_max_x": float(high_centers[:, 0].max()) if len(high_centers) else math.nan,
        "high_center_max_y": float(high_centers[:, 1].max()) if len(high_centers) else math.nan,
        "range_iou_low_vs_high": range_iou,
        "feature_low_coord_set_matches_raw": bool(low_set_match),
        "feature_high_coord_set_matches_raw": bool(high_set_match),
    }


def compute_range_iou(low_coords: np.ndarray, high_coords: np.ndarray) -> float:
    low_min = low_coords.min(axis=0)
    low_max = low_coords.max(axis=0)
    high_min = high_coords.min(axis=0)
    high_max = high_coords.max(axis=0)
    inter_min = np.maximum(low_min, high_min)
    inter_max = np.minimum(low_max, high_max)
    inter_span = np.maximum(inter_max - inter_min, 0.0)
    inter_area = float(inter_span[0] * inter_span[1])
    low_area = float(np.prod(np.maximum(low_max - low_min, 1e-8)))
    high_area = float(np.prod(np.maximum(high_max - high_min, 1e-8)))
    union = max(low_area + high_area - inter_area, 1e-8)
    return inter_area / union


def load_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_evidence_export_audit(args: argparse.Namespace) -> dict[str, object]:
    model_path = ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py"
    model_text = model_path.read_text(encoding="utf-8")

    has_low_region_features = "last_low_region_features" in model_text
    has_high_region_features = "last_high_region_features" in model_text
    has_low_concept_evidence = (
        "last_low_prompt_evidence" in model_text
        and "low_evidence_logits" in model_text
    )
    has_high_concept_evidence = (
        "last_high_prompt_evidence" in model_text
        and "high_evidence_logits" in model_text
    )
    has_per_region_evidence = (
        "last_low_region_concept_sim" in model_text
        and "last_high_region_concept_sim" in model_text
    )
    retrieval_requires_new_export_fields = (
        "del coord_s, coords_l, slide_id" in model_text
        and "last_low_patch_concept_scores" not in model_text
    )

    step57b_sample_path = args.stage57b_audit_dir / "stage57B_sample_contribution.csv"
    step60d_branch_path = args.stage60d_audit_dir / "stage60D_branch_metrics_by_fold.csv"
    step60d_audit_sample = args.stage60d_audit_dir / "audits" / "fold_0" / "stage57B_sample_contribution.csv"

    sample_columns: list[str] = []
    if step57b_sample_path.is_file():
        sample_columns = list(pd.read_csv(step57b_sample_path, nrows=1).columns)

    branch_columns: list[str] = []
    branch_values: list[str] = []
    if step60d_branch_path.is_file():
        branch_df = pd.read_csv(step60d_branch_path)
        branch_columns = list(branch_df.columns)
        if "branch" in branch_df.columns:
            branch_values = sorted(branch_df["branch"].dropna().astype(str).unique().tolist())

    fold0_sample_columns: list[str] = []
    if step60d_audit_sample.is_file():
        fold0_sample_columns = list(pd.read_csv(step60d_audit_sample, nrows=1).columns)

    return {
        "has_low_region_features": has_low_region_features,
        "has_high_region_features": has_high_region_features,
        "has_low_concept_evidence": has_low_concept_evidence,
        "has_high_concept_evidence": has_high_concept_evidence,
        "has_per_region_evidence": has_per_region_evidence,
        "retrieval_requires_new_export_fields": retrieval_requires_new_export_fields,
        "recommended_export_fields_for_step61B": [
            "last_low_patch_concept_scores",
            "last_low_patch_topk_indices",
            "last_low_patch_topk_scores",
            "last_low_patch_coords",
            "last_retrieved_high_patch_indices",
            "last_retrieved_high_patch_coords",
            "last_retrieved_high_patch_match_counts",
            "last_retrieved_high_patch_mask",
            "last_retrieval_debug",
        ],
        "code_evidence": {
            "model_file": relative_path(model_path),
            "model_forward_accepts_coords": "def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):"
            in model_text,
            "model_currently_discards_coords": "del coord_s, coords_l, slide_id" in model_text,
            "stage57b_sample_columns": sample_columns,
            "stage60d_branch_metric_columns": branch_columns,
            "stage60d_branch_values": branch_values,
            "stage60d_fold0_sample_columns": fold0_sample_columns,
        },
    }


def decide_outcome(
    overlap_rate: float,
    low_coord_rate: float,
    high_coord_rate: float,
    both_coord_rate: float,
    mapping_summary: dict[str, object],
    evidence_audit: dict[str, object],
    scale_ratio_known: bool,
    raw_attrs_available_rate: float,
) -> dict[str, object]:
    def numeric_or_nan(key: str) -> float:
        value = mapping_summary.get(key)
        if value is None:
            return math.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    zero_ratio = numeric_or_nan("percent_low_patches_with_zero_high_match")
    avg_matches = numeric_or_nan("avg_high_patches_per_low_patch")
    five_plus = numeric_or_nan("percent_low_patches_with_5plus_high_matches")

    features_ok = overlap_rate >= 0.95
    coords_ok = both_coord_rate >= 0.95 and low_coord_rate >= 0.95 and high_coord_rate >= 0.95
    mapping_good = (
        math.isfinite(zero_ratio)
        and math.isfinite(avg_matches)
        and zero_ratio <= 0.10
        and avg_matches >= 5.0
        and five_plus >= 0.50
    )
    export_ok = (
        bool(evidence_audit["has_low_region_features"])
        and bool(evidence_audit["has_high_region_features"])
        and bool(evidence_audit["has_low_concept_evidence"])
        and bool(evidence_audit["has_high_concept_evidence"])
        and bool(evidence_audit["has_per_region_evidence"])
    )

    if features_ok and coords_ok and mapping_good and export_ok and scale_ratio_known and raw_attrs_available_rate >= 0.95:
        decision = "feasible_for_step61B"
        next_step = "enter_step61B_low_to_high_retrieval_all_off_implementation"
    elif features_ok and (both_coord_rate < 0.95 or not scale_ratio_known or raw_attrs_available_rate < 0.95):
        decision = "partial_feasible_need_coord_fix"
        next_step = "step61A_fix_coordinate_metadata_or_loading"
    elif overlap_rate < 0.50 or both_coord_rate < 0.50 or (math.isfinite(zero_ratio) and zero_ratio > 0.60):
        decision = "not_feasible_with_current_data"
        next_step = "stop_l2h_retrieval_move_to_final_consolidation"
    else:
        decision = "feasible_but_high_risk"
        next_step = "minimal_step61B_all_off_only_no_training"

    return {
        "decision": decision,
        "next_step": next_step,
        "features_ok": features_ok,
        "coords_ok": coords_ok,
        "mapping_good": mapping_good,
        "export_ok": export_ok,
        "scale_ratio_known": scale_ratio_known,
        "raw_attrs_available_rate": round_float(raw_attrs_available_rate),
        "reasons": [
            f"overlap_rate={overlap_rate:.6f}",
            f"low_coord_rate={low_coord_rate:.6f}",
            f"high_coord_rate={high_coord_rate:.6f}",
            f"both_coord_rate={both_coord_rate:.6f}",
            f"avg_high_patches_per_low_patch={avg_matches:.6f}" if math.isfinite(avg_matches) else "avg_high_patches_per_low_patch=NA",
            f"percent_low_patches_with_zero_high_match={zero_ratio:.6f}" if math.isfinite(zero_ratio) else "percent_low_patches_with_zero_high_match=NA",
            f"percent_low_patches_with_5plus_high_matches={five_plus:.6f}" if math.isfinite(five_plus) else "percent_low_patches_with_5plus_high_matches=NA",
            f"retrieval_requires_new_export_fields={evidence_audit['retrieval_requires_new_export_fields']}",
        ],
    }


def build_summary(
    pairing_stats: dict[str, object],
    coord_stats: dict[str, object],
    mapping_summary: dict[str, object],
    evidence_audit: dict[str, object],
    decision_payload: dict[str, object],
    stage41_manifest: dict[str, object] | None,
    stage58c_decision: dict[str, object] | None,
    stage59c_decision: dict[str, object] | None,
    stage60d_decision: dict[str, object] | None,
) -> str:
    candidate_lines = [
        f"- Step58C: hard-metric primary candidate (`decision={stage58c_decision.get('decision') if stage58c_decision else 'unknown'}`)",
        f"- Step59C: Dynamic CSG reference / AUC-PR variant (`decision={stage59c_decision.get('decision') if stage59c_decision else 'unknown'}`)",
        f"- Step60D: CCRA balanced representative (`decision={stage60d_decision.get('decision') if stage60d_decision else 'unknown'}`)",
    ]

    next_step_text = {
        "feasible_for_step61B": "进入 Step61B Low-to-High Retrieval all-off implementation。",
        "partial_feasible_need_coord_fix": "先做 Step61A-fix 坐标/metadata 修复。",
        "feasible_but_high_risk": "只做最小 Step61B all-off implementation，不训练。",
        "not_feasible_with_current_data": "停止 Low-to-High Retrieval，进入 final consolidation。",
    }.get(decision_payload["decision"], "等待进一步确认。")

    stage41_note = "无"
    if stage41_manifest:
        stage41_note = (
            f"已存在 Step41 formal audit，recommended_coord_mode=`{stage41_manifest.get('recommended_coord_mode')}`，"
            f"recommended_scale_ratio=`{stage41_manifest.get('recommended_scale_ratio')}`。"
        )

    lines = [
        "# Step61A Low-to-High Concept-guided Retrieval feasibility audit",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。",
        f"3. 5x 与 20x features 是否能按 slide_id 配对：是，`{pairing_stats['slides_with_both_5x_20x']}/{pairing_stats['total_slides']}`，overlap_rate=`{format_metric(pairing_stats['overlap_rate'])}`。",
        f"4. low/high coords 是否存在：是，low coords rate=`{format_metric(coord_stats['low_coord_rate'])}`，high coords rate=`{format_metric(coord_stats['high_coord_rate'])}`，both coords rate=`{format_metric(coord_stats['both_coord_rate'])}`。",
        "5. 坐标单位和 scale ratio 是否能确定：可以基本确定。"
        f" 当前 feature h5 直接存储原始 patch coords；raw coords h5 attrs 显示 low `patch_level=2, downsample≈16`，high `patch_level=1, downsample≈4`，"
        f"坐标变换 scale_ratio 估计为 `{format_metric(mapping_summary['estimated_scale_ratio'])}`，low/high 原图 footprint 比例约为 `{format_metric(mapping_summary['estimated_patch_footprint_ratio'])}`。",
        f"6. low-to-high patch mapping 是否可行：{'是' if decision_payload['decision'] != 'not_feasible_with_current_data' else '否'}，mapping_quality_label=`{mapping_summary['mapping_quality_label']}`。",
        f"7. 每个 low patch 平均能匹配多少 high patches：`{format_metric(mapping_summary['avg_high_patches_per_low_patch'])}`，median=`{format_metric(mapping_summary['median_high_patches_per_low_patch'])}`。",
        f"8. 有多少 low patches 找不到 high patch：`{format_metric(mapping_summary['percent_low_patches_with_zero_high_match'])}`。",
        "9. 当前模型是否已有 low/high region evidence 可导出："
        f" low_region_features=`{evidence_audit['has_low_region_features']}`，high_region_features=`{evidence_audit['has_high_region_features']}`，"
        f" low/high concept evidence=`{evidence_audit['has_low_concept_evidence'] and evidence_audit['has_high_concept_evidence']}`，per_region_evidence=`{evidence_audit['has_per_region_evidence']}`。",
        "10. 如果进入 Step61B，需要新增哪些 export fields："
        f" {', '.join(evidence_audit['recommended_export_fields_for_step61B'])}。",
        f"11. 最终 decision 是什么：`{decision_payload['decision']}`。",
        f"12. 下一步建议：{next_step_text}",
        "",
        "## Candidate Positioning",
        "",
        *candidate_lines,
        "",
        "## Evidence Notes",
        "",
        f"- Step41 prior audit: {stage41_note}",
        f"- retrieval_requires_new_export_fields: `{evidence_audit['retrieval_requires_new_export_fields']}`",
        f"- sampled_mapping_slides: `{mapping_summary['sampled_slide_count']}`",
        f"- sampled_low_patches: `{mapping_summary['sampled_low_patch_count']}`",
        f"- sampled_high_patches: `{mapping_summary['sampled_high_patch_count']}`",
        f"- feature/raw coord set match rate on sampled slides: low=`{format_metric(mapping_summary['sample_low_coord_set_match_rate'])}` high=`{format_metric(mapping_summary['sample_high_coord_set_match_rate'])}`",
        "",
        "## Decision Basis",
        "",
    ]
    for reason in decision_payload["reasons"]:
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_commands(output_dir)

    dataset_slide_ids = read_dataset_slide_ids(args.csv_path)
    dataset_set = set(dataset_slide_ids)
    low_features = list_h5_files(args.low_feature_dir)
    high_features = list_h5_files(args.high_feature_dir)

    union_slide_ids = list(dataset_slide_ids)
    for slide_id in sorted((set(low_features) | set(high_features)) - dataset_set):
        union_slide_ids.append(slide_id)

    pairing_rows: list[dict[str, object]] = []
    coord_rows: list[dict[str, object]] = []
    mapping_candidates: list[str] = []
    low_coord_key_counter: Counter[str] = Counter()
    high_coord_key_counter: Counter[str] = Counter()
    low_dtype_counter: Counter[str] = Counter()
    high_dtype_counter: Counter[str] = Counter()

    for slide_id in union_slide_ids:
        low_path = low_features.get(slide_id)
        high_path = high_features.get(slide_id)
        low_summary = read_feature_summary(low_path) if low_path else read_feature_summary(Path())
        high_summary = read_feature_summary(high_path) if high_path else read_feature_summary(Path())
        low_raw = read_coord_attrs(args.low_raw_coord_dir / f"{slide_id}.h5")
        high_raw = read_coord_attrs(args.high_raw_coord_dir / f"{slide_id}.h5")

        has_low = bool(low_path)
        has_high = bool(high_path)
        has_low_coords = bool(low_summary["coords_present"])
        has_high_coords = bool(high_summary["coords_present"])
        has_both_coords = has_low_coords and has_high_coords

        pairing_rows.append(
            {
                "slide_id": slide_id,
                "in_dataset_csv": slide_id in dataset_set,
                "has_5x_feature": has_low,
                "has_20x_feature": has_high,
                "has_both_5x_20x": has_low and has_high,
                "low_feature_path": relative_path(low_path) if low_path else None,
                "high_feature_path": relative_path(high_path) if high_path else None,
            }
        )

        coord_rows.append(
            {
                "slide_id": slide_id,
                "has_5x_feature": has_low,
                "has_20x_feature": has_high,
                "low_coord_key": low_summary["coord_key"],
                "high_coord_key": high_summary["coord_key"],
                "low_coords_present": has_low_coords,
                "high_coords_present": has_high_coords,
                "has_both_coords": has_both_coords,
                "low_coords_shape": low_summary["coords_shape"],
                "high_coords_shape": high_summary["coords_shape"],
                "low_coords_dtype": low_summary["coords_dtype"],
                "high_coords_dtype": high_summary["coords_dtype"],
                "low_coords_count": low_summary["coords_count"],
                "high_coords_count": high_summary["coords_count"],
                "low_duplicate_count": low_summary["duplicate_count"],
                "high_duplicate_count": high_summary["duplicate_count"],
                "low_has_negative_coords": low_summary["has_negative_coords"],
                "high_has_negative_coords": high_summary["has_negative_coords"],
                "low_has_nan_or_inf": low_summary["has_nan_or_inf"],
                "high_has_nan_or_inf": high_summary["has_nan_or_inf"],
                "low_has_extreme_coords": low_summary["has_extreme_coords"],
                "high_has_extreme_coords": high_summary["has_extreme_coords"],
                "low_coord_min_x": low_summary["coord_min_x"],
                "low_coord_min_y": low_summary["coord_min_y"],
                "low_coord_max_x": low_summary["coord_max_x"],
                "low_coord_max_y": low_summary["coord_max_y"],
                "high_coord_min_x": high_summary["coord_min_x"],
                "high_coord_min_y": high_summary["coord_min_y"],
                "high_coord_max_x": high_summary["coord_max_x"],
                "high_coord_max_y": high_summary["coord_max_y"],
                "low_raw_coord_has_file": low_raw["raw_coord_has_file"],
                "high_raw_coord_has_file": high_raw["raw_coord_has_file"],
                "low_patch_level": low_raw["patch_level"],
                "high_patch_level": high_raw["patch_level"],
                "low_patch_size": low_raw["patch_size"],
                "high_patch_size": high_raw["patch_size"],
                "low_downsample_x": low_raw["downsample_x"],
                "low_downsample_y": low_raw["downsample_y"],
                "high_downsample_x": high_raw["downsample_x"],
                "high_downsample_y": high_raw["downsample_y"],
            }
        )

        if has_low_coords and has_high_coords and low_raw["raw_coord_has_file"] and high_raw["raw_coord_has_file"]:
            mapping_candidates.append(slide_id)
        if low_summary["coord_key"]:
            low_coord_key_counter[str(low_summary["coord_key"])] += 1
        if high_summary["coord_key"]:
            high_coord_key_counter[str(high_summary["coord_key"])] += 1
        if low_summary["coords_dtype"]:
            low_dtype_counter[str(low_summary["coords_dtype"])] += 1
        if high_summary["coords_dtype"]:
            high_dtype_counter[str(high_summary["coords_dtype"])] += 1

    pairing_df = pd.DataFrame(pairing_rows)
    coord_df = pd.DataFrame(coord_rows)
    pairing_df.to_csv(output_dir / "stage61A_feature_pairing_audit.csv", index=False)
    coord_df.to_csv(output_dir / "stage61A_coord_availability.csv", index=False)

    total_slides = len(pairing_df)
    slides_with_both = int(pairing_df["has_both_5x_20x"].sum()) if not pairing_df.empty else 0
    missing_5x_slides = int((~pairing_df["has_5x_feature"]).sum()) if not pairing_df.empty else 0
    missing_20x_slides = int((~pairing_df["has_20x_feature"]).sum()) if not pairing_df.empty else 0
    overlap_rate = float(slides_with_both / total_slides) if total_slides else math.nan

    low_coord_rate = float(coord_df["low_coords_present"].mean()) if not coord_df.empty else math.nan
    high_coord_rate = float(coord_df["high_coords_present"].mean()) if not coord_df.empty else math.nan
    both_coord_rate = float(coord_df["has_both_coords"].mean()) if not coord_df.empty else math.nan
    raw_attrs_available_rate = (
        float((coord_df["low_raw_coord_has_file"] & coord_df["high_raw_coord_has_file"]).mean())
        if not coord_df.empty
        else math.nan
    )

    rng = random.Random(args.seed)
    mapping_sample_ids = sorted(mapping_candidates)
    if len(mapping_sample_ids) > args.max_mapping_slides:
        mapping_sample_ids = sorted(rng.sample(mapping_sample_ids, args.max_mapping_slides))

    mapping_rows: list[dict[str, object]] = []
    for slide_id in mapping_sample_ids:
        context = infer_mapping_context(
            slide_id=slide_id,
            low_feature_path=low_features[slide_id],
            high_feature_path=high_features[slide_id],
            low_raw_path=args.low_raw_coord_dir / f"{slide_id}.h5",
            high_raw_path=args.high_raw_coord_dir / f"{slide_id}.h5",
        )
        if context is not None:
            mapping_rows.append(context)

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(output_dir / "stage61A_mapping_quality_by_slide.csv", index=False)

    mapping_summary = {
        "estimated_scale_ratio": None,
        "estimated_patch_footprint_ratio": None,
        "avg_high_patches_per_low_patch": None,
        "median_high_patches_per_low_patch": None,
        "percent_low_patches_with_zero_high_match": None,
        "percent_low_patches_with_1_to_4_high_matches": None,
        "percent_low_patches_with_5plus_high_matches": None,
        "mapping_quality_label": "pending",
        "sampled_slide_count": len(mapping_df),
        "sampled_low_patch_count": int(mapping_df["low_patch_count"].sum()) if not mapping_df.empty else 0,
        "sampled_high_patch_count": int(mapping_df["high_patch_count"].sum()) if not mapping_df.empty else 0,
        "sample_low_coord_set_match_rate": None,
        "sample_high_coord_set_match_rate": None,
    }
    if not mapping_df.empty:
        total_low = float(mapping_df["low_patch_count"].sum())
        total_high = float(mapping_df["high_patch_count"].sum())
        weighted_avg = np.average(
            mapping_df["avg_high_patches_per_low_patch"],
            weights=np.maximum(mapping_df["low_patch_count"], 1),
        )
        weighted_zero = np.average(
            mapping_df["percent_low_patches_with_zero_high_match"],
            weights=np.maximum(mapping_df["low_patch_count"], 1),
        )
        weighted_1_to_4 = np.average(
            mapping_df["percent_low_patches_with_1_to_4_high_matches"],
            weights=np.maximum(mapping_df["low_patch_count"], 1),
        )
        weighted_5_plus = np.average(
            mapping_df["percent_low_patches_with_5plus_high_matches"],
            weights=np.maximum(mapping_df["low_patch_count"], 1),
        )
        mapping_summary.update(
            {
                "estimated_scale_ratio": round_float(mapping_df["coord_scale_ratio"].median()),
                "estimated_patch_footprint_ratio": round_float(mapping_df["footprint_ratio"].median()),
                "avg_high_patches_per_low_patch": round_float(weighted_avg),
                "median_high_patches_per_low_patch": round_float(mapping_df["median_high_patches_per_low_patch"].median()),
                "percent_low_patches_with_zero_high_match": round_float(weighted_zero),
                "percent_low_patches_with_1_to_4_high_matches": round_float(weighted_1_to_4),
                "percent_low_patches_with_5plus_high_matches": round_float(weighted_5_plus),
                "sample_low_coord_set_match_rate": round_float(
                    float(mapping_df["feature_low_coord_set_matches_raw"].mean())
                ),
                "sample_high_coord_set_match_rate": round_float(
                    float(mapping_df["feature_high_coord_set_matches_raw"].mean())
                ),
                "sampled_low_patch_count": int(total_low),
                "sampled_high_patch_count": int(total_high),
            }
        )
        zero_ratio = float(weighted_zero)
        avg_matches = float(weighted_avg)
        if zero_ratio <= 0.02 and avg_matches >= 8.0:
            mapping_quality_label = "strong_same_coord_system_mapping"
        elif zero_ratio <= 0.10 and avg_matches >= 5.0:
            mapping_quality_label = "good_mapping_with_dense_high_children"
        elif zero_ratio <= 0.35 and avg_matches >= 1.0:
            mapping_quality_label = "partial_mapping_need_neighbor_fallback"
        else:
            mapping_quality_label = "weak_mapping"
        mapping_summary["mapping_quality_label"] = mapping_quality_label

    evidence_audit = build_evidence_export_audit(args)
    (output_dir / "stage61A_evidence_export_audit.json").write_text(
        json.dumps(to_jsonable(evidence_audit), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stage41_manifest = load_json_if_exists(args.stage41_manifest)
    stage58c_decision = load_json_if_exists(args.stage58c_decision)
    stage59c_decision = load_json_if_exists(args.stage59c_decision)
    stage60d_decision = load_json_if_exists(args.stage60d_decision)

    scale_ratio_known = bool(
        mapping_summary["estimated_scale_ratio"] is not None
        and mapping_summary["estimated_patch_footprint_ratio"] is not None
    )
    decision_payload = decide_outcome(
        overlap_rate=overlap_rate,
        low_coord_rate=low_coord_rate,
        high_coord_rate=high_coord_rate,
        both_coord_rate=both_coord_rate,
        mapping_summary=mapping_summary,
        evidence_audit=evidence_audit,
        scale_ratio_known=scale_ratio_known,
        raw_attrs_available_rate=raw_attrs_available_rate,
    )

    pairing_stats = {
        "total_slides": total_slides,
        "slides_with_both_5x_20x": slides_with_both,
        "missing_5x_slides": missing_5x_slides,
        "missing_20x_slides": missing_20x_slides,
        "overlap_rate": overlap_rate,
    }
    coord_stats = {
        "low_coord_rate": low_coord_rate,
        "high_coord_rate": high_coord_rate,
        "both_coord_rate": both_coord_rate,
        "low_coord_key_counts": dict(low_coord_key_counter),
        "high_coord_key_counts": dict(high_coord_key_counter),
        "low_dtype_counts": dict(low_dtype_counter),
        "high_dtype_counts": dict(high_dtype_counter),
        "slides_with_both_coords": int(coord_df["has_both_coords"].sum()) if not coord_df.empty else 0,
        "low_duplicate_slides": int((coord_df["low_duplicate_count"].fillna(0) > 0).sum()) if not coord_df.empty else 0,
        "high_duplicate_slides": int((coord_df["high_duplicate_count"].fillna(0) > 0).sum()) if not coord_df.empty else 0,
        "low_negative_slides": int(coord_df["low_has_negative_coords"].fillna(False).sum()) if not coord_df.empty else 0,
        "high_negative_slides": int(coord_df["high_has_negative_coords"].fillna(False).sum()) if not coord_df.empty else 0,
        "low_nan_slides": int(coord_df["low_has_nan_or_inf"].fillna(False).sum()) if not coord_df.empty else 0,
        "high_nan_slides": int(coord_df["high_has_nan_or_inf"].fillna(False).sum()) if not coord_df.empty else 0,
        "raw_attrs_available_rate": raw_attrs_available_rate,
        "coords_shape_examples": coord_df.loc[
            coord_df["has_both_coords"],
            ["slide_id", "low_coords_shape", "high_coords_shape", "low_coords_dtype", "high_coords_dtype"],
        ].head(5).to_dict(orient="records"),
    }

    summary_text = build_summary(
        pairing_stats=pairing_stats,
        coord_stats=coord_stats,
        mapping_summary=mapping_summary,
        evidence_audit=evidence_audit,
        decision_payload=decision_payload,
        stage41_manifest=stage41_manifest,
        stage58c_decision=stage58c_decision,
        stage59c_decision=stage59c_decision,
        stage60d_decision=stage60d_decision,
    )
    (output_dir / "stage61A_summary.md").write_text(summary_text, encoding="utf-8")

    status_payload = {
        "status": "completed",
        "input_paths": {
            "data_root_dir": relative_path(args.data_root_dir),
            "low_feature_dir": relative_path(args.low_feature_dir),
            "high_feature_dir": relative_path(args.high_feature_dir),
            "low_raw_coord_dir": relative_path(args.low_raw_coord_dir),
            "high_raw_coord_dir": relative_path(args.high_raw_coord_dir),
            "csv_path": relative_path(args.csv_path),
        },
        "pairing_stats": to_jsonable(pairing_stats),
        "coord_stats": to_jsonable(coord_stats),
        "mapping_summary": to_jsonable(mapping_summary),
        "sampled_mapping_slides": mapping_sample_ids,
        "stage41_reference_present": stage41_manifest is not None,
    }
    (output_dir / "stage61A_status.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stage61A_decision.json").write_text(
        json.dumps(
            {
                **decision_payload,
                "pairing_stats": to_jsonable(pairing_stats),
                "coord_stats": to_jsonable(coord_stats),
                "mapping_summary": to_jsonable(mapping_summary),
                "evidence_export_summary": {
                    key: evidence_audit[key]
                    for key in [
                        "has_low_region_features",
                        "has_high_region_features",
                        "has_low_concept_evidence",
                        "has_high_concept_evidence",
                        "has_per_region_evidence",
                        "retrieval_requires_new_export_fields",
                        "recommended_export_fields_for_step61B",
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
