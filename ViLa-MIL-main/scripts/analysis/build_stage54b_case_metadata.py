from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_stage54_rce_evidence_interpretability"

FULL_RUN_DIR = ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1"
WO_CSG_RUN_DIR = ROOT / "results_stage52_rce_core_ablation" / "wo_csg_5fold_e20_s1"
FULL_DIRECT_DIR = RESULTS_DIR / "full"
WO_CSG_DIRECT_DIR = RESULTS_DIR / "wo_csg"
FULL_STAGE32_FALLBACK_DIR = ROOT / "results_stage32" / "stage32_rce_v4_csg_evidence_export"

LABEL_NAMES = {0: "Adenocarcinoma", 1: "NonAdenocarcinoma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step54B case-level evidence metadata.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def safe_read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_path_text(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\\", "/").rstrip("/")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def path_matches(manifest_path: str | None, expected_path: Path) -> bool:
    expected = normalize_path_text(str(expected_path))
    actual = normalize_path_text(manifest_path)
    return bool(actual) and (actual == expected or actual.endswith(rel(expected_path)))


def resolve_full_source() -> tuple[Path, str, bool]:
    direct_manifest = safe_read_json(FULL_DIRECT_DIR / "stage32_manifest.json")
    direct_slide = safe_read_csv(FULL_DIRECT_DIR / "stage32_slide_evidence_summary.csv")
    if direct_manifest is not None and direct_slide is not None:
        if path_matches(str(direct_manifest.get("results_dir", "")), FULL_RUN_DIR) and path_matches(
            str(direct_manifest.get("ckpt_path", "")), FULL_RUN_DIR / "s_0_checkpoint.pt"
        ):
            return FULL_DIRECT_DIR, "stage54_full_direct_export", False
    return FULL_STAGE32_FALLBACK_DIR, "stage32_legacy_deg_export_fallback", True


def resolve_wo_csg_source() -> tuple[Path, str]:
    return WO_CSG_DIRECT_DIR, "stage54_wo_csg_direct_export"


def add_confidence_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_confidence"] = out[["prob_class_0", "prob_class_1"]].max(axis=1)
    out[f"{prefix}_pred_name"] = out["pred"].map(LABEL_NAMES)
    out[f"{prefix}_correct"] = out["correct"].astype(bool)
    rename_map = {
        "pred": f"{prefix}_pred",
        "label": "true_label",
        "split": "split",
        "fold": "fold",
        "slide_id": "slide_id",
    }
    out = out.rename(columns=rename_map)
    keep_cols = [
        "slide_id",
        "fold",
        "split",
        "true_label",
        f"{prefix}_pred",
        f"{prefix}_pred_name",
        f"{prefix}_confidence",
        f"{prefix}_correct",
    ]
    return out[keep_cols].copy()


def classify_case(row: pd.Series) -> str:
    same_prediction_confidence_shift = (
        row["full_pred"] == row["wo_csg_pred"] and abs(row["confidence_delta"]) >= 0.10
    )
    if row["is_full_correct"] and not row["is_wo_csg_correct"]:
        return "full_correct_wo_csg_wrong"
    if not row["is_full_correct"]:
        return "full_wrong"
    if same_prediction_confidence_shift:
        return "same_prediction_confidence_shift"
    if row["is_full_correct"]:
        return "full_correct"
    return "other"


def recommend_figure_use(row: pd.Series) -> str:
    if row["case_type"] == "full_correct_wo_csg_wrong":
        return (
            "supplementary_comparison_due_source_difference"
            if row["uses_stage32_fallback"]
            else "supplementary_matched_direct_export_comparison"
        )
    if row["case_type"] == "same_prediction_confidence_shift":
        return (
            "supplementary_comparison_due_source_difference"
            if row["uses_stage32_fallback"]
            else "supplementary_matched_direct_export_comparison"
        )
    if row["case_type"] == "full_wrong":
        return "supplementary_single_case"
    if row["case_type"] == "full_correct" and row["full_confidence"] >= 0.99:
        return (
            "main_text_single_case_with_fallback_disclosure"
            if row["uses_stage32_fallback"]
            else "main_text_single_case_direct_export"
        )
    if row["case_type"] == "full_correct":
        return "supplementary_single_case"
    return "not_recommended"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    full_source_dir, full_source_label, uses_stage32_fallback = resolve_full_source()
    wo_source_dir, wo_source_label = resolve_wo_csg_source()

    full_df = safe_read_csv(full_source_dir / "stage32_slide_evidence_summary.csv")
    wo_df = safe_read_csv(wo_source_dir / "stage32_slide_evidence_summary.csv")
    if full_df is None or wo_df is None:
        raise FileNotFoundError("Required slide evidence summaries are missing for Step54B case metadata.")

    full_base = add_confidence_columns(full_df, "full")
    wo_base = add_confidence_columns(wo_df, "wo_csg")
    merged = full_base.merge(wo_base, on=["slide_id", "fold", "split", "true_label"], how="inner")

    merged["case_id"] = merged.apply(lambda row: f"fold{int(row['fold'])}_{row['slide_id']}", axis=1)
    merged["true_label_name"] = merged["true_label"].map(LABEL_NAMES)
    merged["is_full_correct"] = merged["full_correct"].astype(bool)
    merged["is_wo_csg_correct"] = merged["wo_csg_correct"].astype(bool)
    merged["confidence_delta"] = merged["full_confidence"] - merged["wo_csg_confidence"]
    merged["evidence_source_full"] = full_source_label
    merged["evidence_source_wo_csg"] = wo_source_label
    merged["uses_stage32_fallback"] = uses_stage32_fallback
    merged["case_type"] = merged.apply(classify_case, axis=1)
    merged["recommended_figure_use"] = merged.apply(recommend_figure_use, axis=1)
    merged["source_difference_note"] = (
        "full_uses_stage32_legacy_fallback_vs_wo_csg_direct_export"
        if uses_stage32_fallback
        else "same_step54_export_family"
    )

    priority_map = {
        "full_correct_wo_csg_wrong": 0,
        "same_prediction_confidence_shift": 1,
        "full_wrong": 2,
        "full_correct": 3,
        "other": 4,
    }
    merged["_priority"] = merged["case_type"].map(priority_map).fillna(99)
    merged = merged.sort_values(
        by=["_priority", "confidence_delta", "full_confidence", "slide_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    merged["case_rank_within_type"] = merged.groupby("case_type").cumcount() + 1

    output_columns = [
        "case_id",
        "slide_id",
        "fold",
        "split",
        "true_label",
        "true_label_name",
        "full_pred",
        "full_pred_name",
        "full_confidence",
        "wo_csg_pred",
        "wo_csg_pred_name",
        "wo_csg_confidence",
        "case_type",
        "is_full_correct",
        "is_wo_csg_correct",
        "confidence_delta",
        "evidence_source_full",
        "evidence_source_wo_csg",
        "uses_stage32_fallback",
        "recommended_figure_use",
        "source_difference_note",
        "case_rank_within_type",
    ]
    output_df = merged[output_columns].copy()
    output_df.to_csv(args.results_dir / "stage54b_case_level_metadata.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
