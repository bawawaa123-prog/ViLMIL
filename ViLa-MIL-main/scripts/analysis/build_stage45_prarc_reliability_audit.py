from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE32_DIR = "results_stage32/stage32_rce_v4_csg_evidence_export"
DEFAULT_STAGE33_DIR = "results_stage33/stage33_evidence_failure_analysis"
DEFAULT_STAGE39_DIR = "results_stage39/final_evidence_package"
DEFAULT_STAGE44_DIR = "results_stage44/stage44_hcrc_light_summary"
DEFAULT_OUTPUT_DIR = "results_stage45/prarc_reliability_audit"

NUMERIC_COLUMNS = [
    "label",
    "pred",
    "prob_class_0",
    "prob_class_1",
    "final_logit_class_0",
    "final_logit_class_1",
    "pred_margin",
    "low_logit_class_0",
    "low_logit_class_1",
    "high_logit_class_0",
    "high_logit_class_1",
    "visual_logit_class_0",
    "visual_logit_class_1",
    "visual_alpha",
    "csg_logit_class_0",
    "csg_logit_class_1",
    "csg_alpha",
    "final_true_vs_wrong_margin",
    "low_true_vs_wrong_margin",
    "high_true_vs_wrong_margin",
    "visual_true_vs_wrong_margin",
    "visual_effective_margin",
    "csg_true_vs_wrong_margin",
    "csg_effective_margin",
    "concept_margin_mean",
    "concept_abs_margin_max",
    "low_source_ratio",
    "high_source_ratio",
    "visual_source_ratio",
    "csg_source_ratio",
    "dominant_source_ratio",
    "prompt_confusion_score",
    "top_csg_pair_score_class_0",
    "top_csg_pair_score_class_1",
    "csg_max_pair_score",
]

BOOL_COLUMNS = [
    "correct",
    "low_available",
    "high_available",
    "low_supports_true",
    "high_supports_true",
    "low_high_same_support",
    "low_high_conflict",
    "csg_pair_near_zero",
    "csg_margin_near_zero",
    "csg_logits_near_zero",
    "csg_inactive_or_zero_flag",
    "visual_override_candidate",
    "prompt_confusion_flag",
]

FEATURE_COLUMNS = [
    "concept_margin_abs",
    "low_margin_abs",
    "high_margin_abs",
    "low_high_margin_agreement",
    "low_high_sign_agreement",
    "visual_concept_conflict",
    "visual_margin_abs",
    "visual_over_concept_ratio",
    "dominant_source_ratio",
    "prompt_error_risk_score",
    "prompt_reliability_score",
    "csg_abs_margin",
    "prediction_confidence_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step45 PRARC reliability audit from existing stage outputs.")
    parser.add_argument("--stage32_dir", default=DEFAULT_STAGE32_DIR)
    parser.add_argument("--stage33_dir", default=DEFAULT_STAGE33_DIR)
    parser.add_argument("--stage39_dir", default=DEFAULT_STAGE39_DIR)
    parser.add_argument("--stage44_dir", default=DEFAULT_STAGE44_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_prompt_occurrences", type=int, default=3)
    parser.add_argument("--visual_override_threshold", type=float, default=0.5)
    parser.add_argument("--low_margin_threshold", type=float, default=0.05)
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


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def drop_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    column_names = list(df.columns.astype(str))
    row_match_ratio = (
        df.astype(str)
        .apply(lambda row: np.mean([str(row.iloc[i]) == column_names[i] for i in range(len(column_names))]), axis=1)
    )
    return df.loc[row_match_ratio < 0.8].copy()


def safe_read_csv(path: Path, warning_log: list[str], files_read: dict[str, dict[str, object]]) -> pd.DataFrame:
    if not path.is_file():
        warn_message(f"Missing optional CSV: {path}", warning_log)
        files_read[relative_path(path)] = {"status": "missing", "type": "csv"}
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df = drop_repeated_header_rows(df)
        files_read[relative_path(path)] = {
            "status": "loaded",
            "type": "csv",
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        return df
    except Exception as exc:
        warn_message(f"Failed to read CSV {path}: {exc}", warning_log)
        files_read[relative_path(path)] = {"status": "error", "type": "csv", "error": str(exc)}
        return pd.DataFrame()


def safe_read_json(path: Path, warning_log: list[str], files_read: dict[str, dict[str, object]]) -> dict[str, object]:
    if not path.is_file():
        warn_message(f"Missing optional JSON: {path}", warning_log)
        files_read[relative_path(path)] = {"status": "missing", "type": "json"}
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files_read[relative_path(path)] = {"status": "loaded", "type": "json", "keys": sorted(data.keys())}
        return data
    except Exception as exc:
        warn_message(f"Failed to read JSON {path}: {exc}", warning_log)
        files_read[relative_path(path)] = {"status": "error", "type": "json", "error": str(exc)}
        return {}


def safe_read_text(path: Path, warning_log: list[str], files_read: dict[str, dict[str, object]]) -> str:
    if not path.is_file():
        warn_message(f"Missing optional text file: {path}", warning_log)
        files_read[relative_path(path)] = {"status": "missing", "type": "text"}
        return ""
    try:
        text = path.read_text(encoding="utf-8")
        files_read[relative_path(path)] = {"status": "loaded", "type": "text", "num_chars": len(text)}
        return text
    except Exception as exc:
        warn_message(f"Failed to read text file {path}: {exc}", warning_log)
        files_read[relative_path(path)] = {"status": "error", "type": "text", "error": str(exc)}
        return ""


def safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"})


def class_name(class_id: object) -> str:
    value = safe_float(class_id)
    if value is None:
        return "Unknown"
    idx = int(value)
    if idx == 0:
        return "Adenocarcinoma"
    if idx == 1:
        return "NonAdenocarcinoma"
    return f"class_{idx}"


def compute_pred_margin(class0: object, class1: object, pred: object) -> float | None:
    value0 = safe_float(class0)
    value1 = safe_float(class1)
    pred_value = safe_float(pred)
    if value0 is None or value1 is None or pred_value is None:
        return None
    return value0 - value1 if int(pred_value) == 0 else value1 - value0


def compute_true_vs_wrong_margin(class0: object, class1: object, label: object) -> float | None:
    value0 = safe_float(class0)
    value1 = safe_float(class1)
    label_value = safe_float(label)
    if value0 is None or value1 is None or label_value is None:
        return None
    return value0 - value1 if int(label_value) == 0 else value1 - value0


def support_class_from_pair(class0: object, class1: object, eps: float = 1e-12) -> float | None:
    value0 = safe_float(class0)
    value1 = safe_float(class1)
    if value0 is None or value1 is None:
        return None
    if abs(value0 - value1) <= eps:
        return None
    return 0.0 if value0 > value1 else 1.0


def sign_nonzero(value: object, eps: float = 1e-12) -> int:
    number = safe_float(value)
    if number is None:
        return 0
    if number > eps:
        return 1
    if number < -eps:
        return -1
    return 0


def format_num(value: object, digits: int = 4) -> str:
    number = safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.copy()
    for column in safe_df.columns:
        if pd.api.types.is_float_dtype(safe_df[column]) or pd.api.types.is_integer_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].map(lambda x: format_num(x) if pd.notna(x) else "N/A")
    safe_df = safe_df.fillna("N/A").astype(str)
    header = "| " + " | ".join(safe_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(safe_df.columns)) + " |"
    rows = ["| " + " | ".join(row[column] for column in safe_df.columns) + " |" for _, row in safe_df.iterrows()]
    return "\n".join([header, separator] + rows)


def normalize_slide_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOL_COLUMNS:
        if column in frame.columns:
            frame[column] = parse_bool_series(frame[column]).astype(bool)
    for column in ["failure_labels", "primary_failure_type", "dominant_source", "evidence_agreement", "low_high_joint_state", "prompt_confusion_note"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    if "slide_id" in frame.columns:
        frame["slide_id"] = frame["slide_id"].astype(str)
    return frame


def build_stage32_fallback(stage32_slide_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_slide_dataframe(stage32_slide_df)
    if df.empty:
        return df
    if "correct" not in df.columns:
        df["correct"] = df["label"] == df["pred"]
    df["label_name"] = df["label"].apply(class_name)
    df["pred_name"] = df["pred"].apply(class_name)

    if "pred_margin" not in df.columns:
        df["pred_margin"] = df.apply(
            lambda row: compute_pred_margin(row.get("final_logit_class_0"), row.get("final_logit_class_1"), row.get("pred")),
            axis=1,
        )

    for source_name, class0_col, class1_col in [
        ("final", "final_logit_class_0", "final_logit_class_1"),
        ("low", "low_logit_class_0", "low_logit_class_1"),
        ("high", "high_logit_class_0", "high_logit_class_1"),
        ("visual", "visual_logit_class_0", "visual_logit_class_1"),
        ("csg", "csg_logit_class_0", "csg_logit_class_1"),
    ]:
        margin_col = f"{source_name}_true_vs_wrong_margin"
        if margin_col not in df.columns:
            df[margin_col] = df.apply(
                lambda row: compute_true_vs_wrong_margin(row.get(class0_col), row.get(class1_col), row.get("label")),
                axis=1,
            )
        support_col = f"{source_name}_support_class"
        if support_col not in df.columns:
            df[support_col] = df.apply(lambda row: support_class_from_pair(row.get(class0_col), row.get(class1_col)), axis=1)
        label_col = f"{source_name}_support_label"
        if label_col not in df.columns:
            df[label_col] = df[support_col].apply(class_name)
        abs_col = f"{source_name}_abs_margin"
        if abs_col not in df.columns:
            df[abs_col] = df[margin_col].abs()

    df["visual_effective_margin"] = df.apply(
        lambda row: (
            safe_float(row.get("visual_true_vs_wrong_margin")) * safe_float(row.get("visual_alpha"))
            if safe_float(row.get("visual_true_vs_wrong_margin")) is not None and safe_float(row.get("visual_alpha")) is not None
            else math.nan
        ),
        axis=1,
    )
    df["csg_effective_margin"] = df.apply(
        lambda row: (
            safe_float(row.get("csg_true_vs_wrong_margin")) * safe_float(row.get("csg_alpha"))
            if safe_float(row.get("csg_true_vs_wrong_margin")) is not None and safe_float(row.get("csg_alpha")) is not None
            else math.nan
        ),
        axis=1,
    )
    df["visual_effective_abs_margin"] = df["visual_effective_margin"].abs()
    df["csg_effective_abs_margin"] = df["csg_effective_margin"].abs()
    df["concept_margin_mean"] = df[["low_true_vs_wrong_margin", "high_true_vs_wrong_margin"]].mean(axis=1, skipna=True)
    df["concept_abs_margin_max"] = df[["low_abs_margin", "high_abs_margin"]].max(axis=1, skipna=True)
    source_values = pd.DataFrame(
        {
            "low": df["low_abs_margin"],
            "high": df["high_abs_margin"],
            "visual": df["visual_effective_abs_margin"],
            "csg": df["csg_effective_abs_margin"],
        }
    )
    total = source_values.sum(axis=1, skipna=True).replace(0.0, np.nan)
    df["low_source_ratio"] = df["low_abs_margin"] / total
    df["high_source_ratio"] = df["high_abs_margin"] / total
    df["visual_source_ratio"] = df["visual_effective_abs_margin"] / total
    df["csg_source_ratio"] = df["csg_effective_abs_margin"] / total
    df["dominant_source"] = source_values.idxmax(axis=1)
    df["dominant_source_ratio"] = source_values.max(axis=1) / total
    low_support = df["low_support_class"]
    high_support = df["high_support_class"]
    df["low_available"] = df["low_true_vs_wrong_margin"].notna()
    df["high_available"] = df["high_true_vs_wrong_margin"].notna()
    df["low_supports_true"] = low_support == df["label"]
    df["high_supports_true"] = high_support == df["label"]
    df["low_high_same_support"] = low_support.notna() & high_support.notna() & (low_support == high_support)
    df["low_high_conflict"] = low_support.notna() & high_support.notna() & (low_support != high_support)
    df["low_high_joint_state"] = np.where(
        df["low_high_conflict"],
        "conflict",
        np.where(
            df["low_high_same_support"] & df["low_supports_true"],
            "both_support_true",
            np.where(df["low_high_same_support"], "both_support_wrong", "partial"),
        ),
    )
    df["failure_labels"] = np.where(df["correct"], "", "unclassified_failure")
    df["primary_failure_type"] = np.where(df["correct"], "correct_prediction", "unclassified_failure")
    df["prompt_confusion_flag"] = False
    df["prompt_confusion_score"] = math.nan
    df["prompt_confusion_note"] = ""
    df["visual_override_candidate"] = False
    df["csg_export_capture_status"] = ""
    df["csg_inactive_or_zero_flag"] = False
    return df


def build_slide_reliability_table(
    stage32_slide_df: pd.DataFrame,
    stage33_slide_df: pd.DataFrame,
    visual_override_threshold: float,
    low_margin_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    missing_feature_notes: list[str] = []
    if not stage33_slide_df.empty:
        slide_df = normalize_slide_dataframe(stage33_slide_df)
        if not stage32_slide_df.empty:
            stage32_df = normalize_slide_dataframe(stage32_slide_df)
            add_cols = [column for column in stage32_df.columns if column not in slide_df.columns]
            if add_cols:
                slide_df = slide_df.merge(stage32_df[["slide_id"] + add_cols], on="slide_id", how="left")
    else:
        slide_df = build_stage32_fallback(stage32_slide_df)

    if slide_df.empty:
        raise RuntimeError("Neither Stage33 slide failure labels nor Stage32 slide evidence summary could be loaded.")

    if "slide_id" not in slide_df.columns:
        raise RuntimeError("Slide-level table is missing slide_id.")

    slide_df["slide_id"] = slide_df["slide_id"].astype(str)
    slide_df["correct"] = parse_bool_series(slide_df["correct"]).astype(bool)
    slide_df["label_name"] = slide_df.get("label_name", slide_df["label"].apply(class_name)).replace("", np.nan)
    slide_df["label_name"] = slide_df["label_name"].fillna(slide_df["label"].apply(class_name))
    slide_df["pred_name"] = slide_df.get("pred_name", slide_df["pred"].apply(class_name)).replace("", np.nan)
    slide_df["pred_name"] = slide_df["pred_name"].fillna(slide_df["pred"].apply(class_name))
    slide_df["failure_labels"] = slide_df.get("failure_labels", pd.Series("", index=slide_df.index)).fillna("").astype(str)
    slide_df["failure_type"] = slide_df.get("primary_failure_type", pd.Series("", index=slide_df.index)).replace("", np.nan)
    default_failure_type = pd.Series(
        np.where(slide_df["correct"], "correct_prediction", "unclassified_failure"),
        index=slide_df.index,
    )
    slide_df["failure_type"] = slide_df["failure_type"].fillna(default_failure_type)

    if "final_true_vs_wrong_margin" not in slide_df.columns:
        slide_df["final_true_vs_wrong_margin"] = slide_df.apply(
            lambda row: compute_true_vs_wrong_margin(row.get("final_logit_class_0"), row.get("final_logit_class_1"), row.get("label")),
            axis=1,
        )
    if "pred_margin" not in slide_df.columns:
        slide_df["pred_margin"] = slide_df.apply(
            lambda row: compute_pred_margin(row.get("final_logit_class_0"), row.get("final_logit_class_1"), row.get("pred")),
            axis=1,
        )
    for source_name, class0_col, class1_col in [
        ("low", "low_logit_class_0", "low_logit_class_1"),
        ("high", "high_logit_class_0", "high_logit_class_1"),
        ("visual", "visual_logit_class_0", "visual_logit_class_1"),
        ("csg", "csg_logit_class_0", "csg_logit_class_1"),
    ]:
        margin_col = f"{source_name}_true_vs_wrong_margin"
        if margin_col not in slide_df.columns:
            slide_df[margin_col] = slide_df.apply(
                lambda row: compute_true_vs_wrong_margin(row.get(class0_col), row.get(class1_col), row.get("label")),
                axis=1,
            )
        support_col = f"{source_name}_support_class"
        if support_col not in slide_df.columns:
            slide_df[support_col] = slide_df.apply(lambda row: support_class_from_pair(row.get(class0_col), row.get(class1_col)), axis=1)
        label_col = f"{source_name}_support_label"
        if label_col not in slide_df.columns:
            slide_df[label_col] = slide_df[support_col].apply(class_name)

    slide_df["visual_effective_margin"] = slide_df.get("visual_effective_margin", pd.Series(np.nan, index=slide_df.index))
    visual_missing = slide_df["visual_effective_margin"].isna()
    slide_df.loc[visual_missing, "visual_effective_margin"] = slide_df.loc[visual_missing].apply(
        lambda row: (
            safe_float(row.get("visual_true_vs_wrong_margin")) * safe_float(row.get("visual_alpha"))
            if safe_float(row.get("visual_true_vs_wrong_margin")) is not None and safe_float(row.get("visual_alpha")) is not None
            else math.nan
        ),
        axis=1,
    )
    slide_df["csg_effective_margin"] = slide_df.get("csg_effective_margin", pd.Series(np.nan, index=slide_df.index))
    csg_missing = slide_df["csg_effective_margin"].isna()
    slide_df.loc[csg_missing, "csg_effective_margin"] = slide_df.loc[csg_missing].apply(
        lambda row: (
            safe_float(row.get("csg_true_vs_wrong_margin")) * safe_float(row.get("csg_alpha"))
            if safe_float(row.get("csg_true_vs_wrong_margin")) is not None and safe_float(row.get("csg_alpha")) is not None
            else math.nan
        ),
        axis=1,
    )
    slide_df["visual_effective_abs_margin"] = slide_df["visual_effective_margin"].abs()
    slide_df["csg_effective_abs_margin"] = slide_df["csg_effective_margin"].abs()
    slide_df["low_margin_abs"] = slide_df["low_true_vs_wrong_margin"].abs()
    slide_df["high_margin_abs"] = slide_df["high_true_vs_wrong_margin"].abs()
    slide_df["visual_margin_abs"] = slide_df["visual_effective_margin"].abs()
    slide_df["csg_abs_margin"] = slide_df["csg_effective_margin"].abs()
    if "concept_margin_mean" not in slide_df.columns:
        slide_df["concept_margin_mean"] = slide_df[["low_true_vs_wrong_margin", "high_true_vs_wrong_margin"]].mean(axis=1, skipna=True)
    if "concept_abs_margin_max" not in slide_df.columns:
        slide_df["concept_abs_margin_max"] = slide_df[["low_margin_abs", "high_margin_abs"]].max(axis=1, skipna=True)
    slide_df["concept_margin_abs"] = slide_df["concept_margin_mean"].abs()
    total_margin = (
        slide_df["low_margin_abs"].fillna(0.0)
        + slide_df["high_margin_abs"].fillna(0.0)
        + slide_df["visual_margin_abs"].fillna(0.0)
        + slide_df["csg_abs_margin"].fillna(0.0)
    ).replace(0.0, np.nan)
    if "low_source_ratio" not in slide_df.columns:
        slide_df["low_source_ratio"] = slide_df["low_margin_abs"] / total_margin
    if "high_source_ratio" not in slide_df.columns:
        slide_df["high_source_ratio"] = slide_df["high_margin_abs"] / total_margin
    if "visual_source_ratio" not in slide_df.columns:
        slide_df["visual_source_ratio"] = slide_df["visual_margin_abs"] / total_margin
    if "csg_source_ratio" not in slide_df.columns:
        slide_df["csg_source_ratio"] = slide_df["csg_abs_margin"] / total_margin
    if "dominant_source" not in slide_df.columns:
        source_matrix = pd.DataFrame(
            {
                "low": slide_df["low_margin_abs"],
                "high": slide_df["high_margin_abs"],
                "visual": slide_df["visual_margin_abs"],
                "csg": slide_df["csg_abs_margin"],
            }
        )
        slide_df["dominant_source"] = source_matrix.idxmax(axis=1)
        slide_df["dominant_source_ratio"] = source_matrix.max(axis=1) / total_margin

    if "low_available" not in slide_df.columns:
        slide_df["low_available"] = slide_df["low_true_vs_wrong_margin"].notna()
    if "high_available" not in slide_df.columns:
        slide_df["high_available"] = slide_df["high_true_vs_wrong_margin"].notna()
    slide_df["low_supports_true"] = slide_df.get("low_supports_true", slide_df["low_support_class"] == slide_df["label"]).fillna(False).astype(bool)
    slide_df["high_supports_true"] = slide_df.get("high_supports_true", slide_df["high_support_class"] == slide_df["label"]).fillna(False).astype(bool)
    if "low_high_same_support" not in slide_df.columns:
        slide_df["low_high_same_support"] = slide_df["low_support_class"].notna() & slide_df["high_support_class"].notna() & (
            slide_df["low_support_class"] == slide_df["high_support_class"]
        )
    if "low_high_conflict" not in slide_df.columns:
        slide_df["low_high_conflict"] = slide_df["low_support_class"].notna() & slide_df["high_support_class"].notna() & (
            slide_df["low_support_class"] != slide_df["high_support_class"]
        )
    if "low_high_joint_state" not in slide_df.columns:
        slide_df["low_high_joint_state"] = np.where(
            slide_df["low_high_conflict"],
            "conflict",
            np.where(
                slide_df["low_high_same_support"] & slide_df["low_supports_true"],
                "both_support_true",
                np.where(slide_df["low_high_same_support"], "both_support_wrong", "partial"),
            ),
        )

    slide_df["visual_supports_true"] = slide_df["visual_support_class"].notna() & (slide_df["visual_support_class"] == slide_df["label"])
    slide_df["visual_supports_pred"] = slide_df["visual_support_class"].notna() & (slide_df["visual_support_class"] == slide_df["pred"])
    slide_df["visual_supports_wrong"] = slide_df["visual_support_class"].notna() & (slide_df["visual_support_class"] != slide_df["label"])
    slide_df["low_high_agreement"] = slide_df["low_high_same_support"].fillna(False).astype(bool)
    slide_df["concept_agreement"] = slide_df["low_high_joint_state"].replace("", "unknown")
    slide_df["prediction_confidence_margin"] = slide_df["pred_margin"]

    slide_df["low_high_sign_agreement"] = slide_df.apply(
        lambda row: (
            float(sign_nonzero(row.get("low_true_vs_wrong_margin")) == sign_nonzero(row.get("high_true_vs_wrong_margin")))
            if sign_nonzero(row.get("low_true_vs_wrong_margin")) != 0 and sign_nonzero(row.get("high_true_vs_wrong_margin")) != 0
            else math.nan
        ),
        axis=1,
    )
    slide_df["low_high_margin_agreement"] = slide_df.apply(
        lambda row: compute_low_high_margin_agreement(row.get("low_true_vs_wrong_margin"), row.get("high_true_vs_wrong_margin")),
        axis=1,
    )
    slide_df["visual_concept_conflict"] = slide_df.apply(
        lambda row: compute_visual_concept_conflict(row.get("visual_effective_margin"), row.get("concept_margin_mean")),
        axis=1,
    )
    slide_df["visual_over_concept_ratio"] = slide_df.apply(
        lambda row: (
            safe_float(row.get("visual_margin_abs")) / max(safe_float(row.get("concept_margin_abs")) or 0.0, 1e-6)
            if safe_float(row.get("visual_margin_abs")) is not None
            else math.nan
        ),
        axis=1,
    )
    slide_df["visual_residual_override_flag"] = (
        slide_df["failure_labels"].str.contains("visual_residual_override", regex=False)
        | (
            slide_df["visual_supports_wrong"]
            & (slide_df["visual_source_ratio"].fillna(0.0) >= float(visual_override_threshold))
            & (slide_df["visual_effective_margin"].fillna(0.0) < 0.0)
        )
    )
    slide_df["uncertain_margin_flag"] = slide_df["pred_margin"].fillna(np.inf) <= float(low_margin_threshold)
    slide_df["high_scale_dominant_wrong_flag"] = slide_df.apply(
        lambda row: (
            safe_float(row.get("high_true_vs_wrong_margin")) is not None
            and safe_float(row.get("high_true_vs_wrong_margin")) < 0.0
            and abs(safe_float(row.get("high_true_vs_wrong_margin")) or 0.0) >= abs(safe_float(row.get("low_true_vs_wrong_margin")) or 0.0)
        ),
        axis=1,
    )
    slide_df["low_scale_dominant_wrong_flag"] = slide_df.apply(
        lambda row: (
            safe_float(row.get("low_true_vs_wrong_margin")) is not None
            and safe_float(row.get("low_true_vs_wrong_margin")) < 0.0
            and abs(safe_float(row.get("low_true_vs_wrong_margin")) or 0.0) > abs(safe_float(row.get("high_true_vs_wrong_margin")) or 0.0)
        ),
        axis=1,
    )
    slide_df["csg_misleading_flag"] = slide_df.apply(
        lambda row: (
            safe_float(row.get("csg_effective_margin")) is not None
            and safe_float(row.get("csg_effective_margin")) < 0.0
            and (safe_float(row.get("csg_source_ratio")) or 0.0) >= 0.01
        ),
        axis=1,
    )

    required_columns = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "pred_margin",
        "failure_type",
        "failure_labels",
        "dominant_source",
        "dominant_source_ratio",
        "low_true_vs_wrong_margin",
        "high_true_vs_wrong_margin",
        "visual_effective_margin",
        "csg_effective_margin",
        "visual_source_ratio",
        "visual_supports_true",
        "visual_supports_pred",
        "visual_supports_wrong",
        "low_high_agreement",
        "concept_agreement",
        "prompt_confusion_flag",
        "visual_residual_override_flag",
        "uncertain_margin_flag",
        "high_scale_dominant_wrong_flag",
        "low_scale_dominant_wrong_flag",
        "csg_misleading_flag",
    ]
    for column in required_columns:
        if column not in slide_df.columns:
            missing_feature_notes.append(f"Required slide-level field could not be built: {column}")

    for feature_name in [name for name in FEATURE_COLUMNS if not name.startswith("prompt_")]:
        if feature_name not in slide_df.columns or slide_df[feature_name].notna().sum() == 0:
            missing_feature_notes.append(f"Candidate reliability feature is missing or fully NaN: {feature_name}")

    return slide_df, missing_feature_notes


def compute_low_high_margin_agreement(low_margin: object, high_margin: object) -> float:
    low_value = safe_float(low_margin)
    high_value = safe_float(high_margin)
    if low_value is None or high_value is None:
        return math.nan
    low_abs = abs(low_value)
    high_abs = abs(high_value)
    if max(low_abs, high_abs) <= 1e-12:
        return 1.0
    ratio = min(low_abs, high_abs) / max(low_abs, high_abs)
    same_sign = sign_nonzero(low_value) == sign_nonzero(high_value)
    return float(ratio if same_sign else -ratio)


def compute_visual_concept_conflict(visual_margin: object, concept_margin: object) -> float:
    visual_sign = sign_nonzero(visual_margin)
    concept_sign = sign_nonzero(concept_margin)
    if visual_sign == 0 or concept_sign == 0:
        return math.nan
    return float(visual_sign != concept_sign)


def build_prompt_reliability_scores(
    top_concepts_df: pd.DataFrame,
    prompt_confusion_df: pd.DataFrame,
    prompt_reliability_preview_df: pd.DataFrame,
    slide_table: pd.DataFrame,
    min_prompt_occurrences: int,
    warning_log: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_concepts_df.empty:
        warn_message("Stage32 top concepts CSV is empty; prompt reliability analysis will be skipped.", warning_log)
        slide_scores = slide_table[["slide_id"]].copy()
        slide_scores["prompt_error_risk_score"] = math.nan
        slide_scores["prompt_reliability_score"] = math.nan
        return pd.DataFrame(), slide_scores

    concepts = top_concepts_df.copy()
    concepts["slide_id"] = concepts["slide_id"].astype(str)
    for column in ["label", "pred", "class_id", "concept_rank", "evidence", "weight", "contribution"]:
        if column in concepts.columns:
            concepts[column] = pd.to_numeric(concepts[column], errors="coerce")
    concepts["correct"] = parse_bool_series(concepts["correct"]).astype(bool)
    pred_concepts = concepts[concepts["class_type"].astype(str) == "pred"].copy()
    if pred_concepts.empty:
        warn_message("No predicted-class prompt rows found in stage32_top_concepts_long.csv.", warning_log)
        slide_scores = slide_table[["slide_id"]].copy()
        slide_scores["prompt_error_risk_score"] = math.nan
        slide_scores["prompt_reliability_score"] = math.nan
        return pd.DataFrame(), slide_scores

    slide_flags = slide_table[
        [
            "slide_id",
            "correct",
            "visual_residual_override_flag",
            "low_high_agreement",
            "low_high_sign_agreement",
            "pred",
            "label",
        ]
    ].copy()
    slide_flags["low_high_conflict_flag"] = ~slide_flags["low_high_agreement"].fillna(False)
    pred_concepts = pred_concepts.merge(slide_flags, on="slide_id", how="left", suffixes=("", "_slide"))
    if "correct_slide" in pred_concepts.columns:
        pred_concepts["correct"] = pred_concepts["correct_slide"].combine_first(pred_concepts["correct"])
        pred_concepts = pred_concepts.drop(columns=["correct_slide"])
    pred_concepts["correct"] = pred_concepts["correct"].fillna(False).astype(bool)

    grouped = (
        pred_concepts.groupby(["scale", "class_id", "class_name", "concept_text"], dropna=False)
        .agg(
            occurrences=("slide_id", "size"),
            correct_count=("correct", "sum"),
            error_count=("correct", lambda values: int((~values.astype(bool)).sum())),
            error_rate=("correct", lambda values: float((~values.astype(bool)).mean())),
            visual_override_count=("visual_residual_override_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
            low_high_conflict_count=("low_high_conflict_flag", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
            wrong_class_support_count=("correct", lambda values: int((~values.astype(bool)).sum())),
            mean_evidence=("evidence", "mean"),
            mean_weight=("weight", "mean"),
            mean_contribution=("contribution", "mean"),
        )
        .reset_index()
    )

    if not prompt_confusion_df.empty:
        prompt_conf = prompt_confusion_df.copy()
        if "class_type" in prompt_conf.columns:
            prompt_conf = prompt_conf[prompt_conf["class_type"].astype(str) == "pred"].copy()
        merge_columns = [
            "scale",
            "class_id",
            "class_name",
            "concept_text",
            "occurrences",
            "correct_count",
            "error_count",
            "error_rate_among_occurrences",
        ]
        available_columns = [column for column in merge_columns if column in prompt_conf.columns]
        prompt_conf = prompt_conf[available_columns].copy()
        grouped = grouped.merge(
            prompt_conf,
            on=["scale", "class_id", "class_name", "concept_text"],
            how="left",
            suffixes=("", "_stage33"),
        )
        for local_col, stage33_col in [
            ("occurrences", "occurrences_stage33"),
            ("correct_count", "correct_count_stage33"),
            ("error_count", "error_count_stage33"),
            ("error_rate", "error_rate_among_occurrences"),
        ]:
            if stage33_col in grouped.columns:
                grouped[local_col] = grouped[stage33_col].combine_first(grouped[local_col])
        grouped = grouped.drop(columns=[column for column in grouped.columns if column.endswith("_stage33")], errors="ignore")

    if not prompt_reliability_preview_df.empty:
        preview = prompt_reliability_preview_df.copy()
        preview = preview.rename(columns={"reliability_hint": "stage33_reliability_hint"})
        keep_cols = [column for column in ["scale", "class_id", "class_name", "concept_text", "stage33_reliability_hint"] if column in preview.columns]
        grouped = grouped.merge(preview[keep_cols], on=["scale", "class_id", "class_name", "concept_text"], how="left")

    grouped["visual_override_rate"] = grouped["visual_override_count"] / grouped["occurrences"].replace(0, np.nan)
    grouped["low_high_conflict_rate"] = grouped["low_high_conflict_count"] / grouped["occurrences"].replace(0, np.nan)
    grouped["occurrence_confidence"] = grouped["occurrences"].clip(lower=0) / max(int(min_prompt_occurrences), 1)
    grouped["occurrence_confidence"] = grouped["occurrence_confidence"].clip(upper=1.0)
    grouped["prompt_reliability_score"] = (
        0.5 * (1.0 - grouped["occurrence_confidence"])
        + grouped["occurrence_confidence"] * (1.0 - grouped["error_rate"].fillna(0.5))
    )
    grouped["prompt_error_risk_score"] = 1.0 - grouped["prompt_reliability_score"]
    grouped["reliability_label"] = grouped.apply(
        lambda row: classify_prompt_label(
            occurrences=row["occurrences"],
            error_rate=row["error_rate"],
            visual_override_rate=row["visual_override_rate"],
            low_high_conflict_rate=row["low_high_conflict_rate"],
            min_prompt_occurrences=min_prompt_occurrences,
        ),
        axis=1,
    )
    grouped["recommended_action"] = grouped.apply(
        lambda row: classify_prompt_action(
            reliability_label=row["reliability_label"],
            occurrences=row["occurrences"],
            error_rate=row["error_rate"],
            visual_override_rate=row["visual_override_rate"],
            min_prompt_occurrences=min_prompt_occurrences,
        ),
        axis=1,
    )
    grouped = grouped.sort_values(
        ["reliability_label", "error_rate", "occurrences", "visual_override_count"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    prompt_key = grouped.set_index(["scale", "class_id", "class_name", "concept_text"])
    pred_concepts["contribution_abs"] = pred_concepts["contribution"].abs().fillna(0.0)
    slide_rows: list[dict[str, object]] = []
    for slide_id, group in pred_concepts.groupby("slide_id"):
        weights = []
        reliabilities = []
        risks = []
        for row in group.itertuples(index=False):
            key = (row.scale, row.class_id, row.class_name, row.concept_text)
            if key not in prompt_key.index:
                continue
            prompt_row = prompt_key.loc[key]
            weight = safe_float(getattr(row, "contribution_abs", 0.0)) or 0.0
            if weight <= 0.0:
                weight = abs(safe_float(getattr(row, "weight", 0.0)) or 0.0)
            if weight <= 0.0:
                weight = 1.0
            reliability = safe_float(prompt_row.get("prompt_reliability_score"))
            risk = safe_float(prompt_row.get("prompt_error_risk_score"))
            if reliability is None or risk is None:
                continue
            weights.append(weight)
            reliabilities.append(reliability)
            risks.append(risk)
        if weights:
            slide_rows.append(
                {
                    "slide_id": slide_id,
                    "prompt_reliability_score": float(np.average(reliabilities, weights=weights)),
                    "prompt_error_risk_score": float(np.average(risks, weights=weights)),
                }
            )
        else:
            slide_rows.append(
                {
                    "slide_id": slide_id,
                    "prompt_reliability_score": math.nan,
                    "prompt_error_risk_score": math.nan,
                }
            )

    slide_scores = pd.DataFrame(slide_rows)
    return grouped, slide_scores


def classify_prompt_label(
    occurrences: object,
    error_rate: object,
    visual_override_rate: object,
    low_high_conflict_rate: object,
    min_prompt_occurrences: int,
) -> str:
    occ = safe_float(occurrences) or 0.0
    err = safe_float(error_rate)
    override = safe_float(visual_override_rate) or 0.0
    conflict = safe_float(low_high_conflict_rate) or 0.0
    if occ < float(min_prompt_occurrences):
        return "rare"
    if err is None:
        return "mixed"
    if err >= 0.5 or override >= 0.3:
        return "error_prone"
    if err <= 0.15 and override <= 0.1 and conflict <= 0.15:
        return "reliable"
    return "mixed"


def classify_prompt_action(
    reliability_label: str,
    occurrences: object,
    error_rate: object,
    visual_override_rate: object,
    min_prompt_occurrences: int,
) -> str:
    occ = safe_float(occurrences) or 0.0
    err = safe_float(error_rate) or 0.0
    override = safe_float(visual_override_rate) or 0.0
    if reliability_label == "reliable":
        return "keep"
    if reliability_label == "rare":
        return "monitor"
    if reliability_label == "error_prone":
        if occ >= max(2 * min_prompt_occurrences, 6) and err >= 0.5 and override >= 0.2:
            return "candidate_remove"
        return "downweight"
    return "downweight" if err >= 0.3 or override >= 0.2 else "monitor"


def build_visual_override_patterns(slide_table: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, str, pd.DataFrame]] = [
        ("overall", "all", slide_table),
        ("correctness", "correct", slide_table[slide_table["correct"] == True]),
        ("correctness", "error", slide_table[slide_table["correct"] == False]),
        ("override", "visual_override", slide_table[slide_table["visual_residual_override_flag"] == True]),
        ("override", "non_visual_override", slide_table[slide_table["visual_residual_override_flag"] == False]),
    ]
    for failure_type, group_df in slide_table.groupby("failure_type", dropna=False):
        groups.append(("failure_type", str(failure_type), group_df))

    rows: list[dict[str, object]] = []
    for family, value, group in groups:
        if group.empty:
            continue
        rows.append(
            {
                "group_family": family,
                "group_value": value,
                "n_slides": int(len(group)),
                "error_rate": float((~group["correct"]).mean()),
                "visual_source_ratio_mean": float(group["visual_source_ratio"].mean()),
                "visual_source_ratio_std": float(group["visual_source_ratio"].std(ddof=0)),
                "visual_source_ratio_median": float(group["visual_source_ratio"].median()),
                "visual_effective_margin_mean": float(group["visual_effective_margin"].mean()),
                "visual_effective_margin_std": float(group["visual_effective_margin"].std(ddof=0)),
                "visual_effective_margin_median": float(group["visual_effective_margin"].median()),
                "visual_supports_wrong_ratio": float(group["visual_supports_wrong"].mean()),
                "visual_concept_conflict_mean": float(group["visual_concept_conflict"].mean()),
                "dominant_visual_ratio": float((group["dominant_source"] == "visual").mean()),
                "dominant_high_ratio": float((group["dominant_source"] == "high").mean()),
                "dominant_low_ratio": float((group["dominant_source"] == "low").mean()),
                "dominant_csg_ratio": float((group["dominant_source"] == "csg").mean()),
                "low_margin_abs_mean": float(group["low_margin_abs"].mean()),
                "high_margin_abs_mean": float(group["high_margin_abs"].mean()),
                "concept_margin_abs_mean": float(group["concept_margin_abs"].mean()),
                "prediction_confidence_margin_mean": float(group["prediction_confidence_margin"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["group_family", "group_value"]).reset_index(drop=True)


def auc_rank_based(values: pd.Series, target: pd.Series) -> float:
    valid = values.notna() & target.notna()
    if valid.sum() == 0:
        return math.nan
    scores = values.loc[valid].astype(float)
    labels = target.loc[valid].astype(int)
    positives = int(labels.sum())
    negatives = int((1 - labels).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = scores.rank(method="average")
    rank_sum = float(ranks[labels == 1].sum())
    u_stat = rank_sum - positives * (positives + 1) / 2.0
    return float(u_stat / (positives * negatives))


def build_feature_ranking(slide_table: pd.DataFrame) -> pd.DataFrame:
    target_error = (~slide_table["correct"]).astype(int)
    target_override = slide_table["visual_residual_override_flag"].fillna(False).astype(int)
    rows: list[dict[str, object]] = []

    for feature_name in FEATURE_COLUMNS:
        values = pd.to_numeric(slide_table.get(feature_name), errors="coerce")
        missing_ratio = float(values.isna().mean())
        correct_values = values[slide_table["correct"] == True]
        error_values = values[slide_table["correct"] == False]
        override_values = values[target_override == 1]
        non_override_values = values[target_override == 0]

        correct_mean = float(correct_values.mean()) if correct_values.notna().any() else math.nan
        error_mean = float(error_values.mean()) if error_values.notna().any() else math.nan
        visual_override_mean = float(override_values.mean()) if override_values.notna().any() else math.nan
        non_visual_override_mean = float(non_override_values.mean()) if non_override_values.notna().any() else math.nan

        direction_error = "higher=>error" if error_mean >= correct_mean else "lower=>error"
        direction_override = "higher=>override" if visual_override_mean >= non_visual_override_mean else "lower=>override"
        oriented_error = values if direction_error == "higher=>error" else -values
        oriented_override = values if direction_override == "higher=>override" else -values
        auc_error = auc_rank_based(oriented_error, target_error)
        auc_override = auc_rank_based(oriented_override, target_override)
        separability_score = float(np.nanmean([auc_error, auc_override])) if not np.isnan([auc_error, auc_override]).all() else math.nan

        rows.append(
            {
                "feature_name": feature_name,
                "correct_mean": correct_mean,
                "correct_std": float(correct_values.std(ddof=0)) if correct_values.notna().any() else math.nan,
                "error_mean": error_mean,
                "error_std": float(error_values.std(ddof=0)) if error_values.notna().any() else math.nan,
                "visual_override_mean": visual_override_mean,
                "visual_override_std": float(override_values.std(ddof=0)) if override_values.notna().any() else math.nan,
                "non_visual_override_mean": non_visual_override_mean,
                "non_visual_override_std": float(non_override_values.std(ddof=0)) if non_override_values.notna().any() else math.nan,
                "threshold_direction": direction_error,
                "override_direction": direction_override,
                "auc_error": auc_error,
                "auc_visual_override": auc_override,
                "missing_ratio": missing_ratio,
                "separability_score": separability_score,
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["separability_score", "auc_error", "auc_visual_override", "missing_ratio"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking


def build_feature_set_proposal(feature_ranking: pd.DataFrame) -> dict[str, object]:
    ranking_map = feature_ranking.set_index("feature_name")
    safe_core = [
        "concept_margin_abs",
        "low_high_margin_agreement",
        "visual_concept_conflict",
        "dominant_source_ratio",
        "prediction_confidence_margin",
        "low_high_sign_agreement",
    ]
    safe_features_v1: list[str] = []
    optional_features_v2: list[str] = []
    exclude_features: list[str] = []
    reasons: dict[str, str] = {}

    for feature_name in FEATURE_COLUMNS:
        row = ranking_map.loc[feature_name]
        missing_ratio = safe_float(row.get("missing_ratio"))
        auc_error = safe_float(row.get("auc_error"))
        auc_override = safe_float(row.get("auc_visual_override"))
        separability = safe_float(row.get("separability_score"))
        missing_ratio = 1.0 if missing_ratio is None else missing_ratio
        auc_error = 0.5 if auc_error is None else auc_error
        auc_override = 0.5 if auc_override is None else auc_override
        separability = 0.5 if separability is None else separability
        leakage_risk = feature_name.startswith("prompt_")

        if feature_name in safe_core and missing_ratio <= 0.1 and auc_error >= 0.58 and auc_override >= 0.55:
            safe_features_v1.append(feature_name)
            reasons[feature_name] = "complete enough, interpretable, and separates both error and visual override targets."
            continue

        if leakage_risk:
            optional_features_v2.append(feature_name)
            reasons[feature_name] = "audit signal is useful, but the score is derived from test-set reliability statistics and must be recomputed inside training folds to avoid leakage."
            continue

        if missing_ratio <= 0.2 and separability >= 0.65:
            optional_features_v2.append(feature_name)
            reasons[feature_name] = "potentially useful signal, but less stable or less interpretable than the v1 core set."
        else:
            exclude_features.append(feature_name)
            reasons[feature_name] = "weak separability, unstable behavior, or too little practical value for the first adaptive gate."

    preferred_safe_order = [
        "concept_margin_abs",
        "low_high_margin_agreement",
        "visual_concept_conflict",
        "dominant_source_ratio",
        "prediction_confidence_margin",
        "low_high_sign_agreement",
    ]
    safe_features_v1 = [feature for feature in preferred_safe_order if feature in safe_features_v1]
    optional_preferred = [
        "visual_margin_abs",
        "high_margin_abs",
        "low_margin_abs",
        "visual_over_concept_ratio",
        "prompt_error_risk_score",
        "prompt_reliability_score",
        "csg_abs_margin",
    ]
    existing_optional = set(optional_features_v2)
    optional_features_v2 = [feature for feature in optional_preferred if feature in existing_optional] + [
        feature for feature in optional_features_v2 if feature not in optional_preferred
    ]
    existing_exclude = set(exclude_features)
    exclude_features = [feature for feature in FEATURE_COLUMNS if feature in existing_exclude]

    return {
        "safe_features_v1": safe_features_v1,
        "optional_features_v2": optional_features_v2,
        "exclude_features": exclude_features,
        "feature_reasons": reasons,
        "audit_only_note": (
            "Prompt-derived reliability scores are audit features only in Step45. "
            "They should not be turned into fixed prompt weights from this test export; "
            "if used in Step46, they must be recomputed inside the training fold or learned without test leakage."
        ),
    }


def build_hcrc_negative_ablation_note(
    stage44_summary_df: pd.DataFrame,
    stage44_delta_df: pd.DataFrame,
) -> str:
    merged = stage44_summary_df.merge(stage44_delta_df, on="variant", how="left", suffixes=("_summary", "_delta"))
    lines = [
        "# Stage45 HCRC Negative Ablation Note",
        "",
        "## Step44 Results",
    ]
    for row in merged.itertuples(index=False):
        lines.extend(
            [
                f"### {row.variant}",
                f"- test_auc: `{format_num(getattr(row, 'test_auc_mean', None))}`",
                f"- test_acc: `{format_num(getattr(row, 'test_acc_mean', None))}`",
                f"- test_f1: `{format_num(getattr(row, 'test_f1_mean', None))}`",
                f"- balanced_acc: `{format_num(getattr(row, 'balanced_acc_mean', None))}`",
                f"- sensitivity: `{format_num(getattr(row, 'sensitivity_mean', None))}`",
                f"- specificity: `{format_num(getattr(row, 'specificity_mean', None))}`",
                f"- pr_auc: `{format_num(getattr(row, 'pr_auc_mean', None))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Why HCRC Is Not The Main Model",
            "- Step44 completed cleanly for `hcrc_a002_b8`, `hcrc_a005_b8`, and `hcrc_a01_b8`, but none beat the baseline by the Step44 decision rules.",
            "- `hcrc_a002_b8` improves `ACC / F1 / Balanced ACC`, but `AUC` drops from `0.9702` to `0.9613` and `PR-AUC` drops from `0.9444` to `0.9209`.",
            "- `hcrc_a01_b8` is the closest variant on `AUC` (`0.9683`), but its classification metrics and `PR-AUC` still remain below baseline.",
            "- `hcrc_a005_b8` is weaker than baseline on both ranking metrics and classification metrics.",
            "- The current evidence therefore supports keeping HCRC as a negative ablation or future-work branch rather than the default model direction.",
            "",
            "## Why Step46 Moves To PRARC",
            "- Step33 showed that visual residual evidence is often strong, but its failures are sample-dependent rather than globally suppressible.",
            "- Step39 already concluded that a scalar visual gate is a negative ablation, which points toward sample-adaptive reliability calibration instead of another fixed residual design.",
            "- Step45 therefore audits reliability features that can drive a sample-adaptive residual gate in Step46 without changing the current forward path yet.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(
    args: argparse.Namespace,
    files_read: dict[str, dict[str, object]],
    warning_log: list[str],
    slide_table: pd.DataFrame,
    prompt_scores: pd.DataFrame,
    visual_patterns: pd.DataFrame,
    feature_ranking: pd.DataFrame,
    feature_set: dict[str, object],
    stage39_negative_df: pd.DataFrame,
    stage39_calibration_df: pd.DataFrame,
    stage44_manifest: dict[str, object],
    missing_feature_notes: list[str],
) -> str:
    loaded_files = [path for path, meta in files_read.items() if meta.get("status") == "loaded"]
    missing_files = [path for path, meta in files_read.items() if meta.get("status") != "loaded"]
    error_count = int((~slide_table["correct"]).sum())
    override_count = int(slide_table["visual_residual_override_flag"].sum())
    prompt_preview = prompt_scores[
        [
            "scale",
            "class_name",
            "concept_text",
            "occurrences",
            "error_rate",
            "visual_override_count",
            "reliability_label",
            "recommended_action",
        ]
    ].head(10)
    ranking_preview = feature_ranking[
        ["rank", "feature_name", "auc_error", "auc_visual_override", "missing_ratio", "threshold_direction"]
    ].head(10)
    pattern_preview = visual_patterns[
        [
            "group_family",
            "group_value",
            "n_slides",
            "visual_source_ratio_mean",
            "visual_effective_margin_mean",
            "visual_supports_wrong_ratio",
            "visual_concept_conflict_mean",
        ]
    ].head(12)
    top_failures = (
        slide_table[slide_table["correct"] == False]["failure_type"].value_counts(dropna=False).rename_axis("failure_type").reset_index(name="count")
    )
    lines = [
        "# Step45 PRARC Reliability Audit",
        "",
        "## Purpose",
        "- Step45 is an audit stage. It does not train a model and does not modify the existing forward path.",
        "- The goal is to identify reliability features that separate correct vs error slides and visual-override vs non-override slides, so that Step46 can build a sample-adaptive residual gate on stable inputs.",
        "",
        "## Inputs Read",
        f"- Loaded files: `{len(loaded_files)}`",
    ]
    lines.extend([f"- {path}" for path in loaded_files])
    lines.append("")
    lines.append("## Missing Or Warning Inputs")
    if missing_files:
        lines.extend([f"- {path}" for path in missing_files])
    else:
        lines.append("- No requested input file was missing.")
    if warning_log:
        lines.append("- Runtime warnings:")
        lines.extend([f"  - {message}" for message in warning_log[:20]])
    lines.extend(
        [
            "",
            "## Missing Feature Notes",
        ]
    )
    if missing_feature_notes:
        lines.extend([f"- {note}" for note in missing_feature_notes])
    else:
        lines.append("- No required Step45 feature was fully missing in the current exports.")
    lines.extend(
        [
            "",
            "## Slide-Level Reliability Summary",
            f"- Slides audited: `{len(slide_table)}`",
            f"- Errors: `{error_count}`",
            f"- Visual residual override flags: `{override_count}`",
            f"- Dominant visual source ratio mean: `{format_num(slide_table['visual_source_ratio'].mean())}`",
            f"- Error slides with visual supporting the wrong class: `{format_num(slide_table.loc[slide_table['correct'] == False, 'visual_supports_wrong'].mean())}`",
            f"- Error slides with low-high conflict: `{format_num(slide_table.loc[slide_table['correct'] == False, 'low_high_agreement'].eq(False).mean())}`",
            "",
            "Top failure types:",
            markdown_table(top_failures.head(10)),
            "",
            "## Prompt Reliability Summary",
            "- Prompt reliability is audited from existing test-export evidence only.",
            "- Step45 must not be used to directly freeze prompt weights from this test export. Any Step46 use must recompute prompt-derived statistics inside training folds to avoid leakage.",
            markdown_table(prompt_preview),
            "",
            "## Visual Residual Override Summary",
            "- Visual residual is most useful when it agrees with concept evidence and increases confidence on already-correct slides.",
            "- Visual residual is most misleading when it dominates source ratio, supports the wrong class, and conflicts with the concept margin sign.",
            "- This supports an adaptive gate that sees concept margin, low-high agreement, visual-concept conflict, dominant-source ratio, and prediction margin together.",
            markdown_table(pattern_preview),
            "",
            "## Feature Separability Ranking",
            markdown_table(ranking_preview),
            "",
            "## PRARC Gate Feature Proposal",
            f"- safe_features_v1: `{', '.join(feature_set['safe_features_v1'])}`",
            f"- optional_features_v2: `{', '.join(feature_set['optional_features_v2'])}`",
            f"- exclude_features: `{', '.join(feature_set['exclude_features'])}`",
            "",
            "## Stage39 / Stage44 Context",
        ]
    )
    if not stage39_calibration_df.empty:
        row = stage39_calibration_df.iloc[0]
        lines.extend(
            [
                f"- Stage39 evidence calibration summary: fixed `{int(row['fixed_cases'])}`, regressed `{int(row['regressed_cases'])}`, persistent `{int(row['persistent_errors'])}`.",
                f"- Low-high conflicts moved from `{int(row['low_high_conflict_skeleton'])}` to `{int(row['low_high_conflict_lh'])}`, but visual override increased from `{int(row['visual_residual_override_skeleton'])}` to `{int(row['visual_residual_override_lh'])}`.",
            ]
        )
    if not stage39_negative_df.empty:
        lines.append("- Stage39 negative ablation summary:")
        for row in stage39_negative_df.itertuples(index=False):
            lines.append(f"  - {row.module}: {row.negative_ablation_statement}")
    if stage44_manifest:
        lines.extend(
            [
                f"- Stage44 enter_step45 from manifest: `{stage44_manifest.get('enter_step45')}`",
                f"- Stage44 reason: `{stage44_manifest.get('enter_step45_reason', 'N/A')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def build_manifest(
    args: argparse.Namespace,
    files_read: dict[str, dict[str, object]],
    warning_log: list[str],
    slide_table: pd.DataFrame,
    prompt_scores: pd.DataFrame,
    visual_patterns: pd.DataFrame,
    feature_ranking: pd.DataFrame,
    feature_set: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    return {
        "step": 45,
        "name": "PRARC Reliability Audit",
        "root_dir": str(ROOT),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "args": vars(args),
        "inputs": files_read,
        "warnings": warning_log,
        "summary": {
            "num_slides": int(len(slide_table)),
            "num_errors": int((~slide_table["correct"]).sum()),
            "num_visual_override_flags": int(slide_table["visual_residual_override_flag"].sum()),
            "top_safe_features": feature_set["safe_features_v1"],
            "top_feature_by_separability": feature_ranking.iloc[0]["feature_name"] if not feature_ranking.empty else None,
            "num_prompt_records": int(len(prompt_scores)),
            "num_visual_pattern_groups": int(len(visual_patterns)),
        },
        "audit_only_note": feature_set["audit_only_note"],
        "output_paths": {
            "slide_reliability_table": str(output_dir / "stage45_slide_reliability_table.csv"),
            "prompt_reliability_scores": str(output_dir / "stage45_prompt_reliability_scores.csv"),
            "visual_override_patterns": str(output_dir / "stage45_visual_override_patterns.csv"),
            "reliability_feature_ranking": str(output_dir / "stage45_reliability_feature_ranking.csv"),
            "prarc_feature_set": str(output_dir / "stage45_prarc_feature_set.json"),
            "hcrc_negative_ablation_note": str(output_dir / "stage45_hcrc_negative_ablation_note.md"),
            "prarc_reliability_report": str(output_dir / "stage45_prarc_reliability_report.md"),
            "manifest": str(output_dir / "stage45_manifest.json"),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []
    files_read: dict[str, dict[str, object]] = {}

    stage32_dir = resolve_path(args.stage32_dir)
    stage33_dir = resolve_path(args.stage33_dir)
    stage39_dir = resolve_path(args.stage39_dir)
    stage44_dir = resolve_path(args.stage44_dir)

    stage32_slide_df = safe_read_csv(stage32_dir / "stage32_slide_evidence_summary.csv", warning_log, files_read)
    stage32_top_concepts_df = safe_read_csv(stage32_dir / "stage32_top_concepts_long.csv", warning_log, files_read)
    stage32_error_df = safe_read_csv(stage32_dir / "stage32_error_cases.csv", warning_log, files_read)
    stage32_csg_pairs_df = safe_read_csv(stage32_dir / "stage32_top_csg_pairs.csv", warning_log, files_read)
    stage32_manifest = safe_read_json(stage32_dir / "stage32_manifest.json", warning_log, files_read)

    stage33_slide_df = safe_read_csv(stage33_dir / "stage33_slide_failure_labels.csv", warning_log, files_read)
    stage33_error_cases_df = safe_read_csv(stage33_dir / "stage33_error_failure_cases.csv", warning_log, files_read)
    stage33_visual_df = safe_read_csv(stage33_dir / "stage33_visual_residual_diagnostics.csv", warning_log, files_read)
    stage33_source_stats_df = safe_read_csv(stage33_dir / "stage33_evidence_source_stats.csv", warning_log, files_read)
    stage33_prompt_confusion_df = safe_read_csv(stage33_dir / "stage33_prompt_confusion_summary.csv", warning_log, files_read)
    stage33_prompt_preview_df = safe_read_csv(stage33_dir / "stage33_prompt_reliability_preview.csv", warning_log, files_read)
    stage33_failure_counts_df = safe_read_csv(stage33_dir / "stage33_failure_type_counts.csv", warning_log, files_read)
    stage33_recommendations = safe_read_json(stage33_dir / "stage33_recommendations.json", warning_log, files_read)

    stage39_calibration_df = safe_read_csv(stage39_dir / "stage39_evidence_calibration_summary.csv", warning_log, files_read)
    stage39_final_performance_df = safe_read_csv(stage39_dir / "stage39_final_performance_summary.csv", warning_log, files_read)
    stage39_negative_df = safe_read_csv(stage39_dir / "stage39_negative_ablation_summary.csv", warning_log, files_read)
    stage39_recommendation = safe_read_json(stage39_dir / "stage39_final_model_recommendation.json", warning_log, files_read)

    stage44_summary_df = safe_read_csv(stage44_dir / "stage44_hcrc_5fold_summary.csv", warning_log, files_read)
    stage44_delta_df = safe_read_csv(stage44_dir / "stage44_hcrc_vs_baseline.csv", warning_log, files_read)
    stage44_manifest = safe_read_json(stage44_dir / "stage44_manifest.json", warning_log, files_read)
    stage44_report = safe_read_text(stage44_dir / "stage44_hcrc_light_report.md", warning_log, files_read)
    del stage32_error_df, stage32_csg_pairs_df, stage33_error_cases_df, stage33_visual_df, stage33_source_stats_df
    del stage33_failure_counts_df, stage32_manifest, stage33_recommendations, stage39_recommendation, stage44_report
    del stage39_final_performance_df

    slide_table, missing_feature_notes = build_slide_reliability_table(
        stage32_slide_df=stage32_slide_df,
        stage33_slide_df=stage33_slide_df,
        visual_override_threshold=float(args.visual_override_threshold),
        low_margin_threshold=float(args.low_margin_threshold),
    )

    prompt_scores_df, slide_prompt_scores_df = build_prompt_reliability_scores(
        top_concepts_df=stage32_top_concepts_df,
        prompt_confusion_df=stage33_prompt_confusion_df,
        prompt_reliability_preview_df=stage33_prompt_preview_df,
        slide_table=slide_table,
        min_prompt_occurrences=int(args.min_prompt_occurrences),
        warning_log=warning_log,
    )

    slide_table = slide_table.merge(slide_prompt_scores_df, on="slide_id", how="left", suffixes=("", "_prompt"))
    for column in ["prompt_error_risk_score", "prompt_reliability_score"]:
        prompt_column = f"{column}_prompt"
        if prompt_column in slide_table.columns:
            if column not in slide_table.columns:
                slide_table[column] = slide_table[prompt_column]
            else:
                slide_table[column] = slide_table[prompt_column].combine_first(slide_table[column])
            slide_table = slide_table.drop(columns=[prompt_column])
    if "prompt_error_risk_score" not in slide_table.columns:
        slide_table["prompt_error_risk_score"] = math.nan
    if "prompt_reliability_score" not in slide_table.columns:
        slide_table["prompt_reliability_score"] = math.nan

    visual_patterns_df = build_visual_override_patterns(slide_table)
    feature_ranking_df = build_feature_ranking(slide_table)
    feature_set = build_feature_set_proposal(feature_ranking_df)

    hcrc_note = build_hcrc_negative_ablation_note(stage44_summary_df, stage44_delta_df)
    report_md = build_report(
        args=args,
        files_read=files_read,
        warning_log=warning_log,
        slide_table=slide_table,
        prompt_scores=prompt_scores_df,
        visual_patterns=visual_patterns_df,
        feature_ranking=feature_ranking_df,
        feature_set=feature_set,
        stage39_negative_df=stage39_negative_df,
        stage39_calibration_df=stage39_calibration_df,
        stage44_manifest=stage44_manifest,
        missing_feature_notes=missing_feature_notes,
    )
    manifest = build_manifest(
        args=args,
        files_read=files_read,
        warning_log=warning_log,
        slide_table=slide_table,
        prompt_scores=prompt_scores_df,
        visual_patterns=visual_patterns_df,
        feature_ranking=feature_ranking_df,
        feature_set=feature_set,
        output_dir=output_dir,
    )

    slide_output_columns = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "pred_margin",
        "failure_type",
        "failure_labels",
        "dominant_source",
        "dominant_source_ratio",
        "low_true_vs_wrong_margin",
        "high_true_vs_wrong_margin",
        "visual_effective_margin",
        "csg_effective_margin",
        "visual_source_ratio",
        "visual_supports_true",
        "visual_supports_pred",
        "visual_supports_wrong",
        "low_high_agreement",
        "concept_agreement",
        "prompt_confusion_flag",
        "visual_residual_override_flag",
        "uncertain_margin_flag",
        "high_scale_dominant_wrong_flag",
        "low_scale_dominant_wrong_flag",
        "csg_misleading_flag",
    ] + [feature for feature in FEATURE_COLUMNS if feature != "dominant_source_ratio"]
    extra_columns = [column for column in slide_table.columns if column not in slide_output_columns]
    slide_table[slide_output_columns + extra_columns].to_csv(
        output_dir / "stage45_slide_reliability_table.csv",
        index=False,
    )
    prompt_scores_df.to_csv(output_dir / "stage45_prompt_reliability_scores.csv", index=False)
    visual_patterns_df.to_csv(output_dir / "stage45_visual_override_patterns.csv", index=False)
    feature_ranking_df.to_csv(output_dir / "stage45_reliability_feature_ranking.csv", index=False)
    (output_dir / "stage45_prarc_feature_set.json").write_text(
        json.dumps(feature_set, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "stage45_hcrc_negative_ablation_note.md").write_text(hcrc_note, encoding="utf-8")
    (output_dir / "stage45_prarc_reliability_report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "stage45_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[Step45] Wrote audit outputs to: {output_dir}")


if __name__ == "__main__":
    main()
