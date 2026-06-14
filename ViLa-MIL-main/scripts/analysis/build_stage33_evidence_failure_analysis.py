from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = Path("results_stage32/stage32_rce_v4_csg_evidence_export")
DEFAULT_OUTPUT_DIR = Path("results_stage33/stage33_evidence_failure_analysis")


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step33 evidence failure/conflict analysis from Step32 exports.")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--near_zero_eps", type=float, default=1e-6)
    parser.add_argument("--dominance_ratio", type=float, default=0.5)
    parser.add_argument("--low_margin_quantile", type=float, default=0.25)
    parser.add_argument("--top_error_cases", type=int, default=10)
    parser.add_argument("--top_concepts", type=int, default=10)
    parser.add_argument("--label_names", type=str, default="Adenocarcinoma,NonAdenocarcinoma")
    return parser.parse_args()


def resolve_path(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def relative_path_str(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_read_csv(path: Path, warning_log: list[str], required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing required CSV: {path}")
        warn_message(f"Missing optional CSV: {path}", warning_log)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Failed to read CSV {path}: {exc}") from exc
        warn_message(f"Failed to read optional CSV {path}: {exc}", warning_log)
        return pd.DataFrame()


def safe_read_json(path: Path, warning_log: list[str]) -> dict:
    if not path.is_file():
        warn_message(f"Missing optional JSON: {path}", warning_log)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn_message(f"Failed to read JSON {path}: {exc}", warning_log)
        return {}


def safe_read_text(path: Path, warning_log: list[str]) -> str:
    if not path.is_file():
        warn_message(f"Missing optional text file: {path}", warning_log)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warn_message(f"Failed to read text file {path}: {exc}", warning_log)
        return ""


def first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"})


def nanmean_pair(a: float | None, b: float | None) -> float:
    values = [value for value in [a, b] if value is not None and not math.isnan(value)]
    if not values:
        return math.nan
    return float(np.mean(values))


def safe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def class_name(class_id: int | float | None, label_names: list[str]) -> str:
    class_id = safe_float(class_id)
    if class_id is None:
        return "Unknown"
    idx = int(class_id)
    if 0 <= idx < len(label_names):
        return label_names[idx]
    return f"class_{idx}"


def compute_pred_margin_from_pair(value0: float | None, value1: float | None, pred: int | None) -> float | None:
    if value0 is None or value1 is None or pred is None:
        return None
    if pred == 0:
        return value0 - value1
    if pred == 1:
        return value1 - value0
    return None


def compute_true_vs_wrong_margin(value0: float | None, value1: float | None, label: int | None) -> float | None:
    if value0 is None or value1 is None or label is None:
        return None
    if label == 0:
        return value0 - value1
    if label == 1:
        return value1 - value0
    return None


def support_class_from_pair(value0: float | None, value1: float | None, eps: float) -> int | None:
    if value0 is None or value1 is None:
        return None
    if abs(value0 - value1) <= eps:
        return None
    return 0 if value0 > value1 else 1


def quantile_or_nan(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    q = min(max(float(q), 0.0), 1.0)
    return float(clean.quantile(q))


def build_slide_table(
    slide_df: pd.DataFrame,
    csg_pair_df: pd.DataFrame,
    label_names: list[str],
    near_zero_eps: float,
    dominance_ratio: float,
    low_margin_quantile: float,
    warning_log: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = slide_df.copy()

    slide_id_col = first_existing(df.columns, ["slide_id", "case_id"])
    label_col = first_existing(df.columns, ["label", "true_label"])
    pred_col = first_existing(df.columns, ["pred", "pred_label"])
    correct_col = first_existing(df.columns, ["correct", "is_correct"])
    if slide_id_col is None or label_col is None or pred_col is None:
        raise ValueError("Required slide columns are missing from Step32 summary CSV.")

    standardized = pd.DataFrame()
    standardized["slide_id"] = df[slide_id_col].astype(str)
    standardized["label"] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    standardized["pred"] = pd.to_numeric(df[pred_col], errors="coerce").astype("Int64")
    if correct_col is not None:
        standardized["correct"] = parse_bool_series(df[correct_col]).astype(bool)
    else:
        standardized["correct"] = standardized["label"] == standardized["pred"]

    for field in [
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
        "top_csg_pair_class_0",
        "top_csg_pair_class_1",
        "top_csg_pair_score_class_0",
        "top_csg_pair_score_class_1",
        "top_low_concepts_for_pred",
        "top_high_concepts_for_pred",
        "top_low_concepts_for_true",
        "top_high_concepts_for_true",
    ]:
        column = first_existing(df.columns, [field])
        if column is not None:
            standardized[field] = df[column]

    standardized["label_name"] = standardized["label"].apply(lambda x: class_name(x, label_names))
    standardized["pred_name"] = standardized["pred"].apply(lambda x: class_name(x, label_names))

    standardized["pred_margin"] = pd.to_numeric(standardized.get("pred_margin"), errors="coerce")
    pred_margin_missing = standardized["pred_margin"].isna()
    if pred_margin_missing.any():
        fallback_logits = standardized.loc[pred_margin_missing, ["final_logit_class_0", "final_logit_class_1", "pred"]]
        standardized.loc[pred_margin_missing, "pred_margin"] = fallback_logits.apply(
            lambda row: compute_pred_margin_from_pair(
                safe_float(row.get("final_logit_class_0")),
                safe_float(row.get("final_logit_class_1")),
                None if pd.isna(row.get("pred")) else int(row.get("pred")),
            ),
            axis=1,
        )
    pred_margin_missing = standardized["pred_margin"].isna()
    if pred_margin_missing.any():
        fallback_probs = standardized.loc[pred_margin_missing, ["prob_class_0", "prob_class_1", "pred"]]
        standardized.loc[pred_margin_missing, "pred_margin"] = fallback_probs.apply(
            lambda row: compute_pred_margin_from_pair(
                safe_float(row.get("prob_class_0")),
                safe_float(row.get("prob_class_1")),
                None if pd.isna(row.get("pred")) else int(row.get("pred")),
            ),
            axis=1,
        )

    source_specs = {
        "final": ("final_logit_class_0", "final_logit_class_1"),
        "low": ("low_logit_class_0", "low_logit_class_1"),
        "high": ("high_logit_class_0", "high_logit_class_1"),
        "visual": ("visual_logit_class_0", "visual_logit_class_1"),
        "csg": ("csg_logit_class_0", "csg_logit_class_1"),
    }

    for source_name, (class0_col, class1_col) in source_specs.items():
        standardized[f"{source_name}_true_vs_wrong_margin"] = standardized.apply(
            lambda row: compute_true_vs_wrong_margin(
                safe_float(row.get(class0_col)),
                safe_float(row.get(class1_col)),
                None if pd.isna(row.get("label")) else int(row.get("label")),
            ),
            axis=1,
        )
        standardized[f"{source_name}_support_class"] = standardized.apply(
            lambda row: support_class_from_pair(
                safe_float(row.get(class0_col)),
                safe_float(row.get(class1_col)),
                near_zero_eps,
            ),
            axis=1,
        )
        standardized[f"{source_name}_support_label"] = standardized[f"{source_name}_support_class"].apply(
            lambda x: class_name(x, label_names) if pd.notna(x) else "Unknown"
        )
        standardized[f"{source_name}_abs_margin"] = standardized[f"{source_name}_true_vs_wrong_margin"].abs()

    standardized["visual_effective_margin"] = standardized.apply(
        lambda row: (
            safe_float(row.get("visual_true_vs_wrong_margin")) * safe_float(row.get("visual_alpha"))
            if safe_float(row.get("visual_true_vs_wrong_margin")) is not None
            and safe_float(row.get("visual_alpha")) is not None
            else math.nan
        ),
        axis=1,
    )
    standardized["visual_effective_abs_margin"] = standardized["visual_effective_margin"].abs()
    standardized["csg_effective_margin"] = standardized.apply(
        lambda row: (
            safe_float(row.get("csg_true_vs_wrong_margin")) * safe_float(row.get("csg_alpha"))
            if safe_float(row.get("csg_true_vs_wrong_margin")) is not None
            and safe_float(row.get("csg_alpha")) is not None
            else math.nan
        ),
        axis=1,
    )
    standardized["csg_effective_abs_margin"] = standardized["csg_effective_margin"].abs()

    standardized["concept_margin_mean"] = standardized.apply(
        lambda row: nanmean_pair(
            safe_float(row.get("low_true_vs_wrong_margin")),
            safe_float(row.get("high_true_vs_wrong_margin")),
        ),
        axis=1,
    )
    standardized["concept_abs_margin_max"] = standardized[["low_abs_margin", "high_abs_margin"]].max(axis=1, skipna=True)

    source_margin_cols = [
        "low_abs_margin",
        "high_abs_margin",
        "visual_effective_abs_margin",
        "csg_effective_abs_margin",
    ]
    standardized["available_source_count"] = standardized[source_margin_cols].notna().sum(axis=1)
    standardized["total_abs_source_margin"] = standardized[source_margin_cols].sum(axis=1, skipna=True)
    for source_name in ["low", "high", "visual", "csg"]:
        standardized[f"{source_name}_source_ratio"] = standardized.apply(
            lambda row: (
                safe_float(
                    row.get(
                        f"{source_name}_effective_abs_margin"
                        if source_name in {"visual", "csg"}
                        else f"{source_name}_abs_margin"
                    )
                ) / safe_float(row.get("total_abs_source_margin"))
                if safe_float(
                    row.get(
                        f"{source_name}_effective_abs_margin"
                        if source_name in {"visual", "csg"}
                        else f"{source_name}_abs_margin"
                    )
                ) is not None
                and safe_float(row.get("total_abs_source_margin")) not in {None, 0.0}
                else math.nan
            ),
            axis=1,
        )

    def dominant_source(row: pd.Series) -> tuple[str, float]:
        pairs: list[tuple[str, float]] = []
        for source_name in ["low", "high", "visual", "csg"]:
            value = safe_float(
                row.get(
                    f"{source_name}_effective_abs_margin"
                    if source_name in {"visual", "csg"}
                    else f"{source_name}_abs_margin"
                )
            )
            if value is not None:
                pairs.append((source_name, value))
        if not pairs:
            return "none", math.nan
        source_name, value = max(pairs, key=lambda item: item[1])
        total = safe_float(row.get("total_abs_source_margin"))
        ratio = value / total if total not in {None, 0.0} else math.nan
        return source_name, ratio

    dominant_pairs = standardized.apply(dominant_source, axis=1, result_type="expand")
    standardized["dominant_source"] = dominant_pairs[0]
    standardized["dominant_source_ratio"] = pd.to_numeric(dominant_pairs[1], errors="coerce")

    standardized["evidence_agreement"] = standardized.apply(
        lambda row: summarize_evidence_agreement(row, ["low", "high", "visual", "csg"]),
        axis=1,
    )

    pred_margin_q25 = quantile_or_nan(standardized["pred_margin"], low_margin_quantile)
    pred_margin_q75 = quantile_or_nan(standardized["pred_margin"], 1.0 - low_margin_quantile)
    if math.isnan(pred_margin_q75):
        pred_margin_q75 = quantile_or_nan(standardized["pred_margin"], 0.75)
    standardized["confidence_level"] = standardized["pred_margin"].apply(
        lambda value: classify_confidence(safe_float(value), pred_margin_q25, pred_margin_q75)
    )

    low_high_conflict = standardized.apply(
        lambda row: summarize_low_high_relation(
            safe_float(row.get("low_true_vs_wrong_margin")),
            safe_float(row.get("high_true_vs_wrong_margin")),
            row.get("low_support_class"),
            row.get("high_support_class"),
            row.get("label"),
            near_zero_eps,
            label_names,
        ),
        axis=1,
        result_type="expand",
    )
    for column in low_high_conflict.columns:
        standardized[column] = low_high_conflict[column]

    max_pair_score_by_slide = pd.Series(dtype=float)
    if not csg_pair_df.empty and "slide_id" in csg_pair_df.columns and "pair_score" in csg_pair_df.columns:
        grouped = csg_pair_df.copy()
        grouped["pair_score"] = pd.to_numeric(grouped["pair_score"], errors="coerce")
        max_pair_score_by_slide = grouped.groupby("slide_id")["pair_score"].apply(
            lambda values: float(np.nanmax(np.abs(values.to_numpy(dtype=float)))) if len(values) else math.nan
        )

    standardized["csg_max_pair_score"] = standardized["slide_id"].map(max_pair_score_by_slide)
    if "top_csg_pair_score_class_0" in standardized.columns or "top_csg_pair_score_class_1" in standardized.columns:
        standardized["csg_max_pair_score"] = standardized[
            ["csg_max_pair_score", "top_csg_pair_score_class_0", "top_csg_pair_score_class_1"]
        ].apply(lambda row: np.nanmax(pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)), axis=1)

    standardized["csg_pair_near_zero"] = standardized["csg_max_pair_score"].abs() <= near_zero_eps
    standardized["csg_margin_near_zero"] = standardized["csg_effective_abs_margin"] <= near_zero_eps
    standardized["csg_logits_near_zero"] = standardized[["csg_logit_class_0", "csg_logit_class_1"]].abs().max(axis=1, skipna=True) <= near_zero_eps
    standardized["csg_export_capture_status"] = standardized.apply(
        lambda row: classify_csg_capture_status(
            safe_float(row.get("csg_effective_abs_margin")),
            safe_float(row.get("csg_max_pair_score")),
            near_zero_eps,
        ),
        axis=1,
    )
    standardized["csg_inactive_or_zero_flag"] = standardized["csg_export_capture_status"].isin(
        ["inactive_or_very_weak", "missing_pair_signal"]
    )

    standardized["visual_override_candidate"] = standardized.apply(
        lambda row: is_visual_override_candidate(
            visual_margin=safe_float(row.get("visual_effective_margin")),
            low_margin=safe_float(row.get("low_true_vs_wrong_margin")),
            high_margin=safe_float(row.get("high_true_vs_wrong_margin")),
            visual_ratio=safe_float(row.get("visual_source_ratio")),
            concept_abs_max=safe_float(row.get("concept_abs_margin_max")),
            eps=near_zero_eps,
            dominance_ratio=dominance_ratio,
        ),
        axis=1,
    )

    standardized["prompt_confusion_flag"] = False
    standardized["prompt_confusion_score"] = math.nan
    standardized["prompt_confusion_note"] = ""

    thresholds = {
        "pred_margin_q25": pred_margin_q25,
        "pred_margin_q75": pred_margin_q75,
        "low_margin_quantile_threshold": quantile_or_nan(standardized["low_abs_margin"], low_margin_quantile),
        "high_margin_quantile_threshold": quantile_or_nan(standardized["high_abs_margin"], low_margin_quantile),
        "visual_margin_quantile_threshold": quantile_or_nan(standardized["visual_effective_abs_margin"], low_margin_quantile),
        "csg_margin_quantile_threshold": quantile_or_nan(standardized["csg_effective_abs_margin"], max(0.5, low_margin_quantile)),
    }

    return standardized, thresholds


def summarize_evidence_agreement(row: pd.Series, sources: list[str]) -> str:
    support_classes: list[int] = []
    for source_name in sources:
        value = row.get(f"{source_name}_support_class")
        if pd.notna(value):
            support_classes.append(int(value))
    if not support_classes:
        return "no_available_source"
    unique = sorted(set(support_classes))
    if len(unique) == 1:
        label = row.get("label")
        pred = row.get("pred")
        if pd.notna(label) and unique[0] == int(label):
            return "all_support_true"
        if pd.notna(pred) and unique[0] == int(pred):
            return "all_support_pred"
        return "all_support_same_other"
    return "source_conflict"


def classify_confidence(value: float | None, q25: float, q75: float) -> str:
    if value is None or math.isnan(value):
        return "unknown"
    if not math.isnan(q25) and value <= q25:
        return "low"
    if not math.isnan(q75) and value >= q75:
        return "high"
    return "medium"


def summarize_low_high_relation(
    low_margin: float | None,
    high_margin: float | None,
    low_support_class,
    high_support_class,
    label,
    eps: float,
    label_names: list[str],
) -> dict[str, object]:
    low_available = low_margin is not None
    high_available = high_margin is not None
    low_support_class = None if pd.isna(low_support_class) else int(low_support_class)
    high_support_class = None if pd.isna(high_support_class) else int(high_support_class)
    true_label = None if pd.isna(label) else int(label)

    result = {
        "low_available": low_available,
        "high_available": high_available,
        "low_support_class": low_support_class,
        "high_support_class": high_support_class,
        "low_support_label": class_name(low_support_class, label_names) if low_support_class is not None else "Unknown",
        "high_support_label": class_name(high_support_class, label_names) if high_support_class is not None else "Unknown",
        "low_supports_true": low_support_class == true_label if low_support_class is not None and true_label is not None else False,
        "high_supports_true": high_support_class == true_label if high_support_class is not None and true_label is not None else False,
        "low_high_same_support": False,
        "low_high_conflict": False,
        "low_high_joint_state": "missing",
    }

    if low_available and high_available and low_support_class is not None and high_support_class is not None:
        result["low_high_same_support"] = low_support_class == high_support_class
        result["low_high_conflict"] = low_support_class != high_support_class
        if low_support_class == high_support_class:
            if true_label is not None and low_support_class == true_label:
                result["low_high_joint_state"] = "both_support_true"
            else:
                result["low_high_joint_state"] = "both_support_wrong"
        else:
            result["low_high_joint_state"] = "conflict"
    elif low_available or high_available:
        result["low_high_joint_state"] = "partial"

    if low_margin is not None and abs(low_margin) <= eps:
        result["low_high_joint_state"] = (
            "partial_near_zero" if result["low_high_joint_state"] == "partial" else result["low_high_joint_state"]
        )
    if high_margin is not None and abs(high_margin) <= eps:
        result["low_high_joint_state"] = (
            "partial_near_zero" if result["low_high_joint_state"] == "partial" else result["low_high_joint_state"]
        )
    return result


def classify_csg_capture_status(csg_abs_margin: float | None, pair_score: float | None, eps: float) -> str:
    if csg_abs_margin is None and pair_score is None:
        return "missing"
    csg_small = csg_abs_margin is None or abs(csg_abs_margin) <= eps
    pair_small = pair_score is None or abs(pair_score) <= eps
    if csg_small and pair_small:
        return "inactive_or_very_weak"
    if not csg_small and pair_small:
        return "pair_export_may_need_refinement"
    if csg_small and not pair_small:
        return "missing_pair_signal"
    return "active_and_captured"


def is_visual_override_candidate(
    visual_margin: float | None,
    low_margin: float | None,
    high_margin: float | None,
    visual_ratio: float | None,
    concept_abs_max: float | None,
    eps: float,
    dominance_ratio: float,
) -> bool:
    if visual_margin is None or visual_ratio is None:
        return False
    concept_supports_true = (
        (low_margin is not None and low_margin > eps) or
        (high_margin is not None and high_margin > eps)
    )
    concept_weak = concept_abs_max is None or concept_abs_max <= eps or (visual_margin is not None and abs(visual_margin) > concept_abs_max)
    return visual_margin < -eps and visual_ratio >= dominance_ratio and (concept_supports_true or concept_weak)


def build_prompt_tables(
    concept_df: pd.DataFrame,
    slide_table: pd.DataFrame,
    top_concepts: int,
    warning_log: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if concept_df.empty:
        warn_message("Top concept CSV is empty; prompt confusion analysis will be limited.", warning_log)
        empty = pd.DataFrame()
        return empty, empty, slide_table

    df = concept_df.copy()
    for column in ["concept_rank", "class_id", "label", "pred", "evidence", "weight", "contribution"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "correct" in df.columns:
        df["correct"] = parse_bool_series(df["correct"]).astype(bool)
    else:
        correct_map = slide_table.set_index("slide_id")["correct"].to_dict()
        df["correct"] = df["slide_id"].map(correct_map).fillna(False).astype(bool)

    df["occurrence"] = 1
    summary = (
        df.groupby(["scale", "class_id", "class_name", "class_type", "concept_text"], dropna=False)
        .agg(
            occurrences=("occurrence", "sum"),
            correct_count=("correct", "sum"),
            error_count=("correct", lambda values: int((~values.astype(bool)).sum())),
            mean_evidence=("evidence", "mean"),
            mean_weight=("weight", "mean"),
            mean_contribution=("contribution", "mean"),
            best_rank=("concept_rank", "min"),
        )
        .reset_index()
    )
    summary["error_rate_among_occurrences"] = summary.apply(
        lambda row: float(row["error_count"] / row["occurrences"]) if row["occurrences"] else math.nan,
        axis=1,
    )
    summary = summary.sort_values(
        ["class_type", "scale", "error_count", "mean_contribution", "occurrences"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)

    reliability = (
        df[df["class_type"] == "pred"]
        .groupby(["scale", "class_id", "class_name", "concept_text"], dropna=False)
        .agg(
            occurrences=("occurrence", "sum"),
            correct_count=("correct", "sum"),
            error_count=("correct", lambda values: int((~values.astype(bool)).sum())),
            correct_mean_evidence=("evidence", lambda values: float(np.nanmean(values[df.loc[values.index, "correct"]])) if np.any(df.loc[values.index, "correct"]) else math.nan),
            error_mean_evidence=("evidence", lambda values: float(np.nanmean(values[~df.loc[values.index, "correct"]])) if np.any(~df.loc[values.index, "correct"]) else math.nan),
            mean_contribution=("contribution", "mean"),
        )
        .reset_index()
    )
    reliability["error_rate_among_occurrences"] = reliability.apply(
        lambda row: float(row["error_count"] / row["occurrences"]) if row["occurrences"] else math.nan,
        axis=1,
    )
    reliability["reliability_hint"] = reliability.apply(classify_prompt_reliability, axis=1)
    reliability = reliability.sort_values(
        ["error_rate_among_occurrences", "error_count", "occurrences", "mean_contribution"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    global_error_rate = float((~slide_table["correct"]).mean()) if len(slide_table) else 0.0
    pred_reliability = reliability.set_index(["scale", "class_id", "concept_text"])
    df_for_flag = df[df["class_type"] == "pred"].copy()
    df_for_flag["concept_rank"] = pd.to_numeric(df_for_flag["concept_rank"], errors="coerce")
    df_for_flag = df_for_flag[df_for_flag["concept_rank"] <= top_concepts]

    per_slide_records = []
    for slide_id, group in df_for_flag.groupby("slide_id"):
        error_rates = []
        flagged_concepts = []
        for row in group.itertuples(index=False):
            key = (row.scale, int(row.class_id), row.concept_text)
            if key not in pred_reliability.index:
                continue
            rel_row = pred_reliability.loc[key]
            error_rate = safe_float(rel_row.get("error_rate_among_occurrences"))
            occurrences = safe_float(rel_row.get("occurrences"))
            if error_rate is None:
                continue
            error_rates.append(error_rate)
            if occurrences is not None and occurrences >= 3 and error_rate >= max(0.30, global_error_rate * 2.0):
                flagged_concepts.append(str(row.concept_text))
        score = float(np.mean(error_rates)) if error_rates else math.nan
        per_slide_records.append(
            {
                "slide_id": slide_id,
                "prompt_confusion_flag": bool(flagged_concepts or (not math.isnan(score) and score >= max(0.30, global_error_rate * 2.0))),
                "prompt_confusion_score": score,
                "prompt_confusion_note": " | ".join(flagged_concepts[:5]),
            }
        )
    prompt_flag_df = pd.DataFrame(per_slide_records)
    if not prompt_flag_df.empty:
        slide_table = slide_table.merge(prompt_flag_df, on="slide_id", how="left", suffixes=("", "_new"))
        for field in ["prompt_confusion_flag", "prompt_confusion_score", "prompt_confusion_note"]:
            new_field = f"{field}_new"
            if new_field in slide_table.columns:
                slide_table[field] = slide_table[new_field].combine_first(slide_table[field])
                slide_table = slide_table.drop(columns=[new_field])
        slide_table["prompt_confusion_flag"] = slide_table["prompt_confusion_flag"].fillna(False).astype(bool)

    return summary, reliability, slide_table


def classify_prompt_reliability(row: pd.Series) -> str:
    occurrences = safe_float(row.get("occurrences")) or 0.0
    error_rate = safe_float(row.get("error_rate_among_occurrences"))
    if occurrences < 3:
        return "rare"
    if error_rate is None:
        return "unknown"
    if error_rate >= 0.50:
        return "error_prone"
    if error_rate <= 0.10:
        return "reliable"
    return "mixed"


def assign_failure_labels(
    slide_table: pd.DataFrame,
    thresholds: dict[str, float],
    near_zero_eps: float,
    dominance_ratio: float,
    label_names: list[str],
) -> pd.DataFrame:
    df = slide_table.copy()
    low_conf_threshold = thresholds.get("pred_margin_q25", math.nan)
    csg_active_threshold = thresholds.get("csg_margin_quantile_threshold", math.nan)

    rows = []
    for row in df.itertuples(index=False):
        record = row._asdict()
        labels: list[str] = []
        notes: list[str] = []
        correct = bool(record.get("correct"))
        label = None if pd.isna(record.get("label")) else int(record.get("label"))
        pred = None if pd.isna(record.get("pred")) else int(record.get("pred"))
        low_margin = safe_float(record.get("low_true_vs_wrong_margin"))
        high_margin = safe_float(record.get("high_true_vs_wrong_margin"))
        visual_margin = safe_float(record.get("visual_effective_margin"))
        csg_margin = safe_float(record.get("csg_effective_margin"))
        pred_margin = safe_float(record.get("pred_margin"))
        dominant_source = record.get("dominant_source")
        dominant_ratio_value = safe_float(record.get("dominant_source_ratio"))

        if not correct:
            if bool(record.get("visual_override_candidate")):
                labels.append("visual_residual_override")
                notes.append("visual residual supports the wrong class and dominates available source margin.")

            concept_margins = [value for value in [low_margin, high_margin] if value is not None]
            if concept_margins and sum(value < -near_zero_eps for value in concept_margins) >= max(1, len(concept_margins)):
                labels.append("concept_wrong_class_drift")
                notes.append("concept evidence mainly supports the predicted wrong class.")

            if bool(record.get("low_high_conflict")):
                labels.append("low_high_conflict")
                notes.append("low and high evidence support different classes.")

            if high_margin is not None and high_margin < -near_zero_eps:
                low_abs = abs(low_margin) if low_margin is not None else 0.0
                if abs(high_margin) > low_abs:
                    labels.append("high_scale_dominant_wrong")
                    notes.append("high-scale concept evidence is the stronger wrong-class driver.")

            if low_margin is not None and low_margin < -near_zero_eps:
                high_abs = abs(high_margin) if high_margin is not None else 0.0
                if abs(low_margin) > high_abs:
                    labels.append("low_scale_dominant_wrong")
                    notes.append("low-scale concept evidence is the stronger wrong-class driver.")

            if csg_margin is not None and csg_margin < -near_zero_eps:
                if not math.isnan(csg_active_threshold) and abs(csg_margin) > max(csg_active_threshold, near_zero_eps):
                    labels.append("csg_misleading")
                    notes.append("CSG margin supports the predicted wrong class with non-trivial magnitude.")

            if bool(record.get("csg_inactive_or_zero_flag")):
                labels.append("csg_inactive_or_zero")
                if record.get("csg_export_capture_status") == "pair_export_may_need_refinement":
                    notes.append("CSG logits are non-zero but exported pair scores are near-zero; pair-level export may need refinement.")
                else:
                    notes.append("CSG appears inactive or very weak on this slide.")

            if pred_margin is not None and not math.isnan(low_conf_threshold) and pred_margin <= low_conf_threshold:
                labels.append("uncertain_low_margin")
                notes.append("final prediction margin is in the low-confidence region.")

            if bool(record.get("prompt_confusion_flag")):
                labels.append("prompt_confusion")
                note = str(record.get("prompt_confusion_note") or "").strip()
                notes.append(f"error-prone top concepts are present: {note}" if note else "error-prone top concepts are present.")

            if not labels:
                labels.append("unclassified_failure")
                notes.append("no heuristic failure pattern dominated this error case.")

        primary_failure_type = select_primary_failure_type(labels, correct)
        rows.append(
            {
                **record,
                "failure_labels": "|".join(labels),
                "primary_failure_type": primary_failure_type,
                "supporting_notes": " ".join(dict.fromkeys(notes)),
                "is_error": not correct,
                "true_vs_wrong_margin": safe_float(record.get("final_true_vs_wrong_margin")),
                "dominant_source_ratio": dominant_ratio_value,
                "label_name": class_name(label, label_names),
                "pred_name": class_name(pred, label_names),
            }
        )
    return pd.DataFrame(rows)


def select_primary_failure_type(labels: list[str], correct: bool) -> str:
    if correct:
        return "correct_prediction"
    priority = [
        "visual_residual_override",
        "concept_wrong_class_drift",
        "low_high_conflict",
        "high_scale_dominant_wrong",
        "low_scale_dominant_wrong",
        "csg_misleading",
        "uncertain_low_margin",
        "prompt_confusion",
        "csg_inactive_or_zero",
        "unclassified_failure",
    ]
    for item in priority:
        if item in labels:
            return item
    return "unclassified_failure"


def build_evidence_source_stats(slide_table: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {
        "all": slide_table,
        "correct": slide_table[slide_table["correct"] == True],
        "error": slide_table[slide_table["correct"] == False],
    }
    for group_name, group_df in groups.items():
        for source_name in ["low", "high", "visual", "csg"]:
            margin_col = (
                f"{source_name}_effective_margin"
                if source_name in {"visual", "csg"}
                else f"{source_name}_true_vs_wrong_margin"
            )
            support_col = f"{source_name}_support_class"
            available = group_df[margin_col].dropna()
            support = pd.to_numeric(group_df[support_col], errors="coerce")
            true_ratio = float((support == group_df["label"]).mean()) if len(group_df) else math.nan
            pred_ratio = float((support == group_df["pred"]).mean()) if len(group_df) else math.nan
            rows.append(
                {
                    "group": group_name,
                    "source": source_name,
                    "n_slides": len(group_df),
                    "n_available": int(available.notna().sum()),
                    "mean_margin": float(available.mean()) if not available.empty else math.nan,
                    "mean_abs_margin": float(available.abs().mean()) if not available.empty else math.nan,
                    "support_true_ratio": true_ratio,
                    "support_pred_ratio": pred_ratio,
                    "support_wrong_ratio": float(1.0 - true_ratio) if not math.isnan(true_ratio) else math.nan,
                    "dominant_ratio_mean": float(group_df.loc[group_df["dominant_source"] == source_name, "dominant_source_ratio"].mean())
                    if len(group_df)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_visual_diagnostics(slide_table: pd.DataFrame, near_zero_eps: float) -> pd.DataFrame:
    df = slide_table.copy()
    df["visual_supports_true"] = df["visual_support_class"] == df["label"]
    df["visual_supports_pred"] = df["visual_support_class"] == df["pred"]
    df["visual_supports_wrong"] = df["visual_support_class"].notna() & (df["visual_support_class"] != df["label"])
    df["visual_dominant"] = df["dominant_source"] == "visual"
    df["visual_margin_near_zero"] = df["visual_abs_margin"] <= near_zero_eps
    columns = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "pred_margin",
        "visual_true_vs_wrong_margin",
        "visual_effective_margin",
        "visual_abs_margin",
        "visual_effective_abs_margin",
        "visual_source_ratio",
        "visual_support_class",
        "visual_support_label",
        "visual_supports_true",
        "visual_supports_pred",
        "visual_supports_wrong",
        "visual_dominant",
        "visual_override_candidate",
        "visual_margin_near_zero",
        "dominant_source",
        "dominant_source_ratio",
        "concept_margin_mean",
        "concept_abs_margin_max",
    ]
    return df[columns].copy()


def build_csg_diagnostics(slide_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "pred_margin",
        "csg_logit_class_0",
        "csg_logit_class_1",
        "csg_true_vs_wrong_margin",
        "csg_effective_margin",
        "csg_abs_margin",
        "csg_effective_abs_margin",
        "csg_support_class",
        "csg_support_label",
        "csg_alpha",
        "csg_max_pair_score",
        "top_csg_pair_class_0",
        "top_csg_pair_class_1",
        "top_csg_pair_score_class_0",
        "top_csg_pair_score_class_1",
        "csg_pair_near_zero",
        "csg_margin_near_zero",
        "csg_logits_near_zero",
        "csg_export_capture_status",
        "csg_inactive_or_zero_flag",
    ]
    return slide_table[columns].copy()


def build_low_high_conflict_table(slide_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "pred_margin",
        "low_true_vs_wrong_margin",
        "high_true_vs_wrong_margin",
        "low_support_class",
        "low_support_label",
        "high_support_class",
        "high_support_label",
        "low_supports_true",
        "high_supports_true",
        "low_high_same_support",
        "low_high_conflict",
        "low_high_joint_state",
        "evidence_agreement",
    ]
    return slide_table[columns].copy()


def build_failure_type_counts(labeled_slides: pd.DataFrame) -> pd.DataFrame:
    error_df = labeled_slides[labeled_slides["correct"] == False].copy()
    any_label_counts: dict[str, int] = {}
    for value in error_df["failure_labels"].fillna(""):
        for item in [token.strip() for token in str(value).split("|") if token.strip()]:
            any_label_counts[item] = any_label_counts.get(item, 0) + 1

    primary_counts = error_df["primary_failure_type"].value_counts(dropna=False)
    rows = []
    total_errors = len(error_df)
    for failure_type in sorted(set(primary_counts.index.tolist()) | set(any_label_counts.keys())):
        count_as_primary = int(primary_counts.get(failure_type, 0))
        count_any = int(any_label_counts.get(failure_type, 0))
        rows.append(
            {
                "failure_type": failure_type,
                "count_as_primary": count_as_primary,
                "count_any_label": count_any,
                "proportion_among_errors": float(count_as_primary / total_errors) if total_errors else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["count_as_primary", "count_any_label"], ascending=[False, False]).reset_index(drop=True)


def summarize_conflict_counts(conflict_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for scope_name, group_df in {
        "all": conflict_df,
        "correct": conflict_df[conflict_df["correct"] == True],
        "error": conflict_df[conflict_df["correct"] == False],
    }.items():
        summary[scope_name] = {
            "low_support_true": int(group_df["low_supports_true"].sum()),
            "high_support_true": int(group_df["high_supports_true"].sum()),
            "both_support_true": int((group_df["low_high_joint_state"] == "both_support_true").sum()),
            "both_support_wrong": int((group_df["low_high_joint_state"] == "both_support_wrong").sum()),
            "conflict": int(group_df["low_high_conflict"].sum()),
            "num_slides": int(len(group_df)),
        }
    return summary


def summarize_visual_diagnostics(visual_df: pd.DataFrame) -> dict[str, float | str | None]:
    error_df = visual_df[visual_df["correct"] == False].copy()
    mean_ratio = safe_float(visual_df["visual_source_ratio"].mean()) if len(visual_df) else None
    error_wrong_ratio = safe_float(error_df["visual_supports_wrong"].mean()) if len(error_df) else None
    overall_wrong_ratio = safe_float(visual_df["visual_supports_wrong"].mean()) if len(visual_df) else None

    if error_wrong_ratio is not None and mean_ratio is not None and error_wrong_ratio >= 0.5 and mean_ratio >= 0.5:
        conclusion = "visual residual appears strong enough to justify an explicit gate in Step34."
        gate_init = "0.00 or 0.01"
        strategy = "start with a scalar visual gate, then compare a slide-adaptive gate only if scalar gating helps."
    elif mean_ratio is not None and mean_ratio >= 0.35:
        conclusion = "visual residual is substantial; a conservative visual gate should be compared in Step34."
        gate_init = "0.01 or 0.05"
        strategy = "start with a scalar visual gate and keep slide-adaptive gating as a secondary ablation."
    else:
        conclusion = "visual residual is not obviously over-strong on exported evidence."
        gate_init = "0.05"
        strategy = "keep the current residual path as baseline and only test lightweight gating ablations."

    return {
        "mean_visual_source_ratio": mean_ratio,
        "visual_supports_true_ratio": safe_float(visual_df["visual_supports_true"].mean()) if len(visual_df) else None,
        "visual_supports_pred_ratio": safe_float(visual_df["visual_supports_pred"].mean()) if len(visual_df) else None,
        "visual_supports_wrong_ratio": overall_wrong_ratio,
        "error_visual_supports_wrong_ratio": error_wrong_ratio,
        "conclusion": conclusion,
        "recommended_gate_init": gate_init,
        "recommended_strategy": strategy,
    }


def summarize_csg_diagnostics(csg_df: pd.DataFrame, near_zero_eps: float) -> dict[str, float | str | None]:
    mean_abs_logit = safe_float(csg_df["csg_effective_abs_margin"].mean()) if len(csg_df) else None
    max_abs_logit = safe_float(csg_df["csg_effective_abs_margin"].max()) if len(csg_df) else None
    mean_abs_raw = safe_float(csg_df["csg_abs_margin"].mean()) if len(csg_df) else None
    max_abs_raw = safe_float(csg_df["csg_abs_margin"].max()) if len(csg_df) else None
    mean_pair_score = safe_float(csg_df["csg_max_pair_score"].mean()) if len(csg_df) else None
    max_pair_score = safe_float(csg_df["csg_max_pair_score"].max()) if len(csg_df) else None
    mismatch_ratio = safe_float((csg_df["csg_export_capture_status"] == "pair_export_may_need_refinement").mean()) if len(csg_df) else None
    inactive_ratio = safe_float((csg_df["csg_export_capture_status"] == "inactive_or_very_weak").mean()) if len(csg_df) else None

    if mean_abs_logit is not None and mean_abs_logit > near_zero_eps and mean_pair_score is not None and mean_pair_score <= near_zero_eps:
        conclusion = "CSG logits are non-zero but exported pair scores are near-zero; pair-level export may need refinement."
    elif mean_abs_logit is not None and mean_abs_logit <= near_zero_eps and mean_pair_score is not None and mean_pair_score <= near_zero_eps:
        conclusion = "CSG appears inactive or very weak on exported fold0/test evidence."
    elif mismatch_ratio is not None and mismatch_ratio >= 0.25:
        conclusion = "CSG has some non-zero logit signal, but pair-level export may be under-reporting the effective contribution."
    else:
        conclusion = "CSG is present, but it remains much weaker than the visual and concept branches on exported evidence."

    return {
        "mean_abs_csg_margin": mean_abs_logit,
        "max_abs_csg_margin": max_abs_logit,
        "mean_abs_csg_raw_margin": mean_abs_raw,
        "max_abs_csg_raw_margin": max_abs_raw,
        "mean_top_pair_score": mean_pair_score,
        "max_top_pair_score": max_pair_score,
        "pair_export_mismatch_ratio": mismatch_ratio,
        "inactive_ratio": inactive_ratio,
        "conclusion": conclusion,
    }


def collect_error_case_table(labeled_slides: pd.DataFrame, top_error_cases: int) -> pd.DataFrame:
    error_df = labeled_slides[labeled_slides["correct"] == False].copy()
    error_df = error_df.sort_values(["pred_margin", "dominant_source_ratio"], ascending=[True, False])
    columns = [
        "slide_id",
        "label",
        "label_name",
        "pred",
        "pred_name",
        "pred_margin",
        "dominant_source",
        "dominant_source_ratio",
        "low_true_vs_wrong_margin",
        "high_true_vs_wrong_margin",
        "visual_effective_margin",
        "csg_effective_margin",
        "evidence_agreement",
        "primary_failure_type",
        "failure_labels",
        "supporting_notes",
    ]
    return error_df[columns].head(top_error_cases).copy()


def markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["| none |", "| --- |"]
    header = "| " + " | ".join(df.columns.astype(str).tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return [header, separator] + rows


def build_report(
    root: Path,
    input_dir: Path,
    output_dir: Path,
    slide_table: pd.DataFrame,
    failure_counts: pd.DataFrame,
    evidence_stats: pd.DataFrame,
    conflict_df: pd.DataFrame,
    visual_summary: dict[str, float | str | None],
    csg_summary: dict[str, float | str | None],
    conflict_summary: dict[str, dict[str, int]],
    prompt_summary: pd.DataFrame,
    reliability_preview: pd.DataFrame,
    error_case_table: pd.DataFrame,
    manifest: dict,
    warning_log: list[str],
) -> str:
    metrics = manifest.get("metrics", {}) if isinstance(manifest, dict) else {}
    correct_count = int(slide_table["correct"].sum()) if len(slide_table) else 0
    error_count = int((~slide_table["correct"]).sum()) if len(slide_table) else 0

    prompt_error_top = reliability_preview.head(10)[
        ["scale", "class_name", "concept_text", "occurrences", "error_count", "error_rate_among_occurrences", "reliability_hint"]
    ] if not reliability_preview.empty else pd.DataFrame()

    lines = [
        "# Step33 Evidence Failure / Conflict Analysis",
        "",
        "## Scope",
        "- This step does not train the model.",
        "- This step does not modify the model.",
        "- This step only analyzes Step32 evidence export.",
        "",
        "## Inputs / Outputs",
        f"- Input directory: `{relative_path_str(root, input_dir)}`",
        f"- Output directory: `{relative_path_str(root, output_dir)}`",
        "- Read files:",
        "  - `stage32_slide_evidence_summary.csv`",
        "  - `stage32_top_concepts_long.csv`",
        "  - `stage32_top_csg_pairs.csv`",
        "  - `stage32_error_cases.csv`",
        "  - `stage32_manifest.json`",
        "  - `stage32_evidence_export_report.md` (optional)",
        "",
        "## Exported Slide Counts",
        f"- Slides: `{len(slide_table)}`",
        f"- Correct: `{correct_count}`",
        f"- Error: `{error_count}`",
        "",
        "## Repeated Metrics",
        f"- test AUC: `{format_float(safe_float(metrics.get('test_auc')) )}`",
        f"- test ACC: `{format_float(safe_float(metrics.get('test_acc')) )}`",
        f"- test F1: `{format_float(safe_float(metrics.get('test_f1')) )}`",
        f"- Balanced ACC: `{format_float(safe_float(metrics.get('balanced_acc')) )}`",
        f"- PR-AUC: `{format_float(safe_float(metrics.get('pr_auc')) )}`",
        "",
        "## Evidence Source Magnitude Comparison",
    ]
    for source_name in ["low", "high", "visual", "csg"]:
        row = evidence_stats[(evidence_stats["group"] == "all") & (evidence_stats["source"] == source_name)]
        if row.empty:
            continue
        lines.append(
            f"- `{source_name}` mean abs margin: `{format_float(safe_float(row.iloc[0]['mean_abs_margin']))}`"
        )

    lines.extend(
        [
            "",
            "## Visual Residual Diagnosis",
            f"- Conclusion: {visual_summary.get('conclusion')}",
            "- Ratios and dominance below use `visual_alpha * visual_margin`, not the raw visual logits alone.",
            f"- Mean visual source ratio: `{format_float(safe_float(visual_summary.get('mean_visual_source_ratio')) )}`",
            f"- Visual supports true ratio: `{format_float(safe_float(visual_summary.get('visual_supports_true_ratio')) )}`",
            f"- Visual supports predicted ratio: `{format_float(safe_float(visual_summary.get('visual_supports_pred_ratio')) )}`",
            f"- Visual supports wrong ratio on errors: `{format_float(safe_float(visual_summary.get('error_visual_supports_wrong_ratio')) )}`",
            f"- Suggested Step34 visual gate init: `{visual_summary.get('recommended_gate_init')}`",
            f"- Suggested Step34 strategy: {visual_summary.get('recommended_strategy')}",
            "",
            "## CSG Diagnosis",
            f"- Conclusion: {csg_summary.get('conclusion')}",
            "- CSG margin below is the effective contribution `csg_alpha * csg_margin`; raw branch magnitude is listed separately.",
            f"- Mean abs effective CSG margin: `{format_float(safe_float(csg_summary.get('mean_abs_csg_margin')) , 6)}`",
            f"- Max abs effective CSG margin: `{format_float(safe_float(csg_summary.get('max_abs_csg_margin')) , 6)}`",
            f"- Mean abs raw CSG margin: `{format_float(safe_float(csg_summary.get('mean_abs_csg_raw_margin')) , 6)}`",
            f"- Max abs raw CSG margin: `{format_float(safe_float(csg_summary.get('max_abs_csg_raw_margin')) , 6)}`",
            f"- Mean top CSG pair score: `{format_float(safe_float(csg_summary.get('mean_top_pair_score')) , 6)}`",
            f"- Max top CSG pair score: `{format_float(safe_float(csg_summary.get('max_top_pair_score')) , 6)}`",
            f"- Pair export mismatch ratio: `{format_float(safe_float(csg_summary.get('pair_export_mismatch_ratio')) )}`",
            f"- Inactive ratio: `{format_float(safe_float(csg_summary.get('inactive_ratio')) )}`",
            "",
            "## Low / High Conflict Diagnosis",
            f"- All slides conflict count: `{conflict_summary['all']['conflict']}` / `{conflict_summary['all']['num_slides']}`",
            f"- Correct slides conflict count: `{conflict_summary['correct']['conflict']}` / `{conflict_summary['correct']['num_slides']}`",
            f"- Error slides conflict count: `{conflict_summary['error']['conflict']}` / `{conflict_summary['error']['num_slides']}`",
            f"- Error slides both-support-wrong count: `{conflict_summary['error']['both_support_wrong']}`",
            f"- Error slides both-support-true count: `{conflict_summary['error']['both_support_true']}`",
        ]
    )

    if conflict_summary["error"]["conflict"] > 0 and conflict_summary["error"]["conflict"] >= conflict_summary["correct"]["conflict"]:
        lines.append("- Recommendation: Step35 should compare a low-high consistency loss or a high-branch gate.")
    if conflict_summary["error"]["both_support_wrong"] > conflict_summary["error"]["both_support_true"]:
        lines.append("- Recommendation: high-scale or concept-level margin control is worth testing because wrong-class concept agreement exists in errors.")

    lines.extend(
        [
            "",
            "## Prompt Confusion Diagnosis",
            "- Error-prone concept preview:",
            *markdown_table(prompt_error_top),
            "",
            "## Failure Type Counts",
            *markdown_table(failure_counts[["failure_type", "count_as_primary", "count_any_label", "proportion_among_errors"]]),
            "",
            "## Top Error Cases",
            *markdown_table(error_case_table),
            "",
            "## Step34 Recommendation",
            f"- Evidence-level gated fusion: {'yes' if 'gate' in str(visual_summary.get('recommended_strategy', '')).lower() else 'consider'}",
            f"- Start with: {visual_summary.get('recommended_strategy')}",
            "- Suggested ablations:",
            "  - scalar visual gate init `0.00` vs `0.01` vs current-equivalent `0.05`",
            "  - keep / remove CSG residual path while visual gate is active",
            "  - scalar-only gate vs slide-adaptive gate if scalar gating helps",
            "",
            "## Step35 / Step36 Recommendation",
            "- Evidence consistency / margin loss: yes, if Step34 shows that source conflict remains a major error pattern.",
            "- Most worth constraining first: the high-scale concept branch when it dominates wrong-class errors, then low-high consistency, then CSG only after export fidelity is clarified.",
            "",
            "## Warnings",
        ]
    )

    if warning_log:
        lines.extend([f"- {message}" for message in warning_log])
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def build_recommendations_json(
    slide_table: pd.DataFrame,
    visual_summary: dict[str, float | str | None],
    csg_summary: dict[str, float | str | None],
    conflict_summary: dict[str, dict[str, int]],
    failure_counts: pd.DataFrame,
) -> dict[str, object]:
    top_failure = None
    if not failure_counts.empty:
        top_failure = failure_counts.iloc[0]["failure_type"]
    return {
        "step": 33,
        "num_slides": int(len(slide_table)),
        "num_errors": int((~slide_table["correct"]).sum()) if len(slide_table) else 0,
        "top_primary_failure_type": top_failure,
        "visual_residual": visual_summary,
        "csg": csg_summary,
        "low_high_conflict": conflict_summary,
        "step34_recommendation": {
            "use_evidence_level_gated_fusion": True,
            "start_with_scalar_gate": True,
            "consider_slide_adaptive_gate_after_scalar": True,
            "suggested_visual_gate_init": visual_summary.get("recommended_gate_init"),
            "keep_csg_gate_as_ablation": True,
        },
        "step35_recommendation": {
            "consider_consistency_loss": True,
            "prioritize_high_scale_margin_control": conflict_summary["error"]["both_support_wrong"] > 0,
            "prioritize_low_high_consistency": conflict_summary["error"]["conflict"] > 0,
            "defer_csg_constraint_until_export_is_clarified": "pair_export" in str(csg_summary.get("conclusion", "")).lower(),
        },
    }


def main() -> int:
    args = parse_args()
    root = DEFAULT_ROOT
    warning_log: list[str] = []

    label_names = [item.strip() for item in str(args.label_names).split(",") if item.strip()]
    if len(label_names) < 2:
        label_names = ["Adenocarcinoma", "NonAdenocarcinoma"]

    input_dir = resolve_path(root, args.input_dir)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_csv = input_dir / "stage32_slide_evidence_summary.csv"
    concept_csv = input_dir / "stage32_top_concepts_long.csv"
    csg_csv = input_dir / "stage32_top_csg_pairs.csv"
    error_csv = input_dir / "stage32_error_cases.csv"
    manifest_json = input_dir / "stage32_manifest.json"
    report_md = input_dir / "stage32_evidence_export_report.md"

    slide_df = safe_read_csv(slide_csv, warning_log, required=True)
    concept_df = safe_read_csv(concept_csv, warning_log, required=False)
    csg_pair_df = safe_read_csv(csg_csv, warning_log, required=False)
    safe_read_csv(error_csv, warning_log, required=False)
    manifest = safe_read_json(manifest_json, warning_log)
    safe_read_text(report_md, warning_log)

    slide_table, thresholds = build_slide_table(
        slide_df=slide_df,
        csg_pair_df=csg_pair_df,
        label_names=label_names,
        near_zero_eps=args.near_zero_eps,
        dominance_ratio=args.dominance_ratio,
        low_margin_quantile=args.low_margin_quantile,
        warning_log=warning_log,
    )

    prompt_summary, reliability_preview, slide_table = build_prompt_tables(
        concept_df=concept_df,
        slide_table=slide_table,
        top_concepts=args.top_concepts,
        warning_log=warning_log,
    )

    labeled_slides = assign_failure_labels(
        slide_table=slide_table,
        thresholds=thresholds,
        near_zero_eps=args.near_zero_eps,
        dominance_ratio=args.dominance_ratio,
        label_names=label_names,
    )
    evidence_stats = build_evidence_source_stats(labeled_slides, label_names)
    conflict_df = build_low_high_conflict_table(labeled_slides)
    visual_df = build_visual_diagnostics(labeled_slides, args.near_zero_eps)
    csg_diag_df = build_csg_diagnostics(labeled_slides)
    failure_counts = build_failure_type_counts(labeled_slides)
    error_case_table = collect_error_case_table(labeled_slides, args.top_error_cases)

    visual_summary = summarize_visual_diagnostics(visual_df)
    csg_summary = summarize_csg_diagnostics(csg_diag_df, args.near_zero_eps)
    conflict_summary = summarize_conflict_counts(conflict_df)

    slide_output = labeled_slides.copy()
    slide_output.to_csv(output_dir / "stage33_slide_failure_labels.csv", index=False, encoding="utf-8")
    labeled_slides[labeled_slides["correct"] == False].to_csv(
        output_dir / "stage33_error_failure_cases.csv", index=False, encoding="utf-8"
    )
    evidence_stats.to_csv(output_dir / "stage33_evidence_source_stats.csv", index=False, encoding="utf-8")
    conflict_df.to_csv(output_dir / "stage33_low_high_conflict_summary.csv", index=False, encoding="utf-8")
    visual_df.to_csv(output_dir / "stage33_visual_residual_diagnostics.csv", index=False, encoding="utf-8")
    csg_diag_df.to_csv(output_dir / "stage33_csg_diagnostics.csv", index=False, encoding="utf-8")
    prompt_summary.to_csv(output_dir / "stage33_prompt_confusion_summary.csv", index=False, encoding="utf-8")
    reliability_preview.to_csv(output_dir / "stage33_prompt_reliability_preview.csv", index=False, encoding="utf-8")
    failure_counts.to_csv(output_dir / "stage33_failure_type_counts.csv", index=False, encoding="utf-8")

    recommendations = build_recommendations_json(
        slide_table=labeled_slides,
        visual_summary=visual_summary,
        csg_summary=csg_summary,
        conflict_summary=conflict_summary,
        failure_counts=failure_counts,
    )
    (output_dir / "stage33_recommendations.json").write_text(
        json.dumps(recommendations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_text = build_report(
        root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        slide_table=labeled_slides,
        failure_counts=failure_counts,
        evidence_stats=evidence_stats,
        conflict_df=conflict_df,
        visual_summary=visual_summary,
        csg_summary=csg_summary,
        conflict_summary=conflict_summary,
        prompt_summary=prompt_summary,
        reliability_preview=reliability_preview,
        error_case_table=error_case_table,
        manifest=manifest,
        warning_log=warning_log,
    )
    (output_dir / "stage33_evidence_failure_report.md").write_text(report_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
