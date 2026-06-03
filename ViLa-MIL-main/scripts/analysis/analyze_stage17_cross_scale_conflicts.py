from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = Path("results_stage9/stage13_rce_evidence_export_fold0_test_full")
DEFAULT_GRAPH_DIR = Path("results_stage9/stage14_concept_class_graph_fold0")
DEFAULT_STAGE16_DIR = Path("results_stage9/stage16_failure_case_narratives_fold0")
DEFAULT_OUT_DIR = Path("results_stage9/stage17_cross_scale_conflict_analysis_fold0")
CLASS_NAME_MAP = {0: "Adenocarcinoma", 1: "NonAdenocarcinoma"}
CONFLICT_TYPES = [
    "consistent_correct_support",
    "consistent_wrong_class_drift",
    "high_scale_dominant_correct",
    "high_scale_dominant_wrong",
    "low_scale_dominant_correct",
    "low_scale_dominant_wrong",
    "weak_or_mixed_conflict",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Step17 cross-scale evidence conflicts on fold0 full-export slides.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to ViLa-MIL-main root.")
    parser.add_argument(
        "--evidence_dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing Step13d evidence export files.",
    )
    parser.add_argument(
        "--graph_dir",
        type=Path,
        default=DEFAULT_GRAPH_DIR,
        help="Directory containing Step14 graph outputs.",
    )
    parser.add_argument(
        "--stage16_dir",
        type=Path,
        default=DEFAULT_STAGE16_DIR,
        help="Directory containing Step16 failure-case narrative outputs.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for Step17 cross-scale conflict outputs.",
    )
    parser.add_argument("--top_k_concepts", type=int, default=5, help="Top concepts per slide-scale-class used in aggregation.")
    return parser.parse_args()


def resolve_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


def warn_message(message: str, warning_log: list[str]) -> None:
    text = f"[Step17 warning] {message}"
    print(text)
    warning_log.append(message)


def safe_read_csv(path: Path, warning_log: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warn_message(f"Missing CSV: {path}", warning_log)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn_message(f"Failed to read CSV {path}: {exc}", warning_log)
        return None


def ensure_prediction_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    columns = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "prob_0",
        "prob_1",
        "low_visual_logit_0",
        "low_visual_logit_1",
        "high_visual_logit_0",
        "high_visual_logit_1",
    ]
    if df is None:
        return pd.DataFrame(columns=columns)
    required = ["slide_id", "label", "pred", "correct", "prob_0", "prob_1"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Prediction CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=columns)
    result = df.copy()
    for column in columns:
        if column in result.columns and column != "slide_id":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["label"] = result["label"].fillna(-1).astype(int)
    result["pred"] = result["pred"].fillna(-1).astype(int)
    result["correct"] = result["correct"].fillna(0).astype(int)
    return result


def ensure_top_concepts_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    columns = [
        "slide_id",
        "scale",
        "class_id",
        "concept_id",
        "concept_text",
        "evidence_score",
        "prompt_weight",
        "rank",
        "prompt_id",
    ]
    if df is None:
        return pd.DataFrame(columns=columns)
    required = ["slide_id", "scale", "class_id", "concept_id", "evidence_score", "prompt_weight", "rank"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Top concepts CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=columns)
    result = df.copy()
    for column in ["class_id", "evidence_score", "prompt_weight", "rank", "prompt_id"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        else:
            result[column] = np.nan
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    result["rank"] = result["rank"].fillna(9999).astype(int)
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    return result


def ensure_edges_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    columns = [
        "scale",
        "class_id",
        "class_name",
        "concept_id",
        "concept_text",
        "edge_strength",
        "slide_coverage",
        "mean_rank",
        "passes_min_count",
    ]
    if df is None:
        return pd.DataFrame(columns=columns)
    required = ["scale", "class_id", "concept_id", "edge_strength"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Edges CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=columns)
    result = df.copy()
    for column in ["class_id", "edge_strength", "slide_coverage", "mean_rank"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "passes_min_count" in result.columns:
        result["passes_min_count"] = result["passes_min_count"].fillna(False).astype(bool)
    else:
        result["passes_min_count"] = True
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["class_name"] = result.get("class_name", pd.Series(dtype=object))
    if "class_name" not in result.columns:
        result["class_name"] = result["class_id"].map(CLASS_NAME_MAP).fillna("Unknown")
    else:
        result["class_name"] = result["class_name"].fillna(result["class_id"].map(CLASS_NAME_MAP)).fillna("Unknown")
    result["concept_text"] = result.get("concept_text", pd.Series(dtype=object))
    if "concept_text" not in result.columns:
        result["concept_text"] = ""
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    return result


def ensure_failure_cases_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["slide_id", "error_direction", "low_scale_relation", "high_scale_relation"])
    required = ["slide_id", "error_direction", "low_scale_relation", "high_scale_relation"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Step16 failure cases CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=required)
    return df.copy()


def ensure_selected_narratives_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["slide_id", "correct", "selection_reason", "narrative_summary"])
    required = ["slide_id", "correct"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Step16 selected narratives CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=["slide_id", "correct", "selection_reason", "narrative_summary"])
    return df.copy()


def class_name(class_id: int) -> str:
    return CLASS_NAME_MAP.get(int(class_id), "Unknown")


def build_merged_concepts(concepts_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    if concepts_df.empty:
        return concepts_df.copy()
    edge_subset = edges_df[
        [column for column in ["scale", "class_id", "concept_id", "edge_strength", "slide_coverage", "mean_rank", "passes_min_count"] if column in edges_df.columns]
    ].drop_duplicates(["scale", "class_id", "concept_id"])
    merged = concepts_df.merge(edge_subset, on=["scale", "class_id", "concept_id"], how="left")
    merged["edge_strength"] = pd.to_numeric(merged["edge_strength"], errors="coerce").fillna(1.0)
    merged["slide_coverage"] = pd.to_numeric(merged["slide_coverage"], errors="coerce")
    merged["mean_rank"] = pd.to_numeric(merged["mean_rank"], errors="coerce")
    merged["passes_min_count"] = merged["passes_min_count"].fillna(False).astype(bool)
    merged["rank_factor"] = 1.0 / np.maximum(merged["rank"].astype(float), 1.0)
    merged["aggregated_component"] = (
        merged["evidence_score"].astype(float)
        * merged["prompt_weight"].astype(float)
        * merged["edge_strength"].astype(float)
        * merged["rank_factor"].astype(float)
    )
    return merged


def aggregate_scale_scores(merged_concepts_df: pd.DataFrame, top_k_concepts: int) -> pd.DataFrame:
    if merged_concepts_df.empty:
        return pd.DataFrame(
            columns=[
                "slide_id",
                "scale",
                "class_id",
                "scale_class_score",
                "scale_class_top_concepts",
                "scale_class_top_concept_ids",
            ]
        )

    filtered = (
        merged_concepts_df.sort_values(
            ["slide_id", "scale", "class_id", "rank", "evidence_score", "prompt_weight", "edge_strength"],
            ascending=[True, True, True, True, False, False, False],
        )
        .groupby(["slide_id", "scale", "class_id"], as_index=False, group_keys=False)
        .head(int(top_k_concepts))
        .copy()
    )

    rows = []
    for keys, group in filtered.groupby(["slide_id", "scale", "class_id"], dropna=False):
        slide_id, scale, class_id = keys
        ordered = group.sort_values(
            ["rank", "evidence_score", "prompt_weight", "edge_strength"],
            ascending=[True, False, False, False],
        )
        concept_strings = []
        concept_ids = []
        for _, row in ordered.iterrows():
            concept_ids.append(str(row["concept_id"]))
            concept_strings.append(
                f"{row['concept_id']} (ev={float(row['evidence_score']):.4f}, pw={float(row['prompt_weight']):.4f}, edge={float(row['edge_strength']):.4f}, rank={int(row['rank'])})"
            )
        rows.append(
            {
                "slide_id": str(slide_id),
                "scale": str(scale),
                "class_id": int(class_id),
                "scale_class_score": float(ordered["aggregated_component"].sum()),
                "scale_class_top_concepts": " | ".join(concept_strings),
                "scale_class_top_concept_ids": json.dumps(concept_ids, ensure_ascii=True),
            }
        )
    return pd.DataFrame(rows)


def pivot_scale_scores(scale_scores_df: pd.DataFrame) -> pd.DataFrame:
    if scale_scores_df.empty:
        return pd.DataFrame(columns=["slide_id"])
    rows = []
    for slide_id, slide_group in scale_scores_df.groupby("slide_id", dropna=False):
        row: dict[str, object] = {"slide_id": str(slide_id)}
        for scale in ["low", "high"]:
            scale_group = slide_group.loc[slide_group["scale"] == scale].copy()
            for class_id in [0, 1]:
                class_group = scale_group.loc[scale_group["class_id"] == class_id].copy()
                prefix = f"{scale}_score_{class_id}"
                row[prefix] = float(class_group["scale_class_score"].iloc[0]) if not class_group.empty else np.nan
                text_col = f"{scale}_top_concepts_{class_id}"
                id_col = f"{scale}_top_concept_ids_{class_id}"
                row[text_col] = class_group["scale_class_top_concepts"].iloc[0] if not class_group.empty else ""
                row[id_col] = class_group["scale_class_top_concept_ids"].iloc[0] if not class_group.empty else "[]"
        rows.append(row)
    return pd.DataFrame(rows)


def support_class_from_margin(margin: float) -> int | None:
    if pd.isna(margin):
        return None
    return 0 if float(margin) >= 0 else 1


def predicted_scale_relation(pred: int, low_support: int | None, high_support: int | None) -> str:
    low_match = low_support == int(pred)
    high_match = high_support == int(pred)
    if low_match and high_match:
        return "both_scales_support_predicted"
    if (not low_match) and high_match:
        return "high_only_supports_predicted"
    if low_match and (not high_match):
        return "low_only_supports_predicted"
    return "neither_scale_supports_predicted"


def classify_conflict_type(correct: int, pred: int, low_support: int | None, high_support: int | None) -> str:
    relation = predicted_scale_relation(pred=pred, low_support=low_support, high_support=high_support)
    if relation == "both_scales_support_predicted":
        return "consistent_correct_support" if int(correct) == 1 else "consistent_wrong_class_drift"
    if relation == "high_only_supports_predicted":
        return "high_scale_dominant_correct" if int(correct) == 1 else "high_scale_dominant_wrong"
    if relation == "low_only_supports_predicted":
        return "low_scale_dominant_correct" if int(correct) == 1 else "low_scale_dominant_wrong"
    return "weak_or_mixed_conflict"


def build_slide_scores_df(
    prediction_df: pd.DataFrame,
    pivot_scores_df: pd.DataFrame,
    selected_narratives_df: pd.DataFrame,
) -> pd.DataFrame:
    base = prediction_df.merge(pivot_scores_df, on="slide_id", how="left")
    if not selected_narratives_df.empty:
        selected_subset = selected_narratives_df[
            [column for column in ["slide_id", "selection_reason", "narrative_summary"] if column in selected_narratives_df.columns]
        ].drop_duplicates("slide_id")
        base = base.merge(selected_subset, on="slide_id", how="left")
    else:
        base["selection_reason"] = ""
        base["narrative_summary"] = ""

    base["label_name"] = base["label"].map(CLASS_NAME_MAP).fillna("Unknown")
    base["pred_name"] = base["pred"].map(CLASS_NAME_MAP).fillna("Unknown")
    base["low_margin"] = pd.to_numeric(base.get("low_score_0"), errors="coerce") - pd.to_numeric(base.get("low_score_1"), errors="coerce")
    base["high_margin"] = pd.to_numeric(base.get("high_score_0"), errors="coerce") - pd.to_numeric(base.get("high_score_1"), errors="coerce")
    base["low_support_class"] = base["low_margin"].map(support_class_from_margin)
    base["high_support_class"] = base["high_margin"].map(support_class_from_margin)
    base["low_support_class_name"] = base["low_support_class"].map(lambda value: class_name(value) if value is not None else "Unknown")
    base["high_support_class_name"] = base["high_support_class"].map(lambda value: class_name(value) if value is not None else "Unknown")
    base["low_high_agree"] = (
        (base["low_support_class"].notna())
        & (base["high_support_class"].notna())
        & (base["low_support_class"] == base["high_support_class"])
    ).astype(int)
    base["predicted_scale_relation"] = base.apply(
        lambda row: predicted_scale_relation(
            pred=int(row["pred"]),
            low_support=None if pd.isna(row["low_support_class"]) else int(row["low_support_class"]),
            high_support=None if pd.isna(row["high_support_class"]) else int(row["high_support_class"]),
        ),
        axis=1,
    )
    base["conflict_type"] = base.apply(
        lambda row: classify_conflict_type(
            correct=int(row["correct"]),
            pred=int(row["pred"]),
            low_support=None if pd.isna(row["low_support_class"]) else int(row["low_support_class"]),
            high_support=None if pd.isna(row["high_support_class"]) else int(row["high_support_class"]),
        ),
        axis=1,
    )
    return base


def build_summary_rows(slide_scores_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = int(len(slide_scores_df))
    if total == 0:
        return rows
    rows.append(
        {
            "summary_scope": "overall",
            "group_key": "all_slides",
            "group_value": "all_slides",
            "conflict_type": "all",
            "count": total,
            "rate": 1.0,
        }
    )
    for conflict_type, count in slide_scores_df["conflict_type"].value_counts().sort_index().items():
        rows.append(
            {
                "summary_scope": "overall",
                "group_key": "all_slides",
                "group_value": "all_slides",
                "conflict_type": str(conflict_type),
                "count": int(count),
                "rate": float(count / total),
            }
        )

    summary_specs = [
        ("correctness", "correct"),
        ("true_label", "label_name"),
        ("predicted_label", "pred_name"),
    ]
    for scope, column in summary_specs:
        for group_value, group_df in slide_scores_df.groupby(column, dropna=False):
            group_total = int(len(group_df))
            for conflict_type in CONFLICT_TYPES:
                count = int((group_df["conflict_type"] == conflict_type).sum())
                rows.append(
                    {
                        "summary_scope": scope,
                        "group_key": column,
                        "group_value": str(group_value),
                        "conflict_type": conflict_type,
                        "count": count,
                        "rate": float(count / group_total) if group_total > 0 else np.nan,
                    }
                )
    return rows


def expected_conflict_from_stage16(row: pd.Series) -> str:
    low_relation = str(row.get("low_scale_relation", ""))
    high_relation = str(row.get("high_scale_relation", ""))
    if low_relation == "predicted_class_stronger" and high_relation == "predicted_class_stronger":
        return "consistent_wrong_class_drift"
    if high_relation == "predicted_class_stronger" and low_relation != "predicted_class_stronger":
        return "high_scale_dominant_wrong"
    if low_relation == "predicted_class_stronger" and high_relation != "predicted_class_stronger":
        return "low_scale_dominant_wrong"
    return "weak_or_mixed_conflict"


def build_failure_pattern_cases_df(
    slide_scores_df: pd.DataFrame,
    failure_cases_df: pd.DataFrame,
) -> pd.DataFrame:
    pattern_df = slide_scores_df.loc[
        slide_scores_df["conflict_type"].isin(
            [
                "consistent_wrong_class_drift",
                "high_scale_dominant_wrong",
                "low_scale_dominant_wrong",
                "weak_or_mixed_conflict",
            ]
        )
        | (slide_scores_df["correct"] == 0)
    ].copy()
    pattern_df["step16_failure_case"] = 0
    pattern_df["step16_expected_conflict_type"] = ""
    pattern_df["step16_pattern_match"] = np.nan

    if not failure_cases_df.empty:
        failure_cases_df = failure_cases_df.copy()
        failure_cases_df["step16_expected_conflict_type"] = failure_cases_df.apply(expected_conflict_from_stage16, axis=1)
        merge_subset = failure_cases_df[["slide_id", "error_direction", "step16_expected_conflict_type"]].drop_duplicates("slide_id")
        pattern_df = pattern_df.drop(columns=["step16_expected_conflict_type"], errors="ignore")
        pattern_df = pattern_df.merge(merge_subset, on="slide_id", how="left")
        if "error_direction" not in pattern_df.columns:
            pattern_df["error_direction"] = np.nan
        pattern_df["step16_failure_case"] = pattern_df["error_direction"].notna().astype(int)
        pattern_df["step16_pattern_match"] = np.where(
            pattern_df["step16_failure_case"] == 1,
            (pattern_df["conflict_type"] == pattern_df["step16_expected_conflict_type"]).astype(int),
            np.nan,
        )
    pattern_df = pattern_df.sort_values(
        ["correct", "conflict_type", "slide_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return pattern_df


def build_cross_scale_summary_df(
    slide_scores_df: pd.DataFrame,
    failure_pattern_cases_df: pd.DataFrame,
    warning_log: list[str],
    generated_files: list[Path],
) -> pd.DataFrame:
    rows = []
    total = int(len(slide_scores_df))
    rows.append({"section": "inputs", "metric": "slide_count", "value": total, "note": ""})
    if total > 0:
        rows.append(
            {
                "section": "overall",
                "metric": "low_high_agreement_rate",
                "value": float(slide_scores_df["low_high_agree"].mean()),
                "note": "",
            }
        )
        rows.append(
            {
                "section": "overall",
                "metric": "incorrect_count",
                "value": int((slide_scores_df["correct"] == 0).sum()),
                "note": "",
            }
        )
        for conflict_type in CONFLICT_TYPES:
            rows.append(
                {
                    "section": "overall",
                    "metric": conflict_type,
                    "value": int((slide_scores_df["conflict_type"] == conflict_type).sum()),
                    "note": "",
                }
            )
    wrong_high = slide_scores_df.loc[slide_scores_df["conflict_type"] == "high_scale_dominant_wrong", "slide_id"].astype(str).tolist()
    wrong_drift = slide_scores_df.loc[slide_scores_df["conflict_type"] == "consistent_wrong_class_drift", "slide_id"].astype(str).tolist()
    rows.append({"section": "failure_patterns", "metric": "high_scale_dominant_wrong_slides", "value": int(len(wrong_high)), "note": json.dumps(wrong_high, ensure_ascii=True)})
    rows.append({"section": "failure_patterns", "metric": "consistent_wrong_class_drift_slides", "value": int(len(wrong_drift)), "note": json.dumps(wrong_drift, ensure_ascii=True)})
    step16_matches = failure_pattern_cases_df.loc[failure_pattern_cases_df["step16_failure_case"] == 1, "step16_pattern_match"]
    if not step16_matches.empty:
        rows.append(
            {
                "section": "step16_mapping",
                "metric": "mapped_failure_case_count",
                "value": int(len(step16_matches)),
                "note": "",
            }
        )
        rows.append(
            {
                "section": "step16_mapping",
                "metric": "mapped_failure_case_match_count",
                "value": int(step16_matches.fillna(0).sum()),
                "note": "",
            }
        )
    rows.append({"section": "quality", "metric": "warning_count", "value": int(len(warning_log)), "note": ""})
    for path in generated_files:
        rows.append({"section": "outputs", "metric": path.name, "value": 1, "note": str(path)})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows available._"
    safe_df = df.fillna("NA").astype(str)
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def build_report(
    root: Path,
    evidence_dir: Path,
    graph_dir: Path,
    stage16_dir: Path,
    slide_scores_df: pd.DataFrame,
    conflict_summary_df: pd.DataFrame,
    failure_pattern_cases_df: pd.DataFrame,
    warning_log: list[str],
) -> str:
    total = int(len(slide_scores_df))
    correct_df = slide_scores_df.loc[slide_scores_df["correct"] == 1].copy()
    incorrect_df = slide_scores_df.loc[slide_scores_df["correct"] == 0].copy()
    agreement_rate = float(slide_scores_df["low_high_agree"].mean()) if total > 0 else np.nan

    correct_summary = conflict_summary_df.loc[conflict_summary_df["summary_scope"] == "correctness"].copy()
    correct_summary = correct_summary.loc[correct_summary["group_value"].isin(["0", "1"])].copy()
    if not correct_summary.empty:
        correct_summary["group_value"] = correct_summary["group_value"].map({"0": "incorrect", "1": "correct"}).fillna(correct_summary["group_value"])

    failure_mapping_df = failure_pattern_cases_df.loc[failure_pattern_cases_df["step16_failure_case"] == 1].copy()
    high_wrong_slides = slide_scores_df.loc[slide_scores_df["conflict_type"] == "high_scale_dominant_wrong", "slide_id"].astype(str).tolist()
    wrong_drift_slides = slide_scores_df.loc[slide_scores_df["conflict_type"] == "consistent_wrong_class_drift", "slide_id"].astype(str).tolist()

    lines = [
        "# Stage17 Cross-Scale Evidence Conflict Analysis",
        "",
        "Step17 is a post-hoc cross-scale evidence analysis over the fold0 test full export. It does not modify the model and does not run training.",
        "",
        "## Inputs",
        "",
        f"- root: `{root}`",
        f"- evidence_dir: `{evidence_dir}`",
        f"- graph_dir: `{graph_dir}`",
        f"- stage16_dir: `{stage16_dir}`",
        f"- fold0_slide_count: `{total}`",
        "",
        "## Scale-Evidence Definition",
        "",
        "For each slide, scale, and class, Step17 aggregates the top concept paths using:",
        "",
        "`aggregated_component = evidence_score * prompt_weight * edge_strength * (1 / rank)`",
        "",
        "The slide-level low/high margins below use the signed convention `class0_score - class1_score`.",
        "",
        "## Overall Agreement",
        "",
        f"- low_high_agreement_rate: `{agreement_rate:.4f}`" if not np.isnan(agreement_rate) else "- low_high_agreement_rate: `NA`",
        f"- correct_slide_count: `{len(correct_df)}`",
        f"- incorrect_slide_count: `{len(incorrect_df)}`",
        "",
        "## Correct vs Incorrect Conflict Distribution",
        "",
    ]

    if not correct_summary.empty:
        display_df = correct_summary[["group_value", "conflict_type", "count", "rate"]].copy()
        display_df["rate"] = pd.to_numeric(display_df["rate"], errors="coerce").map(lambda value: f"{float(value):.4f}")
        lines.extend([markdown_table(display_df), ""])
    else:
        lines.extend(["_No correctness summary available._", ""])

    true_label_summary = conflict_summary_df.loc[conflict_summary_df["summary_scope"] == "true_label", ["group_value", "conflict_type", "count", "rate"]].copy()
    if not true_label_summary.empty:
        true_label_summary["rate"] = pd.to_numeric(true_label_summary["rate"], errors="coerce").map(lambda value: f"{float(value):.4f}")
        lines.extend(["## True Label Pattern Distribution", "", markdown_table(true_label_summary), ""])

    pred_label_summary = conflict_summary_df.loc[conflict_summary_df["summary_scope"] == "predicted_label", ["group_value", "conflict_type", "count", "rate"]].copy()
    if not pred_label_summary.empty:
        pred_label_summary["rate"] = pd.to_numeric(pred_label_summary["rate"], errors="coerce").map(lambda value: f"{float(value):.4f}")
        lines.extend(["## Predicted Label Pattern Distribution", "", markdown_table(pred_label_summary), ""])

    lines.extend(["## Step16 Failure-Case Mapping", ""])
    if not failure_mapping_df.empty:
        mapping_display = failure_mapping_df[
            [
                "slide_id",
                "error_direction",
                "low_support_class_name",
                "high_support_class_name",
                "predicted_scale_relation",
                "conflict_type",
                "step16_expected_conflict_type",
                "step16_pattern_match",
            ]
        ].copy()
        lines.extend([markdown_table(mapping_display), ""])
        for _, row in failure_mapping_df.iterrows():
            lines.extend(
                [
                    f"### {row['slide_id']}",
                    "",
                    f"- predicted_scale_relation: `{row['predicted_scale_relation']}`",
                    f"- conflict_type: `{row['conflict_type']}`",
                    f"- step16_expected_conflict_type: `{row['step16_expected_conflict_type']}`",
                    f"- pattern_match: `{int(row['step16_pattern_match']) if not pd.isna(row['step16_pattern_match']) else 'NA'}`",
                    "",
                ]
            )
    else:
        lines.extend(["_No Step16 failure-case mapping rows available._", ""])

    lines.extend(
        [
            "## Failure Pattern Lists",
            "",
            f"- high_scale_dominant_wrong_slides: `{json.dumps(high_wrong_slides, ensure_ascii=True)}`",
            f"- consistent_wrong_class_drift_slides: `{json.dumps(wrong_drift_slides, ensure_ascii=True)}`",
            "",
        ]
    )

    if high_wrong_slides:
        lines.append("High-scale override errors appear when high scale supports the predicted class while low scale does not.")
    if wrong_drift_slides:
        lines.append("Consistent wrong-class drift appears when both scales move toward the same wrong class.")
    lines.append("")

    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "This is a post-hoc cross-scale evidence analysis only. It does not retrain the model, does not alter Step13/14/15/16 artifacts, and should be treated as model-behavior analysis rather than pathology validation.",
            "",
            "## Warnings",
            "",
        ]
    )
    if warning_log:
        for message in warning_log:
            lines.append(f"- {message}")
    else:
        lines.append("- None")
    lines.append("")

    lines.extend(
        [
            "## Next Suggested Step",
            "",
            "- Step18 cross-scale evidence graph prototype",
            "- Step18 learnable concept-class graph prototype",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    warning_log: list[str] = []
    root = resolve_path(DEFAULT_ROOT, args.root)
    evidence_dir = resolve_path(root, args.evidence_dir)
    graph_dir = resolve_path(root, args.graph_dir)
    stage16_dir = resolve_path(root, args.stage16_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = evidence_dir / "slide_prediction_evidence.csv"
    concepts_path = evidence_dir / "slide_top_concepts.csv"
    edges_path = graph_dir / "stage14_concept_class_edges.csv"
    failure_cases_path = stage16_dir / "stage16_failure_cases.csv"
    selected_narratives_path = stage16_dir / "stage16_selected_slide_narratives.csv"

    prediction_df = ensure_prediction_df(safe_read_csv(prediction_path, warning_log), warning_log)
    concepts_df = ensure_top_concepts_df(safe_read_csv(concepts_path, warning_log), warning_log)
    edges_df = ensure_edges_df(safe_read_csv(edges_path, warning_log), warning_log)
    failure_cases_df = ensure_failure_cases_df(safe_read_csv(failure_cases_path, warning_log), warning_log)
    selected_narratives_df = ensure_selected_narratives_df(safe_read_csv(selected_narratives_path, warning_log), warning_log)

    merged_concepts_df = build_merged_concepts(concepts_df, edges_df)
    scale_scores_df = aggregate_scale_scores(merged_concepts_df, top_k_concepts=max(int(args.top_k_concepts), 1))
    pivot_scores_df = pivot_scale_scores(scale_scores_df)
    slide_scores_df = build_slide_scores_df(
        prediction_df=prediction_df,
        pivot_scores_df=pivot_scores_df,
        selected_narratives_df=selected_narratives_df,
    )

    slide_scores_csv = out_dir / "stage17_cross_scale_slide_scores.csv"
    conflict_summary_csv = out_dir / "stage17_conflict_type_summary.csv"
    failure_pattern_csv = out_dir / "stage17_failure_pattern_cases.csv"
    report_md = out_dir / "stage17_scale_conflict_report.md"
    cross_summary_csv = out_dir / "stage17_cross_scale_summary.csv"

    slide_scores_df.to_csv(slide_scores_csv, index=False)

    conflict_summary_df = pd.DataFrame(build_summary_rows(slide_scores_df))
    conflict_summary_df.to_csv(conflict_summary_csv, index=False)

    failure_pattern_cases_df = build_failure_pattern_cases_df(slide_scores_df, failure_cases_df)
    failure_pattern_cases_df.to_csv(failure_pattern_csv, index=False)

    generated_files = [slide_scores_csv, conflict_summary_csv, failure_pattern_csv]
    cross_summary_df = build_cross_scale_summary_df(
        slide_scores_df=slide_scores_df,
        failure_pattern_cases_df=failure_pattern_cases_df,
        warning_log=warning_log,
        generated_files=generated_files,
    )
    cross_summary_df.to_csv(cross_summary_csv, index=False)
    generated_files.append(cross_summary_csv)

    report_text = build_report(
        root=root,
        evidence_dir=evidence_dir,
        graph_dir=graph_dir,
        stage16_dir=stage16_dir,
        slide_scores_df=slide_scores_df,
        conflict_summary_df=conflict_summary_df,
        failure_pattern_cases_df=failure_pattern_cases_df,
        warning_log=warning_log,
    )
    report_md.write_text(report_text, encoding="utf-8")
    generated_files.append(report_md)

    print(f"[Step17] Slides analyzed: {len(slide_scores_df)}")
    print(f"[Step17] Failure-pattern rows: {len(failure_pattern_cases_df)}")
    print(f"[Step17] Warnings: {len(warning_log)}")
    print(f"[Step17] Output directory: {out_dir}")

    return {
        "root": root,
        "out_dir": out_dir,
        "slide_scores_df": slide_scores_df,
        "conflict_summary_df": conflict_summary_df,
        "failure_pattern_cases_df": failure_pattern_cases_df,
        "cross_summary_df": cross_summary_df,
        "warning_log": warning_log,
    }


def main() -> None:
    args = parse_args()
    try:
        run_analysis(args)
    except Exception as exc:
        root = resolve_path(DEFAULT_ROOT, args.root)
        out_dir = resolve_path(root, args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "stage17_scale_conflict_report.md"
        lines = [
            "# Stage17 Cross-Scale Evidence Conflict Analysis",
            "",
            "The script exited early because an unexpected error occurred.",
            "",
            f"- error: `{exc}`",
            "- This step is intended to remain post-hoc only: no model change, no training, no feature extraction.",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Step17 warning] Unexpected failure: {exc}")
        print(f"[Step17] Wrote fallback report to: {report_path}")


if __name__ == "__main__":
    main()
