from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE15_DIR = Path("results_stage9/stage15_rce_evidence_visualization_fold0")
DEFAULT_EVIDENCE_DIR = Path("results_stage9/stage13_rce_evidence_export_fold0_test_full")
DEFAULT_GRAPH_DIR = Path("results_stage9/stage14_concept_class_graph_fold0")
DEFAULT_OUT_DIR = Path("results_stage9/stage16_failure_case_narratives_fold0")
CLASS_NAME_MAP = {0: "Adenocarcinoma", 1: "NonAdenocarcinoma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step16 failure-case evidence narrative summaries.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to ViLa-MIL-main root.")
    parser.add_argument(
        "--stage15_dir",
        type=Path,
        default=DEFAULT_STAGE15_DIR,
        help="Directory containing Step15 visualization outputs.",
    )
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
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for Step16 narrative outputs.",
    )
    parser.add_argument("--top_k_paths", type=int, default=6, help="Number of strongest paths summarized per slide.")
    return parser.parse_args()


def resolve_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


def warn_message(message: str, warning_log: list[str]) -> None:
    text = f"[Step16 warning] {message}"
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


def safe_read_text(path: Path, warning_log: list[str]) -> str:
    if not path.is_file():
        warn_message(f"Missing report text: {path}", warning_log)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warn_message(f"Failed to read report text {path}: {exc}", warning_log)
        return ""


def empty_selected_slides_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "slide_id",
            "label",
            "pred",
            "correct",
            "class_name_true",
            "class_name_pred",
            "true_prob",
            "pred_prob",
            "confidence_margin",
            "selection_reason",
            "prob_0",
            "prob_1",
        ]
    )


def empty_concepts_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "slide_id",
            "scale",
            "class_id",
            "concept_id",
            "concept_text",
            "evidence_score",
            "prompt_weight",
            "rank",
            "prompt_id",
            "edge_strength",
            "class_name",
            "is_high_frequency_low_distinction",
            "passes_min_count",
        ]
    )


def ensure_selected_slides(
    selected_df: pd.DataFrame | None,
    prediction_df: pd.DataFrame | None,
    warning_log: list[str],
) -> pd.DataFrame:
    if selected_df is None:
        return empty_selected_slides_df()
    required = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "class_name_true",
        "class_name_pred",
        "true_prob",
        "pred_prob",
        "confidence_margin",
        "selection_reason",
    ]
    missing = [column for column in required if column not in selected_df.columns]
    if missing:
        warn_message(f"Step15 selected slides CSV missing required columns: {missing}", warning_log)
        return empty_selected_slides_df()

    result = selected_df.copy()
    for column in ["label", "pred", "correct", "true_prob", "pred_prob", "confidence_margin"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["label"] = result["label"].fillna(-1).astype(int)
    result["pred"] = result["pred"].fillna(-1).astype(int)
    result["correct"] = result["correct"].fillna(0).astype(int)
    result["class_name_true"] = result["class_name_true"].fillna(result["label"].map(CLASS_NAME_MAP)).fillna("Unknown")
    result["class_name_pred"] = result["class_name_pred"].fillna(result["pred"].map(CLASS_NAME_MAP)).fillna("Unknown")

    if prediction_df is not None and not prediction_df.empty and "slide_id" in prediction_df.columns:
        pred_subset = prediction_df.copy()
        for column in ["prob_0", "prob_1"]:
            if column in pred_subset.columns:
                pred_subset[column] = pd.to_numeric(pred_subset[column], errors="coerce")
        pred_subset = pred_subset[[column for column in ["slide_id", "prob_0", "prob_1"] if column in pred_subset.columns]].drop_duplicates("slide_id")
        result = result.merge(pred_subset, on="slide_id", how="left")
    else:
        result["prob_0"] = np.nan
        result["prob_1"] = np.nan

    if "prob_0" not in result.columns:
        result["prob_0"] = np.nan
    if "prob_1" not in result.columns:
        result["prob_1"] = np.nan
    return result


def ensure_prediction_df(prediction_df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if prediction_df is None:
        return empty_selected_slides_df()
    required = ["slide_id", "label", "pred", "correct", "prob_0", "prob_1"]
    missing = [column for column in required if column not in prediction_df.columns]
    if missing:
        warn_message(f"Prediction CSV missing required columns: {missing}", warning_log)
        return empty_selected_slides_df()
    result = prediction_df.copy()
    for column in ["label", "pred", "correct", "prob_0", "prob_1"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["label"] = result["label"].fillna(-1).astype(int)
    result["pred"] = result["pred"].fillna(-1).astype(int)
    result["correct"] = result["correct"].fillna(0).astype(int)
    return result


def ensure_concepts_df(concepts_df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if concepts_df is None:
        return empty_concepts_df()
    required = [
        "slide_id",
        "scale",
        "class_id",
        "concept_id",
        "concept_text",
        "evidence_score",
        "prompt_weight",
        "rank",
    ]
    missing = [column for column in required if column not in concepts_df.columns]
    if missing:
        warn_message(f"Step13 top concepts CSV missing required columns: {missing}", warning_log)
        return empty_concepts_df()
    result = concepts_df.copy()
    for column in ["class_id", "evidence_score", "prompt_weight", "rank"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "prompt_id" in result.columns:
        result["prompt_id"] = pd.to_numeric(result["prompt_id"], errors="coerce")
    else:
        result["prompt_id"] = np.nan
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    result["rank"] = result["rank"].fillna(9999).astype(int)
    return result


def ensure_stage15_paths_df(paths_df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if paths_df is None:
        return pd.DataFrame(
            columns=[
                "slide_id",
                "label",
                "pred",
                "correct",
                "scale",
                "class_id",
                "concept_id",
                "concept_text",
                "rank",
                "evidence_score",
                "prompt_weight",
                "edge_strength",
                "region_id",
                "region_score",
            ]
        )
    required = [
        "slide_id",
        "scale",
        "class_id",
        "concept_id",
        "concept_text",
        "rank",
        "evidence_score",
        "prompt_weight",
        "edge_strength",
    ]
    missing = [column for column in required if column not in paths_df.columns]
    if missing:
        warn_message(f"Step15 evidence paths CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=required)
    result = paths_df.copy()
    for column in ["label", "pred", "correct", "class_id", "rank", "evidence_score", "prompt_weight", "edge_strength", "region_id", "region_score"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    result["rank"] = result["rank"].fillna(9999).astype(int)
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    return result


def ensure_edges_df(edges_df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if edges_df is None:
        return empty_concepts_df()
    required = [
        "scale",
        "class_id",
        "class_name",
        "concept_id",
        "concept_text",
        "edge_strength",
    ]
    missing = [column for column in required if column not in edges_df.columns]
    if missing:
        warn_message(f"Step14 edges CSV missing required columns: {missing}", warning_log)
        return empty_concepts_df()
    result = edges_df.copy()
    for column in [
        "class_id",
        "edge_strength",
        "mean_evidence_score",
        "slide_coverage",
        "mean_rank",
        "n_topk",
        "label_evidence_gap",
        "correctness_evidence_gap",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ["is_high_frequency_low_distinction", "passes_min_count"]:
        if column in result.columns:
            result[column] = result[column].fillna(False).astype(bool)
        else:
            result[column] = False if column == "is_high_frequency_low_distinction" else True
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    result["class_name"] = result["class_name"].fillna(result["class_id"].map(CLASS_NAME_MAP)).fillna("Unknown")
    return result


def join_concept_strings(group: pd.DataFrame, top_k: int) -> str:
    if group.empty:
        return ""
    rows = []
    for _, row in group.head(int(top_k)).iterrows():
        rows.append(
            f"{row['concept_id']} (ev={float(row['evidence_score']):.4f}, edge={float(row.get('edge_strength', np.nan)):.4f}, rank={int(row['rank'])})"
        )
    return " | ".join(rows)


def join_path_strings(group: pd.DataFrame, top_k: int) -> str:
    if group.empty:
        return ""
    rows = []
    sorted_group = group.sort_values(
        ["evidence_score", "edge_strength", "prompt_weight"],
        ascending=[False, False, False],
    )
    for _, row in sorted_group.head(int(top_k)).iterrows():
        rows.append(
            f"{row['slide_id']} -> {row['scale']} -> {row['concept_id']} -> {CLASS_NAME_MAP.get(int(row['class_id']), 'Unknown')}"
            f" (ev={float(row['evidence_score']):.4f}, edge={float(row['edge_strength']):.4f})"
        )
    return " | ".join(rows)


def top_group_rows(merged_df: pd.DataFrame, slide_id: str, scale: str | None, class_id: int | None, top_k: int) -> pd.DataFrame:
    subset = merged_df.loc[merged_df["slide_id"] == slide_id].copy()
    if scale is not None:
        subset = subset.loc[subset["scale"] == scale].copy()
    if class_id is not None:
        subset = subset.loc[subset["class_id"] == int(class_id)].copy()
    subset = subset.sort_values(
        ["rank", "evidence_score", "prompt_weight", "edge_strength"],
        ascending=[True, False, False, False],
    )
    return subset.head(int(top_k)).copy()


def strongest_path_row(merged_df: pd.DataFrame, slide_id: str, class_id: int) -> pd.Series | None:
    subset = merged_df.loc[(merged_df["slide_id"] == slide_id) & (merged_df["class_id"] == int(class_id))].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(
        ["evidence_score", "edge_strength", "prompt_weight", "rank"],
        ascending=[False, False, False, True],
    )
    return subset.iloc[0]


def scale_support_metrics(merged_df: pd.DataFrame, slide_id: str, scale: str, pred_class: int, true_class: int) -> dict[str, float | str]:
    pred_rows = top_group_rows(merged_df, slide_id=slide_id, scale=scale, class_id=pred_class, top_k=1)
    true_rows = top_group_rows(merged_df, slide_id=slide_id, scale=scale, class_id=true_class, top_k=1)
    pred_ev = float(pred_rows.iloc[0]["evidence_score"]) if not pred_rows.empty else np.nan
    true_ev = float(true_rows.iloc[0]["evidence_score"]) if not true_rows.empty else np.nan
    pred_edge = float(pred_rows.iloc[0]["edge_strength"]) if not pred_rows.empty else np.nan
    true_edge = float(true_rows.iloc[0]["edge_strength"]) if not true_rows.empty else np.nan
    evidence_gap = pred_ev - true_ev if not (np.isnan(pred_ev) or np.isnan(true_ev)) else np.nan
    edge_gap = pred_edge - true_edge if not (np.isnan(pred_edge) or np.isnan(true_edge)) else np.nan

    direction = "unclear"
    if not np.isnan(evidence_gap):
        if evidence_gap > 0.02:
            direction = "predicted_class_stronger"
        elif evidence_gap < -0.02:
            direction = "true_class_stronger"
        else:
            direction = "mixed_or_close"
    return {
        "pred_ev": pred_ev,
        "true_ev": true_ev,
        "pred_edge": pred_edge,
        "true_edge": true_edge,
        "evidence_gap": evidence_gap,
        "edge_gap": edge_gap,
        "direction": direction,
    }


def contrast_class(label: int, pred: int) -> int:
    if int(label) != int(pred):
        return int(label)
    return 1 - int(pred) if int(pred) in (0, 1) else int(label)


def build_failure_hypothesis(
    pred_class_name: str,
    true_class_name: str,
    low_direction: str,
    high_direction: str,
    pred_top_ids: list[str],
    true_top_ids: list[str],
) -> str:
    pred_phrase = ", ".join(pred_top_ids[:3]) if pred_top_ids else pred_class_name
    true_phrase = ", ".join(true_top_ids[:3]) if true_top_ids else true_class_name
    if low_direction == "predicted_class_stronger" and high_direction == "predicted_class_stronger":
        return (
            f"Model evidence suggests a cross-scale drift toward {pred_class_name}, with both low and high scale relying on "
            f"{pred_phrase} more strongly than the true-class concepts {true_phrase}."
        )
    if high_direction == "predicted_class_stronger" and low_direction != "predicted_class_stronger":
        return (
            f"Model appears to rely mainly on high-scale {pred_class_name} cues such as {pred_phrase}, while low-scale evidence is weaker or conflicted against the true class."
        )
    if low_direction == "predicted_class_stronger" and high_direction != "predicted_class_stronger":
        return (
            f"Model appears to rely mainly on low-scale {pred_class_name} cues such as {pred_phrase}, while high-scale evidence is weaker or conflicted against the true class."
        )
    if low_direction == "true_class_stronger" and high_direction == "true_class_stronger":
        return (
            f"Model still predicts {pred_class_name} despite both scales showing stronger true-class evidence around {true_phrase}; this suggests the final decision may be dominated by a small set of misleading predicted-class concepts."
        )
    return (
        f"Model evidence appears mixed: predicted-class concepts {pred_phrase} compete with true-class concepts {true_phrase}, creating a cross-scale conflict rather than a clean wrong-class signature."
    )


def build_success_hypothesis(
    pred_class_name: str,
    low_direction: str,
    high_direction: str,
    support_ids: list[str],
) -> str:
    support_phrase = ", ".join(support_ids[:3]) if support_ids else pred_class_name
    if low_direction == "predicted_class_stronger" and high_direction == "predicted_class_stronger":
        return f"Model evidence suggests both scales consistently support {pred_class_name}, especially through {support_phrase}."
    if high_direction == "predicted_class_stronger":
        return f"Model appears to rely more on high-scale {pred_class_name} evidence, with {support_phrase} providing the strongest support."
    if low_direction == "predicted_class_stronger":
        return f"Model appears to rely more on low-scale {pred_class_name} evidence, with {support_phrase} providing the strongest support."
    return f"Model evidence appears more mixed, but the final prediction still aligns with {pred_class_name} through concepts such as {support_phrase}."


def json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=True)


def build_merged_concepts(concepts_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    edge_columns = [
        column
        for column in [
            "scale",
            "class_id",
            "concept_id",
            "class_name",
            "edge_strength",
            "mean_evidence_score",
            "slide_coverage",
            "mean_rank",
            "n_topk",
            "is_high_frequency_low_distinction",
            "passes_min_count",
            "label_evidence_gap",
            "correctness_evidence_gap",
        ]
        if column in edges_df.columns
    ]
    edge_subset = edges_df[edge_columns].drop_duplicates(["scale", "class_id", "concept_id"]).copy() if edge_columns else pd.DataFrame()
    if edge_subset.empty:
        result = concepts_df.copy()
        result["class_name"] = result["class_id"].map(CLASS_NAME_MAP).fillna("Unknown")
        result["edge_strength"] = np.nan
        result["is_high_frequency_low_distinction"] = False
        result["passes_min_count"] = False
        return result
    result = concepts_df.merge(edge_subset, on=["scale", "class_id", "concept_id"], how="left")
    result["class_name"] = result["class_name"].fillna(result["class_id"].map(CLASS_NAME_MAP)).fillna("Unknown")
    result["edge_strength"] = pd.to_numeric(result["edge_strength"], errors="coerce")
    result["is_high_frequency_low_distinction"] = result["is_high_frequency_low_distinction"].fillna(False).astype(bool)
    result["passes_min_count"] = result["passes_min_count"].fillna(False).astype(bool)
    return result


def build_selected_slide_narratives(
    selected_slides_df: pd.DataFrame,
    merged_concepts_df: pd.DataFrame,
    stage15_paths_df: pd.DataFrame,
    top_k_paths: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, slide_row in selected_slides_df.iterrows():
        slide_id = str(slide_row["slide_id"])
        label = int(slide_row["label"])
        pred = int(slide_row["pred"])
        contrast = contrast_class(label=label, pred=pred)

        low_pred_df = top_group_rows(merged_concepts_df, slide_id, "low", pred, top_k_paths)
        high_pred_df = top_group_rows(merged_concepts_df, slide_id, "high", pred, top_k_paths)
        path_df = stage15_paths_df.loc[stage15_paths_df["slide_id"] == slide_id].copy()
        if path_df.empty:
            path_df = top_group_rows(merged_concepts_df, slide_id, None, pred, top_k_paths)

        low_metrics = scale_support_metrics(merged_concepts_df, slide_id, "low", pred, contrast)
        high_metrics = scale_support_metrics(merged_concepts_df, slide_id, "high", pred, contrast)
        support_ids = low_pred_df["concept_id"].head(2).tolist() + high_pred_df["concept_id"].head(2).tolist()
        narrative = (
            build_success_hypothesis(slide_row["class_name_pred"], low_metrics["direction"], high_metrics["direction"], support_ids)
            if int(slide_row["correct"]) == 1
            else build_failure_hypothesis(
                slide_row["class_name_pred"],
                slide_row["class_name_true"],
                low_metrics["direction"],
                high_metrics["direction"],
                high_pred_df["concept_id"].head(3).tolist() + low_pred_df["concept_id"].head(3).tolist(),
                top_group_rows(merged_concepts_df, slide_id, None, label, top_k_paths)["concept_id"].head(3).tolist(),
            )
        )
        rows.append(
            {
                "slide_id": slide_id,
                "true_label": label,
                "predicted_label": pred,
                "true_class_name": slide_row["class_name_true"],
                "predicted_class_name": slide_row["class_name_pred"],
                "correct": int(slide_row["correct"]),
                "prob_0": slide_row.get("prob_0", np.nan),
                "prob_1": slide_row.get("prob_1", np.nan),
                "confidence_margin": slide_row.get("confidence_margin", np.nan),
                "selection_reason": slide_row.get("selection_reason", ""),
                "top_low_scale_evidence_concepts": join_concept_strings(low_pred_df, top_k_paths),
                "top_high_scale_evidence_concepts": join_concept_strings(high_pred_df, top_k_paths),
                "strongest_paths": join_path_strings(path_df, top_k_paths),
                "low_scale_support_direction": low_metrics["direction"],
                "high_scale_support_direction": high_metrics["direction"],
                "narrative_summary": narrative,
            }
        )
    return pd.DataFrame(rows)


def build_failure_cases(
    selected_slides_df: pd.DataFrame,
    merged_concepts_df: pd.DataFrame,
    top_k_paths: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    incorrect_df = selected_slides_df.loc[selected_slides_df["correct"] == 0].copy()
    for _, slide_row in incorrect_df.iterrows():
        slide_id = str(slide_row["slide_id"])
        true_label = int(slide_row["label"])
        pred_label = int(slide_row["pred"])

        pred_top = top_group_rows(merged_concepts_df, slide_id, None, pred_label, top_k_paths)
        true_top = top_group_rows(merged_concepts_df, slide_id, None, true_label, top_k_paths)
        pred_best = strongest_path_row(merged_concepts_df, slide_id, pred_label)
        true_best = strongest_path_row(merged_concepts_df, slide_id, true_label)
        low_metrics = scale_support_metrics(merged_concepts_df, slide_id, "low", pred_label, true_label)
        high_metrics = scale_support_metrics(merged_concepts_df, slide_id, "high", pred_label, true_label)

        pred_best_ev = float(pred_best["evidence_score"]) if pred_best is not None else np.nan
        true_best_ev = float(true_best["evidence_score"]) if true_best is not None else np.nan
        pred_best_edge = float(pred_best["edge_strength"]) if pred_best is not None else np.nan
        true_best_edge = float(true_best["edge_strength"]) if true_best is not None else np.nan
        evidence_gap = pred_best_ev - true_best_ev if not (np.isnan(pred_best_ev) or np.isnan(true_best_ev)) else np.nan
        edge_gap = pred_best_edge - true_best_edge if not (np.isnan(pred_best_edge) or np.isnan(true_best_edge)) else np.nan

        if low_metrics["direction"] == high_metrics["direction"]:
            scale_relation = f"consistent_{low_metrics['direction']}"
        else:
            scale_relation = "cross_scale_conflict"

        rows.append(
            {
                "slide_id": slide_id,
                "error_direction": f"{slide_row['class_name_true']} -> {slide_row['class_name_pred']}",
                "true_class_name": slide_row["class_name_true"],
                "predicted_class_name": slide_row["class_name_pred"],
                "prob_0": slide_row.get("prob_0", np.nan),
                "prob_1": slide_row.get("prob_1", np.nan),
                "confidence_margin": slide_row.get("confidence_margin", np.nan),
                "predicted_class_top_concepts": join_concept_strings(pred_top, top_k_paths),
                "true_class_top_concepts": join_concept_strings(true_top, top_k_paths),
                "predicted_best_evidence_score": pred_best_ev,
                "true_best_evidence_score": true_best_ev,
                "evidence_score_gap": evidence_gap,
                "predicted_best_edge_strength": pred_best_edge,
                "true_best_edge_strength": true_best_edge,
                "edge_strength_gap": edge_gap,
                "low_scale_relation": low_metrics["direction"],
                "high_scale_relation": high_metrics["direction"],
                "scale_relation": scale_relation,
                "machine_generated_failure_hypothesis": build_failure_hypothesis(
                    str(slide_row["class_name_pred"]),
                    str(slide_row["class_name_true"]),
                    str(low_metrics["direction"]),
                    str(high_metrics["direction"]),
                    pred_top["concept_id"].head(3).tolist(),
                    true_top["concept_id"].head(3).tolist(),
                ),
                "predicted_top_concept_ids": json_list(pred_top["concept_id"].head(top_k_paths).tolist()),
                "true_top_concept_ids": json_list(true_top["concept_id"].head(top_k_paths).tolist()),
            }
        )
    return pd.DataFrame(rows)


def build_success_cases(
    selected_slides_df: pd.DataFrame,
    merged_concepts_df: pd.DataFrame,
    top_k_paths: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    correct_df = selected_slides_df.loc[selected_slides_df["correct"] == 1].copy()
    for _, slide_row in correct_df.iterrows():
        slide_id = str(slide_row["slide_id"])
        pred_label = int(slide_row["pred"])
        contrast = contrast_class(label=int(slide_row["label"]), pred=pred_label)
        support_df = top_group_rows(merged_concepts_df, slide_id, None, pred_label, top_k_paths)
        low_metrics = scale_support_metrics(merged_concepts_df, slide_id, "low", pred_label, contrast)
        high_metrics = scale_support_metrics(merged_concepts_df, slide_id, "high", pred_label, contrast)

        if low_metrics["direction"] == "predicted_class_stronger" and high_metrics["direction"] == "predicted_class_stronger":
            scale_support = "both_scales_support_prediction"
        elif low_metrics["direction"] == "predicted_class_stronger":
            scale_support = "low_scale_stronger"
        elif high_metrics["direction"] == "predicted_class_stronger":
            scale_support = "high_scale_stronger"
        else:
            scale_support = "mixed_support"

        edge_alignment = bool((support_df["passes_min_count"] == True).any()) if "passes_min_count" in support_df.columns else False
        rows.append(
            {
                "slide_id": slide_id,
                "predicted_class_name": slide_row["class_name_pred"],
                "prob_0": slide_row.get("prob_0", np.nan),
                "prob_1": slide_row.get("prob_1", np.nan),
                "confidence_margin": slide_row.get("confidence_margin", np.nan),
                "strongest_supporting_concepts": join_concept_strings(support_df, top_k_paths),
                "low_scale_relation": low_metrics["direction"],
                "high_scale_relation": high_metrics["direction"],
                "scale_support_summary": scale_support,
                "matches_step14_top_edge": int(edge_alignment),
                "machine_generated_success_hypothesis": build_success_hypothesis(
                    str(slide_row["class_name_pred"]),
                    str(low_metrics["direction"]),
                    str(high_metrics["direction"]),
                    support_df["concept_id"].head(3).tolist(),
                ),
                "support_concept_ids": json_list(support_df["concept_id"].head(top_k_paths).tolist()),
            }
        )
    return pd.DataFrame(rows)


def build_summary_df(
    selected_slides_df: pd.DataFrame,
    narratives_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    success_df: pd.DataFrame,
    merged_concepts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    warning_log: list[str],
    generated_files: list[Path],
) -> pd.DataFrame:
    rows = [
        {"section": "inputs", "metric": "selected_slides", "value": int(len(selected_slides_df)), "note": ""},
        {"section": "inputs", "metric": "correct_slides", "value": int((selected_slides_df["correct"] == 1).sum()) if not selected_slides_df.empty else 0, "note": ""},
        {"section": "inputs", "metric": "incorrect_slides", "value": int((selected_slides_df["correct"] == 0).sum()) if not selected_slides_df.empty else 0, "note": ""},
        {"section": "outputs", "metric": "narrative_rows", "value": int(len(narratives_df)), "note": ""},
        {"section": "outputs", "metric": "failure_case_rows", "value": int(len(failure_df)), "note": ""},
        {"section": "outputs", "metric": "success_case_rows", "value": int(len(success_df)), "note": ""},
        {"section": "quality", "metric": "warning_count", "value": int(len(warning_log)), "note": ""},
    ]

    if not failure_df.empty:
        suspicious_ids = []
        for column in ["predicted_top_concept_ids"]:
            for raw_value in failure_df[column].dropna().tolist():
                try:
                    suspicious_ids.extend(json.loads(raw_value))
                except Exception:
                    continue
        suspicious_series = pd.Series(suspicious_ids).value_counts()
        for concept_id, count in suspicious_series.head(12).items():
            rows.append(
                {
                    "section": "failure",
                    "metric": "suspicious_predicted_concept",
                    "value": int(count),
                    "note": str(concept_id),
                }
            )

    if not edges_df.empty and "is_high_frequency_low_distinction" in edges_df.columns:
        flagged = edges_df.loc[edges_df["is_high_frequency_low_distinction"] == True].copy()
        rows.append(
            {
                "section": "graph",
                "metric": "high_frequency_low_distinction_edges",
                "value": int(len(flagged)),
                "note": "",
            }
        )
        if flagged.empty:
            high_coverage = edges_df.sort_values(["slide_coverage", "edge_strength"], ascending=[False, False]).head(4)
            for _, row in high_coverage.iterrows():
                rows.append(
                    {
                        "section": "graph",
                        "metric": "high_coverage_edge",
                        "value": float(row.get("slide_coverage", np.nan)),
                        "note": f"{row['scale']}::{row['concept_id']}",
                    }
                )

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
    body = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + body)


def build_report(
    root: Path,
    stage15_dir: Path,
    evidence_dir: Path,
    graph_dir: Path,
    selected_slides_df: pd.DataFrame,
    narratives_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    success_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    step15_summary_text: str,
    top_k_paths: int,
    warning_log: list[str],
) -> str:
    correct_count = int((selected_slides_df["correct"] == 1).sum()) if not selected_slides_df.empty else 0
    incorrect_count = int((selected_slides_df["correct"] == 0).sum()) if not selected_slides_df.empty else 0
    lines = [
        "# Stage16 Failure-Case Evidence Narrative Summary",
        "",
        "Step16 is a post-hoc narrative summary built from Step15 selected slides and upstream Step13d/Step14 evidence.",
        "",
        "- It does not modify the model.",
        "- It does not run training.",
        "- It does not claim pathology-doctor validation; the wording below stays at the level of model evidence suggests / model appears to rely on.",
        "",
        "## Inputs",
        "",
        f"- root: `{root}`",
        f"- stage15_dir: `{stage15_dir}`",
        f"- evidence_dir: `{evidence_dir}`",
        f"- graph_dir: `{graph_dir}`",
        f"- selected_slide_count: `{len(selected_slides_df)}`",
        f"- correct_selected_slides: `{correct_count}`",
        f"- incorrect_selected_slides: `{incorrect_count}`",
        f"- top_k_paths: `{top_k_paths}`",
        "",
        "## Correct vs Incorrect Summary",
        "",
    ]

    if not failure_df.empty:
        mean_gap = pd.to_numeric(failure_df["evidence_score_gap"], errors="coerce").mean()
        lines.append(
            f"- Incorrect slides show mean predicted-vs-true best evidence gap `{mean_gap:.4f}`."
            if not np.isnan(mean_gap)
            else "- Incorrect slides show mixed predicted-vs-true evidence gaps."
        )
    else:
        lines.append("- No incorrect selected slides were available.")

    if not success_df.empty:
        both_support = int((success_df["scale_support_summary"] == "both_scales_support_prediction").sum())
        lines.append(f"- Correct slides with both scales supporting the predicted class: `{both_support}` / `{len(success_df)}`.")
    else:
        lines.append("- No correct selected slides were available.")

    if step15_summary_text:
        lines.extend(["", "## Step15 Context", "", "Step15 summary file was available and used as upstream context.", ""])

    if not failure_df.empty:
        lines.extend(["## Failure Cases", ""])
        preview_columns = [
            "slide_id",
            "error_direction",
            "evidence_score_gap",
            "edge_strength_gap",
            "low_scale_relation",
            "high_scale_relation",
            "scale_relation",
        ]
        failure_preview = failure_df[preview_columns].copy()
        for column in ["evidence_score_gap", "edge_strength_gap"]:
            failure_preview[column] = pd.to_numeric(failure_preview[column], errors="coerce").map(
                lambda value: "NA" if pd.isna(value) else f"{float(value):.4f}"
            )
        lines.extend([markdown_table(failure_preview), ""])

        for _, row in failure_df.iterrows():
            lines.extend(
                [
                    f"### {row['slide_id']}",
                    "",
                    f"- error_direction: `{row['error_direction']}`",
                    f"- predicted_class_top_concepts: {row['predicted_class_top_concepts']}",
                    f"- true_class_top_concepts: {row['true_class_top_concepts']}",
                    f"- low/high relation: `{row['low_scale_relation']}` / `{row['high_scale_relation']}`",
                    f"- hypothesis: {row['machine_generated_failure_hypothesis']}",
                    "",
                ]
            )

    if not success_df.empty:
        lines.extend(["## Representative Success Cases", ""])
        success_preview = success_df.head(min(4, len(success_df))).copy()
        lines.extend(
            [
                markdown_table(
                    success_preview[
                        [
                            "slide_id",
                            "predicted_class_name",
                            "scale_support_summary",
                            "matches_step14_top_edge",
                            "strongest_supporting_concepts",
                        ]
                    ]
                ),
                "",
            ]
        )
        for _, row in success_df.head(min(3, len(success_df))).iterrows():
            lines.extend(
                [
                    f"### {row['slide_id']}",
                    "",
                    f"- support_summary: `{row['scale_support_summary']}`",
                    f"- supporting_concepts: {row['strongest_supporting_concepts']}",
                    f"- narrative: {row['machine_generated_success_hypothesis']}",
                    "",
                ]
            )

    lines.extend(["## Possible Misleading Or High-Coverage Concepts", ""])
    if not failure_df.empty:
        suspicious_ids = []
        for raw_value in failure_df["predicted_top_concept_ids"].dropna().tolist():
            try:
                suspicious_ids.extend(json.loads(raw_value))
            except Exception:
                continue
        suspicious_series = pd.Series(suspicious_ids).value_counts()
        if not suspicious_series.empty:
            for concept_id, count in suspicious_series.head(12).items():
                lines.append(f"- failure-case predicted concept `{concept_id}` appeared `{int(count)}` times across the incorrect-slide top concepts.")
        else:
            lines.append("- No recurring predicted-class concept was extracted from incorrect slides.")
    else:
        lines.append("- No incorrect slides were available for misleading-concept inspection.")

    if not edges_df.empty and "is_high_frequency_low_distinction" in edges_df.columns:
        flagged = edges_df.loc[edges_df["is_high_frequency_low_distinction"] == True].copy()
        if flagged.empty:
            lines.append("- Step14 did not flag any concept-class edge as high-frequency-low-distinction under its saved thresholds.")
        else:
            for _, row in flagged.head(6).iterrows():
                lines.append(f"- flagged low-distinction edge: `{row['scale']}::{row['concept_id']}`")
    lines.append("")

    lines.extend(["## Interpretation Boundary", ""])
    lines.extend(
        [
            "This step is a post-hoc narrative summary only. It does not retrain the model, does not alter Step13/14/15 artifacts, and should be treated as qualitative interpretation rather than validated pathology evidence.",
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
            "- Step17 learnable concept-class graph prototype",
            "- Step17 cross-scale evidence conflict analysis",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    warning_log: list[str] = []
    root = resolve_path(DEFAULT_ROOT, args.root)
    stage15_dir = resolve_path(root, args.stage15_dir)
    evidence_dir = resolve_path(root, args.evidence_dir)
    graph_dir = resolve_path(root, args.graph_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_slides_path = stage15_dir / "stage15_selected_slides.csv"
    stage15_paths_path = stage15_dir / "stage15_slide_evidence_paths.csv"
    stage15_summary_path = stage15_dir / "stage15_visualization_summary.csv"
    prediction_path = evidence_dir / "slide_prediction_evidence.csv"
    concepts_path = evidence_dir / "slide_top_concepts.csv"
    edges_path = graph_dir / "stage14_concept_class_edges.csv"

    prediction_df = ensure_prediction_df(safe_read_csv(prediction_path, warning_log), warning_log)
    selected_slides_df = ensure_selected_slides(
        safe_read_csv(selected_slides_path, warning_log),
        prediction_df,
        warning_log,
    )
    stage15_paths_df = ensure_stage15_paths_df(safe_read_csv(stage15_paths_path, warning_log), warning_log)
    stage15_summary_df = safe_read_csv(stage15_summary_path, warning_log)
    concepts_df = ensure_concepts_df(safe_read_csv(concepts_path, warning_log), warning_log)
    edges_df = ensure_edges_df(safe_read_csv(edges_path, warning_log), warning_log)
    merged_concepts_df = build_merged_concepts(concepts_df, edges_df)
    step15_summary_text = markdown_table(stage15_summary_df.head(12)) if stage15_summary_df is not None and not stage15_summary_df.empty else ""

    narratives_df = build_selected_slide_narratives(
        selected_slides_df=selected_slides_df,
        merged_concepts_df=merged_concepts_df,
        stage15_paths_df=stage15_paths_df,
        top_k_paths=max(int(args.top_k_paths), 1),
    )
    failure_df = build_failure_cases(
        selected_slides_df=selected_slides_df,
        merged_concepts_df=merged_concepts_df,
        top_k_paths=max(int(args.top_k_paths), 1),
    )
    success_df = build_success_cases(
        selected_slides_df=selected_slides_df,
        merged_concepts_df=merged_concepts_df,
        top_k_paths=max(int(args.top_k_paths), 1),
    )

    narratives_csv = out_dir / "stage16_selected_slide_narratives.csv"
    failure_csv = out_dir / "stage16_failure_cases.csv"
    success_csv = out_dir / "stage16_success_cases.csv"
    report_md = out_dir / "stage16_case_narrative_report.md"
    summary_csv = out_dir / "stage16_case_narrative_summary.csv"

    narratives_df.to_csv(narratives_csv, index=False)
    failure_df.to_csv(failure_csv, index=False)
    success_df.to_csv(success_csv, index=False)

    generated_files = [narratives_csv, failure_csv, success_csv]
    summary_df = build_summary_df(
        selected_slides_df=selected_slides_df,
        narratives_df=narratives_df,
        failure_df=failure_df,
        success_df=success_df,
        merged_concepts_df=merged_concepts_df,
        edges_df=edges_df,
        warning_log=warning_log,
        generated_files=generated_files,
    )
    summary_df.to_csv(summary_csv, index=False)
    generated_files.append(summary_csv)

    report_text = build_report(
        root=root,
        stage15_dir=stage15_dir,
        evidence_dir=evidence_dir,
        graph_dir=graph_dir,
        selected_slides_df=selected_slides_df,
        narratives_df=narratives_df,
        failure_df=failure_df,
        success_df=success_df,
        edges_df=edges_df,
        step15_summary_text=step15_summary_text,
        top_k_paths=max(int(args.top_k_paths), 1),
        warning_log=warning_log,
    )
    report_md.write_text(report_text, encoding="utf-8")
    generated_files.append(report_md)

    print(f"[Step16] Selected slides: {len(selected_slides_df)}")
    print(f"[Step16] Failure cases: {len(failure_df)}")
    print(f"[Step16] Success cases: {len(success_df)}")
    print(f"[Step16] Warnings: {len(warning_log)}")
    print(f"[Step16] Output directory: {out_dir}")

    return {
        "root": root,
        "out_dir": out_dir,
        "selected_slides_df": selected_slides_df,
        "narratives_df": narratives_df,
        "failure_df": failure_df,
        "success_df": success_df,
        "summary_df": summary_df,
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
        report_path = out_dir / "stage16_case_narrative_report.md"
        lines = [
            "# Stage16 Failure-Case Evidence Narrative Summary",
            "",
            "The script exited early because an unexpected error occurred.",
            "",
            f"- error: `{exc}`",
            "- This step is intended to remain post-hoc only: no model change, no training, no feature extraction.",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Step16 warning] Unexpected failure: {exc}")
        print(f"[Step16] Wrote fallback report to: {report_path}")


if __name__ == "__main__":
    main()
