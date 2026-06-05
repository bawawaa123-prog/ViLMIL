from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = Path("results_stage9/stage13_rce_evidence_export_fold0_test_full")
DEFAULT_GRAPH_DIR = Path("results_stage9/stage14_concept_class_graph_fold0")
DEFAULT_CONFLICT_DIR = Path("results_stage9/stage17_cross_scale_conflict_analysis_fold0")
DEFAULT_OUT_DIR = Path("results_stage9/stage18_cross_scale_evidence_graph_fold0")
CLASS_NAME_MAP = {0: "Adenocarcinoma", 1: "NonAdenocarcinoma"}
OVERRIDE_CONFLICT_TYPES = {"high_scale_dominant_wrong", "high_scale_dominant_correct"}
DRIFT_CONFLICT_TYPES = {"consistent_wrong_class_drift"}
EXPECTED_FAILURE_SLIDES = ["25001929B3", "25023972B3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step18 post-hoc cross-scale evidence graph prototype.")
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
        help="Directory containing Step14 concept-class graph outputs.",
    )
    parser.add_argument(
        "--conflict_dir",
        type=Path,
        default=DEFAULT_CONFLICT_DIR,
        help="Directory containing Step17 conflict analysis outputs.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for Step18 graph prototype outputs.",
    )
    parser.add_argument("--top_k_edges", type=int, default=20, help="Top-K concept-class edges retained per scale/class.")
    parser.add_argument("--min_count", type=int, default=3, help="Minimum co-occurrence count retained for low/high concept pairs.")
    return parser.parse_args()


def resolve_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


def warn_message(message: str, warning_log: list[str]) -> None:
    text = f"[Step18 warning] {message}"
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
    ]
    if df is None:
        return pd.DataFrame(columns=columns)
    required = ["slide_id", "scale", "class_id", "concept_id", "concept_text", "evidence_score", "prompt_weight", "rank"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Top concepts CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=columns)
    result = df.copy()
    for column in ["class_id", "evidence_score", "prompt_weight", "rank"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    result["rank"] = result["rank"].fillna(9999).astype(int)
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    return result


def ensure_stage14_edges_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    columns = [
        "scale",
        "class_id",
        "class_name",
        "concept_id",
        "concept_text",
        "edge_strength",
        "mean_evidence_score",
        "slide_coverage",
        "passes_min_count",
    ]
    if df is None:
        return pd.DataFrame(columns=columns)
    required = ["scale", "class_id", "concept_id", "concept_text", "edge_strength"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Stage14 edges CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=columns)
    result = df.copy()
    for column in ["class_id", "edge_strength", "mean_evidence_score", "slide_coverage"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "passes_min_count" in result.columns:
        result["passes_min_count"] = result["passes_min_count"].fillna(False).astype(bool)
    else:
        result["passes_min_count"] = True
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    if "class_name" not in result.columns:
        result["class_name"] = result["class_id"].map(CLASS_NAME_MAP).fillna("Unknown")
    else:
        result["class_name"] = result["class_name"].fillna(result["class_id"].map(CLASS_NAME_MAP)).fillna("Unknown")
    return result


def ensure_slide_scores_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(
            columns=[
                "slide_id",
                "label",
                "pred",
                "correct",
                "conflict_type",
                "predicted_scale_relation",
                "low_support_class",
                "high_support_class",
                "label_name",
                "pred_name",
            ]
        )
    required = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "conflict_type",
        "predicted_scale_relation",
        "low_support_class",
        "high_support_class",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Stage17 slide scores CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=required)
    result = df.copy()
    for column in ["label", "pred", "correct", "low_support_class", "high_support_class"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["label"] = result["label"].fillna(-1).astype(int)
    result["pred"] = result["pred"].fillna(-1).astype(int)
    result["correct"] = result["correct"].fillna(0).astype(int)
    result["low_support_class"] = result["low_support_class"].fillna(-1).astype(int)
    result["high_support_class"] = result["high_support_class"].fillna(-1).astype(int)
    if "label_name" not in result.columns:
        result["label_name"] = result["label"].map(CLASS_NAME_MAP).fillna("Unknown")
    if "pred_name" not in result.columns:
        result["pred_name"] = result["pred"].map(CLASS_NAME_MAP).fillna("Unknown")
    return result


def ensure_conflict_summary_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["summary_scope", "group_key", "group_value", "conflict_type", "count", "rate"])
    required = ["summary_scope", "group_key", "group_value", "conflict_type", "count", "rate"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Stage17 conflict summary CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=required)
    result = df.copy()
    for column in ["count", "rate"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def ensure_failure_pattern_df(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["slide_id", "conflict_type", "step16_failure_case", "error_direction"])
    required = ["slide_id", "conflict_type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        warn_message(f"Stage17 failure pattern CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=required)
    result = df.copy()
    if "step16_failure_case" in result.columns:
        result["step16_failure_case"] = pd.to_numeric(result["step16_failure_case"], errors="coerce").fillna(0).astype(int)
    else:
        result["step16_failure_case"] = 0
    if "error_direction" not in result.columns:
        result["error_direction"] = ""
    return result


def truncate_text(text: str, limit: int = 120) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def build_merged_concepts(top_concepts_df: pd.DataFrame, stage14_edges_df: pd.DataFrame) -> pd.DataFrame:
    if top_concepts_df.empty:
        return top_concepts_df.copy()
    edge_subset = stage14_edges_df[
        [column for column in ["scale", "class_id", "concept_id", "edge_strength", "mean_evidence_score", "slide_coverage", "passes_min_count"] if column in stage14_edges_df.columns]
    ].drop_duplicates(["scale", "class_id", "concept_id"])
    merged = top_concepts_df.merge(edge_subset, on=["scale", "class_id", "concept_id"], how="left")
    merged["edge_strength"] = pd.to_numeric(merged["edge_strength"], errors="coerce").fillna(0.0)
    merged["mean_evidence_score"] = pd.to_numeric(merged["mean_evidence_score"], errors="coerce")
    merged["slide_coverage"] = pd.to_numeric(merged["slide_coverage"], errors="coerce")
    merged["passes_min_count"] = merged["passes_min_count"].fillna(False).astype(bool)
    merged["aggregated_component"] = (
        merged["evidence_score"].astype(float)
        * merged["prompt_weight"].astype(float)
        * merged["edge_strength"].astype(float)
        / np.maximum(merged["rank"].astype(float), 1.0)
    )
    return merged


def build_concept_class_edges(stage14_edges_df: pd.DataFrame, top_k_edges: int) -> pd.DataFrame:
    if stage14_edges_df.empty:
        return pd.DataFrame(
            columns=[
                "edge_type",
                "source_node_id",
                "target_node_id",
                "scale",
                "class_id",
                "class_name",
                "concept_id",
                "concept_text",
                "edge_strength",
                "mean_evidence_score",
                "slide_coverage",
            ]
        )
    filtered = stage14_edges_df.loc[stage14_edges_df["passes_min_count"] == True].copy() if "passes_min_count" in stage14_edges_df.columns else stage14_edges_df.copy()
    if filtered.empty:
        filtered = stage14_edges_df.copy()

    selected_frames = []
    for (scale, class_id), group in filtered.groupby(["scale", "class_id"], dropna=False):
        selected_frames.append(group.sort_values(["edge_strength", "mean_evidence_score", "slide_coverage"], ascending=[False, False, False]).head(int(top_k_edges)).copy())
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(columns=filtered.columns)
    if selected.empty:
        return pd.DataFrame(columns=["edge_type"])
    selected["node_prefix"] = np.where(selected["scale"] == "low", "low_concept", "high_concept")
    selected["source_node_id"] = selected["node_prefix"] + "::" + selected["concept_id"].astype(str)
    selected["target_node_id"] = "class::" + selected["class_id"].astype(str)
    selected["edge_type"] = "concept_to_class"
    return selected[
        [
            "edge_type",
            "source_node_id",
            "target_node_id",
            "scale",
            "class_id",
            "class_name",
            "concept_id",
            "concept_text",
            "edge_strength",
            "mean_evidence_score",
            "slide_coverage",
        ]
    ].reset_index(drop=True)


def build_slide_scale_support_concepts(
    merged_concepts_df: pd.DataFrame,
    slide_scores_df: pd.DataFrame,
    min_count: int,
) -> pd.DataFrame:
    if merged_concepts_df.empty or slide_scores_df.empty:
        return pd.DataFrame(
            columns=[
                "slide_id",
                "scale",
                "support_class",
                "concept_id",
                "concept_text",
                "aggregated_component",
                "evidence_score",
                "prompt_weight",
                "rank",
                "edge_strength",
                "conflict_type",
                "correct",
                "label",
                "pred",
            ]
        )
    slide_subset = slide_scores_df[
        [column for column in ["slide_id", "label", "pred", "correct", "conflict_type", "low_support_class", "high_support_class"] if column in slide_scores_df.columns]
    ].drop_duplicates("slide_id")
    rows = []
    for scale_name, support_col in [("low", "low_support_class"), ("high", "high_support_class")]:
        scale_df = merged_concepts_df.loc[merged_concepts_df["scale"] == scale_name].copy()
        merged_scale = scale_df.merge(slide_subset, on="slide_id", how="inner")
        merged_scale = merged_scale.loc[merged_scale["class_id"] == merged_scale[support_col]].copy()
        if merged_scale.empty:
            continue
        merged_scale = merged_scale.sort_values(
            ["slide_id", "rank", "evidence_score", "prompt_weight", "edge_strength"],
            ascending=[True, True, False, False, False],
        )
        merged_scale = merged_scale.groupby("slide_id", as_index=False, group_keys=False).head(int(max(min_count, 1) + 2)).copy()
        merged_scale["support_class"] = merged_scale[support_col].astype(int)
        rows.append(
            merged_scale[
                [
                    "slide_id",
                    "scale",
                    "support_class",
                    "concept_id",
                    "concept_text",
                    "aggregated_component",
                    "evidence_score",
                    "prompt_weight",
                    "rank",
                    "edge_strength",
                    "conflict_type",
                    "correct",
                    "label",
                    "pred",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(columns=["slide_id"])
    return pd.concat(rows, ignore_index=True)


def build_low_high_concept_pairs(
    support_concepts_df: pd.DataFrame,
    slide_scores_df: pd.DataFrame,
    min_count: int,
) -> pd.DataFrame:
    if support_concepts_df.empty or slide_scores_df.empty:
        return pd.DataFrame(
            columns=[
                "low_concept_id",
                "low_concept_text",
                "high_concept_id",
                "high_concept_text",
                "source_node_id",
                "target_node_id",
                "co_occurrence_count",
                "co_occurrence_rate",
                "mean_joint_evidence",
                "correct_count",
                "incorrect_count",
                "dominant_conflict_type",
                "high_scale_override_rate",
                "wrong_class_drift_rate",
                "example_slides",
            ]
        )

    total_slides = max(int(slide_scores_df["slide_id"].nunique()), 1)
    rows = []
    for slide_id, slide_group in support_concepts_df.groupby("slide_id", dropna=False):
        low_group = slide_group.loc[slide_group["scale"] == "low"].copy()
        high_group = slide_group.loc[slide_group["scale"] == "high"].copy()
        if low_group.empty or high_group.empty:
            continue
        slide_meta = slide_scores_df.loc[slide_scores_df["slide_id"] == slide_id].head(1)
        if slide_meta.empty:
            continue
        meta_row = slide_meta.iloc[0]
        for _, low_row in low_group.iterrows():
            for _, high_row in high_group.iterrows():
                joint_evidence = 0.5 * (float(low_row["aggregated_component"]) + float(high_row["aggregated_component"]))
                rows.append(
                    {
                        "slide_id": str(slide_id),
                        "low_concept_id": str(low_row["concept_id"]),
                        "low_concept_text": str(low_row["concept_text"]),
                        "high_concept_id": str(high_row["concept_id"]),
                        "high_concept_text": str(high_row["concept_text"]),
                        "low_support_class": int(low_row["support_class"]),
                        "high_support_class": int(high_row["support_class"]),
                        "correct": int(meta_row["correct"]),
                        "conflict_type": str(meta_row["conflict_type"]),
                        "predicted_scale_relation": str(meta_row["predicted_scale_relation"]),
                        "joint_evidence": joint_evidence,
                    }
                )
    pair_instances_df = pd.DataFrame(rows)
    if pair_instances_df.empty:
        return pd.DataFrame(columns=["low_concept_id"])

    aggregated_rows = []
    for keys, group in pair_instances_df.groupby(["low_concept_id", "low_concept_text", "high_concept_id", "high_concept_text"], dropna=False):
        low_concept_id, low_concept_text, high_concept_id, high_concept_text = keys
        conflict_counts = group["conflict_type"].value_counts()
        dominant_conflict_type = conflict_counts.index[0] if not conflict_counts.empty else ""
        slide_ids = group["slide_id"].astype(str).drop_duplicates().tolist()
        aggregated_rows.append(
            {
                "low_concept_id": str(low_concept_id),
                "low_concept_text": str(low_concept_text),
                "high_concept_id": str(high_concept_id),
                "high_concept_text": str(high_concept_text),
                "source_node_id": f"low_concept::{low_concept_id}",
                "target_node_id": f"high_concept::{high_concept_id}",
                "co_occurrence_count": int(group["slide_id"].nunique()),
                "co_occurrence_rate": float(group["slide_id"].nunique() / total_slides),
                "mean_joint_evidence": float(group["joint_evidence"].mean()),
                "correct_count": int((group["correct"] == 1).sum()),
                "incorrect_count": int((group["correct"] == 0).sum()),
                "dominant_conflict_type": str(dominant_conflict_type),
                "high_scale_override_rate": float(group["conflict_type"].isin(OVERRIDE_CONFLICT_TYPES).mean()),
                "wrong_class_drift_rate": float(group["conflict_type"].isin(DRIFT_CONFLICT_TYPES).mean()),
                "example_slides": json.dumps(slide_ids[:12], ensure_ascii=True),
            }
        )
    pairs_df = pd.DataFrame(aggregated_rows)
    pairs_df = pairs_df.loc[pairs_df["co_occurrence_count"] >= int(min_count)].copy()
    pairs_df = pairs_df.sort_values(
        ["co_occurrence_count", "mean_joint_evidence", "high_scale_override_rate", "wrong_class_drift_rate"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return pairs_df


def build_conflict_pattern_edges(slide_scores_df: pd.DataFrame) -> pd.DataFrame:
    if slide_scores_df.empty:
        return pd.DataFrame(
            columns=[
                "edge_type",
                "source_node_id",
                "target_node_id",
                "conflict_type",
                "class_id",
                "class_name",
                "count",
                "incorrect_rate",
                "main_true_label",
                "main_pred_label",
            ]
        )
    rows = []
    for (conflict_type, pred_class), group in slide_scores_df.groupby(["conflict_type", "pred"], dropna=False):
        true_label_counts = group["label_name"].value_counts()
        pred_label_counts = group["pred_name"].value_counts()
        rows.append(
            {
                "edge_type": "conflict_to_class",
                "source_node_id": f"conflict_type::{conflict_type}",
                "target_node_id": f"class::{int(pred_class)}",
                "conflict_type": str(conflict_type),
                "class_id": int(pred_class),
                "class_name": CLASS_NAME_MAP.get(int(pred_class), "Unknown"),
                "count": int(len(group)),
                "incorrect_rate": float((group["correct"] == 0).mean()),
                "main_true_label": str(true_label_counts.index[0]) if not true_label_counts.empty else "",
                "main_pred_label": str(pred_label_counts.index[0]) if not pred_label_counts.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "incorrect_rate"], ascending=[False, False]).reset_index(drop=True)


def build_nodes_df(
    concept_class_edges_df: pd.DataFrame,
    low_high_pairs_df: pd.DataFrame,
    conflict_pattern_edges_df: pd.DataFrame,
) -> pd.DataFrame:
    node_rows = []
    for class_id, class_name in CLASS_NAME_MAP.items():
        node_rows.append(
            {
                "node_id": f"class::{class_id}",
                "node_type": "class",
                "scale": "both",
                "class_id": class_id,
                "class_name": class_name,
                "concept_id": "",
                "concept_text": class_name,
                "conflict_type": "",
            }
        )

    concept_records = {}
    if not concept_class_edges_df.empty:
        for _, row in concept_class_edges_df.iterrows():
            node_id = str(row["source_node_id"])
            concept_records[node_id] = {
                "node_id": node_id,
                "node_type": "low_concept" if str(row["scale"]) == "low" else "high_concept",
                "scale": str(row["scale"]),
                "class_id": np.nan,
                "class_name": "",
                "concept_id": str(row["concept_id"]),
                "concept_text": str(row["concept_text"]),
                "conflict_type": "",
            }
    if not low_high_pairs_df.empty:
        for _, row in low_high_pairs_df.iterrows():
            low_node_id = str(row["source_node_id"])
            high_node_id = str(row["target_node_id"])
            if low_node_id not in concept_records:
                concept_records[low_node_id] = {
                    "node_id": low_node_id,
                    "node_type": "low_concept",
                    "scale": "low",
                    "class_id": np.nan,
                    "class_name": "",
                    "concept_id": str(row["low_concept_id"]),
                    "concept_text": str(row["low_concept_text"]),
                    "conflict_type": "",
                }
            if high_node_id not in concept_records:
                concept_records[high_node_id] = {
                    "node_id": high_node_id,
                    "node_type": "high_concept",
                    "scale": "high",
                    "class_id": np.nan,
                    "class_name": "",
                    "concept_id": str(row["high_concept_id"]),
                    "concept_text": str(row["high_concept_text"]),
                    "conflict_type": "",
                }
    node_rows.extend(concept_records.values())

    if not conflict_pattern_edges_df.empty:
        for conflict_type in conflict_pattern_edges_df["conflict_type"].dropna().astype(str).unique():
            node_rows.append(
                {
                    "node_id": f"conflict_type::{conflict_type}",
                    "node_type": "conflict_type",
                    "scale": "both",
                    "class_id": np.nan,
                    "class_name": "",
                    "concept_id": "",
                    "concept_text": conflict_type,
                    "conflict_type": conflict_type,
                }
            )

    nodes_df = pd.DataFrame(node_rows).drop_duplicates(subset=["node_id"]).reset_index(drop=True)
    return nodes_df


def build_combined_edges_df(
    concept_class_edges_df: pd.DataFrame,
    low_high_pairs_df: pd.DataFrame,
    conflict_pattern_edges_df: pd.DataFrame,
) -> pd.DataFrame:
    edge_frames = []
    if not concept_class_edges_df.empty:
        edge_frames.append(concept_class_edges_df.copy())
    if not low_high_pairs_df.empty:
        pairs_df = low_high_pairs_df.copy()
        pairs_df["edge_type"] = "low_high_pair"
        edge_frames.append(pairs_df)
    if not conflict_pattern_edges_df.empty:
        edge_frames.append(conflict_pattern_edges_df.copy())
    if not edge_frames:
        return pd.DataFrame(columns=["edge_type", "source_node_id", "target_node_id"])
    return pd.concat(edge_frames, ignore_index=True, sort=False)


def build_graph_json(
    nodes_df: pd.DataFrame,
    concept_class_edges_df: pd.DataFrame,
    low_high_pairs_df: pd.DataFrame,
    conflict_pattern_edges_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "graph_type": "post_hoc_cross_scale_evidence_graph_prototype",
        "node_count": int(len(nodes_df)),
        "concept_class_edge_count": int(len(concept_class_edges_df)),
        "low_high_pair_edge_count": int(len(low_high_pairs_df)),
        "conflict_pattern_edge_count": int(len(conflict_pattern_edges_df)),
        "nodes": nodes_df.fillna("").to_dict(orient="records"),
        "concept_class_edges": concept_class_edges_df.fillna("").to_dict(orient="records"),
        "low_high_concept_pairs": low_high_pairs_df.fillna("").to_dict(orient="records"),
        "conflict_pattern_edges": conflict_pattern_edges_df.fillna("").to_dict(orient="records"),
    }


def build_summary_df(
    slide_scores_df: pd.DataFrame,
    concept_class_edges_df: pd.DataFrame,
    low_high_pairs_df: pd.DataFrame,
    conflict_pattern_edges_df: pd.DataFrame,
    warning_log: list[str],
    generated_files: list[Path],
) -> pd.DataFrame:
    rows = [
        {"section": "inputs", "metric": "slide_count", "value": int(slide_scores_df["slide_id"].nunique()) if not slide_scores_df.empty else 0, "note": ""},
        {"section": "graph", "metric": "concept_class_edges", "value": int(len(concept_class_edges_df)), "note": ""},
        {"section": "graph", "metric": "low_high_pair_edges", "value": int(len(low_high_pairs_df)), "note": ""},
        {"section": "graph", "metric": "conflict_pattern_edges", "value": int(len(conflict_pattern_edges_df)), "note": ""},
        {"section": "quality", "metric": "warning_count", "value": int(len(warning_log)), "note": ""},
    ]
    if not low_high_pairs_df.empty:
        top_override = low_high_pairs_df.loc[
            (low_high_pairs_df["high_scale_override_rate"] > 0.0) & (low_high_pairs_df["incorrect_count"] > 0)
        ].sort_values(["incorrect_count", "high_scale_override_rate", "co_occurrence_count"], ascending=[False, False, False]).head(5)
        if top_override.empty:
            top_override = low_high_pairs_df.sort_values(["high_scale_override_rate", "co_occurrence_count"], ascending=[False, False]).head(5)
        for _, row in top_override.iterrows():
            rows.append(
                {
                    "section": "override_pairs",
                    "metric": "pair",
                    "value": float(row["high_scale_override_rate"]),
                    "note": f"{row['low_concept_id']} <-> {row['high_concept_id']}",
                }
            )
        top_drift = low_high_pairs_df.loc[
            (low_high_pairs_df["wrong_class_drift_rate"] > 0.0) & (low_high_pairs_df["incorrect_count"] > 0)
        ].sort_values(["wrong_class_drift_rate", "incorrect_count", "co_occurrence_count"], ascending=[False, False, False]).head(5)
        if top_drift.empty:
            top_drift = low_high_pairs_df.sort_values(["wrong_class_drift_rate", "co_occurrence_count"], ascending=[False, False]).head(5)
        for _, row in top_drift.iterrows():
            rows.append(
                {
                    "section": "drift_pairs",
                    "metric": "pair",
                    "value": float(row["wrong_class_drift_rate"]),
                    "note": f"{row['low_concept_id']} <-> {row['high_concept_id']}",
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
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def failure_case_pair_summary(
    slide_scores_df: pd.DataFrame,
    low_high_pairs_df: pd.DataFrame,
    slide_id: str,
) -> dict[str, object]:
    if slide_scores_df.empty or low_high_pairs_df.empty:
        return {}
    row_df = slide_scores_df.loc[slide_scores_df["slide_id"] == slide_id].head(1)
    if row_df.empty:
        return {}
    row = row_df.iloc[0]
    low_support = int(row["low_support_class"])
    high_support = int(row["high_support_class"])
    low_ids_raw = row.get(f"low_top_concept_ids_{low_support}", "[]")
    high_ids_raw = row.get(f"high_top_concept_ids_{high_support}", "[]")
    try:
        low_ids = json.loads(low_ids_raw) if isinstance(low_ids_raw, str) else []
    except Exception:
        low_ids = []
    try:
        high_ids = json.loads(high_ids_raw) if isinstance(high_ids_raw, str) else []
    except Exception:
        high_ids = []
    for low_id in low_ids:
        for high_id in high_ids:
            match = low_high_pairs_df.loc[
                (low_high_pairs_df["low_concept_id"] == str(low_id))
                & (low_high_pairs_df["high_concept_id"] == str(high_id))
            ].head(1)
            if not match.empty:
                matched = match.iloc[0]
                return {
                    "slide_id": slide_id,
                    "low_concept_id": str(matched["low_concept_id"]),
                    "high_concept_id": str(matched["high_concept_id"]),
                    "co_occurrence_count": int(matched["co_occurrence_count"]),
                    "high_scale_override_rate": float(matched["high_scale_override_rate"]),
                    "wrong_class_drift_rate": float(matched["wrong_class_drift_rate"]),
                    "dominant_conflict_type": str(matched["dominant_conflict_type"]),
                }
    return {"slide_id": slide_id}


def build_report(
    root: Path,
    evidence_dir: Path,
    graph_dir: Path,
    conflict_dir: Path,
    slide_scores_df: pd.DataFrame,
    concept_class_edges_df: pd.DataFrame,
    low_high_pairs_df: pd.DataFrame,
    conflict_pattern_edges_df: pd.DataFrame,
    warning_log: list[str],
) -> str:
    slide_count = int(slide_scores_df["slide_id"].nunique()) if not slide_scores_df.empty else 0
    top_pairs = low_high_pairs_df.head(12).copy() if not low_high_pairs_df.empty else pd.DataFrame()
    if not top_pairs.empty:
        top_pairs = top_pairs[
            [
                "low_concept_id",
                "high_concept_id",
                "co_occurrence_count",
                "co_occurrence_rate",
                "mean_joint_evidence",
                "dominant_conflict_type",
                "high_scale_override_rate",
                "wrong_class_drift_rate",
            ]
        ].copy()
        for column in ["co_occurrence_rate", "mean_joint_evidence", "high_scale_override_rate", "wrong_class_drift_rate"]:
            top_pairs[column] = pd.to_numeric(top_pairs[column], errors="coerce").map(lambda value: f"{float(value):.4f}")

    override_pairs = (
        low_high_pairs_df.loc[(low_high_pairs_df["high_scale_override_rate"] > 0.0) & (low_high_pairs_df["incorrect_count"] > 0)]
        .sort_values(["incorrect_count", "high_scale_override_rate", "co_occurrence_count"], ascending=[False, False, False])
        .head(8)
        .copy()
        if not low_high_pairs_df.empty
        else pd.DataFrame()
    )
    if override_pairs.empty and not low_high_pairs_df.empty:
        override_pairs = low_high_pairs_df.sort_values(["high_scale_override_rate", "co_occurrence_count"], ascending=[False, False]).head(8).copy()

    drift_pairs = (
        low_high_pairs_df.loc[(low_high_pairs_df["wrong_class_drift_rate"] > 0.0) & (low_high_pairs_df["incorrect_count"] > 0)]
        .sort_values(["wrong_class_drift_rate", "incorrect_count", "co_occurrence_count"], ascending=[False, False, False])
        .head(8)
        .copy()
        if not low_high_pairs_df.empty
        else pd.DataFrame()
    )
    if drift_pairs.empty and not low_high_pairs_df.empty:
        drift_pairs = low_high_pairs_df.sort_values(["wrong_class_drift_rate", "co_occurrence_count"], ascending=[False, False]).head(8).copy()

    for frame in [override_pairs, drift_pairs]:
        if not frame.empty:
            for column in ["co_occurrence_rate", "mean_joint_evidence", "high_scale_override_rate", "wrong_class_drift_rate"]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").map(lambda value: f"{float(value):.4f}")

    failure_summaries = [failure_case_pair_summary(slide_scores_df, low_high_pairs_df, slide_id) for slide_id in EXPECTED_FAILURE_SLIDES]

    lines = [
        "# Stage18 Cross-Scale Evidence Graph Prototype",
        "",
        "Step18 builds a post-hoc cross-scale evidence graph prototype from the existing Step13d, Step14, and Step17 outputs.",
        "",
        "- It is not a learnable model.",
        "- It does not modify the current model.",
        "- It does not run training.",
        "",
        "## Inputs",
        "",
        f"- root: `{root}`",
        f"- evidence_dir: `{evidence_dir}`",
        f"- graph_dir: `{graph_dir}`",
        f"- conflict_dir: `{conflict_dir}`",
        f"- fold0_slide_count: `{slide_count}`",
        "",
        "## Graph Structure",
        "",
        "- nodes: `low_concept`, `high_concept`, `class`, `conflict_type`",
        "- edges: `concept_to_class`, `low_high_pair`, `conflict_to_class`",
        "",
    ]

    if not top_pairs.empty:
        lines.extend(["## Top Low/High Concept Pairs", "", markdown_table(top_pairs), ""])

    if not override_pairs.empty:
        lines.extend(
            [
                "## High-Scale Override Pairs",
                "",
                markdown_table(
                    override_pairs[
                        [
                            "low_concept_id",
                            "high_concept_id",
                            "co_occurrence_count",
                            "incorrect_count",
                            "high_scale_override_rate",
                            "dominant_conflict_type",
                        ]
                    ]
                ),
                "",
            ]
        )

    if not drift_pairs.empty:
        lines.extend(
            [
                "## Consistent Wrong-Class Drift Pairs",
                "",
                markdown_table(
                    drift_pairs[
                        [
                            "low_concept_id",
                            "high_concept_id",
                            "co_occurrence_count",
                            "incorrect_count",
                            "wrong_class_drift_rate",
                            "dominant_conflict_type",
                        ]
                    ]
                ),
                "",
            ]
        )

    if not conflict_pattern_edges_df.empty:
        conflict_display = conflict_pattern_edges_df[
            ["conflict_type", "class_name", "count", "incorrect_rate", "main_true_label", "main_pred_label"]
        ].copy()
        conflict_display["incorrect_rate"] = pd.to_numeric(conflict_display["incorrect_rate"], errors="coerce").map(lambda value: f"{float(value):.4f}")
        lines.extend(["## Conflict-Type To Class Edges", "", markdown_table(conflict_display), ""])

    lines.extend(["## Step17 Failure Pattern Interpretation", ""])
    if failure_summaries:
        for summary in failure_summaries:
            slide_id = summary.get("slide_id", "")
            lines.append(f"### {slide_id}")
            if not summary or len(summary.keys()) == 1:
                lines.extend(["", "- No aggregated low/high concept pair was recovered for this slide under the retained graph thresholds.", ""])
                continue
            lines.extend(
                [
                    "",
                    f"- low/high concept pair: `{summary['low_concept_id']} <-> {summary['high_concept_id']}`",
                    f"- co_occurrence_count: `{summary['co_occurrence_count']}`",
                    f"- dominant_conflict_type: `{summary['dominant_conflict_type']}`",
                    f"- high_scale_override_rate: `{summary['high_scale_override_rate']:.4f}`",
                    f"- wrong_class_drift_rate: `{summary['wrong_class_drift_rate']:.4f}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "Step18 is a post-hoc graph prototype only. It structures existing cross-scale evidence into graph-form CSV/JSON outputs, but it is not a learnable graph module and does not change the trained model.",
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
            "- Step19 learnable cross-scale graph module prototype",
            "- Step19 learnable concept-class graph module prototype",
            "",
        ]
    )
    return "\n".join(lines)


def run_graph_builder(args: argparse.Namespace) -> dict[str, object]:
    warning_log: list[str] = []
    root = resolve_path(DEFAULT_ROOT, args.root)
    evidence_dir = resolve_path(root, args.evidence_dir)
    graph_dir = resolve_path(root, args.graph_dir)
    conflict_dir = resolve_path(root, args.conflict_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_concepts_path = evidence_dir / "slide_top_concepts.csv"
    stage14_edges_path = graph_dir / "stage14_concept_class_edges.csv"
    slide_scores_path = conflict_dir / "stage17_cross_scale_slide_scores.csv"
    conflict_summary_path = conflict_dir / "stage17_conflict_type_summary.csv"
    failure_patterns_path = conflict_dir / "stage17_failure_pattern_cases.csv"

    top_concepts_df = ensure_top_concepts_df(safe_read_csv(top_concepts_path, warning_log), warning_log)
    stage14_edges_df = ensure_stage14_edges_df(safe_read_csv(stage14_edges_path, warning_log), warning_log)
    slide_scores_df = ensure_slide_scores_df(safe_read_csv(slide_scores_path, warning_log), warning_log)
    conflict_summary_df = ensure_conflict_summary_df(safe_read_csv(conflict_summary_path, warning_log), warning_log)
    failure_patterns_df = ensure_failure_pattern_df(safe_read_csv(failure_patterns_path, warning_log), warning_log)

    merged_concepts_df = build_merged_concepts(top_concepts_df, stage14_edges_df)
    concept_class_edges_df = build_concept_class_edges(stage14_edges_df, top_k_edges=max(int(args.top_k_edges), 1))
    support_concepts_df = build_slide_scale_support_concepts(
        merged_concepts_df=merged_concepts_df,
        slide_scores_df=slide_scores_df,
        min_count=max(int(args.min_count), 1),
    )
    low_high_pairs_df = build_low_high_concept_pairs(
        support_concepts_df=support_concepts_df,
        slide_scores_df=slide_scores_df,
        min_count=max(int(args.min_count), 1),
    )
    conflict_pattern_edges_df = build_conflict_pattern_edges(slide_scores_df)
    nodes_df = build_nodes_df(concept_class_edges_df, low_high_pairs_df, conflict_pattern_edges_df)
    combined_edges_df = build_combined_edges_df(concept_class_edges_df, low_high_pairs_df, conflict_pattern_edges_df)

    nodes_csv = out_dir / "stage18_cross_scale_nodes.csv"
    edges_csv = out_dir / "stage18_cross_scale_edges.csv"
    pairs_csv = out_dir / "stage18_low_high_concept_pairs.csv"
    conflict_edges_csv = out_dir / "stage18_conflict_pattern_edges.csv"
    graph_json_path = out_dir / "stage18_cross_scale_graph.json"
    report_md = out_dir / "stage18_cross_scale_graph_report.md"
    summary_csv = out_dir / "stage18_cross_scale_graph_summary.csv"

    nodes_df.to_csv(nodes_csv, index=False)
    combined_edges_df.to_csv(edges_csv, index=False)
    low_high_pairs_df.to_csv(pairs_csv, index=False)
    conflict_pattern_edges_df.to_csv(conflict_edges_csv, index=False)

    graph_payload = build_graph_json(nodes_df, concept_class_edges_df, low_high_pairs_df, conflict_pattern_edges_df)
    graph_json_path.write_text(json.dumps(graph_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    generated_files = [nodes_csv, edges_csv, pairs_csv, conflict_edges_csv, graph_json_path]
    summary_df = build_summary_df(
        slide_scores_df=slide_scores_df,
        concept_class_edges_df=concept_class_edges_df,
        low_high_pairs_df=low_high_pairs_df,
        conflict_pattern_edges_df=conflict_pattern_edges_df,
        warning_log=warning_log,
        generated_files=generated_files,
    )
    summary_df.to_csv(summary_csv, index=False)
    generated_files.append(summary_csv)

    report_text = build_report(
        root=root,
        evidence_dir=evidence_dir,
        graph_dir=graph_dir,
        conflict_dir=conflict_dir,
        slide_scores_df=slide_scores_df,
        concept_class_edges_df=concept_class_edges_df,
        low_high_pairs_df=low_high_pairs_df,
        conflict_pattern_edges_df=conflict_pattern_edges_df,
        warning_log=warning_log,
    )
    report_md.write_text(report_text, encoding="utf-8")
    generated_files.append(report_md)

    print(f"[Step18] Slide count: {slide_scores_df['slide_id'].nunique() if not slide_scores_df.empty else 0}")
    print(f"[Step18] Nodes: {len(nodes_df)}")
    print(f"[Step18] Combined edges: {len(combined_edges_df)}")
    print(f"[Step18] Pair edges: {len(low_high_pairs_df)}")
    print(f"[Step18] Warnings: {len(warning_log)}")
    print(f"[Step18] Output directory: {out_dir}")

    return {
        "root": root,
        "out_dir": out_dir,
        "nodes_df": nodes_df,
        "edges_df": combined_edges_df,
        "pairs_df": low_high_pairs_df,
        "conflict_edges_df": conflict_pattern_edges_df,
        "summary_df": summary_df,
        "warning_log": warning_log,
        "conflict_summary_df": conflict_summary_df,
        "failure_patterns_df": failure_patterns_df,
    }


def main() -> None:
    args = parse_args()
    try:
        run_graph_builder(args)
    except Exception as exc:
        root = resolve_path(DEFAULT_ROOT, args.root)
        out_dir = resolve_path(root, args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "stage18_cross_scale_graph_report.md"
        lines = [
            "# Stage18 Cross-Scale Evidence Graph Prototype",
            "",
            "The script exited early because an unexpected error occurred.",
            "",
            f"- error: `{exc}`",
            "- This step is intended to remain post-hoc only: no model change, no training, no feature extraction.",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Step18 warning] Unexpected failure: {exc}")
        print(f"[Step18] Wrote fallback report to: {report_path}")


if __name__ == "__main__":
    main()
