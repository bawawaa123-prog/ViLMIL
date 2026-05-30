from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = Path("results_stage9/stage13_rce_evidence_export_fold0_test_full")
DEFAULT_OUT_DIR = Path("results_stage9/stage14_concept_class_graph_fold0")
CLASS_NAME_MAP = {0: "Adenocarcinoma", 1: "NonAdenocarcinoma"}
CORE_INPUTS = {
    "prediction_csv": "slide_prediction_evidence.csv",
    "top_concepts_csv": "slide_top_concepts.csv",
    "evidence_pkl": "region_concept_evidence.pkl",
    "quality_report_md": "stage13_rce_evidence_quality_report.md",
}


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post-hoc Stage14 concept-class evidence graph.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to ViLa-MIL-main root.")
    parser.add_argument(
        "--evidence_dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing Step13d full evidence export files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for graph-analysis outputs.",
    )
    parser.add_argument("--top_k_edges", type=int, default=20, help="Top-K edges highlighted in report.")
    parser.add_argument("--min_count", type=int, default=5, help="Minimum top-k count for edges retained in summary.")
    return parser.parse_args()


def resolve_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


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
        warn_message(f"Missing Markdown report: {path}", warning_log)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warn_message(f"Failed to read text {path}: {exc}", warning_log)
        return ""


def safe_load_pickle(path: Path, warning_log: list[str]):
    if not path.is_file():
        warn_message(f"Missing pickle: {path}", warning_log)
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        warn_message(f"Failed to load pickle {path}: {exc}", warning_log)
        return None


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
        return np.asarray(value)
    try:
        return np.asarray(value)
    except Exception:
        return None


def shape_string(value) -> str:
    arr = to_numpy(value)
    if arr is None:
        return "missing"
    return str(tuple(arr.shape))


def contains_nan_or_inf(value) -> tuple[bool, bool]:
    arr = to_numpy(value)
    if arr is None or arr.size == 0:
        return False, False
    try:
        return bool(np.isnan(arr).any()), bool(np.isinf(arr).any())
    except Exception:
        return False, False


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("NA").astype(str)
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def extract_exported_slides(report_text: str) -> int | None:
    for line in report_text.splitlines():
        if "exported_slides" in line:
            try:
                return int(line.split("`")[1])
            except Exception:
                return None
    return None


def build_merged_evidence(prediction_df: pd.DataFrame | None, concepts_df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if prediction_df is None or concepts_df is None:
        return pd.DataFrame()

    required_pred_cols = ["slide_id", "label", "pred", "correct", "prob_0", "prob_1"]
    required_concept_cols = [
        "slide_id",
        "scale",
        "class_id",
        "concept_id",
        "concept_text",
        "evidence_score",
        "prompt_weight",
        "rank",
    ]
    missing_pred = [col for col in required_pred_cols if col not in prediction_df.columns]
    missing_concept = [col for col in required_concept_cols if col not in concepts_df.columns]
    if missing_pred:
        warn_message(f"Prediction CSV missing required columns: {missing_pred}", warning_log)
        return pd.DataFrame()
    if missing_concept:
        warn_message(f"Top concepts CSV missing required columns: {missing_concept}", warning_log)
        return pd.DataFrame()

    merged = concepts_df.merge(
        prediction_df[required_pred_cols],
        on="slide_id",
        how="left",
        validate="many_to_one",
    )
    if merged["label"].isna().any():
        warn_message("Some top-concept rows could not be aligned to prediction slide_id values.", warning_log)

    merged["class_id"] = pd.to_numeric(merged["class_id"], errors="coerce")
    merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce")
    merged["evidence_score"] = pd.to_numeric(merged["evidence_score"], errors="coerce")
    merged["prompt_weight"] = pd.to_numeric(merged["prompt_weight"], errors="coerce")
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
    merged["pred"] = pd.to_numeric(merged["pred"], errors="coerce")
    merged["correct"] = pd.to_numeric(merged["correct"], errors="coerce")
    merged["class_name"] = merged["class_id"].map(CLASS_NAME_MAP).fillna("Unknown")
    merged["concept_text"] = merged["concept_text"].fillna("").astype(str).str.strip()
    merged["concept_label"] = np.where(merged["concept_text"] != "", merged["concept_text"], merged["concept_id"].astype(str))
    return merged


def aggregate_edges(merged_df: pd.DataFrame, total_slides: int, min_count: int) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame()

    group_cols = ["scale", "class_id", "class_name", "concept_id", "concept_label"]
    rows: list[dict[str, object]] = []
    for keys, group in merged_df.groupby(group_cols, dropna=False):
        scale, class_id, class_name, concept_id, concept_label = keys
        evidence = group["evidence_score"].dropna()
        weights = group["prompt_weight"].dropna()
        rank = group["rank"].dropna()
        slide_count = group["slide_id"].nunique()
        slide_coverage = float(slide_count / total_slides) if total_slides > 0 else math.nan
        mean_evidence = float(evidence.mean()) if not evidence.empty else math.nan
        std_evidence = float(evidence.std(ddof=0)) if not evidence.empty else math.nan
        mean_weight = float(weights.mean()) if not weights.empty else math.nan
        mean_rank = float(rank.mean()) if not rank.empty else math.nan

        correct_group = group.loc[group["correct"] == 1, "evidence_score"].dropna()
        incorrect_group = group.loc[group["correct"] == 0, "evidence_score"].dropna()
        label0_group = group.loc[group["label"] == 0, "evidence_score"].dropna()
        label1_group = group.loc[group["label"] == 1, "evidence_score"].dropna()
        pred_class_group = group.loc[group["pred"] == class_id, "evidence_score"].dropna()

        rank_factor = 1.0 / mean_rank if not math.isnan(mean_rank) and mean_rank > 0 else 0.0
        edge_strength = mean_evidence * slide_coverage * rank_factor if not math.isnan(mean_evidence) else math.nan
        label_gap = (
            float(label0_group.mean()) - float(label1_group.mean())
            if not label0_group.empty and not label1_group.empty
            else math.nan
        )
        correctness_gap = (
            float(correct_group.mean()) - float(incorrect_group.mean())
            if not correct_group.empty and not incorrect_group.empty
            else math.nan
        )

        rows.append(
            {
                "scale": scale,
                "class_id": int(class_id) if not pd.isna(class_id) else -1,
                "class_name": class_name,
                "concept_id": concept_id,
                "concept_text": concept_label,
                "n_topk": int(len(group)),
                "slide_coverage_count": int(slide_count),
                "slide_coverage": slide_coverage,
                "mean_evidence_score": mean_evidence,
                "std_evidence_score": std_evidence,
                "mean_prompt_weight": mean_weight,
                "mean_rank": mean_rank,
                "correct_slide_mean_evidence": float(correct_group.mean()) if not correct_group.empty else math.nan,
                "incorrect_slide_mean_evidence": float(incorrect_group.mean()) if not incorrect_group.empty else math.nan,
                "label0_mean_evidence": float(label0_group.mean()) if not label0_group.empty else math.nan,
                "label1_mean_evidence": float(label1_group.mean()) if not label1_group.empty else math.nan,
                "predicted_class_mean_evidence": float(pred_class_group.mean()) if not pred_class_group.empty else math.nan,
                "label_evidence_gap": label_gap,
                "correctness_evidence_gap": correctness_gap,
                "edge_strength": edge_strength,
            }
        )

    edges_df = pd.DataFrame(rows)
    if edges_df.empty:
        return edges_df
    edges_df = edges_df.sort_values(
        ["edge_strength", "mean_evidence_score", "slide_coverage", "mean_prompt_weight"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    edges_df["is_high_frequency_low_distinction"] = (
        (edges_df["slide_coverage"] >= 0.8)
        & (edges_df["label_evidence_gap"].abs().fillna(0.0) < 0.02)
        & (edges_df["correctness_evidence_gap"].abs().fillna(0.0) < 0.02)
    )
    edges_df["passes_min_count"] = edges_df["n_topk"] >= int(min_count)
    return edges_df


def build_nodes(edges_df: pd.DataFrame) -> pd.DataFrame:
    class_rows = [
        {"node_type": "class", "node_id": f"class_{class_id}", "class_id": class_id, "class_name": class_name, "scale": "both", "concept_id": "", "concept_text": class_name}
        for class_id, class_name in CLASS_NAME_MAP.items()
    ]
    if edges_df.empty:
        return pd.DataFrame(class_rows)

    concept_rows = (
        edges_df[["scale", "concept_id", "concept_text"]]
        .drop_duplicates()
        .assign(
            node_type="concept",
            node_id=lambda df: df["scale"].astype(str) + "::" + df["concept_id"].astype(str),
            class_id=np.nan,
            class_name="",
        )
    )
    concept_rows = concept_rows[["node_type", "node_id", "class_id", "class_name", "scale", "concept_id", "concept_text"]]
    return pd.concat([pd.DataFrame(class_rows), concept_rows], ignore_index=True)


def summarize_pickle(payload, expected_slides: int, warning_log: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    summary = {
        "pkl_entry_count": 0 if payload is None or not hasattr(payload, "__len__") else len(payload),
        "pkl_matches_slide_count": False,
        "final_logits_shapes": "",
        "visual_logits_shapes": "",
        "has_nan": False,
        "has_inf": False,
    }
    if payload is None or not isinstance(payload, list):
        if payload is not None:
            warn_message(f"Evidence pickle root object is not a list: {type(payload).__name__}", warning_log)
        return summary, pd.DataFrame()

    final_shapes = []
    visual_shapes = []
    shape_rows = []
    any_nan = False
    any_inf = False
    for item in payload:
        if not isinstance(item, dict):
            continue
        final_shape = shape_string(item.get("final_logits"))
        visual_shape = shape_string(item.get("visual_logits"))
        final_shapes.append(final_shape)
        visual_shapes.append(visual_shape)
        shape_rows.append(
            {
                "slide_id": item.get("slide_id", ""),
                "final_logits_shape": final_shape,
                "visual_logits_shape": visual_shape,
                "low_region_shape": shape_string(item.get("low_region_concept_sim")),
                "high_region_shape": shape_string(item.get("high_region_concept_sim")),
            }
        )
        for key in ["final_logits", "visual_logits", "low_region_concept_sim", "high_region_concept_sim"]:
            has_nan, has_inf = contains_nan_or_inf(item.get(key))
            any_nan = any_nan or has_nan
            any_inf = any_inf or has_inf

    summary["pkl_matches_slide_count"] = summary["pkl_entry_count"] == expected_slides
    summary["final_logits_shapes"] = "; ".join(f"{shape}:{count}" for shape, count in pd.Series(final_shapes).value_counts().sort_index().items())
    summary["visual_logits_shapes"] = "; ".join(f"{shape}:{count}" for shape, count in pd.Series(visual_shapes).value_counts().sort_index().items())
    summary["has_nan"] = any_nan
    summary["has_inf"] = any_inf
    if any_nan:
        warn_message("NaN values detected in evidence pickle tensors.", warning_log)
    if any_inf:
        warn_message("Inf values detected in evidence pickle tensors.", warning_log)
    if not summary["pkl_matches_slide_count"]:
        warn_message(
            f"Evidence pickle entry count {summary['pkl_entry_count']} does not match merged slide count {expected_slides}.",
            warning_log,
        )
    return summary, pd.DataFrame(shape_rows)


def build_graph_json(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, pkl_summary: dict[str, object]) -> dict[str, object]:
    class_nodes = []
    concept_nodes = []
    for _, row in nodes_df.iterrows():
        node = {
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "scale": row["scale"],
            "class_id": None if pd.isna(row["class_id"]) else int(row["class_id"]),
            "class_name": row["class_name"],
            "concept_id": row["concept_id"],
            "concept_text": row["concept_text"],
        }
        if row["node_type"] == "class":
            class_nodes.append(node)
        else:
            concept_nodes.append(node)

    edge_rows = []
    for _, row in edges_df.iterrows():
        edge_rows.append(
            {
                "source": f"{row['scale']}::{row['concept_id']}",
                "target": f"class_{int(row['class_id'])}",
                "scale": row["scale"],
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "concept_id": row["concept_id"],
                "concept_text": row["concept_text"],
                "edge_strength": float(row["edge_strength"]),
                "mean_evidence_score": float(row["mean_evidence_score"]),
                "slide_coverage": float(row["slide_coverage"]),
                "mean_rank": float(row["mean_rank"]),
                "n_topk": int(row["n_topk"]),
            }
        )

    return {
        "graph_type": "post_hoc_concept_class_evidence_graph",
        "class_nodes": class_nodes,
        "concept_nodes": concept_nodes,
        "concept_to_class_edges": edge_rows,
        "pkl_summary": pkl_summary,
    }


def build_summary_df(merged_df: pd.DataFrame, edges_df: pd.DataFrame, pkl_summary: dict[str, object], warning_log: list[str]) -> pd.DataFrame:
    summary_rows = []
    summary_rows.append(
        {
            "section": "inputs",
            "metric": "merged_rows",
            "value": int(len(merged_df)),
            "note": "",
        }
    )
    if not merged_df.empty:
        summary_rows.extend(
            [
                {"section": "inputs", "metric": "slide_count", "value": int(merged_df["slide_id"].nunique()), "note": ""},
                {"section": "quality", "metric": "export_accuracy", "value": float(merged_df[["slide_id", "correct"]].drop_duplicates()["correct"].mean()), "note": ""},
                {"section": "edges", "metric": "edge_count", "value": int(len(edges_df)), "note": ""},
                {"section": "edges", "metric": "high_frequency_low_distinction_edges", "value": int(edges_df["is_high_frequency_low_distinction"].sum()) if not edges_df.empty else 0, "note": ""},
            ]
        )
    for key, value in pkl_summary.items():
        summary_rows.append({"section": "pkl", "metric": key, "value": value, "note": ""})
    summary_rows.append({"section": "quality", "metric": "warning_count", "value": int(len(warning_log)), "note": ""})
    return pd.DataFrame(summary_rows)


def build_report(
    evidence_dir: Path,
    merged_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    pkl_summary: dict[str, object],
    quality_report_text: str,
    top_k_edges: int,
    warning_log: list[str],
) -> str:
    slide_count = int(merged_df["slide_id"].nunique()) if not merged_df.empty else 0
    accuracy = (
        float(merged_df[["slide_id", "correct"]].drop_duplicates()["correct"].mean())
        if not merged_df.empty
        else math.nan
    )

    top_edges_all = edges_df.loc[edges_df["passes_min_count"]].head(top_k_edges).copy() if not edges_df.empty else pd.DataFrame()
    top_edges_by_class = {}
    top_edges_by_scale = {}
    for class_id, class_name in CLASS_NAME_MAP.items():
        top_edges_by_class[class_id] = (
            edges_df.loc[(edges_df["class_id"] == class_id) & (edges_df["passes_min_count"])]
            .head(top_k_edges)
            .copy()
        )
    for scale in ["low", "high"]:
        top_edges_by_scale[scale] = (
            edges_df.loc[(edges_df["scale"] == scale) & (edges_df["passes_min_count"])]
            .head(top_k_edges)
            .copy()
        )

    correct_gap_df = (
        edges_df.loc[edges_df["passes_min_count"], ["scale", "class_name", "concept_id", "concept_text", "correctness_evidence_gap", "label_evidence_gap", "edge_strength"]]
        .sort_values(["correctness_evidence_gap", "edge_strength"], ascending=[False, False])
        .head(top_k_edges)
        .copy()
        if not edges_df.empty
        else pd.DataFrame()
    )
    high_freq_low_distinction_df = (
        edges_df.loc[edges_df["is_high_frequency_low_distinction"], ["scale", "class_name", "concept_id", "concept_text", "slide_coverage", "label_evidence_gap", "correctness_evidence_gap", "edge_strength"]]
        .sort_values(["slide_coverage", "edge_strength"], ascending=[False, False])
        .head(top_k_edges)
        .copy()
        if not edges_df.empty
        else pd.DataFrame()
    )

    lines = [
        "# Stage14 Concept-Class Evidence Graph",
        "",
        "This is a post-hoc concept-class evidence graph built from exported RCE evidence. It is not a learnable graph model and does not change the current model structure.",
        "",
        "## Input Summary",
        "",
        f"- evidence_dir: `{evidence_dir}`",
        f"- merged_slide_count: `{slide_count}`",
        f"- fold0 full export accuracy: `{accuracy:.4f}`" if not math.isnan(accuracy) else "- fold0 full export accuracy: `NA`",
        f"- pkl_entry_count: `{pkl_summary.get('pkl_entry_count', 0)}`",
        f"- pkl_matches_slide_count: `{pkl_summary.get('pkl_matches_slide_count', False)}`",
        f"- final_logits_shapes: `{pkl_summary.get('final_logits_shapes', '')}`",
        f"- visual_logits_shapes: `{pkl_summary.get('visual_logits_shapes', '')}`",
        "",
        "## Edge Strength Formula",
        "",
        "`edge_strength = mean_evidence_score * slide_coverage * (1 / mean_rank)`",
        "",
        "This favors concepts that are strong, broadly covering, and consistently ranked near the top, without reducing the graph to frequency alone.",
        "",
    ]

    if not top_edges_all.empty:
        lines.extend(["## Overall Top Edges", "", markdown_table(top_edges_all.head(top_k_edges)), ""])

    for class_id, class_name in CLASS_NAME_MAP.items():
        df = top_edges_by_class[class_id]
        if not df.empty:
            lines.extend([f"## Top Edges For {class_name}", "", markdown_table(df.head(top_k_edges)), ""])

    for scale in ["low", "high"]:
        df = top_edges_by_scale[scale]
        if not df.empty:
            lines.extend([f"## Top {scale.capitalize()}-Scale Edges", "", markdown_table(df.head(top_k_edges)), ""])

    if not correct_gap_df.empty:
        lines.extend(
            [
                "## Correct vs Incorrect Evidence Gap",
                "",
                markdown_table(correct_gap_df),
                "",
            ]
        )

    if not high_freq_low_distinction_df.empty:
        lines.extend(
            [
                "## High-Frequency But Low-Distinction Concepts",
                "",
                "These concepts appear often but show limited label/correctness separation, so they should not be over-interpreted as class-defining on frequency alone.",
                "",
                markdown_table(high_freq_low_distinction_df),
                "",
            ]
        )

    if quality_report_text:
        extracted = []
        for line in quality_report_text.splitlines():
            if "warnings_found" in line or "prediction accuracy on exported slides" in line:
                extracted.append(line)
        if extracted:
            lines.extend(["## Quality Check Notes", ""])
            lines.extend(f"- {line}" for line in extracted)
            lines.append("")

    lines.extend(["## Warnings", ""])
    if warning_log:
        for warning in warning_log:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Step15 can focus on evidence visualization for selected concept-class edges, or a learnable concept-class graph prototype if you want to move beyond post-hoc analysis.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    evidence_dir = resolve_path(root, args.evidence_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warning_log: list[str] = []

    prediction_df = safe_read_csv(evidence_dir / CORE_INPUTS["prediction_csv"], warning_log)
    top_concepts_df = safe_read_csv(evidence_dir / CORE_INPUTS["top_concepts_csv"], warning_log)
    evidence_payload = safe_load_pickle(evidence_dir / CORE_INPUTS["evidence_pkl"], warning_log)
    quality_report_text = safe_read_text(evidence_dir / CORE_INPUTS["quality_report_md"], warning_log)

    merged_df = build_merged_evidence(prediction_df, top_concepts_df, warning_log)
    slide_count = int(merged_df["slide_id"].nunique()) if not merged_df.empty else 0
    edges_df = aggregate_edges(merged_df, slide_count, args.min_count)
    nodes_df = build_nodes(edges_df)
    pkl_summary, _ = summarize_pickle(evidence_payload, slide_count, warning_log)

    edges_csv = out_dir / "stage14_concept_class_edges.csv"
    nodes_csv = out_dir / "stage14_concept_class_nodes.csv"
    graph_json_path = out_dir / "stage14_concept_class_graph.json"
    report_md = out_dir / "stage14_concept_class_report.md"
    summary_csv = out_dir / "stage14_concept_class_summary.csv"

    edges_df.to_csv(edges_csv, index=False)
    nodes_df.to_csv(nodes_csv, index=False)
    graph_payload = build_graph_json(nodes_df, edges_df, pkl_summary)
    graph_json_path.write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_df = build_summary_df(merged_df, edges_df, pkl_summary, warning_log)
    summary_df.to_csv(summary_csv, index=False)
    report_text = build_report(
        evidence_dir=evidence_dir,
        merged_df=merged_df,
        edges_df=edges_df,
        pkl_summary=pkl_summary,
        quality_report_text=quality_report_text,
        top_k_edges=args.top_k_edges,
        warning_log=warning_log,
    )
    report_md.write_text(report_text, encoding="utf-8")

    print(f"Saved concept-class edges CSV to: {edges_csv}")
    print(f"Saved concept-class nodes CSV to: {nodes_csv}")
    print(f"Saved concept-class graph JSON to: {graph_json_path}")
    print(f"Saved concept-class report Markdown to: {report_md}")
    print(f"Saved concept-class summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
