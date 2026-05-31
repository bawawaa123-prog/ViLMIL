from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = Path("results_stage9/stage13_rce_evidence_export_fold0_test_full")
DEFAULT_GRAPH_DIR = Path("results_stage9/stage14_concept_class_graph_fold0")
DEFAULT_OUT_DIR = Path("results_stage9/stage15_rce_evidence_visualization_fold0")
CLASS_NAME_MAP = {0: "Adenocarcinoma", 1: "NonAdenocarcinoma"}
SCALE_ORDER = {"low": 0, "high": 1}
SELECTED_SLIDE_COLUMNS = [
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
EVIDENCE_PATH_COLUMNS = [
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
    "prompt_id",
    "region_id",
    "region_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Step15 RCE evidence paths from Stage13d and Stage14 outputs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to ViLa-MIL-main root.")
    parser.add_argument(
        "--evidence_dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing Stage13d evidence export files.",
    )
    parser.add_argument(
        "--graph_dir",
        type=Path,
        default=DEFAULT_GRAPH_DIR,
        help="Directory containing Stage14 concept-class graph files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for Step15 visualization outputs.",
    )
    parser.add_argument("--top_k_edges", type=int, default=8, help="Top concept-class edges to highlight per view.")
    parser.add_argument("--slides_per_class", type=int, default=3, help="Correct high-confidence slides selected per true label.")
    parser.add_argument("--include_incorrect", type=int, default=1, help="Incorrect slides selected per true label.")
    parser.add_argument("--max_heatmaps", type=int, default=12, help="Maximum slide-level heatmap PNGs to generate.")
    return parser.parse_args()


def resolve_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


def warn_message(message: str, warning_log: list[str]) -> None:
    text = f"[Step15 warning] {message}"
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


def safe_read_json(path: Path, warning_log: list[str]) -> dict | None:
    if not path.is_file():
        warn_message(f"Missing JSON: {path}", warning_log)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn_message(f"Failed to read JSON {path}: {exc}", warning_log)
        return None


def safe_load_pickle(path: Path, warning_log: list[str]):
    if not path.is_file():
        warn_message(f"Missing pickle: {path}", warning_log)
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        warn_message(f"Failed to load pickle {path}: {exc}", warning_log)
        return None


def safe_read_text(path: Path, warning_log: list[str]) -> str:
    if not path.is_file():
        warn_message(f"Missing text report: {path}", warning_log)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warn_message(f"Failed to read text {path}: {exc}", warning_log)
        return ""


def ensure_prediction_columns(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=SELECTED_SLIDE_COLUMNS)
    required = ["slide_id", "label", "pred", "correct", "prob_0", "prob_1"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        warn_message(f"Prediction CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=SELECTED_SLIDE_COLUMNS)

    result = df.copy()
    for col in ["label", "pred", "correct", "prob_0", "prob_1"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=["slide_id", "label", "pred", "prob_0", "prob_1"]).copy()
    if result.empty:
        return result
    result["label"] = result["label"].astype(int)
    result["pred"] = result["pred"].astype(int)
    result["correct"] = result["correct"].fillna((result["label"] == result["pred"]).astype(float)).astype(int)
    result["true_prob"] = np.where(result["label"] == 0, result["prob_0"], result["prob_1"])
    result["pred_prob"] = np.where(result["pred"] == 0, result["prob_0"], result["prob_1"])
    result["confidence_margin"] = (result["prob_0"] - result["prob_1"]).abs()
    result["class_name_true"] = result["label"].map(CLASS_NAME_MAP).fillna("Unknown")
    result["class_name_pred"] = result["pred"].map(CLASS_NAME_MAP).fillna("Unknown")
    return result


def ensure_concept_columns(df: pd.DataFrame | None, warning_log: list[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=EVIDENCE_PATH_COLUMNS)
    required = [
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
    missing = [col for col in required if col not in df.columns]
    if missing:
        warn_message(f"Top concepts CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=EVIDENCE_PATH_COLUMNS)

    result = df.copy()
    for col in ["class_id", "evidence_score", "prompt_weight", "rank", "prompt_id"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    result = result.dropna(subset=["slide_id", "class_id", "rank", "prompt_id"]).copy()
    if result.empty:
        return result
    result["class_id"] = result["class_id"].astype(int)
    result["rank"] = result["rank"].astype(int)
    result["prompt_id"] = result["prompt_id"].astype(int)
    return result


def build_edges_from_graph_json(graph_json: dict | None, warning_log: list[str]) -> pd.DataFrame:
    if not graph_json:
        return pd.DataFrame()
    rows = []
    for edge in graph_json.get("concept_to_class_edges", []):
        if not isinstance(edge, dict):
            continue
        rows.append(
            {
                "scale": edge.get("scale", ""),
                "class_id": edge.get("class_id", np.nan),
                "class_name": edge.get("class_name", ""),
                "concept_id": edge.get("concept_id", ""),
                "concept_text": edge.get("concept_text", ""),
                "edge_strength": edge.get("edge_strength", np.nan),
                "mean_evidence_score": edge.get("mean_evidence_score", np.nan),
                "slide_coverage": edge.get("slide_coverage", np.nan),
                "mean_rank": edge.get("mean_rank", np.nan),
                "n_topk": edge.get("n_topk", np.nan),
                "passes_min_count": True,
            }
        )
    if not rows:
        warn_message("Graph JSON did not contain concept_to_class_edges rows.", warning_log)
    return pd.DataFrame(rows)


def ensure_edge_columns(edges_df: pd.DataFrame | None, graph_json: dict | None, warning_log: list[str]) -> pd.DataFrame:
    if edges_df is None or edges_df.empty:
        edges_df = build_edges_from_graph_json(graph_json, warning_log)
    if edges_df is None or edges_df.empty:
        return pd.DataFrame(
            columns=[
                "scale",
                "class_id",
                "class_name",
                "concept_id",
                "concept_text",
                "edge_strength",
                "mean_evidence_score",
                "slide_coverage",
                "mean_rank",
                "n_topk",
                "passes_min_count",
            ]
        )

    required = ["scale", "class_id", "concept_id", "concept_text", "edge_strength"]
    missing = [col for col in required if col not in edges_df.columns]
    if missing:
        warn_message(f"Edges CSV missing required columns: {missing}", warning_log)
        return pd.DataFrame(columns=required)

    result = edges_df.copy()
    for col in ["class_id", "edge_strength", "mean_evidence_score", "slide_coverage", "mean_rank", "n_topk"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    result["scale"] = result["scale"].fillna("").astype(str)
    result["concept_id"] = result["concept_id"].fillna("").astype(str)
    result["concept_text"] = result["concept_text"].fillna("").astype(str)
    result["class_id"] = result["class_id"].fillna(-1).astype(int)
    if "class_name" not in result.columns:
        result["class_name"] = result["class_id"].map(CLASS_NAME_MAP).fillna("Unknown")
    else:
        result["class_name"] = result["class_name"].fillna(result["class_id"].map(CLASS_NAME_MAP)).fillna("Unknown")
    if "passes_min_count" not in result.columns:
        result["passes_min_count"] = True
    result["passes_min_count"] = result["passes_min_count"].fillna(True).astype(bool)
    result = result.sort_values(
        ["edge_strength", "mean_evidence_score", "slide_coverage", "n_topk"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return result


def truncate_text(text: str, limit: int = 42) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
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


def select_representative_slides(
    prediction_df: pd.DataFrame,
    slides_per_class: int,
    include_incorrect: int,
) -> pd.DataFrame:
    if prediction_df.empty:
        return pd.DataFrame(columns=SELECTED_SLIDE_COLUMNS)

    selected_frames = []
    used_slide_ids: set[str] = set()
    for label_value in sorted(prediction_df["label"].dropna().astype(int).unique()):
        label_df = prediction_df.loc[prediction_df["label"] == int(label_value)].copy()
        if label_df.empty:
            continue

        correct_df = label_df.loc[label_df["correct"] == 1].sort_values(
            ["true_prob", "confidence_margin", "pred_prob"],
            ascending=[False, False, False],
        )
        if slides_per_class > 0:
            correct_df = correct_df.loc[~correct_df["slide_id"].isin(used_slide_ids)].head(int(slides_per_class)).copy()
            if not correct_df.empty:
                correct_df["selection_reason"] = "correct_high_confidence"
                selected_frames.append(correct_df)
                used_slide_ids.update(correct_df["slide_id"].astype(str).tolist())

        incorrect_df = label_df.loc[label_df["correct"] == 0].sort_values(
            ["pred_prob", "confidence_margin", "true_prob"],
            ascending=[False, False, False],
        )
        if include_incorrect > 0:
            incorrect_df = incorrect_df.loc[~incorrect_df["slide_id"].isin(used_slide_ids)].head(int(include_incorrect)).copy()
            if not incorrect_df.empty:
                incorrect_df["selection_reason"] = "incorrect_high_confidence"
                selected_frames.append(incorrect_df)
                used_slide_ids.update(incorrect_df["slide_id"].astype(str).tolist())

    if not selected_frames:
        return pd.DataFrame(columns=SELECTED_SLIDE_COLUMNS)

    selected_df = pd.concat(selected_frames, ignore_index=True)
    selected_df = selected_df.sort_values(
        ["label", "correct", "true_prob", "pred_prob"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return selected_df[SELECTED_SLIDE_COLUMNS]


def build_prompt_reference(concepts_df: pd.DataFrame) -> pd.DataFrame:
    if concepts_df.empty:
        return pd.DataFrame(columns=["scale", "class_id", "prompt_id", "concept_id", "concept_text"])
    grouped = (
        concepts_df.groupby(["scale", "class_id", "prompt_id", "concept_id", "concept_text"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["scale", "class_id", "prompt_id", "count"], ascending=[True, True, True, False])
    )
    reference = grouped.drop_duplicates(subset=["scale", "class_id", "prompt_id"]).copy()
    return reference[["scale", "class_id", "prompt_id", "concept_id", "concept_text"]].reset_index(drop=True)


def build_slide_payload_map(payload, warning_log: list[str]) -> dict[str, dict]:
    if payload is None:
        return {}
    if not isinstance(payload, list):
        warn_message(f"Evidence pickle root object is not a list: {type(payload).__name__}", warning_log)
        return {}
    slide_map = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        slide_id = item.get("slide_id")
        if slide_id is None:
            continue
        slide_map[str(slide_id)] = item
    return slide_map


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        try:
            return np.asarray(value.detach().cpu().numpy())
        except Exception:
            return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def extract_region_peak(slide_item: dict | None, scale: str, class_id: int, prompt_id: int) -> tuple[float, float]:
    if slide_item is None:
        return np.nan, np.nan
    tensor_key = "low_region_concept_sim" if scale == "low" else "high_region_concept_sim"
    array = to_numpy(slide_item.get(tensor_key))
    if array is None or array.ndim != 3:
        return np.nan, np.nan
    if class_id < 0 or class_id >= array.shape[0] or prompt_id < 0 or prompt_id >= array.shape[2]:
        return np.nan, np.nan
    region_scores = array[class_id, :, prompt_id]
    if region_scores.size == 0:
        return np.nan, np.nan
    region_idx = int(np.argmax(region_scores))
    region_score = float(region_scores[region_idx])
    return float(region_idx), region_score


def build_top_edge_sets(edges_df: pd.DataFrame, top_k_edges: int) -> tuple[set[tuple[str, int, str]], pd.DataFrame]:
    selected_keys: set[tuple[str, int, str]] = set()
    selected_frames = []
    filtered = edges_df.loc[edges_df["passes_min_count"]].copy() if "passes_min_count" in edges_df.columns else edges_df.copy()

    overall_df = filtered.head(int(top_k_edges)).copy()
    if not overall_df.empty:
        selected_frames.append(overall_df)

    for class_id in sorted(filtered["class_id"].dropna().astype(int).unique()):
        class_df = filtered.loc[filtered["class_id"] == int(class_id)].head(int(top_k_edges)).copy()
        if not class_df.empty:
            selected_frames.append(class_df)

    for scale_name in ["low", "high"]:
        scale_df = filtered.loc[filtered["scale"] == scale_name].head(int(top_k_edges)).copy()
        if not scale_df.empty:
            selected_frames.append(scale_df)

    if selected_frames:
        selected_edge_df = pd.concat(selected_frames, ignore_index=True).drop_duplicates(
            subset=["scale", "class_id", "concept_id"]
        )
    else:
        selected_edge_df = pd.DataFrame(columns=edges_df.columns)

    for _, row in selected_edge_df.iterrows():
        selected_keys.add((str(row["scale"]), int(row["class_id"]), str(row["concept_id"])))
    return selected_keys, selected_edge_df


def build_slide_evidence_paths(
    selected_slides_df: pd.DataFrame,
    concepts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    slide_payload_map: dict[str, dict],
    top_k_edges: int,
) -> pd.DataFrame:
    if selected_slides_df.empty or concepts_df.empty:
        return pd.DataFrame(columns=EVIDENCE_PATH_COLUMNS)

    edge_lookup_df = edges_df[["scale", "class_id", "concept_id", "edge_strength"]].drop_duplicates().copy()
    selected_edge_keys, selected_edge_df = build_top_edge_sets(edges_df, top_k_edges)

    selected_paths = concepts_df.merge(
        selected_slides_df[["slide_id", "label", "pred", "correct"]],
        on="slide_id",
        how="inner",
        validate="many_to_one",
    )
    if selected_paths.empty:
        return pd.DataFrame(columns=EVIDENCE_PATH_COLUMNS)

    selected_paths = selected_paths.loc[selected_paths["class_id"] == selected_paths["pred"]].copy()
    if selected_paths.empty:
        return pd.DataFrame(columns=EVIDENCE_PATH_COLUMNS)

    selected_paths = selected_paths.merge(
        edge_lookup_df,
        on=["scale", "class_id", "concept_id"],
        how="left",
    )
    selected_paths["edge_strength"] = pd.to_numeric(selected_paths["edge_strength"], errors="coerce")
    selected_paths["is_selected_edge"] = selected_paths.apply(
        lambda row: (str(row["scale"]), int(row["class_id"]), str(row["concept_id"])) in selected_edge_keys,
        axis=1,
    )

    filtered_frames = []
    for slide_id, group in selected_paths.groupby("slide_id", dropna=False):
        selected_group = group.loc[group["is_selected_edge"]].copy()
        if selected_group.empty:
            selected_group = group.sort_values(
                ["scale", "rank", "evidence_score", "prompt_weight"],
                ascending=[True, True, False, False],
            ).head(6).copy()
        filtered_frames.append(selected_group)
    selected_paths = pd.concat(filtered_frames, ignore_index=True) if filtered_frames else pd.DataFrame(columns=selected_paths.columns)

    region_ids = []
    region_scores = []
    for _, row in selected_paths.iterrows():
        slide_item = slide_payload_map.get(str(row["slide_id"]))
        region_id, region_score = extract_region_peak(
            slide_item=slide_item,
            scale=str(row["scale"]),
            class_id=int(row["pred"]),
            prompt_id=int(row["prompt_id"]),
        )
        region_ids.append(region_id)
        region_scores.append(region_score)
    selected_paths["region_id"] = region_ids
    selected_paths["region_score"] = region_scores

    selected_paths = selected_paths.sort_values(
        ["slide_id", "scale", "rank", "edge_strength", "evidence_score"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)
    for column in EVIDENCE_PATH_COLUMNS:
        if column not in selected_paths.columns:
            selected_paths[column] = np.nan
    return selected_paths[EVIDENCE_PATH_COLUMNS]


def build_figure_label(row: pd.Series, include_scale: bool, include_class: bool) -> str:
    parts = []
    if include_scale:
        parts.append("L" if str(row.get("scale", "")).lower() == "low" else "H")
    if include_class:
        parts.append("Adeno" if int(row.get("class_id", -1)) == 0 else "NonAdeno")
    concept_id = str(row.get("concept_id", "")).strip()
    if concept_id:
        parts.append(concept_id)
    else:
        parts.append(truncate_text(row.get("concept_text", ""), 28))
    return " | ".join(parts)


def plot_top_edges_by_class(edges_df: pd.DataFrame, out_path: Path, top_k_edges: int, warning_log: list[str]) -> bool:
    filtered = edges_df.loc[edges_df["passes_min_count"]].copy() if "passes_min_count" in edges_df.columns else edges_df.copy()
    if filtered.empty:
        warn_message("No concept-class edges available for class-level figure.", warning_log)
        return False

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    colors = {0: "#4C72B0", 1: "#DD8452"}
    for index, class_id in enumerate([0, 1]):
        axis = axes[index]
        class_df = filtered.loc[filtered["class_id"] == class_id].head(int(top_k_edges)).copy()
        if class_df.empty:
            axis.set_title(f"{CLASS_NAME_MAP.get(class_id, 'Unknown')}\nNo edges")
            axis.axis("off")
            continue
        class_df = class_df.iloc[::-1].copy()
        labels = [build_figure_label(row, include_scale=True, include_class=False) for _, row in class_df.iterrows()]
        axis.barh(np.arange(len(class_df)), class_df["edge_strength"], color=colors.get(class_id, "#888888"))
        axis.set_yticks(np.arange(len(class_df)))
        axis.set_yticklabels([truncate_text(label, 40) for label in labels], fontsize=9)
        axis.set_xlabel("Edge strength")
        axis.set_title(CLASS_NAME_MAP.get(class_id, str(class_id)))
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Top concept-class edges by class", fontsize=14)
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_top_edges_by_scale(edges_df: pd.DataFrame, out_path: Path, top_k_edges: int, warning_log: list[str]) -> bool:
    filtered = edges_df.loc[edges_df["passes_min_count"]].copy() if "passes_min_count" in edges_df.columns else edges_df.copy()
    if filtered.empty:
        warn_message("No concept-class edges available for scale-level figure.", warning_log)
        return False

    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    colors = {"low": "#55A868", "high": "#C44E52"}
    for index, scale_name in enumerate(["low", "high"]):
        axis = axes[index]
        scale_df = filtered.loc[filtered["scale"] == scale_name].head(int(top_k_edges)).copy()
        if scale_df.empty:
            axis.set_title(f"{scale_name.capitalize()} scale\nNo edges")
            axis.axis("off")
            continue
        scale_df = scale_df.iloc[::-1].copy()
        labels = [build_figure_label(row, include_scale=False, include_class=True) for _, row in scale_df.iterrows()]
        axis.barh(np.arange(len(scale_df)), scale_df["edge_strength"], color=colors.get(scale_name, "#888888"))
        axis.set_yticks(np.arange(len(scale_df)))
        axis.set_yticklabels([truncate_text(label, 40) for label in labels], fontsize=9)
        axis.set_xlabel("Edge strength")
        axis.set_title(f"{scale_name.capitalize()} scale")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Top concept-class edges by scale", fontsize=14)
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_selected_slide_probabilities(selected_slides_df: pd.DataFrame, out_path: Path, warning_log: list[str]) -> bool:
    if selected_slides_df.empty:
        warn_message("No selected slides available for probability overview figure.", warning_log)
        return False

    plot_df = selected_slides_df.copy().reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.36
    figure_width = max(10.0, len(plot_df) * 1.1)
    figure, axis = plt.subplots(figsize=(figure_width, 6), constrained_layout=True)
    axis.bar(x - width / 2.0, plot_df["true_prob"], width=width, color="#4C72B0", label="True-class probability")
    axis.bar(x + width / 2.0, plot_df["pred_prob"], width=width, color="#DD8452", label="Predicted-class probability")

    tick_labels = []
    for _, row in plot_df.iterrows():
        status = "OK" if int(row["correct"]) == 1 else "ERR"
        tick_labels.append(f"{row['slide_id']}\nT{int(row['label'])}/P{int(row['pred'])} {status}")
    axis.set_xticks(x)
    axis.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=9)
    axis.set_ylabel("Probability")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    axis.set_title("Selected slide probability overview")
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def build_prompt_lookup(prompt_reference_df: pd.DataFrame) -> dict[tuple[str, int, int], tuple[str, str]]:
    lookup: dict[tuple[str, int, int], tuple[str, str]] = {}
    for _, row in prompt_reference_df.iterrows():
        lookup[(str(row["scale"]), int(row["class_id"]), int(row["prompt_id"]))] = (
            str(row["concept_id"]),
            str(row["concept_text"]),
        )
    return lookup


def plot_slide_heatmap(
    slide_row: pd.Series,
    slide_item: dict | None,
    prompt_lookup: dict[tuple[str, int, int], tuple[str, str]],
    out_path: Path,
    warning_log: list[str],
) -> bool:
    if slide_item is None:
        warn_message(f"Missing pickle entry for slide heatmap: {slide_row['slide_id']}", warning_log)
        return False

    pred_class = int(slide_row["pred"])
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    plotted = False
    for axis, scale_name, tensor_key in zip(
        axes,
        ["low", "high"],
        ["low_region_concept_sim", "high_region_concept_sim"],
    ):
        array = to_numpy(slide_item.get(tensor_key))
        if array is None or array.ndim != 3:
            axis.set_title(f"{scale_name.capitalize()} scale\nMissing heatmap")
            axis.axis("off")
            continue
        if pred_class < 0 or pred_class >= array.shape[0]:
            axis.set_title(f"{scale_name.capitalize()} scale\nInvalid class index")
            axis.axis("off")
            continue
        heatmap = np.asarray(array[pred_class], dtype=float)
        image = axis.imshow(heatmap, aspect="auto", cmap="viridis")
        concept_labels = []
        for prompt_id in range(heatmap.shape[1]):
            concept_id, concept_text = prompt_lookup.get(
                (scale_name, pred_class, int(prompt_id)),
                (f"c{prompt_id}", ""),
            )
            label = concept_id if concept_id else concept_text
            concept_labels.append(truncate_text(label, 16))
        axis.set_xticks(np.arange(heatmap.shape[1]))
        axis.set_xticklabels(concept_labels, rotation=50, ha="right", fontsize=8)
        axis.set_yticks(np.arange(heatmap.shape[0]))
        axis.set_yticklabels([f"R{i}" for i in range(heatmap.shape[0])], fontsize=8)
        axis.set_xlabel("Concept")
        axis.set_ylabel("Region")
        axis.set_title(f"{scale_name.capitalize()} scale | pred class {pred_class}")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        plotted = True

    if not plotted:
        plt.close(figure)
        warn_message(f"No valid heatmap arrays found for slide: {slide_row['slide_id']}", warning_log)
        return False

    figure.suptitle(
        f"{slide_row['slide_id']} | label={int(slide_row['label'])} pred={pred_class} correct={int(slide_row['correct'])}",
        fontsize=13,
    )
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def build_summary_df(
    prediction_df: pd.DataFrame,
    concepts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    selected_slides_df: pd.DataFrame,
    evidence_paths_df: pd.DataFrame,
    graph_json: dict | None,
    generated_files: list[Path],
    warning_log: list[str],
) -> pd.DataFrame:
    rows = [
        {"section": "inputs", "metric": "prediction_slides", "value": int(prediction_df["slide_id"].nunique()) if not prediction_df.empty else 0, "note": ""},
        {"section": "inputs", "metric": "top_concept_rows", "value": int(len(concepts_df)), "note": ""},
        {"section": "inputs", "metric": "graph_edge_rows", "value": int(len(edges_df)), "note": ""},
        {"section": "selection", "metric": "selected_slides", "value": int(len(selected_slides_df)), "note": ""},
        {"section": "selection", "metric": "selected_correct", "value": int((selected_slides_df["correct"] == 1).sum()) if not selected_slides_df.empty else 0, "note": ""},
        {"section": "selection", "metric": "selected_incorrect", "value": int((selected_slides_df["correct"] == 0).sum()) if not selected_slides_df.empty else 0, "note": ""},
        {"section": "paths", "metric": "slide_evidence_paths", "value": int(len(evidence_paths_df)), "note": ""},
        {"section": "outputs", "metric": "generated_file_count", "value": int(len(generated_files)), "note": ""},
        {"section": "quality", "metric": "warning_count", "value": int(len(warning_log)), "note": ""},
    ]
    if graph_json:
        rows.append(
            {
                "section": "inputs",
                "metric": "graph_json_edge_count",
                "value": int(len(graph_json.get("concept_to_class_edges", []))),
                "note": "",
            }
        )
        rows.append(
            {
                "section": "inputs",
                "metric": "graph_json_concept_node_count",
                "value": int(len(graph_json.get("concept_nodes", []))),
                "note": "",
            }
        )
    for path in generated_files:
        rows.append(
            {
                "section": "outputs",
                "metric": path.name,
                "value": 1,
                "note": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    root: Path,
    evidence_dir: Path,
    graph_dir: Path,
    out_dir: Path,
    selected_slides_df: pd.DataFrame,
    evidence_paths_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    graph_json: dict | None,
    generated_files: list[Path],
    warning_log: list[str],
    quality_report_text: str,
    top_k_edges: int,
) -> str:
    lines = [
        "# Stage15 RCE Evidence Visualization",
        "",
        "Step15 is a visualization and interpretation layer built on top of existing Step13d evidence export and Step14 concept-class graph outputs.",
        "",
        "- It does not modify the model.",
        "- It does not run training.",
        "- It does not represent pathology-doctor validation or clinical decision guidance.",
        "",
        "## Inputs",
        "",
        f"- root: `{root}`",
        f"- evidence_dir: `{evidence_dir}`",
        f"- graph_dir: `{graph_dir}`",
        f"- out_dir: `{out_dir}`",
        f"- selected_slide_count: `{len(selected_slides_df)}`",
        f"- evidence_path_rows: `{len(evidence_paths_df)}`",
        f"- graph_edge_rows: `{len(edges_df)}`",
        "",
        "## Selection Logic",
        "",
        "- For each true label, select high-confidence correct slides first.",
        "- If available, also include a small number of high-confidence incorrect slides for failure-case interpretation.",
        "- Slide-level heatmaps visualize predicted-class region-concept similarity from Stage13d pickle arrays with expected shape `(2, 16, 12)`.",
        "",
    ]

    if not selected_slides_df.empty:
        display_df = selected_slides_df.copy()
        for column in ["true_prob", "pred_prob", "confidence_margin"]:
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.4f}")
        lines.extend(["## Selected Slides", "", markdown_table(display_df), ""])

    filtered_edges = edges_df.loc[edges_df["passes_min_count"]].copy() if "passes_min_count" in edges_df.columns else edges_df.copy()
    if not filtered_edges.empty:
        overall_top = filtered_edges.head(int(top_k_edges)).copy()
        overall_top = overall_top[
            ["scale", "class_name", "concept_id", "concept_text", "edge_strength", "mean_evidence_score", "slide_coverage", "mean_rank", "n_topk"]
        ].copy()
        for column in ["edge_strength", "mean_evidence_score", "slide_coverage", "mean_rank"]:
            if column in overall_top.columns:
                overall_top[column] = overall_top[column].map(lambda value: f"{float(value):.4f}")
        lines.extend(["## Overall Top Concept-Class Edges", "", markdown_table(overall_top), ""])

    if not evidence_paths_df.empty:
        preview_df = evidence_paths_df.copy()
        for column in ["evidence_score", "prompt_weight", "edge_strength", "region_score"]:
            preview_df[column] = preview_df[column].map(
                lambda value: "NA" if pd.isna(value) else f"{float(value):.4f}"
            )
        lines.extend(
            [
                "## Evidence Path Preview",
                "",
                "The CSV keeps the full `slide -> region -> concept -> class` path details. A short preview is shown below.",
                "",
                markdown_table(preview_df.head(24)),
                "",
            ]
        )

    if graph_json:
        lines.extend(
            [
                "## Graph JSON Snapshot",
                "",
                f"- class_nodes: `{len(graph_json.get('class_nodes', []))}`",
                f"- concept_nodes: `{len(graph_json.get('concept_nodes', []))}`",
                f"- concept_to_class_edges: `{len(graph_json.get('concept_to_class_edges', []))}`",
                "",
            ]
        )

    if quality_report_text:
        lines.extend(
            [
                "## Step13 Quality Note",
                "",
                "Stage13d quality report was available and used as upstream context for visualization readiness.",
                "",
            ]
        )

    lines.extend(["## Generated Outputs", ""])
    if generated_files:
        for path in generated_files:
            lines.append(f"- `{path.relative_to(out_dir.parent.parent if out_dir.parent.parent.exists() else out_dir.parent)}`")
    else:
        lines.append("- No output artifacts were generated beyond this report.")
    lines.append("")

    lines.extend(["## Warnings", ""])
    if warning_log:
        for message in warning_log:
            lines.append(f"- {message}")
    else:
        lines.append("- None")
    lines.append("")

    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "These figures and tables are post-hoc evidence summaries intended for qualitative interpretation only. They should be treated as model-behavior visualization rather than validated biomedical evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def run_visualization(args: argparse.Namespace) -> dict[str, object]:
    warning_log: list[str] = []
    root = resolve_path(DEFAULT_ROOT, args.root)
    evidence_dir = resolve_path(root, args.evidence_dir)
    graph_dir = resolve_path(root, args.graph_dir)
    out_dir = resolve_path(root, args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = evidence_dir / "slide_prediction_evidence.csv"
    concepts_path = evidence_dir / "slide_top_concepts.csv"
    evidence_pkl_path = evidence_dir / "region_concept_evidence.pkl"
    quality_report_path = evidence_dir / "stage13_rce_evidence_quality_report.md"
    edges_path = graph_dir / "stage14_concept_class_edges.csv"
    graph_json_path = graph_dir / "stage14_concept_class_graph.json"

    prediction_df = ensure_prediction_columns(safe_read_csv(prediction_path, warning_log), warning_log)
    concepts_df = ensure_concept_columns(safe_read_csv(concepts_path, warning_log), warning_log)
    graph_json = safe_read_json(graph_json_path, warning_log)
    edges_df = ensure_edge_columns(safe_read_csv(edges_path, warning_log), graph_json, warning_log)
    payload = safe_load_pickle(evidence_pkl_path, warning_log)
    quality_report_text = safe_read_text(quality_report_path, warning_log)
    slide_payload_map = build_slide_payload_map(payload, warning_log)
    prompt_reference_df = build_prompt_reference(concepts_df)
    prompt_lookup = build_prompt_lookup(prompt_reference_df)

    selected_slides_df = select_representative_slides(
        prediction_df=prediction_df,
        slides_per_class=max(int(args.slides_per_class), 0),
        include_incorrect=max(int(args.include_incorrect), 0),
    )
    evidence_paths_df = build_slide_evidence_paths(
        selected_slides_df=selected_slides_df,
        concepts_df=concepts_df,
        edges_df=edges_df,
        slide_payload_map=slide_payload_map,
        top_k_edges=max(int(args.top_k_edges), 1),
    )

    selected_slides_csv = out_dir / "stage15_selected_slides.csv"
    evidence_paths_csv = out_dir / "stage15_slide_evidence_paths.csv"
    summary_csv = out_dir / "stage15_visualization_summary.csv"
    report_md = out_dir / "stage15_rce_evidence_visualization_report.md"

    selected_slides_df.to_csv(selected_slides_csv, index=False)
    evidence_paths_df.to_csv(evidence_paths_csv, index=False)

    generated_files: list[Path] = [selected_slides_csv, evidence_paths_csv]

    class_figure = figures_dir / "top_concept_class_edges_by_class.png"
    if plot_top_edges_by_class(edges_df, class_figure, max(int(args.top_k_edges), 1), warning_log):
        generated_files.append(class_figure)

    scale_figure = figures_dir / "top_concept_class_edges_by_scale.png"
    if plot_top_edges_by_scale(edges_df, scale_figure, max(int(args.top_k_edges), 1), warning_log):
        generated_files.append(scale_figure)

    probability_figure = figures_dir / "selected_slide_probability_overview.png"
    if plot_selected_slide_probabilities(selected_slides_df, probability_figure, warning_log):
        generated_files.append(probability_figure)

    heatmap_count = 0
    if not selected_slides_df.empty and int(args.max_heatmaps) > 0:
        for _, slide_row in selected_slides_df.head(int(args.max_heatmaps)).iterrows():
            safe_slide_id = str(slide_row["slide_id"]).replace("/", "_")
            heatmap_path = figures_dir / f"slide_{safe_slide_id}_predclass_region_concept_heatmap.png"
            if plot_slide_heatmap(
                slide_row=slide_row,
                slide_item=slide_payload_map.get(str(slide_row["slide_id"])),
                prompt_lookup=prompt_lookup,
                out_path=heatmap_path,
                warning_log=warning_log,
            ):
                generated_files.append(heatmap_path)
                heatmap_count += 1

    summary_df = build_summary_df(
        prediction_df=prediction_df,
        concepts_df=concepts_df,
        edges_df=edges_df,
        selected_slides_df=selected_slides_df,
        evidence_paths_df=evidence_paths_df,
        graph_json=graph_json,
        generated_files=generated_files,
        warning_log=warning_log,
    )
    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame(
                [
                    {
                        "section": "outputs",
                        "metric": "heatmap_png_count",
                        "value": int(heatmap_count),
                        "note": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    summary_df.to_csv(summary_csv, index=False)
    generated_files.append(summary_csv)

    report_text = build_report(
        root=root,
        evidence_dir=evidence_dir,
        graph_dir=graph_dir,
        out_dir=out_dir,
        selected_slides_df=selected_slides_df,
        evidence_paths_df=evidence_paths_df,
        edges_df=edges_df,
        graph_json=graph_json,
        generated_files=generated_files,
        warning_log=warning_log,
        quality_report_text=quality_report_text,
        top_k_edges=max(int(args.top_k_edges), 1),
    )
    report_md.write_text(report_text, encoding="utf-8")
    generated_files.append(report_md)

    print(f"[Step15] Selected slides: {len(selected_slides_df)}")
    print(f"[Step15] Evidence paths: {len(evidence_paths_df)}")
    print(f"[Step15] Heatmaps: {heatmap_count}")
    print(f"[Step15] Warnings: {len(warning_log)}")
    print(f"[Step15] Output directory: {out_dir}")

    return {
        "root": root,
        "out_dir": out_dir,
        "selected_slides_df": selected_slides_df,
        "evidence_paths_df": evidence_paths_df,
        "summary_df": summary_df,
        "warning_log": warning_log,
        "generated_files": generated_files,
    }


def main() -> None:
    args = parse_args()
    try:
        run_visualization(args)
    except Exception as exc:
        root = resolve_path(DEFAULT_ROOT, args.root)
        out_dir = resolve_path(root, args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_md = out_dir / "stage15_rce_evidence_visualization_report.md"
        message = [
            "# Stage15 RCE Evidence Visualization",
            "",
            "The visualization script exited early because an unexpected error occurred.",
            "",
            f"- error: `{exc}`",
            "- This step is intended to stay post-hoc only: no model change, no training, no feature extraction.",
            "",
        ]
        report_md.write_text("\n".join(message), encoding="utf-8")
        print(f"[Step15 warning] Unexpected failure: {exc}")
        print(f"[Step15] Wrote fallback report to: {report_md}")


if __name__ == "__main__":
    main()
