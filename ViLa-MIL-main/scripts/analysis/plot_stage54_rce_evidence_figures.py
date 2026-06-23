from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "stage54_rce_evidence_interpretability_package"
FIGURE_DIR = DOCS_DIR / "figures"
FULL_EVIDENCE_DIR = ROOT / "results_stage32" / "stage32_rce_v4_csg_evidence_export"
WO_CSG_EVIDENCE_DIR = ROOT / "results_stage54_rce_evidence_interpretability" / "wo_csg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Step54 RCE evidence figures.")
    parser.add_argument("--docs_dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--figure_dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--full_evidence_dir", type=Path, default=FULL_EVIDENCE_DIR)
    parser.add_argument("--wo_csg_evidence_dir", type=Path, default=WO_CSG_EVIDENCE_DIR)
    parser.add_argument("--case_selection_csv", type=Path, default=DOCS_DIR / "stage54_case_selection.csv")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_literal_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def parse_concept_summary(text: object) -> list[tuple[str, float]]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    parts = []
    for chunk in str(text).split("|"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, value = chunk.rsplit(":", 1)
        numeric = safe_literal_float(value.strip())
        if numeric is None:
            continue
        parts.append((name.strip(), numeric))
    return parts


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.9), 3.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_barh(labels: list[str], values: list[float], title: str, output_path: Path, color: str = "#C44E52") -> None:
    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.45)))
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=color)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_component_bar(labels: list[str], values: list[float], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors = ["#4C72B0" if value >= 0 else "#DD8452" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.figure_dir)

    case_selection = read_csv_optional(args.case_selection_csv)
    full_summary = read_csv_optional(args.full_evidence_dir / "stage32_slide_evidence_summary.csv")
    full_long = read_csv_optional(args.full_evidence_dir / "stage32_top_concepts_long.csv")
    full_csg = read_csv_optional(args.full_evidence_dir / "stage32_top_csg_pairs.csv")
    wo_summary = read_csv_optional(args.wo_csg_evidence_dir / "stage32_slide_evidence_summary.csv")
    wo_long = read_csv_optional(args.wo_csg_evidence_dir / "stage32_top_concepts_long.csv")

    index_rows: list[dict[str, object]] = []

    def add_index(name: str, status: str, files: list[Path], inputs: list[Path], note: str, placement: str) -> None:
        index_rows.append(
            {
                "figure": name,
                "status": status,
                "output_files": ", ".join(rel(path) for path in files) if files else "missing",
                "input_files": ", ".join(rel(path) for path in inputs if path is not None),
                "placement": placement,
                "note": note,
            }
        )

    # 1. region-concept evidence heatmap
    if case_selection is not None and full_long is not None and not case_selection.empty:
        candidates = case_selection[case_selection["selection_type"] == "full_correct_high_confidence"]
        if not candidates.empty:
            slide_id = str(candidates.iloc[0]["slide_id"])
            subset = full_long[(full_long["slide_id"].astype(str) == slide_id) & (full_long["class_type"] == "pred")]
            low = subset[subset["scale"] == "low"].sort_values("concept_rank").head(5)
            high = subset[subset["scale"] == "high"].sort_values("concept_rank").head(5)
            concept_labels = [*low["concept_id"].fillna(low["concept_text"]).astype(str).tolist(), *high["concept_id"].fillna(high["concept_text"]).astype(str).tolist()]
            values = [
                low["contribution"].astype(float).tolist() + [0.0] * len(high.index),
                [0.0] * len(low.index) + high["contribution"].astype(float).tolist(),
            ]
            if concept_labels:
                matrix = np.asarray(values)
                output_path = args.figure_dir / "stage54_region_concept_heatmap.png"
                save_heatmap(matrix, ["low-scale", "high-scale"], concept_labels, f"Region-Concept Heatmap: {slide_id}", output_path)
                add_index(
                    "region-concept evidence heatmap",
                    "ready",
                    [output_path],
                    [args.case_selection_csv, args.full_evidence_dir / "stage32_top_concepts_long.csv"],
                    "Built from the existing full-equivalent fold0/test concept table.",
                    "main paper",
                )
            else:
                add_index(
                    "region-concept evidence heatmap",
                    "missing",
                    [],
                    [args.case_selection_csv, args.full_evidence_dir / "stage32_top_concepts_long.csv"],
                    "No concept rows were found for the selected correct case.",
                    "main paper",
                )
        else:
            add_index(
                "region-concept evidence heatmap",
                "missing",
                [],
                [args.case_selection_csv],
                "No selected full-correct case was available.",
                "main paper",
            )
    else:
        add_index(
            "region-concept evidence heatmap",
            "missing",
            [],
            [args.case_selection_csv, args.full_evidence_dir / "stage32_top_concepts_long.csv"],
            "Required case selection or concept evidence table is missing.",
            "main paper",
        )

    # 2. low-high concept evidence comparison
    if case_selection is not None and full_summary is not None and not case_selection.empty:
        row = case_selection[case_selection["selection_type"] == "full_correct_high_confidence"]
        if not row.empty:
            chosen = row.iloc[0]
            low_items = parse_concept_summary(chosen.get("full_evidence_top_low_concepts_for_pred"))
            high_items = parse_concept_summary(chosen.get("full_evidence_top_high_concepts_for_pred"))
            labels = [f"L:{name}" for name, _ in low_items[:5]] + [f"H:{name}" for name, _ in high_items[:5]]
            values = [value for _, value in low_items[:5]] + [value for _, value in high_items[:5]]
            if labels:
                output_path = args.figure_dir / "stage54_low_high_concept_comparison.png"
                save_barh(labels, values, f"Low vs High Concept Evidence: {chosen['slide_id']}", output_path, color="#4C72B0")
                add_index(
                    "low-high concept evidence comparison",
                    "ready",
                    [output_path],
                    [args.case_selection_csv, args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
                    "Built from selected-case concept summaries.",
                    "main paper or supplementary",
                )
            else:
                add_index(
                    "low-high concept evidence comparison",
                    "missing",
                    [],
                    [args.case_selection_csv],
                    "Selected case did not contain parsed concept summaries.",
                    "main paper or supplementary",
                )
        else:
            add_index(
                "low-high concept evidence comparison",
                "missing",
                [],
                [args.case_selection_csv],
                "No selected correct case was available.",
                "main paper or supplementary",
            )
    else:
        add_index(
            "low-high concept evidence comparison",
            "missing",
            [],
            [args.case_selection_csv, args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
            "Required selected case or full evidence summary is missing.",
            "main paper or supplementary",
        )

    # 3. full vs wo_csg concept ranking delta
    if case_selection is not None and full_long is not None and wo_long is not None and wo_summary is not None:
        candidates = case_selection[case_selection["selection_type"] == "same_pred_confidence_shift"]
        if not candidates.empty:
            chosen = candidates.iloc[0]
            slide_id = str(chosen["slide_id"])
            full_case = full_long[(full_long["slide_id"].astype(str) == slide_id) & (full_long["class_type"] == "pred")].copy()
            wo_case = wo_long[(wo_long["slide_id"].astype(str) == slide_id) & (wo_long["class_type"] == "pred")].copy()
            if not full_case.empty and not wo_case.empty:
                full_case["concept_name"] = full_case["concept_id"].fillna(full_case["concept_text"]).astype(str)
                wo_case["concept_name"] = wo_case["concept_id"].fillna(wo_case["concept_text"]).astype(str)
                full_case = full_case.groupby("concept_name", as_index=False)["contribution"].mean()
                wo_case = wo_case.groupby("concept_name", as_index=False)["contribution"].mean()
                merged = full_case.merge(wo_case, on="concept_name", how="outer", suffixes=("_full", "_wo_csg")).fillna(0.0)
                merged["delta"] = merged["contribution_full"] - merged["contribution_wo_csg"]
                merged = merged.sort_values("delta", ascending=False).head(12)
                output_path = args.figure_dir / "stage54_full_vs_wo_csg_concept_ranking_delta.png"
                save_barh(
                    merged["concept_name"].tolist(),
                    merged["delta"].astype(float).tolist(),
                    f"Full vs w/o CSG Concept Delta: {slide_id}",
                    output_path,
                    color="#55A868",
                )
                add_index(
                    "full vs wo_csg concept ranking delta",
                    "ready",
                    [output_path],
                    [args.full_evidence_dir / "stage32_top_concepts_long.csv", args.wo_csg_evidence_dir / "stage32_top_concepts_long.csv"],
                    "Requires both full and w/o CSG concept exports.",
                    "supplementary",
                )
            else:
                add_index(
                    "full vs wo_csg concept ranking delta",
                    "missing",
                    [],
                    [args.full_evidence_dir / "stage32_top_concepts_long.csv", args.wo_csg_evidence_dir / "stage32_top_concepts_long.csv"],
                    "The selected matched case was not found in both concept tables.",
                    "supplementary",
                )
        else:
            add_index(
                "full vs wo_csg concept ranking delta",
                "missing",
                [],
                [args.case_selection_csv],
                "No same-prediction confidence-shift case was available.",
                "supplementary",
            )
    else:
        add_index(
            "full vs wo_csg concept ranking delta",
            "missing",
            [],
            [args.full_evidence_dir / "stage32_top_concepts_long.csv", args.wo_csg_evidence_dir / "stage32_top_concepts_long.csv"],
            "w/o CSG evidence export is missing, so matched evidence-ranking figures cannot be generated yet.",
            "supplementary",
        )

    # 4. correct case evidence figure
    if case_selection is not None and full_summary is not None:
        chosen_df = case_selection[case_selection["selection_type"] == "full_correct_high_confidence"]
        if not chosen_df.empty:
            chosen = chosen_df.iloc[0]
            full_case = full_summary[full_summary["slide_id"].astype(str) == str(chosen["slide_id"])]
            if not full_case.empty:
                row = full_case.iloc[0]
                labels = ["low_c0", "low_c1", "high_c0", "high_c1", "visual_c0", "visual_c1", "csg_c0", "csg_c1"]
                values = [
                    safe_literal_float(row.get("low_logit_class_0")) or 0.0,
                    safe_literal_float(row.get("low_logit_class_1")) or 0.0,
                    safe_literal_float(row.get("high_logit_class_0")) or 0.0,
                    safe_literal_float(row.get("high_logit_class_1")) or 0.0,
                    safe_literal_float(row.get("visual_logit_class_0")) or 0.0,
                    safe_literal_float(row.get("visual_logit_class_1")) or 0.0,
                    safe_literal_float(row.get("csg_logit_class_0")) or 0.0,
                    safe_literal_float(row.get("csg_logit_class_1")) or 0.0,
                ]
                output_path = args.figure_dir / "stage54_correct_case_evidence_components.png"
                save_component_bar(labels, values, f"Correct Case Components: {chosen['slide_id']}", output_path)
                add_index(
                    "correct case evidence figure",
                    "ready",
                    [output_path],
                    [args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
                    "Component-level logit decomposition from the full-equivalent evidence export.",
                    "main paper",
                )
            else:
                add_index(
                    "correct case evidence figure",
                    "missing",
                    [],
                    [args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
                    "The selected correct case was not found in the full evidence summary.",
                    "main paper",
                )
        else:
            add_index(
                "correct case evidence figure",
                "missing",
                [],
                [args.case_selection_csv],
                "No correct case selection was available.",
                "main paper",
            )
    else:
        add_index(
            "correct case evidence figure",
            "missing",
            [],
            [args.case_selection_csv, args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
            "Required selected case or full evidence summary is missing.",
            "main paper",
        )

    # 5. failure case evidence figure
    if case_selection is not None and full_summary is not None:
        chosen_df = case_selection[case_selection["selection_type"] == "full_wrong_failure"]
        if not chosen_df.empty:
            chosen = chosen_df.iloc[0]
            full_case = full_summary[full_summary["slide_id"].astype(str) == str(chosen["slide_id"])]
            if not full_case.empty:
                row = full_case.iloc[0]
                labels = ["full_logit_c0", "full_logit_c1", "visual_alpha", "csg_alpha", "pred_margin"]
                values = [
                    safe_literal_float(row.get("final_logit_class_0")) or 0.0,
                    safe_literal_float(row.get("final_logit_class_1")) or 0.0,
                    safe_literal_float(row.get("visual_alpha")) or 0.0,
                    safe_literal_float(row.get("csg_alpha")) or 0.0,
                    safe_literal_float(row.get("pred_margin")) or 0.0,
                ]
                output_path = args.figure_dir / "stage54_failure_case_evidence_components.png"
                save_component_bar(labels, values, f"Failure Case Components: {chosen['slide_id']}", output_path)
                add_index(
                    "failure case evidence figure",
                    "ready",
                    [output_path],
                    [args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
                    "Failure-case confidence and component structure from the full-equivalent evidence export.",
                    "supplementary",
                )
            else:
                add_index(
                    "failure case evidence figure",
                    "missing",
                    [],
                    [args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
                    "The selected failure case was not found in the full evidence summary.",
                    "supplementary",
                )
        else:
            add_index(
                "failure case evidence figure",
                "missing",
                [],
                [args.case_selection_csv],
                "No failure case selection was available.",
                "supplementary",
            )
    else:
        add_index(
            "failure case evidence figure",
            "missing",
            [],
            [args.case_selection_csv, args.full_evidence_dir / "stage32_slide_evidence_summary.csv"],
            "Required selected case or evidence summary is missing.",
            "supplementary",
        )

    # 6. CSG-benefit case figure
    if case_selection is not None and wo_summary is not None and full_summary is not None:
        chosen_df = case_selection[case_selection["selection_type"] == "csg_benefit_full_correct_wo_csg_wrong"]
        if not chosen_df.empty:
            chosen = chosen_df.iloc[0]
            slide_id = str(chosen["slide_id"])
            full_case = full_summary[full_summary["slide_id"].astype(str) == slide_id]
            wo_case = wo_summary[wo_summary["slide_id"].astype(str) == slide_id]
            if not full_case.empty and not wo_case.empty:
                full_row = full_case.iloc[0]
                wo_row = wo_case.iloc[0]
                labels = ["full_margin", "wo_csg_margin", "full_prob1", "wo_csg_prob1"]
                values = [
                    safe_literal_float(full_row.get("pred_margin")) or 0.0,
                    safe_literal_float(wo_row.get("pred_margin")) or 0.0,
                    safe_literal_float(full_row.get("prob_class_1")) or 0.0,
                    safe_literal_float(wo_row.get("prob_class_1")) or 0.0,
                ]
                output_path = args.figure_dir / "stage54_csg_benefit_case_comparison.png"
                save_component_bar(labels, values, f"CSG Benefit Case: {slide_id}", output_path)
                add_index(
                    "CSG-benefit case figure",
                    "ready",
                    [output_path],
                    [args.full_evidence_dir / "stage32_slide_evidence_summary.csv", args.wo_csg_evidence_dir / "stage32_slide_evidence_summary.csv"],
                    "Direct matched-case comparison between full and w/o CSG evidence exports.",
                    "supplementary",
                )
            else:
                add_index(
                    "CSG-benefit case figure",
                    "missing",
                    [],
                    [args.full_evidence_dir / "stage32_slide_evidence_summary.csv", args.wo_csg_evidence_dir / "stage32_slide_evidence_summary.csv"],
                    "The selected CSG-benefit case was not found in both evidence summaries.",
                    "supplementary",
                )
        else:
            add_index(
                "CSG-benefit case figure",
                "missing",
                [],
                [args.case_selection_csv],
                "No CSG-benefit case was available.",
                "supplementary",
            )
    else:
        add_index(
            "CSG-benefit case figure",
            "missing",
            [],
            [args.full_evidence_dir / "stage32_slide_evidence_summary.csv", args.wo_csg_evidence_dir / "stage32_slide_evidence_summary.csv"],
            "w/o CSG evidence export is missing, so the CSG-benefit figure cannot be generated yet.",
            "supplementary",
        )

    # 7. aggregate top concept frequency / importance figure
    if full_long is not None and not full_long.empty:
        subset = full_long[(full_long["class_type"] == "pred") & (full_long["concept_rank"] <= 3)].copy()
        subset["concept_name"] = subset["concept_id"].fillna(subset["concept_text"]).astype(str)
        grouped = subset.groupby("concept_name", as_index=False).agg(
            count=("slide_id", "count"),
            mean_contribution=("contribution", "mean"),
        )
        grouped = grouped.sort_values(["count", "mean_contribution"], ascending=False).head(12)
        if not grouped.empty:
            output_path = args.figure_dir / "stage54_aggregate_top_concept_frequency.png"
            save_barh(
                grouped["concept_name"].tolist(),
                grouped["count"].astype(float).tolist(),
                "Top Concept Frequency Across Full Evidence Export",
                output_path,
                color="#8172B3",
            )
            add_index(
                "aggregate top concept frequency / importance figure",
                "ready",
                [output_path],
                [args.full_evidence_dir / "stage32_top_concepts_long.csv"],
                "Counts the most frequent top-ranked concepts in the existing full-equivalent evidence export.",
                "supplementary",
            )
        else:
            add_index(
                "aggregate top concept frequency / importance figure",
                "missing",
                [],
                [args.full_evidence_dir / "stage32_top_concepts_long.csv"],
                "The concept table did not contain usable rows.",
                "supplementary",
            )
    else:
        add_index(
            "aggregate top concept frequency / importance figure",
            "missing",
            [],
            [args.full_evidence_dir / "stage32_top_concepts_long.csv"],
            "The full concept evidence table is missing.",
            "supplementary",
        )

    rows = index_rows
    pd.DataFrame(rows).to_csv(args.docs_dir / "stage54_figure_index.csv", index=False)
    md_rows = pd.DataFrame(rows).fillna("missing").to_dict("records")
    columns = ["figure", "status", "output_files", "input_files", "placement", "note"]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in md_rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    index_md = "# Stage54 Figure Index\n\n" + "\n".join([header, sep, *body]) + "\n"
    (args.docs_dir / "stage54_figure_index.md").write_text(index_md, encoding="utf-8")


if __name__ == "__main__":
    main()
