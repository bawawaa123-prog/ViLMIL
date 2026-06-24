from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_stage54_rce_evidence_interpretability"
DOCS_DIR = ROOT / "docs" / "stage54b_rce_evidence_figure_polish"
FIGURE_DIR = DOCS_DIR / "figures"

FULL_RUN_DIR = ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1"
FULL_DIRECT_DIR = RESULTS_DIR / "full"
FULL_STAGE32_FALLBACK_DIR = ROOT / "results_stage32" / "stage32_rce_v4_csg_evidence_export"
WO_CSG_DIRECT_DIR = RESULTS_DIR / "wo_csg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Step54B evidence figures and write paper-ready figure docs.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--docs_dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--figure_dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_path_text(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\\", "/").rstrip("/")


def path_matches(manifest_path: str | None, expected_path: Path) -> bool:
    expected = normalize_path_text(str(expected_path))
    actual = normalize_path_text(manifest_path)
    return bool(actual) and (actual == expected or actual.endswith(rel(expected_path)))


def resolve_full_source() -> tuple[Path, str, bool]:
    manifest = safe_read_json(FULL_DIRECT_DIR / "stage32_manifest.json")
    slide_df = safe_read_csv(FULL_DIRECT_DIR / "stage32_slide_evidence_summary.csv")
    if manifest is not None and slide_df is not None:
        if path_matches(str(manifest.get("results_dir", "")), FULL_RUN_DIR) and path_matches(
            str(manifest.get("ckpt_path", "")), FULL_RUN_DIR / "s_0_checkpoint.pt"
        ):
            return FULL_DIRECT_DIR, "stage54_full_direct_export", False
    return FULL_STAGE32_FALLBACK_DIR, "stage32_legacy_deg_export_fallback", True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if math.isnan(numeric):
        return 0.0
    return numeric


def parse_concept_summary(text: object) -> list[tuple[str, float]]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    items: list[tuple[str, float]] = []
    for part in str(text).split("|"):
        piece = part.strip()
        if not piece or ":" not in piece:
            continue
        name, value = piece.rsplit(":", 1)
        try:
            items.append((name.strip(), float(value.strip())))
        except Exception:
            continue
    return items


def choose_case(case_df: pd.DataFrame, case_type: str) -> pd.Series | None:
    subset = case_df[case_df["case_type"] == case_type].copy()
    if subset.empty:
        return None
    if case_type == "full_correct_wo_csg_wrong":
        subset = subset.sort_values(["confidence_delta", "full_confidence"], ascending=[False, False])
    elif case_type == "same_prediction_confidence_shift":
        subset = subset.sort_values(["confidence_delta", "full_confidence"], ascending=[False, False])
    elif case_type == "full_wrong":
        subset = subset.sort_values(["full_confidence", "slide_id"], ascending=[False, True])
    else:
        subset = subset.sort_values(["full_confidence", "slide_id"], ascending=[False, True])
    return subset.iloc[0]


def pick_evidence_row(df: pd.DataFrame, row: pd.Series) -> pd.Series | None:
    subset = df[(df["slide_id"].astype(str) == str(row["slide_id"])) & (df["fold"].astype(int) == int(row["fold"]))]
    if subset.empty:
        return None
    return subset.iloc[0]


def title_for_single(base: str, row: pd.Series, source_label: str) -> str:
    return (
        f"{base}\n"
        f"slide={row['slide_id']} fold={int(row['fold'])} true={row['true_label_name']} "
        f"pred={row['full_pred_name']} conf={row['full_confidence']:.3f} source={source_label}"
    )


def title_for_compare(base: str, row: pd.Series) -> str:
    return (
        f"{base}\n"
        f"slide={row['slide_id']} fold={int(row['fold'])} true={row['true_label_name']} "
        f"full={row['full_pred_name']}({row['full_confidence']:.3f}) "
        f"wo_csg={row['wo_csg_pred_name']}({row['wo_csg_confidence']:.3f})"
    )


def save_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.8), 4.0))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_barh(labels: list[str], values: list[float], title: str, output_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.42)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    for idx, value in enumerate(values):
        offset = 0.01 if value >= 0 else -0.01
        ax.text(value + offset, idx, f"{value:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_component_bar(labels: list[str], values: list[float], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2E86AB" if value >= 0 else "#D1495B" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=28)
    for idx, value in enumerate(values):
        va = "bottom" if value >= 0 else "top"
        ax.text(idx, value, f"{value:.3f}", ha="center", va=va, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def format_claims_row(row: dict[str, object]) -> str:
    return (
        f"| {row['figure_filename']} | {row['figure_type']} | {row['input_data_source']} | "
        f"{row['case_or_aggregate']} | {row['uses_stage32_fallback']} | {row['suggest_main_text']} | "
        f"{row['suggest_supplementary']} | {row['supports_claim']} | {row['cannot_support_claim']} |"
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.docs_dir)
    ensure_dir(args.figure_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    audit_df = safe_read_csv(args.docs_dir / "stage54b_evidence_source_audit.csv")
    case_df = safe_read_csv(args.results_dir / "stage54b_case_level_metadata.csv")
    if audit_df is None or case_df is None:
        raise FileNotFoundError("Run Step54B audit and case metadata scripts before plotting figures.")

    full_source_dir, full_source_label, uses_stage32_fallback = resolve_full_source()
    main_single_case_flag = "yes_with_fallback_disclosure" if uses_stage32_fallback else "yes"
    comparison_note = (
        "Full side uses fallback while wo_csg side uses direct Step54 export."
        if uses_stage32_fallback
        else "Both full and wo_csg sides use Step54 direct export."
    )
    comparison_support = (
        "In a selected matched case, CSG may alter concept ranking and confidence behavior."
        if uses_stage32_fallback
        else "Matched Step54 direct exports show that CSG may alter concept ranking and confidence behavior in selected cases."
    )
    wo_source_dir = WO_CSG_DIRECT_DIR
    full_summary = safe_read_csv(full_source_dir / "stage32_slide_evidence_summary.csv")
    full_long = safe_read_csv(full_source_dir / "stage32_top_concepts_long.csv")
    wo_summary = safe_read_csv(wo_source_dir / "stage32_slide_evidence_summary.csv")
    wo_long = safe_read_csv(wo_source_dir / "stage32_top_concepts_long.csv")
    if full_summary is None or full_long is None or wo_summary is None or wo_long is None:
        raise FileNotFoundError("Required evidence tables are missing for Step54B plotting.")

    correct_case = choose_case(case_df, "full_correct")
    failure_case = choose_case(case_df, "full_wrong")
    benefit_case = choose_case(case_df, "full_correct_wo_csg_wrong")
    shift_case = choose_case(case_df, "same_prediction_confidence_shift")

    figure_rows: list[dict[str, object]] = []

    def add_figure_row(
        *,
        figure_filename: str,
        figure_type: str,
        case_or_aggregate: str,
        slide_id: str,
        fold: str,
        true_label: str,
        pred_label: str,
        confidence: str,
        case_type: str,
        evidence_source: str,
        input_data_source: str,
        uses_fallback: bool,
        suggest_main_text: str,
        suggest_supplementary: str,
        supports_claim: str,
        cannot_support_claim: str,
        provenance_note: str,
    ) -> None:
        figure_rows.append(
            {
                "figure_filename": figure_filename,
                "figure_type": figure_type,
                "input_data_source": input_data_source,
                "case_or_aggregate": case_or_aggregate,
                "slide_id": slide_id,
                "fold": fold,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "case_type": case_type,
                "evidence_source": evidence_source,
                "uses_stage32_fallback": uses_fallback,
                "suggest_main_text": suggest_main_text,
                "suggest_supplementary": suggest_supplementary,
                "supports_claim": supports_claim,
                "cannot_support_claim": cannot_support_claim,
                "provenance_note": provenance_note,
            }
        )

    if correct_case is not None:
        full_case_long = full_long[
            (full_long["slide_id"].astype(str) == str(correct_case["slide_id"]))
            & (full_long["fold"].astype(int) == int(correct_case["fold"]))
            & (full_long["class_type"] == "pred")
        ].copy()
        low = full_case_long[full_case_long["scale"] == "low"].sort_values("concept_rank").head(5)
        high = full_case_long[full_case_long["scale"] == "high"].sort_values("concept_rank").head(5)
        concept_labels = (
            low["concept_id"].fillna(low["concept_text"]).astype(str).tolist()
            + high["concept_id"].fillna(high["concept_text"]).astype(str).tolist()
        )
        matrix = np.asarray(
            [
                low["contribution"].astype(float).tolist() + [0.0] * len(high.index),
                [0.0] * len(low.index) + high["contribution"].astype(float).tolist(),
            ]
        )
        heatmap_path = args.figure_dir / "stage54b_region_concept_heatmap.png"
        if concept_labels and matrix.size > 0:
            save_heatmap(
                matrix,
                ["low-scale", "high-scale"],
                concept_labels,
                title_for_single("Stage54B Region-Concept Heatmap", correct_case, full_source_label),
                heatmap_path,
            )
        add_figure_row(
            figure_filename=rel(heatmap_path),
            figure_type="single-case region-concept heatmap",
            case_or_aggregate="case",
            slide_id=str(correct_case["slide_id"]),
            fold=str(int(correct_case["fold"])),
            true_label=str(correct_case["true_label_name"]),
            pred_label=str(correct_case["full_pred_name"]),
            confidence=f"{correct_case['full_confidence']:.3f}",
            case_type=str(correct_case["case_type"]),
            evidence_source=full_source_label,
            input_data_source=(
                f"{rel(full_source_dir / 'stage32_top_concepts_long.csv')}; "
                f"{rel(args.results_dir / 'stage54b_case_level_metadata.csv')}"
            ),
            uses_fallback=uses_stage32_fallback,
            suggest_main_text=main_single_case_flag,
            suggest_supplementary="yes",
            supports_claim="Example region-concept evidence can be inspected at low and high scales.",
            cannot_support_claim="This figure is not a localization benchmark or expert annotation surrogate.",
            provenance_note="Single-case full-side evidence uses fallback provenance." if uses_stage32_fallback else "Direct full export.",
        )

        low_items = parse_concept_summary(
            pick_evidence_row(full_summary, correct_case).get("top_low_concepts_for_pred")
            if pick_evidence_row(full_summary, correct_case) is not None
            else None
        )
        high_items = parse_concept_summary(
            pick_evidence_row(full_summary, correct_case).get("top_high_concepts_for_pred")
            if pick_evidence_row(full_summary, correct_case) is not None
            else None
        )
        labels = [f"L:{name}" for name, _ in low_items[:5]] + [f"H:{name}" for name, _ in high_items[:5]]
        values = [value for _, value in low_items[:5]] + [value for _, value in high_items[:5]]
        low_high_path = args.figure_dir / "stage54b_low_high_concept_comparison.png"
        if labels:
            save_barh(
                labels,
                values,
                title_for_single("Stage54B Low-High Concept Comparison", correct_case, full_source_label),
                low_high_path,
                color="#3B7EA1",
            )
        add_figure_row(
            figure_filename=rel(low_high_path),
            figure_type="single-case low-vs-high concept bar chart",
            case_or_aggregate="case",
            slide_id=str(correct_case["slide_id"]),
            fold=str(int(correct_case["fold"])),
            true_label=str(correct_case["true_label_name"]),
            pred_label=str(correct_case["full_pred_name"]),
            confidence=f"{correct_case['full_confidence']:.3f}",
            case_type=str(correct_case["case_type"]),
            evidence_source=full_source_label,
            input_data_source=(
                f"{rel(full_source_dir / 'stage32_slide_evidence_summary.csv')}; "
                f"{rel(args.results_dir / 'stage54b_case_level_metadata.csv')}"
            ),
            uses_fallback=uses_stage32_fallback,
            suggest_main_text="no",
            suggest_supplementary="yes",
            supports_claim="Concept contributions from low and high scales can be compared within one evidence example.",
            cannot_support_claim="This figure cannot prove systematic localization quality or statistically significant benefit.",
            provenance_note="Caption should state fallback provenance for the full-side example." if uses_stage32_fallback else "Direct full export.",
        )

        full_case_summary = pick_evidence_row(full_summary, correct_case)
        component_path = args.figure_dir / "stage54b_correct_case_evidence_components.png"
        if full_case_summary is not None:
            component_labels = ["low_c0", "low_c1", "high_c0", "high_c1", "visual_c0", "visual_c1", "csg_c0", "csg_c1"]
            component_values = [
                safe_float(full_case_summary.get("low_logit_class_0")),
                safe_float(full_case_summary.get("low_logit_class_1")),
                safe_float(full_case_summary.get("high_logit_class_0")),
                safe_float(full_case_summary.get("high_logit_class_1")),
                safe_float(full_case_summary.get("visual_logit_class_0")),
                safe_float(full_case_summary.get("visual_logit_class_1")),
                safe_float(full_case_summary.get("csg_logit_class_0")),
                safe_float(full_case_summary.get("csg_logit_class_1")),
            ]
            save_component_bar(
                component_labels,
                component_values,
                title_for_single("Stage54B Correct-Case Evidence Components", correct_case, full_source_label),
                component_path,
            )
        add_figure_row(
            figure_filename=rel(component_path),
            figure_type="single-case component decomposition",
            case_or_aggregate="case",
            slide_id=str(correct_case["slide_id"]),
            fold=str(int(correct_case["fold"])),
            true_label=str(correct_case["true_label_name"]),
            pred_label=str(correct_case["full_pred_name"]),
            confidence=f"{correct_case['full_confidence']:.3f}",
            case_type=str(correct_case["case_type"]),
            evidence_source=full_source_label,
            input_data_source=rel(full_source_dir / "stage32_slide_evidence_summary.csv"),
            uses_fallback=uses_stage32_fallback,
            suggest_main_text=main_single_case_flag,
            suggest_supplementary="yes",
            supports_claim="Evidence components can be decomposed into low-scale, high-scale, visual residual, and CSG terms.",
            cannot_support_claim="This figure cannot prove that the model localizes pathology at pathologist level.",
            provenance_note="Full-side component plot uses fallback evidence." if uses_stage32_fallback else "Direct full export.",
        )

    if shift_case is not None:
        full_case = full_long[
            (full_long["slide_id"].astype(str) == str(shift_case["slide_id"]))
            & (full_long["fold"].astype(int) == int(shift_case["fold"]))
            & (full_long["class_type"] == "pred")
        ].copy()
        wo_case = wo_long[
            (wo_long["slide_id"].astype(str) == str(shift_case["slide_id"]))
            & (wo_long["fold"].astype(int) == int(shift_case["fold"]))
            & (wo_long["class_type"] == "pred")
        ].copy()
        if not full_case.empty and not wo_case.empty:
            full_case["concept_name"] = full_case["concept_id"].fillna(full_case["concept_text"]).astype(str)
            wo_case["concept_name"] = wo_case["concept_id"].fillna(wo_case["concept_text"]).astype(str)
            delta_df = (
                full_case.groupby("concept_name", as_index=False)["contribution"]
                .mean()
                .merge(
                    wo_case.groupby("concept_name", as_index=False)["contribution"].mean(),
                    on="concept_name",
                    how="outer",
                    suffixes=("_full", "_wo_csg"),
                )
                .fillna(0.0)
            )
            delta_df["delta"] = delta_df["contribution_full"] - delta_df["contribution_wo_csg"]
            delta_df = delta_df.sort_values("delta", ascending=False).head(12)
            delta_path = args.figure_dir / "stage54b_full_vs_wo_csg_concept_ranking_delta.png"
            save_barh(
                delta_df["concept_name"].tolist(),
                delta_df["delta"].astype(float).tolist(),
                title_for_compare("Stage54B Full vs w/o CSG Concept Delta", shift_case),
                delta_path,
                color="#5A9367",
            )
            add_figure_row(
                figure_filename=rel(delta_path),
                figure_type="matched-case concept ranking delta",
                case_or_aggregate="case",
                slide_id=str(shift_case["slide_id"]),
                fold=str(int(shift_case["fold"])),
                true_label=str(shift_case["true_label_name"]),
                pred_label=(
                    f"full={shift_case['full_pred_name']}; "
                    f"wo_csg={shift_case['wo_csg_pred_name']}"
                ),
                confidence=(
                    f"full={shift_case['full_confidence']:.3f}; "
                    f"wo_csg={shift_case['wo_csg_confidence']:.3f}"
                ),
                case_type=str(shift_case["case_type"]),
                evidence_source=f"full={full_source_label}; wo_csg=stage54_wo_csg_direct_export",
                input_data_source=(
                    f"{rel(full_source_dir / 'stage32_top_concepts_long.csv')}; "
                    f"{rel(wo_source_dir / 'stage32_top_concepts_long.csv')}"
                ),
                uses_fallback=uses_stage32_fallback,
                suggest_main_text="no",
                suggest_supplementary="yes",
                supports_claim=comparison_support,
                cannot_support_claim="This cross-source example does not establish a fully same-source evidence comparison or statistical significance.",
                provenance_note=comparison_note,
            )

    if failure_case is not None:
        failure_summary = pick_evidence_row(full_summary, failure_case)
        failure_path = args.figure_dir / "stage54b_failure_case_evidence_components.png"
        if failure_summary is not None:
            failure_labels = ["final_logit_c0", "final_logit_c1", "visual_alpha", "csg_alpha", "pred_margin"]
            failure_values = [
                safe_float(failure_summary.get("final_logit_class_0")),
                safe_float(failure_summary.get("final_logit_class_1")),
                safe_float(failure_summary.get("visual_alpha")),
                safe_float(failure_summary.get("csg_alpha")),
                safe_float(failure_summary.get("pred_margin")),
            ]
            save_component_bar(
                failure_labels,
                failure_values,
                title_for_single("Stage54B Failure-Case Evidence Components", failure_case, full_source_label),
                failure_path,
            )
        add_figure_row(
            figure_filename=rel(failure_path),
            figure_type="failure-case component decomposition",
            case_or_aggregate="case",
            slide_id=str(failure_case["slide_id"]),
            fold=str(int(failure_case["fold"])),
            true_label=str(failure_case["true_label_name"]),
            pred_label=str(failure_case["full_pred_name"]),
            confidence=f"{failure_case['full_confidence']:.3f}",
            case_type=str(failure_case["case_type"]),
            evidence_source=full_source_label,
            input_data_source=rel(full_source_dir / "stage32_slide_evidence_summary.csv"),
            uses_fallback=uses_stage32_fallback,
            suggest_main_text="no",
            suggest_supplementary="yes",
            supports_claim="Failure cases can be inspected for component imbalance or conflicting evidence.",
            cannot_support_claim="This figure cannot prove the exact clinical reason for the model error.",
            provenance_note="Failure example uses fallback provenance on the full side." if uses_stage32_fallback else "Direct full export.",
        )

    if benefit_case is not None:
        full_benefit = pick_evidence_row(full_summary, benefit_case)
        wo_benefit = pick_evidence_row(wo_summary, benefit_case)
        benefit_path = args.figure_dir / "stage54b_csg_benefit_case_comparison.png"
        if full_benefit is not None and wo_benefit is not None:
            benefit_labels = ["full_margin", "wo_csg_margin", "full_prob1", "wo_csg_prob1"]
            benefit_values = [
                safe_float(full_benefit.get("pred_margin")),
                safe_float(wo_benefit.get("pred_margin")),
                safe_float(full_benefit.get("prob_class_1")),
                safe_float(wo_benefit.get("prob_class_1")),
            ]
            save_component_bar(
                benefit_labels,
                benefit_values,
                title_for_compare("Stage54B CSG-Benefit Case Comparison", benefit_case),
                benefit_path,
            )
        add_figure_row(
            figure_filename=rel(benefit_path),
            figure_type="selected full-vs-wo_csg comparison",
            case_or_aggregate="case",
            slide_id=str(benefit_case["slide_id"]),
            fold=str(int(benefit_case["fold"])),
            true_label=str(benefit_case["true_label_name"]),
            pred_label=(
                f"full={benefit_case['full_pred_name']}; "
                f"wo_csg={benefit_case['wo_csg_pred_name']}"
            ),
            confidence=(
                f"full={benefit_case['full_confidence']:.3f}; "
                f"wo_csg={benefit_case['wo_csg_confidence']:.3f}"
            ),
            case_type=str(benefit_case["case_type"]),
            evidence_source=f"full={full_source_label}; wo_csg=stage54_wo_csg_direct_export",
            input_data_source=(
                f"{rel(full_source_dir / 'stage32_slide_evidence_summary.csv')}; "
                f"{rel(wo_source_dir / 'stage32_slide_evidence_summary.csv')}"
            ),
            uses_fallback=uses_stage32_fallback,
            suggest_main_text="no",
            suggest_supplementary="yes",
            supports_claim=(
                "A selected case suggests CSG can improve margin or confidence behavior."
                if uses_stage32_fallback
                else "A matched direct-export case suggests CSG can improve margin or confidence behavior."
            ),
            cannot_support_claim="This single case cannot support large ACC gains or a same-source causal proof.",
            provenance_note=(
                "Comparison mixes full fallback evidence with direct wo_csg export."
                if uses_stage32_fallback
                else "Comparison uses direct Step54 exports on both sides."
            ),
        )

    aggregate_subset = full_long[(full_long["class_type"] == "pred") & (full_long["concept_rank"] <= 3)].copy()
    aggregate_subset["concept_name"] = aggregate_subset["concept_id"].fillna(aggregate_subset["concept_text"]).astype(str)
    grouped = (
        aggregate_subset.groupby("concept_name", as_index=False)
        .agg(count=("slide_id", "count"), mean_contribution=("contribution", "mean"))
        .sort_values(["count", "mean_contribution"], ascending=[False, False])
        .head(12)
    )
    aggregate_path = args.figure_dir / "stage54b_aggregate_top_concept_frequency.png"
    save_barh(
        grouped["concept_name"].tolist(),
        grouped["count"].astype(float).tolist(),
        "Stage54B Aggregate Top Concept Frequency\naggregate figure source="
        + full_source_label,
        aggregate_path,
        color="#8F6AAE",
    )
    add_figure_row(
        figure_filename=rel(aggregate_path),
        figure_type="aggregate concept frequency bar chart",
        case_or_aggregate="aggregate",
        slide_id="aggregate_figure",
        fold="aggregate_figure",
        true_label="aggregate_figure",
        pred_label="aggregate_predicted_class_frequency",
        confidence="aggregate_figure",
        case_type="aggregate",
        evidence_source=full_source_label,
        input_data_source=rel(full_source_dir / "stage32_top_concepts_long.csv"),
        uses_fallback=uses_stage32_fallback,
        suggest_main_text="no",
        suggest_supplementary="yes",
        supports_claim="Frequently recurring top predicted-class concepts can be summarized over the audited export.",
        cannot_support_claim="This aggregate count does not measure localization accuracy or cross-model statistical significance.",
        provenance_note="Aggregate figure inherits full-side provenance." if uses_stage32_fallback else "Direct full export.",
    )

    figure_df = pd.DataFrame(figure_rows)
    figure_csv_path = args.docs_dir / "stage54b_figure_index.csv"
    figure_df.to_csv(figure_csv_path, index=False, encoding="utf-8")

    md_lines = [
        "# Stage54B Figure Index",
        "",
        "| figure_filename | figure_type | input_data_source | case_or_aggregate | evidence_source | uses_stage32_fallback | suggest_main_text | suggest_supplementary | supports_claim | cannot_support_claim | provenance_note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in figure_rows:
        md_lines.append(
            f"| {row['figure_filename']} | {row['figure_type']} | {row['input_data_source']} | "
            f"{row['case_or_aggregate']} | {row['evidence_source']} | {row['uses_stage32_fallback']} | "
            f"{row['suggest_main_text']} | {row['suggest_supplementary']} | {row['supports_claim']} | "
            f"{row['cannot_support_claim']} | {row['provenance_note']} |"
        )
    (args.docs_dir / "stage54b_figure_index.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    caption_lines = ["# Stage54B Paper Figure Caption Drafts", ""]
    for row in figure_rows:
        caption_lines.extend(
            [
                f"## {Path(str(row['figure_filename'])).name}",
                f"{row['figure_type']}. Source: {row['evidence_source']}. "
                f"Context: {row['case_or_aggregate']}, slide={row['slide_id']}, fold={row['fold']}, "
                f"true={row['true_label']}, pred={row['pred_label']}, confidence={row['confidence']}. "
                f"{row['supports_claim']} {row['cannot_support_claim']} "
                f"Provenance note: {row['provenance_note']}",
                "",
            ]
        )
    (args.docs_dir / "stage54b_paper_figure_caption_drafts.md").write_text(
        "\n".join(caption_lines), encoding="utf-8"
    )

    full_direct_ready = any(
        (audit_df["source_id"] == "full_direct_expected")
        & (audit_df["provenance_status"] == "direct_export_verified")
    )
    summary_lines = [
        "# Stage54B Summary",
        "",
        "## Audit Outcome",
        f"- Full direct export ready in Step54 layout: `{full_direct_ready}`.",
        "- Step54B did not run full or wo_csg evidence export commands.",
        f"- Full-side figure package currently uses `{full_source_label}`.",
        "- wo_csg-side figure package uses `stage54_wo_csg_direct_export`.",
        (
            "- Full vs wo_csg figure provenance is now direct-export based on both sides."
            if full_direct_ready and not uses_stage32_fallback
            else "- Cross-source provenance must be disclosed in any full vs wo_csg figure or caption."
        ),
        "",
        "## Generated Outputs",
        f"- Figure directory: `{rel(args.figure_dir)}`",
        f"- Figure index: `{rel(figure_csv_path)}` and `docs/stage54b_rce_evidence_figure_polish/stage54b_figure_index.md`",
        "- Caption drafts, summary, claims, and next steps were regenerated for Step54B.",
        "",
        "## Practical Reading",
        (
            "- Single-case figures now use the refreshed Step54 full direct export."
            if full_direct_ready and not uses_stage32_fallback
            else "- Single-case fallback figures can support interpretability illustration with explicit provenance disclosure."
        ),
        (
            "- Matched full vs wo_csg comparison figures now come from the Step54 evidence pipeline on both sides."
            if full_direct_ready and not uses_stage32_fallback
            else "- Cross-source comparison figures should stay in supplementary material unless the direct full export is regenerated."
        ),
        "- None of the figures should be used to claim localization benchmarking or pathologist-level alignment.",
    ]
    (args.docs_dir / "stage54b_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    claims_lines = [
        "# Stage54B Claims To Make And Avoid",
        "",
        "## Claims To Make",
        "- RCE evidence visualizations help inspect region-concept evidence behavior.",
        "- Concept-level evidence provides interpretable clues for model behavior in selected cases.",
        (
            "- Full and w/o CSG evidence are both exported through the Step54 evidence pipeline."
            if full_direct_ready and not uses_stage32_fallback
            else "- Selected full vs w/o CSG case comparisons suggest CSG can affect evidence ranking or confidence behavior, with provenance differences disclosed when fallback is used."
        ),
        (
            "- Full vs w/o CSG evidence-level comparisons can be used as matched interpretability illustrations with explicit variant provenance."
            if full_direct_ready and not uses_stage32_fallback
            else "- Concept-level evidence comparisons should remain explicitly provenance-aware when sources differ."
        ),
        "",
        "## Claims To Avoid",
        "- Do not claim pathologist-level localization.",
        "- Do not claim statistically significant localization improvement.",
        "- Do not claim that CSG produces a large ACC gain.",
        (
            "- Do not describe the current full vs w/o CSG figures as training-statistical proof."
            if full_direct_ready and not uses_stage32_fallback
            else "- Do not describe the current full vs w/o CSG figures as a fully same-source 5-fold evidence comparison while full-side fallback remains in use."
        ),
        "- Do not equate heatmaps or concept evidence with expert pathology annotations.",
    ]
    (args.docs_dir / "stage54b_claims_to_make_and_avoid.md").write_text(
        "\n".join(claims_lines) + "\n", encoding="utf-8"
    )

    next_steps_lines = [
        "# Stage54B Next Steps",
        "",
        (
            "- Full direct export is already available; preserve `results_stage54_rce_evidence_interpretability/full/` as the preferred full-side source."
            if full_direct_ready and not uses_stage32_fallback
            else "- If a direct full export is needed, run `MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh` manually."
        ),
        (
            "- If full export is regenerated later, rerun the three Step54B scripts to refresh provenance and figures."
            if full_direct_ready and not uses_stage32_fallback
            else "- After a successful full direct export, rerun `python scripts/analysis/build_stage54b_evidence_source_audit.py`."
        ),
        (
            "- Keep claims constrained to interpretability illustrations rather than statistical training claims."
            if full_direct_ready and not uses_stage32_fallback
            else "- Then rerun `python scripts/analysis/build_stage54b_case_metadata.py` and `python scripts/analysis/plot_stage54b_rce_evidence_figures.py` to replace fallback provenance where possible."
        ),
        (
            "- Full vs w/o CSG comparison figures can remain supplementary unless one is specifically promoted into the paper with cautious captioning."
            if full_direct_ready and not uses_stage32_fallback
            else "- Keep full vs w/o CSG comparison figures in supplementary material until provenance is fully same-source."
        ),
    ]
    (args.docs_dir / "stage54b_next_steps.md").write_text("\n".join(next_steps_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
