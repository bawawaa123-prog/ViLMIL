from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "results_stage38" / "stage38_lh_consistency_failure_comparison"
DEFAULT_SKELETON_EVIDENCE_DIR = ROOT / "results_stage38" / "evidence_export_skeleton_fold0_test"
DEFAULT_LH_EVIDENCE_DIR = ROOT / "results_stage38" / "evidence_export_lh_l001_m0_fold0_test"
DEFAULT_SKELETON_FAILURE_DIR = ROOT / "results_stage38" / "failure_analysis_skeleton_fold0_test"
DEFAULT_LH_FAILURE_DIR = ROOT / "results_stage38" / "failure_analysis_lh_l001_m0_fold0_test"
FAILURE_TYPES = [
    "visual_residual_override",
    "low_high_conflict",
    "concept_wrong_class_drift",
    "uncertain_low_margin",
    "high_scale_dominant_wrong",
    "low_scale_dominant_wrong",
    "prompt_confusion",
    "csg_misleading",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Step38 skeleton vs low-high consistency failure exports.")
    parser.add_argument("--skeleton_evidence_dir", type=str, default=str(DEFAULT_SKELETON_EVIDENCE_DIR))
    parser.add_argument("--lh_evidence_dir", type=str, default=str(DEFAULT_LH_EVIDENCE_DIR))
    parser.add_argument("--skeleton_failure_dir", type=str, default=str(DEFAULT_SKELETON_FAILURE_DIR))
    parser.add_argument("--lh_failure_dir", type=str, default=str(DEFAULT_LH_FAILURE_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skeleton_name", type=str, default="skeleton")
    parser.add_argument("--lh_name", type=str, default="lh_l001_m0")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path_str


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def compute_metrics(slide_df: pd.DataFrame) -> dict[str, float | int | None]:
    if slide_df.empty:
        return {
            "test_auc": None,
            "test_acc": None,
            "test_f1": None,
            "balanced_acc": None,
            "sensitivity": None,
            "specificity": None,
            "pr_auc": None,
            "num_correct": 0,
            "num_error": 0,
            "num_slides": 0,
        }

    y_true = pd.to_numeric(slide_df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = pd.to_numeric(slide_df["pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_score = pd.to_numeric(slide_df.get("prob_class_1"), errors="coerce").fillna(0.0).astype(float).to_numpy()

    metrics: dict[str, float | int | None] = {
        "test_auc": None,
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "pr_auc": None,
        "num_correct": int((y_true == y_pred).sum()),
        "num_error": int((y_true != y_pred).sum()),
        "num_slides": int(len(slide_df.index)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["test_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


def load_variant_bundle(evidence_dir: Path, failure_dir: Path, variant_name: str) -> dict[str, object]:
    slide_evidence = read_csv(evidence_dir / "stage32_slide_evidence_summary.csv")
    manifest = read_json(evidence_dir / "stage32_manifest.json")
    slide_failure = read_csv(failure_dir / "stage33_slide_failure_labels.csv")
    error_failure = read_csv(failure_dir / "stage33_error_failure_cases.csv")
    low_high_conflict = read_csv(failure_dir / "stage33_low_high_conflict_summary.csv")
    visual_diag = read_csv(failure_dir / "stage33_visual_residual_diagnostics.csv")
    failure_type_counts = read_csv(failure_dir / "stage33_failure_type_counts.csv")
    return {
        "variant_name": variant_name,
        "evidence_dir": evidence_dir,
        "failure_dir": failure_dir,
        "slide_evidence": slide_evidence,
        "manifest": manifest,
        "slide_failure": slide_failure,
        "error_failure": error_failure,
        "low_high_conflict": low_high_conflict,
        "visual_diag": visual_diag,
        "failure_type_counts": failure_type_counts,
    }


def build_metric_comparison(skeleton: dict[str, object], lh: dict[str, object]) -> pd.DataFrame:
    rows = []
    for bundle in [skeleton, lh]:
        slide_df = bundle["slide_evidence"]
        metrics = compute_metrics(slide_df)
        rows.append(
            {
                "variant": bundle["variant_name"],
                **metrics,
                "manifest_test_auc": safe_float(bundle["manifest"].get("metrics", {}).get("test_auc")),
                "manifest_test_acc": safe_float(bundle["manifest"].get("metrics", {}).get("test_acc")),
                "manifest_test_f1": safe_float(bundle["manifest"].get("metrics", {}).get("test_f1")),
                "manifest_balanced_acc": safe_float(bundle["manifest"].get("metrics", {}).get("balanced_acc")),
                "manifest_pr_auc": safe_float(bundle["manifest"].get("metrics", {}).get("pr_auc")),
            }
        )
    df = pd.DataFrame(rows)
    if len(df.index) == 2:
        base = df.iloc[0]
        compare = df.iloc[1]
        delta_row = {"variant": f"{compare['variant']} - {base['variant']}"}
        for column in [
            "test_auc",
            "test_acc",
            "test_f1",
            "balanced_acc",
            "sensitivity",
            "specificity",
            "pr_auc",
            "num_correct",
            "num_error",
        ]:
            delta_row[column] = (
                float(compare[column]) - float(base[column])
                if pd.notna(compare[column]) and pd.notna(base[column])
                else math.nan
            )
        df = pd.concat([df, pd.DataFrame([delta_row])], ignore_index=True)
    return df


def parse_failure_labels(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {item.strip() for item in str(value).split("|") if item.strip()}


def build_failure_type_comparison(skeleton: dict[str, object], lh: dict[str, object]) -> pd.DataFrame:
    def count_any_label(df: pd.DataFrame, failure_type: str) -> int:
        if df.empty:
            return 0
        return int(df["failure_labels"].apply(lambda value: failure_type in parse_failure_labels(value)).sum())

    def count_primary(df: pd.DataFrame, failure_type: str) -> int:
        if df.empty or "primary_failure_type" not in df.columns:
            return 0
        return int((df["primary_failure_type"] == failure_type).sum())

    rows = []
    sk_df = skeleton["slide_failure"]
    lh_df = lh["slide_failure"]
    sk_error_df = sk_df[sk_df["correct"] == False].copy()
    lh_error_df = lh_df[lh_df["correct"] == False].copy()
    for failure_type in FAILURE_TYPES:
        rows.append(
            {
                "failure_type": failure_type,
                f"{skeleton['variant_name']}_count_any_label": count_any_label(sk_error_df, failure_type),
                f"{skeleton['variant_name']}_count_as_primary": count_primary(sk_error_df, failure_type),
                f"{lh['variant_name']}_count_any_label": count_any_label(lh_error_df, failure_type),
                f"{lh['variant_name']}_count_as_primary": count_primary(lh_error_df, failure_type),
            }
        )
    df = pd.DataFrame(rows)
    df[f"{lh['variant_name']}_minus_{skeleton['variant_name']}_any_label"] = (
        df[f"{lh['variant_name']}_count_any_label"] - df[f"{skeleton['variant_name']}_count_any_label"]
    )
    df[f"{lh['variant_name']}_minus_{skeleton['variant_name']}_primary"] = (
        df[f"{lh['variant_name']}_count_as_primary"] - df[f"{skeleton['variant_name']}_count_as_primary"]
    )
    return df.sort_values(by=[f"{lh['variant_name']}_minus_{skeleton['variant_name']}_any_label", "failure_type"], ascending=[True, True]).reset_index(drop=True)


def build_error_overlap(skeleton: dict[str, object], lh: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sk_df = skeleton["slide_failure"].copy()
    lh_df = lh["slide_failure"].copy()
    sk_index = sk_df.set_index("slide_id")
    lh_index = lh_df.set_index("slide_id")
    common_ids = sorted(set(sk_index.index) & set(lh_index.index))

    records = []
    fixed_rows = []
    regressed_rows = []
    persistent_rows = []
    for slide_id in common_ids:
        sk_row = sk_index.loc[slide_id]
        lh_row = lh_index.loc[slide_id]
        sk_correct = bool(sk_row["correct"])
        lh_correct = bool(lh_row["correct"])
        records.append(
            {
                "slide_id": slide_id,
                "label": sk_row.get("label"),
                "skeleton_pred": sk_row.get("pred"),
                "lh_pred": lh_row.get("pred"),
                "skeleton_correct": sk_correct,
                "lh_correct": lh_correct,
                "status": (
                    "fixed" if (not sk_correct and lh_correct)
                    else "regressed" if (sk_correct and not lh_correct)
                    else "persistent_error" if (not sk_correct and not lh_correct)
                    else "both_correct"
                ),
                "skeleton_failure_labels": sk_row.get("failure_labels"),
                "lh_failure_labels": lh_row.get("failure_labels"),
                "skeleton_primary_failure_type": sk_row.get("primary_failure_type"),
                "lh_primary_failure_type": lh_row.get("primary_failure_type"),
            }
        )
        if not sk_correct and lh_correct:
            fixed_rows.append(records[-1])
        elif sk_correct and not lh_correct:
            regressed_rows.append(records[-1])
        elif not sk_correct and not lh_correct:
            persistent_rows.append(records[-1])

    overlap_df = pd.DataFrame(records)
    return (
        overlap_df,
        pd.DataFrame(fixed_rows),
        pd.DataFrame(regressed_rows),
        pd.DataFrame(persistent_rows),
    )


def build_conflict_comparison(
    skeleton: dict[str, object],
    lh: dict[str, object],
    fixed_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for bundle in [skeleton, lh]:
        df = bundle["slide_failure"].copy()
        error_df = df[df["correct"] == False].copy()
        low_margin = pd.to_numeric(error_df.get("low_true_vs_wrong_margin"), errors="coerce")
        high_margin = pd.to_numeric(error_df.get("high_true_vs_wrong_margin"), errors="coerce")
        margin_gap = (low_margin - high_margin).abs()
        rows.append(
            {
                "variant": bundle["variant_name"],
                "error_low_high_conflict_count": int(error_df["failure_labels"].apply(lambda value: "low_high_conflict" in parse_failure_labels(value)).sum()),
                "error_both_support_wrong_count": int((error_df.get("low_high_joint_state") == "both_support_wrong").sum()),
                "error_margin_gap_mean": safe_float(margin_gap.mean()),
                "error_margin_gap_median": safe_float(margin_gap.median()),
            }
        )

    fixed_conflict_repaired = 0
    if not fixed_df.empty:
        sk_df = skeleton["slide_failure"].set_index("slide_id")
        for slide_id in fixed_df["slide_id"].tolist():
            sk_labels = parse_failure_labels(sk_df.loc[slide_id].get("failure_labels"))
            if "low_high_conflict" in sk_labels:
                fixed_conflict_repaired += 1

    rows.append(
        {
            "variant": f"{lh['variant_name']} - {skeleton['variant_name']}",
            "error_low_high_conflict_count": rows[1]["error_low_high_conflict_count"] - rows[0]["error_low_high_conflict_count"],
            "error_both_support_wrong_count": rows[1]["error_both_support_wrong_count"] - rows[0]["error_both_support_wrong_count"],
            "error_margin_gap_mean": (
                rows[1]["error_margin_gap_mean"] - rows[0]["error_margin_gap_mean"]
                if rows[0]["error_margin_gap_mean"] is not None and rows[1]["error_margin_gap_mean"] is not None
                else math.nan
            ),
            "error_margin_gap_median": (
                rows[1]["error_margin_gap_median"] - rows[0]["error_margin_gap_median"]
                if rows[0]["error_margin_gap_median"] is not None and rows[1]["error_margin_gap_median"] is not None
                else math.nan
            ),
            "fixed_cases_with_conflict_repaired": fixed_conflict_repaired,
        }
    )
    return pd.DataFrame(rows)


def build_visual_override_comparison(skeleton: dict[str, object], lh: dict[str, object]) -> pd.DataFrame:
    rows = []
    for bundle in [skeleton, lh]:
        df = bundle["slide_failure"].copy()
        error_df = df[df["correct"] == False].copy()
        visual_ratio_error = pd.to_numeric(error_df.get("visual_source_ratio"), errors="coerce")
        visual_ratio_all = pd.to_numeric(df.get("visual_source_ratio"), errors="coerce")
        wrong_support_error = pd.to_numeric(error_df.get("visual_supports_wrong"), errors="coerce")
        rows.append(
            {
                "variant": bundle["variant_name"],
                "visual_residual_override_error_count": int(error_df["failure_labels"].apply(lambda value: "visual_residual_override" in parse_failure_labels(value)).sum()),
                "error_visual_source_ratio_mean": safe_float(visual_ratio_error.mean()),
                "all_visual_source_ratio_mean": safe_float(visual_ratio_all.mean()),
                "error_wrong_class_visual_support_ratio": safe_float(wrong_support_error.mean()),
            }
        )
    rows.append(
        {
            "variant": f"{lh['variant_name']} - {skeleton['variant_name']}",
            "visual_residual_override_error_count": rows[1]["visual_residual_override_error_count"] - rows[0]["visual_residual_override_error_count"],
            "error_visual_source_ratio_mean": (
                rows[1]["error_visual_source_ratio_mean"] - rows[0]["error_visual_source_ratio_mean"]
                if rows[0]["error_visual_source_ratio_mean"] is not None and rows[1]["error_visual_source_ratio_mean"] is not None
                else math.nan
            ),
            "all_visual_source_ratio_mean": (
                rows[1]["all_visual_source_ratio_mean"] - rows[0]["all_visual_source_ratio_mean"]
                if rows[0]["all_visual_source_ratio_mean"] is not None and rows[1]["all_visual_source_ratio_mean"] is not None
                else math.nan
            ),
            "error_wrong_class_visual_support_ratio": (
                rows[1]["error_wrong_class_visual_support_ratio"] - rows[0]["error_wrong_class_visual_support_ratio"]
                if rows[0]["error_wrong_class_visual_support_ratio"] is not None and rows[1]["error_wrong_class_visual_support_ratio"] is not None
                else math.nan
            ),
        }
    )
    return pd.DataFrame(rows)


def build_recommendations(
    metric_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
    visual_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    regressed_df: pd.DataFrame,
    persistent_df: pd.DataFrame,
    skeleton_name: str,
    lh_name: str,
) -> dict[str, object]:
    metric_lookup = {str(row["variant"]): row for _, row in metric_df.iterrows() if row["variant"] in {skeleton_name, lh_name}}
    conflict_lookup = {str(row["variant"]): row for _, row in conflict_df.iterrows() if row["variant"] in {skeleton_name, lh_name, f"{lh_name} - {skeleton_name}"}}
    visual_lookup = {str(row["variant"]): row for _, row in visual_df.iterrows() if row["variant"] in {skeleton_name, lh_name, f"{lh_name} - {skeleton_name}"}}

    fixed_count = int(len(fixed_df.index))
    regressed_count = int(len(regressed_df.index))
    persistent_count = int(len(persistent_df.index))
    low_high_conflict_delta = conflict_lookup.get(f"{lh_name} - {skeleton_name}", {}).get("error_low_high_conflict_count", math.nan)
    visual_override_delta = visual_lookup.get(f"{lh_name} - {skeleton_name}", {}).get("visual_residual_override_error_count", math.nan)
    auc_delta = None
    pr_auc_delta = None
    if skeleton_name in metric_lookup and lh_name in metric_lookup:
        sk_metrics = metric_lookup[skeleton_name]
        lh_metrics = metric_lookup[lh_name]
        if pd.notna(sk_metrics.get("test_auc")) and pd.notna(lh_metrics.get("test_auc")):
            auc_delta = float(lh_metrics["test_auc"]) - float(sk_metrics["test_auc"])
        if pd.notna(sk_metrics.get("pr_auc")) and pd.notna(lh_metrics.get("pr_auc")):
            pr_auc_delta = float(lh_metrics["pr_auc"]) - float(sk_metrics["pr_auc"])

    keep_as_final_candidate = False
    recommendation = "keep_skeleton_as_final_main_model"
    rationale = []
    if fixed_count > regressed_count and pd.notna(low_high_conflict_delta) and float(low_high_conflict_delta) < 0:
        if pd.isna(visual_override_delta) or float(visual_override_delta) <= 0:
            if (auc_delta is None or auc_delta > -0.01) and (pr_auc_delta is None or pr_auc_delta > -0.01):
                keep_as_final_candidate = True
                recommendation = "keep_lh_consistency_as_final_candidate_module"
                rationale.append("fixed cases exceed regressed cases")
                rationale.append("low_high_conflict decreases")
                rationale.append("visual_residual_override does not increase materially")
    if not keep_as_final_candidate:
        if fixed_count < regressed_count:
            rationale.append("fixed cases are fewer than regressed cases")
        if pd.notna(low_high_conflict_delta) and float(low_high_conflict_delta) >= 0:
            rationale.append("low_high_conflict does not decrease")
        if pd.notna(visual_override_delta) and float(visual_override_delta) > 0:
            rationale.append("visual_residual_override increases")
        if (auc_delta is not None and auc_delta <= -0.01) or (pr_auc_delta is not None and pr_auc_delta <= -0.01):
            recommendation = "keep_lh_consistency_as_secondary_tradeoff_variant"
            rationale.append("AUC or PR-AUC drops noticeably despite some point-metric gains")

    return {
        "skeleton_name": skeleton_name,
        "lh_name": lh_name,
        "fixed_cases": fixed_count,
        "regressed_cases": regressed_count,
        "persistent_error_cases": persistent_count,
        "low_high_conflict_delta": None if pd.isna(low_high_conflict_delta) else int(low_high_conflict_delta),
        "visual_residual_override_delta": None if pd.isna(visual_override_delta) else int(visual_override_delta),
        "auc_delta": auc_delta,
        "pr_auc_delta": pr_auc_delta,
        "keep_lh_as_final_candidate": keep_as_final_candidate,
        "recommendation": recommendation,
        "rationale": rationale,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("NA").astype(str)
    header = "| " + " | ".join(safe_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(safe_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in safe_df.columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def build_report(
    output_dir: Path,
    metric_df: pd.DataFrame,
    failure_type_df: pd.DataFrame,
    error_overlap_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    regressed_df: pd.DataFrame,
    persistent_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
    visual_df: pd.DataFrame,
    recommendations: dict[str, object],
) -> str:
    lines = [
        "# Stage38 LH-Consistency Failure Comparison",
        "",
        "## Scope",
        "- This step does not train any model.",
        "- This step re-exports evidence and compares Step33-style failure analysis between two existing variants.",
        "",
        "## Compared Variants",
        f"- Skeleton: `{recommendations['skeleton_name']}`",
        f"- LH consistency: `{recommendations['lh_name']}`",
        "",
        "## Key Findings",
        f"- Fixed cases: `{recommendations['fixed_cases']}`",
        f"- Regressed cases: `{recommendations['regressed_cases']}`",
        f"- Persistent errors: `{recommendations['persistent_error_cases']}`",
        f"- Low-high conflict delta (`{recommendations['lh_name']} - {recommendations['skeleton_name']}`): `{recommendations['low_high_conflict_delta']}`",
        f"- Visual residual override delta (`{recommendations['lh_name']} - {recommendations['skeleton_name']}`): `{recommendations['visual_residual_override_delta']}`",
        f"- AUC delta: `{format_float(recommendations['auc_delta'])}`",
        f"- PR-AUC delta: `{format_float(recommendations['pr_auc_delta'])}`",
        f"- Recommendation: `{recommendations['recommendation']}`",
        "",
        "## Rationale",
    ]
    if recommendations["rationale"]:
        lines.extend([f"- {item}" for item in recommendations["rationale"]])
    else:
        lines.append("- No strong negative or positive trigger was detected from the implemented rules.")

    lines.extend(
        [
            "",
            "## Metric Comparison",
            "",
            markdown_table(metric_df),
            "",
            "## Failure Type Comparison",
            "",
            markdown_table(failure_type_df),
            "",
            "## Error Overlap Summary",
            "",
            f"- Total overlap rows: `{len(error_overlap_df.index)}`",
            f"- Fixed rows: `{len(fixed_df.index)}`",
            f"- Regressed rows: `{len(regressed_df.index)}`",
            f"- Persistent rows: `{len(persistent_df.index)}`",
            "",
            "## Low / High Conflict Comparison",
            "",
            markdown_table(conflict_df),
            "",
            "## Visual Override Comparison",
            "",
            markdown_table(visual_df),
            "",
            "## Fixed Cases Preview",
            "",
            markdown_table(fixed_df.head(20)),
            "",
            "## Regressed Cases Preview",
            "",
            markdown_table(regressed_df.head(20)),
            "",
            "## Persistent Error Preview",
            "",
            markdown_table(persistent_df.head(20)),
            "",
            "## Output Files",
            "",
            f"- `{output_dir / 'stage38_variant_metric_comparison.csv'}`",
            f"- `{output_dir / 'stage38_failure_type_comparison.csv'}`",
            f"- `{output_dir / 'stage38_error_overlap.csv'}`",
            f"- `{output_dir / 'stage38_fixed_cases.csv'}`",
            f"- `{output_dir / 'stage38_regressed_cases.csv'}`",
            f"- `{output_dir / 'stage38_persistent_error_cases.csv'}`",
            f"- `{output_dir / 'stage38_low_high_conflict_comparison.csv'}`",
            f"- `{output_dir / 'stage38_visual_override_comparison.csv'}`",
            f"- `{output_dir / 'stage38_recommendations.json'}`",
            f"- `{output_dir / 'stage38_lh_consistency_failure_comparison_report.md'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton = load_variant_bundle(
        evidence_dir=resolve(args.skeleton_evidence_dir),
        failure_dir=resolve(args.skeleton_failure_dir),
        variant_name=args.skeleton_name,
    )
    lh = load_variant_bundle(
        evidence_dir=resolve(args.lh_evidence_dir),
        failure_dir=resolve(args.lh_failure_dir),
        variant_name=args.lh_name,
    )

    metric_df = build_metric_comparison(skeleton, lh)
    failure_type_df = build_failure_type_comparison(skeleton, lh)
    error_overlap_df, fixed_df, regressed_df, persistent_df = build_error_overlap(skeleton, lh)
    conflict_df = build_conflict_comparison(skeleton, lh, fixed_df)
    visual_df = build_visual_override_comparison(skeleton, lh)
    recommendations = build_recommendations(
        metric_df=metric_df,
        conflict_df=conflict_df,
        visual_df=visual_df,
        fixed_df=fixed_df,
        regressed_df=regressed_df,
        persistent_df=persistent_df,
        skeleton_name=args.skeleton_name,
        lh_name=args.lh_name,
    )

    metric_df.to_csv(output_dir / "stage38_variant_metric_comparison.csv", index=False, encoding="utf-8")
    failure_type_df.to_csv(output_dir / "stage38_failure_type_comparison.csv", index=False, encoding="utf-8")
    error_overlap_df.to_csv(output_dir / "stage38_error_overlap.csv", index=False, encoding="utf-8")
    fixed_df.to_csv(output_dir / "stage38_fixed_cases.csv", index=False, encoding="utf-8")
    regressed_df.to_csv(output_dir / "stage38_regressed_cases.csv", index=False, encoding="utf-8")
    persistent_df.to_csv(output_dir / "stage38_persistent_error_cases.csv", index=False, encoding="utf-8")
    conflict_df.to_csv(output_dir / "stage38_low_high_conflict_comparison.csv", index=False, encoding="utf-8")
    visual_df.to_csv(output_dir / "stage38_visual_override_comparison.csv", index=False, encoding="utf-8")
    (output_dir / "stage38_recommendations.json").write_text(
        json.dumps(recommendations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "stage38_lh_consistency_failure_comparison_report.md").write_text(
        build_report(
            output_dir=output_dir,
            metric_df=metric_df,
            failure_type_df=failure_type_df,
            error_overlap_df=error_overlap_df,
            fixed_df=fixed_df,
            regressed_df=regressed_df,
            persistent_df=persistent_df,
            conflict_df=conflict_df,
            visual_df=visual_df,
            recommendations=recommendations,
        ),
        encoding="utf-8",
    )

    print(f"[Done] Wrote: {output_dir / 'stage38_variant_metric_comparison.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_failure_type_comparison.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_error_overlap.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_fixed_cases.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_regressed_cases.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_persistent_error_cases.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_low_high_conflict_comparison.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_visual_override_comparison.csv'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_recommendations.json'}")
    print(f"[Done] Wrote: {output_dir / 'stage38_lh_consistency_failure_comparison_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
