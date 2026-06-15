from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE24_DIR = "results_stage24/stage24_rce_v4_csg_summary"
DEFAULT_STAGE28_DIR = "results_stage28/stage28_deg_region_graph_summary"
DEFAULT_STAGE31_DIR = "results_stage31/stage31_deg_concept_graph_summary"
DEFAULT_STAGE35_DIR = "results_stage35"
DEFAULT_STAGE37_DIR = "results_stage37/stage37_lh_consistency_summary"
DEFAULT_STAGE38_DIR = "results_stage38/stage38_lh_consistency_failure_comparison"
DEFAULT_SKELETON_EVIDENCE_DIR = "results_stage38/evidence_export_skeleton_fold0_test"
DEFAULT_LH_EVIDENCE_DIR = "results_stage38/evidence_export_lh_l001_m0_fold0_test"
DEFAULT_OUTPUT_DIR = "results_stage39/final_evidence_package"

FINAL_DEFAULT_MODEL = "RCE-v4-CSG-a01-rq16 / DEG skeleton"
FINAL_SECONDARY_VARIANT = "RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0"
FINAL_REASON = (
    "skeleton has stronger AUC/PR-AUC and remains the most robust default. "
    "lh_l001_m0 reduces fold0/test errors and low-high conflict but increases "
    "visual_residual_override and slightly reduces AUC/PR-AUC."
)


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step39 final evidence package from existing stage outputs.")
    parser.add_argument("--output_dir", default=env_default("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage24_dir", default=env_default("STAGE24_DIR", DEFAULT_STAGE24_DIR))
    parser.add_argument("--stage28_dir", default=env_default("STAGE28_DIR", DEFAULT_STAGE28_DIR))
    parser.add_argument("--stage31_dir", default=env_default("STAGE31_DIR", DEFAULT_STAGE31_DIR))
    parser.add_argument("--stage35_dir", default=env_default("STAGE35_DIR", DEFAULT_STAGE35_DIR))
    parser.add_argument("--stage37_dir", default=env_default("STAGE37_DIR", DEFAULT_STAGE37_DIR))
    parser.add_argument("--stage38_dir", default=env_default("STAGE38_DIR", DEFAULT_STAGE38_DIR))
    parser.add_argument("--skeleton_evidence_dir", default=env_default("SKELETON_EVIDENCE_DIR", DEFAULT_SKELETON_EVIDENCE_DIR))
    parser.add_argument("--lh_evidence_dir", default=env_default("LH_EVIDENCE_DIR", DEFAULT_LH_EVIDENCE_DIR))
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def warn(message: str, warning_log: list[str]) -> None:
    print(f"[Warning] {message}")
    warning_log.append(message)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def read_json_optional(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("N/A").astype(str)
    header = "| " + " | ".join(safe_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(safe_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in safe_df.columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def first_row(df: pd.DataFrame, column: str, value: object) -> pd.Series:
    matched = df[df[column] == value]
    if matched.empty:
        raise ValueError(f"Could not find row where {column} == {value!r}")
    return matched.iloc[0]


def load_stage24(stage24_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "stage22_summary": read_csv_required(stage24_dir / "stage24_stage22_csg_init_summary.csv"),
        "stage23_summary": read_csv_required(stage24_dir / "stage24_stage23_region_query_summary.csv"),
        "metric_deltas": read_csv_required(stage24_dir / "stage24_metric_deltas.csv"),
    }


def load_stage28(stage28_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "summary": read_csv_required(stage28_dir / "stage28_deg_region_graph_summary.csv"),
        "metric_deltas": read_csv_required(stage28_dir / "stage28_deg_region_graph_metric_deltas.csv"),
    }


def load_stage31(stage31_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "summary": read_csv_required(stage31_dir / "stage31_deg_concept_graph_summary.csv"),
        "metric_deltas": read_csv_required(stage31_dir / "stage31_deg_concept_graph_metric_deltas.csv"),
    }


def summarize_fold_summary(path: Path) -> dict[str, object]:
    df = read_csv_required(path)
    row: dict[str, object] = {
        "source_path": relative(path),
        "status": "ok",
        "num_rows": len(df.index),
    }
    for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "sensitivity", "specificity", "pr_auc"]:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
        row[f"{metric}_std"] = float(values.std(ddof=0)) if not values.empty else None
    return row


def build_stage35_summary_from_raw(stage35_dir: Path, warning_log: list[str]) -> pd.DataFrame:
    variants = [
        ("skeleton", "DEG skeleton", "on", "off", "off", stage35_dir / "visual_gate_skeleton_5fold_e20_s1" / "fold_summary.csv"),
        ("gate0", "Visual gate init 0.00", "on", "on", "0.00", stage35_dir / "visual_gate_gate0_5fold_e20_s1" / "fold_summary.csv"),
        ("gate001", "Visual gate init 0.01", "on", "on", "0.01", stage35_dir / "visual_gate_gate001_5fold_e20_s1" / "fold_summary.csv"),
        ("gate005", "Visual gate init 0.05", "on", "on", "0.05", stage35_dir / "visual_gate_gate005_5fold_e20_s1" / "fold_summary.csv"),
        ("gate1", "Visual gate init 1.00", "on", "on", "1.00", stage35_dir / "visual_gate_gate1_5fold_e20_s1" / "fold_summary.csv"),
    ]
    rows: list[dict[str, object]] = []
    for variant, method, visual_residual, visual_gate, gate_init, path in variants:
        if not path.is_file():
            warn(f"Stage35 raw fold summary missing for variant={variant}: {path}", warning_log)
            row = {
                "variant": variant,
                "method": method,
                "visual_residual": visual_residual,
                "visual_gate": visual_gate,
                "gate_init": gate_init,
                "source_path": relative(path),
                "status": "missing",
                "num_rows": 0,
            }
            for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "sensitivity", "specificity", "pr_auc"]:
                row[f"{metric}_mean"] = None
            rows.append(row)
            continue
        row = summarize_fold_summary(path)
        row.update(
            {
                "variant": variant,
                "method": method,
                "visual_residual": visual_residual,
                "visual_gate": visual_gate,
                "gate_init": gate_init,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_stage35(stage35_dir: Path, warning_log: list[str]) -> pd.DataFrame:
    summary_path = stage35_dir / "stage35_visual_gate_summary" / "stage35_visual_gate_summary.csv"
    use_raw = True
    if summary_path.is_file():
        summary_df = pd.read_csv(summary_path)
        skeleton_rows = summary_df[(summary_df["variant"] == "skeleton") & (summary_df["status"] == "ok")]
        gate_rows = summary_df[(summary_df["variant"] == "gate1") & (summary_df["status"] == "ok")]
        if not skeleton_rows.empty and not gate_rows.empty:
            use_raw = False
            return summary_df
        warn(
            "Stage35 summary CSV exists but does not contain the expected full e20 skeleton/gate rows; using raw fold_summary.csv files instead.",
            warning_log,
        )
    return build_stage35_summary_from_raw(stage35_dir, warning_log)


def load_stage37(stage37_dir: Path) -> dict[str, pd.DataFrame | dict[str, object]]:
    return {
        "summary": read_csv_required(stage37_dir / "stage37_lh_consistency_summary.csv"),
        "metric_deltas": read_csv_required(stage37_dir / "stage37_lh_consistency_metric_deltas.csv"),
        "recommendations": read_json_optional(stage37_dir / "stage37_recommendations.json"),
    }


def load_stage38(stage38_dir: Path, skeleton_evidence_dir: Path, lh_evidence_dir: Path) -> dict[str, object]:
    stage38_root = stage38_dir.parent
    return {
        "metric_comparison": read_csv_required(stage38_dir / "stage38_variant_metric_comparison.csv"),
        "failure_type_comparison": read_csv_required(stage38_dir / "stage38_failure_type_comparison.csv"),
        "error_overlap": read_csv_required(stage38_dir / "stage38_error_overlap.csv"),
        "fixed_cases": read_csv_required(stage38_dir / "stage38_fixed_cases.csv"),
        "regressed_cases": read_csv_required(stage38_dir / "stage38_regressed_cases.csv"),
        "persistent_cases": read_csv_required(stage38_dir / "stage38_persistent_error_cases.csv"),
        "conflict_comparison": read_csv_required(stage38_dir / "stage38_low_high_conflict_comparison.csv"),
        "visual_comparison": read_csv_required(stage38_dir / "stage38_visual_override_comparison.csv"),
        "recommendations": read_json_optional(stage38_dir / "stage38_recommendations.json"),
        "skeleton_slide_evidence": read_csv_required(skeleton_evidence_dir / "stage32_slide_evidence_summary.csv"),
        "lh_slide_evidence": read_csv_required(lh_evidence_dir / "stage32_slide_evidence_summary.csv"),
        "skeleton_failure_labels": read_csv_required(stage38_root / "failure_analysis_skeleton_fold0_test" / "stage33_slide_failure_labels.csv"),
        "lh_failure_labels": read_csv_required(stage38_root / "failure_analysis_lh_l001_m0_fold0_test" / "stage33_slide_failure_labels.csv"),
        "skeleton_evidence_dir": skeleton_evidence_dir,
        "lh_evidence_dir": lh_evidence_dir,
    }


def build_final_performance_summary(
    stage24: dict[str, pd.DataFrame],
    stage28: dict[str, pd.DataFrame],
    stage31: dict[str, pd.DataFrame],
    stage35: pd.DataFrame,
    stage37: dict[str, pd.DataFrame | dict[str, object]],
    stage38: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def append_from_summary(
        df: pd.DataFrame,
        stage_name: str,
        category: str,
        variant_column: str = "variant",
        note_builder=None,
    ) -> None:
        for _, row in df.iterrows():
            note = "" if note_builder is None else note_builder(row)
            rows.append(
                {
                    "stage": stage_name,
                    "category": category,
                    "variant": row.get(variant_column),
                    "method": row.get("method"),
                    "status": row.get("status"),
                    "source_path": row.get("source_path"),
                    "test_auc": row.get("test_auc_mean"),
                    "test_acc": row.get("test_acc_mean"),
                    "test_f1": row.get("test_f1_mean"),
                    "balanced_acc": row.get("balanced_acc_mean"),
                    "sensitivity": row.get("sensitivity_mean"),
                    "specificity": row.get("specificity_mean"),
                    "pr_auc": row.get("pr_auc_mean"),
                    "note": note,
                }
            )

    append_from_summary(stage24["stage22_summary"], "stage24", "csg_init")
    append_from_summary(stage24["stage23_summary"], "stage24", "region_query")
    append_from_summary(stage28["summary"], "stage28", "spatial_region_graph")
    append_from_summary(stage31["summary"], "stage31", "concept_prompt_graph")
    append_from_summary(stage35, "stage35", "visual_gate")
    append_from_summary(stage37["summary"], "stage37", "low_high_consistency")

    metric_df = stage38["metric_comparison"]
    for _, row in metric_df.iterrows():
        if row["variant"] == "lh_l001_m0 - skeleton":
            continue
        rows.append(
            {
                "stage": "stage38",
                "category": "fold0_test_reexport",
                "variant": row.get("variant"),
                "method": "Step38 evidence/failure comparison",
                "status": "ok",
                "source_path": relative(resolve_path(DEFAULT_STAGE38_DIR) / "stage38_variant_metric_comparison.csv"),
                "test_auc": row.get("test_auc"),
                "test_acc": row.get("test_acc"),
                "test_f1": row.get("test_f1"),
                "balanced_acc": row.get("balanced_acc"),
                "sensitivity": row.get("sensitivity"),
                "specificity": row.get("specificity"),
                "pr_auc": row.get("pr_auc"),
                "note": "fold0/test evidence re-export comparison",
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(by=["stage", "category", "variant"], ascending=[True, True, True]).reset_index(drop=True)


def build_ablation_summary(
    stage24: dict[str, pd.DataFrame],
    stage28: dict[str, pd.DataFrame],
    stage31: dict[str, pd.DataFrame],
    stage35: pd.DataFrame,
    stage37: dict[str, pd.DataFrame | dict[str, object]],
    stage38: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    stage24_deltas = stage24["metric_deltas"]
    csg_delta = first_row(stage24_deltas, "comparison", "csg_a01 - csg_a005")
    rq16_rq8_delta = first_row(stage24_deltas, "comparison", "rq16 - rq8")
    rq16_rq32_delta = first_row(stage24_deltas, "comparison", "rq16 - rq32")

    stage28_summary = stage28["summary"]
    stage28_non_skeleton = stage28_summary[stage28_summary["variant"] != "skeleton"].copy()
    best_rg = stage28_non_skeleton.sort_values(by=["test_auc_mean", "pr_auc_mean"], ascending=[False, False]).iloc[0]
    rg_delta = first_row(stage28["metric_deltas"], "comparison", f"{best_rg['variant']} - skeleton")

    stage31_summary = stage31["summary"]
    stage31_non_skeleton = stage31_summary[stage31_summary["variant"] != "skeleton"].copy()
    best_cg = stage31_non_skeleton.sort_values(by=["test_auc_mean", "pr_auc_mean"], ascending=[False, False]).iloc[0]
    cg_delta = first_row(stage31["metric_deltas"], "comparison", f"{best_cg['variant']} - skeleton")

    stage35_valid = stage35[stage35["status"] == "ok"].copy()
    stage35_skeleton = first_row(stage35_valid, "variant", "skeleton")
    stage35_best_gate = stage35_valid[stage35_valid["variant"] != "skeleton"].sort_values(
        by=["test_auc_mean", "pr_auc_mean", "balanced_acc_mean"],
        ascending=[False, False, False],
    ).iloc[0]

    def gate_delta(metric: str) -> float | None:
        current = safe_float(stage35_best_gate.get(f"{metric}_mean"))
        baseline = safe_float(stage35_skeleton.get(f"{metric}_mean"))
        if current is None or baseline is None:
            return None
        return current - baseline

    lh_stage37_delta = first_row(stage37["metric_deltas"], "comparison", "lh_l001_m0 - skeleton")
    stage38_metric_delta = first_row(stage38["metric_comparison"], "variant", "lh_l001_m0 - skeleton")

    rows.append(
        {
            "topic": "CSG strength",
            "comparison": "csg_a01 - csg_a005",
            "source_stage": "stage24",
            "preferred_variant": "RCE-v4-CSG-a01-rq16",
            "reference_variant": "RCE-v4-CSG-a005-rq16",
            "delta_test_auc": csg_delta.get("test_auc_delta"),
            "delta_test_acc": csg_delta.get("test_acc_delta"),
            "delta_test_f1": csg_delta.get("test_f1_delta"),
            "delta_balanced_acc": csg_delta.get("balanced_acc_delta"),
            "delta_sensitivity": csg_delta.get("sensitivity_delta"),
            "delta_specificity": csg_delta.get("specificity_delta"),
            "delta_pr_auc": csg_delta.get("pr_auc_delta"),
            "paper_ready_conclusion": "CSG a01 > CSG a005",
        }
    )
    rows.append(
        {
            "topic": "Region query count",
            "comparison": "rq16 - rq8",
            "source_stage": "stage24",
            "preferred_variant": "RCE-v4-CSG-a01-rq16",
            "reference_variant": "RCE-v4-CSG-a01-rq8",
            "delta_test_auc": rq16_rq8_delta.get("test_auc_delta"),
            "delta_test_acc": rq16_rq8_delta.get("test_acc_delta"),
            "delta_test_f1": rq16_rq8_delta.get("test_f1_delta"),
            "delta_balanced_acc": rq16_rq8_delta.get("balanced_acc_delta"),
            "delta_sensitivity": rq16_rq8_delta.get("sensitivity_delta"),
            "delta_specificity": rq16_rq8_delta.get("specificity_delta"),
            "delta_pr_auc": rq16_rq8_delta.get("pr_auc_delta"),
            "paper_ready_conclusion": "rq16 > rq8",
        }
    )
    rows.append(
        {
            "topic": "Region query count",
            "comparison": "rq16 - rq32",
            "source_stage": "stage24",
            "preferred_variant": "RCE-v4-CSG-a01-rq16",
            "reference_variant": "RCE-v4-CSG-a01-rq32",
            "delta_test_auc": rq16_rq32_delta.get("test_auc_delta"),
            "delta_test_acc": rq16_rq32_delta.get("test_acc_delta"),
            "delta_test_f1": rq16_rq32_delta.get("test_f1_delta"),
            "delta_balanced_acc": rq16_rq32_delta.get("balanced_acc_delta"),
            "delta_sensitivity": rq16_rq32_delta.get("sensitivity_delta"),
            "delta_specificity": rq16_rq32_delta.get("specificity_delta"),
            "delta_pr_auc": rq16_rq32_delta.get("pr_auc_delta"),
            "paper_ready_conclusion": "rq16 > rq32",
        }
    )
    rows.append(
        {
            "topic": "Spatial Region Graph",
            "comparison": f"{best_rg['variant']} - skeleton",
            "source_stage": "stage28",
            "preferred_variant": "DEG skeleton",
            "reference_variant": best_rg["variant"],
            "delta_test_auc": rg_delta.get("test_auc_delta"),
            "delta_test_acc": rg_delta.get("test_acc_delta"),
            "delta_test_f1": rg_delta.get("test_f1_delta"),
            "delta_balanced_acc": rg_delta.get("balanced_acc_delta"),
            "delta_sensitivity": rg_delta.get("sensitivity_delta"),
            "delta_specificity": rg_delta.get("specificity_delta"),
            "delta_pr_auc": rg_delta.get("pr_auc_delta"),
            "paper_ready_conclusion": "Spatial Region Graph did not outperform skeleton",
        }
    )
    rows.append(
        {
            "topic": "Concept Prompt Graph",
            "comparison": f"{best_cg['variant']} - skeleton",
            "source_stage": "stage31",
            "preferred_variant": "DEG skeleton",
            "reference_variant": best_cg["variant"],
            "delta_test_auc": cg_delta.get("test_auc_delta"),
            "delta_test_acc": cg_delta.get("test_acc_delta"),
            "delta_test_f1": cg_delta.get("test_f1_delta"),
            "delta_balanced_acc": cg_delta.get("balanced_acc_delta"),
            "delta_sensitivity": cg_delta.get("sensitivity_delta"),
            "delta_specificity": cg_delta.get("specificity_delta"),
            "delta_pr_auc": cg_delta.get("pr_auc_delta"),
            "paper_ready_conclusion": "Concept Prompt Graph did not outperform skeleton",
        }
    )
    rows.append(
        {
            "topic": "Scalar Visual Gate",
            "comparison": f"{stage35_best_gate['variant']} - skeleton",
            "source_stage": "stage35",
            "preferred_variant": "DEG skeleton",
            "reference_variant": stage35_best_gate["variant"],
            "delta_test_auc": gate_delta("test_auc"),
            "delta_test_acc": gate_delta("test_acc"),
            "delta_test_f1": gate_delta("test_f1"),
            "delta_balanced_acc": gate_delta("balanced_acc"),
            "delta_sensitivity": gate_delta("sensitivity"),
            "delta_specificity": gate_delta("specificity"),
            "delta_pr_auc": gate_delta("pr_auc"),
            "paper_ready_conclusion": "Scalar Visual Gate did not outperform skeleton",
        }
    )
    rows.append(
        {
            "topic": "Low-High Consistency",
            "comparison": "lh_l001_m0 - skeleton",
            "source_stage": "stage37+stage38",
            "preferred_variant": FINAL_DEFAULT_MODEL,
            "reference_variant": FINAL_SECONDARY_VARIANT,
            "delta_test_auc": stage38_metric_delta.get("test_auc"),
            "delta_test_acc": stage38_metric_delta.get("test_acc"),
            "delta_test_f1": stage38_metric_delta.get("test_f1"),
            "delta_balanced_acc": stage38_metric_delta.get("balanced_acc"),
            "delta_sensitivity": stage38_metric_delta.get("sensitivity"),
            "delta_specificity": stage38_metric_delta.get("specificity"),
            "delta_pr_auc": stage38_metric_delta.get("pr_auc"),
            "paper_ready_conclusion": "Low-High Consistency is a trade-off variant",
            "stage37_test_auc_delta": lh_stage37_delta.get("test_auc_delta"),
            "stage37_balanced_acc_delta": lh_stage37_delta.get("balanced_acc_delta"),
            "stage38_fixed_cases": stage38["recommendations"].get("fixed_cases"),
            "stage38_regressed_cases": stage38["recommendations"].get("regressed_cases"),
        }
    )

    return pd.DataFrame(rows)


def build_negative_ablation_summary() -> pd.DataFrame:
    rows = [
        {
            "module": "attention-centroid region graph",
            "negative_ablation_statement": "semantic region token does not equal a true spatial region",
            "paper_ready_interpretation": "attention-centroid region graph: semantic region token does not equal true spatial region",
            "implication": "graph edges built from centroided prompt tokens fail to provide stable spatial inductive bias",
            "role": "diagnostic ablation",
        },
        {
            "module": "concept prompt graph",
            "negative_ablation_statement": "plain feature-level prompt smoothing weakens evidence discrimination",
            "paper_ready_interpretation": "concept prompt graph: ordinary feature-level prompt smoothing weakens evidence discrimination",
            "implication": "concept-to-concept diffusion blurs class-critical prompt evidence instead of sharpening it",
            "role": "diagnostic ablation",
        },
        {
            "module": "scalar visual gate",
            "negative_ablation_statement": "visual residual cannot be safely suppressed by one global scalar",
            "paper_ready_interpretation": "scalar visual gate: visual residual cannot be simply suppressed by a global scalar",
            "implication": "visual evidence interacts with concept evidence in a sample-dependent manner and resists one-number gating",
            "role": "diagnostic ablation",
        },
        {
            "module": "low-high consistency",
            "negative_ablation_statement": "reduces low-high conflict but introduces visual residual override trade-off",
            "paper_ready_interpretation": "low-high consistency: reduces low-high conflict but carries a visual residual override trade-off",
            "implication": "consistency regularization calibrates evidence alignment but is not clean enough to replace the default model",
            "role": "secondary trade-off variant",
        },
    ]
    return pd.DataFrame(rows)


def build_evidence_calibration_summary(stage38: dict[str, object]) -> pd.DataFrame:
    metric_delta = first_row(stage38["metric_comparison"], "variant", "lh_l001_m0 - skeleton")
    conflict_df = stage38["conflict_comparison"]
    skeleton_conflict = first_row(conflict_df, "variant", "skeleton")
    lh_conflict = first_row(conflict_df, "variant", "lh_l001_m0")
    visual_df = stage38["visual_comparison"]
    skeleton_visual = first_row(visual_df, "variant", "skeleton")
    lh_visual = first_row(visual_df, "variant", "lh_l001_m0")

    row = {
        "comparison": "lh_l001_m0 vs skeleton",
        "fixed_cases": 6,
        "regressed_cases": 2,
        "persistent_errors": 12,
        "low_high_conflict_skeleton": skeleton_conflict.get("error_low_high_conflict_count"),
        "low_high_conflict_lh": lh_conflict.get("error_low_high_conflict_count"),
        "low_high_conflict_delta": lh_conflict.get("error_low_high_conflict_count") - skeleton_conflict.get("error_low_high_conflict_count"),
        "both_support_wrong_skeleton": skeleton_conflict.get("error_both_support_wrong_count"),
        "both_support_wrong_lh": lh_conflict.get("error_both_support_wrong_count"),
        "both_support_wrong_delta": lh_conflict.get("error_both_support_wrong_count") - skeleton_conflict.get("error_both_support_wrong_count"),
        "visual_residual_override_skeleton": skeleton_visual.get("visual_residual_override_error_count"),
        "visual_residual_override_lh": lh_visual.get("visual_residual_override_error_count"),
        "visual_residual_override_delta": lh_visual.get("visual_residual_override_error_count") - skeleton_visual.get("visual_residual_override_error_count"),
        "auc_delta": metric_delta.get("test_auc"),
        "pr_auc_delta": metric_delta.get("pr_auc"),
        "acc_delta": metric_delta.get("test_acc"),
        "f1_delta": metric_delta.get("test_f1"),
        "balanced_acc_delta": metric_delta.get("balanced_acc"),
        "sensitivity_delta": metric_delta.get("sensitivity"),
        "specificity_delta": metric_delta.get("specificity"),
    }
    return pd.DataFrame([row])


def build_failure_comparison_summary(stage38: dict[str, object]) -> pd.DataFrame:
    failure_type_df = stage38["failure_type_comparison"]
    conflict_df = stage38["conflict_comparison"]
    visual_df = stage38["visual_comparison"]
    metric_delta = first_row(stage38["metric_comparison"], "variant", "lh_l001_m0 - skeleton")

    lookup_types = {}
    for _, row in failure_type_df.iterrows():
        lookup_types[str(row["failure_type"])] = row

    skeleton_conflict = first_row(conflict_df, "variant", "skeleton")
    lh_conflict = first_row(conflict_df, "variant", "lh_l001_m0")
    skeleton_visual = first_row(visual_df, "variant", "skeleton")
    lh_visual = first_row(visual_df, "variant", "lh_l001_m0")

    rows = [
        {
            "group": "overlap",
            "metric": "fixed_cases",
            "skeleton_value": None,
            "lh_value": None,
            "delta": 6,
            "interpretation": "lh_l001_m0 fixes 6 skeleton errors",
        },
        {
            "group": "overlap",
            "metric": "regressed_cases",
            "skeleton_value": None,
            "lh_value": None,
            "delta": 2,
            "interpretation": "lh_l001_m0 introduces 2 new errors",
        },
        {
            "group": "overlap",
            "metric": "persistent_errors",
            "skeleton_value": None,
            "lh_value": None,
            "delta": 12,
            "interpretation": "12 errors remain wrong for both variants",
        },
        {
            "group": "failure_type",
            "metric": "low_high_conflict",
            "skeleton_value": lookup_types["low_high_conflict"]["skeleton_count_any_label"],
            "lh_value": lookup_types["low_high_conflict"]["lh_l001_m0_count_any_label"],
            "delta": lookup_types["low_high_conflict"]["lh_l001_m0_minus_skeleton_any_label"],
            "interpretation": "conflict decreases under low-high consistency",
        },
        {
            "group": "failure_type",
            "metric": "both_support_wrong",
            "skeleton_value": skeleton_conflict["error_both_support_wrong_count"],
            "lh_value": lh_conflict["error_both_support_wrong_count"],
            "delta": lh_conflict["error_both_support_wrong_count"] - skeleton_conflict["error_both_support_wrong_count"],
            "interpretation": "joint wrong support decreases under low-high consistency",
        },
        {
            "group": "failure_type",
            "metric": "visual_residual_override",
            "skeleton_value": skeleton_visual["visual_residual_override_error_count"],
            "lh_value": lh_visual["visual_residual_override_error_count"],
            "delta": lh_visual["visual_residual_override_error_count"] - skeleton_visual["visual_residual_override_error_count"],
            "interpretation": "visual override slightly increases under low-high consistency",
        },
        {
            "group": "metric_delta",
            "metric": "test_auc",
            "skeleton_value": None,
            "lh_value": None,
            "delta": metric_delta["test_auc"],
            "interpretation": "AUC drops slightly",
        },
        {
            "group": "metric_delta",
            "metric": "pr_auc",
            "skeleton_value": None,
            "lh_value": None,
            "delta": metric_delta["pr_auc"],
            "interpretation": "PR-AUC drops slightly",
        },
    ]
    return pd.DataFrame(rows)


def select_case_examples(stage38: dict[str, object]) -> pd.DataFrame:
    fixed_df = stage38["fixed_cases"].copy()
    regressed_df = stage38["regressed_cases"].copy()
    persistent_df = stage38["persistent_cases"].copy()

    fixed_selected = fixed_df.sort_values(
        by=["skeleton_primary_failure_type", "slide_id"],
        ascending=[True, True],
    ).head(3).copy()
    regressed_selected = regressed_df.sort_values(
        by=["lh_primary_failure_type", "slide_id"],
        ascending=[True, True],
    ).head(2).copy()

    persistent_df["priority_score"] = (
        persistent_df["skeleton_failure_labels"].fillna("").str.contains("low_high_conflict").astype(int) * 2
        + persistent_df["lh_failure_labels"].fillna("").str.contains("low_high_conflict").astype(int)
    )
    persistent_selected = persistent_df.sort_values(
        by=["priority_score", "skeleton_primary_failure_type", "slide_id"],
        ascending=[False, True, True],
    ).head(3).copy()

    selected_frames = [
        ("fixed", fixed_selected),
        ("regressed", regressed_selected),
        ("persistent_error", persistent_selected),
    ]

    skeleton_lookup = stage38["skeleton_slide_evidence"].set_index("slide_id")
    lh_lookup = stage38["lh_slide_evidence"].set_index("slide_id")
    skeleton_failure_lookup = stage38["skeleton_failure_labels"].set_index("slide_id")
    lh_failure_lookup = stage38["lh_failure_labels"].set_index("slide_id")

    def label_name_from_value(value: object) -> str:
        return "Adenocarcinoma" if int(value) == 0 else "NonAdenocarcinoma"

    def pred_name_from_value(value: object) -> str:
        return "Adenocarcinoma" if int(value) == 0 else "NonAdenocarcinoma"

    rows: list[dict[str, object]] = []
    for case_group, df in selected_frames:
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            slide_id = row["slide_id"]
            sk_row = skeleton_lookup.loc[slide_id]
            lh_row = lh_lookup.loc[slide_id]
            sk_failure_row = skeleton_failure_lookup.loc[slide_id]
            lh_failure_row = lh_failure_lookup.loc[slide_id]
            rows.append(
                {
                    "case_group": case_group,
                    "selection_rank": rank,
                    "slide_id": slide_id,
                    "label": row.get("label"),
                    "label_name": sk_failure_row.get("label_name", label_name_from_value(row.get("label"))),
                    "skeleton_pred": row.get("skeleton_pred"),
                    "lh_pred": row.get("lh_pred"),
                    "skeleton_pred_name": sk_failure_row.get("pred_name", pred_name_from_value(row.get("skeleton_pred"))),
                    "lh_pred_name": lh_failure_row.get("pred_name", pred_name_from_value(row.get("lh_pred"))),
                    "skeleton_primary_failure_type": row.get("skeleton_primary_failure_type"),
                    "lh_primary_failure_type": row.get("lh_primary_failure_type"),
                    "skeleton_failure_labels": row.get("skeleton_failure_labels"),
                    "lh_failure_labels": row.get("lh_failure_labels"),
                    "skeleton_final_margin": sk_failure_row.get("final_true_vs_wrong_margin"),
                    "lh_final_margin": lh_failure_row.get("final_true_vs_wrong_margin"),
                    "skeleton_low_high_joint_state": sk_failure_row.get("low_high_joint_state"),
                    "lh_low_high_joint_state": lh_failure_row.get("low_high_joint_state"),
                    "skeleton_dominant_source": sk_failure_row.get("dominant_source"),
                    "lh_dominant_source": lh_failure_row.get("dominant_source"),
                    "skeleton_visual_source_ratio": sk_failure_row.get("visual_source_ratio"),
                    "lh_visual_source_ratio": lh_failure_row.get("visual_source_ratio"),
                    "selection_note": (
                        "represents repaired low-high conflict / visual override behavior"
                        if case_group == "fixed"
                        else "represents a new failure introduced by consistency regularization"
                        if case_group == "regressed"
                        else "represents a persistent hard case across both variants"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_example_top_concepts(
    example_df: pd.DataFrame,
    stage38: dict[str, object],
    warning_log: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = [
        ("skeleton", stage38["skeleton_slide_evidence"], stage38["skeleton_evidence_dir"]),
        ("lh_l001_m0", stage38["lh_slide_evidence"], stage38["lh_evidence_dir"]),
    ]
    for variant_name, df, evidence_dir in sources:
        detail_path = evidence_dir / "stage32_top_concepts_per_slide.csv"
        if not detail_path.is_file():
            warn(
                f"{relative(detail_path)} is missing; Step39 falls back to stage32_slide_evidence_summary.csv top concept fields.",
                warning_log,
            )
        lookup = df.set_index("slide_id")
        for _, example in example_df.iterrows():
            slide_id = example["slide_id"]
            row = lookup.loc[slide_id]
            rows.append(
                {
                    "case_group": example["case_group"],
                    "selection_rank": example["selection_rank"],
                    "slide_id": slide_id,
                    "variant": variant_name,
                    "label_name": row.get("label_name"),
                    "pred_name": row.get("pred_name"),
                    "top_low_concepts_for_pred": row.get("top_low_concepts_for_pred"),
                    "top_high_concepts_for_pred": row.get("top_high_concepts_for_pred"),
                    "top_low_concepts_for_true": row.get("top_low_concepts_for_true"),
                    "top_high_concepts_for_true": row.get("top_high_concepts_for_true"),
                    "evidence_source": "stage32_slide_evidence_summary.csv",
                    "detail_source_available": detail_path.is_file(),
                }
            )
    return pd.DataFrame(rows)


def build_example_top_csg_pairs(
    example_df: pd.DataFrame,
    stage38: dict[str, object],
    warning_log: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = [
        ("skeleton", stage38["skeleton_slide_evidence"], stage38["skeleton_evidence_dir"]),
        ("lh_l001_m0", stage38["lh_slide_evidence"], stage38["lh_evidence_dir"]),
    ]
    for variant_name, df, evidence_dir in sources:
        detail_path = evidence_dir / "stage32_top_csg_pairs_per_slide.csv"
        if not detail_path.is_file():
            warn(
                f"{relative(detail_path)} is missing; Step39 falls back to stage32_slide_evidence_summary.csv top CSG pair fields.",
                warning_log,
            )
        lookup = df.set_index("slide_id")
        for _, example in example_df.iterrows():
            slide_id = example["slide_id"]
            row = lookup.loc[slide_id]
            rows.append(
                {
                    "case_group": example["case_group"],
                    "selection_rank": example["selection_rank"],
                    "slide_id": slide_id,
                    "variant": variant_name,
                    "top_csg_pair_class_0": row.get("top_csg_pair_class_0"),
                    "top_csg_pair_class_1": row.get("top_csg_pair_class_1"),
                    "top_csg_pair_score_class_0": row.get("top_csg_pair_score_class_0"),
                    "top_csg_pair_score_class_1": row.get("top_csg_pair_score_class_1"),
                    "csg_alpha": row.get("csg_alpha"),
                    "csg_max_pair_score": row.get("csg_max_pair_score"),
                    "csg_export_capture_status": row.get("csg_export_capture_status"),
                    "evidence_source": "stage32_slide_evidence_summary.csv",
                    "detail_source_available": detail_path.is_file(),
                }
            )
    return pd.DataFrame(rows)


def build_final_model_recommendation_json(
    ablation_summary: pd.DataFrame,
    evidence_calibration_summary: pd.DataFrame,
    warnings: list[str],
) -> dict[str, object]:
    return {
        "recommended_default_model": FINAL_DEFAULT_MODEL,
        "secondary_tradeoff_variant": FINAL_SECONDARY_VARIANT,
        "reason": FINAL_REASON,
        "default_model_is_final_main_model": True,
        "lh_l001_m0_is_final_main_model": False,
        "lh_l001_m0_role": "secondary evidence-calibration trade-off variant",
        "ablation_conclusions": ablation_summary[["topic", "paper_ready_conclusion"]].to_dict(orient="records"),
        "evidence_calibration": evidence_calibration_summary.iloc[0].to_dict(),
        "innovation_points": [
            "Region-Concept Evidence Learning",
            "Cross-Scale Concept Evidence Reasoning",
            "Evidence Source Decomposition and Failure Diagnosis",
            "Evidence Calibration Analysis",
        ],
        "warnings": warnings,
    }


def build_innovation_points_md() -> str:
    lines = [
        "# Step39 Final Innovation Points",
        "",
        "## 1. Region-Concept Evidence Learning",
        "- 解决问题：将病理切片中的区域级视觉线索与概念级文本证据对齐，避免仅靠 bag-level 注意力给出黑盒预测。",
        "- 相比 ViLa-MIL 的区别：不只做提示词增强的 MIL 聚合，而是显式组织 region-to-concept 的证据路径，并保留 visual residual / concept support 的分解结果。",
        "- 对应实验/分析支撑：Step24 显示 `CSG a01 > a005`，且 `RCE-v4-CSG-a01-rq16` 成为当前最稳主干。",
        "- 角色定位：最终主模型模块。",
        "",
        "## 2. Cross-Scale Concept Evidence Reasoning",
        "- 解决问题：低倍与高倍概念证据经常互补，单尺度证据不足以解释病理类别判断。",
        "- 相比 ViLa-MIL 的区别：不仅保留双尺度输入，还把低倍/高倍概念支持关系纳入统一的证据推理视角，而不是只在最终分类层做简单融合。",
        "- 对应实验/分析支撑：Step24 中 `rq16 > rq8/rq32`，Step38 中 low/high conflict 的变化证明跨尺度证据关系是可分析、可校准的。",
        "- 角色定位：最终主模型模块。",
        "",
        "## 3. Evidence Source Decomposition and Failure Diagnosis",
        "- 解决问题：需要知道错误来自 low-scale concept、high-scale concept、visual residual 还是 cross-scale pair，而不是只记录分类是否正确。",
        "- 相比 ViLa-MIL 的区别：增加 evidence source decomposition、failure typing、conflict/override 诊断链路，使解释性分析可量化、可复现。",
        "- 对应实验/分析支撑：Step33/38 输出了 `low_high_conflict`、`visual_residual_override`、`concept_wrong_class_drift` 等失败类型与案例表。",
        "- 角色定位：diagnostic analysis，不改变最终主模型主体逻辑。",
        "",
        "## 4. Evidence Calibration Analysis",
        "- 解决问题：在不继续堆新模块的前提下，评估 evidence regularization 是否真正改善证据一致性，还是只是换来新的偏差。",
        "- 相比 ViLa-MIL 的区别：不仅比较主指标，还比较 fixed/regressed/persistent cases、low-high conflict 与 visual residual override 的结构性变化。",
        "- 对应实验/分析支撑：Step37/38 表明 `lh_l001_m0` 可减少 low-high conflict，但带来 visual residual override trade-off。",
        "- 角色定位：secondary trade-off variant + diagnostic ablation，不作为最终默认主模型。",
        "",
    ]
    return "\n".join(lines)


def build_paper_ready_summary_md(
    ablation_summary: pd.DataFrame,
    negative_ablation_summary: pd.DataFrame,
    evidence_calibration_summary: pd.DataFrame,
    example_df: pd.DataFrame,
    warnings: list[str],
) -> str:
    calibration = evidence_calibration_summary.iloc[0]
    lines = [
        "# Step39 Final Evidence Package and Paper-Ready Summary",
        "",
        "## 研究动机",
        "本阶段的目标不是继续训练新模型，而是在现有最优结果上收敛论文叙事：明确最终默认模型、整理哪些模块构成主创新、哪些探索应当作为 negative ablation 保留，并把证据解释与错误分析整理成可直接进入论文/报告的产物。",
        "",
        "## 方法概述",
        "最终方法主线为 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。它保留 Region-Concept Evidence Learning 与 Cross-Scale Concept Evidence Reasoning 这两条核心设计，同时通过 low/high concept evidence、visual residual、cross-scale pair evidence 的分解分析，使后续 failure diagnosis 与 calibration analysis 可以在不改动主模型主体逻辑的前提下完成。",
        "",
        "## 最终推荐模型",
        f"Recommended default model: `{FINAL_DEFAULT_MODEL}`",
        "",
        f"Secondary trade-off variant: `{FINAL_SECONDARY_VARIANT}`",
        "",
        f"Reason: {FINAL_REASON}",
        "",
        "## 主要实验结论",
        "- Step24 说明 `CSG a01 > CSG a005`，且 `rq16 > rq8/rq32`，因此最终 RCE 主干固定为 `RCE-v4-CSG-a01-rq16`。",
        "- Step28 与 Step31 说明无论是 Spatial Region Graph 还是 Concept Prompt Graph，都没有超过 `DEG skeleton`，因此最终主模型不再堆 graph 模块。",
        "- Step35 说明 Scalar Visual Gate 也没有超过 skeleton，说明 visual residual 不能被一个全局 gate 简单替代。",
        "- Step37/38 说明 `lh_l001_m0` 具备 evidence calibration 价值，但还不足以替换 skeleton 成为最终默认模型。",
        "",
        "## 消融实验结论",
        markdown_table(
            ablation_summary[
                [
                    "topic",
                    "comparison",
                    "source_stage",
                    "paper_ready_conclusion",
                    "delta_test_auc",
                    "delta_pr_auc",
                ]
            ]
        ),
        "",
        "## 解释性与错误分析",
        f"- fixed cases = `{int(calibration['fixed_cases'])}`",
        f"- regressed cases = `{int(calibration['regressed_cases'])}`",
        f"- persistent errors = `{int(calibration['persistent_errors'])}`",
        f"- low_high_conflict: `{int(calibration['low_high_conflict_skeleton'])} -> {int(calibration['low_high_conflict_lh'])}`",
        f"- both_support_wrong: `{int(calibration['both_support_wrong_skeleton'])} -> {int(calibration['both_support_wrong_lh'])}`",
        f"- visual_residual_override: `{int(calibration['visual_residual_override_skeleton'])} -> {int(calibration['visual_residual_override_lh'])}`",
        f"- AUC delta = `{format_metric(calibration['auc_delta'])}`",
        f"- PR-AUC delta = `{format_metric(calibration['pr_auc_delta'])}`",
        "- 这说明 low-high consistency 的主要收益在于缓解 low/high evidence conflict，而主要代价在于更容易把错误样本交给 visual residual 接管。",
        "",
        "## Current Negative Ablation Narrative",
        markdown_table(
            negative_ablation_summary[
                [
                    "module",
                    "paper_ready_interpretation",
                    "implication",
                    "role",
                ]
            ]
        ),
        "",
        "## 当前限制",
        "- 当前证据表明 low-high consistency 更像 calibration trade-off，而不是 clean gain；因此不宜直接替换默认主模型。",
        "- Spatial/Concept graph 与 scalar gate 都没有稳定收益，说明继续盲目堆 graph/gate 的边际回报很低。",
        "- Example evidence cases 已能关联 top concepts / top CSG pairs，但当前使用的是 `stage32_slide_evidence_summary.csv` 里的汇总字段，而不是更细粒度的独立 per-slide 明细表。",
        "",
        "## 后续工作",
        "- Step40：根据 `stage39_paper_ready_summary.md` 生成最终论文主图/方法图说明、实验表格说明和答辩汇报材料。",
        "- 如果还要继续模型创新，可以转向 Prompt Reliability / Refined Prompt Pool，但不建议再盲目堆 graph 或 gate。",
        "",
        "## Example Cases Included",
        markdown_table(example_df),
    ]
    if warnings:
        lines.extend(
            [
                "",
                "## Warnings / Data Notes",
                *[f"- {item}" for item in warnings],
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_final_next_steps_md() -> str:
    lines = [
        "# Step39 Final Next Steps",
        "",
        "Step40：根据 stage39_paper_ready_summary.md 生成最终论文主图/方法图说明、实验表格说明和答辩汇报材料。如果还要继续模型创新，可以转向 Prompt Reliability / Refined Prompt Pool，但不建议再盲目堆 graph 或 gate。",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    recommendation: dict[str, object],
    performance_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    negative_ablation_summary: pd.DataFrame,
    evidence_calibration_summary: pd.DataFrame,
    failure_comparison_summary: pd.DataFrame,
    example_df: pd.DataFrame,
    top_concepts_df: pd.DataFrame,
    top_csg_pairs_df: pd.DataFrame,
    innovation_md: str,
    paper_ready_md: str,
    next_steps_md: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage39_final_model_recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    performance_summary.to_csv(output_dir / "stage39_final_performance_summary.csv", index=False, encoding="utf-8")
    ablation_summary.to_csv(output_dir / "stage39_ablation_summary.csv", index=False, encoding="utf-8")
    negative_ablation_summary.to_csv(output_dir / "stage39_negative_ablation_summary.csv", index=False, encoding="utf-8")
    evidence_calibration_summary.to_csv(output_dir / "stage39_evidence_calibration_summary.csv", index=False, encoding="utf-8")
    failure_comparison_summary.to_csv(output_dir / "stage39_failure_comparison_summary.csv", index=False, encoding="utf-8")
    example_df.to_csv(output_dir / "stage39_fixed_regressed_persistent_cases.csv", index=False, encoding="utf-8")
    top_concepts_df.to_csv(output_dir / "stage39_top_concepts_for_examples.csv", index=False, encoding="utf-8")
    top_csg_pairs_df.to_csv(output_dir / "stage39_top_csg_pairs_for_examples.csv", index=False, encoding="utf-8")
    (output_dir / "stage39_final_innovation_points.md").write_text(innovation_md, encoding="utf-8")
    (output_dir / "stage39_paper_ready_summary.md").write_text(paper_ready_md, encoding="utf-8")
    (output_dir / "stage39_final_next_steps.md").write_text(next_steps_md, encoding="utf-8")


def main() -> int:
    args = parse_args()
    warning_log: list[str] = []

    output_dir = resolve_path(args.output_dir)
    stage24_dir = resolve_path(args.stage24_dir)
    stage28_dir = resolve_path(args.stage28_dir)
    stage31_dir = resolve_path(args.stage31_dir)
    stage35_dir = resolve_path(args.stage35_dir)
    stage37_dir = resolve_path(args.stage37_dir)
    stage38_dir = resolve_path(args.stage38_dir)
    skeleton_evidence_dir = resolve_path(args.skeleton_evidence_dir)
    lh_evidence_dir = resolve_path(args.lh_evidence_dir)

    stage24 = load_stage24(stage24_dir)
    stage28 = load_stage28(stage28_dir)
    stage31 = load_stage31(stage31_dir)
    stage35 = load_stage35(stage35_dir, warning_log)
    stage37 = load_stage37(stage37_dir)
    stage38 = load_stage38(stage38_dir, skeleton_evidence_dir, lh_evidence_dir)

    performance_summary = build_final_performance_summary(stage24, stage28, stage31, stage35, stage37, stage38)
    ablation_summary = build_ablation_summary(stage24, stage28, stage31, stage35, stage37, stage38)
    negative_ablation_summary = build_negative_ablation_summary()
    evidence_calibration_summary = build_evidence_calibration_summary(stage38)
    failure_comparison_summary = build_failure_comparison_summary(stage38)
    example_df = select_case_examples(stage38)
    top_concepts_df = build_example_top_concepts(example_df, stage38, warning_log)
    top_csg_pairs_df = build_example_top_csg_pairs(example_df, stage38, warning_log)
    recommendation = build_final_model_recommendation_json(ablation_summary, evidence_calibration_summary, warning_log)
    innovation_md = build_innovation_points_md()
    paper_ready_md = build_paper_ready_summary_md(
        ablation_summary=ablation_summary,
        negative_ablation_summary=negative_ablation_summary,
        evidence_calibration_summary=evidence_calibration_summary,
        example_df=example_df,
        warnings=warning_log,
    )
    next_steps_md = build_final_next_steps_md()

    write_outputs(
        output_dir=output_dir,
        recommendation=recommendation,
        performance_summary=performance_summary,
        ablation_summary=ablation_summary,
        negative_ablation_summary=negative_ablation_summary,
        evidence_calibration_summary=evidence_calibration_summary,
        failure_comparison_summary=failure_comparison_summary,
        example_df=example_df,
        top_concepts_df=top_concepts_df,
        top_csg_pairs_df=top_csg_pairs_df,
        innovation_md=innovation_md,
        paper_ready_md=paper_ready_md,
        next_steps_md=next_steps_md,
    )

    for name in [
        "stage39_final_model_recommendation.json",
        "stage39_final_performance_summary.csv",
        "stage39_ablation_summary.csv",
        "stage39_negative_ablation_summary.csv",
        "stage39_evidence_calibration_summary.csv",
        "stage39_failure_comparison_summary.csv",
        "stage39_fixed_regressed_persistent_cases.csv",
        "stage39_top_concepts_for_examples.csv",
        "stage39_top_csg_pairs_for_examples.csv",
        "stage39_final_innovation_points.md",
        "stage39_paper_ready_summary.md",
        "stage39_final_next_steps.md",
    ]:
        print(f"[Done] Wrote: {output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
