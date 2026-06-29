from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results_stage62_final_innovation_consolidation"
STEP57B_DIR = ROOT / "results_stage57B_logit_contribution_audit"
STEP57C_PACKAGE_DIR = ROOT / "results_stage57C_rce_v2_copy_reproduction"
STEP57C_RUN_DIR = STEP57C_PACKAGE_DIR / "rce_v2_copy_csg_a01_rq16_5fold_e20_s1"
STEP58C_PACKAGE_DIR = ROOT / "results_stage58C_residual_constrained_configD_5fold"
STEP59C_PACKAGE_DIR = ROOT / "results_stage59C_dynamic_csg_configA_5fold"
STEP60D_PACKAGE_DIR = ROOT / "results_stage60D_ccra_configC_formal"
STEP61D_PACKAGE_DIR = ROOT / "results_stage61D_l2h_configG_5fold"

METRIC_ORDER = ["AUC", "ACC", "F1", "BACC", "PR_AUC"]
FINAL_PRIMARY_ID = "step58C_residual_constrained_primary"
SECONDARY_VARIANT_IDS = [
    "step59C_dynamic_csg_variant",
    "step60D_ccra_variant",
]
REJECTED_VARIANT_IDS = ["step61D_l2h_not_selected"]
DISPLAY_ROOT_CANDIDATES = [
    Path("/xiangmu/ViLMIL/ViLa-MIL-main"),
    ROOT,
]


MODEL_SPECS = [
    {
        "model_id": "stage57C_rce_v2_baseline",
        "short_name": "Stage57C baseline",
        "package_dir": STEP57C_PACKAGE_DIR,
        "run_dir": STEP57C_RUN_DIR,
        "result_csv": STEP57C_RUN_DIR / "result.csv",
        "fold_csv": STEP57C_RUN_DIR / "fold_summary.csv",
        "summary_path": STEP57C_PACKAGE_DIR / "stage57C_summary.md",
        "status_path": STEP57C_PACKAGE_DIR / "stage57C_reproduction_status.json",
        "decision_path": None,
        "branch_csv": None,
        "contribution_csv": None,
        "module_csv": None,
        "module_type": "none",
        "decision": "reproduced_baseline",
        "model_role": "baseline_reference",
        "selected_as_primary": False,
        "selected_as_variant": False,
        "not_selected_reason": "",
    },
    {
        "model_id": "step58C_residual_constrained_primary",
        "short_name": "Step58C residual constraint",
        "package_dir": STEP58C_PACKAGE_DIR,
        "run_dir": STEP58C_PACKAGE_DIR / "rce_v2_rcD_l003_t050_aux020_5fold_e20_s1",
        "result_csv": STEP58C_PACKAGE_DIR / "rce_v2_rcD_l003_t050_aux020_5fold_e20_s1" / "result.csv",
        "fold_csv": STEP58C_PACKAGE_DIR / "rce_v2_rcD_l003_t050_aux020_5fold_e20_s1" / "fold_summary.csv",
        "summary_path": STEP58C_PACKAGE_DIR / "stage58C_summary.md",
        "status_path": STEP58C_PACKAGE_DIR / "stage58C_status.json",
        "decision_path": STEP58C_PACKAGE_DIR / "stage58C_decision.json",
        "branch_csv": STEP58C_PACKAGE_DIR / "stage58C_branch_metrics_by_fold.csv",
        "contribution_csv": STEP58C_PACKAGE_DIR / "stage58C_contribution_by_fold.csv",
        "module_csv": None,
        "module_type": "residual_constraint",
        "decision": None,
        "model_role": "final_primary_model",
        "selected_as_primary": True,
        "selected_as_variant": False,
        "not_selected_reason": "",
    },
    {
        "model_id": "step59C_dynamic_csg_variant",
        "short_name": "Step59C dynamic CSG",
        "package_dir": STEP59C_PACKAGE_DIR,
        "run_dir": STEP59C_PACKAGE_DIR / "rce_v2_rcD_dynCSG_A_5fold_e20_s1",
        "result_csv": STEP59C_PACKAGE_DIR / "rce_v2_rcD_dynCSG_A_5fold_e20_s1" / "result.csv",
        "fold_csv": STEP59C_PACKAGE_DIR / "rce_v2_rcD_dynCSG_A_5fold_e20_s1" / "fold_summary.csv",
        "summary_path": STEP59C_PACKAGE_DIR / "stage59C_summary.md",
        "status_path": STEP59C_PACKAGE_DIR / "stage59C_status.json",
        "decision_path": STEP59C_PACKAGE_DIR / "stage59C_decision.json",
        "branch_csv": STEP59C_PACKAGE_DIR / "stage59C_branch_metrics_by_fold.csv",
        "contribution_csv": STEP59C_PACKAGE_DIR / "stage59C_contribution_by_fold.csv",
        "module_csv": STEP59C_PACKAGE_DIR / "stage59C_dynamic_csg_by_fold.csv",
        "module_type": "dynamic_csg",
        "decision": None,
        "model_role": "secondary_variant",
        "selected_as_primary": False,
        "selected_as_variant": True,
        "not_selected_reason": "",
    },
    {
        "model_id": "step60D_ccra_variant",
        "short_name": "Step60D CCRA",
        "package_dir": STEP60D_PACKAGE_DIR,
        "run_dir": ROOT / "results_stage60C_ccra_configD_5fold" / "rce_v2_rcD_ccraC_5fold_e20_s1",
        "result_csv": ROOT / "results_stage60C_ccra_configD_5fold" / "rce_v2_rcD_ccraC_5fold_e20_s1" / "result.csv",
        "fold_csv": ROOT / "results_stage60C_ccra_configD_5fold" / "rce_v2_rcD_ccraC_5fold_e20_s1" / "fold_summary.csv",
        "summary_path": STEP60D_PACKAGE_DIR / "stage60D_summary.md",
        "status_path": STEP60D_PACKAGE_DIR / "stage60D_status.json",
        "decision_path": STEP60D_PACKAGE_DIR / "stage60D_decision.json",
        "branch_csv": STEP60D_PACKAGE_DIR / "stage60D_branch_metrics_by_fold.csv",
        "contribution_csv": STEP60D_PACKAGE_DIR / "stage60D_contribution_by_fold.csv",
        "module_csv": STEP60D_PACKAGE_DIR / "stage60D_ccra_by_fold.csv",
        "module_type": "ccra",
        "decision": None,
        "model_role": "secondary_variant",
        "selected_as_primary": False,
        "selected_as_variant": True,
        "not_selected_reason": "",
    },
    {
        "model_id": "step61D_l2h_not_selected",
        "short_name": "Step61D L2H retrieval",
        "package_dir": STEP61D_PACKAGE_DIR,
        "run_dir": STEP61D_PACKAGE_DIR / "rce_v2_rcD_l2hG_5fold_e20_s1",
        "result_csv": STEP61D_PACKAGE_DIR / "rce_v2_rcD_l2hG_5fold_e20_s1" / "result.csv",
        "fold_csv": STEP61D_PACKAGE_DIR / "rce_v2_rcD_l2hG_5fold_e20_s1" / "fold_summary.csv",
        "summary_path": STEP61D_PACKAGE_DIR / "stage61D_summary.md",
        "status_path": STEP61D_PACKAGE_DIR / "stage61D_status.json",
        "decision_path": STEP61D_PACKAGE_DIR / "stage61D_decision.json",
        "branch_csv": STEP61D_PACKAGE_DIR / "stage61D_branch_metrics_by_fold.csv",
        "contribution_csv": STEP61D_PACKAGE_DIR / "stage61D_contribution_by_fold.csv",
        "module_csv": STEP61D_PACKAGE_DIR / "stage61D_l2h_by_fold.csv",
        "module_type": "l2h_retrieval",
        "decision": None,
        "model_role": "not_selected",
        "selected_as_primary": False,
        "selected_as_variant": False,
        "not_selected_reason": (
            "Stable retrieval coverage was observed, but ACC/F1/BACC remained weaker than the selected primary model."
        ),
    },
]


def display_root() -> Path:
    for candidate in DISPLAY_ROOT_CANDIDATES:
        if (candidate / "main.py").is_file() and (candidate / "scripts").is_dir():
            return candidate
    return ROOT


def relative_path_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def round_or_none(value: object, digits: int = 6) -> float | None:
    numeric = safe_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def format_metric(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.6f}"


def format_delta(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:+.6f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def metric_row_to_standard(row: pd.Series) -> dict[str, float]:
    keys = {str(idx).strip().lower(): row[idx] for idx in row.index}
    mapping = {
        "AUC": ["test_auc", "auc"],
        "ACC": ["test_acc", "acc"],
        "F1": ["test_f1", "f1", "macro_f1"],
        "BACC": ["balanced_acc", "bacc", "balanced acc"],
        "PR_AUC": ["pr_auc", "prauc"],
    }
    result: dict[str, float] = {}
    for metric_name, aliases in mapping.items():
        result[metric_name] = math.nan
        for alias in aliases:
            if alias in keys:
                value = safe_float(keys[alias])
                result[metric_name] = math.nan if value is None else value
                break
    return result


def parse_result_metrics(path: Path) -> dict[str, float]:
    df = read_csv(path)
    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    if "metric" in normalized.columns:
        metric_rows = normalized["metric"].astype(str).str.strip().str.lower()
        mean_rows = normalized.loc[metric_rows == "mean"]
        if not mean_rows.empty:
            return metric_row_to_standard(mean_rows.iloc[0])
    return metric_row_to_standard(normalized.iloc[0])


def parse_branch_metrics(path: Path) -> dict[str, object]:
    df = read_csv(path)
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    rename_map = {}
    for column in normalized.columns:
        lowered = column.lower()
        if lowered == "acc":
            rename_map[column] = "ACC"
        elif lowered in {"auc"}:
            rename_map[column] = "AUC"
        elif lowered in {"pr_auc"}:
            rename_map[column] = "PR_AUC"
        elif lowered in {"balanced_acc", "bacc"}:
            rename_map[column] = "BACC"
        elif lowered in {"f1", "macro_f1"}:
            rename_map[column] = "F1"
    normalized = normalized.rename(columns=rename_map)
    if "available" in normalized.columns:
        normalized = normalized.loc[normalized["available"].astype(str).str.lower() != "false"]
    summary = {
        "full_acc": None,
        "full_auc": None,
        "concept_only_acc": None,
        "concept_only_auc": None,
        "visual_only_acc": None,
        "visual_only_auc": None,
        "full_minus_concept_acc": None,
        "full_minus_concept_auc": None,
    }
    branch_means = normalized.groupby("branch", dropna=False)[["ACC", "AUC"]].mean(numeric_only=True)
    for branch_name, prefix in [
        ("full", "full"),
        ("concept_only", "concept_only"),
        ("visual_only", "visual_only"),
    ]:
        if branch_name not in branch_means.index:
            continue
        branch_row = branch_means.loc[branch_name]
        summary[f"{prefix}_acc"] = safe_float(branch_row.get("ACC"))
        summary[f"{prefix}_auc"] = safe_float(branch_row.get("AUC"))
    if summary["full_acc"] is not None and summary["concept_only_acc"] is not None:
        summary["full_minus_concept_acc"] = summary["full_acc"] - summary["concept_only_acc"]
    if summary["full_auc"] is not None and summary["concept_only_auc"] is not None:
        summary["full_minus_concept_auc"] = summary["full_auc"] - summary["concept_only_auc"]
    return summary


def parse_contribution_metrics(path: Path) -> dict[str, object]:
    df = read_csv(path)
    summary = {}
    for column in [
        "visual_ratio_mean",
        "concept_ratio_mean",
        "csg_ratio_mean",
        "visual_ratio_gt_0_5_percent",
    ]:
        summary[column] = safe_float(df[column].mean()) if column in df.columns else None
    return summary


def parse_step57b_reference() -> dict[str, object]:
    branch_path = STEP57B_DIR / "stage57B_branch_metrics.csv"
    margin_path = STEP57B_DIR / "stage57B_margin_stats.csv"
    status_path = STEP57B_DIR / "stage57B_audit_status.json"
    if not branch_path.is_file() or not margin_path.is_file() or not status_path.is_file():
        return {}
    branch_df = read_csv(branch_path)
    branch_summary = {
        "full_acc": None,
        "full_auc": None,
        "concept_only_acc": None,
        "concept_only_auc": None,
        "visual_only_acc": None,
        "visual_only_auc": None,
        "full_minus_concept_acc": None,
        "full_minus_concept_auc": None,
    }
    for branch_name, prefix in [
        ("full", "full"),
        ("concept_only", "concept_only"),
        ("visual_only", "visual_only"),
    ]:
        rows = branch_df.loc[branch_df["branch"] == branch_name]
        if rows.empty:
            continue
        branch_summary[f"{prefix}_acc"] = safe_float(rows.iloc[0].get("acc"))
        branch_summary[f"{prefix}_auc"] = safe_float(rows.iloc[0].get("auc"))
    if branch_summary["full_acc"] is not None and branch_summary["concept_only_acc"] is not None:
        branch_summary["full_minus_concept_acc"] = (
            branch_summary["full_acc"] - branch_summary["concept_only_acc"]
        )
    if branch_summary["full_auc"] is not None and branch_summary["concept_only_auc"] is not None:
        branch_summary["full_minus_concept_auc"] = (
            branch_summary["full_auc"] - branch_summary["concept_only_auc"]
        )

    margin_df = read_csv(margin_path)
    margin_lookup = {
        str(row["metric_name"]).strip(): safe_float(row["mean"])
        for _, row in margin_df.iterrows()
    }
    status_payload = read_json(status_path)
    return {
        **branch_summary,
        "visual_ratio_mean": margin_lookup.get("visual_contribution_ratio"),
        "concept_ratio_mean": margin_lookup.get("concept_contribution_ratio"),
        "csg_ratio_mean": margin_lookup.get("csg_contribution_ratio"),
        "visual_ratio_gt_0_5_percent": safe_float(
            (status_payload.get("visual_details") or {}).get("pct_visual_ratio_gt_0_5")
        ),
        "interpretability_source": "step57B_single_fold_reference",
        "interpretability_is_formal_5fold": False,
    }


def parse_module_summary(model_id: str, module_type: str, path: Path | None) -> dict[str, object]:
    if module_type == "none":
        return {
            "module_type": "none",
            "module_enabled": False,
            "module_alpha_mean": None,
            "module_delta_abs_mean": None,
            "module_signal_interpretation": "Baseline reference only; no extra innovation module is enabled.",
            "claim_strength": "reference_only",
        }
    if module_type == "residual_constraint":
        return {
            "module_type": "residual_constraint",
            "module_enabled": True,
            "module_alpha_mean": None,
            "module_delta_abs_mean": None,
            "module_signal_interpretation": (
                "Residual constraint is a training-time regularizer. No standalone module alpha/delta artifact was exported, "
                "so its signal is inferred from the stable shift from visual-dominant evidence toward concept-dominant evidence."
            ),
            "claim_strength": "strong_primary",
        }
    if path is None or not path.is_file():
        return {
            "module_type": module_type,
            "module_enabled": None,
            "module_alpha_mean": None,
            "module_delta_abs_mean": None,
            "module_signal_interpretation": "Module summary file is missing.",
            "claim_strength": "missing_artifact",
        }

    df = read_csv(path)
    if module_type == "dynamic_csg":
        alpha_mean = safe_float(df["learned_alpha_final"].mean()) if "learned_alpha_final" in df.columns else None
        delta_mean = (
            safe_float(df["csg_logits_delta_abs_mean"].mean())
            if "csg_logits_delta_abs_mean" in df.columns
            else None
        )
        return {
            "module_type": "dynamic_csg",
            "module_enabled": safe_float(df["dynamic_csg_enabled"].mean()) == 1.0 if "dynamic_csg_enabled" in df.columns else True,
            "module_alpha_mean": alpha_mean,
            "module_delta_abs_mean": delta_mean,
            "module_signal_interpretation": (
                "Dynamic CSG produced nonzero sample-adaptive graph updates across folds, but the logit-level delta remained very small."
            ),
            "claim_strength": "conservative_variant",
        }
    if module_type == "ccra":
        alpha_mean = safe_float(df["learned_alpha_final"].mean()) if "learned_alpha_final" in df.columns else None
        low_delta = safe_float(df["low_ccra_delta_abs_mean"].mean()) if "low_ccra_delta_abs_mean" in df.columns else None
        high_delta = safe_float(df["high_ccra_delta_abs_mean"].mean()) if "high_ccra_delta_abs_mean" in df.columns else None
        avg_delta = None
        if low_delta is not None and high_delta is not None:
            avg_delta = (low_delta + high_delta) / 2.0
        return {
            "module_type": "ccra",
            "module_enabled": safe_float(df["ccra_enabled"].mean()) == 1.0 if "ccra_enabled" in df.columns else True,
            "module_alpha_mean": alpha_mean,
            "module_delta_abs_mean": avg_delta,
            "module_signal_interpretation": (
                "CCRA produced stable nonzero concept-conditioned low/high region changes across all audited folds."
            ),
            "claim_strength": "moderate_variant",
            "low_ccra_delta_abs_mean": low_delta,
            "high_ccra_delta_abs_mean": high_delta,
        }
    if module_type == "l2h_retrieval":
        alpha_mean = safe_float(df["learned_alpha_final"].mean()) if "learned_alpha_final" in df.columns else None
        delta_mean = safe_float(df["l2h_delta_abs_mean"].mean()) if "l2h_delta_abs_mean" in df.columns else None
        match_count = (
            safe_float(df["retrieved_high_match_counts_mean"].mean())
            if "retrieved_high_match_counts_mean" in df.columns
            else None
        )
        zero_match = (
            safe_float(df["retrieved_high_zero_match_percent"].mean())
            if "retrieved_high_zero_match_percent" in df.columns
            else None
        )
        return {
            "module_type": "l2h_retrieval",
            "module_enabled": safe_float(df["l2h_enabled"].mean()) == 1.0 if "l2h_enabled" in df.columns else True,
            "module_alpha_mean": alpha_mean,
            "module_delta_abs_mean": delta_mean,
            "module_signal_interpretation": (
                "L2H retrieval achieved stable coordinate-level coverage and nonzero learned alpha, "
                "but the exported debug package did not contain a populated l2h_delta_abs_mean field."
            ),
            "claim_strength": "exploratory_not_selected",
            "retrieved_high_match_counts_mean": match_count,
            "retrieved_high_zero_match_percent": zero_match,
        }
    return {
        "module_type": module_type,
        "module_enabled": None,
        "module_alpha_mean": None,
        "module_delta_abs_mean": None,
        "module_signal_interpretation": "Unsupported module type.",
        "claim_strength": "unsupported",
    }


def evidence_shift_interpretation(visual_ratio: object, concept_ratio: object) -> str:
    visual = safe_float(visual_ratio)
    concept = safe_float(concept_ratio)
    if visual is None or concept is None:
        return "Evidence-shift summary unavailable because branch/contribution artifacts are missing."
    if visual < 0.4 and concept > 0.6:
        return "Evidence is shifted away from visual residual and toward concept evidence."
    if visual > 0.6 and concept < 0.4:
        return "Evidence remains visually dominant."
    return "Evidence is mixed without a strong visual- or concept-dominant pattern."


def metric_delta(current: object, reference: object) -> float | None:
    lhs = safe_float(current)
    rhs = safe_float(reference)
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def git_root() -> Path:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def git_path_modified(path: Path) -> bool:
    repo_root = git_root()
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short", "--", str(rel)],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def load_model_record(spec: dict[str, object], baseline_metrics: dict[str, float], warnings: list[str]) -> dict[str, object]:
    record = dict(spec)
    record["source_dir"] = relative_path_str(spec["package_dir"])
    record["run_dir_rel"] = relative_path_str(spec["run_dir"])
    record["metrics"] = parse_result_metrics(spec["result_csv"])
    status_payload = read_json(spec["status_path"]) if spec["status_path"] and Path(spec["status_path"]).is_file() else {}
    record["status_payload"] = status_payload
    decision_payload = read_json(spec["decision_path"]) if spec["decision_path"] and Path(spec["decision_path"]).is_file() else {}
    record["decision_payload"] = decision_payload
    record["decision"] = spec["decision"] or decision_payload.get("decision") or status_payload.get("decision") or "unknown"

    branch_summary: dict[str, object]
    contribution_summary: dict[str, object]
    if spec["model_id"] == "stage57C_rce_v2_baseline":
        step57b_reference = parse_step57b_reference()
        branch_summary = {
            key: step57b_reference.get(key)
            for key in [
                "full_acc",
                "full_auc",
                "concept_only_acc",
                "concept_only_auc",
                "visual_only_acc",
                "visual_only_auc",
                "full_minus_concept_acc",
                "full_minus_concept_auc",
            ]
        }
        contribution_summary = {
            key: step57b_reference.get(key)
            for key in [
                "visual_ratio_mean",
                "concept_ratio_mean",
                "csg_ratio_mean",
                "visual_ratio_gt_0_5_percent",
            ]
        }
        record["interpretability_source"] = step57b_reference.get(
            "interpretability_source", "missing_reference"
        )
        record["interpretability_is_formal_5fold"] = bool(
            step57b_reference.get("interpretability_is_formal_5fold", False)
        )
        warnings.append(
            "Stage57C baseline branch/contribution fields are populated from Step57B single-fold audit as reference only; "
            "formal 5-fold baseline branch/contribution artifacts are not available."
        )
    else:
        branch_summary = parse_branch_metrics(spec["branch_csv"]) if spec["branch_csv"] and Path(spec["branch_csv"]).is_file() else {}
        contribution_summary = (
            parse_contribution_metrics(spec["contribution_csv"])
            if spec["contribution_csv"] and Path(spec["contribution_csv"]).is_file()
            else {}
        )
        record["interpretability_source"] = relative_path_str(spec["package_dir"])
        record["interpretability_is_formal_5fold"] = True

    record["branch_summary"] = branch_summary
    record["contribution_summary"] = contribution_summary
    record["module_summary"] = parse_module_summary(
        model_id=str(spec["model_id"]),
        module_type=str(spec["module_type"]),
        path=spec["module_csv"],
    )
    record["evidence_shift_interpretation"] = evidence_shift_interpretation(
        contribution_summary.get("visual_ratio_mean"),
        contribution_summary.get("concept_ratio_mean"),
    )

    for metric_name in METRIC_ORDER:
        record[f"delta_{metric_name.lower()}_vs_stage57C"] = metric_delta(
            record["metrics"].get(metric_name), baseline_metrics.get(metric_name)
        )
    return record


def choose_primary_and_variants(records: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    by_id = {record["model_id"]: record for record in records}
    primary = by_id[FINAL_PRIMARY_ID]
    secondary = [by_id[model_id] for model_id in SECONDARY_VARIANT_IDS]
    rejected = [by_id[model_id] for model_id in REJECTED_VARIANT_IDS]
    return primary, secondary, rejected


def build_main_results_table(records: list[dict[str, object]], primary: dict[str, object]) -> pd.DataFrame:
    primary_metrics = primary["metrics"]
    rows = []
    for record in records:
        row = {
            "model_id": record["model_id"],
            "model_role": record["model_role"],
            "source_dir": record["source_dir"],
            "run_dir": record["run_dir_rel"],
            "decision": record["decision"],
        }
        for metric_name in METRIC_ORDER:
            row[metric_name] = round_or_none(record["metrics"].get(metric_name))
            row[f"delta_{metric_name.lower()}_vs_stage57C"] = round_or_none(
                record.get(f"delta_{metric_name.lower()}_vs_stage57C")
            )
            row[f"delta_{metric_name.lower()}_vs_step58C"] = round_or_none(
                metric_delta(record["metrics"].get(metric_name), primary_metrics.get(metric_name))
            )
        row["selected_as_primary"] = bool(record["selected_as_primary"])
        row["selected_as_variant"] = bool(record["selected_as_variant"])
        row["not_selected_reason"] = record["not_selected_reason"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_branch_contribution_table(records: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        branch_summary = record["branch_summary"]
        contribution_summary = record["contribution_summary"]
        rows.append(
            {
                "model_id": record["model_id"],
                "interpretability_source": record["interpretability_source"],
                "interpretability_is_formal_5fold": record["interpretability_is_formal_5fold"],
                "visual_ratio_mean": round_or_none(contribution_summary.get("visual_ratio_mean")),
                "concept_ratio_mean": round_or_none(contribution_summary.get("concept_ratio_mean")),
                "csg_ratio_mean": round_or_none(contribution_summary.get("csg_ratio_mean")),
                "visual_ratio_gt_0_5_percent": round_or_none(
                    contribution_summary.get("visual_ratio_gt_0_5_percent")
                ),
                "full_acc": round_or_none(branch_summary.get("full_acc")),
                "full_auc": round_or_none(branch_summary.get("full_auc")),
                "concept_only_acc": round_or_none(branch_summary.get("concept_only_acc")),
                "concept_only_auc": round_or_none(branch_summary.get("concept_only_auc")),
                "visual_only_acc": round_or_none(branch_summary.get("visual_only_acc")),
                "visual_only_auc": round_or_none(branch_summary.get("visual_only_auc")),
                "full_minus_concept_acc": round_or_none(branch_summary.get("full_minus_concept_acc")),
                "full_minus_concept_auc": round_or_none(branch_summary.get("full_minus_concept_auc")),
            }
        )
    return pd.DataFrame(rows)


def build_interpretability_table(records: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        module = record["module_summary"]
        contribution_summary = record["contribution_summary"]
        rows.append(
            {
                "model_id": record["model_id"],
                "module_type": module.get("module_type"),
                "module_enabled": module.get("module_enabled"),
                "module_alpha_mean": round_or_none(module.get("module_alpha_mean")),
                "module_delta_abs_mean": round_or_none(module.get("module_delta_abs_mean")),
                "module_signal_interpretation": module.get("module_signal_interpretation"),
                "visual_ratio_mean": round_or_none(contribution_summary.get("visual_ratio_mean")),
                "concept_ratio_mean": round_or_none(contribution_summary.get("concept_ratio_mean")),
                "evidence_shift_interpretation": record.get("evidence_shift_interpretation"),
                "claim_strength": module.get("claim_strength"),
            }
        )
    return pd.DataFrame(rows)


def build_variant_comparison_table(records: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        module = record["module_summary"]
        contribution_summary = record["contribution_summary"]
        rows.append(
            {
                "model_id": record["model_id"],
                "model_role": record["model_role"],
                "decision": record["decision"],
                "module_type": module.get("module_type"),
                "AUC": round_or_none(record["metrics"].get("AUC")),
                "ACC": round_or_none(record["metrics"].get("ACC")),
                "F1": round_or_none(record["metrics"].get("F1")),
                "BACC": round_or_none(record["metrics"].get("BACC")),
                "PR_AUC": round_or_none(record["metrics"].get("PR_AUC")),
                "visual_ratio_mean": round_or_none(contribution_summary.get("visual_ratio_mean")),
                "concept_ratio_mean": round_or_none(contribution_summary.get("concept_ratio_mean")),
                "module_alpha_mean": round_or_none(module.get("module_alpha_mean")),
                "module_delta_abs_mean": round_or_none(module.get("module_delta_abs_mean")),
                "selected_as_primary": bool(record["selected_as_primary"]),
                "selected_as_variant": bool(record["selected_as_variant"]),
                "variant_positioning": record["model_role"],
                "claim_strength": module.get("claim_strength"),
            }
        )
    return pd.DataFrame(rows)


def build_ablation_variant_table(records: list[dict[str, object]]) -> pd.DataFrame:
    label_map = {
        "stage57C_rce_v2_baseline": "baseline RCE-v2",
        "step58C_residual_constrained_primary": "+ residual constraint",
        "step59C_dynamic_csg_variant": "+ residual constraint + dynamic CSG",
        "step60D_ccra_variant": "+ residual constraint + CCRA",
        "step61D_l2h_not_selected": "+ residual constraint + L2H retrieval",
    }
    rows = []
    for record in records:
        module = record["module_summary"]
        contribution_summary = record["contribution_summary"]
        module_signal = module.get("module_signal_interpretation", "")
        if record["model_id"] == "stage57C_rce_v2_baseline":
            module_signal = "No extra module; reference baseline."
        rows.append(
            {
                "variant_label": label_map[record["model_id"]],
                "model_id": record["model_id"],
                "AUC": round_or_none(record["metrics"].get("AUC")),
                "ACC": round_or_none(record["metrics"].get("ACC")),
                "F1": round_or_none(record["metrics"].get("F1")),
                "BACC": round_or_none(record["metrics"].get("BACC")),
                "PR_AUC": round_or_none(record["metrics"].get("PR_AUC")),
                "visual_ratio_mean": round_or_none(contribution_summary.get("visual_ratio_mean")),
                "concept_ratio_mean": round_or_none(contribution_summary.get("concept_ratio_mean")),
                "module_signal": module_signal,
                "final_role": record["model_role"],
            }
        )
    return pd.DataFrame(rows)


def build_final_decision(primary: dict[str, object], secondary: list[dict[str, object]], rejected: list[dict[str, object]]) -> dict[str, object]:
    primary_reason = [
        "Step58C is the only completed formal 5-fold innovation model that clearly improves ACC, F1, and BACC over the reproduced Stage57C baseline.",
        "Its evidence composition shifts strongly away from visual residuals and toward concept evidence while the full branch remains stronger than concept_only.",
        "Its AUC trade-off versus Stage57C is small enough to remain acceptable for the final main result.",
    ]
    secondary_variants = [
        {
            "model_id": record["model_id"],
            "source_dir": record["source_dir"],
            "reason": (
                "AUC/PR-AUC improved relative to Step58C and dynamic alpha stayed nonzero, but ACC/F1/BACC were weaker and the CSG logit delta stayed small."
                if record["model_id"] == "step59C_dynamic_csg_variant"
                else "CCRA stayed performance-competitive and produced stable nonzero concept-conditioned region changes, but the full branch was slightly weaker than concept_only."
            ),
        }
        for record in secondary
    ]
    rejected_variants = [
        {
            "model_id": record["model_id"],
            "source_dir": record["source_dir"],
            "reason": record["not_selected_reason"],
        }
        for record in rejected
    ]
    return {
        "final_primary_model_id": primary["model_id"],
        "final_primary_source_dir": primary["source_dir"],
        "final_primary_reason": primary_reason,
        "final_primary_metrics": {
            metric_name: round_or_none(primary["metrics"].get(metric_name)) for metric_name in METRIC_ORDER
        },
        "secondary_variants": secondary_variants,
        "rejected_variants": rejected_variants,
        "primary_selection_policy": (
            "Select only among completed formal 5-fold models. Prefer the model that improves ACC/F1/BACC over the reproduced baseline, "
            "keeps AUC/PR-AUC within an acceptable trade-off range, maintains low visual_ratio/high concept_ratio, "
            "and avoids branch collapse or reliance on a weaker concept_only branch."
        ),
        "paper_main_claim": (
            "Residual-constrained RCE shifts final evidence from visual residual toward concept evidence and improves hard classification metrics over the reproduced RCE-v2 baseline."
        ),
        "paper_secondary_claims": [
            "Dynamic CSG introduces nonzero sample-adaptive graph updates, but its direct logit effect is small and should be positioned as a secondary variant rather than the main performance driver.",
            "CCRA produces stable nonzero concept-conditioned region changes and serves as an interpretable balanced variant.",
            "L2H retrieval is feasible at the coordinate level and yields stable retrieval coverage, but it is not selected because hard classification metrics are weaker than the chosen primary model.",
        ],
        "claims_to_avoid": [
            "Do not claim that Dynamic CSG is the dominant classification contributor.",
            "Do not claim that CCRA clearly surpasses every baseline.",
            "Do not present L2H retrieval as the final effective primary model.",
            "Do not cherry-pick a single metric while hiding the ACC/AUC trade-off.",
            "Do not claim that all innovation modules have been jointly validated as one unified best model.",
        ],
        "next_recommended_action": "freeze_results_and_prepare_paper_assets",
    }


def build_claims_md() -> str:
    lines = [
        "# Step62 Claims To Make And Avoid",
        "",
        "## 一、可以写的 claims",
        "",
        "1. Residual-constrained training shifts final evidence from visual residual toward concept evidence.",
        "2. Step58C improves ACC / F1 / BACC over the reproduced RCE-v2 baseline while keeping AUC within an acceptable range.",
        "3. Dynamic CSG produces nonzero sample-adaptive graph updates, but its logit-level effect is small.",
        "4. CCRA produces stable nonzero concept-conditioned region changes and can be reported as an interpretable variant.",
        "5. L2H retrieval is feasible at the data/coordinate level and produces stable retrieval coverage, but is not selected due to weaker hard classification metrics.",
        "",
        "## 二、不能写或需要避免的 claims",
        "",
        "1. 不要说 Dynamic CSG 是主要分类贡献来源。",
        "2. 不要说 CCRA 明确超过所有 baseline。",
        "3. 不要说 L2H retrieval 是最终有效主模型。",
        "4. 不要只挑 ACC 或只挑 AUC，而忽略 trade-off。",
        "5. 不要声称所有创新模块组合后形成最终统一最优模型，因为当前没有完成组合模型验证。",
        "6. 不要把 Step61D 的 L2H 写成 positive main result，只能写成 explored but not selected。",
        "",
    ]
    return "\n".join(lines)


def build_paper_ready_md(
    main_table: pd.DataFrame,
    branch_table: pd.DataFrame,
    interpretability_table: pd.DataFrame,
    primary: dict[str, object],
    secondary: list[dict[str, object]],
    rejected: list[dict[str, object]],
    warnings: list[str],
) -> str:
    primary_metrics = primary["metrics"]
    primary_branch = primary["branch_summary"]
    primary_contrib = primary["contribution_summary"]
    lines = [
        "# Step62 论文可用最终结果整理",
        "",
        "## 1. 当前最终模型选择结论",
        "",
        f"最终主模型选择为 `{primary['model_id']}`，对应结果目录 `{primary['source_dir']}`。该模型来自 Step58C residual-constrained RCE config D，是当前所有已完成正式 5-fold 创新模型中最适合作为论文主结果的模型。",
        "",
        "## 2. 五个正式模型的结果表",
        "",
        markdown_table(
            main_table[
                [
                    "model_id",
                    "model_role",
                    "decision",
                    "AUC",
                    "ACC",
                    "F1",
                    "BACC",
                    "PR_AUC",
                ]
            ]
        ),
        "",
        "## 3. 为什么 Step58C 是主模型",
        "",
        f"Step58C 的 5-fold 指标为 AUC={format_metric(primary_metrics.get('AUC'))}、ACC={format_metric(primary_metrics.get('ACC'))}、F1={format_metric(primary_metrics.get('F1'))}、BACC={format_metric(primary_metrics.get('BACC'))}、PR_AUC={format_metric(primary_metrics.get('PR_AUC'))}。相对 Stage57C baseline，它提升了 ACC/F1/BACC，同时 visual_ratio 降到 {format_metric(primary_contrib.get('visual_ratio_mean'))}、concept_ratio 升到 {format_metric(primary_contrib.get('concept_ratio_mean'))}，且 full 分支仍强于 concept_only（ACC 差值 {format_delta(primary_branch.get('full_minus_concept_acc'))}，AUC 差值 {format_delta(primary_branch.get('full_minus_concept_auc'))}）。",
        "",
        "## 4. 为什么 Step59C / Step60D 是变体",
        "",
        "Step59C Dynamic CSG 的 AUC / PR-AUC 相对 Step58C 更高，但 ACC / F1 / BACC 更低；同时 dynamic alpha 多 fold 非零，但 csg logit delta 很小，因此更适合作为“有可观测 sample-adaptive 更新，但主分类贡献有限”的变体。",
        "",
        "Step60D CCRA 的整体性能接近主线模型，并且 learned alpha、low delta、high delta 都稳定非零，说明 concept-conditioned region aggregation 机制是成立的；但 full branch 略弱于 concept_only，因此更适合作为可解释 balanced representative，而不是主模型。",
        "",
        "## 5. 为什么 Step61D 不选",
        "",
        "Step61D L2H retrieval 在数据/坐标层面是可行的：retrieval match count 稳定、zero-match 几乎为零、learned alpha 非零。但它的 ACC/F1/BACC 不足以与 Step58C 竞争，因此只能写成 explored but not selected。",
        "",
        "## 6. visual_ratio / concept_ratio 的解释",
        "",
        "visual_ratio 表示 final logits 中来自 visual residual 的相对占比，concept_ratio 表示来自 concept evidence 的相对占比。Step57B 的单 fold baseline 审计显示 visual_ratio 约为 0.7196、concept_ratio 约为 0.2804，说明原始 reproduced baseline 更偏视觉残差。Step58C 及之后的几条创新线都把证据结构推向 concept-dominant，这一点是本轮创新的核心可解释结论。",
        "",
        "## 7. CSG、CCRA、L2H 三条探索线的最终定位",
        "",
        "Dynamic CSG：定位为 secondary variant。可以说它确实引入了 nonzero sample-adaptive graph updates，但不能说它是主要分类来源。",
        "",
        "CCRA：定位为 secondary variant。可以说它带来了稳定非零的 concept-conditioned region changes，是更强的解释性变体之一。",
        "",
        "L2H retrieval：定位为 rejected exploratory line。可以说检索覆盖稳定、机制可行，但不能说它形成了更好的最终分类模型。",
        "",
        "## 8. 论文主结果怎么写",
        "",
        "论文主结果应以 Step58C 为主：强调 residual-constrained training 在不引入新的未验证组合模型前提下，将 final evidence 从 visual residual 转向 concept evidence，并提升 ACC/F1/BACC。",
        "",
        "## 9. 消融实验怎么写",
        "",
        "消融表建议按 baseline RCE-v2、+ residual constraint、+ residual constraint + dynamic CSG、+ residual constraint + CCRA、+ residual constraint + L2H retrieval 组织。重点不是声称所有模块都超过 baseline，而是说明每条创新线分别回答了不同问题：主性能、sample-adaptive graph、concept-conditioned region aggregation、coordinate retrieval feasibility。",
        "",
        "## 10. 局限性怎么写",
        "",
        "当前没有完成所有创新模块的联合组合验证，因此不能声称存在统一最优组合模型。Dynamic CSG 的 logit 影响偏小，CCRA 的 full branch 没有稳定压过 concept_only，L2H retrieval 的导出 debug 字段仍有缺项（如 l2h_delta_abs_mean 为空）。",
        "",
        "## 11. 后续工作怎么写",
        "",
        "后续工作建议进入结果冻结与论文资产整理阶段，而不是继续扩展新模块。若未来继续研究，可优先考虑补齐更加标准化的多 fold interpretability export，而不是重新开新的训练线。",
        "",
        "## 附：解释性与分支摘要",
        "",
        markdown_table(branch_table),
        "",
        markdown_table(interpretability_table),
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def build_summary_md(
    main_table: pd.DataFrame,
    final_decision: dict[str, object],
    warnings: list[str],
    protected_status: dict[str, bool],
    generated_files: list[str],
) -> str:
    primary_id = final_decision["final_primary_model_id"]
    primary_row = main_table.loc[main_table["model_id"] == primary_id].iloc[0]
    secondary_ids = [item["model_id"] for item in final_decision["secondary_variants"]]
    rejected_ids = [item["model_id"] for item in final_decision["rejected_variants"]]
    lines = [
        "# Step62 Final Innovation Consolidation",
        "",
        "## Final Positioning",
        "",
        f"- final_primary_model: `{primary_id}`",
        f"- secondary_variants: `{secondary_ids}`",
        f"- rejected_variants: `{rejected_ids}`",
        f"- next_recommended_action: `{final_decision['next_recommended_action']}`",
        "",
        "## Main Result Snapshot",
        "",
        f"- primary AUC={format_metric(primary_row['AUC'])}",
        f"- primary ACC={format_metric(primary_row['ACC'])}",
        f"- primary F1={format_metric(primary_row['F1'])}",
        f"- primary BACC={format_metric(primary_row['BACC'])}",
        f"- primary PR_AUC={format_metric(primary_row['PR_AUC'])}",
        "",
        "## Policy",
        "",
        f"- {final_decision['primary_selection_policy']}",
        "",
        "## Protected Files",
        "",
        f"- original_rce_modified: `{protected_status['original_rce_modified']}`",
        f"- rce_v2_modified: `{protected_status['rce_v2_modified']}`",
        f"- main_modified: `{protected_status['main_modified']}`",
        f"- core_utils_modified: `{protected_status['core_utils_modified']}`",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in generated_files])
    lines.append("")
    return "\n".join(lines)


def build_run_commands_txt() -> str:
    root = display_root()
    script_rel = "scripts/analysis/build_stage62_final_innovation_consolidation.py"
    return "\n".join(
        [
            f"cd {root}",
            f"python -m py_compile {script_rel}",
            f"PYTHONPATH={root} python {script_rel}",
            "# No training commands are executed in Step62.",
            "",
        ]
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    required_sources = [
        STEP57C_RUN_DIR / "result.csv",
        STEP58C_PACKAGE_DIR / "stage58C_summary.md",
        STEP59C_PACKAGE_DIR / "stage59C_summary.md",
        STEP60D_PACKAGE_DIR / "stage60D_summary.md",
        STEP61D_PACKAGE_DIR / "stage61D_summary.md",
    ]
    missing_sources = [relative_path_str(path) for path in required_sources if not path.is_file()]
    all_required_sources_found = not missing_sources

    baseline_metrics = parse_result_metrics(STEP57C_RUN_DIR / "result.csv")
    records = [load_model_record(spec, baseline_metrics, warnings) for spec in MODEL_SPECS]
    primary, secondary, rejected = choose_primary_and_variants(records)

    main_table = build_main_results_table(records, primary)
    branch_table = build_branch_contribution_table(records)
    interpretability_table = build_interpretability_table(records)
    variant_table = build_variant_comparison_table(records)
    ablation_table = build_ablation_variant_table(records)
    final_decision = build_final_decision(primary, secondary, rejected)

    claims_md = build_claims_md()
    paper_ready_md = build_paper_ready_md(
        main_table=main_table,
        branch_table=branch_table,
        interpretability_table=interpretability_table,
        primary=primary,
        secondary=secondary,
        rejected=rejected,
        warnings=warnings,
    )

    protected_paths = {
        "original_rce_modified": ROOT / "models" / "model_RCE_MIL_BiomedCLIP.py",
        "rce_v2_modified": ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py",
        "main_modified": ROOT / "main.py",
        "core_utils_modified": ROOT / "utils" / "core_utils.py",
    }
    protected_status = {
        key: git_path_modified(path) for key, path in protected_paths.items()
    }

    variant_table.to_csv(OUTPUT_DIR / "stage62_variant_comparison.csv", index=False)
    main_table.to_csv(OUTPUT_DIR / "stage62_main_results_table.csv", index=False)
    branch_table.to_csv(OUTPUT_DIR / "stage62_branch_contribution_summary.csv", index=False)
    interpretability_table.to_csv(OUTPUT_DIR / "stage62_interpretability_summary.csv", index=False)
    ablation_table.to_csv(OUTPUT_DIR / "stage62_ablation_and_variant_table.csv", index=False)
    (OUTPUT_DIR / "stage62_final_model_decision.json").write_text(
        json.dumps(final_decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage62_claims_to_make_and_avoid.md").write_text(claims_md, encoding="utf-8")
    (OUTPUT_DIR / "stage62_paper_ready_results.md").write_text(paper_ready_md, encoding="utf-8")
    (OUTPUT_DIR / "stage62_run_commands.txt").write_text(build_run_commands_txt(), encoding="utf-8")

    generated_files = [
        "stage62_summary.md",
        "stage62_status.json",
        "stage62_final_model_decision.json",
        "stage62_main_results_table.csv",
        "stage62_variant_comparison.csv",
        "stage62_branch_contribution_summary.csv",
        "stage62_interpretability_summary.csv",
        "stage62_ablation_and_variant_table.csv",
        "stage62_claims_to_make_and_avoid.md",
        "stage62_paper_ready_results.md",
        "stage62_run_commands.txt",
    ]

    summary_md = build_summary_md(
        main_table=main_table,
        final_decision=final_decision,
        warnings=warnings,
        protected_status=protected_status,
        generated_files=generated_files,
    )
    (OUTPUT_DIR / "stage62_summary.md").write_text(summary_md, encoding="utf-8")

    status_payload = {
        "status": "completed" if all_required_sources_found and not any(protected_status.values()) else "error",
        "all_required_sources_found": all_required_sources_found,
        "missing_sources": missing_sources,
        "parsed_models": [record["model_id"] for record in records],
        "generated_files": generated_files,
        "warnings": warnings,
        "no_training_run": True,
        **protected_status,
    }
    (OUTPUT_DIR / "stage62_status.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
