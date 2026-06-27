from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analysis.build_stage60C_ccra_5fold_summary as stage60c_summary

from scripts.analysis.build_stage60C_ccra_5fold_summary import (
    CCRA_CONFIG,
    PYTHON_BIN,
    STEP57B_CONCEPT_BASELINE,
    STEP57B_VISUAL_BASELINE,
    add_delta_vs_step58c_contribution,
    build_fold_metrics_df,
    collect_audits_and_ccra,
    collect_completed_checkpoint_folds,
    compute_mean_metrics,
    detect_display_root,
    determine_run_status,
    discover_aggregate_metrics,
    discover_fold_metrics,
    format_delta,
    format_metric,
    read_csv_if_exists,
    relative_path_str,
    round_or_none,
    safe_float,
    summarize_branch_means,
    summarize_ccra_means,
    summarize_contribution_means,
    to_float,
)


DEFAULT_STAGE57C_DIR = (
    ROOT
    / "results_stage57C_rce_v2_copy_reproduction"
    / "rce_v2_copy_csg_a01_rq16_5fold_e20_s1"
)
DEFAULT_STAGE58C_DIR = (
    ROOT
    / "results_stage58C_residual_constrained_configD_5fold"
    / "rce_v2_rcD_l003_t050_aux020_5fold_e20_s1"
)
DEFAULT_STAGE59C_DIR = (
    ROOT
    / "results_stage59C_dynamic_csg_configA_5fold"
    / "rce_v2_rcD_dynCSG_A_5fold_e20_s1"
)
DEFAULT_STAGE60C_DIR = (
    ROOT
    / "results_stage60C_ccra_configD_5fold"
    / "rce_v2_rcD_ccraD_5fold_e20_s1"
)
DEFAULT_STAGE60D_SOURCE_DIR = (
    ROOT
    / "results_stage60C_ccra_configD_5fold"
    / "rce_v2_rcD_ccraC_5fold_e20_s1"
)
DEFAULT_OUTPUT_DIR = ROOT / "results_stage60D_ccra_configC_formal"
DEFAULT_AUDIT_SCRIPT = ROOT / "scripts" / "analysis" / "build_stage57B_logit_contribution_audit.py"
STEP60D_CCRA_CONFIG = {
    "rce_use_ccra": True,
    "rce_ccra_mode": "concept_query_residual",
    "rce_ccra_alpha_init": 0.01,
    "rce_ccra_scale": 1.0,
    "rce_ccra_num_queries": 0,
    "rce_ccra_query_source": "prompt_mean",
    "rce_ccra_detach_prompt": False,
    "rce_ccra_norm": "layernorm",
    "rce_ccra_dropout": 0.0,
    "rce_ccra_clip": 5.0,
    "rce_use_dynamic_csg": False,
}
stage60c_summary.CCRA_CONFIG.clear()
stage60c_summary.CCRA_CONFIG.update(STEP60D_CCRA_CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step60D CCRA config C secondary formal audit summary."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage57c-dir", type=Path, default=DEFAULT_STAGE57C_DIR)
    parser.add_argument("--stage58c-dir", type=Path, default=DEFAULT_STAGE58C_DIR)
    parser.add_argument("--stage59c-dir", type=Path, default=DEFAULT_STAGE59C_DIR)
    parser.add_argument("--stage60c-dir", type=Path, default=DEFAULT_STAGE60C_DIR)
    parser.add_argument("--stage60d-source-dir", type=Path, default=DEFAULT_STAGE60D_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT_SCRIPT)
    return parser.parse_args()


def write_run_commands(output_dir: Path, display_root: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = "\n".join(
        [
            f"cd {display_root}",
            "# Default: only inspect existing config C results and refresh formal summary",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage60D_ccra_configC_formal_summary.py",
            "",
            "# Only if config C results are missing or incomplete",
            "TARGET_CONFIG=C RUN_TRAIN=1 bash scripts/experiments/run_stage60C_ccra_configD_5fold.sh",
            "",
            "# Rebuild Step60D outputs after training",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage60D_ccra_configC_formal_summary.py",
        ]
    )
    (output_dir / "stage60D_run_commands.txt").write_text(command_text + "\n", encoding="utf-8")
    return command_text


def build_compare_df(
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage59c_metrics: dict[str, float],
    stage60c_metrics: dict[str, float],
    stage60d_metrics: dict[str, float],
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
    stage60c_dir: Path,
    stage60d_dir: Path,
    contribution_means: dict[str, float],
    ccra_means: dict[str, float | str | bool | None],
) -> pd.DataFrame:
    references = {
        "stage57c": stage57c_metrics,
        "stage58c": stage58c_metrics,
        "stage59c": stage59c_metrics,
        "stage60c": stage60c_metrics,
    }

    def build_row(
        model_name: str,
        source_dir: Path,
        metrics: dict[str, float],
        include_ccra_fields: bool = False,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "model_name": model_name,
            "source_dir": relative_path_str(ROOT, source_dir),
            "AUC": round_or_none(metrics.get("AUC")),
            "ACC": round_or_none(metrics.get("ACC")),
            "F1": round_or_none(metrics.get("F1")),
            "Balanced_ACC": round_or_none(metrics.get("Balanced_ACC")),
            "BACC": round_or_none(metrics.get("Balanced_ACC")),
            "PR_AUC": round_or_none(metrics.get("PR_AUC")),
            "visual_ratio_mean": None,
            "concept_ratio_mean": None,
            "csg_ratio_mean": None,
            "learned_ccra_alpha_mean": None,
            "low_ccra_delta_abs_mean": None,
            "high_ccra_delta_abs_mean": None,
        }
        if include_ccra_fields:
            row.update(
                {
                    "visual_ratio_mean": round_or_none(contribution_means.get("visual_ratio_mean")),
                    "concept_ratio_mean": round_or_none(contribution_means.get("concept_ratio_mean")),
                    "csg_ratio_mean": round_or_none(contribution_means.get("csg_ratio_mean")),
                    "learned_ccra_alpha_mean": round_or_none(ccra_means.get("learned_alpha_final")),
                    "low_ccra_delta_abs_mean": round_or_none(ccra_means.get("low_ccra_delta_abs_mean")),
                    "high_ccra_delta_abs_mean": round_or_none(ccra_means.get("high_ccra_delta_abs_mean")),
                }
            )
        for ref_name, ref_metrics in references.items():
            for metric_name, key in [
                ("auc", "AUC"),
                ("acc", "ACC"),
                ("f1", "F1"),
                ("bacc", "Balanced_ACC"),
                ("pr_auc", "PR_AUC"),
            ]:
                lhs = metrics.get(key, math.nan)
                rhs = ref_metrics.get(key, math.nan)
                row[f"delta_vs_{ref_name}_{metric_name}"] = (
                    round_or_none(lhs - rhs) if not pd.isna(lhs) and not pd.isna(rhs) else None
                )
        return row

    return pd.DataFrame(
        [
            build_row("stage57C_rce_v2_baseline", stage57c_dir, stage57c_metrics),
            build_row("stage58C_residual_constrained_configD", stage58c_dir, stage58c_metrics),
            build_row("stage59C_dynamic_csg_configA", stage59c_dir, stage59c_metrics),
            build_row("stage60C_ccra_configD", stage60c_dir, stage60c_metrics),
            build_row("stage60D_ccra_configC", stage60d_dir, stage60d_metrics, include_ccra_fields=True),
        ]
    )


def build_step60d_fold_metrics_df(
    stage57c_fold_df: pd.DataFrame | None,
    stage58c_fold_df: pd.DataFrame | None,
    stage59c_fold_df: pd.DataFrame | None,
    stage60c_fold_df: pd.DataFrame | None,
    stage60d_fold_df: pd.DataFrame | None,
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
    stage60c_dir: Path,
    stage60d_dir: Path,
) -> pd.DataFrame:
    base_df = build_fold_metrics_df(
        stage57c_fold_df=stage57c_fold_df,
        stage58c_fold_df=stage58c_fold_df,
        stage59c_fold_df=stage59c_fold_df,
        stage60c_fold_df=stage60c_fold_df,
        stage57c_dir=stage57c_dir,
        stage58c_dir=stage58c_dir,
        stage59c_dir=stage59c_dir,
        stage60c_dir=stage60c_dir,
    )
    rows: list[dict[str, object]] = []
    if not base_df.empty:
        rows.extend(base_df.to_dict(orient="records"))
    if stage60d_fold_df is not None and not stage60d_fold_df.empty:
        for _, row in stage60d_fold_df.iterrows():
            rows.append(
                {
                    "model_name": "stage60D_ccra_configC",
                    "source_dir": relative_path_str(ROOT, stage60d_dir),
                    "fold": int(row["fold"]),
                    "AUC": round_or_none(row["AUC"]),
                    "ACC": round_or_none(row["ACC"]),
                    "F1": round_or_none(row["F1"]),
                    "Balanced_ACC": round_or_none(row["Balanced_ACC"]),
                    "BACC": round_or_none(row["Balanced_ACC"]),
                    "PR_AUC": round_or_none(row["PR_AUC"]),
                }
            )
    return pd.DataFrame(rows)


def decide_outcome(
    stage58c_metrics: dict[str, float],
    stage60c_metrics: dict[str, float],
    stage60d_metrics: dict[str, float],
    stage60d_fold_df: pd.DataFrame | None,
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
    ccra_means: dict[str, float | str | bool | None],
    ccra_df: pd.DataFrame,
    source_status: str,
) -> dict[str, object]:
    required = ("ACC", "AUC", "F1", "Balanced_ACC", "PR_AUC")
    if source_status != "completed" or any(pd.isna(stage60d_metrics[key]) for key in required):
        return {
            "decision": "pending",
            "next_step": "run_target_config_c_training",
            "reasons": ["config C 5-fold results incomplete"],
        }

    delta_vs_step58c = {
        "auc": stage60d_metrics["AUC"] - stage58c_metrics["AUC"],
        "acc": stage60d_metrics["ACC"] - stage58c_metrics["ACC"],
        "f1": stage60d_metrics["F1"] - stage58c_metrics["F1"],
        "bacc": stage60d_metrics["Balanced_ACC"] - stage58c_metrics["Balanced_ACC"],
        "pr_auc": stage60d_metrics["PR_AUC"] - stage58c_metrics["PR_AUC"],
    }
    delta_vs_step60c = {
        "auc": stage60d_metrics["AUC"] - stage60c_metrics["AUC"],
        "acc": stage60d_metrics["ACC"] - stage60c_metrics["ACC"],
        "f1": stage60d_metrics["F1"] - stage60c_metrics["F1"],
        "bacc": stage60d_metrics["Balanced_ACC"] - stage60c_metrics["Balanced_ACC"],
        "pr_auc": stage60d_metrics["PR_AUC"] - stage60c_metrics["PR_AUC"],
    }

    visual_ratio_mean = contribution_means.get("visual_ratio_mean", math.nan)
    concept_ratio_mean = contribution_means.get("concept_ratio_mean", math.nan)
    csg_ratio_mean = contribution_means.get("csg_ratio_mean", math.nan)
    learned_alpha_mean = safe_float(ccra_means.get("learned_alpha_final"))
    low_delta_mean = safe_float(ccra_means.get("low_ccra_delta_abs_mean"))
    high_delta_mean = safe_float(ccra_means.get("high_ccra_delta_abs_mean"))

    visual_low = not pd.isna(visual_ratio_mean) and visual_ratio_mean < 0.5
    visual_not_rebounded = not pd.isna(visual_ratio_mean) and visual_ratio_mean < STEP57B_VISUAL_BASELINE
    concept_high = not pd.isna(concept_ratio_mean) and concept_ratio_mean > 0.5
    concept_above_step57b = not pd.isna(concept_ratio_mean) and concept_ratio_mean > STEP57B_CONCEPT_BASELINE

    alpha_nonzero_folds: list[int] = []
    delta_nonzero_folds: list[int] = []
    anomaly_folds: list[int] = []
    for _, row in ccra_df.iterrows():
        fold = int(row["fold"])
        alpha_value = safe_float(row.get("learned_alpha_final"))
        low_delta = safe_float(row.get("low_ccra_delta_abs_mean"))
        high_delta = safe_float(row.get("high_ccra_delta_abs_mean"))
        anomaly_count = int(safe_float(row.get("anomaly_count")) or 0)
        if alpha_value is not None and abs(alpha_value) > 1e-6:
            alpha_nonzero_folds.append(fold)
        if (low_delta is not None and abs(low_delta) > 1e-6) or (
            high_delta is not None and abs(high_delta) > 1e-6
        ):
            delta_nonzero_folds.append(fold)
        if anomaly_count > 0:
            anomaly_folds.append(fold)

    alpha_nonzero_multi_fold = len(alpha_nonzero_folds) >= 2
    delta_nonzero_multi_fold = len(delta_nonzero_folds) >= 2

    full_row = branch_mean_df.loc[branch_mean_df["branch"] == "full"]
    concept_row = branch_mean_df.loc[branch_mean_df["branch"] == "concept_only"]
    full_acc_gap = math.nan
    full_auc_gap = math.nan
    branch_ok = False
    if not full_row.empty and not concept_row.empty:
        full_acc_gap = to_float(full_row.iloc[0]["ACC"]) - to_float(concept_row.iloc[0]["ACC"])
        full_auc_gap = to_float(full_row.iloc[0]["AUC"]) - to_float(concept_row.iloc[0]["AUC"])
        branch_ok = full_acc_gap >= -0.01 and full_auc_gap >= -0.01

    collapse_folds: list[int] = []
    if stage60d_fold_df is not None and not stage60d_fold_df.empty:
        for _, row in stage60d_fold_df.iterrows():
            if to_float(row["ACC"]) < 0.80 or to_float(row["AUC"]) < 0.85:
                collapse_folds.append(int(row["fold"]))
    severe_fold_collapse = bool(collapse_folds)

    beats_step60c = (
        delta_vs_step60c["auc"] > 0
        and delta_vs_step60c["acc"] > 0
        and delta_vs_step60c["f1"] > 0
        and delta_vs_step60c["bacc"] > 0
    )
    close_to_step58c = (
        delta_vs_step58c["acc"] >= -0.01
        and delta_vs_step58c["auc"] >= -0.01
        and delta_vs_step58c["f1"] >= -0.02
        and delta_vs_step58c["bacc"] >= -0.02
    )
    strong_secondary = close_to_step58c and beats_step60c

    if (
        strong_secondary
        and visual_low
        and visual_not_rebounded
        and concept_high
        and concept_above_step57b
        and alpha_nonzero_multi_fold
        and delta_nonzero_multi_fold
        and branch_ok
        and not severe_fold_collapse
        and not anomaly_folds
    ):
        decision = "secondary_preferred_over_config_d"
        next_step = "use_config_c_as_ccra_representative"
    elif (
        close_to_step58c
        and visual_not_rebounded
        and concept_above_step57b
        and alpha_nonzero_multi_fold
        and delta_nonzero_multi_fold
        and branch_ok
        and not severe_fold_collapse
    ):
        decision = "secondary_valid_keep_for_archive"
        next_step = "archive_config_c_as_balanced_secondary"
    else:
        decision = "secondary_not_selected"
        next_step = "retain_step60c_tradeoff_conclusion"

    reasons = [
        f"delta_vs_step58c_acc={delta_vs_step58c['acc']:+.6f}",
        f"delta_vs_step58c_auc={delta_vs_step58c['auc']:+.6f}",
        f"delta_vs_step58c_f1={delta_vs_step58c['f1']:+.6f}",
        f"delta_vs_step58c_bacc={delta_vs_step58c['bacc']:+.6f}",
        f"delta_vs_step58c_pr_auc={delta_vs_step58c['pr_auc']:+.6f}",
        f"delta_vs_step60c_acc={delta_vs_step60c['acc']:+.6f}",
        f"delta_vs_step60c_auc={delta_vs_step60c['auc']:+.6f}",
        f"delta_vs_step60c_f1={delta_vs_step60c['f1']:+.6f}",
        f"delta_vs_step60c_bacc={delta_vs_step60c['bacc']:+.6f}",
        f"delta_vs_step60c_pr_auc={delta_vs_step60c['pr_auc']:+.6f}",
        f"visual_ratio_mean={format_metric(visual_ratio_mean)} vs step57B={STEP57B_VISUAL_BASELINE:.6f}",
        f"concept_ratio_mean={format_metric(concept_ratio_mean)} vs step57B={STEP57B_CONCEPT_BASELINE:.6f}",
        f"csg_ratio_mean={format_metric(csg_ratio_mean)}",
        f"learned_ccra_alpha_mean={format_metric(learned_alpha_mean)}",
        f"low_ccra_delta_abs_mean={format_metric(low_delta_mean)}",
        f"high_ccra_delta_abs_mean={format_metric(high_delta_mean)}",
        f"alpha_nonzero_folds={alpha_nonzero_folds}",
        f"delta_nonzero_folds={delta_nonzero_folds}",
        f"full_minus_concept_acc={format_metric(full_acc_gap)}",
        f"full_minus_concept_auc={format_metric(full_auc_gap)}",
        f"anomaly_folds={anomaly_folds}",
        f"collapse_folds={collapse_folds}",
    ]

    return {
        "decision": decision,
        "next_step": next_step,
        "close_to_step58c": close_to_step58c,
        "beats_step60c": beats_step60c,
        "visual_low": visual_low,
        "visual_not_rebounded": visual_not_rebounded,
        "concept_high": concept_high,
        "concept_above_step57b": concept_above_step57b,
        "alpha_nonzero_multi_fold": alpha_nonzero_multi_fold,
        "delta_nonzero_multi_fold": delta_nonzero_multi_fold,
        "alpha_nonzero_folds": alpha_nonzero_folds,
        "delta_nonzero_folds": delta_nonzero_folds,
        "branch_ok": branch_ok,
        "severe_fold_collapse": severe_fold_collapse,
        "collapse_folds": collapse_folds,
        "anomaly_folds": anomaly_folds,
        "delta_vs_step58c": {k: round_or_none(v) for k, v in delta_vs_step58c.items()},
        "delta_vs_step60c": {k: round_or_none(v) for k, v in delta_vs_step60c.items()},
        "full_minus_concept": {
            "acc": round_or_none(full_acc_gap),
            "auc": round_or_none(full_auc_gap),
        },
        "reasons": reasons,
    }


def build_summary_md(
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage59c_metrics: dict[str, float],
    stage60c_metrics: dict[str, float],
    stage60d_metrics: dict[str, float],
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
    stage60c_dir: Path,
    stage60d_dir: Path,
    source_status: str,
    completed_folds: list[int],
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
    ccra_means: dict[str, float | str | bool | None],
    decision_payload: dict[str, object],
) -> str:
    full_row = branch_mean_df.loc[branch_mean_df["branch"] == "full"]
    concept_row = branch_mean_df.loc[branch_mean_df["branch"] == "concept_only"]
    visual_row = branch_mean_df.loc[branch_mean_df["branch"] == "visual_only"]

    lines = [
        "# Step60D CCRA Config C secondary formal audit",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。仅复用现有 Step57B / Step60C 审计逻辑。",
        f"3. config C 5-fold 结果是否已存在且完整：{'是' if source_status == 'completed' else '否'}。",
        "4. Step60D config C 的 5-fold AUC / ACC / F1 / BACC / PR-AUC："
        f" {format_metric(stage60d_metrics['AUC'])} / {format_metric(stage60d_metrics['ACC'])} /"
        f" {format_metric(stage60d_metrics['F1'])} / {format_metric(stage60d_metrics['Balanced_ACC'])} /"
        f" {format_metric(stage60d_metrics['PR_AUC'])}。",
        "5. 相比 Stage57C baseline 的差异："
        f" AUC {format_delta(stage60d_metrics['AUC'] - stage57c_metrics['AUC'])},"
        f" ACC {format_delta(stage60d_metrics['ACC'] - stage57c_metrics['ACC'])},"
        f" F1 {format_delta(stage60d_metrics['F1'] - stage57c_metrics['F1'])},"
        f" BACC {format_delta(stage60d_metrics['Balanced_ACC'] - stage57c_metrics['Balanced_ACC'])},"
        f" PR-AUC {format_delta(stage60d_metrics['PR_AUC'] - stage57c_metrics['PR_AUC'])}。",
        "6. 相比 Step58C residual-constrained baseline 的差异："
        f" AUC {format_delta(stage60d_metrics['AUC'] - stage58c_metrics['AUC'])},"
        f" ACC {format_delta(stage60d_metrics['ACC'] - stage58c_metrics['ACC'])},"
        f" F1 {format_delta(stage60d_metrics['F1'] - stage58c_metrics['F1'])},"
        f" BACC {format_delta(stage60d_metrics['Balanced_ACC'] - stage58c_metrics['Balanced_ACC'])},"
        f" PR-AUC {format_delta(stage60d_metrics['PR_AUC'] - stage58c_metrics['PR_AUC'])}。",
        "7. 相比 Step59C Dynamic CSG reference 的差异："
        f" AUC {format_delta(stage60d_metrics['AUC'] - stage59c_metrics['AUC'])},"
        f" ACC {format_delta(stage60d_metrics['ACC'] - stage59c_metrics['ACC'])},"
        f" F1 {format_delta(stage60d_metrics['F1'] - stage59c_metrics['F1'])},"
        f" BACC {format_delta(stage60d_metrics['Balanced_ACC'] - stage59c_metrics['Balanced_ACC'])},"
        f" PR-AUC {format_delta(stage60d_metrics['PR_AUC'] - stage59c_metrics['PR_AUC'])}。",
        "8. 相比 Step60C config D 的差异："
        f" AUC {format_delta(stage60d_metrics['AUC'] - stage60c_metrics['AUC'])},"
        f" ACC {format_delta(stage60d_metrics['ACC'] - stage60c_metrics['ACC'])},"
        f" F1 {format_delta(stage60d_metrics['F1'] - stage60c_metrics['F1'])},"
        f" BACC {format_delta(stage60d_metrics['Balanced_ACC'] - stage60c_metrics['Balanced_ACC'])},"
        f" PR-AUC {format_delta(stage60d_metrics['PR_AUC'] - stage60c_metrics['PR_AUC'])}。",
        "9. visual_ratio 是否仍保持低水平："
        f" {'是' if decision_payload.get('visual_low') and decision_payload.get('visual_not_rebounded') else '否'}，"
        f"{format_metric(contribution_means.get('visual_ratio_mean'))}。",
        "10. concept_ratio 是否仍保持高水平："
        f" {'是' if decision_payload.get('concept_high') and decision_payload.get('concept_above_step57b') else '否'}，"
        f"{format_metric(contribution_means.get('concept_ratio_mean'))}。",
        "11. learned CCRA alpha 是否在多个 fold 中非零："
        f" {'是' if decision_payload.get('alpha_nonzero_multi_fold') else '否'}，"
        f"folds={decision_payload.get('alpha_nonzero_folds', [])}。",
        "12. low/high CCRA delta 是否在多个 fold 中非零："
        f" {'是' if decision_payload.get('delta_nonzero_multi_fold') else '否'}，"
        f"folds={decision_payload.get('delta_nonzero_folds', [])}。",
        "13. full / concept_only / visual_only 的 5-fold branch 表现如何：",
    ]
    if not full_row.empty:
        lines.append(
            f"   full: ACC {format_metric(full_row.iloc[0]['ACC'])}, AUC {format_metric(full_row.iloc[0]['AUC'])}, F1 {format_metric(full_row.iloc[0]['F1'])}。"
        )
    else:
        lines.append("   full: NA。")
    if not concept_row.empty:
        lines.append(
            f"   concept_only: ACC {format_metric(concept_row.iloc[0]['ACC'])}, AUC {format_metric(concept_row.iloc[0]['AUC'])}, F1 {format_metric(concept_row.iloc[0]['F1'])}。"
        )
    else:
        lines.append("   concept_only: NA。")
    if not visual_row.empty:
        lines.append(
            f"   visual_only: ACC {format_metric(visual_row.iloc[0]['ACC'])}, AUC {format_metric(visual_row.iloc[0]['AUC'])}, F1 {format_metric(visual_row.iloc[0]['F1'])}。"
        )
    else:
        lines.append("   visual_only: NA。")

    lines.extend(
        [
            "14. Step60D 最终结论："
            f" {decision_payload.get('decision', 'pending')}。",
            "15. 下一步建议："
            + (
                " 用 config C 作为 CCRA 更均衡的正式代表。"
                if decision_payload.get("decision") == "secondary_preferred_over_config_d"
                else " 将 config C 作为平衡型 secondary 归档。"
                if decision_payload.get("decision") == "secondary_valid_keep_for_archive"
                else " 保留 Step60C 的 tradeoff 结论，不替换 config D 正式结论。"
                if decision_payload.get("decision") == "secondary_not_selected"
                else " 需要补跑 TARGET_CONFIG=C。"
            ),
            "",
            "## Status",
            "",
            f"- Stage57C baseline dir: `{relative_path_str(ROOT, stage57c_dir)}`",
            f"- Stage58C baseline dir: `{relative_path_str(ROOT, stage58c_dir)}`",
            f"- Stage59C reference dir: `{relative_path_str(ROOT, stage59c_dir)}`",
            f"- Stage60C config D dir: `{relative_path_str(ROOT, stage60c_dir)}`",
            f"- Stage60D config C source dir: `{relative_path_str(ROOT, stage60d_dir)}`",
            f"- source_status: `{source_status}`",
            f"- completed_folds_detected: `{completed_folds}`",
            f"- ccra_norm: `{ccra_means.get('ccra_norm')}`",
            f"- ccra_query_source: `{ccra_means.get('ccra_query_source')}`",
            "",
            "## Decision Basis",
            "",
        ]
    )
    for reason in decision_payload.get("reasons", []):
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    display_root = detect_display_root(args.root)
    write_run_commands(output_dir, display_root)

    stage57c_metrics, _ = discover_aggregate_metrics(args.stage57c_dir)
    stage57c_fold_df, _ = discover_fold_metrics(args.stage57c_dir)
    stage58c_metrics, _ = discover_aggregate_metrics(args.stage58c_dir)
    stage58c_fold_df, _ = discover_fold_metrics(args.stage58c_dir)
    stage59c_metrics, _ = discover_aggregate_metrics(args.stage59c_dir)
    stage59c_fold_df, _ = discover_fold_metrics(args.stage59c_dir)
    stage60c_metrics, _ = discover_aggregate_metrics(args.stage60c_dir)
    stage60c_fold_df, _ = discover_fold_metrics(args.stage60c_dir)
    stage60d_metrics, stage60d_agg_path = discover_aggregate_metrics(args.stage60d_source_dir)
    stage60d_fold_df, stage60d_fold_path = discover_fold_metrics(args.stage60d_source_dir)
    source_status, checkpoint_folds = determine_run_status(args.stage60d_source_dir, stage60d_fold_df)

    completed_folds = collect_completed_checkpoint_folds(args.stage60d_source_dir)
    if not completed_folds:
        completed_folds = sorted(set(checkpoint_folds))

    branch_df = pd.DataFrame(
        columns=["fold", "branch", "available", "num_samples", "ACC", "BACC", "F1", "AUC", "PR_AUC"]
    )
    contribution_df = pd.DataFrame(
        columns=[
            "fold",
            "visual_ratio_mean",
            "visual_ratio_median",
            "visual_ratio_gt_0_5_percent",
            "concept_ratio_mean",
            "concept_ratio_median",
            "csg_ratio_mean",
            "csg_ratio_median",
            "full_margin_mean",
            "concept_margin_mean",
            "visual_margin_mean",
            "csg_margin_mean",
        ]
    )
    ccra_df = pd.DataFrame(
        columns=[
            "fold",
            "learned_alpha_final",
            "ccra_enabled",
            "ccra_scale",
            "ccra_norm",
            "ccra_dropout",
            "ccra_clip",
            "ccra_query_source",
            "detach_prompt",
            "low_ccra_delta_abs_mean",
            "high_ccra_delta_abs_mean",
            "low_original_region_norm",
            "high_original_region_norm",
            "low_fused_region_norm",
            "high_fused_region_norm",
            "low_ccra_region_norm",
            "high_ccra_region_norm",
            "low_ccra_delta_vs_original_ratio",
            "high_ccra_delta_vs_original_ratio",
            "status",
            "anomaly_count",
        ]
    )
    audit_status: dict[str, object] = {}
    warnings: list[str] = []

    if completed_folds:
        branch_df, contribution_df, ccra_df, audit_status, warnings = collect_audits_and_ccra(
            run_dir=args.stage60d_source_dir,
            fold_ids=completed_folds,
            output_dir=output_dir,
            audit_script=args.audit_script,
        )
        if not ccra_df.empty:
            ccra_df["ccra_scale"] = ccra_df["ccra_scale"].fillna(STEP60D_CCRA_CONFIG["rce_ccra_scale"])
            ccra_df["ccra_norm"] = ccra_df["ccra_norm"].fillna(STEP60D_CCRA_CONFIG["rce_ccra_norm"])
            ccra_df["ccra_dropout"] = ccra_df["ccra_dropout"].fillna(STEP60D_CCRA_CONFIG["rce_ccra_dropout"])
            ccra_df["ccra_clip"] = ccra_df["ccra_clip"].fillna(STEP60D_CCRA_CONFIG["rce_ccra_clip"])
            ccra_df["ccra_query_source"] = ccra_df["ccra_query_source"].fillna(
                STEP60D_CCRA_CONFIG["rce_ccra_query_source"]
            )
            ccra_df["detach_prompt"] = ccra_df["detach_prompt"].fillna(
                STEP60D_CCRA_CONFIG["rce_ccra_detach_prompt"]
            )
    else:
        warnings.append("config_c_source_incomplete_or_missing")

    stage58c_contrib_df = read_csv_if_exists(args.stage58c_dir.parent / "stage58C_contribution_by_fold.csv")
    contribution_df = add_delta_vs_step58c_contribution(contribution_df, stage58c_contrib_df)
    contribution_means = summarize_contribution_means(contribution_df)
    ccra_means = summarize_ccra_means(ccra_df)
    branch_mean_df = summarize_branch_means(branch_df)
    stage60d_mean_metrics = compute_mean_metrics(stage60d_fold_df)
    if not any(pd.isna(stage60d_mean_metrics[key]) for key in stage60d_mean_metrics):
        stage60d_metrics = stage60d_mean_metrics

    compare_df = build_compare_df(
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage59c_metrics=stage59c_metrics,
        stage60c_metrics=stage60c_metrics,
        stage60d_metrics=stage60d_metrics,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
        stage60c_dir=args.stage60c_dir,
        stage60d_dir=args.stage60d_source_dir,
        contribution_means=contribution_means,
        ccra_means=ccra_means,
    )
    fold_metrics_df = build_step60d_fold_metrics_df(
        stage57c_fold_df=stage57c_fold_df,
        stage58c_fold_df=stage58c_fold_df,
        stage59c_fold_df=stage59c_fold_df,
        stage60c_fold_df=stage60c_fold_df,
        stage60d_fold_df=stage60d_fold_df,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
        stage60c_dir=args.stage60c_dir,
        stage60d_dir=args.stage60d_source_dir,
    )
    decision_payload = decide_outcome(
        stage58c_metrics=stage58c_metrics,
        stage60c_metrics=stage60c_metrics,
        stage60d_metrics=stage60d_metrics,
        stage60d_fold_df=stage60d_fold_df,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
        ccra_means=ccra_means,
        ccra_df=ccra_df,
        source_status=source_status,
    )
    summary_text = build_summary_md(
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage59c_metrics=stage59c_metrics,
        stage60c_metrics=stage60c_metrics,
        stage60d_metrics=stage60d_metrics,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
        stage60c_dir=args.stage60c_dir,
        stage60d_dir=args.stage60d_source_dir,
        source_status=source_status,
        completed_folds=completed_folds,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
        ccra_means=ccra_means,
        decision_payload=decision_payload,
    )

    compare_df.to_csv(output_dir / "stage60D_compare_with_baselines.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "stage60D_fold_metrics.csv", index=False)
    branch_df.to_csv(output_dir / "stage60D_branch_metrics_by_fold.csv", index=False)
    contribution_df.to_csv(output_dir / "stage60D_contribution_by_fold.csv", index=False)
    ccra_df.to_csv(output_dir / "stage60D_ccra_by_fold.csv", index=False)
    (output_dir / "stage60D_summary.md").write_text(summary_text, encoding="utf-8")
    (output_dir / "stage60D_decision.json").write_text(
        json.dumps(
            {
                **decision_payload,
                "config_c_source_status": source_status,
                "expected_ccra_config": STEP60D_CCRA_CONFIG,
                "stage57c_baseline_metrics": {
                    key: round_or_none(value) for key, value in stage57c_metrics.items()
                },
                "stage58c_baseline_metrics": {
                    key: round_or_none(value) for key, value in stage58c_metrics.items()
                },
                "stage59c_reference_metrics": {
                    key: round_or_none(value) for key, value in stage59c_metrics.items()
                },
                "stage60c_configd_metrics": {
                    key: round_or_none(value) for key, value in stage60c_metrics.items()
                },
                "stage60d_configc_metrics": {
                    key: round_or_none(value) for key, value in stage60d_metrics.items()
                },
                "contribution_means": {
                    key: round_or_none(value) for key, value in contribution_means.items()
                },
                "ccra_means": {
                    key: (
                        round_or_none(value)
                        if key not in {"ccra_norm", "ccra_query_source", "detach_prompt"}
                        else value
                    )
                    for key, value in ccra_means.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "stage60D_status.json").write_text(
        json.dumps(
            {
                "status": source_status,
                "completed_folds": completed_folds,
                "stage60d_source_dir": relative_path_str(ROOT, args.stage60d_source_dir),
                "stage60d_aggregate_source": None
                if stage60d_agg_path is None
                else relative_path_str(ROOT, stage60d_agg_path),
                "stage60d_fold_source": None
                if stage60d_fold_path is None
                else relative_path_str(ROOT, stage60d_fold_path),
                "audit_status": audit_status,
                "warnings": warnings,
                "pending_train_command": "TARGET_CONFIG=C RUN_TRAIN=1 bash scripts/experiments/run_stage60C_ccra_configD_5fold.sh",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
