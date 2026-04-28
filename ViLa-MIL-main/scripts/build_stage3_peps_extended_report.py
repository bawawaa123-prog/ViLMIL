from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
OUT_DIR = ROOT / "trained_models" / "stage3_peps_comparison"


def _fmt(value):
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map(_fmt)
    headers = list(display_df.columns)
    rows = display_df.values.tolist()
    sep = ["---"] * len(headers)
    table_rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    table_rows.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(table_rows)


def _best_overall_peps(summary_df: pd.DataFrame) -> str:
    peps_df = summary_df[
        summary_df["experiment"].str.contains("PEPS", na=False)
        & (summary_df["status"] == "ok")
    ].copy()
    if peps_df.empty:
        return "No completed PEPS experiment available."
    ranked = peps_df.sort_values(
        ["test_auc_mean", "test_acc_mean", "test_f1_mean", "balanced_acc_mean"],
        ascending=[False, False, False, False],
    )
    best = ranked.iloc[0]
    return (
        f"{best['experiment']} is the current primary recommendation "
        f"(test_auc={best['test_auc_mean']:.6f}, test_acc={best['test_acc_mean']:.6f}, "
        f"test_f1={best['test_f1_mean']:.6f}, balanced_acc={best['balanced_acc_mean']:.6f})."
    )


def main():
    summary_csv = OUT_DIR / "peps_extended_summary.csv"
    folds_csv = OUT_DIR / "peps_extended_folds.csv"
    delta_csv = OUT_DIR / "peps_fold_delta_analysis.csv"
    win_csv = OUT_DIR / "peps_fold_delta_winloss.csv"
    usage_overall_csv = OUT_DIR / "peps_prompt_usage_overall.csv"
    usage_fold_csv = OUT_DIR / "peps_prompt_usage_summary.csv"
    usage_top_csv = OUT_DIR / "peps_prompt_usage_top_prompts.csv"
    usage_top_overall_csv = OUT_DIR / "peps_prompt_usage_top_prompts_overall.csv"
    report_path = OUT_DIR / "peps_extended_report.md"

    summary_df = pd.read_csv(summary_csv) if summary_csv.is_file() else pd.DataFrame()
    folds_df = pd.read_csv(folds_csv) if folds_csv.is_file() else pd.DataFrame()
    delta_df = pd.read_csv(delta_csv) if delta_csv.is_file() else pd.DataFrame()
    win_df = pd.read_csv(win_csv) if win_csv.is_file() else pd.DataFrame()
    usage_overall_df = pd.read_csv(usage_overall_csv) if usage_overall_csv.is_file() else pd.DataFrame()
    usage_fold_df = pd.read_csv(usage_fold_csv) if usage_fold_csv.is_file() else pd.DataFrame()
    usage_top_df = pd.read_csv(usage_top_csv) if usage_top_csv.is_file() else pd.DataFrame()
    usage_top_overall_df = pd.read_csv(usage_top_overall_csv) if usage_top_overall_csv.is_file() else pd.DataFrame()

    tau_subset = summary_df[
        summary_df["experiment"].isin(
            [
                "Concept-12 PEPS topk=3 tau=0.07",
                "Concept-12 PEPS topk=3 tau=0.1",
                "Concept-12 PEPS topk=3 tau=0.15",
                "Concept-12 PEPS topk=5 tau=0.07",
                "Concept-12 PEPS topk=5 tau=0.1",
                "Concept-12 PEPS topk=5 tau=0.15",
            ]
        )
    ].copy()

    topk_compare = tau_subset.copy()
    delta_focus = delta_df.copy()
    win_focus = win_df.copy()

    diversity_focus = usage_overall_df[
        usage_overall_df["experiment"].isin(
            [
                "Concept-12 MLP dynamic_gate",
                "Concept-12 PEPS topk=3 tau=0.1",
                "Concept-12 PEPS topk=5 tau=0.1",
            ]
        )
    ].copy()

    collapse_focus = usage_fold_df[
        usage_fold_df["experiment"].isin(
            [
                "Concept-12 MLP dynamic_gate",
                "Concept-12 PEPS topk=3 tau=0.1",
                "Concept-12 PEPS topk=5 tau=0.1",
            ]
        )
    ].copy()

    top_prompt_focus = usage_top_overall_df[
        usage_top_overall_df["experiment"].isin(
            [
                "Concept-12 PEPS topk=3 tau=0.1",
                "Concept-12 PEPS topk=5 tau=0.1",
            ]
        )
    ].copy()

    lines = []
    lines.append("# Stage3 PEPS Extended Report")
    lines.append("")
    lines.append("## Current Recommendation")
    lines.append("")
    lines.append(_best_overall_peps(summary_df))
    lines.append("")
    lines.append("## All Current PEPS Results")
    lines.append("")
    if not summary_df.empty:
        display_cols = [
            "experiment",
            "status",
            "test_auc_mean",
            "test_acc_mean",
            "test_f1_mean",
            "val_auc_mean",
            "balanced_acc_mean",
            "sensitivity_mean",
            "specificity_mean",
            "pr_auc_mean",
        ]
        lines.append(_table(summary_df[display_cols]))
    else:
        lines.append("_No summary data available._")
    lines.append("")
    lines.append("## Tau Sweep Results")
    lines.append("")
    if not tau_subset.empty:
        display_cols = [
            "experiment",
            "status",
            "test_auc_mean",
            "test_acc_mean",
            "test_f1_mean",
            "balanced_acc_mean",
            "sensitivity_mean",
            "specificity_mean",
            "pr_auc_mean",
        ]
        lines.append(_table(tau_subset[display_cols]))
    else:
        lines.append("_Tau sweep results are not available yet._")
    lines.append("")
    lines.append("## Topk=3 vs Topk=5")
    lines.append("")
    if not topk_compare.empty:
        display_cols = [
            "experiment",
            "test_auc_mean",
            "test_acc_mean",
            "test_f1_mean",
            "balanced_acc_mean",
            "specificity_mean",
            "pr_auc_mean",
        ]
        lines.append(_table(topk_compare[display_cols]))
    else:
        lines.append("_No topk comparison data available._")
    lines.append("")
    lines.append("## Fold-Level Delta Analysis")
    lines.append("")
    if not delta_focus.empty:
        delta_cols = [
            "candidate_experiment",
            "reference_experiment",
            "fold",
            "delta_test_auc",
            "delta_test_acc",
            "delta_test_f1",
            "delta_balanced_acc",
            "delta_sensitivity",
            "delta_specificity",
            "delta_pr_auc",
        ]
        lines.append(_table(delta_focus[delta_cols]))
        lines.append("")
        lines.append("### Win/Loss Counts")
        lines.append("")
        lines.append(_table(win_focus))
    else:
        lines.append("_No fold delta analysis available._")
    lines.append("")
    lines.append("## Prompt Usage Diversity")
    lines.append("")
    if not diversity_focus.empty:
        lines.append(_table(diversity_focus))
    else:
        lines.append("_No prompt diversity summary available._")
    lines.append("")
    lines.append("## Prompt Collapse Check")
    lines.append("")
    if not collapse_focus.empty:
        collapse_cols = [
            "experiment",
            "fold",
            "low_top1_unique_prompts",
            "high_top1_unique_prompts",
            "low_prompt_entropy",
            "high_prompt_entropy",
            "low_prompt_collapse_rate",
            "high_prompt_collapse_rate",
        ]
        lines.append(_table(collapse_focus[collapse_cols]))
    else:
        lines.append("_No collapse analysis available._")
    lines.append("")
    lines.append("## Most Frequent Top-1 Prompts by True Class")
    lines.append("")
    if not top_prompt_focus.empty:
        lines.append(_table(top_prompt_focus))
    else:
        lines.append("_No top-prompt ranking available._")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown report to: {report_path}")


if __name__ == "__main__":
    main()
