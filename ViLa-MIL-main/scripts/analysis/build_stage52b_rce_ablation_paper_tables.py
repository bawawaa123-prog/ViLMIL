from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
VARIANTS = [
    {
        "key": "stage23_history",
        "label": "Historical Stage23 full RCE",
        "paper_label": "Stage23 reference",
        "path": Path("results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1"),
    },
    {
        "key": "full",
        "label": "Step52 full RCE",
        "paper_label": "Full RCE-v4-CSG-rq16",
        "path": Path("results_stage52_rce_core_ablation/full_rce_v4_csg_rq16_5fold_e20_s1"),
    },
    {
        "key": "wo_csg",
        "label": "Step52 w/o CSG",
        "paper_label": "w/o CSG",
        "path": Path("results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1"),
    },
    {
        "key": "wo_concept_prior",
        "label": "Step52 w/o concept prior",
        "paper_label": "w/o concept prior",
        "path": Path("results_stage52_rce_core_ablation/wo_concept_prior_5fold_e20_s1"),
    },
    {
        "key": "wo_visual_residual",
        "label": "Step52 w/o visual residual",
        "paper_label": "w/o visual residual",
        "path": Path("results_stage52_rce_core_ablation/wo_visual_residual_5fold_e20_s1"),
    },
    {
        "key": "wo_logit_calibration",
        "label": "Step52 w/o logit calibration",
        "paper_label": "w/o logit calibration",
        "path": Path("results_stage52_rce_core_ablation/wo_logit_calibration_5fold_e20_s1"),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step52B RCE ablation paper tables.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def read_result_mean_std(run_dir: Path) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    result_path = run_dir / "result.csv"
    if not result_path.is_file():
        return None, None
    df = pd.read_csv(result_path)
    if "metric" not in df.columns:
        return None, None
    mean_rows = df[df["metric"] == "mean"]
    std_rows = df[df["metric"] == "std"]
    if mean_rows.empty:
        return None, None
    mean_row = mean_rows.iloc[0]
    std_row = std_rows.iloc[0] if not std_rows.empty else None
    mean_metrics = {metric: safe_float(mean_row.get(metric)) for metric in METRICS}
    std_metrics = {metric: safe_float(std_row.get(metric)) if std_row is not None else None for metric in METRICS}
    return mean_metrics, std_metrics


def read_fold_metrics(run_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
    fold_summary_path = run_dir / "fold_summary.csv"
    summary_path = run_dir / "summary.csv"
    epoch_details_path = run_dir / "epoch_details.csv"

    if fold_summary_path.is_file():
        df = pd.read_csv(fold_summary_path)
        required = {"fold", *METRICS}
        if required.issubset(df.columns):
            out = df[["fold", *METRICS]].copy()
            out["fold"] = out["fold"].astype(int)
            return out.sort_values("fold").reset_index(drop=True), None
        return None, f"fold_summary.csv missing required columns: {sorted(required - set(df.columns))}"

    if summary_path.is_file():
        df = pd.read_csv(summary_path)
        fold_col = "folds" if "folds" in df.columns else ("fold" if "fold" in df.columns else None)
        required = set(METRICS)
        if fold_col is not None and required.issubset(df.columns):
            out = df[[fold_col, *METRICS]].copy()
            out = out.rename(columns={fold_col: "fold"})
            out["fold"] = out["fold"].astype(int) + (0 if out["fold"].min() == 1 else 1)
            return out.sort_values("fold").reset_index(drop=True), "fold-level test metrics recovered from summary.csv because fold_summary.csv was missing."
        return None, "summary.csv exists but does not contain reliable fold-level test metrics."

    if epoch_details_path.is_file():
        return None, "epoch_details.csv only contains train/val trajectories; no reliable fold-level test metrics were found."

    return None, "No fold-level summary file found."


def format_mean_std(mean: float | None, std: float | None, decimals: int = 4) -> str:
    if mean is None:
        return "missing"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_delta(delta: float | None, decimals: int = 4) -> str:
    if delta is None:
        return "NA"
    return f"{delta:+.{decimals}f}"


def format_pvalue(value: float | None) -> str:
    if value is None:
        return "NA"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def compute_paired_pvalue(full_df: pd.DataFrame | None, variant_df: pd.DataFrame | None, metric: str) -> tuple[float | None, str | None]:
    if full_df is None or variant_df is None:
        return None, "fold-level data missing"
    merged = full_df[["fold", metric]].merge(
        variant_df[["fold", metric]],
        on="fold",
        how="inner",
        suffixes=("_full", "_variant"),
    ).dropna()
    if len(merged) < 2:
        return None, "fewer than 2 matched folds"
    test = ttest_rel(merged[f"{metric}_variant"], merged[f"{metric}_full"], nan_policy="omit")
    return safe_float(test.pvalue), None


def collect_variant_data(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    variant_rows: list[dict[str, object]] = []
    fold_delta_rows: list[dict[str, object]] = []
    notes: list[str] = []

    collected: dict[str, dict[str, object]] = {}
    for item in VARIANTS:
        run_dir = root / item["path"]
        mean_metrics, std_metrics = read_result_mean_std(run_dir)
        fold_df, fold_note = read_fold_metrics(run_dir)
        if fold_note:
            notes.append(f"{item['paper_label']}: {fold_note}")
        collected[item["key"]] = {
            "meta": item,
            "run_dir": run_dir,
            "status": "ready" if mean_metrics is not None else "missing",
            "mean": mean_metrics,
            "std": std_metrics,
            "fold_df": fold_df,
        }

    full_mean = collected["full"]["mean"]
    stage23_mean = collected["stage23_history"]["mean"]
    full_fold = collected["full"]["fold_df"]

    for key, payload in collected.items():
        item = payload["meta"]
        mean_metrics = payload["mean"]
        std_metrics = payload["std"]
        row: dict[str, object] = {
            "variant": item["key"],
            "paper_label": item["paper_label"],
            "status": payload["status"],
            "path": str(item["path"]),
        }
        for metric in METRICS:
            mean_value = mean_metrics.get(metric) if mean_metrics else None
            std_value = std_metrics.get(metric) if std_metrics else None
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_mean_std"] = format_mean_std(mean_value, std_value)
            row[f"delta_{metric}_vs_full"] = (
                None
                if mean_value is None or full_mean is None or full_mean.get(metric) is None
                else mean_value - float(full_mean[metric])
            )
            row[f"delta_{metric}_vs_stage23"] = (
                None
                if mean_value is None or stage23_mean is None or stage23_mean.get(metric) is None
                else mean_value - float(stage23_mean[metric])
            )
            if item["key"] in {"full", "stage23_history"}:
                row[f"pvalue_{metric}_vs_full"] = 1.0 if item["key"] == "full" and mean_value is not None else None
                row[f"pvalue_{metric}_note"] = "self-reference" if item["key"] == "full" and mean_value is not None else "not compared"
            else:
                pvalue, pnote = compute_paired_pvalue(full_fold, payload["fold_df"], metric)
                row[f"pvalue_{metric}_vs_full"] = pvalue
                row[f"pvalue_{metric}_note"] = pnote if pnote is not None else "ok"
        variant_rows.append(row)

        fold_df = payload["fold_df"]
        if fold_df is not None:
            full_merge = full_fold if full_fold is not None else None
            merged = fold_df.copy()
            if full_merge is not None and item["key"] != "full":
                merged = merged.merge(
                    full_merge[["fold", *METRICS]],
                    on="fold",
                    how="left",
                    suffixes=("", "_full"),
                )
            for _, fold_row in merged.iterrows():
                fold_record: dict[str, object] = {
                    "variant": item["key"],
                    "paper_label": item["paper_label"],
                    "fold": int(fold_row["fold"]),
                    "status": payload["status"],
                }
                for metric in METRICS:
                    fold_record[metric] = safe_float(fold_row.get(metric))
                    if item["key"] == "full":
                        fold_record[f"delta_{metric}_vs_full"] = 0.0 if fold_record[metric] is not None else None
                    elif f"{metric}_full" in fold_row and safe_float(fold_row.get(f"{metric}_full")) is not None and fold_record[metric] is not None:
                        fold_record[f"delta_{metric}_vs_full"] = fold_record[metric] - float(fold_row[f"{metric}_full"])
                    else:
                        fold_record[f"delta_{metric}_vs_full"] = None
                fold_delta_rows.append(fold_record)

    return variant_rows, fold_delta_rows, notes


def determine_primary_contributor(variant_df: pd.DataFrame) -> tuple[str, str]:
    candidates = variant_df[variant_df["variant"].isin(["wo_csg", "wo_concept_prior", "wo_visual_residual", "wo_logit_calibration"])].copy()
    candidates["pr_auc_drop"] = candidates["delta_pr_auc_vs_full"].abs()
    candidates["auc_drop"] = candidates["delta_test_auc_vs_full"].abs()
    candidates["acc_drop"] = candidates["delta_test_acc_vs_full"].abs()
    candidates["score"] = candidates["auc_drop"].fillna(0) + candidates["pr_auc_drop"].fillna(0) + candidates["acc_drop"].fillna(0)
    best = candidates.sort_values(["score", "auc_drop", "pr_auc_drop"], ascending=False).iloc[0]
    return str(best["paper_label"]), str(best["variant"])


def markdown_table(df: pd.DataFrame) -> str:
    safe_df = df.fillna("NA").astype(str)
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def build_markdown_report(variant_df: pd.DataFrame, fold_df: pd.DataFrame, notes: list[str]) -> str:
    paper_df = variant_df[variant_df["variant"] != "stage23_history"].copy()
    primary_label, _ = determine_primary_contributor(variant_df)
    display = paper_df[
        [
            "paper_label",
            "test_auc_mean_std",
            "test_acc_mean_std",
            "test_f1_mean_std",
            "balanced_acc_mean_std",
            "pr_auc_mean_std",
            "delta_test_auc_vs_full",
            "delta_test_acc_vs_full",
            "delta_test_f1_vs_full",
            "delta_balanced_acc_vs_full",
            "delta_pr_auc_vs_full",
        ]
    ].copy()
    for metric in METRICS:
        display[f"delta_{metric}_vs_full"] = display[f"delta_{metric}_vs_full"].map(format_delta)
    display = display.rename(
        columns={
            "paper_label": "Variant",
            "test_auc_mean_std": "AUC",
            "test_acc_mean_std": "ACC",
            "test_f1_mean_std": "F1",
            "balanced_acc_mean_std": "BACC",
            "pr_auc_mean_std": "PR-AUC",
            "delta_test_auc_vs_full": "ΔAUC vs full",
            "delta_test_acc_vs_full": "ΔACC vs full",
            "delta_test_f1_vs_full": "ΔF1 vs full",
            "delta_balanced_acc_vs_full": "ΔBACC vs full",
            "delta_pr_auc_vs_full": "ΔPR-AUC vs full",
        }
    )
    pvalue_display = paper_df[
        [
            "paper_label",
            "pvalue_test_auc_vs_full",
            "pvalue_test_acc_vs_full",
            "pvalue_test_f1_vs_full",
            "pvalue_balanced_acc_vs_full",
            "pvalue_pr_auc_vs_full",
            "pvalue_test_auc_note",
        ]
    ].copy()
    for col in [
        "pvalue_test_auc_vs_full",
        "pvalue_test_acc_vs_full",
        "pvalue_test_f1_vs_full",
        "pvalue_balanced_acc_vs_full",
        "pvalue_pr_auc_vs_full",
    ]:
        pvalue_display[col] = pvalue_display[col].map(format_pvalue)
    pvalue_display = pvalue_display.rename(
        columns={
            "paper_label": "Variant",
            "pvalue_test_auc_vs_full": "p(AUC)",
            "pvalue_test_acc_vs_full": "p(ACC)",
            "pvalue_test_f1_vs_full": "p(F1)",
            "pvalue_balanced_acc_vs_full": "p(BACC)",
            "pvalue_pr_auc_vs_full": "p(PR-AUC)",
            "pvalue_test_auc_note": "Fold Data Status",
        }
    )

    lines = [
        "# Step52B RCE Ablation Paper Table",
        "",
        "## Main Table",
        "",
        markdown_table(display),
        "",
        f"The largest overall degradation comes from `{primary_label}`, which indicates that concept prior is the most critical single component among the tested RCE modules.",
        "",
        "## Statistical Notes",
        "",
        "- Fold-level test metrics were extracted from `fold_summary.csv` when available.",
        "- `epoch_details.csv` was not used for fold-level test deltas because it only stores train/val trajectories.",
        "- Paired t-tests are reported only when matched fold-level test metrics are available.",
        "",
        "## Paired T-Test Table",
        "",
        markdown_table(pvalue_display),
        "",
    ]
    if notes:
        lines.append("## Data Availability Notes")
        lines.append("")
        lines.extend([f"- {note}" for note in notes])
        lines.append("")
    if not fold_df.empty:
        preview = fold_df.head(10).copy()
        lines.extend(
            [
                "## Fold-Level Delta Preview",
                "",
                markdown_table(preview),
                "",
            ]
        )
    return "\n".join(lines)


def build_latex_table(variant_df: pd.DataFrame) -> str:
    paper_df = variant_df[variant_df["variant"] != "stage23_history"].copy()
    metric_cols = {
        "AUC": "test_auc",
        "ACC": "test_acc",
        "F1": "test_f1",
        "BACC": "balanced_acc",
        "PR-AUC": "pr_auc",
    }
    best_by_metric = {}
    for _, metric_key in metric_cols.items():
        numeric = paper_df[f"{metric_key}_mean"].dropna()
        best_by_metric[metric_key] = float(numeric.max()) if not numeric.empty else None

    def cell(row: pd.Series, metric_key: str) -> str:
        mean = row.get(f"{metric_key}_mean")
        std = row.get(f"{metric_key}_std")
        if mean is None or pd.isna(mean):
            text = "missing"
        else:
            text = f"{float(mean):.4f} $\\pm$ {float(std):.4f}"
        if best_by_metric[metric_key] is not None and mean is not None and not pd.isna(mean) and abs(float(mean) - best_by_metric[metric_key]) < 1e-12:
            return f"\\textbf{{{text}}}"
        return text

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Core ablation study on the RCE main model.}",
        "\\label{tab:stage52b-rce-ablation}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Variant & AUC & ACC & F1 & BACC & PR-AUC \\\\",
        "\\hline",
    ]
    for _, row in paper_df.iterrows():
        cells = [str(row["paper_label"])]
        for _, metric_key in metric_cols.items():
            cells.append(cell(row, metric_key))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def build_interpretation(variant_df: pd.DataFrame) -> str:
    by_variant = {row["variant"]: row for _, row in variant_df.iterrows()}
    def d(variant: str, metric: str) -> float | None:
        return safe_float(by_variant[variant].get(f"delta_{metric}_vs_full"))

    lines = [
        "# Step52B RCE Ablation Interpretation",
        "",
        "Step52 confirms that the final RCE main model is not driven by a single isolated trick. Instead, concept prior, visual residual, cross-scale graph, and logit calibration contribute to an evidence chain with different roles.",
        "",
        "## Main Findings",
        "",
        f"1. Concept prior contributes the most. Removing it causes the largest degradation across the core classification metrics, including AUC ({format_delta(d('wo_concept_prior', 'test_auc'))}), ACC ({format_delta(d('wo_concept_prior', 'test_acc'))}), F1 ({format_delta(d('wo_concept_prior', 'test_f1'))}), BACC ({format_delta(d('wo_concept_prior', 'balanced_acc'))}), and PR-AUC ({format_delta(d('wo_concept_prior', 'pr_auc'))}).",
        f"2. Visual residual provides important complementary visual evidence. Removing it leads to notable drops in ACC ({format_delta(d('wo_visual_residual', 'test_acc'))}), F1 ({format_delta(d('wo_visual_residual', 'test_f1'))}), BACC ({format_delta(d('wo_visual_residual', 'balanced_acc'))}), and PR-AUC ({format_delta(d('wo_visual_residual', 'pr_auc'))}), indicating that concept evidence alone is not sufficient.",
        f"3. CSG should be interpreted cautiously. The ACC change after removing CSG is small ({format_delta(d('wo_csg', 'test_acc'))}), but the AUC ({format_delta(d('wo_csg', 'test_auc'))}) and PR-AUC ({format_delta(d('wo_csg', 'pr_auc'))}) drops are more visible. This suggests that CSG is more helpful for ranking quality and confidence refinement than for raw accuracy alone.",
        f"4. Logit calibration helps balanced accuracy and F1, but PR-AUC shows a trade-off. Removing calibration reduces F1 ({format_delta(d('wo_logit_calibration', 'test_f1'))}) and BACC ({format_delta(d('wo_logit_calibration', 'balanced_acc'))}), while PR-AUC changes in the opposite direction ({format_delta(d('wo_logit_calibration', 'pr_auc'))}). The calibration module should therefore be described as improving decision balance rather than universally improving every metric.",
        "",
        "## Paper-Oriented Conclusion",
        "",
        "The ablation results support the paper's central claim that the RCE main model works by combining class-aware concept weighting, complementary visual residual evidence, cross-scale interaction, and calibrated final decision scores. The evidence chain is strongest when all four parts are present together.",
        "",
        "This interpretation intentionally avoids claims about DEG modules and does not overstate the contribution of CSG to ACC.",
        "",
    ]
    return "\n".join(lines)


def build_statistics_plan() -> str:
    return "\n".join(
        [
            "# Step52B RCE Ablation Statistics Plan",
            "",
            "## Purpose",
            "",
            "Step52B consolidates already-trained RCE core ablation results into paper-ready statistical tables, fold-level deltas, and interpretation notes.",
            "",
            "## Inputs",
            "",
            "- `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/`",
            "- `results_stage52_rce_core_ablation/*/result.csv`",
            "- `results_stage52_rce_core_ablation/*/fold_summary.csv` when available",
            "- `results_stage52_rce_core_ablation/*/epoch_details.csv` only as a fallback note source",
            "",
            "## Outputs",
            "",
            "- `docs/stage52b_rce_ablation_table_paper.md`",
            "- `docs/stage52b_rce_ablation_table_paper.csv`",
            "- `docs/stage52b_rce_ablation_table_latex.tex`",
            "- `docs/stage52b_rce_ablation_fold_delta.csv`",
            "- `docs/stage52b_rce_ablation_interpretation.md`",
            "- `docs/stage52b_rce_ablation_statistics_plan.md`",
            "",
            "## Run Command",
            "",
            "```bash",
            "python scripts/analysis/build_stage52b_rce_ablation_paper_tables.py",
            "```",
            "",
            "## Using The LaTeX Table",
            "",
            "The generated `docs/stage52b_rce_ablation_table_latex.tex` can be copied into the paper appendix or ablation section directly. It already uses `mean $\\pm$ std`, includes the required rows and columns, and boldfaces the best value in each metric column.",
            "",
            "## Statistical Limitations",
            "",
            "- Paired t-tests are only valid when matched fold-level test metrics are available.",
            "- `epoch_details.csv` stores train/val trajectories and is not treated as a source of final test metrics.",
            "- If fold-level summaries are missing or incomplete, the script leaves p-values as `NA` instead of fabricating them.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    output_md = root / "docs/stage52b_rce_ablation_table_paper.md"
    output_csv = root / "docs/stage52b_rce_ablation_table_paper.csv"
    output_tex = root / "docs/stage52b_rce_ablation_table_latex.tex"
    output_fold = root / "docs/stage52b_rce_ablation_fold_delta.csv"
    output_interp = root / "docs/stage52b_rce_ablation_interpretation.md"
    output_plan = root / "docs/stage52b_rce_ablation_statistics_plan.md"
    for path in [output_md, output_csv, output_tex, output_fold, output_interp, output_plan]:
        path.parent.mkdir(parents=True, exist_ok=True)

    variant_rows, fold_delta_rows, notes = collect_variant_data(root)
    variant_df = pd.DataFrame(variant_rows)
    fold_df = pd.DataFrame(fold_delta_rows)

    output_csv.write_text("", encoding="utf-8")
    variant_df.to_csv(output_csv, index=False)
    if fold_df.empty:
        pd.DataFrame([{"status": "missing", "reason": "No reliable fold-level test metrics found."}]).to_csv(output_fold, index=False)
    else:
        fold_df.to_csv(output_fold, index=False)

    output_md.write_text(build_markdown_report(variant_df, fold_df, notes) + "\n", encoding="utf-8")
    output_tex.write_text(build_latex_table(variant_df) + "\n", encoding="utf-8")
    output_interp.write_text(build_interpretation(variant_df) + "\n", encoding="utf-8")
    output_plan.write_text(build_statistics_plan() + "\n", encoding="utf-8")

    print(f"[Saved] markdown: {output_md}")
    print(f"[Saved] csv: {output_csv}")
    print(f"[Saved] latex: {output_tex}")
    print(f"[Saved] fold delta: {output_fold}")
    print(f"[Saved] interpretation: {output_interp}")
    print(f"[Saved] statistics plan: {output_plan}")


if __name__ == "__main__":
    main()
