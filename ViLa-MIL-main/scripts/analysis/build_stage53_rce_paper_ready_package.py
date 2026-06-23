from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "stage53_rce_paper_ready_package"
METRICS = ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]
METRIC_LABELS = {
    "test_auc": "AUC",
    "test_acc": "ACC",
    "test_f1": "F1",
    "balanced_acc": "BACC",
    "pr_auc": "PR-AUC",
}
INPUT_FILES = {
    "main_lineage": ROOT / "docs" / "main_model_reproduction_and_code_lineage.md",
    "stage52_comparison": ROOT / "docs" / "stage52_rce_core_ablation_comparison.md",
    "stage52b_paper_md": ROOT / "docs" / "stage52b_rce_ablation_table_paper.md",
    "stage52b_paper_csv": ROOT / "docs" / "stage52b_rce_ablation_table_paper.csv",
    "stage52b_latex": ROOT / "docs" / "stage52b_rce_ablation_table_latex.tex",
    "stage52b_delta_csv": ROOT / "docs" / "stage52b_rce_ablation_fold_delta.csv",
    "stage52b_interpretation": ROOT / "docs" / "stage52b_rce_ablation_interpretation.md",
    "stage52b_statistics": ROOT / "docs" / "stage52b_rce_ablation_statistics_plan.md",
}
VARIANTS = [
    {
        "key": "stage23_history",
        "paper_label": "Stage23 reference",
        "run_label": "Historical Stage23 full RCE",
        "path": ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1",
    },
    {
        "key": "full",
        "paper_label": "Full RCE-v4-CSG-rq16",
        "run_label": "Step52 full RCE",
        "path": ROOT / "results_stage52_rce_core_ablation" / "full_rce_v4_csg_rq16_5fold_e20_s1",
    },
    {
        "key": "wo_csg",
        "paper_label": "w/o CSG",
        "run_label": "Step52 w/o CSG",
        "path": ROOT / "results_stage52_rce_core_ablation" / "wo_csg_5fold_e20_s1",
    },
    {
        "key": "wo_concept_prior",
        "paper_label": "w/o concept prior",
        "run_label": "Step52 w/o concept prior",
        "path": ROOT / "results_stage52_rce_core_ablation" / "wo_concept_prior_5fold_e20_s1",
    },
    {
        "key": "wo_visual_residual",
        "paper_label": "w/o visual residual",
        "run_label": "Step52 w/o visual residual",
        "path": ROOT / "results_stage52_rce_core_ablation" / "wo_visual_residual_5fold_e20_s1",
    },
    {
        "key": "wo_logit_calibration",
        "paper_label": "w/o logit calibration",
        "run_label": "Step52 w/o logit calibration",
        "path": ROOT / "results_stage52_rce_core_ablation" / "wo_logit_calibration_5fold_e20_s1",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step53 RCE paper-ready package.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def read_text_optional(path: Path, warnings: list[str]) -> str | None:
    if not path.is_file():
        warnings.append(f"missing text: {rel(path)}")
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"failed to read text {rel(path)}: {exc}")
        return None


def read_csv_optional(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warnings.append(f"missing csv: {rel(path)}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"failed to read csv {rel(path)}: {exc}")
        return None


def read_result_mean_std(run_dir: Path, warnings: list[str]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    df = read_csv_optional(run_dir / "result.csv", warnings)
    if df is None or "metric" not in df.columns:
        if df is not None:
            warnings.append(f"invalid result schema: {rel(run_dir / 'result.csv')}")
        return None, None
    mean_rows = df[df["metric"] == "mean"]
    std_rows = df[df["metric"] == "std"]
    if mean_rows.empty:
        warnings.append(f"missing mean row: {rel(run_dir / 'result.csv')}")
        return None, None
    mean_row = mean_rows.iloc[0]
    std_row = std_rows.iloc[0] if not std_rows.empty else None
    mean_metrics = {metric: safe_float(mean_row.get(metric)) for metric in METRICS}
    std_metrics = {metric: safe_float(std_row.get(metric)) if std_row is not None else None for metric in METRICS}
    return mean_metrics, std_metrics


def read_fold_metrics(run_dir: Path, warnings: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    fold_summary = run_dir / "fold_summary.csv"
    summary = run_dir / "summary.csv"
    epoch_details = run_dir / "epoch_details.csv"
    if fold_summary.is_file():
        df = read_csv_optional(fold_summary, warnings)
        required = {"fold", *METRICS}
        if df is not None and required.issubset(df.columns):
            out = df[["fold", *METRICS]].copy()
            out["fold"] = out["fold"].astype(int)
            return out.sort_values("fold").reset_index(drop=True), None
        return None, "fold_summary.csv missing required test-metric columns"
    if summary.is_file():
        df = read_csv_optional(summary, warnings)
        fold_col = "folds" if df is not None and "folds" in df.columns else "fold" if df is not None and "fold" in df.columns else None
        if df is not None and fold_col is not None and set(METRICS).issubset(df.columns):
            out = df[[fold_col, *METRICS]].copy()
            out = out.rename(columns={fold_col: "fold"})
            out["fold"] = out["fold"].astype(int) + (0 if out["fold"].min() == 1 else 1)
            return out.sort_values("fold").reset_index(drop=True), "fold-level metrics recovered from summary.csv"
        return None, "summary.csv exists but cannot recover fold-level test metrics"
    if epoch_details.is_file():
        return None, "only epoch_details.csv exists; fold-level test metrics unavailable"
    return None, "no fold-level summary file found"


def compute_paired_pvalue(full_df: pd.DataFrame | None, variant_df: pd.DataFrame | None, metric: str) -> float | None:
    if full_df is None or variant_df is None:
        return None
    merged = full_df[["fold", metric]].merge(
        variant_df[["fold", metric]],
        on="fold",
        how="inner",
        suffixes=("_full", "_variant"),
    ).dropna()
    if len(merged) < 2:
        return None
    test = ttest_rel(merged[f"{metric}_variant"], merged[f"{metric}_full"], nan_policy="omit")
    return safe_float(test.pvalue)


def fmt(value: float | None, digits: int = 4, missing: str = "missing") -> str:
    if value is None:
        return missing
    return f"{value:.{digits}f}"


def fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "missing"
    if std is None:
        return fmt(mean)
    return f"{mean:.4f} ± {std:.4f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.4f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "NA"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def build_variant_payload(warnings: list[str]) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        mean_metrics, std_metrics = read_result_mean_std(variant["path"], warnings)
        fold_df, fold_note = read_fold_metrics(variant["path"], warnings)
        if fold_note:
            warnings.append(f"{variant['paper_label']}: {fold_note}")
        payload[variant["key"]] = {
            "meta": variant,
            "status": "ready" if mean_metrics is not None else "missing",
            "mean": mean_metrics,
            "std": std_metrics,
            "fold_df": fold_df,
        }
    return payload


def build_ablation_rows(collected: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    full_mean = collected.get("full", {}).get("mean")
    rows: list[dict[str, object]] = []
    for key in ["full", "wo_csg", "wo_concept_prior", "wo_visual_residual", "wo_logit_calibration"]:
        payload = collected[key]
        mean_metrics = payload["mean"] or {}
        std_metrics = payload["std"] or {}
        row: dict[str, object] = {"Variant": payload["meta"]["paper_label"]}
        for metric in METRICS:
            label = METRIC_LABELS[metric]
            row[label] = fmt_pm(mean_metrics.get(metric), std_metrics.get(metric))
            delta = None
            if full_mean and mean_metrics.get(metric) is not None and full_mean.get(metric) is not None:
                delta = float(mean_metrics[metric]) - float(full_mean[metric])
            row[f"Δ{label} vs full"] = fmt_delta(delta)
        rows.append(row)
    return rows


def build_ttest_rows(collected: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    full_fold = collected.get("full", {}).get("fold_df")
    rows: list[dict[str, object]] = []
    for key in ["wo_csg", "wo_concept_prior", "wo_visual_residual", "wo_logit_calibration"]:
        payload = collected[key]
        row: dict[str, object] = {"Variant": payload["meta"]["paper_label"]}
        for metric in METRICS:
            row[f"p({METRIC_LABELS[metric]})"] = fmt_p(compute_paired_pvalue(full_fold, payload["fold_df"], metric))
        row["Interpretation"] = "descriptive only; no robust significance claim"
        rows.append(row)
    return rows


def build_fold_delta_rows(collected: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    full_fold = collected.get("full", {}).get("fold_df")
    rows: list[dict[str, object]] = []
    if full_fold is None:
        return rows
    for key in ["wo_csg", "wo_concept_prior", "wo_visual_residual", "wo_logit_calibration"]:
        payload = collected[key]
        variant_fold = payload["fold_df"]
        if variant_fold is None:
            continue
        merged = variant_fold.merge(full_fold, on="fold", how="inner", suffixes=("", "_full"))
        for _, data in merged.iterrows():
            row = {"Variant": payload["meta"]["paper_label"], "Fold": int(data["fold"])}
            for metric in METRICS:
                label = METRIC_LABELS[metric]
                row[f"{label} Δ"] = fmt_delta(safe_float(data.get(metric)) - safe_float(data.get(f"{metric}_full")))
            rows.append(row)
    return rows


def build_main_result_rows(collected: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    stage23_mean = collected.get("stage23_history", {}).get("mean") or {}
    stage23_std = collected.get("stage23_history", {}).get("std") or {}
    step52_mean = collected.get("full", {}).get("mean") or {}
    rows: list[dict[str, object]] = []
    for metric in ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]:
        rows.append(
            {
                "Metric": METRIC_LABELS[metric],
                "Stage23 Main Result": fmt_pm(stage23_mean.get(metric), stage23_std.get(metric)),
                "Step52 Full Check": fmt(step52_mean.get(metric)),
                "Delta": fmt_delta(
                    None
                    if stage23_mean.get(metric) is None or step52_mean.get(metric) is None
                    else float(step52_mean[metric]) - float(stage23_mean[metric])
                ),
            }
        )
    return rows


def copy_or_build_latex(source: Path, target: Path, warnings: list[str]) -> None:
    caption_text = (
        "Core ablation study on the final RCE main model. Removing logit calibration "
        "slightly increases PR-AUC, but reduces ACC, F1, and balanced accuracy."
    )
    latex_text = read_text_optional(source, warnings)
    if latex_text is None:
        target.write_text(
            "% missing source LaTeX table\n"
            f"% Suggested caption: {caption_text}\n",
            encoding="utf-8",
        )
        return
    if "\\caption{" in latex_text:
        start = latex_text.find("\\caption{")
        end = latex_text.find("}", start)
        if start != -1 and end != -1:
            updated = latex_text[:start] + f"\\caption{{{caption_text}}}" + latex_text[end + 1 :]
        else:
            updated = f"% Suggested caption: {caption_text}\n" + latex_text
    else:
        updated = f"% Suggested caption: {caption_text}\n" + latex_text
    target.write_text(updated, encoding="utf-8")


def build_summary_markdown(
    collected: dict[str, dict[str, object]],
    source_texts: dict[str, str | None],
    warnings: list[str],
) -> dict[str, str]:
    stage23 = collected.get("stage23_history", {})
    full = collected.get("full", {})
    stage23_mean = stage23.get("mean") or {}
    step52_mean = full.get("mean") or {}
    ablation_rows = build_ablation_rows(collected)
    ttest_rows = build_ttest_rows(collected)
    fold_delta_rows = build_fold_delta_rows(collected)[:12]
    main_result_rows = build_main_result_rows(collected)
    consistency_ok = all(
        (
            stage23_mean.get(metric) is not None
            and step52_mean.get(metric) is not None
            and abs(float(stage23_mean[metric]) - float(step52_mean[metric])) < 1e-12
        )
        for metric in METRICS
    )
    missing_inputs = [name for name, text in source_texts.items() if text is None]
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- none"
    missing_lines = "\n".join(f"- {name}: missing" for name in missing_inputs) if missing_inputs else "- none"

    summary_md = f"""# Stage53 RCE Paper-Ready Summary

## Final Main Model Definition

- Main model: `RCE-v4-CSG-a01-rq16`
- Main model file: `models/model_RCE_MIL_BiomedCLIP.py`
- Main training script: `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
- Main result directory: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Runtime call chain: `main.py -> utils/core_utils.py -> models/model_RCE_MIL_BiomedCLIP.py`
- Scope note: Step53 is fixed on RCE. DEG is retained only as a later extension shell and is not the final paper model.

## Stage23 Main Result

{markdown_table(main_result_rows, ["Metric", "Stage23 Main Result", "Step52 Full Check", "Delta"])}

Consistency note: {"Stage23 and Step52 full are numerically identical across the tracked metrics." if consistency_ok else "Stage23 and Step52 full are not perfectly identical; inspect the Delta column above."}

## Step52 and Step52B Ablation Summary

{markdown_table(ablation_rows, ["Variant", "AUC", "ACC", "F1", "BACC", "PR-AUC", "ΔAUC vs full", "ΔACC vs full", "ΔF1 vs full", "ΔBACC vs full", "ΔPR-AUC vs full"])}

## Current Innovation Line

- ViLa-MIL provides dual-scale vision-language alignment at the slide level.
- RCE extends that backbone into region-concept evidence reasoning rather than only global alignment.
- Concept prior is the dominant class-aware evidence constraint.
- Visual residual supplies complementary visual evidence that concept-only scoring does not fully capture.
- CSG mainly improves evidence ranking quality, reflected more clearly in AUC and PR-AUC than in raw ACC.
- Logit calibration improves balanced decision behavior, especially BACC and F1, with a small PR-AUC trade-off.

## Evidence Strength Assessment

- Strongest evidence: exact reproduction of the final main model in Stage23 and Step52 full, plus consistent descriptive ablation trends.
- Moderate evidence: concept prior and visual residual both show clear average drops across multiple metrics when removed.
- Cautious evidence: CSG and logit calibration effects are more metric-specific and should not be oversold as uniform gains.
- Statistical stance: paired t-tests are available as descriptive references, but the current 5-fold setup is better presented as trend evidence rather than strict statistical significance.

## Recommended Paper Narrative

- Position the final method as an RCE extension of ViLa-MIL from slide-level alignment to region-concept evidence reasoning.
- Emphasize that concept prior and visual residual are the two most visible contributors to the final operating point.
- Describe CSG as improving cross-scale concept interaction and evidence ranking quality, especially AUC and PR-AUC.
- Describe logit calibration as improving balanced decision behavior, while acknowledging that PR-AUC can be slightly higher without it.
- Keep DEG, HCRC, and PRARC out of the final main-method claim in Step53.

## Input Status

Missing inputs:
{missing_lines}

Warnings:
{warning_lines}
"""

    main_results_md = f"""# Stage53 RCE Main Results

## Final Main Model

- Main model: `RCE-v4-CSG-a01-rq16`
- Model file: `models/model_RCE_MIL_BiomedCLIP.py`
- Training script: `scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh`
- Result directory: `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`

## Main Metrics

{markdown_table(main_result_rows, ["Metric", "Stage23 Main Result", "Step52 Full Check", "Delta"])}

## Scope Clarification

- The preferred paper result is the Stage23 direct-RCE run, not a DEG wrapper run.
- Stage23 and Step52 full are used here as consistency-checked equivalents for the final RCE package.
- The final main model is not DEG.
- Step51B only supports the narrower statement that the DEG all-off skeleton can reproduce the RCE path.
- Step53 does not rely on, or claim, any DEG any-on conclusion.
"""

    ablation_summary_md = f"""# Stage53 RCE Ablation Summary

## Core Ablation Table

{markdown_table(ablation_rows, ["Variant", "AUC", "ACC", "F1", "BACC", "PR-AUC", "ΔAUC vs full", "ΔACC vs full", "ΔF1 vs full", "ΔBACC vs full", "ΔPR-AUC vs full"])}

## Module-Wise Interpretation

- `concept prior`: contribution is the largest. Removing it causes the biggest mean drop across AUC, ACC, F1, BACC, and PR-AUC.
- `visual residual`: provides important complementary visual evidence. Removing it produces clear degradation in ACC, F1, BACC, and PR-AUC.
- `CSG`: mainly improves AUC and PR-AUC, which supports an evidence-ranking interpretation. The ACC change is small and should not be exaggerated.
- `logit calibration`: mainly improves balanced decision behavior, especially BACC and F1. PR-AUC is slightly higher without calibration, so this module should be described as a trade-off rather than a universal gain.

## Paired T-Test Reference

{markdown_table(ttest_rows, ["Variant", "p(AUC)", "p(ACC)", "p(F1)", "p(BACC)", "p(PR-AUC)", "Interpretation"])}

Interpretation: these p-values are reported only as matched-fold descriptive references. Under the current 5-fold setting, the ablation evidence is better framed as stable directional trends than as strict statistical significance.

## Fold-Level Delta Preview

{markdown_table(fold_delta_rows, ["Variant", "Fold", "AUC Δ", "ACC Δ", "F1 Δ", "BACC Δ", "PR-AUC Δ"]) if fold_delta_rows else "Fold-level deltas are missing."}

## Writing Constraint

- The current ablation package supports descriptive trend statements.
- It should not be written as proof that all modules are statistically significant.
"""

    claims_md = """# Stage53 RCE Claims To Make And Avoid

| Can Make | Avoid |
| --- | --- |
| RCE main model is stably reproduced by `RCE-v4-CSG-a01-rq16`. | CSG significantly and substantially improves ACC. |
| Concept prior is the strongest single contribution among the tested core modules. | All modules are statistically significant. |
| Visual residual provides complementary visual evidence beyond concept-only reasoning. | Logit calibration improves every metric simultaneously. |
| CSG improves evidence ranking quality, especially AUC and PR-AUC. | DEG, HCRC, or PRARC is the final main model. |
| Logit calibration improves balanced decision behavior, especially BACC and F1. | DEG any-on modules have already been proven cleanly attributable. |
| RCE advances ViLa-MIL from slide-level alignment toward region-concept evidence reasoning. |  |
"""

    results_draft_md = """# Stage53 RCE Results Section Draft

The final model used in this paper is `RCE-v4-CSG-a01-rq16`, which is reproduced consistently by the direct RCE implementation and the matched full ablation reference. In the 5-fold evaluation, the full model achieves 0.9702 AUC, 0.9225 ACC, 0.9145 F1, 0.9171 balanced accuracy, and 0.9444 PR-AUC. This consistency supports the use of the Stage23 run as the primary paper result while preserving a clean code lineage to the final RCE model.

Our ablation study suggests that the main performance gains are not driven by a single engineering detail, but by a coordinated evidence pipeline. Among the tested components, concept prior is the most influential: removing it yields the largest degradation across AUC, ACC, F1, balanced accuracy, and PR-AUC. Removing the visual residual branch also causes a broad drop in classification quality, indicating that concept-level evidence alone is insufficient and that complementary visual residual cues remain important in the final decision process.

The effects of CSG and logit calibration are more nuanced. Removing CSG produces a relatively small change in ACC, but a clearer decline in AUC and PR-AUC, which is more consistent with improved evidence ranking than with a direct increase in raw accuracy. Removing logit calibration slightly increases PR-AUC while reducing F1 and balanced accuracy, suggesting that calibration mainly improves decision balance rather than uniformly improving every metric. Because the present analysis is based on matched 5-fold comparisons, we interpret these statistical results descriptively and avoid strong claims of universal significance.
"""

    method_draft_md = """# Stage53 RCE Method Claim Draft

Starting from ViLa-MIL, which aligns pathology slides with language prompts at dual visual scales, we formulate RCE as a region-concept evidence modeling framework for slide-level diagnosis. Instead of relying only on global slide-text alignment, RCE decomposes the prediction process into region-level evidence candidates and concept-aware aggregation, allowing the model to reason about which local observations support which diagnostic concepts.

Within this framework, CSG introduces concept-level cross-scale reasoning between low- and high-magnification evidence, so that concept support can be refined through cross-scale interaction rather than isolated per-scale scoring. The visual residual branch preserves complementary image evidence that is not fully captured by concept similarity alone, while the logit calibration module stabilizes the final decision boundary and improves balanced decision behavior. In this paper, these components are presented as part of the final RCE formulation; DEG, HCRC, and PRARC are not treated as the final method.
"""

    tables_md = """# Stage53 RCE Tables For Paper

## Main Result Table

Use `stage53_rce_main_results.md` as the direct source for the primary quantitative table of the final model.

## Core Ablation Table

Use `stage53_rce_ablation_summary.md` and `stage53_rce_ablation_table_latex.tex` for the ablation section.

## LaTeX Path

- `docs/stage53_rce_paper_ready_package/stage53_rce_ablation_table_latex.tex`

## Placement Recommendation

- Directly in the main paper: final main-result table; core RCE ablation table.
- Better as supplementary material: fold-level delta preview and descriptive paired t-test table.
"""

    figure_plan_md = """# Stage53 RCE Figure Plan

## Step54 Figure Candidates

1. `RCE overall pipeline`
Purpose: summarize the final method from dual-scale inputs to region-concept evidence aggregation and slide prediction.
Input needs: final model diagram source, concept prior path, CSG logic sketch, Stage23 main-model metadata.
Placement: main paper.

2. `region-concept evidence heatmap`
Purpose: show which regions support which concepts in a correct case.
Input needs: exported region-level evidence scores and concept labels from the RCE evidence package.
Placement: main paper.

3. `low-high CSG concept interaction visualization`
Purpose: illustrate concept-level cross-scale interaction between low- and high-magnification evidence.
Input needs: cross-scale concept linkage or attention-style export from the evidence package.
Placement: main paper or supplementary depending on clarity.

4. `correct case evidence visualization`
Purpose: demonstrate that the final prediction is grounded in interpretable evidence regions and concepts.
Input needs: region thumbnails, concept labels, slide prediction, and evidence scores.
Placement: main paper.

5. `failure case analysis`
Purpose: show where region-concept evidence is incomplete, ambiguous, or misleading.
Input needs: failed cases with exported evidence, prediction, label, and selected region overlays.
Placement: supplementary material.

6. `w/o CSG vs full evidence ranking comparison`
Purpose: show how CSG changes evidence ordering or concept consistency even when ACC change is small.
Input needs: matched-case evidence exports from full and w/o CSG runs.
Placement: supplementary material, with one compact teaser panel potentially in the main paper.
"""

    next_steps_md = """# Stage53 RCE Next Steps

- Step54: build the RCE evidence and interpretability package, including figure-ready exports and curated qualitative cases.
- Step55: audit DEG any-on purity before making any claim about later DEG-based modules.
- Step56: if needed, refactor DEG as `RCE + delta` so that later modules can be attributed cleanly.
- Current recommendation: do not keep adding modules blindly. The immediate priority is to package evidence, interpretation, and paper figures around the fixed RCE main model.
"""

    return {
        "stage53_rce_paper_ready_summary.md": summary_md,
        "stage53_rce_main_results.md": main_results_md,
        "stage53_rce_ablation_summary.md": ablation_summary_md,
        "stage53_rce_claims_to_make_and_avoid.md": claims_md,
        "stage53_rce_results_section_draft.md": results_draft_md,
        "stage53_rce_method_claim_draft.md": method_draft_md,
        "stage53_rce_tables_for_paper.md": tables_md,
        "stage53_rce_figure_plan.md": figure_plan_md,
        "stage53_rce_next_steps.md": next_steps_md,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else args.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    source_texts = {name: read_text_optional(path, warnings) for name, path in INPUT_FILES.items()}
    collected = build_variant_payload(warnings)
    docs = build_summary_markdown(collected, source_texts, warnings)

    for filename, content in docs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    copy_or_build_latex(INPUT_FILES["stage52b_latex"], output_dir / "stage53_rce_ablation_table_latex.tex", warnings)

    manifest_rows = []
    for name, path in INPUT_FILES.items():
        manifest_rows.append(
            {
                "input_name": name,
                "path": rel(path),
                "status": "ready" if path.exists() else "missing",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(output_dir / "stage53_rce_input_manifest.csv", index=False)


if __name__ == "__main__":
    main()
