from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_stage54_rce_evidence_interpretability"
DOCS_DIR = ROOT / "docs" / "stage54_rce_evidence_interpretability_package"
FIGURE_DIR = DOCS_DIR / "figures"

FULL_RUN_DIR = ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1"
WO_CSG_RUN_DIR = ROOT / "results_stage52_rce_core_ablation" / "wo_csg_5fold_e20_s1"
FULL_EXISTING_EVIDENCE_DIR = ROOT / "results_stage32" / "stage32_rce_v4_csg_evidence_export"
FULL_STAGE39_DIR = ROOT / "results_stage39" / "final_evidence_package"
FIGURE_PLAN_PATH = ROOT / "docs" / "stage53_rce_paper_ready_package" / "stage53_rce_figure_plan.md"
WO_CSG_EXPECTED_EVIDENCE_DIR = RESULTS_DIR / "wo_csg"
FULL_EXPECTED_EVIDENCE_DIR = RESULTS_DIR / "full"

LABEL_NAMES = {0: "Adenocarcinoma", 1: "NonAdenocarcinoma"}
METRICS = ["test_auc", "test_acc", "test_f1", "balanced_acc", "pr_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step54 RCE evidence / interpretability package.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--docs_dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--full_run_dir", type=Path, default=FULL_RUN_DIR)
    parser.add_argument("--wo_csg_run_dir", type=Path, default=WO_CSG_RUN_DIR)
    parser.add_argument("--full_existing_evidence_dir", type=Path, default=FULL_EXISTING_EVIDENCE_DIR)
    parser.add_argument("--stage39_dir", type=Path, default=FULL_STAGE39_DIR)
    parser.add_argument("--figure_plan_path", type=Path, default=FIGURE_PLAN_PATH)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def fmt(value: float | None, digits: int = 4, missing: str = "missing") -> str:
    if value is None:
        return missing
    return f"{value:.{digits}f}"


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


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def read_result_metrics(run_dir: Path, warnings: list[str]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    result_path = run_dir / "result.csv"
    df = read_csv_optional(result_path, warnings)
    if df is None or "metric" not in df.columns:
        return None, None
    mean_rows = df[df["metric"] == "mean"]
    std_rows = df[df["metric"] == "std"]
    if mean_rows.empty:
        warnings.append(f"result mean row missing: {rel(result_path)}")
        return None, None
    mean_row = mean_rows.iloc[0]
    std_row = std_rows.iloc[0] if not std_rows.empty else None
    mean_metrics = {metric: safe_float(mean_row.get(metric)) for metric in METRICS}
    std_metrics = {metric: safe_float(std_row.get(metric)) if std_row is not None else None for metric in METRICS}
    return mean_metrics, std_metrics


def read_fold_summary(run_dir: Path, warnings: list[str]) -> pd.DataFrame | None:
    fold_path = run_dir / "fold_summary.csv"
    df = read_csv_optional(fold_path, warnings)
    if df is None:
        return None
    required = {"fold", *METRICS}
    if not required.issubset(df.columns):
        warnings.append(f"fold summary missing required columns: {rel(fold_path)}")
        return None
    out = df[["fold", *METRICS]].copy()
    out["fold"] = out["fold"].astype(int)
    return out


def load_prediction_rows(run_dir: Path, prefix: str, warnings: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        path = run_dir / f"split_{fold}_results.pkl"
        if not path.is_file():
            warnings.append(f"missing prediction pkl: {rel(path)}")
            continue
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            warnings.append(f"failed to load pkl {rel(path)}: {exc}")
            continue
        if not isinstance(payload, dict):
            warnings.append(f"unexpected pkl structure: {rel(path)}")
            continue
        for slide_id, item in payload.items():
            if not isinstance(item, dict):
                continue
            prob = item.get("prob")
            label = item.get("label")
            prob_0 = prob_1 = confidence = None
            pred = None
            if isinstance(prob, np.ndarray):
                arr = np.asarray(prob).reshape(-1)
            elif isinstance(prob, (list, tuple)):
                arr = np.asarray(prob).reshape(-1)
            else:
                arr = None
            if arr is not None and arr.size >= 2:
                prob_0 = safe_float(arr[0])
                prob_1 = safe_float(arr[1])
                pred = int(np.argmax(arr[:2]))
                confidence = max(prob_0 or 0.0, prob_1 or 0.0)
            elif arr is not None and arr.size == 1:
                prob_1 = safe_float(arr[0])
                prob_0 = 1.0 - prob_1 if prob_1 is not None else None
                pred = 1 if (prob_1 or 0.0) >= 0.5 else 0
                confidence = prob_1 if pred == 1 else prob_0
            label_value = None if label is None else int(label)
            rows.append(
                {
                    "slide_id": str(slide_id),
                    "fold": fold,
                    f"{prefix}_true_label": label_value,
                    f"{prefix}_prob_class_0": prob_0,
                    f"{prefix}_prob_class_1": prob_1,
                    f"{prefix}_pred": pred,
                    f"{prefix}_confidence": confidence,
                    f"{prefix}_correct": None if pred is None or label_value is None else pred == label_value,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_case_level_summary(
    full_df: pd.DataFrame,
    wo_csg_df: pd.DataFrame,
    full_fold_summary: pd.DataFrame | None,
    wo_csg_fold_summary: pd.DataFrame | None,
    full_metrics: dict[str, float] | None,
    wo_csg_metrics: dict[str, float] | None,
    full_evidence_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    if full_df.empty:
        return pd.DataFrame()

    merged = full_df.merge(wo_csg_df, on=["slide_id", "fold"], how="left")
    merged["true_label"] = merged["full_true_label"]
    merged["true_label_name"] = merged["true_label"].map(LABEL_NAMES)
    merged["full_pred_name"] = merged["full_pred"].map(LABEL_NAMES)
    merged["wo_csg_pred_name"] = merged["wo_csg_pred"].map(LABEL_NAMES)
    merged["confidence_delta_full_minus_wo_csg"] = merged["full_confidence"] - merged["wo_csg_confidence"]
    merged["same_prediction"] = merged["full_pred"] == merged["wo_csg_pred"]
    merged["csg_benefit_case"] = (merged["full_correct"] == True) & (merged["wo_csg_correct"] == False)
    merged["full_wrong_case"] = merged["full_correct"] == False
    merged["same_pred_confidence_shift"] = merged["same_prediction"] & (
        merged["confidence_delta_full_minus_wo_csg"].abs() >= 0.10
    )

    if full_fold_summary is not None:
        renamed = full_fold_summary.rename(
            columns={metric: f"full_fold_{metric}" for metric in METRICS}
        )
        merged = merged.merge(renamed, on="fold", how="left")
    if wo_csg_fold_summary is not None:
        renamed = wo_csg_fold_summary.rename(
            columns={metric: f"wo_csg_fold_{metric}" for metric in METRICS}
        )
        merged = merged.merge(renamed, on="fold", how="left")

    if full_metrics:
        for metric, value in full_metrics.items():
            merged[f"full_run_{metric}"] = value
    if wo_csg_metrics:
        for metric, value in wo_csg_metrics.items():
            merged[f"wo_csg_run_{metric}"] = value

    if full_evidence_summary is not None and not full_evidence_summary.empty:
        evidence_subset = full_evidence_summary.copy()
        evidence_subset["slide_id"] = evidence_subset["slide_id"].astype(str)
        if "fold" in evidence_subset.columns:
            evidence_subset["fold"] = evidence_subset["fold"].astype(int)
            merge_keys = ["slide_id", "fold"]
        else:
            merge_keys = ["slide_id"]
        keep_cols = [
            col
            for col in [
                *merge_keys,
                "pred_margin",
                "final_logit_class_0",
                "final_logit_class_1",
                "top_csg_pair_class_0",
                "top_csg_pair_class_1",
                "top_low_concepts_for_pred",
                "top_high_concepts_for_pred",
                "top_low_concepts_for_true",
                "top_high_concepts_for_true",
            ]
            if col in evidence_subset.columns
        ]
        evidence_subset = evidence_subset[keep_cols].copy()
        rename_map = {
            col: f"full_evidence_{col}" for col in evidence_subset.columns if col not in merge_keys
        }
        evidence_subset = evidence_subset.rename(columns=rename_map)
        merged = merged.merge(evidence_subset, on=merge_keys, how="left")

    merged["full_evidence_available"] = False
    evidence_cols = [
        "full_evidence_top_low_concepts_for_pred",
        "full_evidence_top_high_concepts_for_pred",
        "full_evidence_pred_margin",
    ]
    present_cols = [col for col in evidence_cols if col in merged.columns]
    if present_cols:
        merged["full_evidence_available"] = merged[present_cols].notna().any(axis=1)

    column_order = [
        "slide_id",
        "fold",
        "true_label",
        "true_label_name",
        "full_pred",
        "full_pred_name",
        "full_confidence",
        "full_correct",
        "wo_csg_pred",
        "wo_csg_pred_name",
        "wo_csg_confidence",
        "wo_csg_correct",
        "confidence_delta_full_minus_wo_csg",
        "same_prediction",
        "csg_benefit_case",
        "same_pred_confidence_shift",
        "full_evidence_available",
    ]
    ordered_cols = [col for col in column_order if col in merged.columns]
    other_cols = [col for col in merged.columns if col not in ordered_cols]
    return merged[ordered_cols + other_cols].sort_values(["fold", "slide_id"]).reset_index(drop=True)


def select_cases(case_df: pd.DataFrame) -> pd.DataFrame:
    if case_df.empty:
        return pd.DataFrame()

    selections: list[pd.DataFrame] = []

    def append_group(group_name: str, subset: pd.DataFrame, sort_cols: list[str], ascending: list[bool], limit: int) -> None:
        if subset.empty:
            return
        chosen = subset.sort_values(sort_cols, ascending=ascending).head(limit).copy()
        chosen["selection_type"] = group_name
        chosen["selection_rank"] = range(1, len(chosen.index) + 1)
        selections.append(chosen)

    append_group(
        "full_correct_high_confidence",
        case_df[case_df["full_correct"] == True],
        ["full_evidence_available", "full_confidence", "fold", "slide_id"],
        [False, False, True, True],
        3,
    )
    append_group(
        "full_wrong_failure",
        case_df[case_df["full_correct"] == False],
        ["full_evidence_available", "full_confidence", "fold", "slide_id"],
        [False, False, True, True],
        3,
    )
    append_group(
        "csg_benefit_full_correct_wo_csg_wrong",
        case_df[case_df["csg_benefit_case"] == True],
        ["full_evidence_available", "confidence_delta_full_minus_wo_csg", "full_confidence", "fold"],
        [False, False, False, True],
        3,
    )
    append_group(
        "same_pred_confidence_shift",
        case_df[(case_df["same_prediction"] == True) & case_df["wo_csg_pred"].notna()],
        ["full_evidence_available", "confidence_delta_full_minus_wo_csg", "full_confidence", "fold"],
        [False, False, False, True],
        3,
    )

    if not selections:
        return pd.DataFrame()
    selected = pd.concat(selections, ignore_index=True)
    keep_cols = [
        "selection_type",
        "selection_rank",
        "slide_id",
        "fold",
        "true_label",
        "true_label_name",
        "full_pred",
        "full_pred_name",
        "full_confidence",
        "full_correct",
        "wo_csg_pred",
        "wo_csg_pred_name",
        "wo_csg_confidence",
        "wo_csg_correct",
        "confidence_delta_full_minus_wo_csg",
        "same_prediction",
        "csg_benefit_case",
        "full_evidence_available",
        "full_evidence_top_csg_pair_class_0",
        "full_evidence_top_csg_pair_class_1",
        "full_evidence_top_low_concepts_for_pred",
        "full_evidence_top_high_concepts_for_pred",
        "full_evidence_top_low_concepts_for_true",
        "full_evidence_top_high_concepts_for_true",
    ]
    keep_cols = [col for col in keep_cols if col in selected.columns]
    return selected[keep_cols].copy()


def build_resource_audit_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    resources = [
        (
            "stage53_figure_plan",
            args.figure_plan_path,
            "paper figure plan from Step53",
        ),
        (
            "stage32_export_script",
            ROOT / "scripts" / "analysis" / "export_stage32_rce_v4_csg_evidence.py",
            "existing evidence export implementation",
        ),
        (
            "stage32_run_script",
            ROOT / "scripts" / "analysis" / "run_stage32_export_evidence.sh",
            "existing shell wrapper for evidence export",
        ),
        (
            "stage32_full_evidence_manifest",
            args.full_existing_evidence_dir / "stage32_manifest.json",
            "existing main-model-equivalent evidence manifest",
        ),
        (
            "stage32_full_slide_summary",
            args.full_existing_evidence_dir / "stage32_slide_evidence_summary.csv",
            "slide-level evidence summary for fold0/test",
        ),
        (
            "stage32_full_top_concepts",
            args.full_existing_evidence_dir / "stage32_top_concepts_long.csv",
            "concept-level evidence details",
        ),
        (
            "stage32_full_top_csg_pairs",
            args.full_existing_evidence_dir / "stage32_top_csg_pairs.csv",
            "cross-scale concept interaction details",
        ),
        (
            "stage39_final_evidence_package",
            args.stage39_dir,
            "previous paper-ready evidence package",
        ),
        (
            "stage39_case_summary",
            args.stage39_dir / "stage39_fixed_regressed_persistent_cases.csv",
            "previous selected cases from another comparison branch",
        ),
        (
            "full_stage23_checkpoint_fold0",
            args.full_run_dir / "s_0_checkpoint.pt",
            "direct full-RCE checkpoint for export",
        ),
        (
            "wo_csg_checkpoint_fold0",
            args.wo_csg_run_dir / "s_0_checkpoint.pt",
            "matched w/o CSG checkpoint for export",
        ),
        (
            "expected_full_stage54_export",
            args.results_dir / "full" / "stage32_slide_evidence_summary.csv",
            "future Step54 full export location",
        ),
        (
            "expected_wo_csg_stage54_export",
            args.results_dir / "wo_csg" / "stage32_slide_evidence_summary.csv",
            "future Step54 w/o CSG export location",
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, path, note in resources:
        rows.append(
            {
                "resource": name,
                "path": rel(path),
                "status": "ready" if path.exists() else "missing",
                "note": note,
            }
        )
    return rows


def write_markdown_outputs(
    args: argparse.Namespace,
    warnings: list[str],
    audit_rows: list[dict[str, object]],
    case_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    full_metrics: dict[str, float] | None,
    wo_csg_metrics: dict[str, float] | None,
    figure_plan_text: str | None,
) -> None:
    audit_table = markdown_table(audit_rows, ["resource", "path", "status", "note"])
    warning_lines = "\n".join(f"- {item}" for item in warnings) if warnings else "- none"

    selection_rows = selection_df.to_dict("records") if not selection_df.empty else []
    selection_table = markdown_table(
        selection_rows,
        [
            col
            for col in [
                "selection_type",
                "selection_rank",
                "slide_id",
                "fold",
                "true_label_name",
                "full_pred_name",
                "full_confidence",
                "wo_csg_pred_name",
                "wo_csg_confidence",
                "confidence_delta_full_minus_wo_csg",
            ]
            if selection_df is not None and not selection_df.empty and col in selection_df.columns
        ],
    )

    matched_prediction_available = not case_df.empty and case_df["wo_csg_pred"].notna().any()
    matched_evidence_available = (args.results_dir / "wo_csg" / "stage32_slide_evidence_summary.csv").is_file()
    full_step54_evidence_available = (args.results_dir / "full" / "stage32_slide_evidence_summary.csv").is_file()
    full_correct_count = int((case_df["full_correct"] == True).sum()) if not case_df.empty else 0
    full_wrong_count = int((case_df["full_correct"] == False).sum()) if not case_df.empty else 0
    csg_benefit_count = int((case_df["csg_benefit_case"] == True).sum()) if not case_df.empty else 0
    same_pred_shift_count = int((case_df["same_pred_confidence_shift"] == True).sum()) if not case_df.empty else 0

    stage54_case_selection_md = f"""# Stage54 Case Selection

## Selection Summary

- Full correct high-confidence cases are selected from Stage23 matched predictions.
- Full wrong failure cases are selected from Stage23 matched predictions.
- CSG-benefit cases are defined as `full correct` and `wo_csg wrong`.
- Same-prediction confidence-shift cases are defined from matched full vs `wo_csg` prediction confidence differences.

## Selected Cases

{selection_table}

## Counts

- Full correct cases available: `{full_correct_count}`
- Full wrong cases available: `{full_wrong_count}`
- CSG-benefit cases available: `{csg_benefit_count}`
- Same-prediction confidence-shift candidates available: `{same_pred_shift_count}`

## Notes

- Matched case selection is available directly from `split_*_results.pkl` for Stage23 full and Step52 `wo_csg`.
- Evidence-level matched comparison still depends on exporting `wo_csg` evidence files under `results_stage54_rce_evidence_interpretability/wo_csg/`.
"""

    stage54_audit_md = f"""# Stage54 Evidence Resource Audit

## Audit Table

{audit_table}

## Reuse Decision

- Step54 reuses `scripts/analysis/export_stage32_rce_v4_csg_evidence.py` as the main evidence export backend.
- Existing `results_stage32/stage32_rce_v4_csg_evidence_export/` is reused as the current full-model-equivalent evidence source for fold0/test.
- Stage39 summaries are reused as reference assets, but they compare skeleton vs low-high consistency rather than full vs `wo_csg`.
- New Step54 export orchestration focuses on direct Stage23 full RCE and Step52 `wo_csg`.

## Current Gaps

- Full model evidence export exists only as the earlier Stage32 equivalent package, not yet in the Step54 directory layout.
- {"`wo_csg` evidence export is now available under `results_stage54_rce_evidence_interpretability/wo_csg/`." if matched_evidence_available else "`wo_csg` evidence export files are not present yet unless the new Step54 export script is run."}
- Region thumbnails or image overlays are not present in the audited resources, so image-based qualitative figures remain pending.

## Warnings

{warning_lines}
"""

    stage54_report_md = f"""# Stage54 RCE Evidence Report

## 1. Purpose

Step54 packages interpretability evidence for the fixed RCE main model so that the paper can support the claims of region-concept evidence reasoning and CSG-driven evidence ranking / cross-scale concept interaction.

## 2. How Interpretability Supports The Paper

- RCE should be explained as moving from slide-level vision-language alignment to region-concept evidence reasoning.
- Evidence exports should show which concepts and regions support the prediction.
- Matched full vs `wo_csg` comparisons should be used to explain CSG mainly through ranking, confidence, and cross-scale interaction rather than through ACC inflation.

## 3. Evidence Resources Found

{audit_table}

## 4. What Can Be Generated Now

- Prediction-level matched case selection across all 5 folds is available now from Stage23 full and Step52 `wo_csg`.
- Fold0/test concept-level evidence plots can be generated now from the existing Stage32 export.
- Aggregate concept-frequency and concept-contribution plots can be generated now from the existing Stage32 long-form concept table.
- {"Full vs `wo_csg` evidence-ranking figures can now be generated from the matched export files." if matched_evidence_available else "Full vs `wo_csg` evidence-ranking figures require running the new Step54 export for `wo_csg`."}

## 5. Missing Evidence

- {"No Step54-format `wo_csg` evidence export has been found yet." if not matched_evidence_available else "Matched `wo_csg` evidence is available for fold0/test, but not yet for folds 1-4."}
- {"No Step54-format direct full export has been generated yet; Step54 currently reuses the earlier Stage32 full-equivalent evidence package." if not full_step54_evidence_available else "A direct Step54 full export is available."}
- No region image crops or overlay assets have been found for direct pathology visual panels.
- No direct fold1-4 evidence exports have been found; the audited full evidence package is fold0/test only.

## 6. Can Full vs w/o CSG Be Compared As Matched Cases?

- Prediction-level matched comparison: {"yes" if matched_prediction_available else "no"}.
- Evidence-level matched comparison: {"yes" if matched_evidence_available else "not yet; export is still required"}.

## 7. Correct / Failure Case Summary

- Full correct cases across all folds: `{full_correct_count}`
- Full wrong cases across all folds: `{full_wrong_count}`
- Cases where full is correct but `wo_csg` is wrong: `{csg_benefit_count}`
- Cases with the same prediction but a notable confidence shift: `{same_pred_shift_count}`

These counts support a targeted qualitative analysis plan instead of an exhaustive slide dump.

## 8. Writing Advice

- Use current evidence to support interpretability as an analysis view, not as a strict localization benchmark.
- Use CSG comparisons to discuss evidence ranking and concept interaction.
- Avoid claiming that the current heatmaps are equivalent to pathologist annotations or causal explanations.
"""

    claims_md = """# Stage54 Interpretability Claims To Make And Avoid

| Can Make | Avoid |
| --- | --- |
| RCE provides a region-concept evidence perspective for slide-level prediction. | Visualization is equivalent to pathologist ground-truth annotation. |
| Concept prior and visual residual show complementary contributions in the ablation package. | CSG substantially improves ACC. |
| CSG is better interpreted as improving evidence ranking and confidence behavior. | Evidence heatmaps are strict causal explanations. |
| Visualization is used as supportive analysis of model behavior. | DEG, HCRC, or PRARC is the final main model. |
| Without exported region-level evidence files, localization claims must remain limited. | Region-level localization is fully validated when evidence files are missing. |
"""

    captions_md = """# Stage54 Paper Figure Caption Drafts

## 1. RCE Pipeline

Overview of the proposed RCE framework built on dual-scale vision-language alignment. The model aggregates low- and high-magnification region evidence into concept-aware slide-level predictions, while concept prior, visual residual evidence, and cross-scale reasoning refine the final decision.

## 2. Region-Concept Evidence Heatmap

Region-concept evidence heatmap for a representative correctly classified slide. Rows denote low- and high-scale concept evidence channels, and columns denote the top concepts supporting the final prediction. The visualization is intended as an interpretability aid rather than a direct localization benchmark.

## 3. Low-High CSG Concept Interaction

Top low-to-high concept interaction pairs under the RCE cross-scale graph for a representative case. The figure illustrates how CSG links concept evidence across magnifications and refines the final ranking of supporting evidence.

## 4. Full vs w/o CSG Evidence Ranking

Matched-case comparison of concept evidence ranking between the full RCE model and its `w/o CSG` counterpart. The comparison highlights that CSG mainly affects evidence ordering and confidence structure, even when the predicted label may remain unchanged.

## 5. Correct / Failure Case Visualization

Representative correct and failure cases from the final RCE model. The panels show prediction outcome, confidence, and the dominant concept evidence at low and high magnification, illustrating both successful evidence alignment and typical error modes.
"""

    plan_md = f"""# Stage54 Plan

## Inputs

- `results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- `results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1`
- `results_stage32/stage32_rce_v4_csg_evidence_export`
- `results_stage39/final_evidence_package`
- `docs/stage53_rce_paper_ready_package/stage53_rce_figure_plan.md`

## Outputs

- `results_stage54_rce_evidence_interpretability/stage54_case_level_summary.csv`
- `docs/stage54_rce_evidence_interpretability_package/stage54_case_selection.csv`
- `docs/stage54_rce_evidence_interpretability_package/stage54_case_selection.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_evidence_resource_audit.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_rce_evidence_report.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_interpretability_claims_to_make_and_avoid.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_paper_figure_caption_drafts.md`
- `docs/stage54_rce_evidence_interpretability_package/stage54_figure_index.md`

## Run Commands

```bash
python scripts/analysis/build_stage54_rce_evidence_interpretability_package.py
bash -n scripts/experiments/run_stage54_export_rce_evidence.sh
python scripts/analysis/plot_stage54_rce_evidence_figures.py
```

## Potentially Time-Consuming Steps

- Evidence export for Stage23 full or Step52 `wo_csg` can be inference-heavy, especially if run for multiple folds.
- Figure generation itself is light, but it depends on whether the export files already exist.

## Recommended Order

1. Build the Step54 package and audit available resources.
2. Run `MODE=dry_run` in the export script to verify commands.
3. Export `wo_csg` evidence first, because that is the main missing comparison resource.
4. Export direct Stage23 full evidence into the Step54 directory only if a direct-RCE copy is required for packaging symmetry.
5. Re-run the figure script after evidence export.

## Relation To Step53

- Step53 fixed the paper main model and the allowed claims.
- Step54 converts that stable result package into an evidence and interpretability package.

## Relation To Step55

- Step54 remains fully within the RCE main-model scope.
- Step55 should audit DEG any-on purity separately and should not be mixed into the Step54 main-model narrative.
"""

    summary_rows = []
    for name, metrics in [("full", full_metrics or {}), ("wo_csg", wo_csg_metrics or {})]:
        row = {"variant": name}
        for metric in METRICS:
            row[metric] = fmt(metrics.get(metric), missing="missing")
        summary_rows.append(row)
    summary_table = markdown_table(summary_rows, ["variant", *METRICS])

    figure_plan_block = figure_plan_text if figure_plan_text is not None else "missing"
    summary_md = f"""# Stage54 Summary Notes

## Main Metrics Snapshot

{summary_table}

## Figure Plan Reference

{figure_plan_block}
"""

    (args.docs_dir / "stage54_case_selection.md").write_text(stage54_case_selection_md, encoding="utf-8")
    (args.docs_dir / "stage54_evidence_resource_audit.md").write_text(stage54_audit_md, encoding="utf-8")
    (args.docs_dir / "stage54_rce_evidence_report.md").write_text(stage54_report_md, encoding="utf-8")
    (args.docs_dir / "stage54_interpretability_claims_to_make_and_avoid.md").write_text(claims_md, encoding="utf-8")
    (args.docs_dir / "stage54_paper_figure_caption_drafts.md").write_text(captions_md, encoding="utf-8")
    (args.docs_dir / "stage54_plan.md").write_text(plan_md, encoding="utf-8")
    (args.docs_dir / "stage54_summary_notes.md").write_text(summary_md, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    full_metrics, full_std = read_result_metrics(args.full_run_dir, warnings)
    wo_csg_metrics, wo_csg_std = read_result_metrics(args.wo_csg_run_dir, warnings)
    full_fold_summary = read_fold_summary(args.full_run_dir, warnings)
    wo_csg_fold_summary = read_fold_summary(args.wo_csg_run_dir, warnings)

    full_pred_df = load_prediction_rows(args.full_run_dir, "full", warnings)
    wo_csg_pred_df = load_prediction_rows(args.wo_csg_run_dir, "wo_csg", warnings)

    full_evidence_summary = read_csv_optional(args.full_existing_evidence_dir / "stage32_slide_evidence_summary.csv", warnings)
    figure_plan_text = read_text_optional(args.figure_plan_path, warnings)

    case_df = build_case_level_summary(
        full_pred_df,
        wo_csg_pred_df,
        full_fold_summary,
        wo_csg_fold_summary,
        full_metrics,
        wo_csg_metrics,
        full_evidence_summary,
    )
    case_df.to_csv(args.results_dir / "stage54_case_level_summary.csv", index=False)

    selection_df = select_cases(case_df)
    selection_df.to_csv(args.docs_dir / "stage54_case_selection.csv", index=False)

    audit_rows = build_resource_audit_rows(args)
    pd.DataFrame(audit_rows).to_csv(args.results_dir / "stage54_resource_audit.csv", index=False)

    manifest_rows = [
        {"item": "full_run_dir", "path": rel(args.full_run_dir), "status": "ready" if args.full_run_dir.exists() else "missing"},
        {"item": "wo_csg_run_dir", "path": rel(args.wo_csg_run_dir), "status": "ready" if args.wo_csg_run_dir.exists() else "missing"},
        {"item": "full_existing_evidence_dir", "path": rel(args.full_existing_evidence_dir), "status": "ready" if args.full_existing_evidence_dir.exists() else "missing"},
        {"item": "stage39_dir", "path": rel(args.stage39_dir), "status": "ready" if args.stage39_dir.exists() else "missing"},
    ]
    pd.DataFrame(manifest_rows).to_csv(args.docs_dir / "stage54_manifest.csv", index=False)

    write_markdown_outputs(
        args,
        warnings,
        audit_rows,
        case_df,
        selection_df,
        full_metrics,
        wo_csg_metrics,
        figure_plan_text,
    )


if __name__ == "__main__":
    main()
