from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_TITLE = (
    "Evidence-Guided Cross-Scale Vision-Language Multiple Instance Learning "
    "for Whole Slide Image Classification"
)


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step50 final paper package.")
    parser.add_argument("--stage39_dir", default=env_default("STAGE39_DIR", "results_stage39/final_evidence_package"))
    parser.add_argument("--stage40_dir", default=env_default("STAGE40_DIR", "results_stage40/paper_ready_assets"))
    parser.add_argument("--stage44_dir", default=env_default("STAGE44_DIR", "results_stage44/stage44_hcrc_light_summary"))
    parser.add_argument("--stage45_dir", default=env_default("STAGE45_DIR", "results_stage45/prarc_reliability_audit"))
    parser.add_argument("--stage47_dir", default=env_default("STAGE47_DIR", "results_stage47/stage47_prarc_gate_summary"))
    parser.add_argument(
        "--stage47_gate_diag_dir",
        default=env_default("STAGE47_GATE_DIAG_DIR", "results_stage47/stage47_prarc_gate_diagnostics"),
    )
    parser.add_argument(
        "--stage48b_dir",
        default=env_default("STAGE48B_DIR", "results_stage48/stage48b_prarc_v2_variant_sweep_summary"),
    )
    parser.add_argument("--stage49_dir", default=env_default("STAGE49_DIR", "results_stage49/final_consolidation"))
    parser.add_argument("--output_dir", default=env_default("OUTPUT_DIR", "results_stage50/final_paper_package"))
    parser.add_argument("--paper_title", default=env_default("PAPER_TITLE", DEFAULT_PAPER_TITLE))
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv_optional(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warnings.append(f"Missing CSV: {relative(path)}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Failed to read CSV {relative(path)}: {exc}")
        return None


def read_json_optional(path: Path, warnings: list[str]) -> dict | None:
    if not path.is_file():
        warnings.append(f"Missing JSON: {relative(path)}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Failed to read JSON {relative(path)}: {exc}")
        return None


def read_text_optional(path: Path, warnings: list[str]) -> str | None:
    if not path.is_file():
        warnings.append(f"Missing text/markdown: {relative(path)}")
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"Failed to read text {relative(path)}: {exc}")
        return None


def safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def fmt_metric(value: object) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.4f}"


def fmt_metric_or_blank(value: object) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.4f}"


def choose_primary_model(stage49_rec: dict | None, stage39_rec: dict | None) -> str:
    if stage49_rec and stage49_rec.get("final_primary_model"):
        return str(stage49_rec["final_primary_model"])
    if stage39_rec and stage39_rec.get("recommended_default_model"):
        return str(stage39_rec["recommended_default_model"])
    return "RCE-v4-CSG-a01-rq16 / DEG skeleton"


def choose_secondary_variant(stage49_rec: dict | None, stage39_rec: dict | None) -> str:
    if stage49_rec and stage49_rec.get("final_secondary_variant"):
        return str(stage49_rec["final_secondary_variant"])
    if stage39_rec and stage39_rec.get("secondary_tradeoff_variant"):
        return str(stage39_rec["secondary_tradeoff_variant"])
    return "RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0"


def find_stage39_row(stage39_perf: pd.DataFrame | None, category: str, variant: str) -> pd.Series | None:
    if stage39_perf is None or stage39_perf.empty:
        return None
    rows = stage39_perf[
        (stage39_perf["category"].astype(str) == category) & (stage39_perf["variant"].astype(str) == variant)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def find_best_stage39_category_row(
    stage39_perf: pd.DataFrame | None,
    category: str,
    include_skeleton: bool = True,
) -> pd.Series | None:
    if stage39_perf is None or stage39_perf.empty:
        return None
    rows = stage39_perf[stage39_perf["category"].astype(str) == category].copy()
    if rows.empty:
        return None
    if not include_skeleton:
        rows = rows[rows["variant"].astype(str) != "skeleton"].copy()
    if rows.empty:
        return None
    rows["test_auc_numeric"] = pd.to_numeric(rows["test_auc"], errors="coerce")
    rows = rows.sort_values("test_auc_numeric", ascending=False)
    return rows.iloc[0]


def find_decision_row(stage49_decision: pd.DataFrame | None, keyword: str) -> pd.Series | None:
    if stage49_decision is None or stage49_decision.empty:
        return None
    mask = stage49_decision["module_or_variant"].astype(str).str.contains(keyword, case=False, na=False)
    rows = stage49_decision[mask].copy()
    if rows.empty:
        return None
    rows["auc_numeric"] = pd.to_numeric(rows["auc"], errors="coerce")
    rows = rows.sort_values("auc_numeric", ascending=False)
    return rows.iloc[0]


def extract_primary_metrics(stage39_perf: pd.DataFrame | None) -> dict[str, object]:
    row = find_stage39_row(stage39_perf, "region_query", "rq16")
    if row is None:
        row = find_best_stage39_category_row(stage39_perf, "region_query")
    if row is None:
        return {}
    return {
        "AUC": row.get("test_auc"),
        "ACC": row.get("test_acc"),
        "F1": row.get("test_f1"),
        "Balanced ACC": row.get("balanced_acc"),
        "PR-AUC": row.get("pr_auc"),
    }


def build_main_results_table(
    stage39_perf: pd.DataFrame | None,
    primary_model: str,
    secondary_variant: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if stage39_perf is None or stage39_perf.empty:
        return pd.DataFrame(columns=["method", "AUC", "ACC", "F1", "Balanced ACC", "PR-AUC", "role", "comment"])

    kept_keys: set[tuple[str, str]] = set()
    for _, row in stage39_perf.iterrows():
        category = str(row.get("category", ""))
        variant = str(row.get("variant", ""))
        status = str(row.get("status", ""))
        if status != "ok":
            continue
        if category == "fold0_test_reexport":
            continue
        key = (category, variant)
        if key in kept_keys:
            continue
        kept_keys.add(key)

        method = str(row.get("method", variant))
        role = "comparison"
        comment = ""
        if category == "csg_init" and variant == "csg_a01":
            method = primary_model
            role = "final_primary_model"
            comment = "Final default model retained after Stage39 and confirmed by Stage49."
        elif category == "low_high_consistency" and variant == "lh_l001_m0":
            method = secondary_variant
            role = "secondary_tradeoff_variant"
            comment = "Better ACC/F1/BalAcc trade-off, but lower AUC/PR-AUC and more visual override."
        elif category == "region_query" and variant == "rq16":
            continue
        elif variant == "skeleton":
            continue
        else:
            role_map = {
                "csg_init": "csg_ablation_reference",
                "region_query": "region_query_ablation",
                "spatial_region_graph": "negative_ablation_reference",
                "concept_prompt_graph": "negative_ablation_reference",
                "visual_gate": "negative_ablation_reference",
                "low_high_consistency": "consistency_variant",
            }
            role = role_map.get(category, "comparison")
            comment = f"Source stage: {row.get('stage', 'unknown')} / category: {category}"

        rows.append(
            {
                "method": method,
                "AUC": fmt_metric_or_blank(row.get("test_auc")),
                "ACC": fmt_metric_or_blank(row.get("test_acc")),
                "F1": fmt_metric_or_blank(row.get("test_f1")),
                "Balanced ACC": fmt_metric_or_blank(row.get("balanced_acc")),
                "PR-AUC": fmt_metric_or_blank(row.get("pr_auc")),
                "role": role,
                "comment": comment,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["role_rank"] = frame["role"].map(
        {
            "final_primary_model": 0,
            "secondary_tradeoff_variant": 1,
            "csg_ablation_reference": 2,
            "region_query_ablation": 3,
            "negative_ablation_reference": 4,
            "consistency_variant": 5,
            "comparison": 6,
        }
    ).fillna(9)
    frame = frame.sort_values(["role_rank", "method"]).drop(columns=["role_rank"])
    return frame


def build_ablation_table(stage39_ablation: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "component",
        "selected_variant",
        "reference_or_context",
        "source_stage",
        "delta_AUC",
        "delta_ACC",
        "delta_F1",
        "delta_Balanced_ACC",
        "delta_PR_AUC",
        "status",
        "paper_takeaway",
    ]
    if stage39_ablation is None or stage39_ablation.empty:
        return pd.DataFrame(columns=columns)

    topic_map = {
        "CSG strength": ("CSG a01", "RCE-v4-CSG-a01-rq16"),
        "Region query count": ("rq16", "RCE-v4-CSG-a01-rq16"),
        "Low-High Consistency": (
            "Low-high consistency secondary variant",
            "RCE-v4-CSG-a01-rq16 + Low-High Consistency, lambda=0.01, margin=0",
        ),
    }
    rows: list[dict[str, object]] = []
    for _, row in stage39_ablation.iterrows():
        topic = str(row.get("topic", ""))
        if topic not in topic_map:
            continue
        component, selected_variant = topic_map[topic]
        rows.append(
            {
                "component": component,
                "selected_variant": selected_variant,
                "reference_or_context": "DEG skeleton" if topic == "Low-High Consistency" else str(row.get("reference_variant", "")),
                "source_stage": str(row.get("source_stage", "")),
                "delta_AUC": fmt_metric_or_blank(row.get("delta_test_auc")),
                "delta_ACC": fmt_metric_or_blank(row.get("delta_test_acc")),
                "delta_F1": fmt_metric_or_blank(row.get("delta_test_f1")),
                "delta_Balanced_ACC": fmt_metric_or_blank(row.get("delta_balanced_acc")),
                "delta_PR_AUC": fmt_metric_or_blank(row.get("delta_pr_auc")),
                "status": "validated_positive" if topic != "Low-High Consistency" else "secondary_tradeoff_variant",
                "paper_takeaway": str(row.get("paper_ready_conclusion", "")),
            }
        )

    rows.extend(
        [
            {
                "component": "Concept prompt pool",
                "selected_variant": "Final concept prompt pool",
                "reference_or_context": "Core final method component",
                "source_stage": "Stage39 narrative",
                "delta_AUC": "",
                "delta_ACC": "",
                "delta_F1": "",
                "delta_Balanced_ACC": "",
                "delta_PR_AUC": "",
                "status": "included_without_standalone_stage39_delta",
                "paper_takeaway": "Kept as a core component of region-concept evidence modeling; no standalone Stage39 delta table available.",
            },
            {
                "component": "Logit calibration",
                "selected_variant": "Final calibrated logits",
                "reference_or_context": "Core final method component",
                "source_stage": "Stage39 narrative",
                "delta_AUC": "",
                "delta_ACC": "",
                "delta_F1": "",
                "delta_Balanced_ACC": "",
                "delta_PR_AUC": "",
                "status": "included_without_standalone_stage39_delta",
                "paper_takeaway": "Retained in the final pipeline, but Step39 inputs do not expose a standalone ablation delta for this component.",
            },
            {
                "component": "Visual residual",
                "selected_variant": "Residual visual branch",
                "reference_or_context": "Core final method component",
                "source_stage": "Stage39 narrative",
                "delta_AUC": "",
                "delta_ACC": "",
                "delta_F1": "",
                "delta_Balanced_ACC": "",
                "delta_PR_AUC": "",
                "status": "included_without_standalone_stage39_delta",
                "paper_takeaway": "Essential for the final evidence decomposition, while also remaining the main unresolved failure source.",
            },
        ]
    )

    return pd.DataFrame(rows, columns=columns)


def build_negative_ablation_table(
    stage39_perf: pd.DataFrame | None,
    stage49_registry: pd.DataFrame | None,
    stage49_decision: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    registry_map = {}
    if stage49_registry is not None and not stage49_registry.empty:
        registry_map = {str(row["module"]): row for _, row in stage49_registry.iterrows()}

    def registry_text(module: str, field: str, fallback: str = "") -> str:
        row = registry_map.get(module)
        if row is None:
            return fallback
        return str(row.get(field, fallback))

    region_row = find_best_stage39_category_row(stage39_perf, "spatial_region_graph", include_skeleton=False)
    concept_row = find_best_stage39_category_row(stage39_perf, "concept_prompt_graph", include_skeleton=False)
    scalar_row = find_best_stage39_category_row(stage39_perf, "visual_gate", include_skeleton=False)
    hcrc_row = find_decision_row(stage49_decision, "HCRC-Light")
    prarc_v1_row = find_decision_row(stage49_decision, "PRARC-v1")
    prarc_v2_row = find_decision_row(stage49_decision, "PRARC-v2")

    def best_result_text(row: pd.Series | None) -> str:
        if row is None:
            return "N/A"
        return (
            f"{row.get('module_or_variant', row.get('method', row.get('variant', 'N/A')))}; "
            f"AUC={fmt_metric(row.get('auc', row.get('test_auc')))}, "
            f"ACC={fmt_metric(row.get('acc', row.get('test_acc')))}, "
            f"F1={fmt_metric(row.get('f1', row.get('test_f1')))}, "
            f"BalAcc={fmt_metric(row.get('balanced_acc'))}, "
            f"PR-AUC={fmt_metric(row.get('pr_auc'))}"
        )

    rows.append(
        {
            "module": "ordinary region graph",
            "tested_stage": "Stage28 / Stage39",
            "best_result_if_available": best_result_text(region_row),
            "why_not_promoted": registry_text(
                "ordinary region graph",
                "reason_failed",
                "Semantic region tokens did not provide stable enough spatial inductive bias.",
            ),
            "paper_wording": registry_text(
                "ordinary region graph",
                "suggested_wording",
                "attention-centroid region graph: semantic region token does not equal true spatial region",
            ),
        }
    )
    rows.append(
        {
            "module": "ordinary concept graph",
            "tested_stage": "Stage31 / Stage39",
            "best_result_if_available": best_result_text(concept_row),
            "why_not_promoted": registry_text(
                "ordinary concept graph",
                "reason_failed",
                "Plain feature-level prompt smoothing weakened evidence discrimination.",
            ),
            "paper_wording": registry_text(
                "ordinary concept graph",
                "suggested_wording",
                "concept prompt graph: ordinary feature-level prompt smoothing weakens evidence discrimination",
            ),
        }
    )
    rows.append(
        {
            "module": "scalar visual evidence gate",
            "tested_stage": "Stage35 / Stage39",
            "best_result_if_available": best_result_text(scalar_row),
            "why_not_promoted": registry_text(
                "scalar visual evidence gate",
                "reason_failed",
                "Visual residual cannot be safely suppressed by one global scalar.",
            ),
            "paper_wording": registry_text(
                "scalar visual evidence gate",
                "suggested_wording",
                "scalar visual gate: visual residual cannot be simply suppressed by a global scalar",
            ),
        }
    )
    rows.append(
        {
            "module": "HCRC-Light",
            "tested_stage": "Stage44 / Stage45",
            "best_result_if_available": best_result_text(hcrc_row),
            "why_not_promoted": registry_text(
                "HCRC-Light",
                "reason_failed",
                "HCRC-Light stayed below baseline on AUC/PR-AUC even when some classification metrics improved slightly.",
            ),
            "paper_wording": registry_text(
                "HCRC-Light",
                "suggested_wording",
                "HCRC-Light was systematically evaluated but did not surpass the primary cross-scale evidence baseline.",
            ),
        }
    )
    rows.append(
        {
            "module": "PRARC-v1",
            "tested_stage": "Stage47",
            "best_result_if_available": best_result_text(prarc_v1_row),
            "why_not_promoted": registry_text(
                "PRARC-v1",
                "reason_failed",
                "5-fold performance fell below baseline and the gate frequently behaved like a weak scalar control.",
            ),
            "paper_wording": registry_text(
                "PRARC-v1",
                "suggested_wording",
                "PRARC-v1 was a disciplined adaptive-gating attempt, but its gate dynamics and ranking metrics were insufficient.",
            ),
        }
    )
    rows.append(
        {
            "module": "PRARC-v2",
            "tested_stage": "Stage48 / Stage48b",
            "best_result_if_available": best_result_text(prarc_v2_row),
            "why_not_promoted": registry_text(
                "PRARC-v2",
                "reason_failed",
                "PRARC-v2 smoke variants were engineering-stable but still failed the gate-dynamics threshold needed for promotion.",
            ),
            "paper_wording": registry_text(
                "PRARC-v2",
                "suggested_wording",
                "PRARC-v2 improved diagnostics only marginally and should be reported as a negative ablation rather than a final module.",
            ),
        }
    )
    return pd.DataFrame(rows)


def build_failure_analysis_table(
    stage33_counts: pd.DataFrame | None,
    stage39_failure: pd.DataFrame | None,
    stage45_override: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    count_map: dict[str, pd.Series] = {}
    if stage33_counts is not None and not stage33_counts.empty:
        count_map = {str(row["failure_type"]): row for _, row in stage33_counts.iterrows()}

    stage39_map: dict[str, pd.Series] = {}
    if stage39_failure is not None and not stage39_failure.empty:
        stage39_map = {str(row["metric"]): row for _, row in stage39_failure.iterrows()}

    override_visual_row = None
    if stage45_override is not None and not stage45_override.empty:
        override_rows = stage45_override[
            (stage45_override["group_family"].astype(str) == "override")
            & (stage45_override["group_value"].astype(str) == "visual_override")
        ]
        if not override_rows.empty:
            override_visual_row = override_rows.iloc[0]

    def count_text_from_stage33(name: str, mode: str = "count_any_label") -> str:
        row = count_map.get(name)
        if row is None:
            return ""
        value = row.get(mode)
        parsed = safe_float(value)
        if parsed is None:
            return ""
        return str(int(parsed))

    rows.append(
        {
            "failure_type": "visual_residual_override",
            "count": count_text_from_stage33("visual_residual_override", "count_any_label"),
            "interpretation": "Most errors are dominated by the visual residual branch rather than missing concept evidence.",
            "implication": "Residual calibration remains the main unresolved bottleneck for the final model.",
            "related_module": "visual residual / future residual calibration",
        }
    )
    rows.append(
        {
            "failure_type": "low_high_conflict",
            "count": count_text_from_stage33("low_high_conflict", "count_any_label"),
            "interpretation": "Low- and high-scale concept evidence can disagree on a non-trivial subset of error slides.",
            "implication": "Explains why low-high consistency helps calibration but does not fully solve final prediction failures.",
            "related_module": "low/high concept evidence and consistency regularization",
        }
    )
    both_support_wrong_row = stage39_map.get("both_support_wrong")
    rows.append(
        {
            "failure_type": "both_support_wrong",
            "count": "" if both_support_wrong_row is None else str(int(float(both_support_wrong_row["skeleton_value"]))),
            "interpretation": "Some errors are jointly supported by low and high concept evidence, so the issue is not only cross-scale disagreement.",
            "implication": "Future work should improve concept margin quality instead of only enforcing agreement.",
            "related_module": "concept prompt pool / evidence-margin learning",
        }
    )
    prompt_confusion_count = count_text_from_stage33("prompt_confusion", "count_any_label")
    rows.append(
        {
            "failure_type": "prompt_confusion",
            "count": prompt_confusion_count,
            "interpretation": "A smaller subset of errors reflects ambiguous prompt-level evidence rather than pure residual override.",
            "implication": "Supports future prompt-reliability or uncertainty-aware supervision, but not as a claim of current resolution.",
            "related_module": "concept prompt pool / train-split-only reliability learning",
        }
    )

    if override_visual_row is not None:
        rows.append(
            {
                "failure_type": "visual_override_group_profile",
                "count": str(int(float(override_visual_row["n_slides"]))),
                "interpretation": (
                    "Visual override slides show lower prediction confidence and negative visual effective margins "
                    "despite concept evidence being present."
                ),
                "implication": "Supports loss-level or uncertainty-aware residual calibration instead of another scalar gate.",
                "related_module": "PRARC motivation / residual calibration future work",
            }
        )

    return pd.DataFrame(rows)


def build_stage50_paper_outline(paper_title: str, primary_model: str, secondary_variant: str) -> str:
    lines = [
        "# Stage50 Paper Outline",
        "",
        "## Title Suggestion",
        f"- `{paper_title}`",
        "- Optional shorter title: `Evidence-Guided Cross-Scale Vision-Language MIL for Whole Slide Classification`",
        "",
        "## Abstract Structure",
        "- Problem: WSI classification needs not only strong slide-level accuracy but also interpretable evidence paths.",
        "- Method: summarize region-concept evidence, low/high concept evidence, CSG reasoning, visual residual, and calibrated logits.",
        f"- Main result: retain `{primary_model}` as the final default model.",
        f"- Boundary: report `{secondary_variant}` as a secondary calibration trade-off variant only.",
        "- Failure analysis: highlight visual residual override as the main unresolved error type.",
        "",
        "## Introduction Structure",
        "- Clinical and technical motivation for evidence-aware WSI classification.",
        "- Gap: stronger graphs or gates do not automatically yield more reliable evidence.",
        "- Core idea: make region-concept evidence and cross-scale concept reasoning the main modeling axis.",
        "- Contributions: final evidence-driven model selection, negative ablation package, and failure-analysis pipeline.",
        "",
        "## Related Work Structure",
        "- Vision-language MIL for pathology.",
        "- Cross-scale reasoning for WSI classification.",
        "- Evidence decomposition / interpretability in medical AI.",
        "- Reliability calibration and residual correction.",
        "",
        "## Method Structure",
        "- BiomedCLIP low/high patch feature extraction.",
        "- Learnable region queries and region-concept evidence.",
        "- Concept prompt pool and low/high concept logits.",
        "- CSG cross-scale concept reasoning.",
        "- Visual residual branch, calibrated logits, and evidence decomposition.",
        "",
        "## Experiments Structure",
        "- Dataset, strict CV protocol, and metrics.",
        "- Main comparison table centered on the final primary model.",
        "- Main ablation on CSG strength and region query count.",
        "- Calibration trade-off comparison with low-high consistency.",
        "",
        "## Ablation Study Structure",
        "- Positive ablation: CSG a01 and rq16.",
        "- Final-method component table: concept prompt pool, logit calibration, visual residual.",
        "- Secondary trade-off variant: low-high consistency.",
        "",
        "## Failure Analysis Structure",
        "- Failure type counts from Stage33.",
        "- Skeleton vs consistency comparison from Stage39.",
        "- Stage45 override-profile interpretation.",
        "",
        "## Limitations Structure",
        "- HCRC relies on loose proposal/bbox settings and can pull weak high-scale evidence.",
        "- PRARC gates still collapse toward near-scalar behavior.",
        "- Visual residual override remains unsolved.",
        "",
        "## Conclusion Structure",
        "- Re-state the final primary model and what is actually validated.",
        "- Emphasize evidence-driven model choice rather than module stacking.",
        "- Close with conservative future-work directions.",
        "",
    ]
    return "\n".join(lines)


def build_method_overview_final(primary_model: str, secondary_variant: str) -> str:
    lines = [
        "# Stage50 Final Method Overview",
        "",
        f"最终主模型保持 `{primary_model}`。",
        "",
        "## Main Pipeline",
        "- 输入由低倍与高倍 WSI patches 构成，并分别提取 `BiomedCLIP` patch features。",
        "- learnable region queries 对 patch features 做区域级聚合，形成可用于诊断解释的 region evidence tokens。",
        "- region evidence tokens 与 concept prompt pool 进行 region-concept similarity 建模，得到 low-scale 与 high-scale concept evidence logits。",
        "- 低倍与高倍 concept evidence 进一步进入 `CSG` cross-scale concept reasoning，以保留 concept-level cross-scale support path。",
        "- 原始视觉分支通过 visual residual 提供补充判别信息，但最终仍以 calibrated logits 汇总各来源证据。",
        "- 预测输出同时支持 evidence decomposition，可拆成 low evidence、high evidence、CSG evidence、visual residual evidence，用于 failure diagnosis。",
        "",
        "## Final Narrative Boundary",
        "- 主模型叙事只包含：BiomedCLIP low/high patch features、learnable region queries、region-concept evidence、concept prompt pool、low/high concept evidence logits、CSG、visual residual、logit calibration、evidence decomposition。",
        f"- `{secondary_variant}` 仅作为 secondary trade-off variant 出现在 calibration / ablation 章节。",
        "- `HCRC-Light`、`PRARC-v1`、`PRARC-v2` 只能出现在 negative ablation、limitations 或 future work 中，不能写成最终主模型组件。",
        "",
        "## Why This Is The Final Main Path",
        "- Stage39 明确支持 `CSG a01 > CSG a005` 与 `rq16 > rq8/rq32`。",
        "- Stage44、Stage47、Stage48b 没有给出比主干更强的 HCRC/PRARC 替代路径。",
        "- 因此最终论文主线应强调 evidence-guided consolidation，而不是继续扩展 graph/gate 模块。",
        "",
    ]
    return "\n".join(lines)


def build_main_method_figure_mermaid() -> str:
    lines = [
        "# Stage50 Main Method Figure Mermaid",
        "",
        "```mermaid",
        "flowchart LR",
        '    A[WSI low patches] --> C[BiomedCLIP feature extraction]',
        '    B[WSI high patches] --> C',
        '    C --> D[Region query aggregation]',
        '    P[Concept prompt pool] --> E[Region-concept similarity]',
        '    D --> E',
        '    E --> F[Low concept evidence logits]',
        '    E --> G[High concept evidence logits]',
        '    F --> H[CSG cross-scale concept reasoning]',
        '    G --> H',
        '    C --> I[Visual residual branch]',
        '    H --> J[Final calibrated logits]',
        '    I --> J',
        '    J --> K[Evidence export]',
        '    K --> L[Failure diagnosis / case review]',
        "```",
        "",
        "说明：HCRC/PRARC 不进入这张最终主方法图，只能在消融或 future work 图表中出现。",
        "",
    ]
    return "\n".join(lines)


def build_evidence_pipeline_figure_mermaid() -> str:
    lines = [
        "# Stage50 Evidence Pipeline Figure Mermaid",
        "",
        "```mermaid",
        "flowchart TD",
        '    A[Final prediction] --> B[Source decomposition]',
        '    B --> C[Low evidence]',
        '    B --> D[High evidence]',
        '    B --> E[CSG evidence]',
        '    B --> F[Visual residual evidence]',
        '    C --> G[Failure type labeling]',
        '    D --> G',
        '    E --> G',
        '    F --> H[Visual residual override diagnosis]',
        '    G --> I[Failure analysis table]',
        '    H --> I',
        "```",
        "",
        "说明：该图用于论文中的 evidence/failure narrative，而不是训练图。",
        "",
    ]
    return "\n".join(lines)


def build_experiment_tables_final(primary_metrics: dict[str, object]) -> str:
    lines = [
        "# Stage50 Experiment Tables Plan",
        "",
        "## Main Comparison Table",
        "- Include the final primary model, the low-high consistency trade-off variant, and representative baseline/ablation references from Stage39.",
        "- Core metrics: `AUC`, `ACC`, `F1`, `Balanced ACC`, `PR-AUC`.",
        f"- Final primary metrics snapshot: `AUC={fmt_metric(primary_metrics.get('AUC'))}`, `ACC={fmt_metric(primary_metrics.get('ACC'))}`, `F1={fmt_metric(primary_metrics.get('F1'))}`, `Balanced ACC={fmt_metric(primary_metrics.get('Balanced ACC'))}`, `PR-AUC={fmt_metric(primary_metrics.get('PR-AUC'))}`.",
        "",
        "## Main Ablation Table",
        "- Focus on positive and retained components: `CSG a01`, `rq16`, concept prompt pool, logit calibration, visual residual, and the low-high consistency trade-off variant.",
        "- Do not invent unavailable standalone deltas; leave them blank and explain that the component is retained as part of the final pipeline.",
        "",
        "## Negative Ablation Table",
        "- Cover ordinary region graph, ordinary concept graph, scalar visual evidence gate, HCRC-Light, PRARC-v1, and PRARC-v2.",
        "- Emphasize why they were not promoted rather than treating them as incomplete engineering attempts.",
        "",
        "## HCRC/PRARC Exploratory Ablation Table",
        "- Optional appendix table summarizing HCRC 5-fold metrics, PRARC-v1 5-fold metrics, and PRARC-v2 gate-diagnostic smoke metrics.",
        "- Keep wording conservative: exploratory branches were completed and rejected by evidence.",
        "",
        "## Failure Analysis Table",
        "- Report failure-type counts from Stage33 plus the Stage39 consistency trade-off comparison and Stage45 override-profile interpretation.",
        "- Make visual residual override the central unresolved failure axis.",
        "",
        "## Optional Sensitivity Table",
        "- If page budget allows, add sensitivity/specificity trade-offs for HCRC-Light and low-high consistency to show why they were not promoted despite selective gains.",
        "",
    ]
    return "\n".join(lines)


def build_claims_to_make_and_avoid() -> str:
    lines = [
        "# Stage50 Claims To Make And Avoid",
        "",
        "## Safe Claims",
        "- Region-concept evidence modeling is effective and remains the strongest validated modeling direction in the current project.",
        "- `CSG` is the strongest validated cross-scale reasoning path among the tested alternatives.",
        "- The final model choice is evidence-driven rather than a late cherry-picked architecture decision.",
        "- Visual residual override remains a major unresolved bottleneck.",
        "- HCRC and PRARC were systematically explored, validated, and not promoted.",
        "",
        "## Claims To Avoid",
        "- Do not claim that HCRC improves the final model.",
        "- Do not claim that PRARC solves visual residual override.",
        "- Do not claim that the PRARC gate is strongly sample-adaptive.",
        "- Do not claim that low-high spatial correspondence is inherently superior to concept-level CSG reasoning.",
        "- Do not redraw HCRC/PRARC as hidden components of the final primary model.",
        "",
    ]
    return "\n".join(lines)


def build_limitations_future_work_final() -> str:
    lines = [
        "# Stage50 Limitations And Future Work",
        "",
        "## Current Limitations",
        "- HCRC currently needs large proposal radius and bbox expansion settings, which can pull in weak or spatially loose high-scale evidence.",
        "- PRARC gates still tend to collapse toward scalar-like behavior instead of maintaining clear sample-adaptive spread.",
        "- Visual residual override remains a major bottleneck and is not solved by the current HCRC/PRARC branches.",
        "",
        "## Recommended Future Work",
        "- Shift toward loss-level or uncertainty-aware residual calibration rather than adding another direct residual gate.",
        "- Use train-split-only reliability learning if prompt or residual reliability signals are introduced into learning.",
        "- Add an evidence-margin auxiliary loss to better separate concept-supported correct slides from residual-overridden failures.",
        "- Strengthen concept evidence construction before adding new graph/gate complexity.",
        "",
    ]
    return "\n".join(lines)


def build_rebuttal_or_defense_points(primary_model: str) -> str:
    qa = [
        (
            "为什么不把 HCRC 作为主模型？",
            "因为 Step44 的三组 HCRC-Light 5-fold 结果都没有超过基线。最接近的 `hcrc_a01_b8` 仍低于基线的 AUC 和 PR-AUC，所以它可以作为系统性探索结果，但不能被写成最终主模型。",
        ),
        (
            "为什么不把 PRARC 作为主模型？",
            "因为 Step47 的 PRARC-v1 5-fold 指标整体低于基线，而 Step48b 的 PRARC-v2 虽然工程稳定，但 gate_std 与 gate_range 仍不足以支撑“强样本自适应门控”的结论。",
        ),
        (
            "为什么 negative ablation 有价值？",
            "因为它说明最终主模型不是凭直觉保留下来的，而是在多个替代 graph/gate 方向被正式验证后，仍由 evidence path 最清晰、指标最稳的方案胜出。",
        ),
        (
            "为什么最终选择 skeleton？",
            f"因为 `{primary_model}` 在 Stage39 已经是最稳默认模型，且后续 Stage44/47/48b 没有出现更强替代者。它同时保留了 region-concept evidence、CSG 和 evidence decomposition 的核心叙事。",
        ),
        (
            "CSG 与 HCRC 的区别是什么？",
            "CSG 是 concept-level 的跨尺度关系建模，直接作用于 low/high concept evidence；HCRC 则更接近 spatial correspondence / child routing 路线。当前证据支持前者更稳定，后者仍受弱 high evidence 引入风险影响。",
        ),
        (
            "visual residual override 是否已经解决？",
            "没有。Stage33/39/45 都表明它仍是主要错误来源，low-high consistency 只能部分缓解 conflict，PRARC 也未能稳定解决 residual override。",
        ),
        (
            "当前方法的主要局限是什么？",
            "主要局限是 residual override 仍强，HCRC 依赖较松的 spatial coverage，PRARC 门控又容易退化为近似 scalar，因此当前工作更像完成了可靠的主模型定稿与负消融收束。",
        ),
        (
            "后续如何继续提升？",
            "优先做 loss-level residual calibration、uncertainty-aware suppression、train-split-only reliability learning，以及 evidence-margin auxiliary loss，而不是继续堆叠 graph/gate 模块。",
        ),
    ]

    lines = ["# Stage50 Rebuttal Or Defense Points", ""]
    for question, answer in qa:
        lines.append(f"## Q: {question}")
        lines.append(answer)
        lines.append("")
    return "\n".join(lines)


def build_paper_section_draft(primary_model: str, secondary_variant: str) -> str:
    lines = [
        "# Stage50 Paper Section Draft",
        "",
        "## Abstract Draft",
        "This work studies evidence-aware whole slide image classification under a vision-language multiple instance learning framework. Instead of continuing to stack new graph or gating modules, we consolidate the strongest validated path around region-concept evidence modeling and cross-scale concept reasoning. The final method uses BiomedCLIP low/high patch features, learnable region queries, a concept prompt pool, low/high concept evidence logits, cross-scale concept graph reasoning, a visual residual branch, calibrated logits, and evidence decomposition for failure diagnosis. Across the Stage39-49 evidence package, `RCE-v4-CSG-a01-rq16 / DEG skeleton` remains the most robust default model, while low-high consistency regularization is retained only as a secondary calibration trade-off variant. Systematic exploratory branches including ordinary region graph, ordinary concept graph, scalar visual gate, HCRC-Light, PRARC-v1, and PRARC-v2 are reported as negative ablations rather than promoted improvements. The consolidated results indicate that visual residual override remains the main unresolved failure source.",
        "",
        "## Introduction Contribution Bullets",
        "- We formulate WSI classification as an evidence-guided vision-language MIL problem in which slide prediction is tied to explicit region-concept evidence rather than slide-level similarity alone.",
        "- We retain a concept-level cross-scale reasoning path through CSG and show that it is stronger than the tested alternative graph/gating routes in the current project branch.",
        "- We provide an evidence-driven consolidation package spanning final metrics, ablations, negative ablations, and failure analysis rather than presenting a single isolated model result.",
        "",
        "## Method Overview Draft",
        "The final model takes low- and high-magnification WSI patches, extracts BiomedCLIP features, and aggregates them with learnable region queries. Each region token is matched against a concept prompt pool to produce region-concept evidence, which is then summarized into low-scale and high-scale concept evidence logits. A cross-scale concept graph reasoning block captures concept-level interactions between low and high evidence streams, while a visual residual branch preserves complementary visual discrimination. The final decision is produced from calibrated logits, and the prediction can be decomposed into low, high, CSG, and visual residual evidence sources for downstream diagnosis. HCRC and PRARC are not part of this final forward narrative and are discussed only as exploratory negative ablations.",
        "",
        "## Ablation Study Draft",
        "The main ablation results support two positive architectural choices. First, Stage24 shows that `CSG a01` outperforms `CSG a005`, supporting the final strength setting for cross-scale concept reasoning. Second, the same stage shows that `rq16` outperforms both `rq8` and `rq32`, fixing the region-query design used by the final model. By contrast, the strongest tested ordinary spatial region graph, concept prompt graph, and scalar visual gate all remain below the skeleton baseline. These results support a narrower claim: the strongest path in the current branch comes from evidence modeling choices inside the region-concept and CSG pipeline, not from adding more generic graph or gate structure.",
        "",
        "## Negative Ablation Discussion Draft",
        "Negative ablations are an important part of the final paper package because they document which directions were completed, tested, and rejected. HCRC-Light completed formal 5-fold evaluation, but none of its variants exceeded the final baseline on the ranking metrics. PRARC-v1 completed formal 5-fold evaluation but stayed below baseline, and its gate diagnostics remained weak overall. PRARC-v2 variants were engineering-stable in smoke evaluation, yet their gate spread was still too small to justify promotion. Reporting these branches as negative ablations makes the final model choice more credible and prevents over-claiming unvalidated improvements.",
        "",
        "## Failure Analysis Draft",
        "The failure analysis across Stage33, Stage39, and Stage45 indicates that visual residual override remains the dominant unresolved error mode. Stage33 reports that visual residual override accounts for 13 error slides as a labeled failure type, while low-high conflict appears on 10 error slides and concept wrong-class drift is less common. Stage39 further shows that low-high consistency regularization reduces low-high conflict and both-support-wrong cases, but slightly increases visual residual override and lowers AUC/PR-AUC, making it a trade-off rather than a clean replacement for the default model. Stage45 reinforces this conclusion by showing that visual override slides have markedly lower confidence margins and negative visual effective margins.",
        "",
        "## Limitations Draft",
        "This study has three main limitations. First, HCRC currently relies on relatively loose proposal coverage, which can introduce weak high-scale evidence and prevents it from serving as the main model path. Second, PRARC gates still tend to collapse toward a near-scalar regime, so the current reliability-gating designs do not yet provide a strong adaptive residual control mechanism. Third, visual residual override remains unresolved, which means the final model is best described as a strong, evidence-supported baseline rather than a finished reliability solution.",
        "",
        "## Conclusion Draft",
        f"Across the consolidated Stage39-49 evidence package, `{primary_model}` remains the final primary model and `{secondary_variant}` remains a secondary evidence-calibration trade-off variant only. The strongest validated paper narrative is therefore centered on region-concept evidence modeling, concept-level cross-scale reasoning through CSG, and explicit evidence decomposition for diagnosis. The completed HCRC and PRARC branches should be reported as systematic negative ablations, while future improvement effort should shift toward residual calibration and uncertainty-aware reliability learning.",
        "",
        "## Writing Constraint Note",
        "- All claims above are grounded in the Stage39-49 result package.",
        "- The draft deliberately avoids presenting HCRC or PRARC as successful final modules.",
        "",
    ]
    return "\n".join(lines)


def build_final_report(
    args: argparse.Namespace,
    output_files: list[str],
    primary_model: str,
    secondary_variant: str,
    primary_metrics: dict[str, object],
    warnings: list[str],
) -> str:
    lines = [
        "# Stage50 Final Paper Package Report",
        "",
        "## Step50 Purpose",
        "- Build the final paper-writing package after HCRC/PRARC consolidation.",
        "- Do not train models, do not modify forward logic, and do not rewrite existing result artifacts.",
        "",
        "## Input Results",
        f"- `{relative(resolve_path(args.stage39_dir))}`",
        f"- `{relative(resolve_path(args.stage40_dir))}`",
        f"- `{relative(resolve_path(args.stage44_dir))}`",
        f"- `{relative(resolve_path(args.stage45_dir))}`",
        f"- `{relative(resolve_path(args.stage47_dir))}`",
        f"- `{relative(resolve_path(args.stage47_gate_diag_dir))}`",
        f"- `{relative(resolve_path(args.stage48b_dir))}`",
        f"- `{relative(resolve_path(args.stage49_dir))}`",
        "",
        "## Output Files",
    ]
    for file_name in output_files:
        lines.append(f"- `{file_name}`")
    lines.extend(
        [
            "",
            "## Final Primary Model",
            f"- `{primary_model}`",
            f"- `AUC={fmt_metric(primary_metrics.get('AUC'))}`",
            f"- `ACC={fmt_metric(primary_metrics.get('ACC'))}`",
            f"- `F1={fmt_metric(primary_metrics.get('F1'))}`",
            f"- `Balanced ACC={fmt_metric(primary_metrics.get('Balanced ACC'))}`",
            f"- `PR-AUC={fmt_metric(primary_metrics.get('PR-AUC'))}`",
            "",
            "## Final Secondary Variant",
            f"- `{secondary_variant}`",
            "- Role: secondary evidence-calibration trade-off variant only.",
            "",
            "## Final Paper Narrative",
            "- Main storyline: region-concept evidence modeling plus concept-level cross-scale reasoning is the strongest validated path.",
            "- HCRC and PRARC belong in negative ablation / future work, not in the final model diagram.",
            "- Visual residual override should be presented as the main unresolved bottleneck.",
            "",
            "## Why More Training Is Not Recommended Now",
            "- The primary model choice is already consolidated by Stage39, Stage44, Stage47, and Stage48b.",
            "- HCRC/PRARC search has already converged to negative ablation conclusions in this branch.",
            "- The highest-value next step is paper-package curation rather than another architecture sweep.",
            "",
            "## Suggested Next Step",
            "- Manually review the generated paper materials.",
            "- Decide the target journal or conference and adjust table density accordingly.",
            "- Prepare figure polishing and representative visual examples for the manuscript.",
            "",
            "## Warnings",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def load_inputs(args: argparse.Namespace, warnings: list[str]) -> dict[str, object]:
    stage39_dir = resolve_path(args.stage39_dir)
    stage40_dir = resolve_path(args.stage40_dir)
    stage44_dir = resolve_path(args.stage44_dir)
    stage45_dir = resolve_path(args.stage45_dir)
    stage47_dir = resolve_path(args.stage47_dir)
    stage47_gate_diag_dir = resolve_path(args.stage47_gate_diag_dir)
    stage48b_dir = resolve_path(args.stage48b_dir)
    stage49_dir = resolve_path(args.stage49_dir)

    inputs: dict[str, object] = {
        "stage39_dir": stage39_dir,
        "stage40_dir": stage40_dir,
        "stage44_dir": stage44_dir,
        "stage45_dir": stage45_dir,
        "stage47_dir": stage47_dir,
        "stage47_gate_diag_dir": stage47_gate_diag_dir,
        "stage48b_dir": stage48b_dir,
        "stage49_dir": stage49_dir,
        "docs_handoff_path": ROOT / "docs/CODEX_HANDOFF.md",
        "stage39_recommendation": read_json_optional(stage39_dir / "stage39_final_model_recommendation.json", warnings),
        "stage39_performance": read_csv_optional(stage39_dir / "stage39_final_performance_summary.csv", warnings),
        "stage39_ablation": read_csv_optional(stage39_dir / "stage39_ablation_summary.csv", warnings),
        "stage39_negative": read_csv_optional(stage39_dir / "stage39_negative_ablation_summary.csv", warnings),
        "stage39_summary_md": read_text_optional(stage39_dir / "stage39_paper_ready_summary.md", warnings),
        "stage40_method_md": read_text_optional(stage40_dir / "stage40_method_overview.md", warnings),
        "stage40_claims_md": read_text_optional(stage40_dir / "stage40_final_claims_and_limitations.md", warnings),
        "stage40_section_md": read_text_optional(stage40_dir / "stage40_paper_section_draft.md", warnings),
        "stage44_summary": read_csv_optional(stage44_dir / "stage44_hcrc_5fold_summary.csv", warnings),
        "stage45_report": read_text_optional(stage45_dir / "stage45_prarc_reliability_report.md", warnings),
        "stage45_override_patterns": read_csv_optional(stage45_dir / "stage45_visual_override_patterns.csv", warnings),
        "stage45_feature_ranking": read_csv_optional(stage45_dir / "stage45_reliability_feature_ranking.csv", warnings),
        "stage47_summary": read_csv_optional(stage47_dir / "stage47_prarc_5fold_summary.csv", warnings),
        "stage47_report": read_text_optional(stage47_dir / "stage47_prarc_gate_report.md", warnings),
        "stage47_gate_distribution": read_csv_optional(
            stage47_gate_diag_dir / "stage47_prarc_gate_distribution_summary.csv", warnings
        ),
        "stage47_gate_report": read_text_optional(
            stage47_gate_diag_dir / "stage47_prarc_gate_diagnostics_report.md", warnings
        ),
        "stage48b_summary": read_csv_optional(stage48b_dir / "stage48b_prarc_v2_variant_sweep_summary.csv", warnings),
        "stage49_recommendation": read_json_optional(stage49_dir / "stage49_final_model_recommendation.json", warnings),
        "stage49_decision_table": read_csv_optional(stage49_dir / "stage49_final_model_decision_table.csv", warnings),
        "stage49_negative_registry": read_csv_optional(stage49_dir / "stage49_negative_ablation_registry.csv", warnings),
        "stage49_claims_md": read_text_optional(stage49_dir / "stage49_paper_claims_and_evidence.md", warnings),
        "stage49_limitations_md": read_text_optional(stage49_dir / "stage49_limitations_and_future_work.md", warnings),
        "stage49_next_routes_md": read_text_optional(stage49_dir / "stage49_next_research_routes.md", warnings),
        "stage33_failure_counts": read_csv_optional(
            ROOT / "results_stage33/stage33_evidence_failure_analysis/stage33_failure_type_counts.csv",
            warnings,
        ),
        "stage33_recommendations": read_json_optional(
            ROOT / "results_stage33/stage33_evidence_failure_analysis/stage33_recommendations.json",
            warnings,
        ),
        "stage39_failure_comparison": read_csv_optional(
            stage39_dir / "stage39_failure_comparison_summary.csv",
            warnings,
        ),
        "docs_handoff": read_text_optional(ROOT / "docs/CODEX_HANDOFF.md", warnings),
    }
    return inputs


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    inputs = load_inputs(args, warnings)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage39_rec = inputs["stage39_recommendation"]
    stage49_rec = inputs["stage49_recommendation"]
    stage39_perf = inputs["stage39_performance"]
    stage39_ablation = inputs["stage39_ablation"]
    stage49_decision = inputs["stage49_decision_table"]
    stage49_registry = inputs["stage49_negative_registry"]
    stage33_counts = inputs["stage33_failure_counts"]
    stage39_failure = inputs["stage39_failure_comparison"]
    stage45_override = inputs["stage45_override_patterns"]

    primary_model = choose_primary_model(stage49_rec, stage39_rec)
    secondary_variant = choose_secondary_variant(stage49_rec, stage39_rec)
    primary_metrics = extract_primary_metrics(stage39_perf)

    main_results_df = build_main_results_table(stage39_perf, primary_model, secondary_variant)
    ablation_df = build_ablation_table(stage39_ablation)
    negative_ablation_df = build_negative_ablation_table(stage39_perf, stage49_registry, stage49_decision)
    failure_analysis_df = build_failure_analysis_table(stage33_counts, stage39_failure, stage45_override)

    output_texts = {
        "stage50_paper_outline.md": build_stage50_paper_outline(args.paper_title, primary_model, secondary_variant),
        "stage50_method_overview_final.md": build_method_overview_final(primary_model, secondary_variant),
        "stage50_main_method_figure_mermaid.md": build_main_method_figure_mermaid(),
        "stage50_evidence_pipeline_figure_mermaid.md": build_evidence_pipeline_figure_mermaid(),
        "stage50_experiment_tables_final.md": build_experiment_tables_final(primary_metrics),
        "stage50_claims_to_make_and_avoid.md": build_claims_to_make_and_avoid(),
        "stage50_limitations_future_work_final.md": build_limitations_future_work_final(),
        "stage50_rebuttal_or_defense_points.md": build_rebuttal_or_defense_points(primary_model),
        "stage50_paper_section_draft.md": build_paper_section_draft(primary_model, secondary_variant),
    }

    csv_outputs = {
        "stage50_main_results_table.csv": main_results_df,
        "stage50_ablation_table_final.csv": ablation_df,
        "stage50_negative_ablation_table_final.csv": negative_ablation_df,
        "stage50_failure_analysis_table_final.csv": failure_analysis_df,
    }

    output_file_names = [
        "stage50_paper_outline.md",
        "stage50_method_overview_final.md",
        "stage50_main_method_figure_mermaid.md",
        "stage50_evidence_pipeline_figure_mermaid.md",
        "stage50_experiment_tables_final.md",
        "stage50_main_results_table.csv",
        "stage50_ablation_table_final.csv",
        "stage50_negative_ablation_table_final.csv",
        "stage50_failure_analysis_table_final.csv",
        "stage50_claims_to_make_and_avoid.md",
        "stage50_limitations_future_work_final.md",
        "stage50_rebuttal_or_defense_points.md",
        "stage50_paper_section_draft.md",
        "stage50_final_paper_package_report.md",
        "stage50_manifest.json",
    ]

    report_text = build_final_report(
        args=args,
        output_files=output_file_names,
        primary_model=primary_model,
        secondary_variant=secondary_variant,
        primary_metrics=primary_metrics,
        warnings=warnings,
    )
    output_texts["stage50_final_paper_package_report.md"] = report_text

    manifest = {
        "step": 50,
        "paper_title": args.paper_title,
        "input_dirs": {
            "stage39_dir": relative(resolve_path(args.stage39_dir)),
            "stage40_dir": relative(resolve_path(args.stage40_dir)),
            "stage44_dir": relative(resolve_path(args.stage44_dir)),
            "stage45_dir": relative(resolve_path(args.stage45_dir)),
            "stage47_dir": relative(resolve_path(args.stage47_dir)),
            "stage47_gate_diag_dir": relative(resolve_path(args.stage47_gate_diag_dir)),
            "stage48b_dir": relative(resolve_path(args.stage48b_dir)),
            "stage49_dir": relative(resolve_path(args.stage49_dir)),
        },
        "output_files": output_file_names,
        "final_primary_model": primary_model,
        "final_secondary_variant": secondary_variant,
        "negative_ablation_modules": [
            "ordinary region graph",
            "ordinary concept graph",
            "scalar visual evidence gate",
            "HCRC-Light",
            "PRARC-v1",
            "PRARC-v2",
        ],
        "recommend_more_training_now": False,
        "recommend_start_paper_writing": True,
        "warnings": warnings,
    }

    for name, text in output_texts.items():
        (output_dir / name).write_text(text, encoding="utf-8")

    for name, frame in csv_outputs.items():
        frame.to_csv(output_dir / name, index=False)

    (output_dir / "stage50_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Stage50] Wrote final paper package to {output_dir}")
    print(f"[Stage50] Final primary model: {primary_model}")
    if warnings:
        print("[Stage50] Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
