from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step49 final consolidation package.")
    parser.add_argument("--stage39_dir", default="results_stage39/final_evidence_package")
    parser.add_argument("--stage44_dir", default="results_stage44/stage44_hcrc_light_summary")
    parser.add_argument("--stage45_dir", default="results_stage45/prarc_reliability_audit")
    parser.add_argument("--stage47_dir", default="results_stage47/stage47_prarc_gate_summary")
    parser.add_argument("--stage47_gate_diag_dir", default="results_stage47/stage47_prarc_gate_diagnostics")
    parser.add_argument("--stage48b_dir", default="results_stage48/stage48b_prarc_v2_variant_sweep_summary")
    parser.add_argument("--output_dir", default="results_stage49/final_consolidation")
    parser.add_argument("--baseline_name", default="RCE-v4-CSG-a01-rq16 / DEG skeleton")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def read_csv_optional(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warnings.append(f"Missing CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Failed to read CSV {path}: {exc}")
        return None


def read_json_optional(path: Path, warnings: list[str]) -> dict | None:
    if not path.is_file():
        warnings.append(f"Missing JSON: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Failed to read JSON {path}: {exc}")
        return None


def read_text_optional(path: Path, warnings: list[str]) -> str | None:
    if not path.is_file():
        warnings.append(f"Missing text/markdown: {path}")
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"Failed to read text {path}: {exc}")
        return None


def safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def fmt_metric(value) -> str:
    value = safe_float(value)
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def metric_delta_text(name: str, delta: float | None) -> str | None:
    delta = safe_float(delta)
    if delta is None:
        return None
    if abs(delta) < 1e-9:
        return f"{name} +0.0000"
    return f"{name} {delta:+.4f}"


def join_metric_text(parts: list[str | None]) -> str:
    items = [part for part in parts if part]
    return "; ".join(items) if items else "N/A"


def choose_baseline_row(stage39_perf: pd.DataFrame | None) -> dict | None:
    if stage39_perf is None or stage39_perf.empty:
        return None

    candidates = stage39_perf[
        stage39_perf["method"].astype(str).str.contains("RCE-v4-CSG-a01-rq16", na=False)
    ].copy()
    if candidates.empty:
        candidates = stage39_perf[
            stage39_perf["variant"].astype(str).isin(["rq16", "csg_a01", "skeleton"])
        ].copy()
    if candidates.empty:
        return stage39_perf.iloc[0].to_dict()

    preferred = candidates[candidates["variant"].astype(str) == "rq16"]
    if preferred.empty:
        preferred = candidates[candidates["variant"].astype(str) == "csg_a01"]
    if preferred.empty:
        preferred = candidates
    return preferred.iloc[0].to_dict()


def choose_stage39_variant(stage39_perf: pd.DataFrame | None, variant_name: str) -> dict | None:
    if stage39_perf is None or stage39_perf.empty:
        return None
    rows = stage39_perf[stage39_perf["variant"].astype(str) == variant_name]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def build_decision_row(
    module_or_variant: str,
    source_stage: str,
    status: str,
    metrics: dict | None,
    main_gain: str,
    main_drop: str,
    decision: str,
    reason: str,
) -> dict[str, object]:
    metrics = metrics or {}
    return {
        "module_or_variant": module_or_variant,
        "source_stage": source_stage,
        "status": status,
        "auc": safe_float(metrics.get("test_auc") or metrics.get("test_auc_mean")),
        "acc": safe_float(metrics.get("test_acc") or metrics.get("test_acc_mean")),
        "f1": safe_float(metrics.get("test_f1") or metrics.get("test_f1_mean")),
        "balanced_acc": safe_float(metrics.get("balanced_acc") or metrics.get("balanced_acc_mean")),
        "pr_auc": safe_float(metrics.get("pr_auc") or metrics.get("pr_auc_mean")),
        "main_gain": main_gain,
        "main_drop": main_drop,
        "decision": decision,
        "reason": reason,
    }


def build_hcrc_rows(stage44_summary: pd.DataFrame | None, stage44_vs_baseline: pd.DataFrame | None) -> list[dict[str, object]]:
    if stage44_summary is None or stage44_summary.empty:
        return []

    delta_map = {}
    if stage44_vs_baseline is not None and not stage44_vs_baseline.empty:
        delta_map = {row["variant"]: row for _, row in stage44_vs_baseline.iterrows()}

    rows: list[dict[str, object]] = []
    for _, row in stage44_summary.iterrows():
        variant = str(row["variant"])
        delta_row = delta_map.get(variant, {})
        gain = join_metric_text(
            [
                metric_delta_text("ACC", delta_row.get("test_acc_delta")),
                metric_delta_text("F1", delta_row.get("test_f1_delta")),
                metric_delta_text("BalAcc", delta_row.get("balanced_acc_delta")),
                metric_delta_text("Sens", delta_row.get("sensitivity_delta")),
                metric_delta_text("Spec", delta_row.get("specificity_delta")),
            ]
        )
        drop = join_metric_text(
            [
                metric_delta_text("AUC", delta_row.get("test_auc_delta")),
                metric_delta_text("PR-AUC", delta_row.get("pr_auc_delta")),
            ]
        )
        reason = (
            "Completed 5-fold cleanly but no HCRC variant exceeded the baseline; "
            "ranking metrics stayed below the default model."
        )
        rows.append(
            build_decision_row(
                module_or_variant=f"HCRC-Light {variant}",
                source_stage="Stage44",
                status="5fold_complete_not_primary",
                metrics=row.to_dict(),
                main_gain=gain,
                main_drop=drop,
                decision="negative_ablation",
                reason=reason,
            )
        )
    return rows


def build_prarc_v1_rows(stage47_summary: pd.DataFrame | None, stage47_vs_baseline: pd.DataFrame | None) -> list[dict[str, object]]:
    if stage47_summary is None or stage47_summary.empty:
        return []

    delta_map = {}
    if stage47_vs_baseline is not None and not stage47_vs_baseline.empty:
        delta_map = {row["variant"]: row for _, row in stage47_vs_baseline.iterrows()}

    rows: list[dict[str, object]] = []
    for _, row in stage47_summary.iterrows():
        variant = str(row["variant"])
        delta_row = delta_map.get(variant, {})
        gain = join_metric_text(
            [
                metric_delta_text("Spec", delta_row.get("specificity_delta")),
            ]
        )
        drop = join_metric_text(
            [
                metric_delta_text("AUC", delta_row.get("test_auc_delta")),
                metric_delta_text("ACC", delta_row.get("test_acc_delta")),
                metric_delta_text("F1", delta_row.get("test_f1_delta")),
                metric_delta_text("BalAcc", delta_row.get("balanced_acc_delta")),
                metric_delta_text("PR-AUC", delta_row.get("pr_auc_delta")),
            ]
        )
        reason = "5-fold performance stayed below baseline; Step47 concluded the gate remained near-scalar."
        rows.append(
            build_decision_row(
                module_or_variant=f"PRARC-v1 {variant}",
                source_stage="Stage47",
                status="5fold_complete_not_primary",
                metrics=row.to_dict(),
                main_gain=gain,
                main_drop=drop,
                decision="negative_ablation",
                reason=reason,
            )
        )
    return rows


def build_prarc_v2_rows(stage48b_summary: pd.DataFrame | None) -> list[dict[str, object]]:
    if stage48b_summary is None or stage48b_summary.empty:
        return []

    rows: list[dict[str, object]] = []
    for _, row in stage48b_summary.iterrows():
        variant = str(row["variant"])
        gain = join_metric_text(
            [
                metric_delta_text("gate_std", row.get("gate_std")),
                metric_delta_text("gate_range", row.get("gate_range")),
                metric_delta_text("conflict_gap", row.get("conflict_minus_nonconflict")),
            ]
        )
        drop = (
            "Smoke-only evidence; gate dynamics still too small for Step49 "
            f"(mean={fmt_metric(row.get('gate_mean'))}, std={fmt_metric(row.get('gate_std'))}, "
            f"range={fmt_metric(row.get('gate_range'))})"
        )
        reason = "Step48b completed without engineering failure, but no PRARC-v2 variant met the gate-dynamics bar."
        rows.append(
            build_decision_row(
                module_or_variant=f"PRARC-v2 {variant}",
                source_stage="Stage48b",
                status="smoke_complete_not_primary",
                metrics=row.to_dict(),
                main_gain=gain,
                main_drop=drop,
                decision="negative_ablation",
                reason=reason,
            )
        )
    return rows


def build_negative_registry(
    stage39_negative: pd.DataFrame | None,
    stage44_manifest: dict | None,
    stage47_manifest: dict | None,
    stage48b_manifest: dict | None,
    stage44_vs_baseline: pd.DataFrame | None,
    stage47_diag_report: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    if stage39_negative is not None and not stage39_negative.empty:
        stage39_map = {
            "attention-centroid region graph": "ordinary region graph",
            "concept prompt graph": "ordinary concept graph",
            "scalar visual gate": "scalar visual evidence gate",
        }
        for _, row in stage39_negative.iterrows():
            module = stage39_map.get(str(row["module"]), str(row["module"]))
            if module == "low-high consistency":
                continue
            rows.append(
                {
                    "module": module,
                    "reason_failed": str(row["negative_ablation_statement"]),
                    "evidence_stage": "Stage39",
                    "metric_effect": str(row["implication"]),
                    "whether_keep_for_paper": "yes",
                    "suggested_wording": str(row["paper_ready_interpretation"]),
                }
            )

    hcrc_reason = "HCRC-Light stayed below baseline on AUC/PR-AUC even when some classification metrics improved slightly."
    hcrc_metric = "Stage44: best AUC delta -0.0019 and best PR-AUC delta -0.0093 versus baseline."
    if stage44_vs_baseline is not None and not stage44_vs_baseline.empty:
        best_auc = stage44_vs_baseline["test_auc_delta"].max()
        best_pr = stage44_vs_baseline["pr_auc_delta"].max()
        hcrc_metric = f"Stage44: best AUC delta {best_auc:+.4f}; best PR-AUC delta {best_pr:+.4f} versus baseline."
    rows.append(
        {
            "module": "HCRC-Light",
            "reason_failed": hcrc_reason,
            "evidence_stage": "Stage44/Stage45",
            "metric_effect": hcrc_metric,
            "whether_keep_for_paper": "yes",
            "suggested_wording": "HCRC-Light was systematically evaluated but did not surpass the primary cross-scale evidence baseline.",
        }
    )

    prarc_v1_metric = "Stage47: best variant prarc_v1_g05 still under baseline and diagnostics reported near-scalar behavior."
    if stage47_diag_report:
        prarc_v1_metric = (
            "Stage47: best variant prarc_v1_g05 remained below baseline; diagnostics showed weak sample-adaptivity except limited spread at g05."
        )
    rows.append(
        {
            "module": "PRARC-v1",
            "reason_failed": "5-fold performance fell below baseline and the gate frequently behaved like a weak scalar control.",
            "evidence_stage": "Stage47",
            "metric_effect": prarc_v1_metric,
            "whether_keep_for_paper": "yes",
            "suggested_wording": "PRARC-v1 was a disciplined adaptive-gating attempt, but its gate dynamics and ranking metrics were insufficient.",
        }
    )

    prarc_v2_reason = "PRARC-v2 smoke variants were engineering-stable but still failed the gate-dynamics threshold needed for promotion."
    prarc_v2_metric = "Stage48b: best conflict-aware variant improved conflict gap modestly, but gate_std and gate_range stayed far below the Step49 bar."
    rows.append(
        {
            "module": "PRARC-v2",
            "reason_failed": prarc_v2_reason,
            "evidence_stage": "Stage48/Stage48b",
            "metric_effect": prarc_v2_metric,
            "whether_keep_for_paper": "yes",
            "suggested_wording": "PRARC-v2 improved diagnostics only marginally and should be reported as a negative ablation rather than a final module.",
        }
    )

    return rows


def build_hcrc_prarc_consolidated_summary(
    stage44_summary: pd.DataFrame | None,
    stage44_vs_baseline: pd.DataFrame | None,
    stage47_summary: pd.DataFrame | None,
    stage47_vs_baseline: pd.DataFrame | None,
    stage48b_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if stage44_summary is not None and not stage44_summary.empty:
        delta_map = {}
        if stage44_vs_baseline is not None and not stage44_vs_baseline.empty:
            delta_map = {row["variant"]: row for _, row in stage44_vs_baseline.iterrows()}
        for _, row in stage44_summary.iterrows():
            delta_row = delta_map.get(row["variant"], {})
            rows.append(
                {
                    "family": "HCRC-Light",
                    "variant": row["variant"],
                    "source_stage": "Stage44",
                    "status": row["status"],
                    "auc": safe_float(row.get("test_auc_mean")),
                    "acc": safe_float(row.get("test_acc_mean")),
                    "f1": safe_float(row.get("test_f1_mean")),
                    "balanced_acc": safe_float(row.get("balanced_acc_mean")),
                    "pr_auc": safe_float(row.get("pr_auc_mean")),
                    "auc_delta_vs_baseline": safe_float(delta_row.get("test_auc_delta")),
                    "pr_auc_delta_vs_baseline": safe_float(delta_row.get("pr_auc_delta")),
                    "gate_std": None,
                    "gate_range": None,
                    "decision": "not_primary",
                    "reason": "Below baseline on the main ranking metrics.",
                }
            )

    if stage47_summary is not None and not stage47_summary.empty:
        delta_map = {}
        if stage47_vs_baseline is not None and not stage47_vs_baseline.empty:
            delta_map = {row["variant"]: row for _, row in stage47_vs_baseline.iterrows()}
        for _, row in stage47_summary.iterrows():
            delta_row = delta_map.get(row["variant"], {})
            rows.append(
                {
                    "family": "PRARC-v1",
                    "variant": row["variant"],
                    "source_stage": "Stage47",
                    "status": row["status"],
                    "auc": safe_float(row.get("test_auc_mean")),
                    "acc": safe_float(row.get("test_acc_mean")),
                    "f1": safe_float(row.get("test_f1_mean")),
                    "balanced_acc": safe_float(row.get("balanced_acc_mean")),
                    "pr_auc": safe_float(row.get("pr_auc_mean")),
                    "auc_delta_vs_baseline": safe_float(delta_row.get("test_auc_delta")),
                    "pr_auc_delta_vs_baseline": safe_float(delta_row.get("pr_auc_delta")),
                    "gate_std": None,
                    "gate_range": None,
                    "decision": "not_primary",
                    "reason": "5-fold metrics below baseline and gate too close to scalar behavior.",
                }
            )

    if stage48b_summary is not None and not stage48b_summary.empty:
        for _, row in stage48b_summary.iterrows():
            rows.append(
                {
                    "family": "PRARC-v2",
                    "variant": row["variant"],
                    "source_stage": "Stage48b",
                    "status": "smoke",
                    "auc": safe_float(row.get("test_auc")),
                    "acc": safe_float(row.get("test_acc")),
                    "f1": safe_float(row.get("test_f1")),
                    "balanced_acc": safe_float(row.get("balanced_acc")),
                    "pr_auc": safe_float(row.get("pr_auc")),
                    "auc_delta_vs_baseline": None,
                    "pr_auc_delta_vs_baseline": None,
                    "gate_std": safe_float(row.get("gate_std")),
                    "gate_range": safe_float(row.get("gate_range")),
                    "decision": "not_primary",
                    "reason": "Smoke metrics were stable but gate dynamics stayed below the Step49 promotion bar.",
                }
            )

    return pd.DataFrame(rows)


def build_paper_claims_md(
    baseline_name: str,
    stage39_rec: dict | None,
    stage44_manifest: dict | None,
    stage47_manifest: dict | None,
    stage48b_manifest: dict | None,
) -> str:
    recommended_default = baseline_name
    if stage39_rec and stage39_rec.get("recommended_default_model"):
        recommended_default = stage39_rec["recommended_default_model"]

    claims = [
        "# Stage49 Paper Claims And Evidence",
        "",
        "## Paper-Ready Claims",
        f"- Final primary model should remain `{recommended_default}` because the strongest evidence package still points to it as the most robust default model.",
        "- Region-concept evidence modeling is effective because the Stage39 final package retained the cross-scale evidence model as the strongest overall design rather than reverting to plain graph or scalar-gate add-ons.",
        "- Cross-scale concept relation modeling is a key contributor because Stage39 concluded `CSG a01 > CSG a005` and retained the CSG-equipped skeleton as the primary model.",
        "- The current main failure source is visual residual override rather than missing graph complexity, because Stage33/39/45 repeatedly showed that wrong visual residual support explains a large share of residual errors.",
        "- HCRC and PRARC should be framed as systematic exploratory branches rather than failed implementation attempts, because they were validated through dedicated smoke, 5-fold, and diagnostic stages before being rejected.",
        "- The final model choice is evidence-driven: Stage39 selected the default model, Stage44 showed HCRC remained below baseline, Stage47 showed PRARC-v1 remained below baseline, and Stage48b showed PRARC-v2 gate dynamics stayed insufficient.",
        "",
        "## Claims To Avoid",
        "- Do not claim that HCRC improved the final model.",
        "- Do not claim that PRARC already solved visual residual override.",
        "- Do not claim that the PRARC gate became strongly sample-adaptive.",
        "- Do not claim that low-high spatial correspondence is inherently superior to concept-level cross-scale evidence reasoning.",
        "",
        "## Evidence Anchors",
        "- Stage39 final recommendation and ablation summaries provide the main primary-model evidence.",
        "- Stage44 HCRC 5-fold summary provides the formal negative result for HCRC-Light.",
        "- Stage45 reliability audit supports the statement that visual residual override is still the main bottleneck.",
        "- Stage47 PRARC 5-fold and gate diagnostics provide the formal PRARC-v1 negative result.",
        "- Stage48b PRARC-v2 variant sweep provides the final PRARC-v2 negative-ablation evidence.",
        "",
    ]
    return "\n".join(claims) + "\n"


def build_limitations_md() -> str:
    lines = [
        "# Stage49 Limitations And Future Work",
        "",
        "## Current Limitations",
        "- Current HCRC relies on relatively large proposal radius and bbox expansion, which can pull in weak or spatially loose high-scale evidence.",
        "- Current PRARC gates can still collapse toward scalar-like behavior instead of maintaining meaningful sample-adaptive spread.",
        "- Visual residual override remains a major bottleneck and is not solved by simply stacking more ordinary graph or gating modules.",
        "",
        "## Better Next Directions",
        "- Loss-level visual residual calibration instead of another direct residual gate.",
        "- Uncertainty-aware residual suppression to penalize visually confident but concept-inconsistent residual corrections.",
        "- Train-split-only reliability learning so prompt or residual reliability signals are learned without test-derived leakage.",
        "- Evidence-margin auxiliary loss to reward cleaner separation between concept-supported correct slides and residual-overridden failures.",
        "- Stronger concept evidence construction rather than continued stacking of ordinary graph smoothing or scalar-style gates.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_next_routes_md() -> str:
    lines = [
        "# Stage49 Next Research Routes",
        "",
        "## Route A: Conservative",
        "- Stop structure stacking and keep the current baseline as the final model.",
        "- Prepare the full ablation package for paper writing, with HCRC and PRARC reported as negative ablations.",
        "- Spend effort on clearer evidence visualizations, failure narratives, and reviewer-facing ablation framing instead of more training.",
        "",
        "## Route B: Moderate",
        "- Add an evidence-margin auxiliary loss that encourages concept-consistent decisions without directly gating the visual residual branch.",
        "- Treat reduced visual override as the primary optimization target rather than raw branch fusion complexity.",
        "- Re-run only targeted experiments that directly test whether residual override frequency drops.",
        "",
        "## Route C: Ambitious",
        "- Redesign the residual branch so visual residual predicts only the residual error on top of concept logits instead of competing as a broad override signal.",
        "- Explore train-fold reliability distillation or uncertainty-aware residual supervision.",
        "- Avoid any test-derived prompt reliability signal in the learning path.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_final_report_md(
    baseline_name: str,
    baseline_row: dict | None,
    secondary_variant: str | None,
    stage44_manifest: dict | None,
    stage47_manifest: dict | None,
    stage48b_manifest: dict | None,
    warnings: list[str],
) -> str:
    lines = [
        "# Stage49 Final Consolidation Report",
        "",
        "## Project Overview",
        "- Up to Step48b, the project completed a full search over cross-scale evidence modeling, low-high consistency, HCRC, PRARC-v1, and PRARC-v2 diagnostics.",
        "- The final objective of Step49 is consolidation rather than further training or architecture changes.",
        "",
        "## Final Primary Model",
        f"- final_primary_model: `{baseline_name}`",
        f"- final_primary_model_auc: `{fmt_metric((baseline_row or {}).get('test_auc'))}`",
        f"- final_primary_model_acc: `{fmt_metric((baseline_row or {}).get('test_acc'))}`",
        f"- final_primary_model_f1: `{fmt_metric((baseline_row or {}).get('test_f1'))}`",
        f"- final_primary_model_balanced_acc: `{fmt_metric((baseline_row or {}).get('balanced_acc'))}`",
        f"- final_primary_model_pr_auc: `{fmt_metric((baseline_row or {}).get('pr_auc'))}`",
    ]
    if secondary_variant:
        lines.append(f"- final_secondary_variant: `{secondary_variant}`")
    lines.extend(
        [
            "",
            "## HCRC Summary",
            f"- enter_step45_from_stage44: `{(stage44_manifest or {}).get('enter_step45')}`",
            "- HCRC-Light completed clean 5-fold evaluation but did not exceed baseline on the main ranking metrics.",
            "- The HCRC branch should be retained as a negative ablation / future-work branch rather than promoted to the main model.",
            "",
            "## PRARC-v1 Summary",
            f"- recommend_enter_step48_from_stage47: `{(stage47_manifest or {}).get('recommend_enter_step48')}`",
            "- PRARC-v1 completed 5-fold evaluation but remained below baseline, and Step47 diagnostics indicated weak or near-scalar gate behavior.",
            "",
            "## PRARC-v2 Summary",
            f"- recommend_enter_step49_from_stage48b: `{(stage48b_manifest or {}).get('recommend_enter_step49')}`",
            "- PRARC-v2 smoke variants were engineering-stable, but Step48b still found insufficient gate dynamics for promotion.",
            "",
            "## Negative Ablation Summary",
            "- Ordinary region graph, ordinary concept graph, scalar visual evidence gate, HCRC-Light, PRARC-v1, and PRARC-v2 should all remain outside the final primary model.",
            "",
            "## Paper Narrative",
            "- The strongest paper narrative is that region-concept evidence modeling plus cross-scale concept reasoning is the current winning path.",
            "- The story should emphasize systematic exploration and rejection of alternative graph/gate modules rather than over-claiming new modules as successful.",
            "- Visual residual override should be presented as the main unresolved bottleneck.",
            "",
            "## Next Step Recommendation",
            "- Do not continue HCRC or PRARC training in the current branch.",
            "- If continuing research, prioritize loss-level or uncertainty-aware residual calibration instead of more direct residual gating.",
            "- Start organizing paper materials now because the primary model and negative-ablation package are already well supported.",
            "",
            "## Execution Decision",
            "- recommend_more_training_now: `False`",
            "- recommend_start_writing_paper_materials: `True`",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage39_dir = resolve_path(args.stage39_dir)
    stage44_dir = resolve_path(args.stage44_dir)
    stage45_dir = resolve_path(args.stage45_dir)
    stage47_dir = resolve_path(args.stage47_dir)
    stage47_diag_dir = resolve_path(args.stage47_gate_diag_dir)
    stage48b_dir = resolve_path(args.stage48b_dir)

    warnings: list[str] = []

    stage39_rec = read_json_optional(stage39_dir / "stage39_final_model_recommendation.json", warnings)
    stage39_perf = read_csv_optional(stage39_dir / "stage39_final_performance_summary.csv", warnings)
    stage39_ablation = read_csv_optional(stage39_dir / "stage39_ablation_summary.csv", warnings)
    stage39_negative = read_csv_optional(stage39_dir / "stage39_negative_ablation_summary.csv", warnings)
    stage39_calib = read_csv_optional(stage39_dir / "stage39_evidence_calibration_summary.csv", warnings)

    stage44_summary = read_csv_optional(stage44_dir / "stage44_hcrc_5fold_summary.csv", warnings)
    stage44_vs_baseline = read_csv_optional(stage44_dir / "stage44_hcrc_vs_baseline.csv", warnings)
    stage44_manifest = read_json_optional(stage44_dir / "stage44_manifest.json", warnings)
    stage44_report = read_text_optional(stage44_dir / "stage44_hcrc_light_report.md", warnings)

    stage45_hcrc_note = read_text_optional(stage45_dir / "stage45_hcrc_negative_ablation_note.md", warnings)
    stage45_prarc_report = read_text_optional(stage45_dir / "stage45_prarc_reliability_report.md", warnings)
    stage45_prarc_features = read_json_optional(stage45_dir / "stage45_prarc_feature_set.json", warnings)

    stage47_summary = read_csv_optional(stage47_dir / "stage47_prarc_5fold_summary.csv", warnings)
    stage47_vs_baseline = read_csv_optional(stage47_dir / "stage47_prarc_vs_baseline.csv", warnings)
    stage47_report = read_text_optional(stage47_dir / "stage47_prarc_gate_report.md", warnings)
    stage47_manifest = read_json_optional(stage47_dir / "stage47_manifest.json", warnings)

    stage47_gate_dist = read_csv_optional(stage47_diag_dir / "stage47_prarc_gate_distribution_summary.csv", warnings)
    stage47_gate_condition = read_csv_optional(stage47_diag_dir / "stage47_prarc_gate_by_condition.csv", warnings)
    stage47_gate_report = read_text_optional(stage47_diag_dir / "stage47_prarc_gate_diagnostics_report.md", warnings)

    stage48b_summary = read_csv_optional(stage48b_dir / "stage48b_prarc_v2_variant_sweep_summary.csv", warnings)
    stage48b_gate_dist = read_csv_optional(stage48b_dir / "stage48b_prarc_v2_gate_distribution.csv", warnings)
    stage48b_report = read_text_optional(stage48b_dir / "stage48b_prarc_v2_variant_report.md", warnings)
    stage48b_manifest = read_json_optional(stage48b_dir / "stage48b_manifest.json", warnings)

    baseline_row = choose_baseline_row(stage39_perf)
    secondary_variant = stage39_rec.get("secondary_tradeoff_variant") if stage39_rec else None
    lh_row = choose_stage39_variant(stage39_perf, "lh_l001_m0")
    rg_row = choose_stage39_variant(stage39_perf, "rg_k8")
    cg_row = choose_stage39_variant(stage39_perf, "cg_k8_a005")
    gate_row = choose_stage39_variant(stage39_perf, "gate1")

    decision_rows: list[dict[str, object]] = []

    primary_reason = "Stage39 final evidence package retained this as the most robust default model on AUC and PR-AUC."
    primary_gain = "Best overall ranking metrics; evidence-driven default after Stage39 consolidation."
    primary_drop = "No stronger replacement found in Stage44/47/48b."
    decision_rows.append(
        build_decision_row(
            module_or_variant=args.baseline_name,
            source_stage="Stage39",
            status="final_primary_model",
            metrics=baseline_row,
            main_gain=primary_gain,
            main_drop=primary_drop,
            decision="keep_primary",
            reason=primary_reason,
        )
    )

    if secondary_variant and lh_row:
        calib_row = stage39_calib.iloc[0].to_dict() if stage39_calib is not None and not stage39_calib.empty else {}
        decision_rows.append(
            build_decision_row(
                module_or_variant=secondary_variant,
                source_stage="Stage39",
                status="secondary_tradeoff_variant",
                metrics=lh_row,
                main_gain=(
                    f"ACC/F1/BalAcc trade-off gain; fixed_cases={int(safe_float(calib_row.get('fixed_cases')) or 0)}, "
                    f"regressed_cases={int(safe_float(calib_row.get('regressed_cases')) or 0)}, "
                    f"low_high_conflict_delta={int(safe_float(calib_row.get('low_high_conflict_delta')) or 0)}"
                ),
                main_drop=(
                    f"AUC delta {fmt_metric(calib_row.get('auc_delta'))}; PR-AUC delta {fmt_metric(calib_row.get('pr_auc_delta'))}; "
                    f"visual_override_delta {int(safe_float(calib_row.get('visual_residual_override_delta')) or 0)}"
                ),
                decision="keep_secondary_only",
                reason="Useful as an evidence-calibration trade-off, but not strong enough to replace the default main model.",
            )
        )

    if rg_row:
        decision_rows.append(
            build_decision_row(
                module_or_variant="ordinary region graph (rg_k8)",
                source_stage="Stage39/Stage28",
                status="negative_ablation",
                metrics=rg_row,
                main_gain="None that exceeded the skeleton baseline.",
                main_drop="Stage39 concluded ordinary spatial region graph did not outperform the skeleton.",
                decision="negative_ablation",
                reason="Semantic region tokens did not provide a stable enough spatial inductive bias in the current setup.",
            )
        )
    if cg_row:
        decision_rows.append(
            build_decision_row(
                module_or_variant="ordinary concept graph (cg_k8_a005)",
                source_stage="Stage39/Stage31",
                status="negative_ablation",
                metrics=cg_row,
                main_gain="None that exceeded the skeleton baseline.",
                main_drop="Stage39 concluded plain prompt-side graph smoothing weakened evidence discrimination.",
                decision="negative_ablation",
                reason="Ordinary concept/prompt graph smoothing blurred class-critical evidence instead of sharpening it.",
            )
        )
    if gate_row:
        decision_rows.append(
            build_decision_row(
                module_or_variant="scalar visual evidence gate (gate1)",
                source_stage="Stage39/Stage35",
                status="negative_ablation",
                metrics=gate_row,
                main_gain="No robust gain over the skeleton baseline.",
                main_drop="Stage39 concluded a global scalar gate could not safely suppress visual residual override.",
                decision="negative_ablation",
                reason="Visual residual interactions were sample-dependent and resisted one-number suppression.",
            )
        )

    decision_rows.extend(build_hcrc_rows(stage44_summary, stage44_vs_baseline))
    decision_rows.extend(build_prarc_v1_rows(stage47_summary, stage47_vs_baseline))
    decision_rows.extend(build_prarc_v2_rows(stage48b_summary))

    decision_table_df = pd.DataFrame(decision_rows)

    negative_registry_df = pd.DataFrame(
        build_negative_registry(
            stage39_negative=stage39_negative,
            stage44_manifest=stage44_manifest,
            stage47_manifest=stage47_manifest,
            stage48b_manifest=stage48b_manifest,
            stage44_vs_baseline=stage44_vs_baseline,
            stage47_diag_report=stage47_gate_report,
        )
    )

    hcrc_prarc_df = build_hcrc_prarc_consolidated_summary(
        stage44_summary=stage44_summary,
        stage44_vs_baseline=stage44_vs_baseline,
        stage47_summary=stage47_summary,
        stage47_vs_baseline=stage47_vs_baseline,
        stage48b_summary=stage48b_summary,
    )

    paper_claims_md = build_paper_claims_md(
        baseline_name=args.baseline_name,
        stage39_rec=stage39_rec,
        stage44_manifest=stage44_manifest,
        stage47_manifest=stage47_manifest,
        stage48b_manifest=stage48b_manifest,
    )
    limitations_md = build_limitations_md()
    next_routes_md = build_next_routes_md()
    final_report_md = build_final_report_md(
        baseline_name=args.baseline_name,
        baseline_row=baseline_row,
        secondary_variant=secondary_variant,
        stage44_manifest=stage44_manifest,
        stage47_manifest=stage47_manifest,
        stage48b_manifest=stage48b_manifest,
        warnings=warnings,
    )

    final_recommendation = {
        "final_primary_model": args.baseline_name,
        "final_secondary_variant": secondary_variant,
        "not_recommended_modules": [
            "ordinary region graph",
            "ordinary concept graph",
            "scalar visual evidence gate",
            "HCRC-Light",
            "PRARC-v1",
            "PRARC-v2",
        ],
        "evidence_supporting_primary_model": [
            "Stage39 final evidence package explicitly retained the CSG-equipped DEG skeleton as the default final model.",
            "Stage39 ablation summary concluded CSG a01 > CSG a005 and rq16 > rq8/rq32.",
            "Stage44 showed all HCRC-Light variants remained below baseline by the formal decision rules.",
            "Stage47 showed all PRARC-v1 variants remained below baseline, with weak or near-scalar gate behavior.",
            "Stage48b showed PRARC-v2 variants were stable but still failed the required gate-dynamics bar.",
        ],
        "why_hcrc_not_primary": [
            "No HCRC-Light variant exceeded the baseline on the main ranking metrics.",
            "The best HCRC AUC remained below the baseline and its PR-AUC was also lower.",
            "The large proposal/bbox settings also keep HCRC in a weaker-evidence-risk regime.",
        ],
        "why_prarc_not_primary": [
            "PRARC-v1 5-fold metrics stayed below baseline and the gate often behaved near-scalar.",
            "PRARC-v2 improved diagnostics only marginally and still failed the Step49 gate-dynamics bar.",
            "Visual residual override remains unresolved by the current PRARC designs.",
        ],
        "paper_ready_claims": [
            "Region-concept evidence modeling is effective.",
            "Cross-scale concept relation reasoning is a key part of the current strongest model.",
            "Visual residual override is the main remaining failure source.",
            "HCRC and PRARC were systematically explored but did not surpass the default model.",
            "The final model choice is evidence-driven rather than ad hoc.",
        ],
        "claims_to_avoid": [
            "HCRC improved the final model.",
            "PRARC solved visual residual override.",
            "The gate became strongly sample-adaptive.",
            "Low-high spatial correspondence is inherently better than concept-level CSG reasoning.",
        ],
        "recommended_next_step": (
            "Keep the current baseline as the paper primary model, report HCRC/PRARC as negative ablations, "
            "and only continue research if the focus shifts to loss-level or uncertainty-aware residual calibration."
        ),
    }

    manifest = {
        "step": "Step49 Final Model Consolidation after HCRC/PRARC Search",
        "baseline_name": args.baseline_name,
        "input_dirs": {
            "stage39_dir": str(stage39_dir),
            "stage44_dir": str(stage44_dir),
            "stage45_dir": str(stage45_dir),
            "stage47_dir": str(stage47_dir),
            "stage47_gate_diag_dir": str(stage47_diag_dir),
            "stage48b_dir": str(stage48b_dir),
        },
        "warnings": warnings,
        "final_primary_model": args.baseline_name,
        "final_secondary_variant": secondary_variant,
        "hcrc_conclusion": "HCRC-Light should not be promoted to the final primary model.",
        "prarc_conclusion": "PRARC-v1 and PRARC-v2 should remain negative ablations in the current branch.",
        "negative_ablation_summary": negative_registry_df["module"].tolist() if not negative_registry_df.empty else [],
        "outputs": {},
    }

    outputs = {
        "stage49_final_model_recommendation.json": output_dir / "stage49_final_model_recommendation.json",
        "stage49_final_model_decision_table.csv": output_dir / "stage49_final_model_decision_table.csv",
        "stage49_negative_ablation_registry.csv": output_dir / "stage49_negative_ablation_registry.csv",
        "stage49_hcrc_prarc_consolidated_summary.csv": output_dir / "stage49_hcrc_prarc_consolidated_summary.csv",
        "stage49_paper_claims_and_evidence.md": output_dir / "stage49_paper_claims_and_evidence.md",
        "stage49_limitations_and_future_work.md": output_dir / "stage49_limitations_and_future_work.md",
        "stage49_next_research_routes.md": output_dir / "stage49_next_research_routes.md",
        "stage49_final_consolidation_report.md": output_dir / "stage49_final_consolidation_report.md",
        "stage49_manifest.json": output_dir / "stage49_manifest.json",
    }

    outputs["stage49_final_model_recommendation.json"].write_text(
        json.dumps(final_recommendation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    decision_table_df.to_csv(outputs["stage49_final_model_decision_table.csv"], index=False)
    negative_registry_df.to_csv(outputs["stage49_negative_ablation_registry.csv"], index=False)
    hcrc_prarc_df.to_csv(outputs["stage49_hcrc_prarc_consolidated_summary.csv"], index=False)
    outputs["stage49_paper_claims_and_evidence.md"].write_text(paper_claims_md, encoding="utf-8")
    outputs["stage49_limitations_and_future_work.md"].write_text(limitations_md, encoding="utf-8")
    outputs["stage49_next_research_routes.md"].write_text(next_routes_md, encoding="utf-8")
    outputs["stage49_final_consolidation_report.md"].write_text(final_report_md, encoding="utf-8")

    manifest["outputs"] = {name: str(path) for name, path in outputs.items()}
    outputs["stage49_manifest.json"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[Done] Wrote Step49 final consolidation to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
