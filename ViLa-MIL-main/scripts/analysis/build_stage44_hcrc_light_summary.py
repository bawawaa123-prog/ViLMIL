from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS = ["hcrc_a002_b8", "hcrc_a005_b8", "hcrc_a01_b8"]
METRIC_COLUMNS = [
    "test_auc",
    "test_acc",
    "test_f1",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "pr_auc",
    "val_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step44 HCRC-Light 5-fold summary.")
    parser.add_argument("--results_root", default="results_stage44")
    parser.add_argument("--baseline_results_root", default="results_stage39/final_evidence_package")
    parser.add_argument(
        "--stage43_summary",
        default="results_stage43/stage43_hcrc_light_smoke_summary/stage43_hcrc_smoke_report.md",
    )
    parser.add_argument("--output_dir", default="results_stage44/stage44_hcrc_light_summary")
    parser.add_argument("--variants", default="hcrc_a002_b8,hcrc_a005_b8,hcrc_a01_b8")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    series = pd.Series(values, dtype="float64")
    return float(series.mean()), float(series.std(ddof=0))


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def parse_variants(raw: str) -> list[str]:
    items = [item.strip() for item in str(raw).split(",")]
    return [item for item in items if item]


def find_run_dir(results_root: Path, variant: str, seed: int) -> Path | None:
    exact = results_root / f"stage44_{variant}_s{seed}"
    if exact.is_dir():
        return exact
    candidates = [
        path
        for path in results_root.glob(f"stage44_{variant}_s{seed}*")
        if path.is_dir()
    ]
    if candidates:
        return sorted(candidates)[-1]
    return None


def read_experiment_settings(run_dir: Path) -> dict[str, object]:
    files = sorted(run_dir.glob("experiment_*.txt"))
    if not files:
        return {}
    try:
        return ast.literal_eval(files[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_fold_summary(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "fold_summary.csv",
        run_dir / "summary.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    extra = sorted(run_dir.glob("**/*fold_summary*.csv"))
    if extra:
        return extra[0]
    extra = sorted(run_dir.glob("**/*summary*.csv"))
    for candidate in extra:
        if "partial" not in candidate.name.lower():
            return candidate
    return None


def normalize_fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(col).strip() for col in renamed.columns]
    if "fold" not in renamed.columns and "folds" in renamed.columns:
        renamed = renamed.rename(columns={"folds": "fold"})
        renamed["fold"] = renamed["fold"] + 1
    if "Unnamed: 0" in renamed.columns and "fold" not in renamed.columns:
        renamed["fold"] = renamed["Unnamed: 0"] + 1
    keep_cols = [col for col in ["fold", *METRIC_COLUMNS] if col in renamed.columns]
    return renamed[keep_cols].copy()


def inspect_variant_logs(results_root: Path, variant: str, seed: int) -> dict[str, object]:
    log_path = results_root / "logs" / f"stage44_{variant}_s{seed}.log"
    info = {
        "log_exists": log_path.is_file(),
        "log_path": str(log_path),
        "has_traceback": None,
        "has_nan_or_inf": None,
    }
    if not log_path.is_file():
        return info
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    info["has_traceback"] = "Traceback" in text
    info["has_nan_or_inf"] = bool(re.search(r"(?i)\b(?:nan|inf)\b", text))
    return info


def collect_variant_outputs(run_dir: Path | None, variant: str, results_root: Path, seed: int) -> dict[str, object]:
    out = {
        "variant": variant,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "fold_summary_path": None,
        "checkpoint_count": 0,
        "missing_folds": None,
        "status": "missing",
        "settings": {},
        "fold_df": pd.DataFrame(),
    }
    out.update(inspect_variant_logs(results_root, variant, seed))
    if run_dir is None:
        return out

    out["settings"] = read_experiment_settings(run_dir)
    out["checkpoint_count"] = len(list(run_dir.glob("s_*_checkpoint.pt")))
    summary_path = find_fold_summary(run_dir)
    out["fold_summary_path"] = str(summary_path) if summary_path is not None else None
    if summary_path is None:
        out["status"] = "missing_fold_summary"
        return out

    try:
        fold_df = normalize_fold_summary(pd.read_csv(summary_path))
    except Exception:
        out["status"] = "invalid_fold_summary"
        return out

    out["fold_df"] = fold_df
    if fold_df.empty:
        out["status"] = "empty_fold_summary"
        return out

    if "fold" in fold_df.columns:
        expected = {1, 2, 3, 4, 5}
        present = {int(value) for value in fold_df["fold"].dropna().tolist()}
        out["missing_folds"] = sorted(expected - present)
    else:
        out["missing_folds"] = [1, 2, 3, 4, 5]

    out["status"] = "ok"
    return out


def summarize_variant(collected: dict[str, object]) -> dict[str, object]:
    row = {
        "variant": collected["variant"],
        "run_dir": collected["run_dir"],
        "status": collected["status"],
        "checkpoint_count": collected["checkpoint_count"],
        "fold_summary_path": collected["fold_summary_path"],
        "log_exists": collected["log_exists"],
        "has_traceback": collected["has_traceback"],
        "has_nan_or_inf": collected["has_nan_or_inf"],
        "missing_folds": ",".join(map(str, collected["missing_folds"])) if collected["missing_folds"] else "",
        "hcrc_alpha_init": collected["settings"].get("rce_hcrc_alpha_init"),
        "hcrc_bbox_expand": collected["settings"].get("rce_hcrc_bbox_expand"),
    }
    fold_df = collected["fold_df"]
    for metric in METRIC_COLUMNS:
        values = []
        if metric in fold_df.columns:
            values = [float(v) for v in fold_df[metric].dropna().tolist()]
        mean_value, std_value = mean_std(values)
        row[f"{metric}_mean"] = mean_value
        row[f"{metric}_std"] = std_value
    return row


def load_baseline_metrics(baseline_root: Path) -> tuple[dict[str, float] | None, str | None]:
    summary_path = baseline_root / "stage39_final_performance_summary.csv"
    if not summary_path.is_file():
        return None, f"Baseline summary not found: {summary_path}"
    try:
        df = pd.read_csv(summary_path)
    except Exception as exc:
        return None, f"Failed to read baseline summary: {exc}"

    mask = df["method"].astype(str).str.contains("RCE-v4-CSG-a01-rq16", na=False)
    rows = df[mask].copy()
    if rows.empty:
        return None, "Baseline row for RCE-v4-CSG-a01-rq16 not found."
    preferred = rows[rows["variant"].astype(str) == "rq16"]
    if preferred.empty:
        preferred = rows.iloc[[0]]
    row = preferred.iloc[0]
    metrics = {}
    for metric in [
        "test_auc",
        "test_acc",
        "test_f1",
        "balanced_acc",
        "sensitivity",
        "specificity",
        "pr_auc",
    ]:
        if metric in row.index:
            metrics[metric] = float(row[metric])
    return metrics, None


def build_delta_rows(summary_rows: list[dict[str, object]], baseline_metrics: dict[str, float] | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        delta_row = {
            "variant": row["variant"],
            "baseline_available": baseline_metrics is not None,
        }
        for metric in [
            "test_auc",
            "test_acc",
            "test_f1",
            "balanced_acc",
            "sensitivity",
            "specificity",
            "pr_auc",
        ]:
            hcrc_value = row.get(f"{metric}_mean")
            baseline_value = baseline_metrics.get(metric) if baseline_metrics is not None else None
            delta_row[f"{metric}_baseline"] = baseline_value
            delta_row[f"{metric}_hcrc"] = hcrc_value
            if baseline_value is None or hcrc_value is None:
                delta_row[f"{metric}_delta"] = None
            else:
                delta_row[f"{metric}_delta"] = float(hcrc_value) - float(baseline_value)
        rows.append(delta_row)
    return rows


def choose_recommended_variant(summary_rows: list[dict[str, object]], baseline_metrics: dict[str, float] | None) -> tuple[str | None, str]:
    valid_rows = [row for row in summary_rows if row.get("status") == "ok"]
    if not valid_rows:
        return None, "No valid Step44 variant results were found."

    if baseline_metrics is None:
        best = max(valid_rows, key=lambda row: row.get("test_auc_mean") or float("-inf"))
        return best["variant"], "Baseline unavailable; selected highest mean test_auc among valid HCRC variants."

    def decision_score(row: dict[str, object]) -> tuple[float, float, float, float]:
        return (
            float(row.get("balanced_acc_mean") or float("-inf")),
            float(row.get("test_f1_mean") or float("-inf")),
            float(row.get("pr_auc_mean") or float("-inf")),
            float(row.get("test_auc_mean") or float("-inf")),
        )

    candidates = []
    for row in valid_rows:
        auc = row.get("test_auc_mean")
        pr_auc = row.get("pr_auc_mean")
        acc = row.get("test_acc_mean")
        f1 = row.get("test_f1_mean")
        bal = row.get("balanced_acc_mean")
        sens = row.get("sensitivity_mean")
        spec = row.get("specificity_mean")
        if auc is None or pr_auc is None or acc is None or f1 is None or bal is None:
            continue
        cond1 = (
            auc >= baseline_metrics["test_auc"]
            and pr_auc >= baseline_metrics["pr_auc"]
            and acc > baseline_metrics["test_acc"]
            and f1 > baseline_metrics["test_f1"]
            and bal > baseline_metrics["balanced_acc"]
        )
        cond2 = (
            auc >= baseline_metrics["test_auc"] - 0.005
            and (
                f1 > baseline_metrics["test_f1"]
                or bal > baseline_metrics["balanced_acc"]
                or (sens is not None and spec is not None and abs(sens - spec) < abs(baseline_metrics["sensitivity"] - baseline_metrics["specificity"]))
            )
        )
        cond3 = (
            abs(auc - baseline_metrics["test_auc"]) <= 0.003
            and abs(pr_auc - baseline_metrics["pr_auc"]) <= 0.003
        )
        if cond1 or cond2 or cond3:
            candidates.append(row)

    if candidates:
        best = max(candidates, key=decision_score)
        return best["variant"], "At least one Step44 decision rule was satisfied."

    best = max(valid_rows, key=decision_score)
    return best["variant"], "No variant beat baseline by the Step44 rules; selected strongest internal HCRC variant for reporting."


def should_enter_step45(
    recommended_variant: str | None,
    summary_rows: list[dict[str, object]],
    baseline_metrics: dict[str, float] | None,
) -> tuple[bool, str]:
    if recommended_variant is None:
        return False, "No valid HCRC result was available."
    row = next((item for item in summary_rows if item["variant"] == recommended_variant), None)
    if row is None:
        return False, "Recommended variant row missing."
    if baseline_metrics is None:
        return True, "Baseline unavailable; Step45 is still useful for evidence-level inspection."

    auc = row.get("test_auc_mean")
    pr_auc = row.get("pr_auc_mean")
    acc = row.get("test_acc_mean")
    f1 = row.get("test_f1_mean")
    bal = row.get("balanced_acc_mean")
    sens = row.get("sensitivity_mean")
    spec = row.get("specificity_mean")
    if auc is None or pr_auc is None or acc is None or f1 is None or bal is None:
        return False, "Recommended variant is missing core metrics."

    if (
        auc >= baseline_metrics["test_auc"]
        and pr_auc >= baseline_metrics["pr_auc"]
        and (acc > baseline_metrics["test_acc"] or f1 > baseline_metrics["test_f1"] or bal > baseline_metrics["balanced_acc"])
    ):
        return True, "AUC/PR-AUC are not below baseline and at least one core classification metric improved."

    if (
        auc >= baseline_metrics["test_auc"] - 0.005
        and (
            f1 > baseline_metrics["test_f1"]
            or bal > baseline_metrics["balanced_acc"]
            or (sens is not None and spec is not None and abs(sens - spec) < abs(baseline_metrics["sensitivity"] - baseline_metrics["specificity"]))
        )
    ):
        return True, "AUC drop is limited while trade-off metrics improved."

    if abs(auc - baseline_metrics["test_auc"]) <= 0.003 and abs(pr_auc - baseline_metrics["pr_auc"]) <= 0.003:
        return True, "Performance is close enough to baseline to justify Step45 evidence analysis."

    return False, "All valid HCRC variants remain below baseline by the Step44 decision rules."


def build_report(
    stage43_summary_path: Path,
    variants: list[str],
    summary_rows: list[dict[str, object]],
    baseline_metrics: dict[str, float] | None,
    baseline_warning: str | None,
    recommended_variant: str | None,
    recommendation_reason: str,
    enter_step45: bool,
    enter_reason: str,
) -> str:
    lines = [
        "# Stage44 HCRC-Light 5-fold Report",
        "",
        "## Purpose",
        "- Formally evaluate whether Step43 HCRC-Light improves over the current `RCE-v4-CSG-a01-rq16 / DEG skeleton` baseline at the 5-fold performance level.",
        "- Step44 is performance-first. HCRC debug/evidence export is not forced here if the stored outputs are insufficient; that work belongs to Step45.",
        "",
        "## Step43 Inheritance",
        f"- Step43 summary source: `{stage43_summary_path}`",
        "- Step42b recommended spatial settings were kept: proposal_radius=4096, nms_radius=512, num_anchors=16, num_high_children=16, coord_mode=top_left, scale_ratio=1.0, child_strategy=bbox_containment.",
        "",
        "## Variants",
        f"- Requested variants: `{', '.join(variants)}`",
        "",
        "## 5-fold Metrics",
    ]

    for row in summary_rows:
        lines.extend(
            [
                f"### {row['variant']}",
                f"- status: `{row['status']}`",
                f"- checkpoints: `{row['checkpoint_count']}`",
                f"- traceback: `{row['has_traceback']}`",
                f"- NaN/Inf in log: `{row['has_nan_or_inf']}`",
                f"- missing folds: `{row['missing_folds'] or 'none'}`",
                f"- alpha_init: `{row.get('hcrc_alpha_init')}`",
                f"- bbox_expand: `{row.get('hcrc_bbox_expand')}`",
                f"- test_auc: `{format_metric(row.get('test_auc_mean'))} +/- {format_metric(row.get('test_auc_std'))}`",
                f"- test_acc: `{format_metric(row.get('test_acc_mean'))} +/- {format_metric(row.get('test_acc_std'))}`",
                f"- test_f1: `{format_metric(row.get('test_f1_mean'))} +/- {format_metric(row.get('test_f1_std'))}`",
                f"- balanced_acc: `{format_metric(row.get('balanced_acc_mean'))} +/- {format_metric(row.get('balanced_acc_std'))}`",
                f"- sensitivity: `{format_metric(row.get('sensitivity_mean'))} +/- {format_metric(row.get('sensitivity_std'))}`",
                f"- specificity: `{format_metric(row.get('specificity_mean'))} +/- {format_metric(row.get('specificity_std'))}`",
                f"- pr_auc: `{format_metric(row.get('pr_auc_mean'))} +/- {format_metric(row.get('pr_auc_std'))}`",
                f"- val_auc: `{format_metric(row.get('val_auc_mean'))} +/- {format_metric(row.get('val_auc_std'))}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Baseline Comparison",
        ]
    )
    if baseline_metrics is None:
        lines.append(f"- Baseline could not be loaded automatically. Reason: `{baseline_warning}`")
        lines.append("- Report falls back to internal HCRC comparison only.")
    else:
        lines.extend(
            [
                f"- baseline test_auc: `{format_metric(baseline_metrics.get('test_auc'))}`",
                f"- baseline test_acc: `{format_metric(baseline_metrics.get('test_acc'))}`",
                f"- baseline test_f1: `{format_metric(baseline_metrics.get('test_f1'))}`",
                f"- baseline balanced_acc: `{format_metric(baseline_metrics.get('balanced_acc'))}`",
                f"- baseline sensitivity: `{format_metric(baseline_metrics.get('sensitivity'))}`",
                f"- baseline specificity: `{format_metric(baseline_metrics.get('specificity'))}`",
                f"- baseline pr_auc: `{format_metric(baseline_metrics.get('pr_auc'))}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Stability",
            "- Step44 checks logs, fold summaries, checkpoint counts, missing folds, and NaN/Inf tokens.",
            "- Performance-level validation is the primary goal in Step44; HCRC debug/evidence details should be exported in Step45 if needed.",
            "",
            "## Recommendation",
            f"- recommended variant: `{recommended_variant}`",
            f"- recommendation reason: `{recommendation_reason}`",
            f"- enter Step45 HCRC Evidence Export and Failure Comparison: `{enter_step45}`",
            f"- Step45 rationale: `{enter_reason}`",
            "",
            "## Risk",
            "- `proposal_radius=4096` can widen the effective low-anchor support and shift anchor coordinates.",
            "- `bbox_expand=8` can introduce broad high-scale evidence regions.",
            "- Even if Step44 performance is acceptable, Step45 must verify whether the evidence is clinically and spatially reliable.",
        ]
    )

    valid_rows = [row for row in summary_rows if row.get("status") == "ok"]
    if valid_rows:
        best_alpha_row = max(
            valid_rows,
            key=lambda row: (
                float(row.get("balanced_acc_mean") or float("-inf")),
                -float(row.get("balanced_acc_std") or float("inf")),
                float(row.get("test_f1_mean") or float("-inf")),
            ),
        )
        lines.insert(
            lines.index("## Stability"),
            f"- most stable alpha among valid runs: `{best_alpha_row.get('hcrc_alpha_init')}` from `{best_alpha_row['variant']}`",
        )

        tradeoff_notes = []
        for row in valid_rows:
            sens = row.get("sensitivity_mean")
            spec = row.get("specificity_mean")
            if sens is not None and spec is not None:
                tradeoff_notes.append(
                    f"{row['variant']}: sens={format_metric(sens)}, spec={format_metric(spec)}"
                )
        if tradeoff_notes:
            lines.insert(lines.index("## Recommendation"), "## Sensitivity / Specificity Trade-off")
            lines.insert(lines.index("## Recommendation") + 1, "- " + " | ".join(tradeoff_notes))
            lines.insert(lines.index("## Recommendation") + 2, "")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    baseline_root = resolve_path(args.baseline_results_root)
    stage43_summary_path = resolve_path(args.stage43_summary)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants)
    if not variants:
        variants = list(DEFAULT_VARIANTS)

    collected = []
    for variant in variants:
        run_dir = find_run_dir(results_root, variant, args.seed)
        collected.append(collect_variant_outputs(run_dir, variant, results_root, args.seed))

    fold_rows = []
    summary_rows = []
    stability_rows = []
    for item in collected:
        if not item["fold_df"].empty:
            fold_df = item["fold_df"].copy()
            fold_df.insert(0, "variant", item["variant"])
            fold_rows.extend(fold_df.to_dict(orient="records"))
        summary_rows.append(summarize_variant(item))
        stability_rows.append(
            {
                "variant": item["variant"],
                "status": item["status"],
                "run_dir": item["run_dir"],
                "fold_summary_path": item["fold_summary_path"],
                "checkpoint_count": item["checkpoint_count"],
                "log_exists": item["log_exists"],
                "log_path": item["log_path"],
                "has_traceback": item["has_traceback"],
                "has_nan_or_inf": item["has_nan_or_inf"],
                "missing_folds": ",".join(map(str, item["missing_folds"])) if item["missing_folds"] else "",
            }
        )

    baseline_metrics, baseline_warning = load_baseline_metrics(baseline_root)
    delta_rows = build_delta_rows(summary_rows, baseline_metrics)
    recommended_variant, recommendation_reason = choose_recommended_variant(summary_rows, baseline_metrics)
    enter_step45, enter_reason = should_enter_step45(recommended_variant, summary_rows, baseline_metrics)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    delta_df = pd.DataFrame(delta_rows)
    stability_df = pd.DataFrame(stability_rows)

    summary_csv = output_dir / "stage44_hcrc_5fold_summary.csv"
    fold_csv = output_dir / "stage44_hcrc_fold_metrics.csv"
    delta_csv = output_dir / "stage44_hcrc_vs_baseline.csv"
    stability_csv = output_dir / "stage44_hcrc_stability_check.csv"
    report_md = output_dir / "stage44_hcrc_light_report.md"
    manifest_json = output_dir / "stage44_manifest.json"

    summary_df.to_csv(summary_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    stability_df.to_csv(stability_csv, index=False)

    report_text = build_report(
        stage43_summary_path=stage43_summary_path,
        variants=variants,
        summary_rows=summary_rows,
        baseline_metrics=baseline_metrics,
        baseline_warning=baseline_warning,
        recommended_variant=recommended_variant,
        recommendation_reason=recommendation_reason,
        enter_step45=enter_step45,
        enter_reason=enter_reason,
    )
    report_md.write_text(report_text, encoding="utf-8")

    manifest = {
        "step": "Step44 HCRC-Light 5-fold",
        "results_root": str(results_root),
        "baseline_results_root": str(baseline_root),
        "stage43_summary": str(stage43_summary_path),
        "output_dir": str(output_dir),
        "variants": variants,
        "seed": args.seed,
        "successful_variants": [row["variant"] for row in summary_rows if row["status"] == "ok"],
        "failed_variants": [row["variant"] for row in summary_rows if row["status"] != "ok"],
        "recommended_variant": recommended_variant,
        "recommendation_reason": recommendation_reason,
        "baseline_available": baseline_metrics is not None,
        "baseline_warning": baseline_warning,
        "enter_step45": enter_step45,
        "enter_step45_reason": enter_reason,
        "output_paths": {
            "summary_csv": str(summary_csv),
            "fold_csv": str(fold_csv),
            "delta_csv": str(delta_csv),
            "stability_csv": str(stability_csv),
            "report_md": str(report_md),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Done] Summary CSV: {summary_csv}")
    print(f"[Done] Fold CSV: {fold_csv}")
    print(f"[Done] Baseline delta CSV: {delta_csv}")
    print(f"[Done] Stability CSV: {stability_csv}")
    print(f"[Done] Report: {report_md}")
    print(f"[Done] Manifest: {manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
