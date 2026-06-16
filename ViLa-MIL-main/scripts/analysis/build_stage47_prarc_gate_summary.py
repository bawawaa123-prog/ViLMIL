from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS = ["prarc_v1_g05", "prarc_v1_g08", "prarc_v1_g10"]
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
    parser = argparse.ArgumentParser(description="Build Step47 PRARC gate 5-fold summary.")
    parser.add_argument("--results_root", default="results_stage47")
    parser.add_argument("--baseline_summary_dir", default="results_stage39/final_evidence_package")
    parser.add_argument(
        "--stage46_summary",
        default="results_stage46/stage46_prarc_gate_smoke_summary/stage46_prarc_smoke_report.md",
    )
    parser.add_argument("--output_dir", default="results_stage47/stage47_prarc_gate_summary")
    parser.add_argument("--variants", default="prarc_v1_g05,prarc_v1_g08,prarc_v1_g10")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def parse_variants(raw: str) -> list[str]:
    items = [item.strip() for item in str(raw).split(",")]
    return [item for item in items if item]


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    series = pd.Series(values, dtype="float64")
    return float(series.mean()), float(series.std(ddof=0))


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def make_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(format_metric(value))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def find_run_dir(results_root: Path, variant: str, seed: int) -> Path | None:
    exact = results_root / f"stage47_{variant}_s{seed}"
    if exact.is_dir():
        return exact
    candidates = [
        path
        for path in results_root.glob(f"stage47_{variant}_s{seed}*")
        if path.is_dir()
    ]
    if candidates:
        return sorted(candidates)[-1]
    return None


def inspect_variant_logs(results_root: Path, variant: str, seed: int) -> dict[str, object]:
    log_path = results_root / "logs" / f"stage47_{variant}_s{seed}.log"
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
    patterns = [
        "**/*fold_summary*.csv",
        "**/summary.csv",
        "**/*summary*.csv",
        "**/result.csv",
        "**/*result*.csv",
    ]
    for pattern in patterns:
        for candidate in sorted(run_dir.glob(pattern)):
            name = candidate.name.lower()
            if "partial" in name:
                continue
            if candidate.is_file():
                return candidate
    return None


def normalize_fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    rename_map = {
        "auc": "test_auc",
        "acc": "test_acc",
        "f1": "test_f1",
        "bal_acc": "balanced_acc",
        "balanced_accuracy": "balanced_acc",
        "sens": "sensitivity",
        "spec": "specificity",
        "prauc": "pr_auc",
    }
    out = out.rename(columns=rename_map)
    if "fold" not in out.columns and "folds" in out.columns:
        out = out.rename(columns={"folds": "fold"})
        out["fold"] = out["fold"] + 1
    if "Unnamed: 0" in out.columns and "fold" not in out.columns:
        out["fold"] = out["Unnamed: 0"] + 1
    keep_cols = [col for col in ["fold", *METRIC_COLUMNS] if col in out.columns]
    out = out[keep_cols].copy()
    if "fold" in out.columns:
        out["fold"] = pd.to_numeric(out["fold"], errors="coerce").astype("Int64")
    for metric in METRIC_COLUMNS:
        if metric in out.columns:
            out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out


def expected_fold_indices(settings: dict[str, object]) -> list[int]:
    k = int(settings.get("num_splits", settings.get("k", 5)) or 5)
    k_start = int(settings.get("k_start", 0) or 0)
    k_end = int(settings.get("k_end", k - 1) or (k - 1))
    if k_start == -1:
        k_start = 0
    if k_end == -1:
        k_end = k - 1
    return list(range(k_start, k_end + 1))


def collect_variant_outputs(run_dir: Path | None, variant: str, results_root: Path, seed: int) -> dict[str, object]:
    out = {
        "variant": variant,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "fold_summary_path": None,
        "status": "missing",
        "settings": {},
        "fold_df": pd.DataFrame(),
        "checkpoint_count": 0,
        "checkpoint_exists": False,
        "missing_checkpoint_folds": [],
        "missing_folds": [],
    }
    out.update(inspect_variant_logs(results_root, variant, seed))
    if run_dir is None:
        return out

    settings = read_experiment_settings(run_dir)
    out["settings"] = settings
    expected_folds = expected_fold_indices(settings) if settings else list(range(5))
    out["checkpoint_count"] = len(list(run_dir.glob("s_*_checkpoint.pt")))
    out["checkpoint_exists"] = out["checkpoint_count"] > 0
    out["missing_checkpoint_folds"] = [
        fold for fold in expected_folds if not (run_dir / f"s_{fold}_checkpoint.pt").is_file()
    ]

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

    present = set()
    if "fold" in fold_df.columns:
        present = {
            int(value) - 1
            for value in fold_df["fold"].dropna().tolist()
            if pd.notna(value)
        }
    out["missing_folds"] = sorted(set(expected_folds) - present)
    out["status"] = "ok" if not out["missing_folds"] else "partial"
    return out


def summarize_variant(collected: dict[str, object]) -> dict[str, object]:
    settings = collected["settings"]
    row = {
        "variant": collected["variant"],
        "status": collected["status"],
        "run_dir": collected["run_dir"],
        "fold_summary_path": collected["fold_summary_path"],
        "checkpoint_exists": collected["checkpoint_exists"],
        "checkpoint_count": collected["checkpoint_count"],
        "log_exists": collected["log_exists"],
        "has_traceback": collected["has_traceback"],
        "has_nan_or_inf": collected["has_nan_or_inf"],
        "missing_folds": ",".join(map(str, collected["missing_folds"])) if collected["missing_folds"] else "",
        "missing_checkpoint_folds": ",".join(map(str, collected["missing_checkpoint_folds"]))
        if collected["missing_checkpoint_folds"]
        else "",
        "gate_init": settings.get("rce_prarc_gate_init"),
        "gate_hidden_dim": settings.get("rce_prarc_gate_hidden_dim"),
        "feature_set": settings.get("rce_prarc_gate_feature_set"),
        "include_optional_features": settings.get("rce_prarc_include_optional_features"),
    }
    fold_df = collected["fold_df"]
    for metric in METRIC_COLUMNS:
        values: list[float] = []
        if metric in fold_df.columns:
            values = [float(v) for v in fold_df[metric].dropna().tolist()]
        mean_value, std_value = mean_std(values)
        row[f"{metric}_mean"] = mean_value
        row[f"{metric}_std"] = std_value
    return row


def build_fold_rows(collected: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fold_df = collected["fold_df"]
    if fold_df.empty:
        return rows
    for _, fold_row in fold_df.iterrows():
        row = {"variant": collected["variant"]}
        for column in fold_df.columns:
            row[column] = fold_row[column]
        rows.append(row)
    return rows


def load_baseline_metrics(baseline_root: Path) -> tuple[dict[str, float] | None, str | None, str | None]:
    perf_path = baseline_root / "stage39_final_performance_summary.csv"
    rec_path = baseline_root / "stage39_final_model_recommendation.json"
    if not perf_path.is_file():
        return None, f"Baseline summary not found: {perf_path}", None

    recommended_name = None
    if rec_path.is_file():
        try:
            recommended_name = json.loads(rec_path.read_text(encoding="utf-8")).get("recommended_default_model")
        except Exception:
            recommended_name = None

    try:
        df = pd.read_csv(perf_path)
    except Exception as exc:
        return None, f"Failed to read baseline summary: {exc}", recommended_name

    if recommended_name:
        rows = df[df["method"].astype(str) == str(recommended_name)].copy()
        if not rows.empty:
            row = rows.iloc[0]
            metrics = {metric: float(row[metric]) for metric in METRIC_COLUMNS if metric in row.index}
            return metrics, None, recommended_name

    mask = df["method"].astype(str).str.contains("RCE-v4-CSG-a01-rq16", na=False)
    rows = df[mask].copy()
    if rows.empty:
        return None, "Baseline row for RCE-v4-CSG-a01-rq16 not found.", recommended_name

    preferred = rows[rows["variant"].astype(str).str.contains("skeleton|rq16", case=False, na=False)]
    if preferred.empty:
        preferred = rows.iloc[[0]]
    row = preferred.iloc[0]
    metrics = {metric: float(row[metric]) for metric in METRIC_COLUMNS if metric in row.index}
    return metrics, None, recommended_name or str(row.get("method"))


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
            variant_value = row.get(f"{metric}_mean")
            baseline_value = baseline_metrics.get(metric) if baseline_metrics is not None else None
            delta_row[f"{metric}_baseline"] = baseline_value
            delta_row[f"{metric}_stage47"] = variant_value
            if baseline_value is None or variant_value is None:
                delta_row[f"{metric}_delta"] = None
            else:
                delta_row[f"{metric}_delta"] = float(variant_value) - float(baseline_value)
        rows.append(delta_row)
    return rows


def load_stage46_smoke(stage46_summary_path: Path) -> dict[str, object]:
    if not stage46_summary_path.is_file():
        return {"available": False, "path": str(stage46_summary_path)}
    text = stage46_summary_path.read_text(encoding="utf-8", errors="ignore")
    keys = [
        "test_auc",
        "test_acc",
        "test_f1",
        "balanced_acc",
        "sensitivity",
        "specificity",
        "pr_auc",
        "gate_mean",
        "gate_min",
        "gate_max",
        "gate_nonconstant",
        "visual_residual_adjusted",
    ]
    data: dict[str, object] = {"available": True, "path": str(stage46_summary_path)}
    for key in keys:
        match = re.search(rf"- {re.escape(key)}: `([^`]+)`", text)
        if match:
            data[key] = match.group(1)
    return data


def load_diagnostics_summary(results_root: Path) -> dict[str, dict[str, object]]:
    diagnostics_dir = results_root / "stage47_prarc_gate_diagnostics"
    manifest_path = diagnostics_dir / "stage47_prarc_gate_diagnostics_manifest.json"
    dist_path = diagnostics_dir / "stage47_prarc_gate_distribution_summary.csv"
    cond_path = diagnostics_dir / "stage47_prarc_gate_by_condition.csv"

    summary: dict[str, dict[str, object]] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("variant_diagnostics", []):
                variant = str(item.get("variant"))
                if variant:
                    summary[variant] = dict(item)
        except Exception:
            pass

    if dist_path.is_file():
        df = pd.read_csv(dist_path)
        for variant, group in df.groupby("variant"):
            entry = summary.setdefault(str(variant), {"variant": str(variant)})
            for col in ["gate_mean", "gate_std", "gate_range", "fraction_gate_below_0.5", "fraction_gate_above_0.9"]:
                if col in group.columns:
                    entry[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())

    if cond_path.is_file():
        df = pd.read_csv(cond_path)
        for variant, group in df.groupby("variant"):
            entry = summary.setdefault(str(variant), {"variant": str(variant)})
            for condition_name, positive_value, negative_value, key_prefix in [
                ("correct", 0, 1, "error_vs_correct"),
                ("visual_concept_conflict", 1, 0, "conflict_vs_nonconflict"),
                ("low_high_sign_agreement", 0, 1, "disagree_vs_agree"),
            ]:
                subset = group[group["condition_name"] == condition_name].copy()
                if subset.empty or "gate_mean" not in subset.columns:
                    continue
                pos = subset[subset["condition_value"] == positive_value]["gate_mean"]
                neg = subset[subset["condition_value"] == negative_value]["gate_mean"]
                if pos.empty or neg.empty:
                    continue
                entry[f"{key_prefix}_delta"] = float(pos.mean() - neg.mean())

    return summary


def evaluate_variant(
    summary_row: dict[str, object],
    delta_row: dict[str, object] | None,
    diagnostics_row: dict[str, object] | None,
) -> tuple[float, bool, str]:
    score = -1e6
    recommend_step48 = False
    reason = "variant incomplete"
    if summary_row.get("status") not in {"ok", "partial"}:
        return score, recommend_step48, reason

    auc_delta = delta_row.get("test_auc_delta") if delta_row else None
    pr_auc_delta = delta_row.get("pr_auc_delta") if delta_row else None
    acc_delta = delta_row.get("test_acc_delta") if delta_row else None
    f1_delta = delta_row.get("test_f1_delta") if delta_row else None
    bal_delta = delta_row.get("balanced_acc_delta") if delta_row else None
    sens_delta = delta_row.get("sensitivity_delta") if delta_row else None
    spec_delta = delta_row.get("specificity_delta") if delta_row else None

    gate_std = diagnostics_row.get("gate_std_mean") if diagnostics_row else None
    gate_range = diagnostics_row.get("gate_range_mean") if diagnostics_row else None
    conflict_delta = diagnostics_row.get("conflict_vs_nonconflict_delta") if diagnostics_row else None
    error_delta = diagnostics_row.get("error_vs_correct_delta") if diagnostics_row else None

    gate_adaptive = False
    if gate_std is not None and gate_range is not None:
        gate_adaptive = bool(gate_std > 1e-3 and gate_range > 1e-2)
    if conflict_delta is not None and conflict_delta < -0.01:
        gate_adaptive = True
    if error_delta is not None and error_delta < -0.01:
        gate_adaptive = True

    performance_noninferior = False
    if auc_delta is not None and pr_auc_delta is not None:
        performance_noninferior = auc_delta >= -0.002 and pr_auc_delta >= -0.002

    performance_improved = False
    if acc_delta is not None and f1_delta is not None and bal_delta is not None:
        performance_improved = acc_delta > 0 and f1_delta > 0 and bal_delta > 0

    tradeoff_better = False
    if f1_delta is not None and bal_delta is not None:
        tradeoff_better = f1_delta > 0.005 and bal_delta > 0.005
        if sens_delta is not None and spec_delta is not None:
            tradeoff_better = tradeoff_better and (sens_delta > 0 or spec_delta > 0)

    flat_but_interesting = False
    if auc_delta is not None and pr_auc_delta is not None:
        flat_but_interesting = abs(auc_delta) <= 0.003 and abs(pr_auc_delta) <= 0.003 and gate_adaptive

    if performance_noninferior and performance_improved:
        recommend_step48 = True
        reason = "AUC/PR-AUC are not below baseline and ACC/F1/Balanced ACC improve."
    elif tradeoff_better:
        recommend_step48 = True
        reason = "AUC trade-off is acceptable and F1/Balanced ACC improve noticeably."
    elif flat_but_interesting:
        recommend_step48 = True
        reason = "Performance is near baseline and gate diagnostics show sample-adaptive behavior."
    elif diagnostics_row and gate_adaptive:
        reason = "Gate shows adaptive behavior, but performance benefit is not yet convincing."
    else:
        reason = "Performance is below baseline or gate still looks near-scalar."

    if summary_row.get("has_traceback") or summary_row.get("has_nan_or_inf"):
        recommend_step48 = False
        reason = "Training/log integrity issue detected."

    score = float(summary_row.get("test_auc_mean") or -1)
    if bal_delta is not None:
        score += 10.0 * float(bal_delta)
    if f1_delta is not None:
        score += 10.0 * float(f1_delta)
    if recommend_step48:
        score += 100.0
    if gate_adaptive:
        score += 5.0
    return score, recommend_step48, reason


def main() -> int:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    baseline_root = resolve_path(args.baseline_summary_dir)
    stage46_summary_path = resolve_path(args.stage46_summary)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants) or list(DEFAULT_VARIANTS)
    collected_list = [collect_variant_outputs(find_run_dir(results_root, variant, args.seed), variant, results_root, args.seed) for variant in variants]
    summary_rows = [summarize_variant(collected) for collected in collected_list]
    fold_rows: list[dict[str, object]] = []
    for collected in collected_list:
        fold_rows.extend(build_fold_rows(collected))

    baseline_metrics, baseline_warning, baseline_name = load_baseline_metrics(baseline_root)
    delta_rows = build_delta_rows(summary_rows, baseline_metrics)
    delta_map = {row["variant"]: row for row in delta_rows}
    diagnostics_map = load_diagnostics_summary(results_root)
    stage46_data = load_stage46_smoke(stage46_summary_path)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    delta_df = pd.DataFrame(delta_rows)
    stability_df = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "status": row["status"],
                "checkpoint_exists": row["checkpoint_exists"],
                "checkpoint_count": row["checkpoint_count"],
                "missing_checkpoint_folds": row["missing_checkpoint_folds"],
                "missing_folds": row["missing_folds"],
                "log_exists": row["log_exists"],
                "has_traceback": row["has_traceback"],
                "has_nan_or_inf": row["has_nan_or_inf"],
                "run_dir": row["run_dir"],
                "fold_summary_path": row["fold_summary_path"],
            }
            for row in summary_rows
        ]
    )

    best_variant = None
    best_reason = "No valid variant available."
    recommend_step48 = False
    best_score = -1e9
    variant_decisions: list[dict[str, object]] = []
    for row in summary_rows:
        delta_row = delta_map.get(row["variant"])
        diagnostics_row = diagnostics_map.get(row["variant"])
        score, variant_step48, reason = evaluate_variant(row, delta_row, diagnostics_row)
        variant_decisions.append(
            {
                "variant": row["variant"],
                "score": score,
                "recommend_step48": variant_step48,
                "reason": reason,
            }
        )
        if row["status"] in {"ok", "partial"} and score > best_score:
            best_score = score
            best_variant = row["variant"]
            best_reason = reason
            recommend_step48 = variant_step48

    successful_variants = [row["variant"] for row in summary_rows if row["status"] in {"ok", "partial"}]
    failed_variants = [row["variant"] for row in summary_rows if row["status"] not in {"ok", "partial"}]

    summary_csv = output_dir / "stage47_prarc_5fold_summary.csv"
    fold_csv = output_dir / "stage47_prarc_fold_metrics.csv"
    delta_csv = output_dir / "stage47_prarc_vs_baseline.csv"
    stability_csv = output_dir / "stage47_prarc_stability_check.csv"
    report_md = output_dir / "stage47_prarc_gate_report.md"
    manifest_json = output_dir / "stage47_manifest.json"

    summary_df.to_csv(summary_csv, index=False)
    fold_df.to_csv(fold_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    stability_df.to_csv(stability_csv, index=False)

    brief_summary_df = summary_df[
        [
            col
            for col in [
                "variant",
                "status",
                "test_auc_mean",
                "test_auc_std",
                "test_acc_mean",
                "test_acc_std",
                "test_f1_mean",
                "test_f1_std",
                "balanced_acc_mean",
                "balanced_acc_std",
                "sensitivity_mean",
                "specificity_mean",
                "pr_auc_mean",
                "pr_auc_std",
            ]
            if col in summary_df.columns
        ]
    ].copy()
    delta_brief_df = delta_df[
        [
            col
            for col in [
                "variant",
                "test_auc_delta",
                "test_acc_delta",
                "test_f1_delta",
                "balanced_acc_delta",
                "sensitivity_delta",
                "specificity_delta",
                "pr_auc_delta",
            ]
            if col in delta_df.columns
        ]
    ].copy()

    diagnostics_lines = []
    diagnostics_dir = results_root / "stage47_prarc_gate_diagnostics"
    diagnostics_manifest_exists = (diagnostics_dir / "stage47_prarc_gate_diagnostics_manifest.json").is_file()
    if diagnostics_map:
        diagnostics_table = pd.DataFrame(
            [
                {
                    "variant": variant,
                    "gate_mean": data.get("gate_mean_mean"),
                    "gate_std": data.get("gate_std_mean"),
                    "gate_range": data.get("gate_range_mean"),
                    "error_minus_correct": data.get("error_vs_correct_delta"),
                    "conflict_minus_nonconflict": data.get("conflict_vs_nonconflict_delta"),
                    "disagree_minus_agree": data.get("disagree_vs_agree_delta"),
                    "sample_adaptive_flag": data.get("sample_adaptive_flag"),
                }
                for variant, data in diagnostics_map.items()
            ]
        )
        diagnostics_lines.append(make_markdown_table(diagnostics_table))
    elif diagnostics_manifest_exists:
        diagnostics_lines.append("_Gate diagnostics outputs exist, but no valid slide-level diagnostic rows are available yet._")
    else:
        diagnostics_lines.append("_Gate diagnostics not found yet. Run `probe_stage47_prarc_gate_diagnostics.py` and rebuild this summary._")

    stage46_lines = []
    if stage46_data.get("available"):
        for key in [
            "test_auc",
            "test_acc",
            "test_f1",
            "balanced_acc",
            "sensitivity",
            "specificity",
            "pr_auc",
            "gate_mean",
            "gate_min",
            "gate_max",
            "gate_nonconstant",
            "visual_residual_adjusted",
        ]:
            if key in stage46_data:
                stage46_lines.append(f"- {key}: `{stage46_data[key]}`")
    else:
        stage46_lines.append(f"- Stage46 smoke summary missing: `{stage46_data['path']}`")

    report_lines = [
        "# Step47 PRARC Gate 5-Fold Report",
        "",
        "## Step47 Purpose",
        "- Run formal 5-fold validation for Step46 PRARC gate variants.",
        "- Compare PRARC variants against the current baseline `RCE-v4-CSG-a01-rq16 / DEG skeleton`.",
        "- Check whether PRARC behaves as a real sample-adaptive gate or collapses toward a near-scalar gate.",
        "",
        "## Step46 Smoke Inheritance",
        *stage46_lines,
        "",
        "## Variants Run",
        f"- requested_variants: `{', '.join(variants)}`",
        f"- successful_variants: `{', '.join(successful_variants) if successful_variants else 'none'}`",
        f"- failed_or_missing_variants: `{', '.join(failed_variants) if failed_variants else 'none'}`",
        "",
        "## 5-Fold Metrics",
        make_markdown_table(brief_summary_df),
        "",
        "## Baseline Comparison",
    ]
    if baseline_metrics is not None:
        report_lines.append(f"- baseline_model: `{baseline_name}`")
        report_lines.append(make_markdown_table(delta_brief_df))
    else:
        report_lines.append(f"- baseline_warning: `{baseline_warning}`")

    report_lines.extend(
        [
            "",
            "## Stability Check",
            make_markdown_table(stability_df),
            "",
            "## Gate Diagnostics Summary",
            *diagnostics_lines,
            "",
            "## Recommendation",
            f"- recommended_variant: `{best_variant or 'none'}`",
            f"- recommend_enter_step48: `{recommend_step48}`",
            f"- recommendation_reason: {best_reason}",
        ]
    )

    if best_variant:
        delta_row = delta_map.get(best_variant, {})
        report_lines.extend(
            [
                f"- best_variant_auc: `{format_metric(summary_df.loc[summary_df['variant'] == best_variant, 'test_auc_mean'].iloc[0] if not summary_df.empty and (summary_df['variant'] == best_variant).any() else None)}`",
                f"- best_variant_pr_auc: `{format_metric(summary_df.loc[summary_df['variant'] == best_variant, 'pr_auc_mean'].iloc[0] if not summary_df.empty and (summary_df['variant'] == best_variant).any() else None)}`",
                f"- best_variant_auc_delta: `{format_metric(delta_row.get('test_auc_delta'))}`",
                f"- best_variant_f1_delta: `{format_metric(delta_row.get('test_f1_delta'))}`",
                f"- best_variant_balanced_acc_delta: `{format_metric(delta_row.get('balanced_acc_delta'))}`",
            ]
        )

    if diagnostics_map and best_variant in diagnostics_map:
        diag = diagnostics_map[best_variant]
        report_lines.extend(
            [
                f"- best_variant_gate_std: `{format_metric(diag.get('gate_std_mean'))}`",
                f"- best_variant_gate_range: `{format_metric(diag.get('gate_range_mean'))}`",
                f"- best_variant_error_minus_correct: `{format_metric(diag.get('error_vs_correct_delta'))}`",
                f"- best_variant_conflict_minus_nonconflict: `{format_metric(diag.get('conflict_vs_nonconflict_delta'))}`",
            ]
        )

    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "step": "Step47 PRARC 5-fold + Gate Diagnostics",
        "results_root": str(results_root),
        "baseline_summary_dir": str(baseline_root),
        "stage46_summary": str(stage46_summary_path),
        "output_dir": str(output_dir),
        "requested_variants": variants,
        "successful_variants": successful_variants,
        "failed_or_missing_variants": failed_variants,
        "baseline_available": baseline_metrics is not None,
        "baseline_warning": baseline_warning,
        "baseline_model": baseline_name,
        "diagnostics_available": bool(diagnostics_map),
        "recommended_variant": best_variant,
        "recommend_enter_step48": recommend_step48,
        "recommendation_reason": best_reason,
        "variant_decisions": variant_decisions,
        "outputs": {
            "summary_csv": str(summary_csv),
            "fold_metrics_csv": str(fold_csv),
            "vs_baseline_csv": str(delta_csv),
            "stability_csv": str(stability_csv),
            "report_md": str(report_md),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[Done] Wrote Step47 summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
