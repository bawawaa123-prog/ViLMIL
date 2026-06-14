from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path("results_stage37")
DEFAULT_OUTPUT_DIR = Path("results_stage37/stage37_lh_consistency_summary")
METRICS = [
    "test_auc",
    "test_acc",
    "test_f1",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "pr_auc",
]
VARIANTS = [
    {"variant": "skeleton", "method": "DEG skeleton", "lambda": "off", "margin": "off", "strength": "control"},
    {"variant": "lh_l0001_m0", "method": "LH consistency lambda 0.001 margin 0.00", "lambda": "0.001", "margin": "0.00", "strength": "light"},
    {"variant": "lh_l0005_m0", "method": "LH consistency lambda 0.005 margin 0.00", "lambda": "0.005", "margin": "0.00", "strength": "light"},
    {"variant": "lh_l001_m0", "method": "LH consistency lambda 0.01 margin 0.00", "lambda": "0.01", "margin": "0.00", "strength": "medium"},
    {"variant": "lh_l0005_m002", "method": "LH consistency lambda 0.005 margin 0.02", "lambda": "0.005", "margin": "0.02", "strength": "light"},
    {"variant": "lh_l001_m002", "method": "LH consistency lambda 0.01 margin 0.02", "lambda": "0.01", "margin": "0.02", "strength": "medium"},
    {"variant": "lh_l001_m005", "method": "LH consistency lambda 0.01 margin 0.05", "lambda": "0.01", "margin": "0.05", "strength": "medium"},
    {"variant": "lh_l005_m0", "method": "LH consistency lambda 0.05 margin 0.00", "lambda": "0.05", "margin": "0.00", "strength": "strong"},
    {"variant": "lh_l005_m005", "method": "LH consistency lambda 0.05 margin 0.05", "lambda": "0.05", "margin": "0.05", "strength": "strong"},
]
RUN_DIR_PATTERN = re.compile(r"^lh_consistency_(?P<variant>.+)_5fold_e(?P<epochs>\d+)_s(?P<seed>\d+)$")


def warn(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def resolve_path(raw: str | os.PathLike[str]) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_run_dir(path: Path) -> dict[str, object] | None:
    match = RUN_DIR_PATTERN.match(path.name)
    if not match:
        return None
    return {
        "variant": match.group("variant"),
        "epochs": int(match.group("epochs")),
        "seed": int(match.group("seed")),
        "path": path,
    }


def discover_fold_summary(
    results_dir: Path,
    variant: str,
    max_epochs_filter: str | None,
    seed_filter: str | None,
    warning_log: list[str],
) -> tuple[Path | None, int | None, int | None]:
    candidates: list[dict[str, object]] = []
    for path in results_dir.glob("lh_consistency_*_5fold_e*_s*"):
        if not path.is_dir():
            continue
        parsed = parse_run_dir(path)
        if parsed is None or parsed["variant"] != variant:
            continue
        if max_epochs_filter and str(parsed["epochs"]) != str(max_epochs_filter):
            continue
        if seed_filter and str(parsed["seed"]) != str(seed_filter):
            continue
        candidates.append(parsed)

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda item: (int(item["epochs"]), int(item["seed"]), str(item["path"])), reverse=True)
    if len(candidates) > 1:
        warn(f"Multiple Stage37 result dirs matched variant={variant}; using {candidates[0]['path']}", warning_log)
    chosen = candidates[0]
    return Path(chosen["path"]) / "fold_summary.csv", int(chosen["epochs"]), int(chosen["seed"])


def read_csv(path: Path | None, warning_log: list[str]) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        warn(f"Missing fold_summary.csv: {path}", warning_log)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn(f"Failed reading {path}: {exc}", warning_log)
        return None


def mean_std(df: pd.DataFrame | None, metric: str, label: str, warning_log: list[str]) -> tuple[float, float]:
    if df is None:
        return math.nan, math.nan
    if metric not in df.columns:
        warn(f"{label}: missing metric column `{metric}`", warning_log)
        return math.nan, math.nan
    values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        warn(f"{label}: no numeric values for `{metric}`", warning_log)
        return math.nan, math.nan
    return float(np.mean(values)), float(np.std(values, ddof=0))


def formatted(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        return f"{float(mean_value):.6f}"
    return f"{float(mean_value):.6f} +/- {float(std_value):.6f}"


def safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def build_summary(
    results_dir: Path,
    max_epochs_filter: str | None,
    seed_filter: str | None,
    warning_log: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lookup: dict[str, dict[str, object]] = {}
    for spec in VARIANTS:
        variant = str(spec["variant"])
        path, epochs, seed = discover_fold_summary(results_dir, variant, max_epochs_filter, seed_filter, warning_log)
        df = read_csv(path, warning_log)
        status = "missing" if df is None else ("empty" if df.empty else "ok")
        row: dict[str, object] = {
            "variant": variant,
            "method": spec["method"],
            "lambda": spec["lambda"],
            "margin": spec["margin"],
            "strength": spec["strength"],
            "matched_epochs": epochs,
            "matched_seed": seed,
            "source_path": "" if path is None else relative(path),
            "status": status,
            "num_rows": 0 if df is None else len(df.index),
        }
        for metric in METRICS:
            mean_value, std_value = mean_std(df, metric, variant, warning_log)
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_formatted"] = formatted(mean_value, std_value)

        sens = safe_float(row["sensitivity_mean"])
        spec_value = safe_float(row["specificity_mean"])
        if sens is None or spec_value is None:
            row["sens_spec_gap"] = math.nan
            row["sens_spec_min"] = math.nan
        else:
            row["sens_spec_gap"] = abs(sens - spec_value)
            row["sens_spec_min"] = min(sens, spec_value)
        rows.append(row)
        lookup[variant] = row

    ordered = ["variant", "method", "lambda", "margin", "strength", "matched_epochs", "matched_seed", "source_path", "status", "num_rows"]
    for metric in METRICS:
        ordered.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_formatted"])
    ordered.extend(["sens_spec_gap", "sens_spec_min"])
    return pd.DataFrame(rows, columns=ordered), lookup


def build_deltas(lookup: dict[str, dict[str, object]]) -> pd.DataFrame:
    skeleton = lookup.get("skeleton")
    rows: list[dict[str, object]] = []
    for spec in VARIANTS:
        variant = str(spec["variant"])
        if variant == "skeleton":
            continue
        current = lookup.get(variant)
        row: dict[str, object] = {
            "comparison": f"{variant} - skeleton",
            "variant": variant,
            "reference_variant": "skeleton",
            "current_status": None if current is None else current.get("status"),
            "reference_status": None if skeleton is None else skeleton.get("status"),
        }
        for metric in METRICS + ["sens_spec_gap", "sens_spec_min"]:
            current_value = None if current is None else current.get(f"{metric}_mean", current.get(metric))
            skeleton_value = None if skeleton is None else skeleton.get(f"{metric}_mean", skeleton.get(metric))
            row[f"{metric}_delta"] = math.nan if current_value is None or skeleton_value is None or pd.isna(current_value) or pd.isna(skeleton_value) else float(current_value) - float(skeleton_value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = summary_df[
        [
            "variant",
            "method",
            "lambda",
            "margin",
            "strength",
            "status",
            "test_auc_mean",
            "test_acc_mean",
            "test_f1_mean",
            "balanced_acc_mean",
            "sensitivity_mean",
            "specificity_mean",
            "pr_auc_mean",
            "sens_spec_gap",
            "sens_spec_min",
        ]
    ].copy()
    for metric in METRICS:
        ranking_df[f"{metric}_rank"] = ranking_df[f"{metric}_mean"].rank(method="min", ascending=False)
    ranking_df["sens_spec_gap_rank"] = ranking_df["sens_spec_gap"].rank(method="min", ascending=True)
    ranking_df["sens_spec_min_rank"] = ranking_df["sens_spec_min"].rank(method="min", ascending=False)
    ranking_df["overall_rank_score"] = ranking_df[
        [
            "test_auc_rank",
            "test_f1_rank",
            "balanced_acc_rank",
            "pr_auc_rank",
            "sens_spec_gap_rank",
            "sens_spec_min_rank",
        ]
    ].sum(axis=1, min_count=1)
    ranking_df["overall_rank"] = ranking_df["overall_rank_score"].rank(method="min", ascending=True)
    return ranking_df.sort_values(
        by=["overall_rank", "test_auc_rank", "balanced_acc_rank", "sens_spec_gap_rank", "variant"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def best_row(summary_df: pd.DataFrame, column: str, ascending: bool = False) -> dict[str, object] | None:
    valid = summary_df.dropna(subset=[column]).copy()
    if valid.empty:
        return None
    valid = valid.sort_values(
        by=[column, "balanced_acc_mean", "test_auc_mean", "sens_spec_min", "variant"],
        ascending=[ascending, False, False, False, True],
    )
    return valid.iloc[0].to_dict()


def best_balance_row(summary_df: pd.DataFrame) -> dict[str, object] | None:
    valid = summary_df.dropna(subset=["sens_spec_gap", "balanced_acc_mean", "test_auc_mean"]).copy()
    if valid.empty:
        return None
    valid = valid.sort_values(
        by=["sens_spec_gap", "balanced_acc_mean", "test_auc_mean", "sens_spec_min", "variant"],
        ascending=[True, False, False, False, True],
    )
    return valid.iloc[0].to_dict()


def collapse_variants(summary_df: pd.DataFrame, metric: str, threshold: float) -> list[str]:
    rows = summary_df[(summary_df["variant"] != "skeleton") & (summary_df[metric].notna())]
    return [str(row["variant"]) for _, row in rows.iterrows() if float(row[metric]) <= threshold]


def clearly_better_than_skeleton(row: pd.Series, skeleton: pd.Series | None) -> bool:
    if skeleton is None:
        return False
    checks = [
        ("test_auc_mean", 0.005),
        ("balanced_acc_mean", 0.005),
    ]
    for metric, threshold in checks:
        current_value = safe_float(row.get(metric))
        skeleton_value = safe_float(skeleton.get(metric))
        if current_value is not None and skeleton_value is not None and current_value >= skeleton_value + threshold:
            return True
    current_gap = safe_float(row.get("sens_spec_gap"))
    skeleton_gap = safe_float(skeleton.get("sens_spec_gap"))
    current_auc = safe_float(row.get("test_auc_mean"))
    skeleton_auc = safe_float(skeleton.get("test_auc_mean"))
    current_bal = safe_float(row.get("balanced_acc_mean"))
    skeleton_bal = safe_float(skeleton.get("balanced_acc_mean"))
    if None not in {current_gap, skeleton_gap, current_auc, skeleton_auc, current_bal, skeleton_bal}:
        if current_gap <= skeleton_gap - 0.05 and current_auc >= skeleton_auc - 0.005 and current_bal >= skeleton_bal - 0.005:
            return True
    return False


def compare_light_vs_strong(summary_df: pd.DataFrame) -> dict[str, object]:
    valid = summary_df[summary_df["variant"] != "skeleton"].dropna(subset=["test_auc_mean", "balanced_acc_mean"]).copy()
    result: dict[str, object] = {"light_variants_loaded": 0, "medium_strong_variants_loaded": 0, "light_more_stable": None}
    light = valid[valid["strength"] == "light"]
    medium_strong = valid[valid["strength"].isin(["medium", "strong"])]
    result["light_variants_loaded"] = int(len(light.index))
    result["medium_strong_variants_loaded"] = int(len(medium_strong.index))
    if light.empty or medium_strong.empty:
        return result
    light_score = float(light["balanced_acc_mean"].mean() + light["test_auc_mean"].mean() - light["sens_spec_gap"].mean())
    other_score = float(medium_strong["balanced_acc_mean"].mean() + medium_strong["test_auc_mean"].mean() - medium_strong["sens_spec_gap"].mean())
    result["light_score"] = light_score
    result["medium_strong_score"] = other_score
    result["light_more_stable"] = light_score >= other_score
    return result


def build_recommendations(summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> dict[str, object]:
    del delta_df
    lookup = {str(row["variant"]): row for _, row in summary_df.iterrows()}
    skeleton = lookup.get("skeleton")
    best_auc = best_row(summary_df, "test_auc_mean", ascending=False)
    best_bal = best_row(summary_df, "balanced_acc_mean", ascending=False)
    best_balance = best_balance_row(summary_df)
    specificity_collapse = collapse_variants(summary_df, "specificity_mean", 0.05)
    sensitivity_collapse = collapse_variants(summary_df, "sensitivity_mean", 0.05)
    valid_variants = summary_df[(summary_df["variant"] != "skeleton") & (summary_df["status"] == "ok")]
    effective_variants = [
        str(row["variant"])
        for _, row in valid_variants.iterrows()
        if clearly_better_than_skeleton(row, skeleton)
    ]
    all_variants_below = False
    if skeleton is not None and not valid_variants.empty:
        skeleton_auc = safe_float(skeleton.get("test_auc_mean"))
        skeleton_bal = safe_float(skeleton.get("balanced_acc_mean"))
        skeleton_f1 = safe_float(skeleton.get("test_f1_mean"))
        if None not in {skeleton_auc, skeleton_bal, skeleton_f1}:
            all_variants_below = True
            for _, row in valid_variants.iterrows():
                auc = safe_float(row.get("test_auc_mean"))
                bal = safe_float(row.get("balanced_acc_mean"))
                f1 = safe_float(row.get("test_f1_mean"))
                if None in {auc, bal, f1}:
                    all_variants_below = False
                    break
                if auc >= skeleton_auc or bal >= skeleton_bal or f1 >= skeleton_f1:
                    all_variants_below = False
                    break

    light_vs_strong = compare_light_vs_strong(summary_df)
    return {
        "best_test_auc_variant": None if best_auc is None else str(best_auc["variant"]),
        "best_balanced_acc_variant": None if best_bal is None else str(best_bal["variant"]),
        "best_sens_spec_balance_variant": None if best_balance is None else str(best_balance["variant"]),
        "variants_clearly_better_than_skeleton": effective_variants,
        "any_variant_clearly_better_than_skeleton": bool(effective_variants),
        "light_lambda_stability": light_vs_strong,
        "specificity_collapse_variants": specificity_collapse,
        "sensitivity_collapse_variants": sensitivity_collapse,
        "all_variants_below_skeleton": all_variants_below,
        "recommend_step38_evidence_reexport_failure_analysis": bool(effective_variants),
        "recommend_negative_ablation": all_variants_below,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    text_df = df.fillna("NA").astype(str)
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in text_df.columns) + " |" for _, row in text_df.iterrows()]
    return "\n".join([header, sep] + rows)


def build_report(
    results_dir: Path,
    output_dir: Path,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    recommendations: dict[str, object],
    warning_log: list[str],
) -> str:
    lookup = {str(row["variant"]): row for _, row in summary_df.iterrows()}
    best_auc = recommendations["best_test_auc_variant"]
    best_bal = recommendations["best_balanced_acc_variant"]
    best_balance = recommendations["best_sens_spec_balance_variant"]
    light_stability = recommendations["light_lambda_stability"]

    lines = [
        "# Stage37 Low-High Consistency Summary",
        "",
        "Step37 compares the DEG skeleton against low-high evidence consistency loss variants using existing `fold_summary.csv` files.",
        "",
        "## Inputs",
        "",
        f"- Results directory: `{relative(results_dir)}`",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"- `{row['variant']}` -> `{row['source_path'] or 'missing'}`")

    lines.extend(["", "## Key Answers", ""])
    if best_auc is None:
        lines.append("1. Best `test_auc`: N/A")
    else:
        lines.append(f"1. Best `test_auc`: `{best_auc}` with `{lookup[best_auc]['test_auc_formatted']}`")
    if best_bal is None:
        lines.append("2. Best `balanced_acc`: N/A")
    else:
        lines.append(f"2. Best `balanced_acc`: `{best_bal}` with `{lookup[best_bal]['balanced_acc_formatted']}`")
    if best_balance is None:
        lines.append("3. Most balanced `sensitivity/specificity`: N/A")
    else:
        row = lookup[best_balance]
        lines.append(
            "3. Most balanced `sensitivity/specificity`: "
            f"`{best_balance}` with gap `{float(row['sens_spec_gap']):.6f}`, "
            f"`sensitivity`={row['sensitivity_formatted']}, `specificity`={row['specificity_formatted']}"
        )

    effective = recommendations["variants_clearly_better_than_skeleton"]
    if effective:
        lines.append("4. Variants clearly better than skeleton: " + ", ".join(f"`{item}`" for item in effective))
    else:
        lines.append("4. No loaded variant is clearly better than skeleton by the current thresholds.")

    if light_stability.get("light_more_stable") is None:
        lines.append("5. Light lambda stability (`0.001/0.005` vs `0.01/0.05`): insufficient loaded data.")
    elif light_stability["light_more_stable"]:
        lines.append("5. Light lambda variants look more stable than medium/strong variants from loaded results.")
    else:
        lines.append("5. Light lambda variants do not look more stable than medium/strong variants from loaded results.")

    spec_collapse = recommendations["specificity_collapse_variants"]
    sens_collapse = recommendations["sensitivity_collapse_variants"]
    if spec_collapse or sens_collapse:
        lines.append(
            "6. Collapse detected: "
            f"specificity={spec_collapse or 'none'}, sensitivity={sens_collapse or 'none'}."
        )
    else:
        lines.append("6. No specificity/sensitivity collapse detected from loaded runs.")

    if recommendations["recommend_negative_ablation"]:
        lines.append("7. All loaded consistency variants are below skeleton; keep Step36/37 as a negative diagnostic ablation.")
    else:
        lines.append("7. The loaded data do not support marking all consistency variants as negative yet.")

    if recommendations["recommend_step38_evidence_reexport_failure_analysis"]:
        lines.append("8. Recommendation: Step38 should re-export evidence and run failure analysis for the best effective variant.")
    else:
        lines.append("8. Recommendation: do not start Step38 from these loaded results unless more complete runs change the ranking.")

    lines.extend(["", "## Mean +/- Std Summary", ""])
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['variant']}`: `test_auc`={row['test_auc_formatted']}, "
            f"`test_acc`={row['test_acc_formatted']}, `test_f1`={row['test_f1_formatted']}, "
            f"`balanced_acc`={row['balanced_acc_formatted']}, `sensitivity`={row['sensitivity_formatted']}, "
            f"`specificity`={row['specificity_formatted']}, `pr_auc`={row['pr_auc_formatted']}"
        )

    lines.extend(["", "## Deltas Vs Skeleton", ""])
    for _, row in delta_df.iterrows():
        parts = []
        for metric in METRICS + ["sens_spec_gap", "sens_spec_min"]:
            value = row.get(f"{metric}_delta", math.nan)
            if not pd.isna(value):
                parts.append(f"`{metric}`={float(value):+.6f}")
        lines.append(f"- `{row['comparison']}`: " + (", ".join(parts) if parts else "N/A"))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{relative(output_dir / 'stage37_lh_consistency_summary.csv')}`",
            f"- `{relative(output_dir / 'stage37_lh_consistency_metric_deltas.csv')}`",
            f"- `{relative(output_dir / 'stage37_lh_consistency_rankings.csv')}`",
            f"- `{relative(output_dir / 'stage37_lh_consistency_report.md')}`",
            f"- `{relative(output_dir / 'stage37_recommendations.json')}`",
        ]
    )
    if warning_log:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {message}" for message in warning_log])
    lines.extend(
        [
            "",
            "## Compact Tables",
            "",
            "### Summary",
            "",
            markdown_table(
                summary_df[
                    [
                        "variant",
                        "status",
                        "matched_epochs",
                        "matched_seed",
                        "test_auc_formatted",
                        "test_f1_formatted",
                        "balanced_acc_formatted",
                        "sensitivity_formatted",
                        "specificity_formatted",
                    ]
                ]
            ),
            "",
            "### Rankings",
            "",
            markdown_table(
                rankings_df[
                    [
                        "variant",
                        "overall_rank",
                        "test_auc_rank",
                        "balanced_acc_rank",
                        "sens_spec_gap_rank",
                        "sens_spec_min_rank",
                    ]
                ]
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    results_dir = resolve_path(os.environ.get("RESULTS_DIR", str(DEFAULT_RESULTS_DIR)))
    output_dir = resolve_path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    max_epochs_filter = os.environ.get("MAX_EPOCHS_FILTER") or None
    seed_filter = os.environ.get("SEED_FILTER") or None
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []
    summary_df, lookup = build_summary(results_dir, max_epochs_filter, seed_filter, warning_log)
    delta_df = build_deltas(lookup)
    rankings_df = build_rankings(summary_df)
    recommendations = build_recommendations(summary_df, delta_df)

    summary_path = output_dir / "stage37_lh_consistency_summary.csv"
    delta_path = output_dir / "stage37_lh_consistency_metric_deltas.csv"
    ranking_path = output_dir / "stage37_lh_consistency_rankings.csv"
    report_path = output_dir / "stage37_lh_consistency_report.md"
    recommendations_path = output_dir / "stage37_recommendations.json"

    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    rankings_df.to_csv(ranking_path, index=False)
    report_path.write_text(
        build_report(results_dir, output_dir, summary_df, delta_df, rankings_df, recommendations, warning_log),
        encoding="utf-8",
    )
    recommendations_path.write_text(json.dumps(recommendations, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] Wrote summary CSV: {summary_path}")
    print(f"[Done] Wrote delta CSV: {delta_path}")
    print(f"[Done] Wrote rankings CSV: {ranking_path}")
    print(f"[Done] Wrote report: {report_path}")
    print(f"[Done] Wrote recommendations: {recommendations_path}")


if __name__ == "__main__":
    main()
