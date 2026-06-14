from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path("results_stage35")
DEFAULT_OUTPUT_DIR = Path("results_stage35/stage35_visual_gate_summary")
DEFAULT_MAX_EPOCHS_FILTER = "20"
DEFAULT_SEED_FILTER = "1"
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
    {
        "variant": "skeleton",
        "method": "DEG skeleton",
        "visual_residual": "on",
        "visual_gate": "off",
        "gate_init": "off",
    },
    {
        "variant": "no_visual_residual",
        "method": "No visual residual",
        "visual_residual": "off",
        "visual_gate": "off",
        "gate_init": "off",
    },
    {
        "variant": "gate0",
        "method": "Visual gate init 0.00",
        "visual_residual": "on",
        "visual_gate": "on",
        "gate_init": "0.00",
    },
    {
        "variant": "gate001",
        "method": "Visual gate init 0.01",
        "visual_residual": "on",
        "visual_gate": "on",
        "gate_init": "0.01",
    },
    {
        "variant": "gate005",
        "method": "Visual gate init 0.05",
        "visual_residual": "on",
        "visual_gate": "on",
        "gate_init": "0.05",
    },
    {
        "variant": "gate05",
        "method": "Visual gate init 0.50",
        "visual_residual": "on",
        "visual_gate": "on",
        "gate_init": "0.50",
    },
    {
        "variant": "gate1",
        "method": "Visual gate init 1.00",
        "visual_residual": "on",
        "visual_gate": "on",
        "gate_init": "1.00",
    },
]
RUN_DIR_PATTERN = re.compile(r"^visual_gate_(?P<variant>.+)_5fold_e(?P<epochs>\d+)_s(?P<seed>\d+)$")


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def resolve_path(root: Path, raw_value: str | os.PathLike[str]) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return root / path


def relative_path_str(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_read_csv(path: Path, warning_log: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warn_message(f"Missing input CSV: {path}", warning_log)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn_message(f"Failed to read CSV {path}: {exc}", warning_log)
        return None


def format_mean_std(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        return f"{float(mean_value):.6f}"
    return f"{float(mean_value):.6f} ± {float(std_value):.6f}"


def parse_run_dir_name(path: Path) -> dict[str, object] | None:
    match = RUN_DIR_PATTERN.match(path.name)
    if not match:
        return None
    return {
        "variant": match.group("variant"),
        "epochs": int(match.group("epochs")),
        "seed": int(match.group("seed")),
        "path": path,
    }


def discover_fold_summary_path(
    results_dir: Path,
    variant: str,
    warning_log: list[str],
    max_epochs_filter: str | None,
    seed_filter: str | None,
) -> tuple[Path | None, str, int | None, int | None]:
    candidates: list[dict[str, object]] = []
    for path in results_dir.glob(f"visual_gate_{variant}_5fold_e*_s*"):
        if not path.is_dir():
            continue
        parsed = parse_run_dir_name(path)
        if parsed is None or parsed["variant"] != variant:
            continue
        if max_epochs_filter is not None and str(parsed["epochs"]) != str(max_epochs_filter):
            continue
        if seed_filter is not None and str(parsed["seed"]) != str(seed_filter):
            continue
        candidates.append(parsed)

    if not candidates:
        return None, "missing", None, None

    candidates.sort(key=lambda item: (int(item["epochs"]), int(item["seed"]), str(item["path"])), reverse=True)
    if len(candidates) > 1:
        warn_message(
            f"Multiple Stage35 result directories matched variant={variant}; using {candidates[0]['path']}",
            warning_log,
        )

    chosen = candidates[0]
    fold_summary_path = Path(chosen["path"]) / "fold_summary.csv"
    return fold_summary_path, "ok", int(chosen["epochs"]), int(chosen["seed"])


def summarize_fold_summary(
    df: pd.DataFrame | None,
    warning_log: list[str],
    label: str,
) -> tuple[dict[str, float], str, int]:
    summary: dict[str, float] = {}
    if df is None:
        for metric in METRICS:
            summary[f"{metric}_mean"] = math.nan
            summary[f"{metric}_std"] = math.nan
        return summary, "missing", 0

    status = "ok"
    num_rows = len(df.index)
    if df.empty:
        warn_message(f"{label}: fold summary is empty.", warning_log)
        status = "empty"

    for metric in METRICS:
        if metric not in df.columns:
            warn_message(f"{label}: missing metric column `{metric}`.", warning_log)
            summary[f"{metric}_mean"] = math.nan
            summary[f"{metric}_std"] = math.nan
            if status == "ok":
                status = "partial"
            continue

        values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            warn_message(f"{label}: no valid numeric values for `{metric}`.", warning_log)
            summary[f"{metric}_mean"] = math.nan
            summary[f"{metric}_std"] = math.nan
            if status == "ok":
                status = "partial"
            continue

        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=0))

    return summary, status, num_rows


def safe_mean_value(row: dict[str, object] | None, metric: str) -> float:
    if row is None:
        return math.nan
    value = row.get(f"{metric}_mean", math.nan)
    return math.nan if pd.isna(value) else float(value)


def build_summary_table(
    root: Path,
    results_dir: Path,
    warning_log: list[str],
    max_epochs_filter: str | None,
    seed_filter: str | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lookup: dict[str, dict[str, object]] = {}

    for experiment in VARIANTS:
        source_path, discovery_status, matched_epochs, matched_seed = discover_fold_summary_path(
            results_dir,
            str(experiment["variant"]),
            warning_log,
            max_epochs_filter,
            seed_filter,
        )
        df = safe_read_csv(source_path, warning_log) if source_path is not None else None
        summary, status, num_rows = summarize_fold_summary(df, warning_log, str(experiment["method"]))
        if discovery_status == "missing":
            status = "missing"

        row: dict[str, object] = {
            "variant": experiment["variant"],
            "method": experiment["method"],
            "visual_residual": experiment["visual_residual"],
            "visual_gate": experiment["visual_gate"],
            "gate_init": experiment["gate_init"],
            "matched_epochs": matched_epochs,
            "matched_seed": matched_seed,
            "source_path": "" if source_path is None else relative_path_str(root, source_path),
            "status": status,
            "num_rows": num_rows,
        }
        for metric in METRICS:
            mean_value = summary[f"{metric}_mean"]
            std_value = summary[f"{metric}_std"]
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_formatted"] = format_mean_std(mean_value, std_value)

        sensitivity_mean = safe_mean_value(row, "sensitivity")
        specificity_mean = safe_mean_value(row, "specificity")
        if pd.isna(sensitivity_mean) or pd.isna(specificity_mean):
            row["sens_spec_gap"] = math.nan
            row["sens_spec_min"] = math.nan
        else:
            row["sens_spec_gap"] = abs(sensitivity_mean - specificity_mean)
            row["sens_spec_min"] = min(sensitivity_mean, specificity_mean)

        rows.append(row)
        lookup[str(experiment["variant"])] = row

    ordered_columns = [
        "variant",
        "method",
        "visual_residual",
        "visual_gate",
        "gate_init",
        "matched_epochs",
        "matched_seed",
        "source_path",
        "status",
        "num_rows",
    ]
    for metric in METRICS:
        ordered_columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_formatted"])
    ordered_columns.extend(["sens_spec_gap", "sens_spec_min"])
    return pd.DataFrame(rows, columns=ordered_columns), lookup


def build_delta_table(summary_lookup: dict[str, dict[str, object]], warning_log: list[str]) -> pd.DataFrame:
    skeleton_row = summary_lookup.get("skeleton")
    rows: list[dict[str, object]] = []
    for experiment in VARIANTS:
        variant = str(experiment["variant"])
        if variant == "skeleton":
            continue
        current_row = summary_lookup.get(variant)
        row: dict[str, object] = {
            "comparison": f"{variant} - skeleton",
            "variant": variant,
            "reference_variant": "skeleton",
            "current_status": None if current_row is None else current_row.get("status"),
            "reference_status": None if skeleton_row is None else skeleton_row.get("status"),
        }
        if current_row is None or skeleton_row is None:
            warn_message(f"Missing summary row for delta computation: {variant}", warning_log)
            for metric in METRICS + ["sens_spec_gap", "sens_spec_min"]:
                row[f"{metric}_delta"] = math.nan
            rows.append(row)
            continue

        for metric in METRICS + ["sens_spec_gap", "sens_spec_min"]:
            current_value = current_row.get(f"{metric}_mean", current_row.get(metric, math.nan))
            reference_value = skeleton_row.get(f"{metric}_mean", skeleton_row.get(metric, math.nan))
            if pd.isna(current_value) or pd.isna(reference_value):
                row[f"{metric}_delta"] = math.nan
            else:
                row[f"{metric}_delta"] = float(current_value) - float(reference_value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_rankings_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = summary_df[
        [
            "variant",
            "method",
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
            "sensitivity_rank",
            "specificity_rank",
            "pr_auc_rank",
            "sens_spec_gap_rank",
        ]
    ].sum(axis=1, min_count=1)
    ranking_df["overall_rank"] = ranking_df["overall_rank_score"].rank(method="min", ascending=True)
    ranking_df = ranking_df.sort_values(
        by=["overall_rank", "test_auc_rank", "balanced_acc_rank", "sens_spec_gap_rank", "variant"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    return ranking_df


def select_best_variant(summary_df: pd.DataFrame, metric_column: str, ascending: bool = False) -> dict[str, object] | None:
    valid = summary_df.dropna(subset=[metric_column]).copy()
    if valid.empty:
        return None
    valid = valid.sort_values(by=[metric_column, "balanced_acc_mean", "test_auc_mean", "variant"], ascending=[ascending, False, False, True])
    return valid.iloc[0].to_dict()


def select_balance_variant(summary_df: pd.DataFrame) -> dict[str, object] | None:
    valid = summary_df.dropna(subset=["sens_spec_gap", "balanced_acc_mean", "test_auc_mean"]).copy()
    if valid.empty:
        return None
    valid = valid.sort_values(
        by=["sens_spec_gap", "balanced_acc_mean", "test_auc_mean", "sens_spec_min", "variant"],
        ascending=[True, False, False, False, True],
    )
    return valid.iloc[0].to_dict()


def is_clearly_lower(delta_row: dict[str, object] | None) -> bool | None:
    if delta_row is None:
        return None
    thresholds = {
        "test_auc_delta": -0.005,
        "test_f1_delta": -0.01,
        "balanced_acc_delta": -0.01,
    }
    results = []
    for key, threshold in thresholds.items():
        value = delta_row.get(key, math.nan)
        if pd.isna(value):
            return None
        results.append(float(value) <= threshold)
    return all(results)


def gate1_requires_check(delta_row: dict[str, object] | None) -> bool | None:
    if delta_row is None:
        return None
    thresholds = {
        "test_auc_delta": 0.005,
        "test_f1_delta": 0.01,
        "balanced_acc_delta": 0.01,
    }
    for key, threshold in thresholds.items():
        value = delta_row.get(key, math.nan)
        if pd.isna(value):
            return None
        if abs(float(value)) > threshold:
            return True
    return False


def specificity_collapse_variants(delta_df: pd.DataFrame) -> list[str]:
    collapsed: list[str] = []
    for _, row in delta_df.iterrows():
        variant = str(row["variant"])
        if variant not in {"gate0", "gate001", "gate005"}:
            continue
        specificity_delta = row.get("specificity_delta", math.nan)
        if not pd.isna(specificity_delta) and float(specificity_delta) <= -0.2:
            collapsed.append(variant)
    return collapsed


def maybe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def build_recommendations(
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> dict[str, object]:
    summary_lookup = {str(row["variant"]): row for _, row in summary_df.iterrows()}
    delta_lookup = {str(row["variant"]): row for _, row in delta_df.iterrows()}

    best_auc = select_best_variant(summary_df, "test_auc_mean", ascending=False)
    best_balanced = select_best_variant(summary_df, "balanced_acc_mean", ascending=False)
    best_balance = select_balance_variant(summary_df)

    no_visual_delta = delta_lookup.get("no_visual_residual")
    gate1_delta = delta_lookup.get("gate1")
    collapse_variants = specificity_collapse_variants(delta_df)

    gate_variants = summary_df[summary_df["variant"].isin(["gate0", "gate001", "gate005", "gate05", "gate1"])].copy()
    valid_gate_variants = gate_variants.dropna(subset=["test_auc_mean", "balanced_acc_mean"])
    skeleton_row = summary_lookup.get("skeleton")

    recommend_step36 = False
    recommend_consistency_loss = False
    if skeleton_row is not None and not valid_gate_variants.empty:
        skeleton_auc = maybe_float(skeleton_row.get("test_auc_mean"))
        skeleton_bal = maybe_float(skeleton_row.get("balanced_acc_mean"))
        skeleton_gap = maybe_float(skeleton_row.get("sens_spec_gap"))
        for _, row in valid_gate_variants.iterrows():
            auc = maybe_float(row.get("test_auc_mean"))
            bal = maybe_float(row.get("balanced_acc_mean"))
            gap = maybe_float(row.get("sens_spec_gap"))
            if auc is None or bal is None or skeleton_auc is None or skeleton_bal is None:
                continue
            if auc > skeleton_auc or bal > skeleton_bal:
                recommend_step36 = True
                break
            if gap is not None and skeleton_gap is not None and gap < skeleton_gap and auc >= skeleton_auc - 0.01 and bal >= skeleton_bal - 0.01:
                recommend_step36 = True
                break
        if not recommend_step36:
            all_worse = True
            for _, row in valid_gate_variants.iterrows():
                auc = maybe_float(row.get("test_auc_mean"))
                bal = maybe_float(row.get("balanced_acc_mean"))
                f1 = maybe_float(row.get("test_f1_mean"))
                skeleton_f1 = maybe_float(skeleton_row.get("test_f1_mean"))
                if None in {auc, bal, f1, skeleton_auc, skeleton_bal, skeleton_f1}:
                    all_worse = False
                    break
                if auc >= skeleton_auc or bal >= skeleton_bal or f1 >= skeleton_f1:
                    all_worse = False
                    break
            recommend_consistency_loss = all_worse

    notes: list[str] = []
    if best_auc is not None:
        notes.append(
            f"Best test_auc variant: {best_auc['variant']} ({float(best_auc['test_auc_mean']):.6f})."
        )
    if best_balanced is not None:
        notes.append(
            f"Best balanced_acc variant: {best_balanced['variant']} ({float(best_balanced['balanced_acc_mean']):.6f})."
        )
    if best_balance is not None:
        notes.append(
            f"Most balanced sensitivity/specificity variant: {best_balance['variant']} "
            f"(gap={float(best_balance['sens_spec_gap']):.6f})."
        )
    if collapse_variants:
        notes.append(
            "Specificity collapse detected for: " + ", ".join(collapse_variants) + "."
        )

    return {
        "best_test_auc_variant": None if best_auc is None else str(best_auc["variant"]),
        "best_balanced_acc_variant": None if best_balanced is None else str(best_balanced["variant"]),
        "best_sens_spec_balance_variant": None if best_balance is None else str(best_balance["variant"]),
        "no_visual_residual_clearly_lower_than_skeleton": is_clearly_lower(no_visual_delta),
        "no_visual_residual_deltas": None if no_visual_delta is None else {
            metric: maybe_float(no_visual_delta.get(metric))
            for metric in ["test_auc_delta", "test_f1_delta", "balanced_acc_delta", "sensitivity_delta", "specificity_delta"]
        },
        "gate1_close_to_skeleton": None if gate1_requires_check(gate1_delta) is None else (not gate1_requires_check(gate1_delta)),
        "gate1_requires_gate_check": gate1_requires_check(gate1_delta),
        "gate1_deltas": None if gate1_delta is None else {
            metric: maybe_float(gate1_delta.get(metric))
            for metric in ["test_auc_delta", "test_f1_delta", "balanced_acc_delta", "sensitivity_delta", "specificity_delta"]
        },
        "specificity_collapse_variants": collapse_variants,
        "recommend_step36_evidence_reexport_and_failure_analysis": recommend_step36,
        "recommend_low_high_evidence_consistency_loss": recommend_consistency_loss,
        "notes": notes,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("NA").astype(str)
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def build_report(
    root: Path,
    results_dir: Path,
    output_dir: Path,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    recommendations: dict[str, object],
    warning_log: list[str],
) -> str:
    summary_lookup = {str(row["variant"]): row for _, row in summary_df.iterrows()}
    delta_lookup = {str(row["variant"]): row for _, row in delta_df.iterrows()}

    best_auc_variant = recommendations.get("best_test_auc_variant")
    best_balanced_variant = recommendations.get("best_balanced_acc_variant")
    best_balance_variant = recommendations.get("best_sens_spec_balance_variant")
    no_visual_lower = recommendations.get("no_visual_residual_clearly_lower_than_skeleton")
    gate1_close = recommendations.get("gate1_close_to_skeleton")
    gate1_requires = recommendations.get("gate1_requires_gate_check")
    collapse_variants = recommendations.get("specificity_collapse_variants", [])

    lines = [
        "# Stage35 Visual Gate Summary",
        "",
        "Step35 reads existing `results_stage35/*/fold_summary.csv` files and compares DEG skeleton against visual-gate ablations.",
        "",
        "## Inputs",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"- `{row['variant']}` -> `{row['source_path'] or 'missing'}`")

    lines.extend(["", "## Key Answers", ""])

    if best_auc_variant is None:
        lines.append("1. `test_auc` best variant: N/A")
    else:
        value = summary_lookup[best_auc_variant]["test_auc_formatted"]
        lines.append(f"1. `test_auc` best variant: `{best_auc_variant}` with `{value}`")

    if best_balanced_variant is None:
        lines.append("2. `balanced_acc` best variant: N/A")
    else:
        value = summary_lookup[best_balanced_variant]["balanced_acc_formatted"]
        lines.append(f"2. `balanced_acc` best variant: `{best_balanced_variant}` with `{value}`")

    if best_balance_variant is None:
        lines.append("3. Most balanced `sensitivity/specificity`: N/A")
    else:
        row = summary_lookup[best_balance_variant]
        lines.append(
            "3. Most balanced `sensitivity/specificity`: "
            f"`{best_balance_variant}` with gap `{float(row['sens_spec_gap']):.6f}`, "
            f"`sensitivity`={row['sensitivity_formatted']}, `specificity`={row['specificity_formatted']}"
        )

    if no_visual_lower is None:
        lines.append("4. `no_visual_residual` vs `skeleton`: insufficient data")
    else:
        lines.append(
            "4. `no_visual_residual` vs `skeleton`: "
            + ("clearly lower." if no_visual_lower else "not clearly lower.")
        )

    if gate1_close is None:
        lines.append("5. `gate1` vs `skeleton`: insufficient data")
    else:
        verdict = "close to skeleton." if gate1_close else "not close to skeleton."
        lines.append(f"5. `gate1` vs `skeleton`: {verdict}")
        if gate1_requires:
            lines.append("   Gate implementation should be re-checked because the sanity-check variant deviates too much.")

    if collapse_variants:
        lines.append(
            "6. Specificity collapse among `gate0/gate001/gate005`: "
            + ", ".join(f"`{variant}`" for variant in collapse_variants)
        )
    else:
        lines.append("6. Specificity collapse among `gate0/gate001/gate005`: not detected from the loaded runs.")

    if recommendations.get("recommend_step36_evidence_reexport_and_failure_analysis"):
        lines.append("7. Recommendation: proceed to Step36 on the best gate variant for evidence re-export + failure analysis.")
    else:
        lines.append("7. Recommendation: do not promote Step36 yet from the currently loaded gate results.")

    if recommendations.get("recommend_low_high_evidence_consistency_loss"):
        lines.append("8. All loaded gate variants remain below skeleton; prefer Low-High Evidence Consistency Loss next.")
    else:
        lines.append("8. Current loaded results do not justify switching to Low-High Evidence Consistency Loss immediately.")

    lines.extend(["", "## Mean ± Std Summary", ""])
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['variant']}`: "
            f"`test_auc`={row['test_auc_formatted']}, "
            f"`test_acc`={row['test_acc_formatted']}, "
            f"`test_f1`={row['test_f1_formatted']}, "
            f"`balanced_acc`={row['balanced_acc_formatted']}, "
            f"`sensitivity`={row['sensitivity_formatted']}, "
            f"`specificity`={row['specificity_formatted']}, "
            f"`pr_auc`={row['pr_auc_formatted']}"
        )

    lines.extend(["", "## Deltas Vs Skeleton", ""])
    for _, row in delta_df.iterrows():
        delta_parts = []
        for metric in METRICS + ["sens_spec_gap"]:
            value = row.get(f"{metric}_delta", math.nan)
            if not pd.isna(value):
                delta_parts.append(f"`{metric}`={float(value):+.6f}")
        lines.append(f"- `{row['comparison']}`: " + (", ".join(delta_parts) if delta_parts else "N/A"))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{relative_path_str(root, output_dir / 'stage35_visual_gate_summary.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage35_visual_gate_metric_deltas.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage35_visual_gate_rankings.csv')}`",
            f"- `{relative_path_str(root, output_dir / 'stage35_visual_gate_report.md')}`",
            f"- `{relative_path_str(root, output_dir / 'stage35_recommendations.json')}`",
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
                        "pr_auc_formatted",
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
                        "sensitivity_rank",
                        "specificity_rank",
                        "sens_spec_gap_rank",
                    ]
                ]
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    root = DEFAULT_ROOT
    results_dir = resolve_path(root, os.environ.get("RESULTS_STAGE35_DIR", str(DEFAULT_RESULTS_DIR)))
    output_dir = resolve_path(root, os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    max_epochs_filter = os.environ.get("MAX_EPOCHS_FILTER", DEFAULT_MAX_EPOCHS_FILTER)
    seed_filter = os.environ.get("SEED_FILTER", DEFAULT_SEED_FILTER)
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []
    summary_df, summary_lookup = build_summary_table(
        root,
        results_dir,
        warning_log,
        max_epochs_filter=max_epochs_filter,
        seed_filter=seed_filter,
    )
    delta_df = build_delta_table(summary_lookup, warning_log)
    rankings_df = build_rankings_table(summary_df)
    recommendations = build_recommendations(summary_df, delta_df)

    summary_path = output_dir / "stage35_visual_gate_summary.csv"
    delta_path = output_dir / "stage35_visual_gate_metric_deltas.csv"
    ranking_path = output_dir / "stage35_visual_gate_rankings.csv"
    report_path = output_dir / "stage35_visual_gate_report.md"
    recommendations_path = output_dir / "stage35_recommendations.json"

    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    rankings_df.to_csv(ranking_path, index=False)
    report_path.write_text(
        build_report(root, results_dir, output_dir, summary_df, delta_df, rankings_df, recommendations, warning_log),
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
