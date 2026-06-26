from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_STAGE57B_DIR = ROOT / "results_stage57B_logit_contribution_audit"
DEFAULT_STAGE57C_DIR = (
    ROOT
    / "results_stage57C_rce_v2_copy_reproduction"
    / "rce_v2_copy_csg_a01_rq16_5fold_e20_s1"
)
DEFAULT_STAGE58C_DIR = (
    ROOT
    / "results_stage58C_residual_constrained_configD_5fold"
    / "rce_v2_rcD_l003_t050_aux020_5fold_e20_s1"
)
DEFAULT_OUTPUT_DIR = ROOT / "results_stage58C_residual_constrained_configD_5fold"
DEFAULT_RUN_SCRIPT = (
    ROOT / "scripts" / "experiments" / "run_stage58C_residual_constrained_configD_5fold.sh"
)
DEFAULT_AUDIT_SCRIPT = ROOT / "scripts" / "analysis" / "build_stage57B_logit_contribution_audit.py"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))
EXPECTED_FOLDS = 5
STEP57B_VISUAL_BASELINE = 0.7195798650806405
STEP57B_CONCEPT_BASELINE = 0.2804201354438285
METRIC_ALIASES = {
    "AUC": ["test_auc", "auc"],
    "ACC": ["test_acc", "acc"],
    "F1": ["test_f1", "f1", "macro_f1"],
    "Balanced_ACC": ["balanced_acc", "bacc", "balanced acc", "balanced_accuracy"],
    "PR_AUC": ["pr_auc", "prauc", "pr_auc_score"],
}
BRANCH_PRIORITY = [
    "full",
    "concept_only",
    "full_without_visual",
    "visual_only",
    "low_only",
    "high_only",
    "csg_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Step58C residual-constrained 5-fold summary."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage57b-dir", type=Path, default=DEFAULT_STAGE57B_DIR)
    parser.add_argument("--stage57c-dir", type=Path, default=DEFAULT_STAGE57C_DIR)
    parser.add_argument("--stage58c-dir", type=Path, default=DEFAULT_STAGE58C_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
    parser.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT_SCRIPT)
    return parser.parse_args()


def relative_path_str(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def detect_display_root(root: Path) -> Path:
    preferred_candidates = [
        Path("/xiangmu/ViLMIL/ViLa-MIL-main"),
        Path(os.environ.get("PWD", str(Path.cwd()))),
    ]
    for candidate in preferred_candidates:
        if (candidate / "main.py").is_file() and (candidate / "scripts").is_dir():
            return candidate
    return root


def to_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def round_or_none(value: float, digits: int = 6) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def format_delta(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):+.6f}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    return normalized


def metric_from_series(series: pd.Series, aliases: list[str]) -> float:
    lowered = {str(idx).strip().lower(): series[idx] for idx in series.index}
    for alias in aliases:
        key = alias.lower()
        if key in lowered:
            return to_float(lowered[key])
    return math.nan


def metrics_from_row(row: pd.Series) -> dict[str, float]:
    return {
        metric_name: metric_from_series(row, aliases)
        for metric_name, aliases in METRIC_ALIASES.items()
    }


def metrics_from_dataframe_mean(df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    normalized = normalize_columns(df)
    for metric_name, aliases in METRIC_ALIASES.items():
        value = math.nan
        for alias in aliases:
            key = alias.lower()
            if key not in normalized.columns:
                continue
            numeric = pd.to_numeric(normalized[key], errors="coerce").dropna()
            if not numeric.empty:
                value = float(numeric.mean())
                break
        metrics[metric_name] = value
    return metrics


def parse_aggregate_csv(path: Path) -> dict[str, float] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    normalized = normalize_columns(df)
    if "metric" in normalized.columns:
        metric_rows = normalized["metric"].astype(str).str.strip().str.lower()
        mean_rows = normalized.loc[metric_rows == "mean"]
        if not mean_rows.empty:
            return metrics_from_row(mean_rows.iloc[0])

    metrics = metrics_from_dataframe_mean(normalized)
    if any(not pd.isna(value) for value in metrics.values()):
        return metrics
    return None


def parse_fold_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    normalized = normalize_columns(df)
    fold_col = None
    for candidate in ("fold", "folds"):
        if candidate in normalized.columns:
            fold_col = candidate
            break
    if fold_col is None:
        return None

    fold_df = pd.DataFrame()
    fold_df["fold"] = pd.to_numeric(normalized[fold_col], errors="coerce")
    for metric_name, aliases in METRIC_ALIASES.items():
        values = pd.Series([math.nan] * len(normalized), index=normalized.index, dtype=float)
        for alias in aliases:
            key = alias.lower()
            if key in normalized.columns:
                values = pd.to_numeric(normalized[key], errors="coerce")
                break
        fold_df[metric_name] = values

    if fold_df["fold"].dropna().empty:
        return None

    fold_df = fold_df.dropna(subset=["fold"]).copy()
    fold_df["fold"] = fold_df["fold"].astype(int)
    fold_df = fold_df.dropna(how="all", subset=list(METRIC_ALIASES.keys()))
    if fold_df.empty:
        return None
    return fold_df.sort_values("fold").reset_index(drop=True)


def discover_csvs(result_dir: Path) -> list[Path]:
    if not result_dir.exists():
        return []
    return sorted(path for path in result_dir.rglob("*.csv") if path.is_file())


def discover_aggregate_metrics(result_dir: Path) -> tuple[dict[str, float], Path | None]:
    csv_paths = discover_csvs(result_dir)
    preferred: list[Path] = []
    preferred.extend(path for path in csv_paths if path.name == "result.csv")
    preferred.extend(
        path for path in csv_paths if "result" in path.stem.lower() and path not in preferred
    )
    preferred.extend(path for path in csv_paths if path.name == "summary.csv" and path not in preferred)
    preferred.extend(
        path
        for path in csv_paths
        if "summary" in path.stem.lower() and "fold" not in path.stem.lower() and path not in preferred
    )
    preferred.extend(path for path in csv_paths if path not in preferred)

    for path in preferred:
        metrics = parse_aggregate_csv(path)
        if metrics is not None:
            return metrics, path
    return {metric_name: math.nan for metric_name in METRIC_ALIASES}, None


def discover_fold_metrics(result_dir: Path) -> tuple[pd.DataFrame | None, Path | None]:
    csv_paths = discover_csvs(result_dir)
    preferred: list[Path] = []
    preferred.extend(path for path in csv_paths if path.name == "fold_summary.csv")
    preferred.extend(path for path in csv_paths if "fold" in path.stem.lower() and path not in preferred)
    preferred.extend(path for path in csv_paths if path.name == "summary.csv" and path not in preferred)
    preferred.extend(path for path in csv_paths if path not in preferred)

    for path in preferred:
        fold_df = parse_fold_csv(path)
        if fold_df is not None:
            return fold_df, path
    return None, None


def read_stage57b_baseline(stage57b_dir: Path) -> dict[str, float]:
    branch_df = pd.read_csv(stage57b_dir / "stage57B_branch_metrics.csv")
    margin_df = pd.read_csv(stage57b_dir / "stage57B_margin_stats.csv")
    status_payload = json.loads((stage57b_dir / "stage57B_audit_status.json").read_text(encoding="utf-8"))
    branch_map = {row["branch"]: row.to_dict() for _, row in branch_df.iterrows()}
    margin_map = {row["metric_name"]: row.to_dict() for _, row in margin_df.iterrows()}
    return {
        "full_acc": to_float(branch_map.get("full", {}).get("acc")),
        "full_auc": to_float(branch_map.get("full", {}).get("auc")),
        "full_f1": to_float(branch_map.get("full", {}).get("macro_f1")),
        "visual_ratio_mean": to_float(margin_map.get("visual_contribution_ratio", {}).get("mean")),
        "visual_ratio_median": to_float(margin_map.get("visual_contribution_ratio", {}).get("median")),
        "visual_ratio_gt_0_5_percent": to_float(
            status_payload.get("visual_details", {}).get("pct_visual_ratio_gt_0_5")
        ),
        "concept_ratio_mean": to_float(margin_map.get("concept_contribution_ratio", {}).get("mean")),
        "concept_ratio_median": to_float(margin_map.get("concept_contribution_ratio", {}).get("median")),
        "csg_ratio_mean": to_float(margin_map.get("csg_contribution_ratio", {}).get("mean")),
    }


def determine_run_status(run_dir: Path, fold_df: pd.DataFrame | None) -> tuple[str, list[int]]:
    if not run_dir.exists():
        return "not_started", []

    checkpoint_folds: list[int] = []
    for path in sorted(run_dir.glob("s_*_checkpoint.pt")):
        try:
            checkpoint_folds.append(int(path.stem.split("_")[1]))
        except Exception:
            continue

    has_result = (run_dir / "result.csv").is_file() or (run_dir / "summary.csv").is_file()
    has_partial = any(run_dir.glob("result_partial_*.csv")) or any(run_dir.glob("summary_partial_*.csv"))
    fold_count = 0 if fold_df is None else int(fold_df["fold"].nunique())

    if has_result and fold_count >= EXPECTED_FOLDS:
        return "completed", checkpoint_folds
    if checkpoint_folds or has_partial or (run_dir / "fold_summary.csv").is_file():
        return "pending", checkpoint_folds
    return "not_started", checkpoint_folds


def write_run_commands(output_dir: Path, display_root: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = "\n".join(
        [
            f"cd {display_root}",
            "RUN_TRAIN=1 bash scripts/experiments/run_stage58C_residual_constrained_configD_5fold.sh",
            "",
            "# Refresh Step58C summary after training",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage58C_residual_constrained_5fold_summary.py",
        ]
    )
    (output_dir / "stage58C_run_commands.txt").write_text(command_text + "\n", encoding="utf-8")
    return command_text


def run_fold_audit(
    run_dir: Path,
    fold: int,
    audit_output_dir: Path,
    audit_script: Path,
) -> dict[str, object]:
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON_BIN),
        str(audit_script),
        "--run_dir",
        str(run_dir),
        "--fold",
        str(fold),
        "--split",
        "test",
        "--output_dir",
        str(audit_output_dir),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def collect_audits(
    run_dir: Path,
    fold_ids: list[int],
    output_dir: Path,
    audit_script: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], list[str]]:
    branch_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    audit_status: dict[str, object] = {}
    warnings: list[str] = []

    for fold in fold_ids:
        audit_dir = output_dir / "audits" / f"fold_{fold}"
        result = run_fold_audit(run_dir, fold, audit_dir, audit_script)
        audit_status[str(fold)] = {
            "ok": result["ok"],
            "returncode": result["returncode"],
            "audit_dir": relative_path_str(ROOT, audit_dir),
        }
        if not result["ok"]:
            warnings.append(f"audit failed for fold {fold}")
            continue

        branch_path = audit_dir / "stage57B_branch_metrics.csv"
        margin_path = audit_dir / "stage57B_margin_stats.csv"
        status_path = audit_dir / "stage57B_audit_status.json"
        if not branch_path.is_file() or not margin_path.is_file():
            warnings.append(f"audit outputs missing for fold {fold}")
            continue

        branch_df = pd.read_csv(branch_path)
        margin_df = pd.read_csv(margin_path)
        status_payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        margin_map = {row["metric_name"]: row.to_dict() for _, row in margin_df.iterrows()}

        for _, row in branch_df.iterrows():
            branch_rows.append(
                {
                    "fold": fold,
                    "branch": row.get("branch"),
                    "available": row.get("available"),
                    "num_samples": row.get("num_samples"),
                    "ACC": to_float(row.get("acc")),
                    "BACC": to_float(row.get("balanced_acc")),
                    "F1": to_float(row.get("macro_f1")),
                    "AUC": to_float(row.get("auc")),
                    "PR_AUC": to_float(row.get("pr_auc")),
                }
            )

        contribution_rows.append(
            {
                "fold": fold,
                "visual_ratio_mean": to_float(margin_map.get("visual_contribution_ratio", {}).get("mean")),
                "visual_ratio_median": to_float(margin_map.get("visual_contribution_ratio", {}).get("median")),
                "visual_ratio_gt_0_5_percent": to_float(
                    status_payload.get("visual_details", {}).get("pct_visual_ratio_gt_0_5")
                ),
                "concept_ratio_mean": to_float(margin_map.get("concept_contribution_ratio", {}).get("mean")),
                "concept_ratio_median": to_float(margin_map.get("concept_contribution_ratio", {}).get("median")),
                "csg_ratio_mean": to_float(margin_map.get("csg_contribution_ratio", {}).get("mean")),
                "full_margin_mean": to_float(margin_map.get("full_margin", {}).get("mean")),
                "concept_margin_mean": to_float(margin_map.get("concept_margin", {}).get("mean")),
                "visual_margin_mean": to_float(margin_map.get("visual_margin", {}).get("mean")),
                "csg_margin_mean": to_float(margin_map.get("csg_margin", {}).get("mean")),
            }
        )

    branch_df = pd.DataFrame(
        branch_rows,
        columns=["fold", "branch", "available", "num_samples", "ACC", "BACC", "F1", "AUC", "PR_AUC"],
    )
    contribution_df = pd.DataFrame(
        contribution_rows,
        columns=[
            "fold",
            "visual_ratio_mean",
            "visual_ratio_median",
            "visual_ratio_gt_0_5_percent",
            "concept_ratio_mean",
            "concept_ratio_median",
            "csg_ratio_mean",
            "full_margin_mean",
            "concept_margin_mean",
            "visual_margin_mean",
            "csg_margin_mean",
        ],
    )
    return branch_df, contribution_df, audit_status, warnings


def summarize_branch_means(branch_df: pd.DataFrame) -> pd.DataFrame:
    if branch_df.empty:
        return pd.DataFrame(columns=["branch", "ACC", "BACC", "F1", "AUC", "PR_AUC"])
    summary = (
        branch_df.groupby("branch", dropna=False)[["ACC", "BACC", "F1", "AUC", "PR_AUC"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary["branch"] = pd.Categorical(summary["branch"], categories=BRANCH_PRIORITY, ordered=True)
    summary = summary.sort_values("branch").reset_index(drop=True)
    summary["branch"] = summary["branch"].astype(str)
    return summary


def summarize_contribution_means(contribution_df: pd.DataFrame) -> dict[str, float]:
    if contribution_df.empty:
        return {
            "visual_ratio_mean": math.nan,
            "visual_ratio_median": math.nan,
            "visual_ratio_gt_0_5_percent": math.nan,
            "concept_ratio_mean": math.nan,
            "concept_ratio_median": math.nan,
            "csg_ratio_mean": math.nan,
            "full_margin_mean": math.nan,
            "concept_margin_mean": math.nan,
            "visual_margin_mean": math.nan,
            "csg_margin_mean": math.nan,
        }
    numeric_cols = [col for col in contribution_df.columns if col != "fold"]
    means = contribution_df[numeric_cols].mean(numeric_only=True)
    return {col: to_float(means.get(col)) for col in numeric_cols}


def build_compare_df(
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage57c_dir: Path,
    stage58c_dir: Path,
    contribution_means: dict[str, float],
) -> pd.DataFrame:
    baseline_row = {
        "model_name": "stage57C_rce_v2_baseline",
        "source_dir": relative_path_str(ROOT, stage57c_dir),
        **{name: round_or_none(value) for name, value in stage57c_metrics.items()},
        "delta_vs_stage57C_auc": 0.0 if not pd.isna(stage57c_metrics["AUC"]) else None,
        "delta_vs_stage57C_acc": 0.0 if not pd.isna(stage57c_metrics["ACC"]) else None,
        "delta_vs_stage57C_f1": 0.0 if not pd.isna(stage57c_metrics["F1"]) else None,
        "delta_vs_stage57C_bacc": 0.0 if not pd.isna(stage57c_metrics["Balanced_ACC"]) else None,
        "delta_vs_stage57C_pr_auc": 0.0 if not pd.isna(stage57c_metrics["PR_AUC"]) else None,
        "visual_ratio_mean": None,
        "concept_ratio_mean": None,
        "csg_ratio_mean": None,
    }
    candidate_row = {
        "model_name": "stage58C_configD_residual_constrained",
        "source_dir": relative_path_str(ROOT, stage58c_dir),
        **{name: round_or_none(value) for name, value in stage58c_metrics.items()},
        "delta_vs_stage57C_auc": round_or_none(stage58c_metrics["AUC"] - stage57c_metrics["AUC"])
        if not pd.isna(stage58c_metrics["AUC"]) and not pd.isna(stage57c_metrics["AUC"])
        else None,
        "delta_vs_stage57C_acc": round_or_none(stage58c_metrics["ACC"] - stage57c_metrics["ACC"])
        if not pd.isna(stage58c_metrics["ACC"]) and not pd.isna(stage57c_metrics["ACC"])
        else None,
        "delta_vs_stage57C_f1": round_or_none(stage58c_metrics["F1"] - stage57c_metrics["F1"])
        if not pd.isna(stage58c_metrics["F1"]) and not pd.isna(stage57c_metrics["F1"])
        else None,
        "delta_vs_stage57C_bacc": round_or_none(
            stage58c_metrics["Balanced_ACC"] - stage57c_metrics["Balanced_ACC"]
        )
        if not pd.isna(stage58c_metrics["Balanced_ACC"]) and not pd.isna(stage57c_metrics["Balanced_ACC"])
        else None,
        "delta_vs_stage57C_pr_auc": round_or_none(stage58c_metrics["PR_AUC"] - stage57c_metrics["PR_AUC"])
        if not pd.isna(stage58c_metrics["PR_AUC"]) and not pd.isna(stage57c_metrics["PR_AUC"])
        else None,
        "visual_ratio_mean": round_or_none(contribution_means.get("visual_ratio_mean", math.nan)),
        "concept_ratio_mean": round_or_none(contribution_means.get("concept_ratio_mean", math.nan)),
        "csg_ratio_mean": round_or_none(contribution_means.get("csg_ratio_mean", math.nan)),
    }
    return pd.DataFrame([baseline_row, candidate_row])


def build_fold_metrics_df(
    stage57c_fold_df: pd.DataFrame | None,
    stage58c_fold_df: pd.DataFrame | None,
    stage57c_dir: Path,
    stage58c_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if stage57c_fold_df is not None:
        for _, row in stage57c_fold_df.iterrows():
            rows.append(
                {
                    "model_name": "stage57C_rce_v2_baseline",
                    "source_dir": relative_path_str(ROOT, stage57c_dir),
                    "fold": int(row["fold"]),
                    "AUC": round_or_none(to_float(row["AUC"])),
                    "ACC": round_or_none(to_float(row["ACC"])),
                    "F1": round_or_none(to_float(row["F1"])),
                    "Balanced_ACC": round_or_none(to_float(row["Balanced_ACC"])),
                    "PR_AUC": round_or_none(to_float(row["PR_AUC"])),
                    "delta_vs_stage57C_auc": 0.0,
                    "delta_vs_stage57C_acc": 0.0,
                    "delta_vs_stage57C_f1": 0.0,
                    "delta_vs_stage57C_bacc": 0.0,
                    "delta_vs_stage57C_pr_auc": 0.0,
                }
            )

    if stage58c_fold_df is not None:
        baseline_map = {}
        if stage57c_fold_df is not None:
            baseline_map = {
                int(row["fold"]): row.to_dict()
                for _, row in stage57c_fold_df.iterrows()
            }
        for _, row in stage58c_fold_df.iterrows():
            fold = int(row["fold"])
            base_row = baseline_map.get(fold, {})
            rows.append(
                {
                    "model_name": "stage58C_configD_residual_constrained",
                    "source_dir": relative_path_str(ROOT, stage58c_dir),
                    "fold": fold,
                    "AUC": round_or_none(to_float(row["AUC"])),
                    "ACC": round_or_none(to_float(row["ACC"])),
                    "F1": round_or_none(to_float(row["F1"])),
                    "Balanced_ACC": round_or_none(to_float(row["Balanced_ACC"])),
                    "PR_AUC": round_or_none(to_float(row["PR_AUC"])),
                    "delta_vs_stage57C_auc": round_or_none(to_float(row["AUC"]) - to_float(base_row.get("AUC"))),
                    "delta_vs_stage57C_acc": round_or_none(to_float(row["ACC"]) - to_float(base_row.get("ACC"))),
                    "delta_vs_stage57C_f1": round_or_none(to_float(row["F1"]) - to_float(base_row.get("F1"))),
                    "delta_vs_stage57C_bacc": round_or_none(
                        to_float(row["Balanced_ACC"]) - to_float(base_row.get("Balanced_ACC"))
                    ),
                    "delta_vs_stage57C_pr_auc": round_or_none(
                        to_float(row["PR_AUC"]) - to_float(base_row.get("PR_AUC"))
                    ),
                }
            )
    return pd.DataFrame(rows)


def decide_outcome(
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage57c_fold_df: pd.DataFrame | None,
    stage58c_fold_df: pd.DataFrame | None,
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
) -> dict[str, object]:
    if any(pd.isna(stage58c_metrics[key]) for key in ("ACC", "AUC", "F1", "Balanced_ACC", "PR_AUC")):
        return {
            "decision": "pending",
            "next_step": "run_step58c_training",
            "reasons": ["5-fold metrics incomplete"],
        }

    delta_acc = stage58c_metrics["ACC"] - stage57c_metrics["ACC"]
    delta_auc = stage58c_metrics["AUC"] - stage57c_metrics["AUC"]
    delta_f1 = stage58c_metrics["F1"] - stage57c_metrics["F1"]
    delta_bacc = stage58c_metrics["Balanced_ACC"] - stage57c_metrics["Balanced_ACC"]
    delta_pr_auc = stage58c_metrics["PR_AUC"] - stage57c_metrics["PR_AUC"]

    visual_ratio_mean = contribution_means.get("visual_ratio_mean", math.nan)
    concept_ratio_mean = contribution_means.get("concept_ratio_mean", math.nan)
    visual_improved = not pd.isna(visual_ratio_mean) and visual_ratio_mean < STEP57B_VISUAL_BASELINE
    concept_improved = not pd.isna(concept_ratio_mean) and concept_ratio_mean > STEP57B_CONCEPT_BASELINE
    ratio_improved = visual_improved and concept_improved

    performance_ok = delta_acc >= -0.01 and delta_auc >= -0.01

    full_row = branch_mean_df.loc[branch_mean_df["branch"] == "full"]
    concept_row = branch_mean_df.loc[branch_mean_df["branch"] == "concept_only"]
    full_acc_gap = math.nan
    full_auc_gap = math.nan
    branch_ok = False
    if not full_row.empty and not concept_row.empty:
        full_acc_gap = to_float(full_row.iloc[0]["ACC"]) - to_float(concept_row.iloc[0]["ACC"])
        full_auc_gap = to_float(full_row.iloc[0]["AUC"]) - to_float(concept_row.iloc[0]["AUC"])
        branch_ok = full_acc_gap >= -0.01 and full_auc_gap >= -0.01

    severe_fold_collapse = False
    collapse_folds: list[int] = []
    if stage57c_fold_df is not None and stage58c_fold_df is not None:
        merged = stage58c_fold_df.merge(
            stage57c_fold_df,
            on="fold",
            how="left",
            suffixes=("_stage58c", "_stage57c"),
        )
        for _, row in merged.iterrows():
            fold = int(row["fold"])
            acc_gap = to_float(row["ACC_stage58c"]) - to_float(row["ACC_stage57c"])
            auc_gap = to_float(row["AUC_stage58c"]) - to_float(row["AUC_stage57c"])
            if acc_gap < -0.05 or auc_gap < -0.05:
                severe_fold_collapse = True
                collapse_folds.append(fold)

    reasons = [
        f"delta_acc={delta_acc:+.6f}",
        f"delta_auc={delta_auc:+.6f}",
        f"delta_f1={delta_f1:+.6f}",
        f"delta_bacc={delta_bacc:+.6f}",
        f"delta_pr_auc={delta_pr_auc:+.6f}",
        f"visual_ratio_mean={format_metric(visual_ratio_mean)} vs step57B={STEP57B_VISUAL_BASELINE:.4f}",
        f"concept_ratio_mean={format_metric(concept_ratio_mean)} vs step57B={STEP57B_CONCEPT_BASELINE:.4f}",
        f"full_minus_concept_acc={format_metric(full_acc_gap)}",
        f"full_minus_concept_auc={format_metric(full_auc_gap)}",
        f"collapse_folds={collapse_folds}",
    ]

    if performance_ok and ratio_improved and branch_ok and not severe_fold_collapse:
        decision = "candidate_primary"
        next_step = "step59A_dynamic_csg_all_off_implementation"
    elif ratio_improved and not severe_fold_collapse:
        decision = "trade_off"
        next_step = "run_config_C_or_B_5fold"
    else:
        decision = "not_selected"
        next_step = "return_to_step58B_reselect_config"

    return {
        "decision": decision,
        "next_step": next_step,
        "performance_ok": performance_ok,
        "visual_ratio_improved": visual_improved,
        "concept_ratio_improved": concept_improved,
        "ratio_improved": ratio_improved,
        "branch_ok": branch_ok,
        "severe_fold_collapse": severe_fold_collapse,
        "collapse_folds": collapse_folds,
        "delta_vs_stage57c": {
            "auc": round_or_none(delta_auc),
            "acc": round_or_none(delta_acc),
            "f1": round_or_none(delta_f1),
            "bacc": round_or_none(delta_bacc),
            "pr_auc": round_or_none(delta_pr_auc),
        },
        "full_minus_concept": {
            "acc": round_or_none(full_acc_gap),
            "auc": round_or_none(full_auc_gap),
        },
        "reasons": reasons,
    }


def build_summary_md(
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage57c_dir: Path,
    stage58c_dir: Path,
    training_status: str,
    completed_folds: list[int],
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
    decision_payload: dict[str, object],
) -> str:
    full_row = branch_mean_df.loc[branch_mean_df["branch"] == "full"]
    concept_row = branch_mean_df.loc[branch_mean_df["branch"] == "concept_only"]
    visual_row = branch_mean_df.loc[branch_mean_df["branch"] == "visual_only"]

    lines = [
        "# Step58C Residual-Constrained Config D 5-fold formal validation",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。",
        f"3. 是否完成 config D 5-fold 训练：{'是' if training_status == 'completed' else '否'}。",
        "4. config D 的 5-fold AUC / ACC / F1 / BACC / PR-AUC："
        f" {format_metric(stage58c_metrics['AUC'])} / {format_metric(stage58c_metrics['ACC'])} /"
        f" {format_metric(stage58c_metrics['F1'])} / {format_metric(stage58c_metrics['Balanced_ACC'])} /"
        f" {format_metric(stage58c_metrics['PR_AUC'])}。",
        "5. 相比 Stage57C baseline 的差异："
        f" AUC {format_delta(stage58c_metrics['AUC'] - stage57c_metrics['AUC'])},"
        f" ACC {format_delta(stage58c_metrics['ACC'] - stage57c_metrics['ACC'])},"
        f" F1 {format_delta(stage58c_metrics['F1'] - stage57c_metrics['F1'])},"
        f" BACC {format_delta(stage58c_metrics['Balanced_ACC'] - stage57c_metrics['Balanced_ACC'])},"
        f" PR-AUC {format_delta(stage58c_metrics['PR_AUC'] - stage57c_metrics['PR_AUC'])}。",
        "6. visual_ratio_mean 是否明显下降："
        f" {'是' if decision_payload.get('visual_ratio_improved') else '否'}，"
        f"{format_metric(contribution_means.get('visual_ratio_mean', math.nan))}"
        f" vs Step57B {STEP57B_VISUAL_BASELINE:.4f}。",
        "7. concept_ratio_mean 是否明显上升："
        f" {'是' if decision_payload.get('concept_ratio_improved') else '否'}，"
        f"{format_metric(contribution_means.get('concept_ratio_mean', math.nan))}"
        f" vs Step57B {STEP57B_CONCEPT_BASELINE:.4f}。",
        "8. full / concept_only / visual_only 的 5-fold branch 表现如何：",
    ]

    if not full_row.empty:
        lines.append(
            f"   full: ACC {format_metric(to_float(full_row.iloc[0]['ACC']))}, AUC {format_metric(to_float(full_row.iloc[0]['AUC']))}, F1 {format_metric(to_float(full_row.iloc[0]['F1']))}。"
        )
    else:
        lines.append("   full: NA。")
    if not concept_row.empty:
        lines.append(
            f"   concept_only: ACC {format_metric(to_float(concept_row.iloc[0]['ACC']))}, AUC {format_metric(to_float(concept_row.iloc[0]['AUC']))}, F1 {format_metric(to_float(concept_row.iloc[0]['F1']))}。"
        )
    else:
        lines.append("   concept_only: NA。")
    if not visual_row.empty:
        lines.append(
            f"   visual_only: ACC {format_metric(to_float(visual_row.iloc[0]['ACC']))}, AUC {format_metric(to_float(visual_row.iloc[0]['AUC']))}, F1 {format_metric(to_float(visual_row.iloc[0]['F1']))}。"
        )
    else:
        lines.append("   visual_only: NA。")

    lines.extend(
        [
            "9. 是否可以把 config D 作为 residual-constrained RCE 候选主模型："
            f" {'是' if decision_payload.get('decision') == 'candidate_primary' else '否'}"
            f"（decision={decision_payload.get('decision', 'pending')}）。",
            "10. 下一步建议："
            + (
                " 进入 Step59A Dynamic CSG all-off implementation。"
                if decision_payload.get("decision") == "candidate_primary"
                else " 补跑 config C 或 B 的 5-fold。"
                if decision_payload.get("decision") == "trade_off"
                else " 回到 Step58B 重新选参。"
                if decision_payload.get("decision") == "not_selected"
                else " 先执行 Step58C 5-fold 训练。"
            ),
            "",
            "## Status",
            "",
            f"- Stage57C baseline dir: `{relative_path_str(ROOT, stage57c_dir)}`",
            f"- Stage58C candidate dir: `{relative_path_str(ROOT, stage58c_dir)}`",
            f"- training_status: `{training_status}`",
            f"- completed_folds_detected: `{completed_folds}`",
            "",
            "## Decision Basis",
            "",
        ]
    )

    for reason in decision_payload.get("reasons", []):
        lines.append(f"- {reason}")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    display_root = detect_display_root(args.root)
    write_run_commands(output_dir, display_root)

    stage57b_baseline = read_stage57b_baseline(args.stage57b_dir)
    stage57c_metrics, _ = discover_aggregate_metrics(args.stage57c_dir)
    stage57c_fold_df, _ = discover_fold_metrics(args.stage57c_dir)
    stage58c_metrics, stage58c_agg_path = discover_aggregate_metrics(args.stage58c_dir)
    stage58c_fold_df, stage58c_fold_path = discover_fold_metrics(args.stage58c_dir)
    training_status, checkpoint_folds = determine_run_status(args.stage58c_dir, stage58c_fold_df)

    completed_folds = checkpoint_folds
    if training_status == "completed":
        completed_folds = sorted(set(checkpoint_folds))[:EXPECTED_FOLDS]

    branch_df = pd.DataFrame(columns=["fold", "branch", "available", "num_samples", "ACC", "BACC", "F1", "AUC", "PR_AUC"])
    contribution_df = pd.DataFrame(
        columns=[
            "fold",
            "visual_ratio_mean",
            "visual_ratio_median",
            "visual_ratio_gt_0_5_percent",
            "concept_ratio_mean",
            "concept_ratio_median",
            "csg_ratio_mean",
            "full_margin_mean",
            "concept_margin_mean",
            "visual_margin_mean",
            "csg_margin_mean",
        ]
    )
    audit_status: dict[str, object] = {}
    warnings: list[str] = []

    if completed_folds:
        branch_df, contribution_df, audit_status, warnings = collect_audits(
            run_dir=args.stage58c_dir,
            fold_ids=completed_folds,
            output_dir=output_dir,
            audit_script=args.audit_script,
        )

    contribution_means = summarize_contribution_means(contribution_df)
    branch_mean_df = summarize_branch_means(branch_df)
    compare_df = build_compare_df(
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        contribution_means=contribution_means,
    )
    fold_metrics_df = build_fold_metrics_df(
        stage57c_fold_df=stage57c_fold_df,
        stage58c_fold_df=stage58c_fold_df,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
    )
    decision_payload = decide_outcome(
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage57c_fold_df=stage57c_fold_df,
        stage58c_fold_df=stage58c_fold_df,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
    )
    summary_text = build_summary_md(
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        training_status=training_status,
        completed_folds=completed_folds,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
        decision_payload=decision_payload,
    )

    compare_df.to_csv(output_dir / "stage58C_compare_with_stage57C.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "stage58C_fold_metrics.csv", index=False)
    branch_df.to_csv(output_dir / "stage58C_branch_metrics_by_fold.csv", index=False)
    contribution_df.to_csv(output_dir / "stage58C_contribution_by_fold.csv", index=False)
    (output_dir / "stage58C_summary.md").write_text(summary_text, encoding="utf-8")
    (output_dir / "stage58C_decision.json").write_text(
        json.dumps(
            {
                **decision_payload,
                "step57b_baseline": {
                    "visual_ratio_mean": round_or_none(stage57b_baseline["visual_ratio_mean"]),
                    "concept_ratio_mean": round_or_none(stage57b_baseline["concept_ratio_mean"]),
                    "csg_ratio_mean": round_or_none(stage57b_baseline["csg_ratio_mean"]),
                },
                "stage57c_baseline_metrics": {
                    key: round_or_none(value) for key, value in stage57c_metrics.items()
                },
                "stage58c_metrics": {
                    key: round_or_none(value) for key, value in stage58c_metrics.items()
                },
                "contribution_means": {
                    key: round_or_none(value) for key, value in contribution_means.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "stage58C_status.json").write_text(
        json.dumps(
            {
                "status": training_status,
                "completed_folds": completed_folds,
                "stage58c_aggregate_source": None
                if stage58c_agg_path is None
                else relative_path_str(ROOT, stage58c_agg_path),
                "stage58c_fold_source": None
                if stage58c_fold_path is None
                else relative_path_str(ROOT, stage58c_fold_path),
                "audit_status": audit_status,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
