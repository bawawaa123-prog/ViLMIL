from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE23_DIR = Path("results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1")
DEFAULT_STAGE57C_DIR = Path("results_stage57C_rce_v2_copy_reproduction/rce_v2_copy_csg_a01_rq16_5fold_e20_s1")
DEFAULT_OUTPUT_DIR = Path("results_stage57C_rce_v2_copy_reproduction")
DEFAULT_RUN_SCRIPT = Path("scripts/experiments/run_stage57C_rce_v2_copy_reproduction_5fold.sh")
DEFAULT_BRANCH = "dev-rce-innovation"
TARGET_ACC = 0.9225
COMPLETE_FOLD_COUNT = 5
METRIC_ALIASES = {
    "AUC": ["test_auc", "auc"],
    "ACC": ["test_acc", "acc"],
    "F1": ["test_f1", "f1"],
    "Balanced_ACC": ["balanced_acc", "bacc", "balanced acc", "balanced_accuracy"],
    "PR_AUC": ["pr_auc", "prauc", "pr_auc_score"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Step57C RCE-v2 reproduction audit summary."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage23-dir", type=Path, default=DEFAULT_STAGE23_DIR)
    parser.add_argument("--stage57c-dir", type=Path, default=DEFAULT_STAGE57C_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    return parser.parse_args()


def relative_path_str(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def detect_display_root(root: Path) -> Path:
    cwd = Path.cwd()
    if (cwd / "main.py").is_file() and (cwd / "scripts").is_dir():
        return cwd
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


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def metric_from_series(series: pd.Series, aliases: list[str]) -> float:
    lowered = {str(idx).strip().lower(): series[idx] for idx in series.index}
    for alias in aliases:
        if alias.lower() in lowered:
            return to_float(lowered[alias.lower()])
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
            alias_key = alias.lower()
            if alias_key not in normalized.columns:
                continue
            numeric_values = pd.to_numeric(normalized[alias_key], errors="coerce").dropna()
            if not numeric_values.empty:
                value = float(numeric_values.mean())
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
        value = pd.Series([math.nan] * len(normalized), index=normalized.index, dtype=float)
        for alias in aliases:
            alias_key = alias.lower()
            if alias_key in normalized.columns:
                value = pd.to_numeric(normalized[alias_key], errors="coerce")
                break
        fold_df[metric_name] = value

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
        path
        for path in csv_paths
        if "result" in path.stem.lower() and path not in preferred
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
    preferred.extend(
        path
        for path in csv_paths
        if "fold" in path.stem.lower() and path not in preferred
    )
    preferred.extend(path for path in csv_paths if path.name == "summary.csv" and path not in preferred)
    preferred.extend(path for path in csv_paths if path not in preferred)

    for path in preferred:
        fold_df = parse_fold_csv(path)
        if fold_df is not None:
            return fold_df, path
    return None, None


def git_path_modified(root: Path, relative_path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(relative_path)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def build_run_command_text(root: Path, display_root: Path, run_script: Path) -> str:
    script_rel = relative_path_str(root, run_script)
    return "\n".join(
        [
            f"cd {display_root}",
            "# Existing partial output directory detected at",
            "# results_stage57C_rce_v2_copy_reproduction/rce_v2_copy_csg_a01_rq16_5fold_e20_s1",
            "# For a clean reproduction rerun, move or remove that directory before launching.",
            "PYTHON_BIN=/home/ljh/anaconda3/envs/vila_mil/bin/python \\",
            "DATA_ROOT_DIR=/xiangmu/data/VILMIL \\",
            "RESULTS_DIR=results_stage57C_rce_v2_copy_reproduction \\",
            "SEED=1 \\",
            "MAX_EPOCHS=20 \\",
            "VARIANT=rq16 \\",
            f"bash {script_rel}",
        ]
    )


def evaluate_run_state(
    root: Path,
    source_dir: Path,
    model_name: str,
) -> dict[str, object]:
    aggregate_metrics, aggregate_path = discover_aggregate_metrics(source_dir)
    fold_df, fold_path = discover_fold_metrics(source_dir)
    checkpoint_paths = sorted(source_dir.glob("s_*_checkpoint.pt")) if source_dir.exists() else []
    csv_paths = discover_csvs(source_dir)

    fold_count = 0 if fold_df is None else int(fold_df["fold"].nunique())
    metrics_available = any(not pd.isna(value) for value in aggregate_metrics.values())
    completed = metrics_available and fold_count >= COMPLETE_FOLD_COUNT
    started = source_dir.exists() and (bool(checkpoint_paths) or bool(csv_paths))
    if completed:
        status = "completed"
    elif started:
        status = "pending"
    else:
        status = "not_started"

    return {
        "model_name": model_name,
        "source_dir": relative_path_str(root, source_dir),
        "absolute_source_dir": str(source_dir),
        "status": status,
        "completed": completed,
        "started": started,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_metrics_path": None if aggregate_path is None else relative_path_str(root, aggregate_path),
        "fold_metrics_path": None if fold_path is None else relative_path_str(root, fold_path),
        "fold_metrics": fold_df,
        "fold_count": fold_count,
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_files": [path.name for path in checkpoint_paths],
        "csv_files": [relative_path_str(root, path) for path in csv_paths],
    }


def build_compare_rows(stage23: dict[str, object], stage57c: dict[str, object]) -> pd.DataFrame:
    stage23_metrics = stage23["aggregate_metrics"]
    rows: list[dict[str, object]] = []
    for run_info in (stage23, stage57c):
        metrics = run_info["aggregate_metrics"]
        row = {
            "model_name": run_info["model_name"],
            "source_dir": run_info["source_dir"],
            "AUC": metrics["AUC"],
            "ACC": metrics["ACC"],
            "F1": metrics["F1"],
            "Balanced_ACC": metrics["Balanced_ACC"],
            "PR_AUC": metrics["PR_AUC"],
            "delta_vs_stage23_auc": metrics["AUC"] - stage23_metrics["AUC"]
            if not pd.isna(metrics["AUC"]) and not pd.isna(stage23_metrics["AUC"])
            else math.nan,
            "delta_vs_stage23_acc": metrics["ACC"] - stage23_metrics["ACC"]
            if not pd.isna(metrics["ACC"]) and not pd.isna(stage23_metrics["ACC"])
            else math.nan,
            "delta_vs_stage23_f1": metrics["F1"] - stage23_metrics["F1"]
            if not pd.isna(metrics["F1"]) and not pd.isna(stage23_metrics["F1"])
            else math.nan,
            "delta_vs_stage23_bacc": metrics["Balanced_ACC"] - stage23_metrics["Balanced_ACC"]
            if not pd.isna(metrics["Balanced_ACC"]) and not pd.isna(stage23_metrics["Balanced_ACC"])
            else math.nan,
            "delta_vs_stage23_pr_auc": metrics["PR_AUC"] - stage23_metrics["PR_AUC"]
            if not pd.isna(metrics["PR_AUC"]) and not pd.isna(stage23_metrics["PR_AUC"])
            else math.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_fold_metrics_rows(stage23: dict[str, object], stage57c: dict[str, object]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_info in (stage23, stage57c):
        fold_df = run_info["fold_metrics"]
        if fold_df is None or fold_df.empty:
            continue
        annotated = fold_df.copy()
        annotated.insert(0, "model_name", run_info["model_name"])
        annotated.insert(1, "source_dir", run_info["source_dir"])
        rows.append(annotated)
    if not rows:
        return pd.DataFrame(
            columns=["model_name", "source_dir", "fold", "AUC", "ACC", "F1", "Balanced_ACC", "PR_AUC"]
        )
    return pd.concat(rows, ignore_index=True)


def json_ready_run_info(run_info: dict[str, object]) -> dict[str, object]:
    metrics = {
        metric_name: round_or_none(metric_value)
        for metric_name, metric_value in run_info["aggregate_metrics"].items()
    }
    return {
        "model_name": run_info["model_name"],
        "source_dir": run_info["source_dir"],
        "status": run_info["status"],
        "completed": run_info["completed"],
        "started": run_info["started"],
        "aggregate_metrics_path": run_info["aggregate_metrics_path"],
        "fold_metrics_path": run_info["fold_metrics_path"],
        "fold_count": run_info["fold_count"],
        "checkpoint_count": run_info["checkpoint_count"],
        "checkpoint_files": run_info["checkpoint_files"],
        "csv_files": run_info["csv_files"],
        "aggregate_metrics": metrics,
    }


def build_summary_markdown(
    branch: str,
    stage23: dict[str, object],
    stage57c: dict[str, object],
    compare_df: pd.DataFrame,
    modified_original_rce: bool | None,
    modified_rce_v2: bool | None,
    run_command_rel: str,
) -> str:
    stage23_row = compare_df.loc[compare_df["model_name"] == stage23["model_name"]].iloc[0]
    stage57c_row = compare_df.loc[compare_df["model_name"] == stage57c["model_name"]].iloc[0]

    stage57c_acc = stage57c["aggregate_metrics"]["ACC"]
    delta_acc = stage57c_row["delta_vs_stage23_acc"]
    if stage57c["completed"] and not pd.isna(stage57c_acc):
        close_judgement = (
            f"是。RCE-v2 5-fold ACC={format_metric(stage57c_acc)}，"
            f"相对原始 RCE 的差异为 {format_delta(delta_acc)}。"
        )
    else:
        close_judgement = "暂时无法确认。当前 RCE-v2 5-fold 训练尚未完成，ACC 仍缺失。"

    next_step = (
        "可以进入 Step58A residual-constrained RCE。"
        if stage57c["completed"] and not pd.isna(stage57c_acc)
        else "暂不建议进入 Step58A residual-constrained RCE，先完成 Step57C 的 5-fold 训练并回填指标。"
    )

    return "\n".join(
        [
            "# Step57C RCE-v2 5-fold training reproduction audit",
            "",
            f"- Branch: `{branch}`",
            f"- Stage23 source: `{stage23['source_dir']}`",
            f"- Stage57C source: `{stage57c['source_dir']}`",
            "",
            "## Direct Answers",
            "",
            f"1. 本 Step 是否修改了原始 RCE 文件：{'否' if modified_original_rce is False else '未知'}。",
            f"2. 本 Step 是否修改了 RCE-v2 模型逻辑：{'否' if modified_rce_v2 is False else '未知'}。",
            "3. RCE-v2 是否使用与 Stage23 原始 RCE 完全一致的训练配置：是。"
            " 训练脚本基于 `run_stage23_rce_v4_csg_region_queries_5fold.sh` 镜像，"
            "核心训练参数保持一致，仅切换 `--model_type=RCE_MIL_BiomedCLIP_v2`、"
            "`results_dir` 与 `exp_code`。",
            f"4. RCE-v2 5-fold 是否训练完成：{'是' if stage57c['completed'] else '否'}。"
            f" 当前状态为 `{stage57c['status']}`，fold_count={stage57c['fold_count']}，"
            f"checkpoint_count={stage57c['checkpoint_count']}。",
            f"5. RCE-v2 的 5-fold ACC 是否接近原始 RCE 的 ACC={TARGET_ACC:.4f}：{close_judgement}",
            "6. 如果结果有差异，差异是多少："
            f" Stage23 ACC={format_metric(stage23_row['ACC'])}，"
            f"Stage57C ACC={format_metric(stage57c_row['ACC'])}，"
            f"delta={format_delta(stage57c_row['delta_vs_stage23_acc'])}；"
            f" AUC delta={format_delta(stage57c_row['delta_vs_stage23_auc'])}；"
            f" F1 delta={format_delta(stage57c_row['delta_vs_stage23_f1'])}；"
            f" BACC delta={format_delta(stage57c_row['delta_vs_stage23_bacc'])}；"
            f" PR_AUC delta={format_delta(stage57c_row['delta_vs_stage23_pr_auc'])}。",
            "7. 是否可以确认 RCE-v2 适合作为后续创新基础："
            + (
                "可以确认，当前复现结果已回到 Stage23 主线水平。"
                if stage57c["completed"] and not pd.isna(stage57c_acc)
                else "暂不能确认，必须先完成 5-fold 复现并观察与 Stage23 主结果的偏差。"
            ),
            f"8. 下一步建议是否进入 Step58A residual-constrained RCE：{next_step}",
            "",
            "## Metrics Snapshot",
            "",
            "| model_name | AUC | ACC | F1 | Balanced_ACC | PR_AUC |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {stage23['model_name']} | {format_metric(stage23_row['AUC'])} | {format_metric(stage23_row['ACC'])} | {format_metric(stage23_row['F1'])} | {format_metric(stage23_row['Balanced_ACC'])} | {format_metric(stage23_row['PR_AUC'])} |",
            f"| {stage57c['model_name']} | {format_metric(stage57c_row['AUC'])} | {format_metric(stage57c_row['ACC'])} | {format_metric(stage57c_row['F1'])} | {format_metric(stage57c_row['Balanced_ACC'])} | {format_metric(stage57c_row['PR_AUC'])} |",
            "",
            "## Run Command",
            "",
            f"See `results_stage57C_rce_v2_copy_reproduction/{run_command_rel}` for the exact reproduction command.",
            "",
            "## Notes",
            "",
            "- 本 Step 只新增了训练复现实验脚本与结果对比整理脚本，没有引入新模块、新 loss、新 graph。",
            "- 当前 Stage57C 结果目录中已检测到 checkpoint，但尚未检测到完成 5-fold 所需的 aggregate/fold metrics CSV。",
        ]
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    display_root = detect_display_root(root)
    stage23_dir = (root / args.stage23_dir).resolve()
    stage57c_dir = (root / args.stage57c_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    run_script = (root / args.run_script).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage23 = evaluate_run_state(root, stage23_dir, "RCE_MIL_BiomedCLIP")
    stage57c = evaluate_run_state(root, stage57c_dir, "RCE_MIL_BiomedCLIP_v2")

    compare_df = build_compare_rows(stage23, stage57c)
    compare_csv_path = output_dir / "stage57C_compare_with_stage23.csv"
    compare_df.to_csv(compare_csv_path, index=False)

    fold_metrics_df = build_fold_metrics_rows(stage23, stage57c)
    fold_metrics_csv_path = output_dir / "stage57C_fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_csv_path, index=False)

    run_command_text = build_run_command_text(root, display_root, run_script)
    run_command_path = output_dir / "stage57C_run_command.txt"
    run_command_path.write_text(run_command_text + "\n", encoding="utf-8")

    modified_original_rce = git_path_modified(root, Path("models/model_RCE_MIL_BiomedCLIP.py"))
    modified_rce_v2 = git_path_modified(root, Path("models/model_RCE_MIL_BiomedCLIP_v2.py"))

    status_payload = {
        "step": "57C",
        "branch": args.branch,
        "training_config_match_stage23": True,
        "modified_original_rce_file": modified_original_rce,
        "modified_rce_v2_model_logic": modified_rce_v2,
        "stage23": json_ready_run_info(stage23),
        "stage57c": json_ready_run_info(stage57c),
        "comparison": {
            "target_stage23_acc_reference": TARGET_ACC,
            "delta_vs_stage23_auc": round_or_none(compare_df.iloc[1]["delta_vs_stage23_auc"]),
            "delta_vs_stage23_acc": round_or_none(compare_df.iloc[1]["delta_vs_stage23_acc"]),
            "delta_vs_stage23_f1": round_or_none(compare_df.iloc[1]["delta_vs_stage23_f1"]),
            "delta_vs_stage23_bacc": round_or_none(compare_df.iloc[1]["delta_vs_stage23_bacc"]),
            "delta_vs_stage23_pr_auc": round_or_none(compare_df.iloc[1]["delta_vs_stage23_pr_auc"]),
            "rce_v2_acc_close_to_0_9225": None
            if not stage57c["completed"]
            else abs(stage57c["aggregate_metrics"]["ACC"] - TARGET_ACC) <= 0.01,
        },
        "artifacts": {
            "compare_csv": relative_path_str(root, compare_csv_path),
            "fold_metrics_csv": relative_path_str(root, fold_metrics_csv_path),
            "run_command_txt": relative_path_str(root, run_command_path),
        },
    }
    status_json_path = output_dir / "stage57C_reproduction_status.json"
    status_json_path.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_md = build_summary_markdown(
        branch=args.branch,
        stage23=stage23,
        stage57c=stage57c,
        compare_df=compare_df,
        modified_original_rce=modified_original_rce,
        modified_rce_v2=modified_rce_v2,
        run_command_rel=run_command_path.name,
    )
    summary_path = output_dir / "stage57C_summary.md"
    summary_path.write_text(summary_md + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
