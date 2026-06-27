from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_RCE_MIL_BiomedCLIP_v2 import RCE_MIL_BiomedCLIP
from utils.utils import get_simple_loader


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
DEFAULT_STAGE59C_DIR = (
    ROOT
    / "results_stage59C_dynamic_csg_configA_5fold"
    / "rce_v2_rcD_dynCSG_A_5fold_e20_s1"
)
DEFAULT_OUTPUT_DIR = ROOT / "results_stage59C_dynamic_csg_configA_5fold"
DEFAULT_RUN_SCRIPT = (
    ROOT / "scripts" / "experiments" / "run_stage59C_dynamic_csg_configA_5fold.sh"
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
DYNAMIC_CONFIG = {
    "rce_use_dynamic_csg": True,
    "rce_dynamic_csg_mode": "evidence_outer",
    "rce_dynamic_csg_alpha_init": 0.0,
    "rce_dynamic_csg_scale": 1.0,
    "rce_dynamic_csg_norm": "softmax",
    "rce_dynamic_csg_clip": 5.0,
    "rce_dynamic_csg_detach_evidence": False,
}
DYNAMIC_KEYS = [
    "dynamic_csg_enabled",
    "dynamic_delta_mean",
    "dynamic_delta_std",
    "dynamic_adj_mean",
    "dynamic_adj_std",
    "static_csg_logits_mean",
    "dynamic_csg_logits_mean",
    "csg_logits_delta_mean",
    "csg_logits_delta_abs_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Step59C Dynamic CSG config A 5-fold formal validation summary."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage57c-dir", type=Path, default=DEFAULT_STAGE57C_DIR)
    parser.add_argument("--stage58c-dir", type=Path, default=DEFAULT_STAGE58C_DIR)
    parser.add_argument("--stage59c-dir", type=Path, default=DEFAULT_STAGE59C_DIR)
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


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def round_or_none(value: object, digits: int = 6) -> float | None:
    numeric = safe_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def format_metric(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.6f}"


def format_delta(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:+.6f}"


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
            "RUN_TRAIN=1 bash scripts/experiments/run_stage59C_dynamic_csg_configA_5fold.sh",
            "",
            "# Refresh Step59C summary after training",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage59C_dynamic_csg_5fold_summary.py",
        ]
    )
    (output_dir / "stage59C_run_commands.txt").write_text(command_text + "\n", encoding="utf-8")
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
        "--rce_use_dynamic_csg",
        "--rce_dynamic_csg_mode",
        str(DYNAMIC_CONFIG["rce_dynamic_csg_mode"]),
        "--rce_dynamic_csg_alpha_init",
        str(DYNAMIC_CONFIG["rce_dynamic_csg_alpha_init"]),
        "--rce_dynamic_csg_scale",
        str(DYNAMIC_CONFIG["rce_dynamic_csg_scale"]),
        "--rce_dynamic_csg_norm",
        str(DYNAMIC_CONFIG["rce_dynamic_csg_norm"]),
        "--rce_dynamic_csg_clip",
        str(DYNAMIC_CONFIG["rce_dynamic_csg_clip"]),
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


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_experiment_settings(run_dir: Path) -> dict[str, object]:
    files = sorted(run_dir.glob("experiment_*.txt"))
    if not files:
        return {}
    return ast.literal_eval(files[0].read_text(encoding="utf-8"))


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_dataset(settings: dict[str, object]) -> Generic_MIL_Dataset:
    task = str(settings["task"])
    data_root_dir = Path(str(settings["data_root_dir"]))
    data_folder_s = str(settings["data_folder_s"])
    data_folder_l = str(settings["data_folder_l"])

    if task == "task_adenocarcinoma":
        csv_path = ROOT / "dataset_csv" / "all_data.csv"
        class_names = ["Adenocarcinoma", "NonAdenocarcinoma"]
    elif task == "task_tcga_lung_subtyping":
        csv_path = ROOT / "dataset_csv" / "TCGA_Lung_subtyping.csv"
        class_names = ["LUAD", "LUSC"]
    elif task == "task_tcga_rcc_subtyping":
        csv_path = ROOT / "dataset_csv" / "TCGA_RCC_subtyping.csv"
        class_names = ["CCRCC", "PRCC", "CRCC"]
    else:
        raise ValueError(f"Unsupported task for Step59C summary: {task}")

    return Generic_MIL_Dataset(
        csv_path=str(csv_path),
        mode=str(settings.get("mode", "transformer")),
        data_dir_s=str(data_root_dir / data_folder_s),
        data_dir_l=str(data_root_dir / data_folder_l),
        shuffle=False,
        print_info=False,
        label_dict={name: idx for idx, name in enumerate(class_names)},
        patient_strat=False,
        ignore=[],
    )


def build_model_config(settings: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        input_size=512,
        hidden_size=192,
        class_names=settings.get("class_names"),
        use_concept_prompt_pool=to_bool(settings.get("use_concept_prompt_pool", False)),
        concept_prompt_path=settings.get("concept_prompt_path"),
        peps_tau=float(settings.get("peps_tau", 0.1)),
        prototype_number=int(settings.get("prototype_number", 16)),
        rce_use_logit_calibration=to_bool(settings.get("rce_use_logit_calibration", False)),
        rce_use_concept_prior=to_bool(settings.get("rce_use_concept_prior", False)),
        rce_logit_scale_init=float(settings.get("rce_logit_scale_init", 10.0)),
        rce_concept_prior_strength=float(settings.get("rce_concept_prior_strength", 1.0)),
        rce_use_visual_residual=to_bool(settings.get("rce_use_visual_residual", False)),
        rce_visual_residual_init=float(settings.get("rce_visual_residual_init", 0.1)),
        rce_use_residual_constraint=to_bool(settings.get("rce_use_residual_constraint", False)),
        rce_residual_constraint_lambda=float(settings.get("rce_residual_constraint_lambda", 0.0)),
        rce_residual_ratio_target=float(settings.get("rce_residual_ratio_target", 0.5)),
        rce_residual_constraint_type=str(settings.get("rce_residual_constraint_type", "relu_l2")),
        rce_use_concept_aux_loss=to_bool(settings.get("rce_use_concept_aux_loss", False)),
        rce_concept_aux_loss_weight=float(settings.get("rce_concept_aux_loss_weight", 0.0)),
        rce_residual_ratio_eps=float(settings.get("rce_residual_ratio_eps", 1e-6)),
        rce_residual_ratio_detach=to_bool(settings.get("rce_residual_ratio_detach", False)),
        rce_use_cross_scale_graph=to_bool(settings.get("rce_use_cross_scale_graph", False)),
        rce_cross_scale_graph_init=float(settings.get("rce_cross_scale_graph_init", 0.05)),
        rce_cross_scale_graph_norm=str(settings.get("rce_cross_scale_graph_norm", "sqrt")),
        rce_use_dynamic_csg=to_bool(settings.get("rce_use_dynamic_csg", False)),
        rce_dynamic_csg_mode=str(settings.get("rce_dynamic_csg_mode", "evidence_outer")),
        rce_dynamic_csg_alpha_init=float(settings.get("rce_dynamic_csg_alpha_init", 0.0)),
        rce_dynamic_csg_scale=float(settings.get("rce_dynamic_csg_scale", 1.0)),
        rce_dynamic_csg_norm=str(settings.get("rce_dynamic_csg_norm", "softmax")),
        rce_dynamic_csg_detach_evidence=to_bool(settings.get("rce_dynamic_csg_detach_evidence", False)),
        rce_dynamic_csg_clip=float(settings.get("rce_dynamic_csg_clip", 5.0)),
        scale_mode=str(settings.get("scale_mode", "dual")),
        finetune_text_encoder=False,
        enable_logit_breakdown_audit=True,
    )


def apply_dynamic_config_overrides(settings: dict[str, object]) -> dict[str, object]:
    merged = dict(settings)
    merged.update(DYNAMIC_CONFIG)
    return merged


def collect_dynamic_metrics_for_fold(run_dir: Path, fold: int) -> dict[str, object]:
    settings = read_experiment_settings(run_dir)
    if not settings:
        return {}
    settings = apply_dynamic_config_overrides(settings)

    dataset = build_dataset(settings)
    split_dir = Path(str(settings["split_dir"]))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    _, _, test_split = dataset.return_splits(
        from_id=False,
        csv_path=str(split_dir / f"splits_{fold}.csv"),
    )
    loader = get_simple_loader(test_split, mode=str(settings.get("mode", "transformer")))
    model = RCE_MIL_BiomedCLIP(
        config=build_model_config(settings),
        num_classes=int(settings["n_classes"]),
    )
    ckpt_path = run_dir / f"s_{fold}_checkpoint.pt"
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.relocate()
    model.eval()
    if hasattr(model, "set_logit_breakdown_audit"):
        model.set_logit_breakdown_audit(True)
    device = next(model.parameters()).device

    dynamic_samples: list[dict[str, float | None]] = []
    anomaly_count = 0

    with torch.no_grad():
        for data_s, coord_s, data_l, coords_l, label, slide_ids in loader:
            slide_id = slide_ids[0] if isinstance(slide_ids, (list, tuple)) and slide_ids else None
            y_prob, _, _ = model(
                data_s.to(device),
                coord_s.to(device),
                data_l.to(device),
                coords_l.to(device),
                label.to(device),
                slide_id=slide_id,
            )
            if torch.isnan(y_prob).any() or torch.isinf(y_prob).any():
                anomaly_count += 1
            breakdown = getattr(model, "last_dynamic_csg_breakdown", None) or {}
            dynamic_samples.append({key: safe_float(breakdown.get(key)) for key in DYNAMIC_KEYS})

    alpha_final = safe_float(getattr(model, "rce_dynamic_csg_alpha", None))
    if alpha_final is None and hasattr(model, "rce_dynamic_csg_alpha") and model.rce_dynamic_csg_alpha is not None:
        alpha_final = safe_float(model.rce_dynamic_csg_alpha.detach())

    summary: dict[str, object] = {
        "learned_alpha_final": alpha_final,
        "dynamic_csg_enabled": None,
        "dynamic_csg_scale": safe_float(settings.get("rce_dynamic_csg_scale")),
        "dynamic_csg_norm": settings.get("rce_dynamic_csg_norm"),
        "dynamic_csg_clip": safe_float(settings.get("rce_dynamic_csg_clip")),
        "anomaly_count": anomaly_count,
    }
    for key in DYNAMIC_KEYS:
        values = [sample.get(key) for sample in dynamic_samples]
        if key == "dynamic_csg_enabled":
            non_null = [value for value in values if value is not None]
            summary[key] = None if not non_null else round(float(np.mean(non_null)), 6)
            continue
        numeric = [value for value in values if value is not None]
        summary[key] = None if not numeric else float(np.mean(numeric))
    return summary


def collect_audits_and_dynamic(
    run_dir: Path,
    fold_ids: list[int],
    output_dir: Path,
    audit_script: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], list[str]]:
    branch_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    dynamic_rows: list[dict[str, object]] = []
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
            warnings.append(f"audit_failed:fold_{fold}")
            dynamic_rows.append(
                {
                    "fold": fold,
                    "learned_alpha_final": None,
                    "dynamic_csg_enabled": None,
                    "dynamic_csg_scale": DYNAMIC_CONFIG["rce_dynamic_csg_scale"],
                    "dynamic_csg_norm": DYNAMIC_CONFIG["rce_dynamic_csg_norm"],
                    "dynamic_csg_clip": DYNAMIC_CONFIG["rce_dynamic_csg_clip"],
                    "dynamic_delta_mean": None,
                    "dynamic_delta_std": None,
                    "dynamic_adj_mean": None,
                    "dynamic_adj_std": None,
                    "static_csg_logits_mean": None,
                    "dynamic_csg_logits_mean": None,
                    "csg_logits_delta_mean": None,
                    "csg_logits_delta_abs_mean": None,
                    "status": "pending",
                    "anomaly_count": None,
                }
            )
            continue

        branch_path = audit_dir / "stage57B_branch_metrics.csv"
        margin_path = audit_dir / "stage57B_margin_stats.csv"
        status_path = audit_dir / "stage57B_audit_status.json"
        if not branch_path.is_file() or not margin_path.is_file():
            warnings.append(f"audit_outputs_missing:fold_{fold}")
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
                "csg_ratio_median": to_float(margin_map.get("csg_contribution_ratio", {}).get("median")),
                "full_margin_mean": to_float(margin_map.get("full_margin", {}).get("mean")),
                "concept_margin_mean": to_float(margin_map.get("concept_margin", {}).get("mean")),
                "visual_margin_mean": to_float(margin_map.get("visual_margin", {}).get("mean")),
                "csg_margin_mean": to_float(margin_map.get("csg_margin", {}).get("mean")),
            }
        )

        try:
            dynamic_payload = collect_dynamic_metrics_for_fold(run_dir, fold)
            dynamic_rows.append(
                {
                    "fold": fold,
                    "learned_alpha_final": dynamic_payload.get("learned_alpha_final"),
                    "dynamic_csg_enabled": dynamic_payload.get("dynamic_csg_enabled"),
                    "dynamic_csg_scale": dynamic_payload.get("dynamic_csg_scale"),
                    "dynamic_csg_norm": dynamic_payload.get("dynamic_csg_norm"),
                    "dynamic_csg_clip": dynamic_payload.get("dynamic_csg_clip"),
                    "dynamic_delta_mean": dynamic_payload.get("dynamic_delta_mean"),
                    "dynamic_delta_std": dynamic_payload.get("dynamic_delta_std"),
                    "dynamic_adj_mean": dynamic_payload.get("dynamic_adj_mean"),
                    "dynamic_adj_std": dynamic_payload.get("dynamic_adj_std"),
                    "static_csg_logits_mean": dynamic_payload.get("static_csg_logits_mean"),
                    "dynamic_csg_logits_mean": dynamic_payload.get("dynamic_csg_logits_mean"),
                    "csg_logits_delta_mean": dynamic_payload.get("csg_logits_delta_mean"),
                    "csg_logits_delta_abs_mean": dynamic_payload.get("csg_logits_delta_abs_mean"),
                    "status": "completed",
                    "anomaly_count": dynamic_payload.get("anomaly_count"),
                }
            )
        except Exception as exc:
            warnings.append(f"dynamic_audit_failed:fold_{fold}:{exc}")
            dynamic_rows.append(
                {
                    "fold": fold,
                    "learned_alpha_final": None,
                    "dynamic_csg_enabled": None,
                    "dynamic_csg_scale": DYNAMIC_CONFIG["rce_dynamic_csg_scale"],
                    "dynamic_csg_norm": DYNAMIC_CONFIG["rce_dynamic_csg_norm"],
                    "dynamic_csg_clip": DYNAMIC_CONFIG["rce_dynamic_csg_clip"],
                    "dynamic_delta_mean": None,
                    "dynamic_delta_std": None,
                    "dynamic_adj_mean": None,
                    "dynamic_adj_std": None,
                    "static_csg_logits_mean": None,
                    "dynamic_csg_logits_mean": None,
                    "csg_logits_delta_mean": None,
                    "csg_logits_delta_abs_mean": None,
                    "status": "pending",
                    "anomaly_count": None,
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
            "csg_ratio_median",
            "full_margin_mean",
            "concept_margin_mean",
            "visual_margin_mean",
            "csg_margin_mean",
        ],
    )
    dynamic_df = pd.DataFrame(
        dynamic_rows,
        columns=[
            "fold",
            "learned_alpha_final",
            "dynamic_csg_enabled",
            "dynamic_csg_scale",
            "dynamic_csg_norm",
            "dynamic_csg_clip",
            "dynamic_delta_mean",
            "dynamic_delta_std",
            "dynamic_adj_mean",
            "dynamic_adj_std",
            "static_csg_logits_mean",
            "dynamic_csg_logits_mean",
            "csg_logits_delta_mean",
            "csg_logits_delta_abs_mean",
            "status",
            "anomaly_count",
        ],
    )
    return branch_df, contribution_df, dynamic_df, audit_status, warnings


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
            "csg_ratio_median": math.nan,
            "full_margin_mean": math.nan,
            "concept_margin_mean": math.nan,
            "visual_margin_mean": math.nan,
            "csg_margin_mean": math.nan,
        }
    numeric_cols = [col for col in contribution_df.columns if col != "fold"]
    means = contribution_df[numeric_cols].mean(numeric_only=True)
    return {col: to_float(means.get(col)) for col in numeric_cols}


def summarize_dynamic_means(dynamic_df: pd.DataFrame) -> dict[str, float | str | None]:
    if dynamic_df.empty:
        return {
            "learned_alpha_final": math.nan,
            "dynamic_csg_enabled": math.nan,
            "dynamic_csg_scale": DYNAMIC_CONFIG["rce_dynamic_csg_scale"],
            "dynamic_csg_norm": DYNAMIC_CONFIG["rce_dynamic_csg_norm"],
            "dynamic_csg_clip": DYNAMIC_CONFIG["rce_dynamic_csg_clip"],
            "dynamic_delta_mean": math.nan,
            "dynamic_delta_std": math.nan,
            "dynamic_adj_mean": math.nan,
            "dynamic_adj_std": math.nan,
            "static_csg_logits_mean": math.nan,
            "dynamic_csg_logits_mean": math.nan,
            "csg_logits_delta_mean": math.nan,
            "csg_logits_delta_abs_mean": math.nan,
        }
    summary: dict[str, float | str | None] = {
        "dynamic_csg_norm": None,
    }
    for col in [
        "learned_alpha_final",
        "dynamic_csg_enabled",
        "dynamic_csg_scale",
        "dynamic_csg_clip",
        "dynamic_delta_mean",
        "dynamic_delta_std",
        "dynamic_adj_mean",
        "dynamic_adj_std",
        "static_csg_logits_mean",
        "dynamic_csg_logits_mean",
        "csg_logits_delta_mean",
        "csg_logits_delta_abs_mean",
    ]:
        summary[col] = to_float(pd.to_numeric(dynamic_df[col], errors="coerce").mean()) if col in dynamic_df.columns else math.nan
    if "dynamic_csg_norm" in dynamic_df.columns and not dynamic_df["dynamic_csg_norm"].dropna().empty:
        summary["dynamic_csg_norm"] = str(dynamic_df["dynamic_csg_norm"].dropna().iloc[0])
    return summary


def build_compare_df(
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage59c_metrics: dict[str, float],
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
    contribution_means: dict[str, float],
    dynamic_means: dict[str, float | str | None],
) -> pd.DataFrame:
    rows = [
        {
            "model_name": "stage57C_rce_v2_baseline",
            "source_dir": relative_path_str(ROOT, stage57c_dir),
            **{name: round_or_none(value) for name, value in stage57c_metrics.items()},
            "BACC": round_or_none(stage57c_metrics["Balanced_ACC"]),
            "delta_vs_stage58C_auc": None,
            "delta_vs_stage58C_acc": None,
            "delta_vs_stage58C_f1": None,
            "delta_vs_stage58C_bacc": None,
            "delta_vs_stage58C_pr_auc": None,
            "visual_ratio_mean": None,
            "concept_ratio_mean": None,
            "csg_ratio_mean": None,
            "csg_logits_delta_abs_mean": None,
            "learned_alpha_final": None,
        },
        {
            "model_name": "stage58C_residual_constrained_configD",
            "source_dir": relative_path_str(ROOT, stage58c_dir),
            **{name: round_or_none(value) for name, value in stage58c_metrics.items()},
            "BACC": round_or_none(stage58c_metrics["Balanced_ACC"]),
            "delta_vs_stage58C_auc": 0.0 if not pd.isna(stage58c_metrics["AUC"]) else None,
            "delta_vs_stage58C_acc": 0.0 if not pd.isna(stage58c_metrics["ACC"]) else None,
            "delta_vs_stage58C_f1": 0.0 if not pd.isna(stage58c_metrics["F1"]) else None,
            "delta_vs_stage58C_bacc": 0.0 if not pd.isna(stage58c_metrics["Balanced_ACC"]) else None,
            "delta_vs_stage58C_pr_auc": 0.0 if not pd.isna(stage58c_metrics["PR_AUC"]) else None,
            "visual_ratio_mean": None,
            "concept_ratio_mean": None,
            "csg_ratio_mean": None,
            "csg_logits_delta_abs_mean": None,
            "learned_alpha_final": None,
        },
        {
            "model_name": "stage59C_dynamic_csg_configA",
            "source_dir": relative_path_str(ROOT, stage59c_dir),
            **{name: round_or_none(value) for name, value in stage59c_metrics.items()},
            "BACC": round_or_none(stage59c_metrics["Balanced_ACC"]),
            "delta_vs_stage58C_auc": round_or_none(stage59c_metrics["AUC"] - stage58c_metrics["AUC"])
            if not pd.isna(stage59c_metrics["AUC"]) and not pd.isna(stage58c_metrics["AUC"])
            else None,
            "delta_vs_stage58C_acc": round_or_none(stage59c_metrics["ACC"] - stage58c_metrics["ACC"])
            if not pd.isna(stage59c_metrics["ACC"]) and not pd.isna(stage58c_metrics["ACC"])
            else None,
            "delta_vs_stage58C_f1": round_or_none(stage59c_metrics["F1"] - stage58c_metrics["F1"])
            if not pd.isna(stage59c_metrics["F1"]) and not pd.isna(stage58c_metrics["F1"])
            else None,
            "delta_vs_stage58C_bacc": round_or_none(
                stage59c_metrics["Balanced_ACC"] - stage58c_metrics["Balanced_ACC"]
            )
            if not pd.isna(stage59c_metrics["Balanced_ACC"]) and not pd.isna(stage58c_metrics["Balanced_ACC"])
            else None,
            "delta_vs_stage58C_pr_auc": round_or_none(stage59c_metrics["PR_AUC"] - stage58c_metrics["PR_AUC"])
            if not pd.isna(stage59c_metrics["PR_AUC"]) and not pd.isna(stage58c_metrics["PR_AUC"])
            else None,
            "visual_ratio_mean": round_or_none(contribution_means.get("visual_ratio_mean")),
            "concept_ratio_mean": round_or_none(contribution_means.get("concept_ratio_mean")),
            "csg_ratio_mean": round_or_none(contribution_means.get("csg_ratio_mean")),
            "csg_logits_delta_abs_mean": round_or_none(dynamic_means.get("csg_logits_delta_abs_mean")),
            "learned_alpha_final": round_or_none(dynamic_means.get("learned_alpha_final")),
        },
    ]
    return pd.DataFrame(rows)


def build_fold_metrics_df(
    stage57c_fold_df: pd.DataFrame | None,
    stage58c_fold_df: pd.DataFrame | None,
    stage59c_fold_df: pd.DataFrame | None,
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if stage57c_fold_df is not None:
        for _, row in stage57c_fold_df.iterrows():
            rows.append(
                {
                    "model_name": "stage57C_rce_v2_baseline",
                    "source_dir": relative_path_str(ROOT, stage57c_dir),
                    "fold": int(row["fold"]),
                    "AUC": round_or_none(row["AUC"]),
                    "ACC": round_or_none(row["ACC"]),
                    "F1": round_or_none(row["F1"]),
                    "Balanced_ACC": round_or_none(row["Balanced_ACC"]),
                    "BACC": round_or_none(row["Balanced_ACC"]),
                    "PR_AUC": round_or_none(row["PR_AUC"]),
                    "delta_vs_stage58C_auc": None,
                    "delta_vs_stage58C_acc": None,
                    "delta_vs_stage58C_f1": None,
                    "delta_vs_stage58C_bacc": None,
                    "delta_vs_stage58C_pr_auc": None,
                }
            )

    if stage58c_fold_df is not None:
        for _, row in stage58c_fold_df.iterrows():
            rows.append(
                {
                    "model_name": "stage58C_residual_constrained_configD",
                    "source_dir": relative_path_str(ROOT, stage58c_dir),
                    "fold": int(row["fold"]),
                    "AUC": round_or_none(row["AUC"]),
                    "ACC": round_or_none(row["ACC"]),
                    "F1": round_or_none(row["F1"]),
                    "Balanced_ACC": round_or_none(row["Balanced_ACC"]),
                    "BACC": round_or_none(row["Balanced_ACC"]),
                    "PR_AUC": round_or_none(row["PR_AUC"]),
                    "delta_vs_stage58C_auc": 0.0 if not pd.isna(row["AUC"]) else None,
                    "delta_vs_stage58C_acc": 0.0 if not pd.isna(row["ACC"]) else None,
                    "delta_vs_stage58C_f1": 0.0 if not pd.isna(row["F1"]) else None,
                    "delta_vs_stage58C_bacc": 0.0 if not pd.isna(row["Balanced_ACC"]) else None,
                    "delta_vs_stage58C_pr_auc": 0.0 if not pd.isna(row["PR_AUC"]) else None,
                }
            )

    if stage59c_fold_df is not None:
        baseline_map = {}
        if stage58c_fold_df is not None:
            baseline_map = {int(row["fold"]): row.to_dict() for _, row in stage58c_fold_df.iterrows()}
        for _, row in stage59c_fold_df.iterrows():
            fold = int(row["fold"])
            base_row = baseline_map.get(fold, {})
            rows.append(
                {
                    "model_name": "stage59C_dynamic_csg_configA",
                    "source_dir": relative_path_str(ROOT, stage59c_dir),
                    "fold": fold,
                    "AUC": round_or_none(row["AUC"]),
                    "ACC": round_or_none(row["ACC"]),
                    "F1": round_or_none(row["F1"]),
                    "Balanced_ACC": round_or_none(row["Balanced_ACC"]),
                    "BACC": round_or_none(row["Balanced_ACC"]),
                    "PR_AUC": round_or_none(row["PR_AUC"]),
                    "delta_vs_stage58C_auc": round_or_none(to_float(row["AUC"]) - to_float(base_row.get("AUC"))),
                    "delta_vs_stage58C_acc": round_or_none(to_float(row["ACC"]) - to_float(base_row.get("ACC"))),
                    "delta_vs_stage58C_f1": round_or_none(to_float(row["F1"]) - to_float(base_row.get("F1"))),
                    "delta_vs_stage58C_bacc": round_or_none(
                        to_float(row["Balanced_ACC"]) - to_float(base_row.get("Balanced_ACC"))
                    ),
                    "delta_vs_stage58C_pr_auc": round_or_none(
                        to_float(row["PR_AUC"]) - to_float(base_row.get("PR_AUC"))
                    ),
                }
            )
    return pd.DataFrame(rows)


def collect_completed_checkpoint_folds(run_dir: Path) -> list[int]:
    folds: list[int] = []
    if not run_dir.exists():
        return folds
    for path in sorted(run_dir.glob("s_*_checkpoint.pt")):
        try:
            folds.append(int(path.stem.split("_")[1]))
        except Exception:
            continue
    return sorted(set(folds))


def decide_outcome(
    stage58c_metrics: dict[str, float],
    stage59c_metrics: dict[str, float],
    stage58c_fold_df: pd.DataFrame | None,
    stage59c_fold_df: pd.DataFrame | None,
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
    dynamic_means: dict[str, float | str | None],
    dynamic_df: pd.DataFrame,
) -> dict[str, object]:
    if any(pd.isna(stage59c_metrics[key]) for key in ("ACC", "AUC", "F1", "Balanced_ACC", "PR_AUC")):
        return {
            "decision": "pending",
            "next_step": "run_step59c_training",
            "reasons": ["5-fold metrics incomplete"],
        }

    delta_acc = stage59c_metrics["ACC"] - stage58c_metrics["ACC"]
    delta_auc = stage59c_metrics["AUC"] - stage58c_metrics["AUC"]
    delta_f1 = stage59c_metrics["F1"] - stage58c_metrics["F1"]
    delta_bacc = stage59c_metrics["Balanced_ACC"] - stage58c_metrics["Balanced_ACC"]
    delta_pr_auc = stage59c_metrics["PR_AUC"] - stage58c_metrics["PR_AUC"]

    visual_ratio_mean = contribution_means.get("visual_ratio_mean", math.nan)
    concept_ratio_mean = contribution_means.get("concept_ratio_mean", math.nan)
    csg_ratio_mean = contribution_means.get("csg_ratio_mean", math.nan)
    csg_delta_abs_mean = safe_float(dynamic_means.get("csg_logits_delta_abs_mean"))
    learned_alpha_mean = safe_float(dynamic_means.get("learned_alpha_final"))

    visual_low = not pd.isna(visual_ratio_mean) and visual_ratio_mean < 0.5
    visual_not_rebounded = not pd.isna(visual_ratio_mean) and visual_ratio_mean < STEP57B_VISUAL_BASELINE
    concept_high = not pd.isna(concept_ratio_mean) and concept_ratio_mean > 0.5
    concept_above_step57b = not pd.isna(concept_ratio_mean) and concept_ratio_mean > STEP57B_CONCEPT_BASELINE
    csg_observable = (not pd.isna(csg_ratio_mean) and csg_ratio_mean > 0.0) or (
        csg_delta_abs_mean is not None and csg_delta_abs_mean > 1e-6
    )

    alpha_nonzero_folds: list[int] = []
    if not dynamic_df.empty and "learned_alpha_final" in dynamic_df.columns:
        for _, row in dynamic_df.iterrows():
            alpha_value = safe_float(row.get("learned_alpha_final"))
            if alpha_value is not None and abs(alpha_value) > 1e-6:
                alpha_nonzero_folds.append(int(row["fold"]))
    alpha_nonzero_multi_fold = len(alpha_nonzero_folds) >= 2

    performance_close = (
        delta_acc >= -0.01
        and delta_auc >= -0.01
        and delta_f1 >= -0.02
        and delta_bacc >= -0.02
        and delta_pr_auc >= -0.02
    )

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
    if stage58c_fold_df is not None and stage59c_fold_df is not None:
        merged = stage59c_fold_df.merge(
            stage58c_fold_df,
            on="fold",
            how="left",
            suffixes=("_stage59c", "_stage58c"),
        )
        for _, row in merged.iterrows():
            fold = int(row["fold"])
            acc_gap = to_float(row["ACC_stage59c"]) - to_float(row["ACC_stage58c"])
            auc_gap = to_float(row["AUC_stage59c"]) - to_float(row["AUC_stage58c"])
            if acc_gap < -0.05 or auc_gap < -0.05:
                severe_fold_collapse = True
                collapse_folds.append(fold)

    reasons = [
        f"delta_acc={delta_acc:+.6f}",
        f"delta_auc={delta_auc:+.6f}",
        f"delta_f1={delta_f1:+.6f}",
        f"delta_bacc={delta_bacc:+.6f}",
        f"delta_pr_auc={delta_pr_auc:+.6f}",
        f"visual_ratio_mean={format_metric(visual_ratio_mean)} vs step57B={STEP57B_VISUAL_BASELINE:.6f}",
        f"concept_ratio_mean={format_metric(concept_ratio_mean)} vs step57B={STEP57B_CONCEPT_BASELINE:.6f}",
        f"csg_ratio_mean={format_metric(csg_ratio_mean)}",
        f"csg_logits_delta_abs_mean={format_metric(csg_delta_abs_mean)}",
        f"learned_alpha_mean={format_metric(learned_alpha_mean)}",
        f"alpha_nonzero_folds={alpha_nonzero_folds}",
        f"full_minus_concept_acc={format_metric(full_acc_gap)}",
        f"full_minus_concept_auc={format_metric(full_auc_gap)}",
        f"collapse_folds={collapse_folds}",
    ]

    if (
        performance_close
        and visual_low
        and visual_not_rebounded
        and concept_high
        and concept_above_step57b
        and csg_observable
        and alpha_nonzero_multi_fold
        and branch_ok
        and not severe_fold_collapse
    ):
        decision = "candidate_primary"
        next_step = "promote_dynamic_csg_candidate"
    elif performance_close and visual_not_rebounded and concept_above_step57b and branch_ok and not severe_fold_collapse:
        decision = "research_variant_only"
        next_step = "do_not_promote_but_archive_findings"
    else:
        decision = "stop_dynamic_csg_move_to_step60"
        next_step = "step60"

    return {
        "decision": decision,
        "next_step": next_step,
        "performance_close_to_step58c": performance_close,
        "visual_low": visual_low,
        "visual_not_rebounded": visual_not_rebounded,
        "concept_high": concept_high,
        "concept_above_step57b": concept_above_step57b,
        "csg_observable": csg_observable,
        "alpha_nonzero_multi_fold": alpha_nonzero_multi_fold,
        "alpha_nonzero_folds": alpha_nonzero_folds,
        "branch_ok": branch_ok,
        "severe_fold_collapse": severe_fold_collapse,
        "collapse_folds": collapse_folds,
        "delta_vs_stage58c": {
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
    stage58c_metrics: dict[str, float],
    stage59c_metrics: dict[str, float],
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
    training_status: str,
    completed_folds: list[int],
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
    dynamic_means: dict[str, float | str | None],
    decision_payload: dict[str, object],
) -> str:
    full_row = branch_mean_df.loc[branch_mean_df["branch"] == "full"]
    concept_row = branch_mean_df.loc[branch_mean_df["branch"] == "concept_only"]
    visual_row = branch_mean_df.loc[branch_mean_df["branch"] == "visual_only"]
    csg_row = branch_mean_df.loc[branch_mean_df["branch"] == "csg_only"]

    lines = [
        "# Step59C Dynamic CSG Config A 5-fold formal validation",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。",
        f"3. 是否完成 config A 5-fold 训练：{'是' if training_status == 'completed' else '否'}。",
        "4. config A 的 5-fold AUC / ACC / F1 / BACC / PR-AUC："
        f" {format_metric(stage59c_metrics['AUC'])} / {format_metric(stage59c_metrics['ACC'])} /"
        f" {format_metric(stage59c_metrics['F1'])} / {format_metric(stage59c_metrics['Balanced_ACC'])} /"
        f" {format_metric(stage59c_metrics['PR_AUC'])}。",
        "5. 相比 Step58C config D 的差异："
        f" AUC {format_delta(stage59c_metrics['AUC'] - stage58c_metrics['AUC'])},"
        f" ACC {format_delta(stage59c_metrics['ACC'] - stage58c_metrics['ACC'])},"
        f" F1 {format_delta(stage59c_metrics['F1'] - stage58c_metrics['F1'])},"
        f" BACC {format_delta(stage59c_metrics['Balanced_ACC'] - stage58c_metrics['Balanced_ACC'])},"
        f" PR-AUC {format_delta(stage59c_metrics['PR_AUC'] - stage58c_metrics['PR_AUC'])}。",
        "6. visual_ratio 是否仍保持较低且未回到 Step57B visual-dominant："
        f" {'是' if decision_payload.get('visual_low') and decision_payload.get('visual_not_rebounded') else '否'}，"
        f"{format_metric(contribution_means.get('visual_ratio_mean'))} vs Step57B {STEP57B_VISUAL_BASELINE:.6f}。",
        "7. concept_ratio 是否仍保持较高："
        f" {'是' if decision_payload.get('concept_high') and decision_payload.get('concept_above_step57b') else '否'}，"
        f"{format_metric(contribution_means.get('concept_ratio_mean'))}。",
        "8. csg_ratio 或 csg_logits_delta_abs_mean 是否有可观察提升："
        f" {'是' if decision_payload.get('csg_observable') else '否'}，"
        f"csg_ratio_mean={format_metric(contribution_means.get('csg_ratio_mean'))}，"
        f"csg_logits_delta_abs_mean={format_metric(dynamic_means.get('csg_logits_delta_abs_mean'))}。",
        "9. learned_dynamic_csg_alpha 是否在多个 fold 上非零："
        f" {'是' if decision_payload.get('alpha_nonzero_multi_fold') else '否'}，"
        f"folds={decision_payload.get('alpha_nonzero_folds', [])}。",
        "10. full / concept_only / visual_only / csg_only 的 5-fold branch 表现如何：",
    ]

    if not full_row.empty:
        lines.append(
            f"   full: ACC {format_metric(full_row.iloc[0]['ACC'])}, AUC {format_metric(full_row.iloc[0]['AUC'])}, F1 {format_metric(full_row.iloc[0]['F1'])}。"
        )
    else:
        lines.append("   full: NA。")
    if not concept_row.empty:
        lines.append(
            f"   concept_only: ACC {format_metric(concept_row.iloc[0]['ACC'])}, AUC {format_metric(concept_row.iloc[0]['AUC'])}, F1 {format_metric(concept_row.iloc[0]['F1'])}。"
        )
    else:
        lines.append("   concept_only: NA。")
    if not visual_row.empty:
        lines.append(
            f"   visual_only: ACC {format_metric(visual_row.iloc[0]['ACC'])}, AUC {format_metric(visual_row.iloc[0]['AUC'])}, F1 {format_metric(visual_row.iloc[0]['F1'])}。"
        )
    else:
        lines.append("   visual_only: NA。")
    if not csg_row.empty:
        lines.append(
            f"   csg_only: ACC {format_metric(csg_row.iloc[0]['ACC'])}, AUC {format_metric(csg_row.iloc[0]['AUC'])}, F1 {format_metric(csg_row.iloc[0]['F1'])}。"
        )
    else:
        lines.append("   csg_only: NA。")

    lines.extend(
        [
            "11. Dynamic CSG 是否应进入新的候选主模型："
            f" {'是' if decision_payload.get('decision') == 'candidate_primary' else '否'}"
            f"（decision={decision_payload.get('decision', 'pending')}）。",
            "12. 是否应停止 Dynamic CSG 并转向 Step60："
            f" {'是' if decision_payload.get('decision') == 'stop_dynamic_csg_move_to_step60' else '否'}。",
            "",
            "## Status",
            "",
            f"- Stage57C baseline dir: `{relative_path_str(ROOT, stage57c_dir)}`",
            f"- Stage58C baseline dir: `{relative_path_str(ROOT, stage58c_dir)}`",
            f"- Stage59C candidate dir: `{relative_path_str(ROOT, stage59c_dir)}`",
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

    stage57c_metrics, _ = discover_aggregate_metrics(args.stage57c_dir)
    stage57c_fold_df, _ = discover_fold_metrics(args.stage57c_dir)
    stage58c_metrics, _ = discover_aggregate_metrics(args.stage58c_dir)
    stage58c_fold_df, _ = discover_fold_metrics(args.stage58c_dir)
    stage59c_metrics, stage59c_agg_path = discover_aggregate_metrics(args.stage59c_dir)
    stage59c_fold_df, stage59c_fold_path = discover_fold_metrics(args.stage59c_dir)
    training_status, checkpoint_folds = determine_run_status(args.stage59c_dir, stage59c_fold_df)

    completed_folds = collect_completed_checkpoint_folds(args.stage59c_dir)
    if not completed_folds:
        completed_folds = sorted(set(checkpoint_folds))

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
            "csg_ratio_median",
            "full_margin_mean",
            "concept_margin_mean",
            "visual_margin_mean",
            "csg_margin_mean",
        ]
    )
    dynamic_df = pd.DataFrame(
        columns=[
            "fold",
            "learned_alpha_final",
            "dynamic_csg_enabled",
            "dynamic_csg_scale",
            "dynamic_csg_norm",
            "dynamic_csg_clip",
            "dynamic_delta_mean",
            "dynamic_delta_std",
            "dynamic_adj_mean",
            "dynamic_adj_std",
            "static_csg_logits_mean",
            "dynamic_csg_logits_mean",
            "csg_logits_delta_mean",
            "csg_logits_delta_abs_mean",
            "status",
            "anomaly_count",
        ]
    )
    audit_status: dict[str, object] = {}
    warnings: list[str] = []

    if completed_folds:
        branch_df, contribution_df, dynamic_df, audit_status, warnings = collect_audits_and_dynamic(
            run_dir=args.stage59c_dir,
            fold_ids=completed_folds,
            output_dir=output_dir,
            audit_script=args.audit_script,
        )

    contribution_means = summarize_contribution_means(contribution_df)
    dynamic_means = summarize_dynamic_means(dynamic_df)
    branch_mean_df = summarize_branch_means(branch_df)
    compare_df = build_compare_df(
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage59c_metrics=stage59c_metrics,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
        contribution_means=contribution_means,
        dynamic_means=dynamic_means,
    )
    fold_metrics_df = build_fold_metrics_df(
        stage57c_fold_df=stage57c_fold_df,
        stage58c_fold_df=stage58c_fold_df,
        stage59c_fold_df=stage59c_fold_df,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
    )

    if not contribution_df.empty:
        step58c_contrib_path = args.stage58c_dir.parent / "stage58C_contribution_by_fold.csv"
        step58c_dyn_path = args.stage58c_dir.parent / "stage59C_placeholder.csv"
        step58c_contrib_df = read_csv_if_exists(step58c_contrib_path)
        if step58c_contrib_df is not None:
            step58c_contrib_map = {int(row["fold"]): row.to_dict() for _, row in step58c_contrib_df.iterrows()}
            contribution_df["delta_visual_ratio_vs_step58C"] = contribution_df.apply(
                lambda row: None
                if int(row["fold"]) not in step58c_contrib_map or pd.isna(row["visual_ratio_mean"])
                else float(row["visual_ratio_mean"]) - to_float(step58c_contrib_map[int(row["fold"])].get("visual_ratio_mean")),
                axis=1,
            )
            contribution_df["delta_concept_ratio_vs_step58C"] = contribution_df.apply(
                lambda row: None
                if int(row["fold"]) not in step58c_contrib_map or pd.isna(row["concept_ratio_mean"])
                else float(row["concept_ratio_mean"]) - to_float(step58c_contrib_map[int(row["fold"])].get("concept_ratio_mean")),
                axis=1,
            )
            contribution_df["delta_csg_ratio_vs_step58C"] = contribution_df.apply(
                lambda row: None
                if int(row["fold"]) not in step58c_contrib_map or pd.isna(row["csg_ratio_mean"])
                else float(row["csg_ratio_mean"]) - to_float(step58c_contrib_map[int(row["fold"])].get("csg_ratio_mean")),
                axis=1,
            )

    decision_payload = decide_outcome(
        stage58c_metrics=stage58c_metrics,
        stage59c_metrics=stage59c_metrics,
        stage58c_fold_df=stage58c_fold_df,
        stage59c_fold_df=stage59c_fold_df,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
        dynamic_means=dynamic_means,
        dynamic_df=dynamic_df,
    )
    summary_text = build_summary_md(
        stage58c_metrics=stage58c_metrics,
        stage59c_metrics=stage59c_metrics,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
        training_status=training_status,
        completed_folds=completed_folds,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
        dynamic_means=dynamic_means,
        decision_payload=decision_payload,
    )

    compare_df.to_csv(output_dir / "stage59C_compare_with_baselines.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "stage59C_fold_metrics.csv", index=False)
    branch_df.to_csv(output_dir / "stage59C_branch_metrics_by_fold.csv", index=False)
    contribution_df.to_csv(output_dir / "stage59C_contribution_by_fold.csv", index=False)
    dynamic_df.to_csv(output_dir / "stage59C_dynamic_csg_by_fold.csv", index=False)
    (output_dir / "stage59C_summary.md").write_text(summary_text, encoding="utf-8")
    (output_dir / "stage59C_decision.json").write_text(
        json.dumps(
            {
                **decision_payload,
                "stage57c_baseline_metrics": {
                    key: round_or_none(value) for key, value in stage57c_metrics.items()
                },
                "stage58c_baseline_metrics": {
                    key: round_or_none(value) for key, value in stage58c_metrics.items()
                },
                "stage59c_metrics": {
                    key: round_or_none(value) for key, value in stage59c_metrics.items()
                },
                "contribution_means": {
                    key: round_or_none(value) for key, value in contribution_means.items()
                },
                "dynamic_means": {
                    key: round_or_none(value) if key != "dynamic_csg_norm" else value
                    for key, value in dynamic_means.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "stage59C_status.json").write_text(
        json.dumps(
            {
                "status": training_status,
                "completed_folds": completed_folds,
                "stage59c_aggregate_source": None
                if stage59c_agg_path is None
                else relative_path_str(ROOT, stage59c_agg_path),
                "stage59c_fold_source": None
                if stage59c_fold_path is None
                else relative_path_str(ROOT, stage59c_fold_path),
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
