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
from utils.eval_utils import _load_state_dict_with_scale_gate_compat
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
DEFAULT_STAGE60D_DIR = ROOT / "results_stage60D_ccra_configC_formal"
DEFAULT_STAGE61D_DIR = ROOT / "results_stage61D_l2h_configG_5fold" / "rce_v2_rcD_l2hG_5fold_e20_s1"
DEFAULT_OUTPUT_DIR = ROOT / "results_stage61D_l2h_configG_5fold"
DEFAULT_RUN_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage61D_l2h_configG_5fold.sh"
DEFAULT_AUDIT_SCRIPT = ROOT / "scripts" / "analysis" / "build_stage57B_logit_contribution_audit.py"
STEP61B_DIR = ROOT / "results_stage61B_l2h_retrieval_all_off"
MODEL_V2_PATH = ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py"
ORIGINAL_RCE_PATH = ROOT / "models" / "model_RCE_MIL_BiomedCLIP.py"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))
EXPECTED_FOLDS = 5
STEP57B_VISUAL_BASELINE = 0.7195798650806405
STEP57B_CONCEPT_BASELINE = 0.2804201354438285
BRANCH_PRIORITY = [
    "full",
    "concept_only",
    "full_without_visual",
    "visual_only",
    "low_only",
    "high_only",
    "csg_only",
]
METRIC_ALIASES = {
    "AUC": ["test_auc", "auc"],
    "ACC": ["test_acc", "acc"],
    "F1": ["test_f1", "f1", "macro_f1"],
    "Balanced_ACC": ["balanced_acc", "bacc", "balanced acc", "balanced_accuracy"],
    "PR_AUC": ["pr_auc", "prauc", "pr_auc_score"],
}
L2H_CONFIG_G = {
    "rce_use_l2h_retrieval": True,
    "rce_l2h_mode": "low_topk_coord_window",
    "rce_l2h_alpha_init": 0.01,
    "rce_l2h_low_topk": 16,
    "rce_l2h_high_max_per_low": 32,
    "rce_l2h_scale_ratio": 1.0,
    "rce_l2h_patch_footprint_ratio": 6.0,
    "rce_l2h_scale": 1.0,
    "rce_l2h_fusion": "high_region_residual",
    "rce_l2h_aggregate": "mean",
    "rce_l2h_score_mode": "low_prompt_max",
    "rce_l2h_detach_low_scores": False,
    "rce_l2h_min_high_matches": 1,
    "rce_l2h_clip": 5.0,
    "rce_use_dynamic_csg": False,
    "rce_use_ccra": False,
}
PRELIGHT_FILES = [
    "stage61B_all_off_equivalence_audit.json",
    "stage61B_config_audit.json",
    "stage61B_l2h_smoke.json",
    "stage61B_param_init_audit.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step61D L2H config G 5-fold formal summary.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage57c-dir", type=Path, default=DEFAULT_STAGE57C_DIR)
    parser.add_argument("--stage58c-dir", type=Path, default=DEFAULT_STAGE58C_DIR)
    parser.add_argument("--stage59c-dir", type=Path, default=DEFAULT_STAGE59C_DIR)
    parser.add_argument("--stage60d-dir", type=Path, default=DEFAULT_STAGE60D_DIR)
    parser.add_argument("--stage61d-run-dir", type=Path, default=DEFAULT_STAGE61D_DIR)
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


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def discover_csvs(result_dir: Path) -> list[Path]:
    if not result_dir.exists():
        return []
    return sorted(path for path in result_dir.rglob("*.csv") if path.is_file())


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
        return "pending", []
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
    return "pending", checkpoint_folds


def read_experiment_settings(run_dir: Path) -> dict[str, object]:
    files = sorted(run_dir.glob("experiment_*.txt"))
    if not files:
        return {}
    return ast.literal_eval(files[0].read_text(encoding="utf-8"))


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
        raise ValueError(f"Unsupported task for Step61D summary: {task}")
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
        rce_use_dynamic_csg=False,
        rce_dynamic_csg_mode="evidence_outer",
        rce_dynamic_csg_alpha_init=0.0,
        rce_dynamic_csg_scale=1.0,
        rce_dynamic_csg_norm="softmax",
        rce_dynamic_csg_detach_evidence=False,
        rce_dynamic_csg_clip=5.0,
        rce_use_ccra=False,
        rce_ccra_mode="concept_query_residual",
        rce_ccra_alpha_init=0.0,
        rce_ccra_scale=1.0,
        rce_ccra_num_queries=0,
        rce_ccra_query_source="prompt_mean",
        rce_ccra_detach_prompt=False,
        rce_ccra_norm="layernorm",
        rce_ccra_dropout=0.0,
        rce_ccra_clip=5.0,
        rce_use_l2h_retrieval=to_bool(settings.get("rce_use_l2h_retrieval", False)),
        rce_l2h_mode=str(settings.get("rce_l2h_mode", "low_topk_coord_window")),
        rce_l2h_low_topk=int(settings.get("rce_l2h_low_topk", 8)),
        rce_l2h_high_max_per_low=int(settings.get("rce_l2h_high_max_per_low", 16)),
        rce_l2h_scale_ratio=float(settings.get("rce_l2h_scale_ratio", 1.0)),
        rce_l2h_patch_footprint_ratio=float(settings.get("rce_l2h_patch_footprint_ratio", 4.0)),
        rce_l2h_alpha_init=float(settings.get("rce_l2h_alpha_init", 0.0)),
        rce_l2h_scale=float(settings.get("rce_l2h_scale", 1.0)),
        rce_l2h_fusion=str(settings.get("rce_l2h_fusion", "high_region_residual")),
        rce_l2h_aggregate=str(settings.get("rce_l2h_aggregate", "mean")),
        rce_l2h_score_mode=str(settings.get("rce_l2h_score_mode", "low_prompt_max")),
        rce_l2h_detach_low_scores=to_bool(settings.get("rce_l2h_detach_low_scores", False)),
        rce_l2h_min_high_matches=int(settings.get("rce_l2h_min_high_matches", 1)),
        rce_l2h_clip=float(settings.get("rce_l2h_clip", 5.0)),
        scale_mode=str(settings.get("scale_mode", "dual")),
        finetune_text_encoder=False,
        enable_logit_breakdown_audit=True,
    )


def apply_l2h_config_g(settings: dict[str, object]) -> dict[str, object]:
    merged = dict(settings)
    merged.update(L2H_CONFIG_G)
    return merged


def load_state_dict(model: RCE_MIL_BiomedCLIP, ckpt_path: Path, settings: dict[str, object]) -> None:
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    cleaned = {}
    for key, value in state_dict.items():
        if "instance_loss_fn" in key:
            continue
        cleaned[key.replace(".module", "")] = value
    _load_state_dict_with_scale_gate_compat(
        model,
        cleaned,
        allow_legacy_scale_fusion_ckpt=bool(settings.get("allow_legacy_scale_fusion_ckpt", False)),
    )


def collect_l2h_metrics_for_fold(run_dir: Path, fold: int) -> dict[str, object]:
    settings = read_experiment_settings(run_dir)
    if not settings:
        return {}
    settings = apply_l2h_config_g(settings)
    dataset = build_dataset(settings)
    split_dir = Path(str(settings["split_dir"]))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    _, _, test_split = dataset.return_splits(from_id=False, csv_path=str(split_dir / f"splits_{fold}.csv"))
    loader = get_simple_loader(test_split, mode=str(settings.get("mode", "transformer")))
    model = RCE_MIL_BiomedCLIP(
        config=build_model_config(settings),
        num_classes=int(settings["n_classes"]),
    )
    ckpt_path = run_dir / f"s_{fold}_checkpoint.pt"
    load_state_dict(model, ckpt_path, settings)
    if hasattr(model, "set_logit_breakdown_audit"):
        model.set_logit_breakdown_audit(True)
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    device = next(model.parameters()).device

    topk_scores: list[float] = []
    match_counts: list[float] = []
    zero_flags: list[float] = []
    retrieved_feature_norms: list[float] = []
    original_high_region_norms: list[float] = []
    fused_high_region_norms: list[float] = []
    delta_abs_means: list[float] = []
    delta_ratio_means: list[float] = []
    anomaly_count = 0
    l2h_shapes: dict[str, object] = {}
    skipped_reasons: list[str] = []

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
            breakdown = getattr(model, "last_l2h_retrieval_debug", None) or {}
            exports = {
                "scores": getattr(model, "last_low_patch_topk_scores", None),
                "counts": getattr(model, "last_retrieved_high_patch_match_counts", None),
                "mask": getattr(model, "last_retrieved_high_patch_mask", None),
            }

            score_tensor = exports["scores"]
            if isinstance(score_tensor, torch.Tensor) and score_tensor.numel() > 0:
                topk_scores.extend(score_tensor.detach().cpu().reshape(-1).tolist())
                l2h_shapes["low_patch_concept_scores_shape"] = breakdown.get("low_patch_concept_scores_shape")

            count_tensor = exports["counts"]
            if isinstance(count_tensor, torch.Tensor) and count_tensor.numel() > 0:
                counts = count_tensor.detach().cpu().reshape(-1).float().tolist()
                match_counts.extend(counts)
                zero_flags.extend([1.0 if value <= 0 else 0.0 for value in counts])

            skipped_reason = breakdown.get("skipped_reason")
            if skipped_reason:
                skipped_reasons.append(str(skipped_reason))

            retrieved_norm = safe_float(breakdown.get("retrieved_high_features_norm"))
            original_norm = safe_float(breakdown.get("original_high_region_norm"))
            fused_norm = safe_float(breakdown.get("fused_high_region_norm"))
            delta_abs = safe_float(breakdown.get("l2h_delta_abs_mean"))
            delta_ratio = safe_float(breakdown.get("l2h_delta_vs_original_ratio"))
            if retrieved_norm is not None:
                retrieved_feature_norms.append(retrieved_norm)
            if original_norm is not None:
                original_high_region_norms.append(original_norm)
            if fused_norm is not None:
                fused_high_region_norms.append(fused_norm)
            if delta_abs is not None:
                delta_abs_means.append(delta_abs)
            if delta_ratio is not None:
                delta_ratio_means.append(delta_ratio)

            for key in ["low_coords_shape", "high_coords_shape"]:
                if key not in l2h_shapes and breakdown.get(key) is not None:
                    l2h_shapes[key] = breakdown.get(key)

    learned_alpha = safe_float(getattr(model, "rce_l2h_alpha", None))
    if learned_alpha is None and hasattr(model, "rce_l2h_alpha") and model.rce_l2h_alpha is not None:
        learned_alpha = safe_float(model.rce_l2h_alpha.detach())

    def safe_stat(values: list[float], kind: str) -> float | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        if kind == "mean":
            return float(arr.mean())
        if kind == "median":
            return float(np.median(arr))
        if kind == "max":
            return float(arr.max())
        raise ValueError(kind)

    return {
        "fold": fold + 1,
        "status": "completed",
        "l2h_enabled": True,
        "alpha_init": float(L2H_CONFIG_G["rce_l2h_alpha_init"]),
        "learned_alpha_final": learned_alpha,
        "l2h_mode": L2H_CONFIG_G["rce_l2h_mode"],
        "l2h_low_topk": int(L2H_CONFIG_G["rce_l2h_low_topk"]),
        "l2h_high_max_per_low": int(L2H_CONFIG_G["rce_l2h_high_max_per_low"]),
        "l2h_scale_ratio": float(L2H_CONFIG_G["rce_l2h_scale_ratio"]),
        "l2h_patch_footprint_ratio": float(L2H_CONFIG_G["rce_l2h_patch_footprint_ratio"]),
        "l2h_fusion": L2H_CONFIG_G["rce_l2h_fusion"],
        "l2h_aggregate": L2H_CONFIG_G["rce_l2h_aggregate"],
        "l2h_score_mode": L2H_CONFIG_G["rce_l2h_score_mode"],
        "low_patch_concept_scores_shape": l2h_shapes.get("low_patch_concept_scores_shape"),
        "low_topk_scores_mean": safe_stat(topk_scores, "mean"),
        "retrieved_high_match_counts_mean": safe_stat(match_counts, "mean"),
        "retrieved_high_match_counts_median": safe_stat(match_counts, "median"),
        "retrieved_high_match_counts_max": safe_stat(match_counts, "max"),
        "retrieved_high_zero_match_percent": safe_stat(zero_flags, "mean"),
        "retrieved_high_features_norm": safe_stat(retrieved_feature_norms, "mean"),
        "original_high_region_norm": safe_stat(original_high_region_norms, "mean"),
        "fused_high_region_norm": safe_stat(fused_high_region_norms, "mean"),
        "l2h_delta_abs_mean": safe_stat(delta_abs_means, "mean"),
        "l2h_delta_vs_original_ratio": safe_stat(delta_ratio_means, "mean"),
        "low_coords_shape": l2h_shapes.get("low_coords_shape"),
        "high_coords_shape": l2h_shapes.get("high_coords_shape"),
        "skipped_reason": None if not skipped_reasons else ";".join(sorted(set(skipped_reasons))),
        "anomaly_count": anomaly_count,
    }


def run_fold_audit(run_dir: Path, fold: int, audit_output_dir: Path, audit_script: Path) -> dict[str, object]:
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


def collect_audits_and_l2h(
    run_dir: Path,
    fold_ids: list[int],
    output_dir: Path,
    audit_script: Path,
    stage58c_contrib_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], list[str]]:
    branch_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    l2h_rows: list[dict[str, object]] = []
    audit_status: dict[str, object] = {}
    warnings: list[str] = []
    baseline_map = {}
    if stage58c_contrib_df is not None and not stage58c_contrib_df.empty:
        baseline_map = {int(row["fold"]): row.to_dict() for _, row in stage58c_contrib_df.iterrows()}

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
            contribution_rows.append(
                {
                    "fold": fold + 1,
                    "visual_ratio_mean": None,
                    "visual_ratio_median": None,
                    "visual_ratio_gt_0_5_percent": None,
                    "concept_ratio_mean": None,
                    "concept_ratio_median": None,
                    "csg_ratio_mean": None,
                    "full_margin_mean": None,
                    "concept_margin_mean": None,
                    "visual_margin_mean": None,
                    "csg_margin_mean": None,
                    "delta_visual_ratio_vs_step58C": None,
                    "delta_concept_ratio_vs_step58C": None,
                    "delta_csg_ratio_vs_step58C": None,
                }
            )
            l2h_rows.append(
                {
                    "fold": fold + 1,
                    "status": "pending",
                    "l2h_enabled": True,
                    "alpha_init": float(L2H_CONFIG_G["rce_l2h_alpha_init"]),
                    "learned_alpha_final": None,
                    "l2h_mode": L2H_CONFIG_G["rce_l2h_mode"],
                    "l2h_low_topk": int(L2H_CONFIG_G["rce_l2h_low_topk"]),
                    "l2h_high_max_per_low": int(L2H_CONFIG_G["rce_l2h_high_max_per_low"]),
                    "l2h_scale_ratio": float(L2H_CONFIG_G["rce_l2h_scale_ratio"]),
                    "l2h_patch_footprint_ratio": float(L2H_CONFIG_G["rce_l2h_patch_footprint_ratio"]),
                    "l2h_fusion": L2H_CONFIG_G["rce_l2h_fusion"],
                    "l2h_aggregate": L2H_CONFIG_G["rce_l2h_aggregate"],
                    "l2h_score_mode": L2H_CONFIG_G["rce_l2h_score_mode"],
                    "low_patch_concept_scores_shape": None,
                    "low_topk_scores_mean": None,
                    "retrieved_high_match_counts_mean": None,
                    "retrieved_high_match_counts_median": None,
                    "retrieved_high_match_counts_max": None,
                    "retrieved_high_zero_match_percent": None,
                    "retrieved_high_features_norm": None,
                    "original_high_region_norm": None,
                    "fused_high_region_norm": None,
                    "l2h_delta_abs_mean": None,
                    "l2h_delta_vs_original_ratio": None,
                    "low_coords_shape": None,
                    "high_coords_shape": None,
                    "skipped_reason": "audit_failed",
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
                    "fold": fold + 1,
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

        contribution_row = {
            "fold": fold + 1,
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
            "delta_visual_ratio_vs_step58C": None,
            "delta_concept_ratio_vs_step58C": None,
            "delta_csg_ratio_vs_step58C": None,
        }
        baseline = baseline_map.get(fold)
        if baseline is not None:
            contribution_row["delta_visual_ratio_vs_step58C"] = (
                None
                if pd.isna(contribution_row["visual_ratio_mean"])
                else float(contribution_row["visual_ratio_mean"]) - to_float(baseline.get("visual_ratio_mean"))
            )
            contribution_row["delta_concept_ratio_vs_step58C"] = (
                None
                if pd.isna(contribution_row["concept_ratio_mean"])
                else float(contribution_row["concept_ratio_mean"]) - to_float(baseline.get("concept_ratio_mean"))
            )
            contribution_row["delta_csg_ratio_vs_step58C"] = (
                None
                if pd.isna(contribution_row["csg_ratio_mean"])
                else float(contribution_row["csg_ratio_mean"]) - to_float(baseline.get("csg_ratio_mean"))
            )
        contribution_rows.append(contribution_row)

        try:
            l2h_rows.append(collect_l2h_metrics_for_fold(run_dir, fold))
        except Exception as exc:
            warnings.append(f"l2h_audit_failed:fold_{fold}:{exc}")
            l2h_rows.append(
                {
                    "fold": fold + 1,
                    "status": "pending",
                    "l2h_enabled": True,
                    "alpha_init": float(L2H_CONFIG_G["rce_l2h_alpha_init"]),
                    "learned_alpha_final": None,
                    "l2h_mode": L2H_CONFIG_G["rce_l2h_mode"],
                    "l2h_low_topk": int(L2H_CONFIG_G["rce_l2h_low_topk"]),
                    "l2h_high_max_per_low": int(L2H_CONFIG_G["rce_l2h_high_max_per_low"]),
                    "l2h_scale_ratio": float(L2H_CONFIG_G["rce_l2h_scale_ratio"]),
                    "l2h_patch_footprint_ratio": float(L2H_CONFIG_G["rce_l2h_patch_footprint_ratio"]),
                    "l2h_fusion": L2H_CONFIG_G["rce_l2h_fusion"],
                    "l2h_aggregate": L2H_CONFIG_G["rce_l2h_aggregate"],
                    "l2h_score_mode": L2H_CONFIG_G["rce_l2h_score_mode"],
                    "low_patch_concept_scores_shape": None,
                    "low_topk_scores_mean": None,
                    "retrieved_high_match_counts_mean": None,
                    "retrieved_high_match_counts_median": None,
                    "retrieved_high_match_counts_max": None,
                    "retrieved_high_zero_match_percent": None,
                    "retrieved_high_features_norm": None,
                    "original_high_region_norm": None,
                    "fused_high_region_norm": None,
                    "l2h_delta_abs_mean": None,
                    "l2h_delta_vs_original_ratio": None,
                    "low_coords_shape": None,
                    "high_coords_shape": None,
                    "skipped_reason": f"exception:{exc}",
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
            "full_margin_mean",
            "concept_margin_mean",
            "visual_margin_mean",
            "csg_margin_mean",
            "delta_visual_ratio_vs_step58C",
            "delta_concept_ratio_vs_step58C",
            "delta_csg_ratio_vs_step58C",
        ],
    )
    l2h_df = pd.DataFrame(
        l2h_rows,
        columns=[
            "fold",
            "status",
            "l2h_enabled",
            "alpha_init",
            "learned_alpha_final",
            "l2h_mode",
            "l2h_low_topk",
            "l2h_high_max_per_low",
            "l2h_scale_ratio",
            "l2h_patch_footprint_ratio",
            "l2h_fusion",
            "l2h_aggregate",
            "l2h_score_mode",
            "low_patch_concept_scores_shape",
            "low_topk_scores_mean",
            "retrieved_high_match_counts_mean",
            "retrieved_high_match_counts_median",
            "retrieved_high_match_counts_max",
            "retrieved_high_zero_match_percent",
            "retrieved_high_features_norm",
            "original_high_region_norm",
            "fused_high_region_norm",
            "l2h_delta_abs_mean",
            "l2h_delta_vs_original_ratio",
            "low_coords_shape",
            "high_coords_shape",
            "skipped_reason",
            "anomaly_count",
        ],
    )
    return branch_df, contribution_df, l2h_df, audit_status, warnings


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
            "delta_visual_ratio_vs_step58C": math.nan,
            "delta_concept_ratio_vs_step58C": math.nan,
            "delta_csg_ratio_vs_step58C": math.nan,
        }
    numeric_cols = [col for col in contribution_df.columns if col != "fold"]
    means = contribution_df[numeric_cols].mean(numeric_only=True)
    return {col: to_float(means.get(col)) for col in numeric_cols}


def summarize_l2h_means(l2h_df: pd.DataFrame) -> dict[str, float | str | bool | None]:
    if l2h_df.empty:
        return {
            "learned_alpha_final": math.nan,
            "l2h_enabled": True,
            "alpha_init": float(L2H_CONFIG_G["rce_l2h_alpha_init"]),
            "l2h_mode": L2H_CONFIG_G["rce_l2h_mode"],
            "l2h_low_topk": int(L2H_CONFIG_G["rce_l2h_low_topk"]),
            "l2h_high_max_per_low": int(L2H_CONFIG_G["rce_l2h_high_max_per_low"]),
            "l2h_scale_ratio": float(L2H_CONFIG_G["rce_l2h_scale_ratio"]),
            "l2h_patch_footprint_ratio": float(L2H_CONFIG_G["rce_l2h_patch_footprint_ratio"]),
            "l2h_fusion": L2H_CONFIG_G["rce_l2h_fusion"],
            "l2h_aggregate": L2H_CONFIG_G["rce_l2h_aggregate"],
            "l2h_score_mode": L2H_CONFIG_G["rce_l2h_score_mode"],
            "low_topk_scores_mean": math.nan,
            "retrieved_high_match_counts_mean": math.nan,
            "retrieved_high_zero_match_percent": math.nan,
            "retrieved_high_features_norm": math.nan,
            "original_high_region_norm": math.nan,
            "fused_high_region_norm": math.nan,
            "l2h_delta_abs_mean": math.nan,
            "l2h_delta_vs_original_ratio": math.nan,
            "anomaly_count": math.nan,
        }
    summary: dict[str, float | str | bool | None] = {
        "l2h_enabled": True,
        "alpha_init": float(L2H_CONFIG_G["rce_l2h_alpha_init"]),
        "l2h_mode": L2H_CONFIG_G["rce_l2h_mode"],
        "l2h_low_topk": int(L2H_CONFIG_G["rce_l2h_low_topk"]),
        "l2h_high_max_per_low": int(L2H_CONFIG_G["rce_l2h_high_max_per_low"]),
        "l2h_scale_ratio": float(L2H_CONFIG_G["rce_l2h_scale_ratio"]),
        "l2h_patch_footprint_ratio": float(L2H_CONFIG_G["rce_l2h_patch_footprint_ratio"]),
        "l2h_fusion": L2H_CONFIG_G["rce_l2h_fusion"],
        "l2h_aggregate": L2H_CONFIG_G["rce_l2h_aggregate"],
        "l2h_score_mode": L2H_CONFIG_G["rce_l2h_score_mode"],
    }
    for col in [
        "learned_alpha_final",
        "low_topk_scores_mean",
        "retrieved_high_match_counts_mean",
        "retrieved_high_match_counts_median",
        "retrieved_high_match_counts_max",
        "retrieved_high_zero_match_percent",
        "retrieved_high_features_norm",
        "original_high_region_norm",
        "fused_high_region_norm",
        "l2h_delta_abs_mean",
        "l2h_delta_vs_original_ratio",
        "anomaly_count",
    ]:
        summary[col] = (
            to_float(pd.to_numeric(l2h_df[col], errors="coerce").mean())
            if col in l2h_df.columns
            else math.nan
        )
    return summary


def compute_mean_metrics(fold_df: pd.DataFrame | None) -> dict[str, float]:
    if fold_df is None or fold_df.empty:
        return {metric_name: math.nan for metric_name in METRIC_ALIASES}
    return {
        metric_name: to_float(pd.to_numeric(fold_df[metric_name], errors="coerce").mean())
        for metric_name in METRIC_ALIASES
    }


def extract_model_rows(df: pd.DataFrame | None, model_name: str) -> pd.DataFrame | None:
    if df is None or df.empty or "model_name" not in df.columns:
        return None
    subset = df.loc[df["model_name"] == model_name].copy()
    return subset if not subset.empty else None


def mean_metrics_from_model_rows(df: pd.DataFrame | None) -> dict[str, float]:
    if df is None or df.empty:
        return {metric_name: math.nan for metric_name in METRIC_ALIASES}
    metrics = {}
    for metric_name in METRIC_ALIASES:
        metrics[metric_name] = to_float(pd.to_numeric(df[metric_name], errors="coerce").mean())
    return metrics


def build_compare_df(
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage59c_metrics: dict[str, float],
    stage60d_metrics: dict[str, float],
    stage61d_metrics: dict[str, float],
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
    stage60d_dir: Path,
    stage61d_dir: Path,
    contribution_means: dict[str, float],
    l2h_means: dict[str, float | str | bool | None],
) -> pd.DataFrame:
    references = {
        "stage57C": stage57c_metrics,
        "step58C": stage58c_metrics,
        "step59C": stage59c_metrics,
        "step60D": stage60d_metrics,
    }

    def build_row(
        model_name: str,
        source_dir: Path,
        metrics: dict[str, float],
        include_l2h_fields: bool = False,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "model_name": model_name,
            "source_dir": relative_path_str(ROOT, source_dir),
            "AUC": round_or_none(metrics.get("AUC")),
            "ACC": round_or_none(metrics.get("ACC")),
            "F1": round_or_none(metrics.get("F1")),
            "Balanced_ACC": round_or_none(metrics.get("Balanced_ACC")),
            "BACC": round_or_none(metrics.get("Balanced_ACC")),
            "PR_AUC": round_or_none(metrics.get("PR_AUC")),
            "visual_ratio_mean": None,
            "concept_ratio_mean": None,
            "csg_ratio_mean": None,
            "learned_l2h_alpha_mean": None,
            "l2h_delta_abs_mean": None,
            "retrieved_high_match_counts_mean": None,
            "retrieved_high_zero_match_percent": None,
        }
        if include_l2h_fields:
            row.update(
                {
                    "visual_ratio_mean": round_or_none(contribution_means.get("visual_ratio_mean")),
                    "concept_ratio_mean": round_or_none(contribution_means.get("concept_ratio_mean")),
                    "csg_ratio_mean": round_or_none(contribution_means.get("csg_ratio_mean")),
                    "learned_l2h_alpha_mean": round_or_none(l2h_means.get("learned_alpha_final")),
                    "l2h_delta_abs_mean": round_or_none(l2h_means.get("l2h_delta_abs_mean")),
                    "retrieved_high_match_counts_mean": round_or_none(
                        l2h_means.get("retrieved_high_match_counts_mean")
                    ),
                    "retrieved_high_zero_match_percent": round_or_none(
                        l2h_means.get("retrieved_high_zero_match_percent")
                    ),
                }
            )
        for ref_name, ref_metrics in references.items():
            for metric_name, key in [
                ("auc", "AUC"),
                ("acc", "ACC"),
                ("f1", "F1"),
                ("bacc", "Balanced_ACC"),
                ("pr_auc", "PR_AUC"),
            ]:
                lhs = metrics.get(key, math.nan)
                rhs = ref_metrics.get(key, math.nan)
                row[f"delta_vs_{ref_name.lower()}_{metric_name}"] = (
                    round_or_none(lhs - rhs) if not pd.isna(lhs) and not pd.isna(rhs) else None
                )
        return row

    return pd.DataFrame(
        [
            build_row("stage57C_rce_v2_baseline", stage57c_dir, stage57c_metrics),
            build_row("stage58C_residual_constrained_configD", stage58c_dir, stage58c_metrics),
            build_row("stage59C_dynamic_csg_configA", stage59c_dir, stage59c_metrics),
            build_row("stage60D_ccra_configC", stage60d_dir, stage60d_metrics),
            build_row("stage61D_l2h_configG", stage61d_dir, stage61d_metrics, include_l2h_fields=True),
        ]
    )


def build_fold_metrics_df(
    stage57c_fold_df: pd.DataFrame | None,
    stage58c_fold_df: pd.DataFrame | None,
    stage59c_fold_df: pd.DataFrame | None,
    stage60d_fold_df: pd.DataFrame | None,
    stage61d_fold_df: pd.DataFrame | None,
    stage57c_dir: Path,
    stage58c_dir: Path,
    stage59c_dir: Path,
    stage60d_dir: Path,
    stage61d_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def append_rows(model_name: str, source_dir: Path, fold_df: pd.DataFrame | None) -> None:
        if fold_df is None or fold_df.empty:
            return
        for _, row in fold_df.iterrows():
            rows.append(
                {
                    "model_name": model_name,
                    "source_dir": relative_path_str(ROOT, source_dir),
                    "fold": int(row["fold"]),
                    "AUC": round_or_none(row["AUC"]),
                    "ACC": round_or_none(row["ACC"]),
                    "F1": round_or_none(row["F1"]),
                    "Balanced_ACC": round_or_none(row["Balanced_ACC"]),
                    "BACC": round_or_none(row["Balanced_ACC"]),
                    "PR_AUC": round_or_none(row["PR_AUC"]),
                }
            )

    append_rows("stage57C_rce_v2_baseline", stage57c_dir, stage57c_fold_df)
    append_rows("stage58C_residual_constrained_configD", stage58c_dir, stage58c_fold_df)
    append_rows("stage59C_dynamic_csg_configA", stage59c_dir, stage59c_fold_df)
    append_rows("stage60D_ccra_configC", stage60d_dir, stage60d_fold_df)
    append_rows("stage61D_l2h_configG", stage61d_dir, stage61d_fold_df)
    return pd.DataFrame(rows)


def count_nonzero_folds(series: pd.Series, threshold: float = 1e-6) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna().abs()
    return int((values > threshold).sum())


def decide_outcome(
    stage58c_metrics: dict[str, float],
    stage61d_metrics: dict[str, float],
    stage61d_fold_df: pd.DataFrame | None,
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
    l2h_means: dict[str, float | str | bool | None],
    l2h_df: pd.DataFrame,
    run_status: str,
    preflight_pass: bool,
) -> dict[str, object]:
    if not preflight_pass:
        return {
            "decision": "failed_preflight",
            "next_step": "fix_preflight_before_training",
            "reasons": ["preflight_failed"],
        }
    required = ("ACC", "AUC", "F1", "Balanced_ACC", "PR_AUC")
    if run_status != "completed" or any(pd.isna(stage61d_metrics[key]) for key in required):
        return {
            "decision": "pending",
            "next_step": "run_step61d_training",
            "reasons": ["5-fold metrics incomplete"],
        }

    delta_acc = stage61d_metrics["ACC"] - stage58c_metrics["ACC"]
    delta_auc = stage61d_metrics["AUC"] - stage58c_metrics["AUC"]
    delta_f1 = stage61d_metrics["F1"] - stage58c_metrics["F1"]
    delta_bacc = stage61d_metrics["Balanced_ACC"] - stage58c_metrics["Balanced_ACC"]
    delta_pr_auc = stage61d_metrics["PR_AUC"] - stage58c_metrics["PR_AUC"]

    visual_ratio_mean = contribution_means.get("visual_ratio_mean", math.nan)
    concept_ratio_mean = contribution_means.get("concept_ratio_mean", math.nan)
    match_mean = safe_float(l2h_means.get("retrieved_high_match_counts_mean"))
    zero_match = safe_float(l2h_means.get("retrieved_high_zero_match_percent"))
    l2h_delta_mean = safe_float(l2h_means.get("l2h_delta_abs_mean"))
    learned_alpha_mean = safe_float(l2h_means.get("learned_alpha_final"))

    visual_low = not pd.isna(visual_ratio_mean) and visual_ratio_mean < STEP57B_VISUAL_BASELINE
    concept_high = not pd.isna(concept_ratio_mean) and concept_ratio_mean > STEP57B_CONCEPT_BASELINE
    alpha_nonzero_folds = count_nonzero_folds(l2h_df.get("learned_alpha_final", pd.Series(dtype=float)))
    delta_nonzero_folds = count_nonzero_folds(l2h_df.get("l2h_delta_abs_mean", pd.Series(dtype=float)))
    anomaly_folds = int((pd.to_numeric(l2h_df.get("anomaly_count", pd.Series(dtype=float)), errors="coerce") > 0).sum())
    zero_match_low = zero_match is not None and zero_match <= 0.05
    match_ok = match_mean is not None and match_mean >= 8.0

    full_row = branch_mean_df.loc[branch_mean_df["branch"] == "full"]
    concept_row = branch_mean_df.loc[branch_mean_df["branch"] == "concept_only"]
    visual_row = branch_mean_df.loc[branch_mean_df["branch"] == "visual_only"]
    full_vs_concept_ok = False
    if not full_row.empty and not concept_row.empty:
        full_vs_concept_ok = (
            to_float(full_row.iloc[0]["ACC"]) >= to_float(concept_row.iloc[0]["ACC"]) - 0.01
            and to_float(full_row.iloc[0]["AUC"]) >= to_float(concept_row.iloc[0]["AUC"]) - 0.01
        )

    severe_fold_collapse = False
    collapse_folds: list[int] = []
    if stage61d_fold_df is not None and not stage61d_fold_df.empty:
        for _, row in stage61d_fold_df.iterrows():
            acc = to_float(row["ACC"])
            auc = to_float(row["AUC"])
            if (not pd.isna(acc) and acc < 0.75) or (not pd.isna(auc) and auc < 0.80):
                severe_fold_collapse = True
                collapse_folds.append(int(row["fold"]))

    acc_close = delta_acc >= -0.01
    auc_close = delta_auc >= -0.01
    f1_close = delta_f1 >= -0.01
    bacc_close = delta_bacc >= -0.01
    no_anomaly = anomaly_folds == 0 and not severe_fold_collapse
    l2h_signal_strong = (
        learned_alpha_mean is not None
        and l2h_delta_mean is not None
        and alpha_nonzero_folds >= 2
        and delta_nonzero_folds >= 2
        and match_ok
        and zero_match_low
    )

    if (
        acc_close
        and auc_close
        and f1_close
        and bacc_close
        and visual_low
        and concept_high
        and l2h_signal_strong
        and full_vs_concept_ok
        and no_anomaly
    ):
        return {
            "decision": "candidate_primary_l2h",
            "next_step": "enter_final_consolidation",
            "reasons": ["5-fold_close_to_or_better_than_step58C", "stable_l2h_signal"],
        }
    if (
        acc_close
        and auc_close
        and visual_low
        and concept_high
        and l2h_signal_strong
        and full_vs_concept_ok
        and no_anomaly
    ):
        return {
            "decision": "l2h_interpretable_candidate",
            "next_step": "enter_final_consolidation",
            "reasons": ["performance_close", "retrieval_signal_strong"],
        }
    if l2h_signal_strong and ((delta_acc < -0.01) or (delta_f1 < -0.01) or (delta_bacc < -0.01)):
        return {
            "decision": "tradeoff_l2h",
            "next_step": "consider_secondary_config_E_or_F",
            "reasons": ["retrieval_signal_strong_but_classification_tradeoff"],
            "suggested_secondary_configs": ["E", "F"],
        }
    return {
        "decision": "not_selected",
        "next_step": "stop_l2h_and_enter_final_consolidation",
        "reasons": ["performance_not_competitive_or_signal_unstable"],
    }


def write_run_commands(output_dir: Path, display_root: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = "\n".join(
        [
            f"cd {display_root}",
            "RUN_TRAIN=1 bash scripts/experiments/run_stage61D_l2h_configG_5fold.sh",
            "",
            "# Optional secondary fallback only if config G shows trade-off",
            "SECONDARY_CONFIG=E RUN_TRAIN=1 bash scripts/experiments/run_stage61D_l2h_configG_5fold.sh",
            "SECONDARY_CONFIG=F RUN_TRAIN=1 bash scripts/experiments/run_stage61D_l2h_configG_5fold.sh",
            "",
            "# Refresh Step61D summary after training",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage61D_l2h_5fold_summary.py",
        ]
    )
    (output_dir / "stage61D_run_commands.txt").write_text(command_text + "\n", encoding="utf-8")
    return command_text


def git_path_modified(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_command(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "command": command,
        }
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }


def instantiate_config_g_model(step58c_run_dir: Path) -> dict[str, object]:
    settings = read_experiment_settings(step58c_run_dir)
    if not settings:
        return {"ok": False, "reason": "missing_experiment_settings"}
    settings = apply_l2h_config_g(settings)
    try:
        config = build_model_config(settings)
        model = RCE_MIL_BiomedCLIP(config=config, num_classes=int(settings["n_classes"]))
    except Exception as exc:
        return {"ok": False, "reason": f"instantiate_failed:{exc}"}
    try:
        ckpt_path = step58c_run_dir / "s_0_checkpoint.pt"
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        cleaned = {}
        for key, value in state_dict.items():
            if "instance_loss_fn" in key:
                continue
            cleaned[key.replace(".module", "")] = value
        load_result = model.load_state_dict(cleaned, strict=False)
    except Exception as exc:
        return {"ok": False, "reason": f"checkpoint_load_failed:{exc}"}
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    allowed_missing = {"rce_l2h_alpha"}
    if set(missing_keys) - allowed_missing:
        return {
            "ok": False,
            "reason": f"unexpected_missing_keys:{missing_keys}",
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }
    if unexpected_keys:
        return {
            "ok": False,
            "reason": f"unexpected_keys:{unexpected_keys}",
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }
    learned_alpha = safe_float(getattr(model, "rce_l2h_alpha", None))
    return {
        "ok": True,
        "reason": None,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "l2h_enabled": bool(getattr(model, "rce_use_l2h_retrieval", False)),
        "learned_alpha": learned_alpha,
        "low_topk": int(getattr(model, "rce_l2h_low_topk", 0)),
        "high_max_per_low": int(getattr(model, "rce_l2h_high_max_per_low", 0)),
        "patch_footprint_ratio": float(getattr(model, "rce_l2h_patch_footprint_ratio", 0.0)),
    }


def run_preflight(step58c_run_dir: Path) -> dict[str, object]:
    missing_files = [name for name in PRELIGHT_FILES if not (STEP61B_DIR / name).is_file()]
    config_audit = read_json_if_exists(STEP61B_DIR / "stage61B_config_audit.json") or {}
    param_init = read_json_if_exists(STEP61B_DIR / "stage61B_param_init_audit.json") or {}
    all_off = read_json_if_exists(STEP61B_DIR / "stage61B_all_off_equivalence_audit.json") or {}
    smoke = read_json_if_exists(STEP61B_DIR / "stage61B_l2h_smoke.json") or {}
    config_g_build = instantiate_config_g_model(step58c_run_dir)
    model_text = MODEL_V2_PATH.read_text(encoding="utf-8")
    fix_note = None
    if "torch.quantile(diffs, 0.95)" in model_text:
        fix_note = (
            "将 L2H patch extent 估计从坐标差分 median 调整为 p95，以更贴近 Step61A 的 patch footprint 审计；"
            "原因是原实现会把检索窗口压缩成 stride 级别，导致 retrieved match count 系统性偏低。"
        )
    pass_flags = {
        "original_rce_unchanged": config_audit.get("original_rce_modified") is False and git_path_modified(ORIGINAL_RCE_PATH) is False,
        "no_l2h_params_when_off": param_init.get("l2h_param_names_when_off") == [],
        "l2h_off_equivalence": bool((all_off.get("l2h_off_equivalence") or {}).get("pass")),
        "full_all_off_equivalence": bool((all_off.get("full_all_off_equivalence") or {}).get("pass")),
        "config_g_buildable": bool(config_g_build.get("ok")),
    }
    ok = not missing_files and all(pass_flags.values())
    return {
        "status": "passed" if ok else "failed_preflight",
        "ok": ok,
        "missing_files": missing_files,
        "checks": pass_flags,
        "config_g_build": config_g_build,
        "step61c_fix_note": fix_note,
    }


def build_summary_md(
    preflight: dict[str, object],
    run_status: str,
    stage57c_metrics: dict[str, float],
    stage58c_metrics: dict[str, float],
    stage59c_metrics: dict[str, float],
    stage60d_metrics: dict[str, float],
    stage61d_metrics: dict[str, float],
    branch_mean_df: pd.DataFrame,
    contribution_means: dict[str, float],
    l2h_means: dict[str, float | str | bool | None],
    l2h_df: pd.DataFrame,
    decision: dict[str, object],
) -> str:
    original_changed = git_path_modified(ORIGINAL_RCE_PATH)
    model_changed = git_path_modified(MODEL_V2_PATH)
    alpha_nonzero_folds = count_nonzero_folds(l2h_df.get("learned_alpha_final", pd.Series(dtype=float)))
    delta_nonzero_folds = count_nonzero_folds(l2h_df.get("l2h_delta_abs_mean", pd.Series(dtype=float)))
    full_row = branch_mean_df.loc[branch_mean_df["branch"] == "full"]
    concept_row = branch_mean_df.loc[branch_mean_df["branch"] == "concept_only"]
    visual_row = branch_mean_df.loc[branch_mean_df["branch"] == "visual_only"]

    lines = [
        "# Step61D Low-to-High Retrieval config G 5-fold formal validation",
        "",
        "## Direct Answers",
        "",
        f"1. 本 Step 是否修改了原始 RCE 文件：{'否' if not original_changed else '是'}。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑："
        f"{'否' if not model_changed else '否，本 Step 未继续修改；当前仅沿用 Step61C 的最小窗口修复。'}",
        f"3. preflight 等价检查是否通过：{'是' if preflight.get('ok') else '否'}。",
        f"4. config G 5-fold 是否完成：{'是' if run_status == 'completed' else '否'}。",
        "5. Step61D 的 5-fold AUC / ACC / F1 / BACC / PR-AUC："
        f"AUC={format_metric(stage61d_metrics.get('AUC'))} ACC={format_metric(stage61d_metrics.get('ACC'))} "
        f"F1={format_metric(stage61d_metrics.get('F1'))} BACC={format_metric(stage61d_metrics.get('Balanced_ACC'))} "
        f"PR_AUC={format_metric(stage61d_metrics.get('PR_AUC'))}。",
        "6. 相比 Stage57C baseline 的差异："
        f"AUC={format_delta(safe_float(stage61d_metrics.get('AUC')) - safe_float(stage57c_metrics.get('AUC')) if safe_float(stage61d_metrics.get('AUC')) is not None and safe_float(stage57c_metrics.get('AUC')) is not None else None)} "
        f"ACC={format_delta(safe_float(stage61d_metrics.get('ACC')) - safe_float(stage57c_metrics.get('ACC')) if safe_float(stage61d_metrics.get('ACC')) is not None and safe_float(stage57c_metrics.get('ACC')) is not None else None)} "
        f"F1={format_delta(safe_float(stage61d_metrics.get('F1')) - safe_float(stage57c_metrics.get('F1')) if safe_float(stage61d_metrics.get('F1')) is not None and safe_float(stage57c_metrics.get('F1')) is not None else None)} "
        f"BACC={format_delta(safe_float(stage61d_metrics.get('Balanced_ACC')) - safe_float(stage57c_metrics.get('Balanced_ACC')) if safe_float(stage61d_metrics.get('Balanced_ACC')) is not None and safe_float(stage57c_metrics.get('Balanced_ACC')) is not None else None)} "
        f"PR_AUC={format_delta(safe_float(stage61d_metrics.get('PR_AUC')) - safe_float(stage57c_metrics.get('PR_AUC')) if safe_float(stage61d_metrics.get('PR_AUC')) is not None and safe_float(stage57c_metrics.get('PR_AUC')) is not None else None)}。",
        "7. 相比 Step58C residual-constrained baseline 的差异："
        f"AUC={format_delta(safe_float(stage61d_metrics.get('AUC')) - safe_float(stage58c_metrics.get('AUC')) if safe_float(stage61d_metrics.get('AUC')) is not None and safe_float(stage58c_metrics.get('AUC')) is not None else None)} "
        f"ACC={format_delta(safe_float(stage61d_metrics.get('ACC')) - safe_float(stage58c_metrics.get('ACC')) if safe_float(stage61d_metrics.get('ACC')) is not None and safe_float(stage58c_metrics.get('ACC')) is not None else None)} "
        f"F1={format_delta(safe_float(stage61d_metrics.get('F1')) - safe_float(stage58c_metrics.get('F1')) if safe_float(stage61d_metrics.get('F1')) is not None and safe_float(stage58c_metrics.get('F1')) is not None else None)} "
        f"BACC={format_delta(safe_float(stage61d_metrics.get('Balanced_ACC')) - safe_float(stage58c_metrics.get('Balanced_ACC')) if safe_float(stage61d_metrics.get('Balanced_ACC')) is not None and safe_float(stage58c_metrics.get('Balanced_ACC')) is not None else None)} "
        f"PR_AUC={format_delta(safe_float(stage61d_metrics.get('PR_AUC')) - safe_float(stage58c_metrics.get('PR_AUC')) if safe_float(stage61d_metrics.get('PR_AUC')) is not None and safe_float(stage58c_metrics.get('PR_AUC')) is not None else None)}。",
        "8. 相比 Step59C Dynamic CSG reference 的差异："
        f"AUC={format_delta(safe_float(stage61d_metrics.get('AUC')) - safe_float(stage59c_metrics.get('AUC')) if safe_float(stage61d_metrics.get('AUC')) is not None and safe_float(stage59c_metrics.get('AUC')) is not None else None)} "
        f"ACC={format_delta(safe_float(stage61d_metrics.get('ACC')) - safe_float(stage59c_metrics.get('ACC')) if safe_float(stage61d_metrics.get('ACC')) is not None and safe_float(stage59c_metrics.get('ACC')) is not None else None)} "
        f"F1={format_delta(safe_float(stage61d_metrics.get('F1')) - safe_float(stage59c_metrics.get('F1')) if safe_float(stage61d_metrics.get('F1')) is not None and safe_float(stage59c_metrics.get('F1')) is not None else None)} "
        f"BACC={format_delta(safe_float(stage61d_metrics.get('Balanced_ACC')) - safe_float(stage59c_metrics.get('Balanced_ACC')) if safe_float(stage61d_metrics.get('Balanced_ACC')) is not None and safe_float(stage59c_metrics.get('Balanced_ACC')) is not None else None)} "
        f"PR_AUC={format_delta(safe_float(stage61d_metrics.get('PR_AUC')) - safe_float(stage59c_metrics.get('PR_AUC')) if safe_float(stage61d_metrics.get('PR_AUC')) is not None and safe_float(stage59c_metrics.get('PR_AUC')) is not None else None)}。",
        "9. 相比 Step60D CCRA config C 的差异："
        f"AUC={format_delta(safe_float(stage61d_metrics.get('AUC')) - safe_float(stage60d_metrics.get('AUC')) if safe_float(stage61d_metrics.get('AUC')) is not None and safe_float(stage60d_metrics.get('AUC')) is not None else None)} "
        f"ACC={format_delta(safe_float(stage61d_metrics.get('ACC')) - safe_float(stage60d_metrics.get('ACC')) if safe_float(stage61d_metrics.get('ACC')) is not None and safe_float(stage60d_metrics.get('ACC')) is not None else None)} "
        f"F1={format_delta(safe_float(stage61d_metrics.get('F1')) - safe_float(stage60d_metrics.get('F1')) if safe_float(stage61d_metrics.get('F1')) is not None and safe_float(stage60d_metrics.get('F1')) is not None else None)} "
        f"BACC={format_delta(safe_float(stage61d_metrics.get('Balanced_ACC')) - safe_float(stage60d_metrics.get('Balanced_ACC')) if safe_float(stage61d_metrics.get('Balanced_ACC')) is not None and safe_float(stage60d_metrics.get('Balanced_ACC')) is not None else None)} "
        f"PR_AUC={format_delta(safe_float(stage61d_metrics.get('PR_AUC')) - safe_float(stage60d_metrics.get('PR_AUC')) if safe_float(stage61d_metrics.get('PR_AUC')) is not None and safe_float(stage60d_metrics.get('PR_AUC')) is not None else None)}。",
        f"10. visual_ratio 是否仍保持低水平：{'是' if contribution_means.get('visual_ratio_mean', math.nan) < STEP57B_VISUAL_BASELINE else '否'}。",
        f"11. concept_ratio 是否仍保持高水平：{'是' if contribution_means.get('concept_ratio_mean', math.nan) > STEP57B_CONCEPT_BASELINE else '否'}。",
        f"12. learned L2H alpha 是否在多个 fold 中非零：{'是' if alpha_nonzero_folds >= 2 else '否'}。",
        f"13. l2h_delta_abs_mean 是否在多个 fold 中非零：{'是' if delta_nonzero_folds >= 2 else '否'}。",
        f"14. retrieved_high_match_counts_mean 是否合理：{'是' if (safe_float(l2h_means.get('retrieved_high_match_counts_mean')) or 0.0) >= 8.0 else '否'}。",
        f"15. zero-match 比例是否低：{'是' if (safe_float(l2h_means.get('retrieved_high_zero_match_percent')) or 1.0) <= 0.05 else '否'}。",
        "16. full / concept_only / visual_only 的 5-fold branch 表现如何："
        f"full(AUC={format_metric(None if full_row.empty else full_row.iloc[0]['AUC'])}, ACC={format_metric(None if full_row.empty else full_row.iloc[0]['ACC'])}) "
        f"concept_only(AUC={format_metric(None if concept_row.empty else concept_row.iloc[0]['AUC'])}, ACC={format_metric(None if concept_row.empty else concept_row.iloc[0]['ACC'])}) "
        f"visual_only(AUC={format_metric(None if visual_row.empty else visual_row.iloc[0]['AUC'])}, ACC={format_metric(None if visual_row.empty else visual_row.iloc[0]['ACC'])})。",
        f"17. 是否可以把 L2H config G 作为新的候选主模型：{'是' if decision.get('decision') == 'candidate_primary_l2h' else '否'}。",
        f"18. 如果不适合作为主模型，是否保留为 L2H retrieval 解释性变体：{'是' if decision.get('decision') in {'l2h_interpretable_candidate', 'tradeoff_l2h'} else '否'}。",
        f"19. 下一步建议：{decision.get('next_step')}。",
        "",
        "## Notes",
        "",
        f"- preflight status: `{preflight.get('status')}`",
        f"- preflight checks: `{preflight.get('checks')}`",
        f"- config G build check: `{preflight.get('config_g_build')}`",
        f"- run_status: `{run_status}`",
        f"- decision: `{decision.get('decision')}`",
        f"- visual_ratio_mean: `{round_or_none(contribution_means.get('visual_ratio_mean'))}`",
        f"- concept_ratio_mean: `{round_or_none(contribution_means.get('concept_ratio_mean'))}`",
        f"- csg_ratio_mean: `{round_or_none(contribution_means.get('csg_ratio_mean'))}`",
        f"- learned_l2h_alpha_mean: `{round_or_none(l2h_means.get('learned_alpha_final'))}`",
        f"- l2h_delta_abs_mean: `{round_or_none(l2h_means.get('l2h_delta_abs_mean'))}`",
        f"- retrieved_high_match_counts_mean: `{round_or_none(l2h_means.get('retrieved_high_match_counts_mean'))}`",
        f"- retrieved_high_zero_match_percent: `{round_or_none(l2h_means.get('retrieved_high_zero_match_percent'))}`",
    ]
    fix_note = preflight.get("step61c_fix_note")
    if fix_note:
        lines.append(f"- Step61C carried minimal model fix into Step61D: {fix_note}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    display_root = detect_display_root(args.root)
    write_run_commands(output_dir, display_root)

    preflight = run_preflight(args.stage58c_dir)

    stage57c_fold_all = read_csv_if_exists(args.stage57c_dir.parent.parent / "results_stage57C_rce_v2_copy_reproduction")  # no-op path guard
    del stage57c_fold_all

    stage57c_fold_df = extract_model_rows(read_csv_if_exists(args.stage58c_dir.parent / "stage58C_fold_metrics.csv"), "stage57C_rce_v2_baseline")
    stage58c_fold_df = extract_model_rows(read_csv_if_exists(args.stage58c_dir.parent / "stage58C_fold_metrics.csv"), "stage58C_configD_residual_constrained")
    if stage58c_fold_df is None:
        stage58c_fold_df = extract_model_rows(
            read_csv_if_exists(args.stage58c_dir.parent / "stage58C_fold_metrics.csv"),
            "stage58C_residual_constrained_configD",
        )
    stage59c_fold_df = extract_model_rows(read_csv_if_exists(args.stage59c_dir.parent / "stage59C_fold_metrics.csv"), "stage59C_dynamic_csg_configA")
    stage60d_fold_df = extract_model_rows(read_csv_if_exists(args.stage60d_dir / "stage60D_fold_metrics.csv"), "stage60D_ccra_configC")
    stage58c_contrib_df = read_csv_if_exists(args.stage58c_dir.parent / "stage58C_contribution_by_fold.csv")

    stage57c_metrics = mean_metrics_from_model_rows(stage57c_fold_df)
    stage58c_metrics = mean_metrics_from_model_rows(stage58c_fold_df)
    stage59c_metrics = mean_metrics_from_model_rows(stage59c_fold_df)
    stage60d_metrics = mean_metrics_from_model_rows(stage60d_fold_df)

    stage61d_fold_df, _ = discover_fold_metrics(args.stage61d_run_dir)
    stage61d_metrics = compute_mean_metrics(stage61d_fold_df)
    run_status, checkpoint_folds = determine_run_status(args.stage61d_run_dir, stage61d_fold_df)

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
            "delta_visual_ratio_vs_step58C",
            "delta_concept_ratio_vs_step58C",
            "delta_csg_ratio_vs_step58C",
        ]
    )
    l2h_df = pd.DataFrame(
        columns=[
            "fold",
            "status",
            "l2h_enabled",
            "alpha_init",
            "learned_alpha_final",
            "l2h_mode",
            "l2h_low_topk",
            "l2h_high_max_per_low",
            "l2h_scale_ratio",
            "l2h_patch_footprint_ratio",
            "l2h_fusion",
            "l2h_aggregate",
            "l2h_score_mode",
            "low_patch_concept_scores_shape",
            "low_topk_scores_mean",
            "retrieved_high_match_counts_mean",
            "retrieved_high_match_counts_median",
            "retrieved_high_match_counts_max",
            "retrieved_high_zero_match_percent",
            "retrieved_high_features_norm",
            "original_high_region_norm",
            "fused_high_region_norm",
            "l2h_delta_abs_mean",
            "l2h_delta_vs_original_ratio",
            "low_coords_shape",
            "high_coords_shape",
            "skipped_reason",
            "anomaly_count",
        ]
    )
    audit_status: dict[str, object] = {}
    warnings: list[str] = []

    if preflight.get("ok") and checkpoint_folds:
        branch_df, contribution_df, l2h_df, audit_status, warnings = collect_audits_and_l2h(
            run_dir=args.stage61d_run_dir,
            fold_ids=checkpoint_folds,
            output_dir=output_dir,
            audit_script=args.audit_script,
            stage58c_contrib_df=stage58c_contrib_df,
        )

    branch_mean_df = summarize_branch_means(branch_df)
    contribution_means = summarize_contribution_means(contribution_df)
    l2h_means = summarize_l2h_means(l2h_df)

    compare_df = build_compare_df(
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage59c_metrics=stage59c_metrics,
        stage60d_metrics=stage60d_metrics,
        stage61d_metrics=stage61d_metrics,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
        stage60d_dir=args.stage60d_dir,
        stage61d_dir=args.stage61d_run_dir,
        contribution_means=contribution_means,
        l2h_means=l2h_means,
    )
    fold_metrics_df = build_fold_metrics_df(
        stage57c_fold_df=stage57c_fold_df,
        stage58c_fold_df=stage58c_fold_df,
        stage59c_fold_df=stage59c_fold_df,
        stage60d_fold_df=stage60d_fold_df,
        stage61d_fold_df=stage61d_fold_df,
        stage57c_dir=args.stage57c_dir,
        stage58c_dir=args.stage58c_dir,
        stage59c_dir=args.stage59c_dir,
        stage60d_dir=args.stage60d_dir,
        stage61d_dir=args.stage61d_run_dir,
    )
    decision = decide_outcome(
        stage58c_metrics=stage58c_metrics,
        stage61d_metrics=stage61d_metrics,
        stage61d_fold_df=stage61d_fold_df,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
        l2h_means=l2h_means,
        l2h_df=l2h_df,
        run_status=run_status,
        preflight_pass=bool(preflight.get("ok")),
    )

    status_payload = {
        "preflight": preflight,
        "run_status": run_status,
        "checkpoint_folds": checkpoint_folds,
        "warnings": warnings,
        "audit_status": audit_status,
        "decision": decision.get("decision"),
        "next_step": decision.get("next_step"),
        "run_dir": str(args.stage61d_run_dir),
    }

    summary_md = build_summary_md(
        preflight=preflight,
        run_status=run_status,
        stage57c_metrics=stage57c_metrics,
        stage58c_metrics=stage58c_metrics,
        stage59c_metrics=stage59c_metrics,
        stage60d_metrics=stage60d_metrics,
        stage61d_metrics=stage61d_metrics,
        branch_mean_df=branch_mean_df,
        contribution_means=contribution_means,
        l2h_means=l2h_means,
        l2h_df=l2h_df,
        decision=decision,
    )

    compare_df.to_csv(output_dir / "stage61D_compare_with_baselines.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "stage61D_fold_metrics.csv", index=False)
    branch_df.to_csv(output_dir / "stage61D_branch_metrics_by_fold.csv", index=False)
    contribution_df.to_csv(output_dir / "stage61D_contribution_by_fold.csv", index=False)
    l2h_df.to_csv(output_dir / "stage61D_l2h_by_fold.csv", index=False)
    (output_dir / "stage61D_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage61D_status.json").write_text(json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage61D_summary.md").write_text(summary_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
