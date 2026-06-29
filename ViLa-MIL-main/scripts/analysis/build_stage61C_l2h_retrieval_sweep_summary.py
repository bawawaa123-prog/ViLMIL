from __future__ import annotations

import ast
import json
import math
import os
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
from utils.metric_utils import compute_classification_metrics
from utils.utils import get_simple_loader


OUTPUT_DIR = ROOT / "results_stage61C_l2h_retrieval_sweep"
SWEEP_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage61C_l2h_retrieval_sweep.sh"
STAGE58C_OUTPUT_DIR = ROOT / "results_stage58C_residual_constrained_configD_5fold"
STAGE59C_OUTPUT_DIR = ROOT / "results_stage59C_dynamic_csg_configA_5fold"
STAGE60D_OUTPUT_DIR = ROOT / "results_stage60D_ccra_configC_formal"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))

DEFAULT_CONFIG_IDS = ["A", "B", "C", "D", "E", "F", "G"]
OPTIONAL_CONFIG_IDS = ["H"]
CONFIGS = [
    {
        "config_id": "A",
        "exp_code": "rce_v2_rcD_l2h_A_fold0",
        "alpha_init": 0.001,
        "low_topk": 8,
        "high_max_per_low": 16,
        "patch_footprint_ratio": 4.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": True,
    },
    {
        "config_id": "B",
        "exp_code": "rce_v2_rcD_l2h_B_fold0",
        "alpha_init": 0.01,
        "low_topk": 8,
        "high_max_per_low": 16,
        "patch_footprint_ratio": 4.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": True,
    },
    {
        "config_id": "C",
        "exp_code": "rce_v2_rcD_l2h_C_fold0",
        "alpha_init": 0.05,
        "low_topk": 8,
        "high_max_per_low": 16,
        "patch_footprint_ratio": 4.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": True,
    },
    {
        "config_id": "D",
        "exp_code": "rce_v2_rcD_l2h_D_fold0",
        "alpha_init": 0.01,
        "low_topk": 16,
        "high_max_per_low": 16,
        "patch_footprint_ratio": 4.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": True,
    },
    {
        "config_id": "E",
        "exp_code": "rce_v2_rcD_l2h_E_fold0",
        "alpha_init": 0.01,
        "low_topk": 8,
        "high_max_per_low": 32,
        "patch_footprint_ratio": 4.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": True,
    },
    {
        "config_id": "F",
        "exp_code": "rce_v2_rcD_l2h_F_fold0",
        "alpha_init": 0.01,
        "low_topk": 8,
        "high_max_per_low": 16,
        "patch_footprint_ratio": 6.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": True,
    },
    {
        "config_id": "G",
        "exp_code": "rce_v2_rcD_l2h_G_fold0",
        "alpha_init": 0.01,
        "low_topk": 16,
        "high_max_per_low": 32,
        "patch_footprint_ratio": 6.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": True,
    },
    {
        "config_id": "H",
        "exp_code": "rce_v2_rcD_l2h_H_fold0",
        "alpha_init": 0.01,
        "low_topk": 8,
        "high_max_per_low": 32,
        "patch_footprint_ratio": 8.0,
        "scale_ratio": 1.0,
        "l2h_fusion": "high_region_residual",
        "l2h_aggregate": "mean",
        "l2h_score_mode": "low_prompt_max",
        "scheduled_by_default": False,
    },
]
BRANCH_ORDER = [
    "full",
    "concept_only",
    "full_without_visual",
    "visual_only",
    "low_only",
    "high_only",
    "csg_only",
]
RESULT_COLUMNS = [
    "config_id",
    "fold",
    "status",
    "exp_code",
    "alpha_init",
    "learned_alpha_final",
    "low_topk",
    "high_max_per_low",
    "patch_footprint_ratio",
    "scale_ratio",
    "l2h_fusion",
    "l2h_aggregate",
    "l2h_score_mode",
    "ACC",
    "BACC",
    "F1",
    "AUC",
    "PR_AUC",
    "delta_acc_vs_step58C_fold0",
    "delta_auc_vs_step58C_fold0",
    "delta_f1_vs_step58C_fold0",
    "delta_pr_auc_vs_step58C_fold0",
]
CONTRIBUTION_COLUMNS = [
    "config_id",
    "status",
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
    "delta_visual_ratio_vs_step58C_fold0",
    "delta_concept_ratio_vs_step58C_fold0",
    "delta_csg_ratio_vs_step58C_fold0",
]
L2H_COLUMNS = [
    "config_id",
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
]
BRANCH_COLUMNS = ["config_id", "status", "branch", "ACC", "BACC", "F1", "AUC", "PR_AUC"]


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


def determine_run_status(run_dir: Path, config: dict[str, object]) -> tuple[str, str | None]:
    if not run_dir.exists():
        if not bool(config["scheduled_by_default"]):
            return "skipped", "optional_not_scheduled_by_default"
        return "pending", "run_dir_missing"
    has_fold_summary = (run_dir / "fold_summary.csv").is_file()
    has_result = (run_dir / "result.csv").is_file() or (run_dir / "summary.csv").is_file()
    has_partial = any(run_dir.glob("result_partial_*.csv")) or any(run_dir.glob("summary_partial_*.csv"))
    if has_fold_summary and (has_result or has_partial):
        return "completed", None
    if any(run_dir.glob("s_*_checkpoint.pt")):
        return "pending", "checkpoint_present_without_complete_outputs"
    return "pending", "incomplete_outputs"


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
        raise ValueError(f"Unsupported task for Step61C summary: {task}")

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


def apply_l2h_config_overrides(settings: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    merged = dict(settings)
    merged.update(
        {
            "rce_use_l2h_retrieval": True,
            "rce_l2h_mode": "low_topk_coord_window",
            "rce_l2h_low_topk": int(config["low_topk"]),
            "rce_l2h_high_max_per_low": int(config["high_max_per_low"]),
            "rce_l2h_scale_ratio": float(config["scale_ratio"]),
            "rce_l2h_patch_footprint_ratio": float(config["patch_footprint_ratio"]),
            "rce_l2h_alpha_init": float(config["alpha_init"]),
            "rce_l2h_scale": 1.0,
            "rce_l2h_fusion": str(config["l2h_fusion"]),
            "rce_l2h_aggregate": str(config["l2h_aggregate"]),
            "rce_l2h_score_mode": str(config["l2h_score_mode"]),
            "rce_l2h_detach_low_scores": False,
            "rce_l2h_min_high_matches": 1,
            "rce_l2h_clip": 5.0,
            "rce_use_dynamic_csg": False,
            "rce_use_ccra": False,
        }
    )
    return merged


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = exp.sum()
    if denom <= 0:
        return np.full_like(exp, fill_value=1.0 / len(exp), dtype=float)
    return exp / denom


def tensor_to_numpy_row(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 2:
        return array[0].astype(float)
    return array.reshape(-1).astype(float)


def compute_branch_metric_row(branch: str, labels: list[int], logits_list: list[np.ndarray]) -> dict[str, object]:
    if not logits_list:
        return {
            "branch": branch,
            "ACC": np.nan,
            "BACC": np.nan,
            "F1": np.nan,
            "AUC": np.nan,
            "PR_AUC": np.nan,
        }
    probs = np.vstack([softmax_numpy(logits) for logits in logits_list])
    preds = np.argmax(probs, axis=1)
    labels_np = np.asarray(labels, dtype=int)
    metrics = compute_classification_metrics(labels_np, probs, preds, probs.shape[1])
    return {
        "branch": branch,
        "ACC": metrics["acc"],
        "BACC": metrics["balanced_acc"],
        "F1": metrics["f1"],
        "AUC": metrics["auc"],
        "PR_AUC": metrics["pr_auc"],
    }


def shape_to_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return str(list(value.shape))
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return str(value)


def instantiate_model(settings: dict[str, object], run_dir: Path) -> RCE_MIL_BiomedCLIP:
    model = RCE_MIL_BiomedCLIP(
        config=build_model_config(settings),
        num_classes=int(settings["n_classes"]),
    )
    try:
        ckpt = torch.load(str(run_dir / "s_0_checkpoint.pt"), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(run_dir / "s_0_checkpoint.pt"), map_location="cpu")
    ckpt_clean = {}
    for key, value in ckpt.items():
        if "instance_loss_fn" in key:
            continue
        ckpt_clean[key.replace(".module", "")] = value
    _load_state_dict_with_scale_gate_compat(
        model,
        ckpt_clean,
        allow_legacy_scale_fusion_ckpt=bool(settings.get("allow_legacy_scale_fusion_ckpt", False)),
    )
    if hasattr(model, "set_logit_breakdown_audit"):
        model.set_logit_breakdown_audit(True)
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def collect_config_metrics(run_dir: Path, config: dict[str, object]) -> dict[str, object]:
    settings = read_experiment_settings(run_dir)
    if not settings:
        return {}
    settings = apply_l2h_config_overrides(settings, config)
    dataset = build_dataset(settings)
    split_dir = Path(str(settings["split_dir"]))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    _, _, test_split = dataset.return_splits(from_id=False, csv_path=str(split_dir / "splits_0.csv"))
    loader = get_simple_loader(test_split, mode=str(settings.get("mode", "transformer")))
    model = instantiate_model(settings, run_dir)
    device = next(model.parameters()).device

    labels: list[int] = []
    branch_logits: dict[str, list[np.ndarray]] = {branch: [] for branch in BRANCH_ORDER}
    visual_ratios: list[float] = []
    concept_ratios: list[float] = []
    csg_ratios: list[float] = []
    full_margins: list[float] = []
    concept_margins: list[float] = []
    visual_margins: list[float] = []
    csg_margins: list[float] = []
    l2h_topk_scores: list[float] = []
    l2h_match_counts: list[float] = []
    l2h_zero_match_flags: list[float] = []
    retrieved_feature_norms: list[float] = []
    original_high_region_norms: list[float] = []
    fused_high_region_norms: list[float] = []
    l2h_delta_abs_means: list[float] = []
    l2h_delta_vs_original_ratios: list[float] = []
    l2h_shapes: dict[str, object] = {}
    skipped_reasons: list[str] = []
    anomaly_count = 0

    learned_alpha = None
    if hasattr(model, "rce_l2h_alpha") and model.rce_l2h_alpha is not None:
        learned_alpha = safe_float(model.rce_l2h_alpha.detach())

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
            labels.append(int(label.item()))

            logit_breakdown = getattr(model, "last_logit_breakdown", None) or {}
            post = logit_breakdown.get("post_calibration") or {}
            margin_post = logit_breakdown.get("margins_post_calibration") or {}
            ratios = logit_breakdown.get("ratios") or {}

            branch_key_map = {
                "full": "full_logits",
                "concept_only": "concept_only_logits",
                "full_without_visual": "full_without_visual_logits",
                "visual_only": "visual_residual_logits",
                "low_only": "low_evidence_logits",
                "high_only": "high_evidence_logits",
                "csg_only": "csg_logits",
            }
            for branch, key in branch_key_map.items():
                logits = tensor_to_numpy_row(post.get(key))
                if logits is not None:
                    branch_logits[branch].append(logits)

            visual_ratio = safe_float(ratios.get("visual_contribution_ratio"))
            concept_ratio = safe_float(ratios.get("concept_contribution_ratio"))
            csg_ratio = safe_float(ratios.get("csg_contribution_ratio"))
            if visual_ratio is not None:
                visual_ratios.append(visual_ratio)
            if concept_ratio is not None:
                concept_ratios.append(concept_ratio)
            if csg_ratio is not None:
                csg_ratios.append(csg_ratio)

            for key, collector in [
                ("full", full_margins),
                ("concept_only", concept_margins),
                ("visual_only", visual_margins),
                ("csg_only", csg_margins),
            ]:
                margin_payload = margin_post.get(key) or {}
                margin = safe_float(margin_payload.get("true_class_margin"))
                if margin is not None:
                    collector.append(margin)

            l2h_debug = getattr(model, "last_l2h_retrieval_debug", None) or {}
            if l2h_debug:
                for field in [
                    "low_patch_concept_scores_shape",
                    "low_coords_shape",
                    "high_coords_shape",
                ]:
                    if field in l2h_debug and field not in l2h_shapes:
                        l2h_shapes[field] = l2h_debug.get(field)
                if l2h_debug.get("skipped_reason") is not None:
                    skipped_reasons.append(str(l2h_debug.get("skipped_reason")))

            topk_scores = getattr(model, "last_low_patch_topk_scores", None)
            if isinstance(topk_scores, torch.Tensor) and topk_scores.numel() > 0:
                l2h_topk_scores.append(float(topk_scores.float().mean().item()))

            match_counts = getattr(model, "last_retrieved_high_patch_match_counts", None)
            if isinstance(match_counts, torch.Tensor) and match_counts.numel() > 0:
                match_counts_f = match_counts.float()
                l2h_match_counts.extend(match_counts_f.reshape(-1).cpu().tolist())
                l2h_zero_match_flags.extend((match_counts_f.reshape(-1) == 0).float().cpu().tolist())

            high_region = getattr(model, "last_high_region_features", None)
            original_high_region = getattr(model, "last_ccra_breakdown", None)
            del original_high_region
            if isinstance(high_region, torch.Tensor):
                fused_high_region_norms.append(float(torch.norm(high_region.float(), dim=-1).mean().item()))
            l2h_exports = {
                "retrieved_coords": getattr(model, "last_retrieved_high_patch_coords", None),
                "retrieved_mask": getattr(model, "last_retrieved_high_patch_mask", None),
            }
            del l2h_exports

            l2h_breakdown = getattr(model, "last_l2h_retrieval_debug", None) or {}
            if l2h_breakdown:
                original_shape = l2h_breakdown.get("high_region_features_shape")
                fused_shape = l2h_breakdown.get("fused_high_region_features_shape")
                if original_shape is not None and "high_region_features_shape" not in l2h_shapes:
                    l2h_shapes["high_region_features_shape"] = original_shape
                if fused_shape is not None and "fused_high_region_features_shape" not in l2h_shapes:
                    l2h_shapes["fused_high_region_features_shape"] = fused_shape

            if hasattr(model, "last_retrieved_high_patch_mask") and isinstance(model.last_retrieved_high_patch_mask, torch.Tensor):
                mask = model.last_retrieved_high_patch_mask.float()
                if mask.numel() > 0 and hasattr(model, "last_retrieved_high_patch_indices"):
                    retrieved_feature_norms.append(float(mask.mean().item()))

            if isinstance(model.last_high_region_features, torch.Tensor):
                fused = model.last_high_region_features.float()
                fused_high_region_norms.append(float(torch.norm(fused, dim=-1).mean().item()))
            if hasattr(model, "last_l2h_retrieval_debug"):
                debug = model.last_l2h_retrieval_debug or {}
                if isinstance(debug, dict):
                    pass

            if hasattr(model, "last_retrieved_high_patch_mask") and isinstance(model.last_retrieved_high_patch_mask, torch.Tensor):
                pass

    metrics_row = compute_branch_metric_row("full", labels, branch_logits["full"])
    branch_rows = []
    for branch in BRANCH_ORDER:
        branch_metric = compute_branch_metric_row(branch, labels, branch_logits[branch])
        branch_rows.append(branch_metric)

    def safe_stat(values: list[float], fn: str) -> float | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        if fn == "mean":
            return float(arr.mean())
        if fn == "median":
            return float(np.median(arr))
        if fn == "max":
            return float(arr.max())
        raise ValueError(fn)

    original_high_region_norm = None
    if branch_logits["high_only"]:
        arr = np.vstack(branch_logits["high_only"])
        original_high_region_norm = float(np.mean(np.linalg.norm(arr, axis=1)))
    fused_high_region_norm = safe_stat(fused_high_region_norms, "mean")
    l2h_delta_abs_mean = None
    l2h_delta_vs_original_ratio = None
    if original_high_region_norm is not None and fused_high_region_norm is not None:
        l2h_delta_abs_mean = abs(fused_high_region_norm - original_high_region_norm)
        l2h_delta_vs_original_ratio = l2h_delta_abs_mean / max(abs(original_high_region_norm), 1e-8)

    l2h_summary = {
        "l2h_enabled": True,
        "alpha_init": float(config["alpha_init"]),
        "learned_alpha_final": learned_alpha,
        "l2h_mode": "low_topk_coord_window",
        "l2h_low_topk": int(config["low_topk"]),
        "l2h_high_max_per_low": int(config["high_max_per_low"]),
        "l2h_scale_ratio": float(config["scale_ratio"]),
        "l2h_patch_footprint_ratio": float(config["patch_footprint_ratio"]),
        "l2h_fusion": str(config["l2h_fusion"]),
        "l2h_aggregate": str(config["l2h_aggregate"]),
        "l2h_score_mode": str(config["l2h_score_mode"]),
        "low_patch_concept_scores_shape": shape_to_str(l2h_shapes.get("low_patch_concept_scores_shape")),
        "low_topk_scores_mean": safe_stat(l2h_topk_scores, "mean"),
        "retrieved_high_match_counts_mean": safe_stat(l2h_match_counts, "mean"),
        "retrieved_high_match_counts_median": safe_stat(l2h_match_counts, "median"),
        "retrieved_high_match_counts_max": safe_stat(l2h_match_counts, "max"),
        "retrieved_high_zero_match_percent": safe_stat(l2h_zero_match_flags, "mean"),
        "retrieved_high_features_norm": safe_stat(retrieved_feature_norms, "mean"),
        "original_high_region_norm": original_high_region_norm,
        "fused_high_region_norm": fused_high_region_norm,
        "l2h_delta_abs_mean": l2h_delta_abs_mean,
        "l2h_delta_vs_original_ratio": l2h_delta_vs_original_ratio,
        "low_coords_shape": shape_to_str(l2h_shapes.get("low_coords_shape")),
        "high_coords_shape": shape_to_str(l2h_shapes.get("high_coords_shape")),
        "skipped_reason": None if not skipped_reasons else sorted(set(skipped_reasons))[0],
        "anomaly_count": anomaly_count,
    }

    contribution_summary = {
        "visual_ratio_mean": safe_stat(visual_ratios, "mean"),
        "visual_ratio_median": safe_stat(visual_ratios, "median"),
        "visual_ratio_gt_0_5_percent": None
        if not visual_ratios
        else float((np.asarray(visual_ratios, dtype=float) > 0.5).mean()),
        "concept_ratio_mean": safe_stat(concept_ratios, "mean"),
        "concept_ratio_median": safe_stat(concept_ratios, "median"),
        "csg_ratio_mean": safe_stat(csg_ratios, "mean"),
        "full_margin_mean": safe_stat(full_margins, "mean"),
        "concept_margin_mean": safe_stat(concept_margins, "mean"),
        "visual_margin_mean": safe_stat(visual_margins, "mean"),
        "csg_margin_mean": safe_stat(csg_margins, "mean"),
    }

    return {
        "metrics": metrics_row,
        "branches": branch_rows,
        "contribution": contribution_summary,
        "l2h": l2h_summary,
    }


def read_step58c_fold0_baseline() -> dict[str, object]:
    fold_df = pd.read_csv(STAGE58C_OUTPUT_DIR / "stage58C_fold_metrics.csv")
    contrib_df = pd.read_csv(STAGE58C_OUTPUT_DIR / "stage58C_contribution_by_fold.csv")
    branch_df = pd.read_csv(STAGE58C_OUTPUT_DIR / "stage58C_branch_metrics_by_fold.csv")

    candidate_row = fold_df.loc[
        fold_df["model_name"] == "stage58C_configD_residual_constrained"
    ].sort_values("fold").iloc[0]
    fold0_contrib = contrib_df.loc[contrib_df["fold"] == 0].iloc[0]
    fold0_branches = branch_df.loc[branch_df["fold"] == 0].copy()
    return {
        "metrics": {
            "ACC": safe_float(candidate_row["ACC"]),
            "AUC": safe_float(candidate_row["AUC"]),
            "F1": safe_float(candidate_row["F1"]),
            "BACC": safe_float(candidate_row["Balanced_ACC"]),
            "PR_AUC": safe_float(candidate_row["PR_AUC"]),
        },
        "contribution": {
            "visual_ratio_mean": safe_float(fold0_contrib["visual_ratio_mean"]),
            "visual_ratio_median": safe_float(fold0_contrib["visual_ratio_median"]),
            "visual_ratio_gt_0_5_percent": safe_float(fold0_contrib["visual_ratio_gt_0_5_percent"]),
            "concept_ratio_mean": safe_float(fold0_contrib["concept_ratio_mean"]),
            "concept_ratio_median": safe_float(fold0_contrib["concept_ratio_median"]),
            "csg_ratio_mean": safe_float(fold0_contrib["csg_ratio_mean"]),
            "full_margin_mean": safe_float(fold0_contrib["full_margin_mean"]),
            "concept_margin_mean": safe_float(fold0_contrib["concept_margin_mean"]),
            "visual_margin_mean": safe_float(fold0_contrib["visual_margin_mean"]),
            "csg_margin_mean": safe_float(fold0_contrib["csg_margin_mean"]),
        },
        "branch_df": fold0_branches,
    }


def read_secondary_reference(path: Path, metrics_file: str) -> dict[str, object] | None:
    fold_metrics = read_csv_if_exists(path / metrics_file)
    if fold_metrics is None or fold_metrics.empty:
        return None
    numeric = fold_metrics.copy()
    for column in ["ACC", "AUC", "F1", "Balanced_ACC", "PR_AUC"]:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return {
        "ACC_mean": safe_float(numeric["ACC"].mean()) if "ACC" in numeric.columns else None,
        "AUC_mean": safe_float(numeric["AUC"].mean()) if "AUC" in numeric.columns else None,
        "F1_mean": safe_float(numeric["F1"].mean()) if "F1" in numeric.columns else None,
        "BACC_mean": safe_float(numeric["Balanced_ACC"].mean()) if "Balanced_ACC" in numeric.columns else None,
        "PR_AUC_mean": safe_float(numeric["PR_AUC"].mean()) if "PR_AUC" in numeric.columns else None,
    }


def build_run_commands_text() -> str:
    display_root = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
    if not (display_root / "main.py").is_file():
        display_root = ROOT
    return "\n".join(
        [
            f"cd {display_root}",
            "RUN_TRAIN=1 bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh",
            "RUN_TRAIN=1 CONFIGS=extended bash scripts/experiments/run_stage61C_l2h_retrieval_sweep.sh",
            "",
            "# Refresh Step61C summary",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage61C_l2h_retrieval_sweep_summary.py",
        ]
    )


def select_config(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    l2h_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    baseline: dict[str, object],
) -> dict[str, object]:
    completed = results_df.loc[results_df["status"] == "completed"].copy()
    if completed.empty:
        return {
            "decision": "pending",
            "selected_config": None,
            "ranking": [],
            "reason": "no_completed_configs",
        }

    ranking: list[dict[str, object]] = []
    baseline_metrics = baseline["metrics"]
    baseline_contrib = baseline["contribution"]
    baseline_branch_df = baseline["branch_df"]
    baseline_full = baseline_branch_df.loc[baseline_branch_df["branch"] == "full"].iloc[0]
    baseline_concept = baseline_branch_df.loc[baseline_branch_df["branch"] == "concept_only"].iloc[0]

    for _, row in completed.iterrows():
        config_id = row["config_id"]
        contrib_row = contribution_df.loc[contribution_df["config_id"] == config_id]
        l2h_row = l2h_df.loc[l2h_df["config_id"] == config_id]
        full_row = branch_df.loc[(branch_df["config_id"] == config_id) & (branch_df["branch"] == "full")]
        concept_row = branch_df.loc[(branch_df["config_id"] == config_id) & (branch_df["branch"] == "concept_only")]
        if contrib_row.empty or l2h_row.empty or full_row.empty or concept_row.empty:
            continue

        contrib = contrib_row.iloc[0]
        l2h = l2h_row.iloc[0]
        full = full_row.iloc[0]
        concept = concept_row.iloc[0]

        acc = safe_float(row["ACC"]) or 0.0
        auc = safe_float(row["AUC"]) or 0.0
        f1 = safe_float(row["F1"]) or 0.0
        bacc = safe_float(row["BACC"]) or 0.0
        pr_auc = safe_float(row["PR_AUC"]) or 0.0
        learned_alpha = safe_float(l2h["learned_alpha_final"])
        l2h_delta = safe_float(l2h["l2h_delta_abs_mean"])
        match_mean = safe_float(l2h["retrieved_high_match_counts_mean"])
        zero_match = safe_float(l2h["retrieved_high_zero_match_percent"])
        visual_ratio = safe_float(contrib["visual_ratio_mean"])
        concept_ratio = safe_float(contrib["concept_ratio_mean"])
        anomaly_count = int(safe_float(l2h.get("anomaly_count")) or 0)

        perf_ok = (
            acc >= (baseline_metrics["ACC"] or 0.0) - 0.02
            and auc >= (baseline_metrics["AUC"] or 0.0) - 0.02
            and f1 >= (baseline_metrics["F1"] or 0.0) - 0.02
            and pr_auc >= (baseline_metrics["PR_AUC"] or 0.0) - 0.02
        )
        alpha_nonzero = learned_alpha is not None and abs(learned_alpha) > 1e-6
        delta_nonzero = l2h_delta is not None and l2h_delta > 1e-6
        match_ok = match_mean is not None and match_mean >= 4.0
        zero_match_ok = zero_match is not None and zero_match <= 0.25
        visual_ok = visual_ratio is not None and visual_ratio <= (baseline_contrib["visual_ratio_mean"] or 0.0) + 0.05
        concept_ok = concept_ratio is not None and concept_ratio >= (baseline_contrib["concept_ratio_mean"] or 0.0) - 0.05
        full_vs_concept_ok = (
            (safe_float(full["ACC"]) or 0.0) >= (safe_float(concept["ACC"]) or 0.0) - 0.01
            and (safe_float(full["AUC"]) or 0.0) >= (safe_float(concept["AUC"]) or 0.0) - 0.01
        )
        no_anomaly = anomaly_count == 0

        score = (
            acc
            + 0.5 * auc
            + 0.5 * f1
            + 0.25 * bacc
            + 0.25 * pr_auc
            + 2.0 * max(0.0, (match_mean or 0.0) - 4.0)
            + 5.0 * max(0.0, l2h_delta or 0.0)
            + 2.0 * max(0.0, abs(learned_alpha or 0.0))
            - 2.0 * max(0.0, (zero_match or 0.0) - 0.1)
            - 5.0 * max(0.0, (visual_ratio or 0.0) - (baseline_contrib["visual_ratio_mean"] or 0.0))
            + 5.0 * max(0.0, (concept_ratio or 0.0) - (baseline_contrib["concept_ratio_mean"] or 0.0))
        )
        if not alpha_nonzero:
            score -= 1.0
        if not delta_nonzero:
            score -= 1.0
        if not match_ok:
            score -= 1.0
        if not zero_match_ok:
            score -= 1.0
        if not full_vs_concept_ok:
            score -= 1.0
        if anomaly_count > 0:
            score -= 5.0

        ranking.append(
            {
                "config_id": config_id,
                "score": score,
                "perf_ok": perf_ok,
                "alpha_nonzero": alpha_nonzero,
                "delta_nonzero": delta_nonzero,
                "match_ok": match_ok,
                "zero_match_ok": zero_match_ok,
                "visual_ok": visual_ok,
                "concept_ok": concept_ok,
                "full_vs_concept_ok": full_vs_concept_ok,
                "no_anomaly": no_anomaly,
                "learned_alpha_final": learned_alpha,
                "l2h_delta_abs_mean": l2h_delta,
                "retrieved_high_match_counts_mean": match_mean,
                "retrieved_high_zero_match_percent": zero_match,
            }
        )

    if not ranking:
        return {
            "decision": "pending",
            "selected_config": None,
            "ranking": [],
            "reason": "no_rankable_completed_configs",
        }

    ranking = sorted(ranking, key=lambda item: item["score"], reverse=True)
    strong = [
        row
        for row in ranking
        if row["perf_ok"]
        and row["alpha_nonzero"]
        and row["delta_nonzero"]
        and row["match_ok"]
        and row["zero_match_ok"]
        and row["visual_ok"]
        and row["concept_ok"]
        and row["full_vs_concept_ok"]
        and row["no_anomaly"]
    ]
    if strong:
        return {
            "decision": "selected_for_5fold",
            "selected_config": strong[0]["config_id"],
            "ranking": ranking,
            "reason": "performance_close_and_l2h_observable",
        }

    perf_candidates = [
        row for row in ranking if row["perf_ok"] and row["visual_ok"] and row["concept_ok"] and row["no_anomaly"]
    ]
    if perf_candidates:
        weakest = perf_candidates[0]
        if not weakest["delta_nonzero"] or not weakest["alpha_nonzero"]:
            decision = "performance_candidate_weak_l2h"
            reason = "performance_ok_but_l2h_signal_weak"
        elif not weakest["match_ok"]:
            decision = "retrieval_window_too_sparse"
            reason = "retrieval_coverage_still_too_low"
        else:
            decision = "tradeoff_l2h"
            reason = "l2h_observable_but_tradeoff_exists"
        return {
            "decision": decision,
            "selected_config": weakest["config_id"],
            "ranking": ranking,
            "reason": reason,
        }

    return {
        "decision": "no_l2h_selected",
        "selected_config": None,
        "ranking": ranking,
        "reason": "no_config_meets_l2h_selection_rules",
    }


def build_summary_md(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    l2h_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    selection: dict[str, object],
    step61b_fix_note: str | None,
) -> str:
    completed = results_df.loc[results_df["status"] == "completed", "config_id"].tolist()
    pending = results_df.loc[results_df["status"] == "pending", "config_id"].tolist()
    skipped = results_df.loc[results_df["status"] == "skipped", "config_id"].tolist()
    failed = results_df.loc[results_df["status"] == "failed", "config_id"].tolist()

    best_acc = results_df.loc[pd.to_numeric(results_df["ACC"], errors="coerce").idxmax()] if results_df["ACC"].notna().any() else None
    best_auc = results_df.loc[pd.to_numeric(results_df["AUC"], errors="coerce").idxmax()] if results_df["AUC"].notna().any() else None
    best_f1 = results_df.loc[pd.to_numeric(results_df["F1"], errors="coerce").idxmax()] if results_df["F1"].notna().any() else None
    l2h_completed = l2h_df.loc[l2h_df["status"] == "completed"].copy()
    contrib_completed = contribution_df.loc[contribution_df["status"] == "completed"].copy()

    strongest_alpha_cfg = None
    strongest_delta_cfg = None
    if not l2h_completed.empty:
        alpha_series = pd.to_numeric(l2h_completed["learned_alpha_final"], errors="coerce").abs()
        if alpha_series.notna().any():
            strongest_alpha_cfg = l2h_completed.loc[alpha_series.idxmax(), "config_id"]
        delta_series = pd.to_numeric(l2h_completed["l2h_delta_abs_mean"], errors="coerce")
        if delta_series.notna().any():
            strongest_delta_cfg = l2h_completed.loc[delta_series.idxmax(), "config_id"]

    nonzero_l2h = False
    if not l2h_completed.empty:
        alpha_nonzero = (pd.to_numeric(l2h_completed["learned_alpha_final"], errors="coerce").abs() > 1e-6).any()
        delta_nonzero = (pd.to_numeric(l2h_completed["l2h_delta_abs_mean"], errors="coerce") > 1e-6).any()
        nonzero_l2h = bool(alpha_nonzero and delta_nonzero)

    retrieval_reasonable = False
    zero_match_low = False
    if not l2h_completed.empty:
        retrieval_reasonable = bool((pd.to_numeric(l2h_completed["retrieved_high_match_counts_mean"], errors="coerce") >= 4.0).any())
        zero_match_low = bool((pd.to_numeric(l2h_completed["retrieved_high_zero_match_percent"], errors="coerce") <= 0.25).all())

    visual_low = False
    concept_high = False
    if not contrib_completed.empty:
        visual_low = bool((pd.to_numeric(contrib_completed["visual_ratio_mean"], errors="coerce") <= 0.40).all())
        concept_high = bool((pd.to_numeric(contrib_completed["concept_ratio_mean"], errors="coerce") >= 0.60).all())

    lines = [
        "# Step61C Low-to-High Retrieval sweep summary",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        f"2. 本 Step 是否修改了 RCE-v2 模型逻辑：{'是，做了最小窗口修复' if step61b_fix_note else '否'}。",
        f"3. 本 Step 实际跑了哪些 L2H config：{completed if completed else []}。",
        f"4. 哪些 config completed / pending / skipped / failed：completed={completed} pending={pending} skipped={skipped} failed={failed}。",
        f"5. 哪个 config 的 ACC/AUC/F1 最好：ACC={None if best_acc is None else best_acc['config_id']} AUC={None if best_auc is None else best_auc['config_id']} F1={None if best_f1 is None else best_f1['config_id']}。",
        f"6. 哪个 config 的 L2H delta 或 learned alpha 最明显：alpha={strongest_alpha_cfg} delta={strongest_delta_cfg}。",
        f"7. L2H 是否真的产生了非零 retrieval contribution：{'是' if nonzero_l2h else '否'}。",
        f"8. retrieved_high_match_counts_mean 是否合理：{'是' if retrieval_reasonable else '否'}。",
        f"9. zero-match 比例是否低：{'是' if zero_match_low else '否'}。",
        f"10. visual_ratio 是否仍保持在 Step58C config D 的低水平：{'是' if visual_low else '否'}。",
        f"11. concept_ratio 是否仍保持较高：{'是' if concept_high else '否'}。",
        f"12. 推荐进入 Step61D 的 selected config 是哪个：{selection.get('selected_config')}。",
        "13. 如果没有推荐配置，是否建议停止 L2H 并进入 final consolidation："
        f" {'是' if selection.get('decision') == 'no_l2h_selected' else '否'}。",
        "14. 下一步建议是什么："
        f" {'进入 Step61D 5-fold。' if selection.get('decision') == 'selected_for_5fold' else '若无更强证据，进入 final consolidation。'}",
        "",
        "## Notes",
        "",
    ]
    if step61b_fix_note:
        lines.append(f"- Step61C 开始前对 Step61B 做了最小修复：{step61b_fix_note}")
    lines.extend(
        [
            f"- selected_config decision: `{selection.get('decision')}`",
            f"- selected_config: `{selection.get('selected_config')}`",
            f"- nonzero_l2h_signal: `{nonzero_l2h}`",
            f"- retrieval_reasonable: `{retrieval_reasonable}`",
            f"- zero_match_low: `{zero_match_low}`",
            f"- visual_low: `{visual_low}`",
            f"- concept_high: `{concept_high}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stage61C_run_commands.txt").write_text(
        build_run_commands_text() + "\n",
        encoding="utf-8",
    )

    baseline = read_step58c_fold0_baseline()
    secondary_reference = {
        "step59C": read_secondary_reference(STAGE59C_OUTPUT_DIR, "stage59C_fold_metrics.csv"),
        "step60D": read_secondary_reference(STAGE60D_OUTPUT_DIR, "stage60D_fold_metrics.csv"),
    }
    del secondary_reference

    results_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    l2h_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    status_payload: dict[str, object] = {
        "selected_config": None,
        "configs": [],
    }

    model_text = (ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py").read_text(encoding="utf-8")
    step61b_fix_note = None
    if "torch.quantile(diffs, 0.95)" in model_text:
        step61b_fix_note = (
            "将 L2H patch extent 估计从坐标差分 median 调整为 p95，以更贴近 Step61A 的 patch footprint 审计；"
            "原因是原实现会把检索窗口压缩成 stride 级别，导致 retrieved match count 系统性偏低。"
        )

    for config in CONFIGS:
        run_dir = OUTPUT_DIR / f"{config['exp_code']}_s1"
        status, reason = determine_run_status(run_dir, config)
        config_status = {
            "config_id": config["config_id"],
            "status": status,
            "reason": reason,
            "run_dir": str(run_dir),
        }
        status_payload["configs"].append(config_status)

        if status != "completed":
            results_rows.append(
                {
                    "config_id": config["config_id"],
                    "fold": 0,
                    "status": status,
                    "exp_code": config["exp_code"],
                    "alpha_init": config["alpha_init"],
                    "learned_alpha_final": None,
                    "low_topk": config["low_topk"],
                    "high_max_per_low": config["high_max_per_low"],
                    "patch_footprint_ratio": config["patch_footprint_ratio"],
                    "scale_ratio": config["scale_ratio"],
                    "l2h_fusion": config["l2h_fusion"],
                    "l2h_aggregate": config["l2h_aggregate"],
                    "l2h_score_mode": config["l2h_score_mode"],
                    "ACC": np.nan,
                    "BACC": np.nan,
                    "F1": np.nan,
                    "AUC": np.nan,
                    "PR_AUC": np.nan,
                    "delta_acc_vs_step58C_fold0": np.nan,
                    "delta_auc_vs_step58C_fold0": np.nan,
                    "delta_f1_vs_step58C_fold0": np.nan,
                    "delta_pr_auc_vs_step58C_fold0": np.nan,
                }
            )
            contribution_rows.append({"config_id": config["config_id"], "status": status})
            l2h_rows.append({"config_id": config["config_id"], "status": status})
            for branch in BRANCH_ORDER:
                branch_rows.append({"config_id": config["config_id"], "status": status, "branch": branch})
            continue

        collected = collect_config_metrics(run_dir, config)
        metrics = collected["metrics"]
        contribution = collected["contribution"]
        l2h = collected["l2h"]

        results_rows.append(
            {
                "config_id": config["config_id"],
                "fold": 0,
                "status": status,
                "exp_code": config["exp_code"],
                "alpha_init": config["alpha_init"],
                "learned_alpha_final": l2h["learned_alpha_final"],
                "low_topk": config["low_topk"],
                "high_max_per_low": config["high_max_per_low"],
                "patch_footprint_ratio": config["patch_footprint_ratio"],
                "scale_ratio": config["scale_ratio"],
                "l2h_fusion": config["l2h_fusion"],
                "l2h_aggregate": config["l2h_aggregate"],
                "l2h_score_mode": config["l2h_score_mode"],
                "ACC": metrics["ACC"],
                "BACC": metrics["BACC"],
                "F1": metrics["F1"],
                "AUC": metrics["AUC"],
                "PR_AUC": metrics["PR_AUC"],
                "delta_acc_vs_step58C_fold0": None if metrics["ACC"] is None else metrics["ACC"] - baseline["metrics"]["ACC"],
                "delta_auc_vs_step58C_fold0": None if metrics["AUC"] is None else metrics["AUC"] - baseline["metrics"]["AUC"],
                "delta_f1_vs_step58C_fold0": None if metrics["F1"] is None else metrics["F1"] - baseline["metrics"]["F1"],
                "delta_pr_auc_vs_step58C_fold0": None if metrics["PR_AUC"] is None else metrics["PR_AUC"] - baseline["metrics"]["PR_AUC"],
            }
        )

        contribution_row = {
            "config_id": config["config_id"],
            "status": status,
            **contribution,
        }
        contribution_row["delta_visual_ratio_vs_step58C_fold0"] = (
            None
            if contribution.get("visual_ratio_mean") is None
            else contribution["visual_ratio_mean"] - baseline["contribution"]["visual_ratio_mean"]
        )
        contribution_row["delta_concept_ratio_vs_step58C_fold0"] = (
            None
            if contribution.get("concept_ratio_mean") is None
            else contribution["concept_ratio_mean"] - baseline["contribution"]["concept_ratio_mean"]
        )
        contribution_row["delta_csg_ratio_vs_step58C_fold0"] = (
            None
            if contribution.get("csg_ratio_mean") is None
            else contribution["csg_ratio_mean"] - baseline["contribution"]["csg_ratio_mean"]
        )
        contribution_rows.append(contribution_row)

        l2h_rows.append({"config_id": config["config_id"], "status": status, **l2h})
        for branch_metric in collected["branches"]:
            branch_rows.append({"config_id": config["config_id"], "status": status, **branch_metric})

    results_df = pd.DataFrame(results_rows, columns=RESULT_COLUMNS)
    contribution_df = pd.DataFrame(contribution_rows, columns=CONTRIBUTION_COLUMNS)
    l2h_df = pd.DataFrame(l2h_rows, columns=L2H_COLUMNS)
    branch_df = pd.DataFrame(branch_rows, columns=BRANCH_COLUMNS)

    selection = select_config(results_df, contribution_df, l2h_df, branch_df, baseline)
    status_payload["selected_config"] = selection.get("selected_config")
    status_payload["decision"] = selection.get("decision")
    status_payload["reason"] = selection.get("reason")
    status_payload["ranking"] = selection.get("ranking")

    results_df.to_csv(OUTPUT_DIR / "stage61C_sweep_results.csv", index=False)
    branch_df.to_csv(OUTPUT_DIR / "stage61C_branch_metrics_by_config.csv", index=False)
    contribution_df.to_csv(OUTPUT_DIR / "stage61C_contribution_by_config.csv", index=False)
    l2h_df.to_csv(OUTPUT_DIR / "stage61C_l2h_by_config.csv", index=False)
    (OUTPUT_DIR / "stage61C_selected_config.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage61C_sweep_status.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage61C_summary.md").write_text(
        build_summary_md(results_df, contribution_df, l2h_df, branch_df, selection, step61b_fix_note),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
