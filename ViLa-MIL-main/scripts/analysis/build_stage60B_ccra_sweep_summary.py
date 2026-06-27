from __future__ import annotations

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


OUTPUT_DIR = ROOT / "results_stage60B_ccra_sweep"
SWEEP_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage60B_ccra_sweep.sh"
STAGE58C_OUTPUT_DIR = ROOT / "results_stage58C_residual_constrained_configD_5fold"
STAGE59C_OUTPUT_DIR = ROOT / "results_stage59C_dynamic_csg_configA_5fold"
STAGE57B_AUDIT_SCRIPT = ROOT / "scripts" / "analysis" / "build_stage57B_logit_contribution_audit.py"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))

DEFAULT_CONFIG_IDS = ["A", "B", "C", "D", "E", "F"]
OPTIONAL_CONFIG_IDS = ["G"]
CONFIGS = [
    {
        "config_id": "A",
        "exp_code": "rce_v2_rcD_ccra_A_fold0",
        "ccra_alpha_init": 0.001,
        "ccra_scale": 0.5,
        "ccra_norm": "layernorm",
        "ccra_dropout": 0.0,
        "ccra_clip": 5.0,
        "ccra_detach_prompt": False,
        "scheduled_by_default": True,
    },
    {
        "config_id": "B",
        "exp_code": "rce_v2_rcD_ccra_B_fold0",
        "ccra_alpha_init": 0.001,
        "ccra_scale": 1.0,
        "ccra_norm": "layernorm",
        "ccra_dropout": 0.0,
        "ccra_clip": 5.0,
        "ccra_detach_prompt": False,
        "scheduled_by_default": True,
    },
    {
        "config_id": "C",
        "exp_code": "rce_v2_rcD_ccra_C_fold0",
        "ccra_alpha_init": 0.01,
        "ccra_scale": 1.0,
        "ccra_norm": "layernorm",
        "ccra_dropout": 0.0,
        "ccra_clip": 5.0,
        "ccra_detach_prompt": False,
        "scheduled_by_default": True,
    },
    {
        "config_id": "D",
        "exp_code": "rce_v2_rcD_ccra_D_fold0",
        "ccra_alpha_init": 0.05,
        "ccra_scale": 1.0,
        "ccra_norm": "layernorm",
        "ccra_dropout": 0.0,
        "ccra_clip": 5.0,
        "ccra_detach_prompt": False,
        "scheduled_by_default": True,
    },
    {
        "config_id": "E",
        "exp_code": "rce_v2_rcD_ccra_E_fold0",
        "ccra_alpha_init": 0.01,
        "ccra_scale": 0.5,
        "ccra_norm": "layernorm",
        "ccra_dropout": 0.1,
        "ccra_clip": 5.0,
        "ccra_detach_prompt": False,
        "scheduled_by_default": True,
    },
    {
        "config_id": "F",
        "exp_code": "rce_v2_rcD_ccra_F_fold0",
        "ccra_alpha_init": 0.01,
        "ccra_scale": 1.0,
        "ccra_norm": "none",
        "ccra_dropout": 0.0,
        "ccra_clip": 5.0,
        "ccra_detach_prompt": False,
        "scheduled_by_default": True,
    },
    {
        "config_id": "G",
        "exp_code": "rce_v2_rcD_ccra_G_fold0",
        "ccra_alpha_init": 0.01,
        "ccra_scale": 1.0,
        "ccra_norm": "layernorm",
        "ccra_dropout": 0.0,
        "ccra_clip": 5.0,
        "ccra_detach_prompt": True,
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
BRANCH_COLUMNS = ["config_id", "status", "branch", "ACC", "BACC", "F1", "AUC", "PR_AUC"]
CONTRIBUTION_COLUMNS = [
    "config_id",
    "status",
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
CCRA_COLUMNS = [
    "config_id",
    "status",
    "ccra_enabled",
    "alpha_init",
    "learned_alpha_final",
    "ccra_scale",
    "ccra_norm",
    "ccra_dropout",
    "ccra_clip",
    "ccra_detach_prompt",
    "low_ccra_delta_abs_mean",
    "high_ccra_delta_abs_mean",
    "low_original_region_norm",
    "high_original_region_norm",
    "low_fused_region_norm",
    "high_fused_region_norm",
    "low_ccra_region_norm",
    "high_ccra_region_norm",
    "anomaly_count",
]
CCRA_KEYS = [
    "ccra_enabled",
    "ccra_alpha",
    "ccra_scale",
    "ccra_query_source",
    "ccra_norm",
    "low_ccra_delta_abs_mean",
    "high_ccra_delta_abs_mean",
    "low_original_region_norm",
    "high_original_region_norm",
    "low_fused_region_norm",
    "high_fused_region_norm",
    "low_ccra_region_norm",
    "high_ccra_region_norm",
]


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


def relative_path_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def detect_supported_ccra_features() -> dict[str, bool]:
    model_text = (ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py").read_text(encoding="utf-8")
    return {
        "layernorm": 'self.rce_ccra_norm not in {"layernorm", "none"}' in model_text,
        "none": '"none"' in model_text,
        "detach_prompt": "self.rce_ccra_detach_prompt" in model_text,
    }


def determine_run_status(
    run_dir: Path,
    config: dict[str, object],
    support: dict[str, bool],
) -> tuple[str, str | None]:
    config_norm = str(config["ccra_norm"])
    if not support.get(config_norm, False):
        return "skipped", f"norm_not_supported:{config_norm}"
    if bool(config["ccra_detach_prompt"]) and not support.get("detach_prompt", False):
        return "skipped", "detach_prompt_not_supported"
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
        raise ValueError(f"Unsupported task for Step60B summary: {task}")

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
        rce_use_ccra=to_bool(settings.get("rce_use_ccra", False)),
        rce_ccra_mode=str(settings.get("rce_ccra_mode", "concept_query_residual")),
        rce_ccra_alpha_init=float(settings.get("rce_ccra_alpha_init", 0.0)),
        rce_ccra_scale=float(settings.get("rce_ccra_scale", 1.0)),
        rce_ccra_num_queries=int(settings.get("rce_ccra_num_queries", 0)),
        rce_ccra_query_source=str(settings.get("rce_ccra_query_source", "prompt_mean")),
        rce_ccra_detach_prompt=to_bool(settings.get("rce_ccra_detach_prompt", False)),
        rce_ccra_norm=str(settings.get("rce_ccra_norm", "layernorm")),
        rce_ccra_dropout=float(settings.get("rce_ccra_dropout", 0.0)),
        rce_ccra_clip=float(settings.get("rce_ccra_clip", 5.0)),
        scale_mode=str(settings.get("scale_mode", "dual")),
        finetune_text_encoder=False,
        enable_logit_breakdown_audit=True,
    )


def run_stage57b_audit(run_dir: Path, config: dict[str, object]) -> dict[str, object]:
    audit_dir = OUTPUT_DIR / "audits" / f"config_{config['config_id']}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON_BIN),
        str(STAGE57B_AUDIT_SCRIPT),
        "--run_dir",
        str(run_dir),
        "--fold",
        "0",
        "--split",
        "test",
        "--output_dir",
        str(audit_dir),
        "--rce_use_ccra",
        "--rce_ccra_mode",
        str(config["ccra_mode"]) if "ccra_mode" in config else "concept_query_residual",
        "--rce_ccra_alpha_init",
        str(config["ccra_alpha_init"]),
        "--rce_ccra_scale",
        str(config["ccra_scale"]),
        "--rce_ccra_norm",
        str(config["ccra_norm"]),
        "--rce_ccra_dropout",
        str(config["ccra_dropout"]),
        "--rce_ccra_clip",
        str(config["ccra_clip"]),
        "--rce_ccra_query_source",
        "prompt_mean",
    ]
    if bool(config["ccra_detach_prompt"]):
        command.append("--rce_ccra_detach_prompt")
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
        "audit_dir": audit_dir,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def apply_ccra_config_overrides(settings: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    merged = dict(settings)
    merged.update(
        {
            "rce_use_ccra": True,
            "rce_ccra_mode": "concept_query_residual",
            "rce_ccra_alpha_init": config["ccra_alpha_init"],
            "rce_ccra_scale": config["ccra_scale"],
            "rce_ccra_num_queries": int(config.get("ccra_num_queries", 0)),
            "rce_ccra_query_source": "prompt_mean",
            "rce_ccra_detach_prompt": bool(config["ccra_detach_prompt"]),
            "rce_ccra_norm": config["ccra_norm"],
            "rce_ccra_dropout": config["ccra_dropout"],
            "rce_ccra_clip": config["ccra_clip"],
            "rce_use_dynamic_csg": False,
        }
    )
    return merged


def collect_ccra_metrics(run_dir: Path, config: dict[str, object]) -> dict[str, object]:
    settings = read_experiment_settings(run_dir)
    if not settings:
        return {}
    settings = apply_ccra_config_overrides(settings, config)

    dataset = build_dataset(settings)
    split_dir = Path(str(settings["split_dir"]))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    _, _, test_split = dataset.return_splits(
        from_id=False,
        csv_path=str(split_dir / "splits_0.csv"),
    )
    loader = get_simple_loader(test_split, mode=str(settings.get("mode", "transformer")))
    model = RCE_MIL_BiomedCLIP(
        config=build_model_config(settings),
        num_classes=int(settings["n_classes"]),
    )
    try:
        state_dict = torch.load(run_dir / "s_0_checkpoint.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(run_dir / "s_0_checkpoint.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.relocate()
    model.eval()
    if hasattr(model, "set_logit_breakdown_audit"):
        model.set_logit_breakdown_audit(True)
    device = next(model.parameters()).device

    ccra_samples: list[dict[str, float | None]] = []
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
            breakdown = getattr(model, "last_ccra_breakdown", None) or {}
            ccra_samples.append({key: safe_float(breakdown.get(key)) for key in CCRA_KEYS})

    learned_alpha = safe_float(getattr(model, "rce_ccra_alpha", None))
    if learned_alpha is None and hasattr(model, "rce_ccra_alpha") and model.rce_ccra_alpha is not None:
        learned_alpha = safe_float(model.rce_ccra_alpha.detach())

    summary: dict[str, object] = {
        "ccra_enabled": None,
        "alpha_init": safe_float(settings.get("rce_ccra_alpha_init")),
        "learned_alpha_final": learned_alpha,
        "ccra_scale": safe_float(settings.get("rce_ccra_scale")),
        "ccra_norm": settings.get("rce_ccra_norm"),
        "ccra_dropout": safe_float(settings.get("rce_ccra_dropout")),
        "ccra_clip": safe_float(settings.get("rce_ccra_clip")),
        "ccra_detach_prompt": to_bool(settings.get("rce_ccra_detach_prompt", False)),
        "anomaly_count": anomaly_count,
    }
    for key in CCRA_KEYS:
        values = [sample.get(key) for sample in ccra_samples]
        if key == "ccra_enabled":
            non_null = [value for value in values if value is not None]
            summary[key] = None if not non_null else round(float(np.mean(non_null)), 6)
            continue
        numeric = [value for value in values if value is not None]
        summary[key] = None if not numeric else float(np.mean(numeric))
    return summary


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


def read_step59c_secondary_reference() -> dict[str, object] | None:
    fold_metrics = read_csv_if_exists(STAGE59C_OUTPUT_DIR / "stage59C_fold_metrics.csv")
    if fold_metrics is None or fold_metrics.empty:
        return None
    numeric = fold_metrics.copy()
    for column in ["ACC", "AUC", "F1", "Balanced_ACC", "PR_AUC"]:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return {
        "ACC_mean": safe_float(numeric["ACC"].mean()),
        "AUC_mean": safe_float(numeric["AUC"].mean()),
        "F1_mean": safe_float(numeric["F1"].mean()),
        "BACC_mean": safe_float(numeric["Balanced_ACC"].mean()),
        "PR_AUC_mean": safe_float(numeric["PR_AUC"].mean()),
    }


def build_run_commands_text() -> str:
    display_root = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
    if not (display_root / "main.py").is_file():
        display_root = ROOT
    return "\n".join(
        [
            f"cd {display_root}",
            "RUN_TRAIN=1 bash scripts/experiments/run_stage60B_ccra_sweep.sh",
            "RUN_TRAIN=1 CONFIGS=extended bash scripts/experiments/run_stage60B_ccra_sweep.sh",
            "",
            "# Refresh Step60B summary",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage60B_ccra_sweep_summary.py",
        ]
    )


def select_config(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    ccra_df: pd.DataFrame,
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

    for _, row in completed.iterrows():
        config_id = row["config_id"]
        contrib_row = contribution_df.loc[contribution_df["config_id"] == config_id]
        ccra_row = ccra_df.loc[ccra_df["config_id"] == config_id]
        full_row = branch_df.loc[(branch_df["config_id"] == config_id) & (branch_df["branch"] == "full")]
        concept_row = branch_df.loc[(branch_df["config_id"] == config_id) & (branch_df["branch"] == "concept_only")]
        if contrib_row.empty or ccra_row.empty or full_row.empty:
            continue
        contrib_payload = contrib_row.iloc[0]
        ccra_payload = ccra_row.iloc[0]
        concept_payload = concept_row.iloc[0] if not concept_row.empty else None

        acc = safe_float(row["ACC"]) or 0.0
        auc = safe_float(row["AUC"]) or 0.0
        f1 = safe_float(row["F1"]) or 0.0
        bacc = safe_float(row["BACC"]) or 0.0
        pr_auc = safe_float(row["PR_AUC"]) or 0.0
        visual_ratio = safe_float(contrib_payload["visual_ratio_mean"])
        concept_ratio = safe_float(contrib_payload["concept_ratio_mean"])
        low_delta = safe_float(ccra_payload["low_ccra_delta_abs_mean"])
        high_delta = safe_float(ccra_payload["high_ccra_delta_abs_mean"])
        learned_alpha = safe_float(ccra_payload["learned_alpha_final"])
        anomaly_count = int(safe_float(ccra_payload["anomaly_count"]) or 0)

        full_acc = safe_float(full_row.iloc[0]["ACC"]) or 0.0
        concept_acc = safe_float(concept_payload["ACC"]) or 0.0 if concept_payload is not None else 0.0
        concept_auc = safe_float(concept_payload["AUC"]) or 0.0 if concept_payload is not None else 0.0

        acc_delta = acc - (baseline_metrics["ACC"] or 0.0)
        auc_delta = auc - (baseline_metrics["AUC"] or 0.0)
        f1_delta = f1 - (baseline_metrics["F1"] or 0.0)
        bacc_delta = bacc - (baseline_metrics["BACC"] or 0.0)
        pr_auc_delta = pr_auc - (baseline_metrics["PR_AUC"] or 0.0)
        visual_delta = (visual_ratio or 0.0) - (baseline_contrib["visual_ratio_mean"] or 0.0)
        concept_delta = (concept_ratio or 0.0) - (baseline_contrib["concept_ratio_mean"] or 0.0)

        perf_ok = (
            acc_delta >= -0.015
            and auc_delta >= -0.015
            and f1_delta >= -0.02
            and bacc_delta >= -0.02
            and pr_auc_delta >= -0.02
        )
        visual_ok = visual_ratio is not None and visual_ratio <= (baseline_contrib["visual_ratio_mean"] or 0.0) + 0.05
        concept_ok = concept_ratio is not None and concept_ratio >= (baseline_contrib["concept_ratio_mean"] or 0.0) - 0.05
        alpha_nonzero = learned_alpha is not None and abs(learned_alpha) > 1e-6
        delta_nonzero = any(value is not None and abs(value) > 1e-8 for value in [low_delta, high_delta])
        branch_gain = concept_acc - (
            safe_float(
                baseline["branch_df"].loc[baseline["branch_df"]["branch"] == "concept_only", "ACC"].iloc[0]
            )
            or 0.0
        )
        branch_auc_gain = concept_auc - (
            safe_float(
                baseline["branch_df"].loc[baseline["branch_df"]["branch"] == "concept_only", "AUC"].iloc[0]
            )
            or 0.0
        )
        full_branch_ok = full_acc >= (safe_float(
            baseline["branch_df"].loc[baseline["branch_df"]["branch"] == "full", "ACC"].iloc[0]
        ) or 0.0) - 0.015
        no_anomaly = anomaly_count == 0

        score = (
            acc
            + 0.5 * auc
            + 0.5 * f1
            + 0.25 * bacc
            + 0.25 * pr_auc
            + 3.0 * max(0.0, branch_gain)
            + 2.0 * max(0.0, branch_auc_gain)
            + 10.0 * max(0.0, concept_delta)
            + 5.0 * max(0.0, low_delta or 0.0)
            + 5.0 * max(0.0, high_delta or 0.0)
            - 10.0 * max(0.0, visual_delta)
            - 2.0 * max(0.0, -acc_delta)
            - 2.0 * max(0.0, -auc_delta)
            - 2.0 * max(0.0, -f1_delta)
        )
        if not alpha_nonzero:
            score -= 0.75
        if not delta_nonzero:
            score -= 0.75
        if anomaly_count > 0:
            score -= 5.0

        ranking.append(
            {
                "config_id": config_id,
                "score": score,
                "perf_ok": perf_ok,
                "visual_ok": visual_ok,
                "concept_ok": concept_ok,
                "alpha_nonzero": alpha_nonzero,
                "delta_nonzero": delta_nonzero,
                "full_branch_ok": full_branch_ok,
                "no_anomaly": no_anomaly,
                "acc_delta": acc_delta,
                "auc_delta": auc_delta,
                "f1_delta": f1_delta,
                "bacc_delta": bacc_delta,
                "pr_auc_delta": pr_auc_delta,
                "visual_ratio_delta": visual_delta,
                "concept_ratio_delta": concept_delta,
                "concept_acc_delta_vs_baseline": branch_gain,
                "concept_auc_delta_vs_baseline": branch_auc_gain,
                "low_ccra_delta_abs_mean": low_delta,
                "high_ccra_delta_abs_mean": high_delta,
                "learned_alpha_final": learned_alpha,
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
    strong_candidates = [
        item
        for item in ranking
        if item["perf_ok"]
        and item["visual_ok"]
        and item["concept_ok"]
        and item["alpha_nonzero"]
        and item["delta_nonzero"]
        and item["full_branch_ok"]
        and item["no_anomaly"]
    ]
    if strong_candidates:
        return {
            "decision": "selected_for_step60C_5fold",
            "selected_config": strong_candidates[0]["config_id"],
            "ranking": ranking,
            "reason": "performance_close_and_ccra_observable",
        }

    performance_candidates = [
        item
        for item in ranking
        if item["perf_ok"] and item["visual_ok"] and item["concept_ok"] and item["no_anomaly"]
    ]
    if performance_candidates:
        return {
            "decision": "performance_candidate_but_ccra_weak",
            "selected_config": performance_candidates[0]["config_id"],
            "ranking": ranking,
            "reason": "performance_ok_but_ccra_signal_weak",
        }

    return {
        "decision": "no_ccra_config_selected",
        "selected_config": None,
        "ranking": ranking,
        "reason": "no_config_meets_ccra_selection_rules",
    }


def build_summary_md(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    ccra_df: pd.DataFrame,
    selection: dict[str, object],
    secondary_reference: dict[str, object] | None,
) -> str:
    completed = results_df.loc[results_df["status"] == "completed", "config_id"].tolist()
    pending = results_df.loc[results_df["status"] == "pending", "config_id"].tolist()
    skipped = results_df.loc[results_df["status"] == "skipped", "config_id"].tolist()

    best_acc = results_df.loc[results_df["ACC"].astype(float).idxmax()] if results_df["ACC"].notna().any() else None
    best_auc = results_df.loc[results_df["AUC"].astype(float).idxmax()] if results_df["AUC"].notna().any() else None
    best_f1 = results_df.loc[results_df["F1"].astype(float).idxmax()] if results_df["F1"].notna().any() else None

    ccra_completed = ccra_df.loc[ccra_df["status"] == "completed"].copy()
    contribution_completed = contribution_df.loc[contribution_df["status"] == "completed"].copy()
    alpha_nonzero = False
    delta_nonzero = False
    visual_low = False
    concept_high = False
    if not ccra_completed.empty:
        alpha_nonzero = bool(
            (pd.to_numeric(ccra_completed["learned_alpha_final"], errors="coerce").fillna(0.0).abs() > 1e-6).any()
        )
        delta_nonzero = bool(
            (
                pd.to_numeric(ccra_completed["low_ccra_delta_abs_mean"], errors="coerce").fillna(0.0).abs() > 1e-8
            ).any()
            or (
                pd.to_numeric(ccra_completed["high_ccra_delta_abs_mean"], errors="coerce").fillna(0.0).abs() > 1e-8
            ).any()
        )
    if not contribution_completed.empty:
        visual_low = bool((pd.to_numeric(contribution_completed["visual_ratio_mean"], errors="coerce") < 0.5).all())
        concept_high = bool((pd.to_numeric(contribution_completed["concept_ratio_mean"], errors="coerce") > 0.5).all())

    lines = [
        "# Step60B CCRA fold0 sweep",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑：是，且仅做最小修复范围外的兼容扩展？否。"
        " 仅扩展了 `build_stage57B_logit_contribution_audit.py` 对 CCRA 参数的重建支持，便于读取 CCRA checkpoint；未改动模型逻辑。",
        f"3. 本 Step 实际跑了哪些 CCRA config：{completed or ['无']}。",
        f"4. 哪些 config completed / pending / skipped：completed={completed or ['无']}；pending={pending or ['无']}；skipped={skipped or ['无']}。",
        "5. 哪个 config 的 ACC/AUC/F1 最好："
        f" ACC={best_acc['config_id'] if best_acc is not None else 'NA'}，"
        f" AUC={best_auc['config_id'] if best_auc is not None else 'NA'}，"
        f" F1={best_f1['config_id'] if best_f1 is not None else 'NA'}。",
        f"6. CCRA 的 learned alpha 是否出现非零：{'是' if alpha_nonzero else '否'}。",
        f"7. CCRA 的 low/high delta 是否出现非零：{'是' if delta_nonzero else '否'}。",
        f"8. visual_ratio 是否仍保持低水平：{'是' if visual_low else '否'}。",
        f"9. concept_ratio 是否保持较高：{'是' if concept_high else '否'}。",
        "10. 是否存在值得进入 Step60C 5-fold 的候选配置："
        f" {'是' if selection.get('selected_config') is not None else '否'}。",
        "11. 推荐进入 Step60C 的 selected config 是哪个："
        f" `{selection.get('selected_config')}`。"
        if selection.get("selected_config") is not None
        else "11. 推荐进入 Step60C 的 selected config 是哪个：NA。",
        "12. 下一步建议是什么："
        + (
            " 进入 Step60C 5-fold。"
            if selection.get("decision") == "selected_for_step60C_5fold"
            else " CCRA 可保留为备选，但其可观察贡献仍偏弱。"
            if selection.get("decision") == "performance_candidate_but_ccra_weak"
            else " 暂不进入 Step60C，先停止 CCRA 主线推进。"
            if selection.get("decision") == "no_ccra_config_selected"
            else " 先完成 fold0 sweep 训练。"
        ),
        "",
        "## Selection Result",
        "",
        f"- decision: `{selection.get('decision')}`",
        f"- reason: `{selection.get('reason')}`",
    ]

    for item in selection.get("ranking", [])[:6]:
        lines.append(
            f"- rank {item['config_id']}: score={item['score']:.6f}, "
            f"acc_delta={item['acc_delta']:+.6f}, auc_delta={item['auc_delta']:+.6f}, "
            f"f1_delta={item['f1_delta']:+.6f}, visual_ratio_delta={item['visual_ratio_delta']:+.6f}, "
            f"concept_ratio_delta={item['concept_ratio_delta']:+.6f}, "
            f"alpha={safe_float(item['learned_alpha_final'])}, "
            f"low_delta={safe_float(item['low_ccra_delta_abs_mean'])}, "
            f"high_delta={safe_float(item['high_ccra_delta_abs_mean'])}"
        )

    if secondary_reference is not None:
        lines.extend(
            [
                "",
                "## Secondary Reference",
                "",
                f"- Step59C mean ACC/AUC/F1/BACC/PR-AUC: "
                f"{secondary_reference.get('ACC_mean')}, "
                f"{secondary_reference.get('AUC_mean')}, "
                f"{secondary_reference.get('F1_mean')}, "
                f"{secondary_reference.get('BACC_mean')}, "
                f"{secondary_reference.get('PR_AUC_mean')}",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stage60B_run_commands.txt").write_text(
        build_run_commands_text() + "\n",
        encoding="utf-8",
    )

    support = detect_supported_ccra_features()
    baseline = read_step58c_fold0_baseline()
    secondary_reference = read_step59c_secondary_reference()

    results_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    ccra_rows: list[dict[str, object]] = []
    audit_invocations: dict[str, object] = {}
    warnings: list[str] = []

    for config in CONFIGS:
        run_dir = OUTPUT_DIR / f"{config['exp_code']}_s1"
        status, status_reason = determine_run_status(run_dir, config, support)
        metrics = {"ACC": None, "BACC": None, "F1": None, "AUC": None, "PR_AUC": None}
        learned_alpha_final = None

        if status == "completed":
            fold_df = read_csv_if_exists(run_dir / "fold_summary.csv")
            if fold_df is not None and not fold_df.empty:
                row = fold_df.iloc[0]
                metrics = {
                    "ACC": safe_float(row.get("test_acc")),
                    "BACC": safe_float(row.get("balanced_acc")),
                    "F1": safe_float(row.get("test_f1")),
                    "AUC": safe_float(row.get("test_auc")),
                    "PR_AUC": safe_float(row.get("pr_auc")),
                }

            audit_result = run_stage57b_audit(run_dir, config)
            audit_invocations[config["config_id"]] = {
                "ok": audit_result["ok"],
                "returncode": audit_result["returncode"],
                "audit_dir": relative_path_str(audit_result["audit_dir"]),
            }
            if not audit_result["ok"]:
                warnings.append(f"audit_failed:{config['config_id']}")
            audit_dir = audit_result["audit_dir"]
            branch_df = read_csv_if_exists(audit_dir / "stage57B_branch_metrics.csv")
            margin_df = read_csv_if_exists(audit_dir / "stage57B_margin_stats.csv")
            status_json_path = audit_dir / "stage57B_audit_status.json"
            audit_status = (
                json.loads(status_json_path.read_text(encoding="utf-8"))
                if status_json_path.is_file()
                else {}
            )
            if branch_df is not None:
                for _, branch_row in branch_df.iterrows():
                    branch_rows.append(
                        {
                            "config_id": config["config_id"],
                            "status": status,
                            "branch": branch_row["branch"],
                            "ACC": safe_float(branch_row.get("acc")),
                            "BACC": safe_float(branch_row.get("balanced_acc")),
                            "F1": safe_float(branch_row.get("macro_f1")),
                            "AUC": safe_float(branch_row.get("auc")),
                            "PR_AUC": safe_float(branch_row.get("pr_auc")),
                        }
                    )

            margin_map = {}
            if margin_df is not None:
                margin_map = {str(row["metric_name"]): row.to_dict() for _, row in margin_df.iterrows()}
            contribution_rows.append(
                {
                    "config_id": config["config_id"],
                    "status": status,
                    "visual_ratio_mean": safe_float(margin_map.get("visual_contribution_ratio", {}).get("mean")),
                    "visual_ratio_median": safe_float(margin_map.get("visual_contribution_ratio", {}).get("median")),
                    "visual_ratio_gt_0_5_percent": safe_float(
                        audit_status.get("visual_details", {}).get("pct_visual_ratio_gt_0_5")
                    ),
                    "concept_ratio_mean": safe_float(margin_map.get("concept_contribution_ratio", {}).get("mean")),
                    "concept_ratio_median": safe_float(margin_map.get("concept_contribution_ratio", {}).get("median")),
                    "csg_ratio_mean": safe_float(margin_map.get("csg_contribution_ratio", {}).get("mean")),
                    "csg_ratio_median": safe_float(margin_map.get("csg_contribution_ratio", {}).get("median")),
                    "full_margin_mean": safe_float(margin_map.get("full_margin", {}).get("mean")),
                    "concept_margin_mean": safe_float(margin_map.get("concept_margin", {}).get("mean")),
                    "visual_margin_mean": safe_float(margin_map.get("visual_margin", {}).get("mean")),
                    "csg_margin_mean": safe_float(margin_map.get("csg_margin", {}).get("mean")),
                }
            )

            ccra_payload = collect_ccra_metrics(run_dir, config)
            learned_alpha_final = ccra_payload.get("learned_alpha_final")
            ccra_rows.append(
                {
                    "config_id": config["config_id"],
                    "status": status,
                    "ccra_enabled": ccra_payload.get("ccra_enabled"),
                    "alpha_init": ccra_payload.get("alpha_init"),
                    "learned_alpha_final": ccra_payload.get("learned_alpha_final"),
                    "ccra_scale": ccra_payload.get("ccra_scale"),
                    "ccra_norm": ccra_payload.get("ccra_norm") or config["ccra_norm"],
                    "ccra_dropout": ccra_payload.get("ccra_dropout"),
                    "ccra_clip": ccra_payload.get("ccra_clip"),
                    "ccra_detach_prompt": ccra_payload.get("ccra_detach_prompt"),
                    "low_ccra_delta_abs_mean": ccra_payload.get("low_ccra_delta_abs_mean"),
                    "high_ccra_delta_abs_mean": ccra_payload.get("high_ccra_delta_abs_mean"),
                    "low_original_region_norm": ccra_payload.get("low_original_region_norm"),
                    "high_original_region_norm": ccra_payload.get("high_original_region_norm"),
                    "low_fused_region_norm": ccra_payload.get("low_fused_region_norm"),
                    "high_fused_region_norm": ccra_payload.get("high_fused_region_norm"),
                    "low_ccra_region_norm": ccra_payload.get("low_ccra_region_norm"),
                    "high_ccra_region_norm": ccra_payload.get("high_ccra_region_norm"),
                    "anomaly_count": ccra_payload.get("anomaly_count"),
                }
            )
        else:
            contribution_rows.append(
                {
                    "config_id": config["config_id"],
                    "status": status,
                    "visual_ratio_mean": None,
                    "visual_ratio_median": None,
                    "visual_ratio_gt_0_5_percent": None,
                    "concept_ratio_mean": None,
                    "concept_ratio_median": None,
                    "csg_ratio_mean": None,
                    "csg_ratio_median": None,
                    "full_margin_mean": None,
                    "concept_margin_mean": None,
                    "visual_margin_mean": None,
                    "csg_margin_mean": None,
                }
            )
            ccra_rows.append(
                {
                    "config_id": config["config_id"],
                    "status": status,
                    "ccra_enabled": None,
                    "alpha_init": config["ccra_alpha_init"],
                    "learned_alpha_final": None,
                    "ccra_scale": config["ccra_scale"],
                    "ccra_norm": config["ccra_norm"],
                    "ccra_dropout": config["ccra_dropout"],
                    "ccra_clip": config["ccra_clip"],
                    "ccra_detach_prompt": config["ccra_detach_prompt"],
                    "low_ccra_delta_abs_mean": None,
                    "high_ccra_delta_abs_mean": None,
                    "low_original_region_norm": None,
                    "high_original_region_norm": None,
                    "low_fused_region_norm": None,
                    "high_fused_region_norm": None,
                    "low_ccra_region_norm": None,
                    "high_ccra_region_norm": None,
                    "anomaly_count": None,
                }
            )

        results_rows.append(
            {
                "config_id": config["config_id"],
                "fold": 0,
                "status": status,
                "status_reason": status_reason,
                "exp_code": config["exp_code"],
                "ccra_alpha_init": config["ccra_alpha_init"],
                "learned_alpha_final": learned_alpha_final,
                "ccra_scale": config["ccra_scale"],
                "ccra_norm": config["ccra_norm"],
                "ccra_dropout": config["ccra_dropout"],
                "ccra_clip": config["ccra_clip"],
                "ccra_detach_prompt": config["ccra_detach_prompt"],
                "ACC": metrics["ACC"],
                "BACC": metrics["BACC"],
                "F1": metrics["F1"],
                "AUC": metrics["AUC"],
                "PR_AUC": metrics["PR_AUC"],
                "delta_acc_vs_step58C_fold0": None if metrics["ACC"] is None else metrics["ACC"] - baseline["metrics"]["ACC"],
                "delta_auc_vs_step58C_fold0": None if metrics["AUC"] is None else metrics["AUC"] - baseline["metrics"]["AUC"],
                "delta_f1_vs_step58C_fold0": None if metrics["F1"] is None else metrics["F1"] - baseline["metrics"]["F1"],
                "delta_bacc_vs_step58C_fold0": None if metrics["BACC"] is None else metrics["BACC"] - baseline["metrics"]["BACC"],
                "delta_pr_auc_vs_step58C_fold0": None if metrics["PR_AUC"] is None else metrics["PR_AUC"] - baseline["metrics"]["PR_AUC"],
            }
        )

    results_df = pd.DataFrame(results_rows)
    branch_df = pd.DataFrame(branch_rows, columns=BRANCH_COLUMNS)
    contribution_df = pd.DataFrame(contribution_rows, columns=CONTRIBUTION_COLUMNS)
    ccra_df = pd.DataFrame(ccra_rows, columns=CCRA_COLUMNS)

    if not contribution_df.empty:
        contribution_df["delta_visual_ratio_vs_step58C_fold0"] = contribution_df["visual_ratio_mean"].apply(
            lambda value: None if pd.isna(value) else float(value) - baseline["contribution"]["visual_ratio_mean"]
        )
        contribution_df["delta_concept_ratio_vs_step58C_fold0"] = contribution_df["concept_ratio_mean"].apply(
            lambda value: None if pd.isna(value) else float(value) - baseline["contribution"]["concept_ratio_mean"]
        )
        contribution_df["delta_csg_ratio_vs_step58C_fold0"] = contribution_df["csg_ratio_mean"].apply(
            lambda value: None if pd.isna(value) else float(value) - baseline["contribution"]["csg_ratio_mean"]
        )

    selection = select_config(results_df, contribution_df, ccra_df, branch_df, baseline)

    results_df.to_csv(OUTPUT_DIR / "stage60B_sweep_results.csv", index=False)
    branch_df.to_csv(OUTPUT_DIR / "stage60B_branch_metrics_by_config.csv", index=False)
    contribution_df.to_csv(OUTPUT_DIR / "stage60B_contribution_by_config.csv", index=False)
    ccra_df.to_csv(OUTPUT_DIR / "stage60B_ccra_by_config.csv", index=False)
    (OUTPUT_DIR / "stage60B_selected_config.json").write_text(
        json.dumps(
            {
                **selection,
                "supported_ccra_features": support,
                "baseline_step58C_fold0": {
                    "metrics": {key: round_or_none(value) for key, value in baseline["metrics"].items()},
                    "contribution": {key: round_or_none(value) for key, value in baseline["contribution"].items()},
                },
                "secondary_reference_step59C": secondary_reference,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage60B_sweep_status.json").write_text(
        json.dumps(
            {
                "status": "completed" if (results_df["status"] == "completed").any() else "pending",
                "completed_configs": results_df.loc[results_df["status"] == "completed", "config_id"].tolist(),
                "pending_configs": results_df.loc[results_df["status"] == "pending", "config_id"].tolist(),
                "skipped_configs": results_df.loc[results_df["status"] == "skipped", "config_id"].tolist(),
                "supported_ccra_features": support,
                "audit_invocations": audit_invocations,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage60B_summary.md").write_text(
        build_summary_md(results_df, contribution_df, ccra_df, selection, secondary_reference),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
