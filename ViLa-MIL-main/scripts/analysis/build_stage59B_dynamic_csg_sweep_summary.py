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


OUTPUT_DIR = ROOT / "results_stage59B_dynamic_csg_sweep"
SWEEP_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage59B_dynamic_csg_sweep.sh"
STAGE58C_OUTPUT_DIR = ROOT / "results_stage58C_residual_constrained_configD_5fold"
STAGE57B_AUDIT_SCRIPT = ROOT / "scripts" / "analysis" / "build_stage57B_logit_contribution_audit.py"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))
DEFAULT_CONFIG_IDS = ["A", "B", "C", "D", "E"]
OPTIONAL_CONFIG_IDS = ["F", "G"]
ALL_CONFIG_IDS = DEFAULT_CONFIG_IDS + OPTIONAL_CONFIG_IDS
CONFIGS = [
    {
        "config_id": "A",
        "exp_code": "rce_v2_rcD_dynCSG_A_fold0",
        "alpha_init": 0.0,
        "dynamic_csg_scale": 1.0,
        "dynamic_csg_norm": "softmax",
        "dynamic_csg_clip": 5.0,
        "scheduled_by_default": True,
    },
    {
        "config_id": "B",
        "exp_code": "rce_v2_rcD_dynCSG_B_fold0",
        "alpha_init": 0.01,
        "dynamic_csg_scale": 1.0,
        "dynamic_csg_norm": "softmax",
        "dynamic_csg_clip": 5.0,
        "scheduled_by_default": True,
    },
    {
        "config_id": "C",
        "exp_code": "rce_v2_rcD_dynCSG_C_fold0",
        "alpha_init": 0.05,
        "dynamic_csg_scale": 1.0,
        "dynamic_csg_norm": "softmax",
        "dynamic_csg_clip": 5.0,
        "scheduled_by_default": True,
    },
    {
        "config_id": "D",
        "exp_code": "rce_v2_rcD_dynCSG_D_fold0",
        "alpha_init": 0.01,
        "dynamic_csg_scale": 2.0,
        "dynamic_csg_norm": "softmax",
        "dynamic_csg_clip": 5.0,
        "scheduled_by_default": True,
    },
    {
        "config_id": "E",
        "exp_code": "rce_v2_rcD_dynCSG_E_fold0",
        "alpha_init": 0.01,
        "dynamic_csg_scale": 1.0,
        "dynamic_csg_norm": "softmax",
        "dynamic_csg_clip": 1.0,
        "scheduled_by_default": True,
    },
    {
        "config_id": "F",
        "exp_code": "rce_v2_rcD_dynCSG_F_fold0",
        "alpha_init": 0.01,
        "dynamic_csg_scale": 1.0,
        "dynamic_csg_norm": "l1",
        "dynamic_csg_clip": 5.0,
        "scheduled_by_default": False,
    },
    {
        "config_id": "G",
        "exp_code": "rce_v2_rcD_dynCSG_G_fold0",
        "alpha_init": 0.01,
        "dynamic_csg_scale": 0.5,
        "dynamic_csg_norm": "none",
        "dynamic_csg_clip": 2.0,
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


def detect_supported_norms() -> dict[str, bool]:
    model_text = (ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py").read_text(encoding="utf-8")
    return {
        "softmax": 'self.rce_dynamic_csg_norm not in {"softmax", "l1", "none"}' in model_text,
        "l1": '"l1"' in model_text,
        "none": '"none"' in model_text,
    }


def determine_run_status(run_dir: Path, config: dict[str, object], supported_norms: dict[str, bool]) -> tuple[str, str | None]:
    config_norm = str(config["dynamic_csg_norm"])
    if not supported_norms.get(config_norm, False):
        return "skipped", f"norm_not_supported:{config_norm}"
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
        raise ValueError(f"Unsupported task for Step59B summary: {task}")

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


def run_stage57b_audit(run_dir: Path, config_id: str) -> dict[str, object]:
    config = next(item for item in CONFIGS if item["config_id"] == config_id)
    audit_dir = OUTPUT_DIR / "audits" / f"config_{config_id}"
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
        "--rce_use_dynamic_csg",
        "--rce_dynamic_csg_mode",
        "evidence_outer",
        "--rce_dynamic_csg_alpha_init",
        str(config["alpha_init"]),
        "--rce_dynamic_csg_scale",
        str(config["dynamic_csg_scale"]),
        "--rce_dynamic_csg_norm",
        str(config["dynamic_csg_norm"]),
        "--rce_dynamic_csg_clip",
        str(config["dynamic_csg_clip"]),
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
        "audit_dir": audit_dir,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def apply_dynamic_config_overrides(
    settings: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    merged = dict(settings)
    merged.update(
        {
            "rce_use_dynamic_csg": True,
            "rce_dynamic_csg_mode": "evidence_outer",
            "rce_dynamic_csg_alpha_init": config["alpha_init"],
            "rce_dynamic_csg_scale": config["dynamic_csg_scale"],
            "rce_dynamic_csg_norm": config["dynamic_csg_norm"],
            "rce_dynamic_csg_clip": config["dynamic_csg_clip"],
            "rce_dynamic_csg_detach_evidence": False,
        }
    )
    return merged


def collect_dynamic_metrics(run_dir: Path, config: dict[str, object]) -> dict[str, object]:
    settings = read_experiment_settings(run_dir)
    if not settings:
        return {}
    settings = apply_dynamic_config_overrides(settings, config)

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
            dynamic_samples.append(
                {
                    key: safe_float(breakdown.get(key))
                    for key in DYNAMIC_KEYS
                }
            )

    alpha_final = safe_float(getattr(model, "rce_dynamic_csg_alpha", None))
    if alpha_final is None and hasattr(model, "rce_dynamic_csg_alpha") and model.rce_dynamic_csg_alpha is not None:
        alpha_final = safe_float(model.rce_dynamic_csg_alpha.detach())

    summary: dict[str, object] = {
        "dynamic_csg_enabled": None,
        "alpha_init": safe_float(settings.get("rce_dynamic_csg_alpha_init")),
        "learned_alpha_final": alpha_final,
        "dynamic_csg_scale": safe_float(settings.get("rce_dynamic_csg_scale")),
        "dynamic_csg_norm": settings.get("rce_dynamic_csg_norm"),
        "dynamic_csg_clip": safe_float(settings.get("rce_dynamic_csg_clip")),
        "dynamic_csg_detach_evidence": to_bool(settings.get("rce_dynamic_csg_detach_evidence", False)),
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
    summary["csg_logits_delta_abs_mean_vs_static"] = summary.get("csg_logits_delta_abs_mean")
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


def build_run_commands_text() -> str:
    display_root = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
    if not (display_root / "main.py").is_file():
        display_root = ROOT
    return "\n".join(
        [
            f"cd {display_root}",
            "RUN_TRAIN=1 bash scripts/experiments/run_stage59B_dynamic_csg_sweep.sh",
            "RUN_TRAIN=1 CONFIGS=extended bash scripts/experiments/run_stage59B_dynamic_csg_sweep.sh",
            "",
            "# Refresh Step59B summary",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage59B_dynamic_csg_sweep_summary.py",
        ]
    )


def select_config(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
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
        dynamic_row = dynamic_df.loc[dynamic_df["config_id"] == config_id]
        full_row = branch_df.loc[(branch_df["config_id"] == config_id) & (branch_df["branch"] == "full")]
        concept_row = branch_df.loc[(branch_df["config_id"] == config_id) & (branch_df["branch"] == "concept_only")]
        if contrib_row.empty or dynamic_row.empty or full_row.empty:
            continue
        contrib_payload = contrib_row.iloc[0]
        dynamic_payload = dynamic_row.iloc[0]
        concept_payload = concept_row.iloc[0] if not concept_row.empty else None

        acc = safe_float(row["ACC"]) or 0.0
        auc = safe_float(row["AUC"]) or 0.0
        f1 = safe_float(row["F1"]) or 0.0
        visual_ratio = safe_float(contrib_payload["visual_ratio_mean"])
        concept_ratio = safe_float(contrib_payload["concept_ratio_mean"])
        csg_ratio = safe_float(contrib_payload["csg_ratio_mean"])
        csg_delta_abs = safe_float(dynamic_payload["csg_logits_delta_abs_mean"])
        learned_alpha = safe_float(dynamic_payload["learned_alpha_final"])
        anomaly_count = int(safe_float(dynamic_payload["anomaly_count"]) or 0)

        full_vs_concept_acc_gap = 0.0
        if concept_payload is not None:
            full_vs_concept_acc_gap = (safe_float(full_row.iloc[0]["ACC"]) or 0.0) - (
                safe_float(concept_payload["ACC"]) or 0.0
            )

        acc_delta = acc - (baseline_metrics["ACC"] or 0.0)
        auc_delta = auc - (baseline_metrics["AUC"] or 0.0)
        f1_delta = f1 - (baseline_metrics["F1"] or 0.0)
        csg_ratio_delta = (csg_ratio or 0.0) - (baseline_contrib["csg_ratio_mean"] or 0.0)
        visual_delta = (visual_ratio or 0.0) - (baseline_contrib["visual_ratio_mean"] or 0.0)
        concept_delta = (concept_ratio or 0.0) - (baseline_contrib["concept_ratio_mean"] or 0.0)

        perf_ok = acc_delta >= -0.01 and auc_delta >= -0.01 and f1_delta >= -0.02
        csg_improved = csg_ratio_delta > 0.0 or (csg_delta_abs or 0.0) > 1e-6
        visual_ok = visual_ratio is not None and visual_ratio <= (baseline_contrib["visual_ratio_mean"] or 0.0) + 0.05
        concept_ok = concept_ratio is not None and concept_ratio >= (baseline_contrib["concept_ratio_mean"] or 0.0) - 0.05
        branch_ok = full_vs_concept_acc_gap >= -0.01
        alpha_nonzero = learned_alpha is not None and abs(learned_alpha) > 1e-6
        no_anomaly = anomaly_count == 0

        score = (
            acc
            + 0.5 * auc
            + 0.5 * f1
            + 100.0 * max(0.0, csg_ratio_delta)
            + 20.0 * max(0.0, csg_delta_abs or 0.0)
            + 5.0 * max(0.0, concept_delta)
            - 5.0 * max(0.0, visual_delta)
            - 4.0 * max(0.0, -full_vs_concept_acc_gap)
            - 3.0 * max(0.0, -acc_delta)
            - 2.0 * max(0.0, -auc_delta)
            - 2.0 * max(0.0, -f1_delta)
        )
        if not alpha_nonzero and not (acc_delta > 0.01):
            score -= 0.5
        if anomaly_count > 0:
            score -= 5.0

        ranking.append(
            {
                "config_id": config_id,
                "score": score,
                "perf_ok": perf_ok,
                "csg_improved": csg_improved,
                "visual_ok": visual_ok,
                "concept_ok": concept_ok,
                "branch_ok": branch_ok,
                "alpha_nonzero": alpha_nonzero,
                "no_anomaly": no_anomaly,
                "acc_delta": acc_delta,
                "auc_delta": auc_delta,
                "f1_delta": f1_delta,
                "csg_ratio_delta": csg_ratio_delta,
                "csg_logits_delta_abs_mean": csg_delta_abs,
                "visual_ratio_delta": visual_delta,
                "concept_ratio_delta": concept_delta,
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
        and item["csg_improved"]
        and item["visual_ok"]
        and item["concept_ok"]
        and item["branch_ok"]
        and item["no_anomaly"]
    ]
    if strong_candidates:
        return {
            "decision": "selected_for_5fold",
            "selected_config": strong_candidates[0]["config_id"],
            "ranking": ranking,
            "reason": "performance_close_and_csg_improved",
        }

    perf_candidates = [
        item
        for item in ranking
        if item["perf_ok"] and item["visual_ok"] and item["concept_ok"] and item["branch_ok"] and item["no_anomaly"]
    ]
    if perf_candidates:
        return {
            "decision": "performance_candidate_but_weak_csg",
            "selected_config": perf_candidates[0]["config_id"],
            "ranking": ranking,
            "reason": "performance_ok_but_csg_improvement_weak",
        }

    return {
        "decision": "no_dynamic_csg_selected",
        "selected_config": None,
        "ranking": ranking,
        "reason": "no_config_meets_dynamic_csg_selection_rules",
    }


def build_summary_md(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    selection: dict[str, object],
) -> str:
    completed = results_df.loc[results_df["status"] == "completed", "config_id"].tolist()
    pending = results_df.loc[results_df["status"] == "pending", "config_id"].tolist()
    skipped = results_df.loc[results_df["status"] == "skipped", "config_id"].tolist()

    best_acc = results_df.loc[results_df["ACC"].astype(float).idxmax()] if results_df["ACC"].notna().any() else None
    best_auc = results_df.loc[results_df["AUC"].astype(float).idxmax()] if results_df["AUC"].notna().any() else None
    best_f1 = results_df.loc[results_df["F1"].astype(float).idxmax()] if results_df["F1"].notna().any() else None

    dynamic_completed = dynamic_df.loc[dynamic_df["status"] == "completed"].copy()
    best_csg_ratio = None
    best_csg_delta = None
    if not contribution_df.empty and contribution_df["csg_ratio_mean"].notna().any():
        contrib_completed = contribution_df.loc[contribution_df["status"] == "completed"].copy()
        if not contrib_completed.empty:
            best_csg_ratio = contrib_completed.loc[contrib_completed["csg_ratio_mean"].astype(float).idxmax()]
    if not dynamic_completed.empty and dynamic_completed["csg_logits_delta_abs_mean"].notna().any():
        best_csg_delta = dynamic_completed.loc[
            dynamic_completed["csg_logits_delta_abs_mean"].astype(float).idxmax()
        ]

    dynamic_nonzero = False
    if not dynamic_completed.empty:
        delta_vals = pd.to_numeric(dynamic_completed["csg_logits_delta_abs_mean"], errors="coerce").fillna(0.0)
        alpha_vals = pd.to_numeric(dynamic_completed["learned_alpha_final"], errors="coerce").fillna(0.0)
        dynamic_nonzero = bool((delta_vals.abs() > 1e-8).any() or (alpha_vals.abs() > 1e-8).any())

    visual_low = False
    contrib_completed = contribution_df.loc[contribution_df["status"] == "completed"].copy()
    if not contrib_completed.empty and contrib_completed["visual_ratio_mean"].notna().any():
        visual_low = bool((pd.to_numeric(contrib_completed["visual_ratio_mean"], errors="coerce") < 0.5).all())

    lines = [
        "# Step59B Dynamic CSG fold0 sweep",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。",
        f"3. 本 Step 实际跑了哪些 Dynamic CSG config：{completed or ['无']}。",
        f"4. 哪些 config completed / pending / skipped：completed={completed or ['无']}；pending={pending or ['无']}；skipped={skipped or ['无']}。",
        "5. 哪个 config 的 ACC/AUC/F1 最好："
        f" ACC={best_acc['config_id'] if best_acc is not None else 'NA'}，"
        f" AUC={best_auc['config_id'] if best_auc is not None else 'NA'}，"
        f" F1={best_f1['config_id'] if best_f1 is not None else 'NA'}。",
        "6. 哪个 config 的 csg_ratio 或 csg_logits_delta 提升最明显："
        f" csg_ratio={best_csg_ratio['config_id'] if best_csg_ratio is not None else 'NA'}，"
        f" csg_logits_delta={best_csg_delta['config_id'] if best_csg_delta is not None else 'NA'}。",
        f"7. Dynamic CSG 是否真的产生了非零动态影响：{'是' if dynamic_nonzero else '否'}。",
        f"8. visual_ratio 是否仍保持在 Step58C config D 的低水平：{'是' if visual_low else '否'}。",
        "9. 推荐进入 Step59C 的 selected config 是哪个："
        f" `{selection.get('selected_config')}`。"
        if selection.get("selected_config") is not None
        else "9. 推荐进入 Step59C 的 selected config 是哪个：NA。",
        "10. 如果没有推荐配置，是否建议停止 Dynamic CSG，转向 Step60："
        f" {'是' if selection.get('decision') == 'no_dynamic_csg_selected' else '否'}。",
        "11. 下一步建议是什么："
        + (
            " 进入 Step59C 5-fold。"
            if selection.get("decision") == "selected_for_5fold"
            else " Dynamic CSG 谨慎继续，若要推进请先评估 Step59C 风险。"
            if selection.get("decision") == "performance_candidate_but_weak_csg"
            else " 暂停 Dynamic CSG，转向 Step60 Concept-conditioned Region Aggregation。"
            if selection.get("decision") == "no_dynamic_csg_selected"
            else " 先完成 fold0 sweep 训练。"
        ),
        "",
        "## Selection Result",
        "",
        f"- decision: `{selection.get('decision')}`",
        f"- reason: `{selection.get('reason')}`",
    ]

    for item in selection.get("ranking", [])[:5]:
        lines.append(
            f"- rank {item['config_id']}: score={item['score']:.6f}, "
            f"acc_delta={item['acc_delta']:+.6f}, auc_delta={item['auc_delta']:+.6f}, "
            f"f1_delta={item['f1_delta']:+.6f}, csg_ratio_delta={item['csg_ratio_delta']:+.6f}, "
            f"csg_logits_delta_abs_mean={safe_float(item['csg_logits_delta_abs_mean'])}"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stage59B_run_commands.txt").write_text(
        build_run_commands_text() + "\n",
        encoding="utf-8",
    )

    supported_norms = detect_supported_norms()
    baseline = read_step58c_fold0_baseline()

    results_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    dynamic_rows: list[dict[str, object]] = []
    audit_invocations: dict[str, object] = {}
    warnings: list[str] = []

    for config in CONFIGS:
        run_dir = OUTPUT_DIR / f"{config['exp_code']}_s1"
        status, status_reason = determine_run_status(run_dir, config, supported_norms)
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

            audit_result = run_stage57b_audit(run_dir, config["config_id"])
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

            dynamic_payload = collect_dynamic_metrics(run_dir, config)
            learned_alpha_final = dynamic_payload.get("learned_alpha_final")
            dynamic_rows.append(
                {
                    "config_id": config["config_id"],
                    "status": status,
                    "dynamic_csg_enabled": dynamic_payload.get("dynamic_csg_enabled"),
                    "alpha_init": dynamic_payload.get("alpha_init"),
                    "learned_alpha_final": dynamic_payload.get("learned_alpha_final"),
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
                    "csg_logits_delta_abs_mean_vs_static": dynamic_payload.get("csg_logits_delta_abs_mean_vs_static"),
                    "anomaly_count": dynamic_payload.get("anomaly_count"),
                }
            )
        else:
            dynamic_rows.append(
                {
                    "config_id": config["config_id"],
                    "status": status,
                    "dynamic_csg_enabled": None,
                    "alpha_init": config["alpha_init"],
                    "learned_alpha_final": None,
                    "dynamic_csg_scale": config["dynamic_csg_scale"],
                    "dynamic_csg_norm": config["dynamic_csg_norm"],
                    "dynamic_csg_clip": config["dynamic_csg_clip"],
                    "dynamic_delta_mean": None,
                    "dynamic_delta_std": None,
                    "dynamic_adj_mean": None,
                    "dynamic_adj_std": None,
                    "static_csg_logits_mean": None,
                    "dynamic_csg_logits_mean": None,
                    "csg_logits_delta_mean": None,
                    "csg_logits_delta_abs_mean": None,
                    "csg_logits_delta_abs_mean_vs_static": None,
                    "anomaly_count": None,
                }
            )
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

        results_rows.append(
            {
                "config_id": config["config_id"],
                "fold": 0,
                "status": status,
                "status_reason": status_reason,
                "exp_code": config["exp_code"],
                "alpha_init": config["alpha_init"],
                "learned_alpha_final": learned_alpha_final,
                "dynamic_csg_scale": config["dynamic_csg_scale"],
                "dynamic_csg_norm": config["dynamic_csg_norm"],
                "dynamic_csg_clip": config["dynamic_csg_clip"],
                "ACC": metrics["ACC"],
                "BACC": metrics["BACC"],
                "F1": metrics["F1"],
                "AUC": metrics["AUC"],
                "PR_AUC": metrics["PR_AUC"],
                "delta_acc_vs_step58C_fold0": None
                if metrics["ACC"] is None
                else metrics["ACC"] - baseline["metrics"]["ACC"],
                "delta_auc_vs_step58C_fold0": None
                if metrics["AUC"] is None
                else metrics["AUC"] - baseline["metrics"]["AUC"],
                "delta_f1_vs_step58C_fold0": None
                if metrics["F1"] is None
                else metrics["F1"] - baseline["metrics"]["F1"],
                "delta_pr_auc_vs_step58C_fold0": None
                if metrics["PR_AUC"] is None
                else metrics["PR_AUC"] - baseline["metrics"]["PR_AUC"],
            }
        )

    results_df = pd.DataFrame(results_rows)
    contribution_df = pd.DataFrame(contribution_rows)
    dynamic_df = pd.DataFrame(dynamic_rows)
    branch_df = pd.DataFrame(branch_rows)

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

    selection = select_config(results_df, contribution_df, dynamic_df, branch_df, baseline)

    results_df.to_csv(OUTPUT_DIR / "stage59B_sweep_results.csv", index=False)
    branch_df.to_csv(OUTPUT_DIR / "stage59B_branch_metrics_by_config.csv", index=False)
    contribution_df.to_csv(OUTPUT_DIR / "stage59B_contribution_by_config.csv", index=False)
    dynamic_df.to_csv(OUTPUT_DIR / "stage59B_dynamic_csg_by_config.csv", index=False)
    (OUTPUT_DIR / "stage59B_selected_config.json").write_text(
        json.dumps(
            {
                **selection,
                "supported_norms": supported_norms,
                "baseline_step58c_fold0": {
                    "metrics": {key: round_or_none(value) for key, value in baseline["metrics"].items()},
                    "contribution": {
                        key: round_or_none(value)
                        for key, value in baseline["contribution"].items()
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage59B_sweep_status.json").write_text(
        json.dumps(
            {
                "status": "completed" if (results_df["status"] == "completed").any() else "pending",
                "completed_configs": results_df.loc[results_df["status"] == "completed", "config_id"].tolist(),
                "pending_configs": results_df.loc[results_df["status"] == "pending", "config_id"].tolist(),
                "skipped_configs": results_df.loc[results_df["status"] == "skipped", "config_id"].tolist(),
                "supported_norms": supported_norms,
                "audit_invocations": audit_invocations,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage59B_summary.md").write_text(
        build_summary_md(results_df, contribution_df, dynamic_df, selection),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
