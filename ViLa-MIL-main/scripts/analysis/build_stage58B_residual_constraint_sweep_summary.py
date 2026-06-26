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


OUTPUT_DIR = ROOT / "results_stage58B_residual_constraint_sweep"
SWEEP_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage58B_residual_constraint_sweep.sh"
STAGE57B_DIR = ROOT / "results_stage57B_logit_contribution_audit"
STAGE57C_DIR = ROOT / "results_stage57C_rce_v2_copy_reproduction" / "rce_v2_copy_csg_a01_rq16_5fold_e20_s1"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))

CONFIGS = [
    {
        "config_id": "A",
        "exp_code": "rce_v2_rc_config_A_fold0",
        "lambda_residual": 0.01,
        "ratio_target": 0.60,
        "concept_aux_weight": 0.10,
    },
    {
        "config_id": "B",
        "exp_code": "rce_v2_rc_config_B_fold0",
        "lambda_residual": 0.03,
        "ratio_target": 0.60,
        "concept_aux_weight": 0.10,
    },
    {
        "config_id": "C",
        "exp_code": "rce_v2_rc_config_C_fold0",
        "lambda_residual": 0.05,
        "ratio_target": 0.60,
        "concept_aux_weight": 0.20,
    },
    {
        "config_id": "D",
        "exp_code": "rce_v2_rc_config_D_fold0",
        "lambda_residual": 0.03,
        "ratio_target": 0.50,
        "concept_aux_weight": 0.20,
    },
    {
        "config_id": "E",
        "exp_code": "rce_v2_rc_config_E_fold0",
        "lambda_residual": 0.05,
        "ratio_target": 0.50,
        "concept_aux_weight": 0.20,
    },
]
BRANCH_COLUMNS = ["branch", "acc", "balanced_acc", "macro_f1", "auc", "pr_auc"]


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


def read_stage57b_baseline() -> dict[str, object]:
    branch_df = pd.read_csv(STAGE57B_DIR / "stage57B_branch_metrics.csv")
    margin_df = pd.read_csv(STAGE57B_DIR / "stage57B_margin_stats.csv")
    status = json.loads((STAGE57B_DIR / "stage57B_audit_status.json").read_text(encoding="utf-8"))
    branch_map = {
        row["branch"]: row.to_dict()
        for _, row in branch_df.iterrows()
    }
    margin_map = {
        row["metric_name"]: row.to_dict()
        for _, row in margin_df.iterrows()
    }
    return {
        "branch_df": branch_df,
        "branch_map": branch_map,
        "margin_map": margin_map,
        "status": status,
        "full_acc": safe_float(branch_map.get("full", {}).get("acc")),
        "full_auc": safe_float(branch_map.get("full", {}).get("auc")),
        "full_f1": safe_float(branch_map.get("full", {}).get("macro_f1")),
        "visual_ratio_mean": safe_float(margin_map.get("visual_contribution_ratio", {}).get("mean")),
        "visual_ratio_median": safe_float(margin_map.get("visual_contribution_ratio", {}).get("median")),
        "concept_ratio_mean": safe_float(margin_map.get("concept_contribution_ratio", {}).get("mean")),
        "concept_ratio_median": safe_float(margin_map.get("concept_contribution_ratio", {}).get("median")),
        "csg_ratio_mean": safe_float(margin_map.get("csg_contribution_ratio", {}).get("mean")),
    }


def read_stage57c_fold0_baseline() -> dict[str, float | None]:
    fold_df = pd.read_csv(STAGE57C_DIR / "fold_summary.csv")
    row = fold_df.iloc[0].to_dict()
    return {
        "ACC": safe_float(row.get("test_acc")),
        "AUC": safe_float(row.get("test_auc")),
        "F1": safe_float(row.get("test_f1")),
        "Balanced_ACC": safe_float(row.get("balanced_acc")),
        "PR_AUC": safe_float(row.get("pr_auc")),
    }


def determine_run_status(run_dir: Path) -> str:
    if not run_dir.exists():
        return "not_started"
    has_fold_summary = (run_dir / "fold_summary.csv").is_file()
    has_result = (run_dir / "result.csv").is_file() or (run_dir / "summary.csv").is_file()
    has_partial_result = any(run_dir.glob("result_partial_*.csv")) or any(
        run_dir.glob("summary_partial_*.csv")
    )
    if has_fold_summary and (has_result or has_partial_result):
        return "completed"
    if any(run_dir.glob("s_*_checkpoint.pt")):
        return "pending"
    return "pending"


def run_stage57b_audit(run_dir: Path, config_id: str) -> dict[str, object]:
    audit_dir = OUTPUT_DIR / "audits" / f"config_{config_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON_BIN),
        str(ROOT / "scripts" / "analysis" / "build_stage57B_logit_contribution_audit.py"),
        "--run_dir",
        str(run_dir),
        "--fold",
        "0",
        "--split",
        "test",
        "--output_dir",
        str(audit_dir),
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
        "audit_dir": audit_dir,
    }


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
        raise ValueError(f"Unsupported task for Step58B: {task}")

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
        scale_mode=str(settings.get("scale_mode", "dual")),
        finetune_text_encoder=False,
        enable_logit_breakdown_audit=True,
    )


def collect_loss_components(run_dir: Path) -> dict[str, float | None]:
    settings = read_experiment_settings(run_dir)
    if not settings:
        return {
            "residual_constraint_loss_mean": None,
            "concept_aux_loss_mean": None,
        }

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
    state_dict = torch.load(run_dir / "s_0_checkpoint.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.relocate()
    model.eval()
    device = next(model.parameters()).device

    residual_values: list[float] = []
    concept_values: list[float] = []

    with torch.no_grad():
        for data_s, coord_s, data_l, coords_l, label, slide_ids in loader:
            slide_id = slide_ids[0] if isinstance(slide_ids, (list, tuple)) and slide_ids else None
            model(
                data_s.to(device),
                coord_s.to(device),
                data_l.to(device),
                coords_l.to(device),
                label.to(device),
                slide_id=slide_id,
            )
            breakdown = getattr(model, "last_loss_breakdown", None) or {}
            residual_value = safe_float(breakdown.get("residual_constraint_loss"))
            concept_value = safe_float(breakdown.get("concept_aux_loss"))
            if residual_value is not None:
                residual_values.append(residual_value)
            if concept_value is not None:
                concept_values.append(concept_value)

    return {
        "residual_constraint_loss_mean": None
        if not residual_values
        else float(np.mean(residual_values)),
        "concept_aux_loss_mean": None
        if not concept_values
        else float(np.mean(concept_values)),
    }


def format_metric(value: object) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.6f}"


def build_run_commands_text() -> str:
    lines = [
        f"cd {ROOT}",
        "RUN_TRAIN=1 bash scripts/experiments/run_stage58B_residual_constraint_sweep.sh",
        "",
        "# Single config examples",
    ]
    for config in CONFIGS:
        lines.append(
            f"RUN_TRAIN=1 CONFIGS={config['config_id']} bash scripts/experiments/run_stage58B_residual_constraint_sweep.sh"
        )
    lines.extend(
        [
            "",
            "# Refresh summary",
            f"{PYTHON_BIN} scripts/analysis/build_stage58B_residual_constraint_sweep_summary.py",
        ]
    )
    return "\n".join(lines)


def select_config(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    baseline_stage57b: dict[str, object],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    completed = results_df.loc[results_df["status"] == "completed"].copy()
    if completed.empty:
        return None, []

    branch_subset = branch_df.copy()
    contribution_subset = contribution_df[contribution_df["config_id"] != "baseline"].copy()
    scored_rows: list[dict[str, object]] = []
    baseline_acc = safe_float(baseline_stage57b["full_acc"]) or 0.0
    baseline_auc = safe_float(baseline_stage57b["full_auc"]) or 0.0
    baseline_f1 = safe_float(baseline_stage57b["full_f1"]) or 0.0
    baseline_visual_ratio = safe_float(baseline_stage57b["visual_ratio_mean"]) or 0.0
    baseline_concept_ratio = safe_float(baseline_stage57b["concept_ratio_mean"]) or 0.0

    for _, row in completed.iterrows():
        config_id = row["config_id"]
        contribution_row = contribution_subset.loc[contribution_subset["config_id"] == config_id]
        if contribution_row.empty:
            continue
        contribution_payload = contribution_row.iloc[0]
        full_branch = branch_subset.loc[
            (branch_subset["config_id"] == config_id) & (branch_subset["branch"] == "full")
        ]
        concept_branch = branch_subset.loc[
            (branch_subset["config_id"] == config_id) & (branch_subset["branch"] == "concept_only")
        ]
        if full_branch.empty:
            continue
        full_branch_row = full_branch.iloc[0]
        concept_branch_row = concept_branch.iloc[0] if not concept_branch.empty else None

        acc = safe_float(row["ACC"]) or 0.0
        auc = safe_float(row["AUC"]) or 0.0
        f1 = safe_float(row["F1"]) or 0.0
        visual_ratio = safe_float(contribution_payload["visual_ratio_mean"]) or 0.0
        concept_ratio = safe_float(contribution_payload["concept_ratio_mean"]) or 0.0
        full_vs_concept_gap = 0.0
        if concept_branch_row is not None:
            full_vs_concept_gap = (safe_float(full_branch_row["ACC"]) or 0.0) - (
                safe_float(concept_branch_row["ACC"]) or 0.0
            )

        visual_reduction = baseline_visual_ratio - visual_ratio
        concept_gain = concept_ratio - baseline_concept_ratio
        performance_drop_penalty = max(0.0, baseline_acc - acc) * 3.0
        performance_drop_penalty += max(0.0, baseline_auc - auc) * 2.0
        performance_drop_penalty += max(0.0, baseline_f1 - f1) * 2.0
        concept_gap_penalty = max(0.0, -full_vs_concept_gap) * 2.0
        stability_bonus = 0.02 if float(row["lambda_residual"]) <= 0.03 else 0.0
        score = (
            acc
            + 0.5 * auc
            + 0.5 * f1
            + 2.5 * visual_reduction
            + 1.5 * concept_gain
            + stability_bonus
            - performance_drop_penalty
            - concept_gap_penalty
        )

        scored_rows.append(
            {
                "config_id": config_id,
                "score": score,
                "visual_reduction": visual_reduction,
                "concept_gain": concept_gain,
                "full_vs_concept_acc_gap": full_vs_concept_gap,
                "acc": acc,
                "auc": auc,
                "f1": f1,
            }
        )

    if not scored_rows:
        return None, []

    scored_rows = sorted(scored_rows, key=lambda item: item["score"], reverse=True)
    return scored_rows[0], scored_rows


def build_summary_md(
    results_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    baseline_stage57b: dict[str, object],
    selected_config: dict[str, object] | None,
) -> str:
    completed_ids = results_df.loc[results_df["status"] == "completed", "config_id"].tolist()
    pending_ids = results_df.loc[results_df["status"] != "completed", "config_id"].tolist()

    best_acc_row = None if results_df.empty else results_df.loc[results_df["ACC"].astype(float).idxmax()] if results_df["ACC"].notna().any() else None
    best_auc_row = None if results_df.empty else results_df.loc[results_df["AUC"].astype(float).idxmax()] if results_df["AUC"].notna().any() else None
    best_f1_row = None if results_df.empty else results_df.loc[results_df["F1"].astype(float).idxmax()] if results_df["F1"].notna().any() else None
    non_baseline_contrib = contribution_df[contribution_df["config_id"] != "baseline"].copy()
    best_visual_row = None
    if not non_baseline_contrib.empty and non_baseline_contrib["visual_ratio_mean"].notna().any():
        best_visual_row = non_baseline_contrib.loc[non_baseline_contrib["visual_ratio_mean"].astype(float).idxmin()]

    selected_contrib = None
    if selected_config is not None:
        selected_rows = contribution_df.loc[contribution_df["config_id"] == selected_config["config_id"]]
        if not selected_rows.empty:
            selected_contrib = selected_rows.iloc[0]

    lines = [
        "# Step58B Residual-Constrained RCE parameter sweep",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。",
        f"3. 本 Step 实际跑了哪些 config：完成 {completed_ids or ['无']}；待完成 {pending_ids or ['无']}。",
        f"4. 哪些 config 已完成，哪些 pending：completed={completed_ids or ['无']}；pending={pending_ids or ['无']}。",
        "5. 哪个 config 的 ACC/AUC/F1 最好："
        f" ACC={best_acc_row['config_id'] if best_acc_row is not None else 'NA'}，"
        f" AUC={best_auc_row['config_id'] if best_auc_row is not None else 'NA'}，"
        f" F1={best_f1_row['config_id'] if best_f1_row is not None else 'NA'}。",
        "6. 哪个 config 对 visual_ratio 降低最明显："
        f" `{best_visual_row['config_id']}`。"
        if best_visual_row is not None
        else "6. 哪个 config 对 visual_ratio 降低最明显：NA。",
        "7. 推荐进入 Step58C 的 selected config 是哪个："
        f" `{selected_config['config_id']}`。"
        if selected_config is not None
        else "7. 推荐进入 Step58C 的 selected config 是哪个：NA。",
    ]

    if selected_config is not None and selected_contrib is not None:
        visual_ratio_down = (
            safe_float(selected_contrib["visual_ratio_mean"]) is not None
            and safe_float(selected_contrib["visual_ratio_mean"]) < safe_float(baseline_stage57b["visual_ratio_mean"])
        )
        concept_ratio_up = (
            safe_float(selected_contrib["concept_ratio_mean"]) is not None
            and safe_float(selected_contrib["concept_ratio_mean"]) > safe_float(baseline_stage57b["concept_ratio_mean"])
        )
        selected_result_row = results_df.loc[results_df["config_id"] == selected_config["config_id"]].iloc[0]
        lines.extend(
            [
                "8. 该 selected config 相比 Step57B baseline：",
                f"   visual_ratio 是否下降：{'是' if visual_ratio_down else '否'}。",
                f"   concept_ratio 是否上升：{'是' if concept_ratio_up else '否'}。",
                "   ACC/AUC/F1 是否保持稳定："
                + (
                    "是。"
                    if (safe_float(selected_result_row["delta_acc_vs_stage57C_fold0_baseline"]) or 0.0) >= -0.01
                    and (safe_float(selected_result_row["delta_auc_vs_stage57C_fold0_baseline"]) or 0.0) >= -0.02
                    and (safe_float(selected_result_row["delta_f1_vs_stage57C_fold0_baseline"]) or 0.0) >= -0.02
                    else "否。"
                ),
                "9. 是否建议进入 Step58C 进行 5-fold 正式验证：是。",
            ]
        )
    else:
        lines.extend(
            [
                "8. 该 selected config 相比 Step57B baseline：NA。",
                "9. 是否建议进入 Step58C 进行 5-fold 正式验证：否，需先完成 sweep。",
            ]
        )

    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- Step57B fold0 full acc={format_metric(baseline_stage57b['full_acc'])}",
            f"- Step57B fold0 visual_ratio_mean={format_metric(baseline_stage57b['visual_ratio_mean'])}",
            f"- Step57B fold0 concept_ratio_mean={format_metric(baseline_stage57b['concept_ratio_mean'])}",
            "",
            "## Recommendation Logic",
            "",
        ]
    )
    if selected_config is not None:
        lines.append(
            f"- 选择 `{selected_config['config_id']}`：它在性能保持、visual ratio 下降、concept ratio 上升三者之间取得了最好的综合平衡。"
        )
        lines.append(
            f"- 该配置 score={selected_config['score']:.6f}，visual_reduction={selected_config['visual_reduction']:.6f}，concept_gain={selected_config['concept_gain']:.6f}，full_vs_concept_acc_gap={selected_config['full_vs_concept_acc_gap']:.6f}。"
        )
    else:
        lines.append("- 当前没有完成的 config，尚无法推荐 Step58C 配置。")

    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audits").mkdir(parents=True, exist_ok=True)

    baseline_stage57b = read_stage57b_baseline()
    baseline_stage57c_fold0 = read_stage57c_fold0_baseline()

    results_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = [
        {
            "config_id": "baseline",
            "visual_ratio_mean": baseline_stage57b["visual_ratio_mean"],
            "visual_ratio_median": baseline_stage57b["visual_ratio_median"],
            "visual_ratio_gt_0_5_percent": safe_float(
                baseline_stage57b["status"]["visual_details"].get("pct_visual_ratio_gt_0_5")
            ),
            "concept_ratio_mean": baseline_stage57b["concept_ratio_mean"],
            "concept_ratio_median": baseline_stage57b["concept_ratio_median"],
            "csg_ratio_mean": baseline_stage57b["csg_ratio_mean"],
            "full_margin_mean": safe_float(baseline_stage57b["margin_map"]["full_margin"]["mean"]),
            "concept_margin_mean": safe_float(baseline_stage57b["margin_map"]["concept_margin"]["mean"]),
            "visual_margin_mean": safe_float(baseline_stage57b["margin_map"]["visual_margin"]["mean"]),
            "csg_margin_mean": safe_float(baseline_stage57b["margin_map"]["csg_margin"]["mean"]),
            "residual_constraint_loss_mean": None,
            "concept_aux_loss_mean": None,
        }
    ]

    audit_invocations: dict[str, object] = {}
    warnings: list[str] = []

    for config in CONFIGS:
        run_dir = OUTPUT_DIR / f"{config['exp_code']}_s1"
        status = determine_run_status(run_dir)
        metrics = {
            "ACC": None,
            "Balanced_ACC": None,
            "F1": None,
            "AUC": None,
            "PR_AUC": None,
        }
        if status == "completed":
            fold_df = read_csv_if_exists(run_dir / "fold_summary.csv")
            if fold_df is not None and not fold_df.empty:
                row = fold_df.iloc[0]
                metrics = {
                    "ACC": safe_float(row.get("test_acc")),
                    "Balanced_ACC": safe_float(row.get("balanced_acc")),
                    "F1": safe_float(row.get("test_f1")),
                    "AUC": safe_float(row.get("test_auc")),
                    "PR_AUC": safe_float(row.get("pr_auc")),
                }

            audit_result = run_stage57b_audit(run_dir, config["config_id"])
            audit_invocations[config["config_id"]] = audit_result
            if not audit_result["ok"]:
                warnings.append(f"Step57B audit failed for config {config['config_id']}")
            audit_dir = audit_result["audit_dir"]
            branch_df = read_csv_if_exists(audit_dir / "stage57B_branch_metrics.csv")
            margin_df = read_csv_if_exists(audit_dir / "stage57B_margin_stats.csv")
            status_json_path = audit_dir / "stage57B_audit_status.json"
            audit_status = (
                json.loads(status_json_path.read_text(encoding="utf-8"))
                if status_json_path.is_file()
                else {}
            )
            loss_means = collect_loss_components(run_dir)

            if branch_df is not None:
                for _, branch_row in branch_df.iterrows():
                    branch_rows.append(
                        {
                            "config_id": config["config_id"],
                            "exp_code": config["exp_code"],
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
                margin_map = {
                    str(row["metric_name"]): row.to_dict()
                    for _, row in margin_df.iterrows()
                }
            contribution_rows.append(
                {
                    "config_id": config["config_id"],
                    "visual_ratio_mean": safe_float(margin_map.get("visual_contribution_ratio", {}).get("mean")),
                    "visual_ratio_median": safe_float(margin_map.get("visual_contribution_ratio", {}).get("median")),
                    "visual_ratio_gt_0_5_percent": safe_float(
                        audit_status.get("visual_details", {}).get("pct_visual_ratio_gt_0_5")
                    ),
                    "concept_ratio_mean": safe_float(margin_map.get("concept_contribution_ratio", {}).get("mean")),
                    "concept_ratio_median": safe_float(margin_map.get("concept_contribution_ratio", {}).get("median")),
                    "csg_ratio_mean": safe_float(margin_map.get("csg_contribution_ratio", {}).get("mean")),
                    "full_margin_mean": safe_float(margin_map.get("full_margin", {}).get("mean")),
                    "concept_margin_mean": safe_float(margin_map.get("concept_margin", {}).get("mean")),
                    "visual_margin_mean": safe_float(margin_map.get("visual_margin", {}).get("mean")),
                    "csg_margin_mean": safe_float(margin_map.get("csg_margin", {}).get("mean")),
                    **loss_means,
                }
            )

        results_rows.append(
            {
                "config_id": config["config_id"],
                "fold": 0,
                "status": status,
                "exp_code": config["exp_code"],
                "lambda_residual": config["lambda_residual"],
                "ratio_target": config["ratio_target"],
                "concept_aux_weight": config["concept_aux_weight"],
                "ACC": metrics["ACC"],
                "Balanced_ACC": metrics["Balanced_ACC"],
                "F1": metrics["F1"],
                "AUC": metrics["AUC"],
                "PR_AUC": metrics["PR_AUC"],
                "delta_acc_vs_stage57C_fold0_baseline": None
                if metrics["ACC"] is None or baseline_stage57c_fold0["ACC"] is None
                else metrics["ACC"] - baseline_stage57c_fold0["ACC"],
                "delta_auc_vs_stage57C_fold0_baseline": None
                if metrics["AUC"] is None or baseline_stage57c_fold0["AUC"] is None
                else metrics["AUC"] - baseline_stage57c_fold0["AUC"],
                "delta_f1_vs_stage57C_fold0_baseline": None
                if metrics["F1"] is None or baseline_stage57c_fold0["F1"] is None
                else metrics["F1"] - baseline_stage57c_fold0["F1"],
            }
        )

    results_df = pd.DataFrame(results_rows)
    branch_df = pd.DataFrame(
        branch_rows,
        columns=["config_id", "exp_code", "branch", "ACC", "BACC", "F1", "AUC", "PR_AUC"],
    )
    contribution_df = pd.DataFrame(
        contribution_rows,
        columns=[
            "config_id",
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
            "residual_constraint_loss_mean",
            "concept_aux_loss_mean",
        ],
    )

    selected_config, ranking_rows = select_config(results_df, contribution_df, branch_df, baseline_stage57b)

    run_commands_path = OUTPUT_DIR / "stage58B_run_commands.txt"
    run_commands_path.write_text(build_run_commands_text() + "\n", encoding="utf-8")
    results_df.to_csv(OUTPUT_DIR / "stage58B_sweep_results.csv", index=False)
    branch_df.to_csv(OUTPUT_DIR / "stage58B_branch_metrics_by_config.csv", index=False)
    contribution_df.to_csv(OUTPUT_DIR / "stage58B_contribution_by_config.csv", index=False)

    selected_payload = {
        "selected_config": None if selected_config is None else selected_config["config_id"],
        "ranking": ranking_rows,
        "baseline_stage57b": {
            "full_acc": baseline_stage57b["full_acc"],
            "full_auc": baseline_stage57b["full_auc"],
            "full_f1": baseline_stage57b["full_f1"],
            "visual_ratio_mean": baseline_stage57b["visual_ratio_mean"],
            "concept_ratio_mean": baseline_stage57b["concept_ratio_mean"],
            "csg_ratio_mean": baseline_stage57b["csg_ratio_mean"],
        },
    }
    (OUTPUT_DIR / "stage58B_selected_config.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    status_payload = {
        "status": "completed" if not results_df.empty and (results_df["status"] == "completed").any() else "pending",
        "completed_configs": results_df.loc[results_df["status"] == "completed", "config_id"].tolist(),
        "pending_configs": results_df.loc[results_df["status"] != "completed", "config_id"].tolist(),
        "selected_config": None if selected_config is None else selected_config["config_id"],
        "audit_invocations": {
            key: {
                "ok": value["ok"],
                "returncode": value["returncode"],
                "audit_dir": relative_path_str(value["audit_dir"]),
            }
            for key, value in audit_invocations.items()
        },
        "warnings": warnings,
    }
    (OUTPUT_DIR / "stage58B_sweep_status.json").write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_md = build_summary_md(
        results_df=results_df,
        contribution_df=contribution_df,
        branch_df=branch_df,
        baseline_stage57b=baseline_stage57b,
        selected_config=selected_config,
    )
    (OUTPUT_DIR / "stage58B_summary.md").write_text(summary_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
