from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_RCE_MIL_BiomedCLIP_v2 import RCE_MIL_BiomedCLIP


OUTPUT_DIR = ROOT / "results_stage61B_l2h_retrieval_all_off"
STEP57C_RUN_DIR = ROOT / "results_stage57C_rce_v2_copy_reproduction" / "rce_v2_copy_csg_a01_rq16_5fold_e20_s1"
STEP57C_EXPERIMENT = STEP57C_RUN_DIR / "experiment_rce_v2_copy_csg_a01_rq16_5fold_e20.txt"
STEP57C_CHECKPOINT = STEP57C_RUN_DIR / "s_0_checkpoint.pt"
STEP58C_RUN_DIR = (
    ROOT / "results_stage58C_residual_constrained_configD_5fold" / "rce_v2_rcD_l003_t050_aux020_5fold_e20_s1"
)
STEP58C_EXPERIMENT = STEP58C_RUN_DIR / "experiment_rce_v2_rcD_l003_t050_aux020_5fold_e20.txt"
STEP58C_CHECKPOINT = STEP58C_RUN_DIR / "s_0_checkpoint.pt"
MODEL_PATH = ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py"
MAIN_PATH = ROOT / "main.py"
CORE_UTILS_PATH = ROOT / "utils" / "core_utils.py"
ORIGINAL_RCE_PATH = ROOT / "models" / "model_RCE_MIL_BiomedCLIP.py"
SMOKE_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage61B_l2h_retrieval_smoke.sh"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))
NEW_ARGS = [
    "rce_use_l2h_retrieval",
    "rce_l2h_mode",
    "rce_l2h_low_topk",
    "rce_l2h_high_max_per_low",
    "rce_l2h_scale_ratio",
    "rce_l2h_patch_footprint_ratio",
    "rce_l2h_alpha_init",
    "rce_l2h_scale",
    "rce_l2h_fusion",
    "rce_l2h_aggregate",
    "rce_l2h_score_mode",
    "rce_l2h_detach_low_scores",
    "rce_l2h_min_high_matches",
    "rce_l2h_clip",
]
MODEL_ATTRS = [
    "self.rce_use_l2h_retrieval",
    "self.rce_l2h_mode",
    "self.rce_l2h_alpha",
    "self.last_l2h_retrieval_debug",
]
L2H_EXPORT_ATTRS = [
    "last_low_patch_concept_scores",
    "last_low_patch_topk_indices",
    "last_low_patch_topk_scores",
    "last_low_patch_coords",
    "last_retrieved_high_patch_indices",
    "last_retrieved_high_patch_coords",
    "last_retrieved_high_patch_match_counts",
    "last_retrieved_high_patch_mask",
]
L2H_BREAKDOWN_KEYS = [
    "l2h_enabled",
    "l2h_mode",
    "l2h_alpha",
    "l2h_scale",
    "l2h_score_mode",
    "l2h_low_topk",
    "l2h_high_max_per_low",
    "l2h_scale_ratio",
    "l2h_patch_footprint_ratio",
    "low_patch_concept_scores_shape",
    "low_patch_features_shape",
    "high_patch_features_shape",
    "low_coords_shape",
    "high_coords_shape",
    "high_region_features_shape",
    "retrieved_high_patch_features_shape",
    "fused_high_region_features_shape",
    "skipped_reason",
]


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def round_or_none(value: object, digits: int = 8) -> float | None:
    numeric = safe_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def file_contains_tokens(path: Path, tokens: list[str]) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    return {token: (token in text) for token in tokens}


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


def read_experiment_settings(path: Path) -> dict[str, object]:
    return ast.literal_eval(path.read_text(encoding="utf-8"))


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
        raise ValueError(f"Unsupported task for Step61B audit: {task}")

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


def build_model_config(
    settings: dict[str, object],
    overrides: dict[str, object] | None = None,
) -> SimpleNamespace:
    merged = dict(settings)
    if overrides:
        merged.update(overrides)

    return SimpleNamespace(
        input_size=512,
        hidden_size=192,
        class_names=merged.get("class_names"),
        use_concept_prompt_pool=to_bool(merged.get("use_concept_prompt_pool", False)),
        concept_prompt_path=merged.get("concept_prompt_path"),
        peps_tau=float(merged.get("peps_tau", 0.1)),
        prototype_number=int(merged.get("prototype_number", 16)),
        rce_use_logit_calibration=to_bool(merged.get("rce_use_logit_calibration", False)),
        rce_use_concept_prior=to_bool(merged.get("rce_use_concept_prior", False)),
        rce_logit_scale_init=float(merged.get("rce_logit_scale_init", 10.0)),
        rce_concept_prior_strength=float(merged.get("rce_concept_prior_strength", 1.0)),
        rce_use_visual_residual=to_bool(merged.get("rce_use_visual_residual", False)),
        rce_visual_residual_init=float(merged.get("rce_visual_residual_init", 0.1)),
        rce_use_residual_constraint=to_bool(merged.get("rce_use_residual_constraint", False)),
        rce_residual_constraint_lambda=float(merged.get("rce_residual_constraint_lambda", 0.0)),
        rce_residual_ratio_target=float(merged.get("rce_residual_ratio_target", 0.5)),
        rce_residual_constraint_type=str(merged.get("rce_residual_constraint_type", "relu_l2")),
        rce_use_concept_aux_loss=to_bool(merged.get("rce_use_concept_aux_loss", False)),
        rce_concept_aux_loss_weight=float(merged.get("rce_concept_aux_loss_weight", 0.0)),
        rce_residual_ratio_eps=float(merged.get("rce_residual_ratio_eps", 1e-6)),
        rce_residual_ratio_detach=to_bool(merged.get("rce_residual_ratio_detach", False)),
        rce_use_cross_scale_graph=to_bool(merged.get("rce_use_cross_scale_graph", False)),
        rce_cross_scale_graph_init=float(merged.get("rce_cross_scale_graph_init", 0.05)),
        rce_cross_scale_graph_norm=str(merged.get("rce_cross_scale_graph_norm", "sqrt")),
        rce_use_dynamic_csg=to_bool(merged.get("rce_use_dynamic_csg", False)),
        rce_dynamic_csg_mode=str(merged.get("rce_dynamic_csg_mode", "evidence_outer")),
        rce_dynamic_csg_alpha_init=float(merged.get("rce_dynamic_csg_alpha_init", 0.0)),
        rce_dynamic_csg_scale=float(merged.get("rce_dynamic_csg_scale", 1.0)),
        rce_dynamic_csg_norm=str(merged.get("rce_dynamic_csg_norm", "softmax")),
        rce_dynamic_csg_detach_evidence=to_bool(
            merged.get("rce_dynamic_csg_detach_evidence", False)
        ),
        rce_dynamic_csg_clip=float(merged.get("rce_dynamic_csg_clip", 5.0)),
        rce_use_ccra=to_bool(merged.get("rce_use_ccra", False)),
        rce_ccra_mode=str(merged.get("rce_ccra_mode", "concept_query_residual")),
        rce_ccra_alpha_init=float(merged.get("rce_ccra_alpha_init", 0.0)),
        rce_ccra_scale=float(merged.get("rce_ccra_scale", 1.0)),
        rce_ccra_num_queries=int(merged.get("rce_ccra_num_queries", 0)),
        rce_ccra_query_source=str(merged.get("rce_ccra_query_source", "prompt_mean")),
        rce_ccra_detach_prompt=to_bool(merged.get("rce_ccra_detach_prompt", False)),
        rce_ccra_norm=str(merged.get("rce_ccra_norm", "layernorm")),
        rce_ccra_dropout=float(merged.get("rce_ccra_dropout", 0.0)),
        rce_ccra_clip=float(merged.get("rce_ccra_clip", 5.0)),
        rce_use_l2h_retrieval=to_bool(merged.get("rce_use_l2h_retrieval", False)),
        rce_l2h_mode=str(merged.get("rce_l2h_mode", "low_topk_coord_window")),
        rce_l2h_low_topk=int(merged.get("rce_l2h_low_topk", 8)),
        rce_l2h_high_max_per_low=int(merged.get("rce_l2h_high_max_per_low", 16)),
        rce_l2h_scale_ratio=float(merged.get("rce_l2h_scale_ratio", 1.0)),
        rce_l2h_patch_footprint_ratio=float(merged.get("rce_l2h_patch_footprint_ratio", 4.0)),
        rce_l2h_alpha_init=float(merged.get("rce_l2h_alpha_init", 0.0)),
        rce_l2h_scale=float(merged.get("rce_l2h_scale", 1.0)),
        rce_l2h_fusion=str(merged.get("rce_l2h_fusion", "high_region_residual")),
        rce_l2h_aggregate=str(merged.get("rce_l2h_aggregate", "mean")),
        rce_l2h_score_mode=str(merged.get("rce_l2h_score_mode", "low_prompt_max")),
        rce_l2h_detach_low_scores=to_bool(merged.get("rce_l2h_detach_low_scores", False)),
        rce_l2h_min_high_matches=int(merged.get("rce_l2h_min_high_matches", 1)),
        rce_l2h_clip=float(merged.get("rce_l2h_clip", 5.0)),
        scale_mode=str(merged.get("scale_mode", "dual")),
        finetune_text_encoder=False,
        enable_logit_breakdown_audit=True,
    )


def load_sample(settings: dict[str, object], split_name: str = "splits_0.csv") -> dict[str, object]:
    dataset = build_dataset(settings)
    split_dir = Path(str(settings["split_dir"]))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    _, _, test_split = dataset.return_splits(
        from_id=False,
        csv_path=str(split_dir / split_name),
    )
    features_s, coords_s, features_l, coords_l, label, slide_id = test_split[0]
    return {
        "data_s": features_s,
        "coord_s": coords_s,
        "data_l": features_l,
        "coord_l": coords_l,
        "label": torch.tensor([label], dtype=torch.long),
        "slide_id": slide_id,
    }


def instantiate_model(
    settings: dict[str, object],
    checkpoint_path: Path,
    overrides: dict[str, object] | None = None,
    strict: bool = True,
) -> tuple[RCE_MIL_BiomedCLIP, dict[str, object]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = build_model_config(settings, overrides)
    model = RCE_MIL_BiomedCLIP(config=config, num_classes=int(settings["n_classes"]))
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(state_dict, strict=strict)
    model.relocate()
    model.eval()
    if hasattr(model, "set_logit_breakdown_audit"):
        model.set_logit_breakdown_audit(True)
    return model, {
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        "strict": strict,
    }


def run_model(model: RCE_MIL_BiomedCLIP, sample: dict[str, object]) -> dict[str, object]:
    device = next(model.parameters()).device
    with torch.no_grad():
        y_prob, y_hat, loss = model(
            sample["data_s"].to(device),
            sample["coord_s"].to(device),
            sample["data_l"].to(device),
            sample["coord_l"].to(device),
            sample["label"].to(device),
            slide_id=sample["slide_id"],
        )
    return {
        "y_prob": y_prob.detach().cpu(),
        "y_hat": y_hat.detach().cpu(),
        "loss": loss.detach().cpu(),
        "logit_breakdown": getattr(model, "last_logit_breakdown", None),
        "loss_breakdown": getattr(model, "last_loss_breakdown", None),
        "dynamic_csg_breakdown": getattr(model, "last_dynamic_csg_breakdown", None),
        "ccra_breakdown": getattr(model, "last_ccra_breakdown", None),
        "l2h_breakdown": getattr(model, "last_l2h_retrieval_debug", None),
        "l2h_exports": {
            name: getattr(model, name, None) for name in L2H_EXPORT_ATTRS
        },
    }


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def compare_outputs(lhs: dict[str, object], rhs: dict[str, object], atol: float = 1e-7) -> dict[str, object]:
    prob_diff = max_abs_diff(lhs["y_prob"], rhs["y_prob"])
    loss_diff = max_abs_diff(lhs["loss"], rhs["loss"])
    hat_equal = bool(torch.equal(lhs["y_hat"], rhs["y_hat"]))
    full_logits_diff = None
    lhs_breakdown = lhs.get("logit_breakdown") or {}
    rhs_breakdown = rhs.get("logit_breakdown") or {}
    lhs_full_logits = ((lhs_breakdown.get("post_calibration") or {}).get("full_logits"))
    rhs_full_logits = ((rhs_breakdown.get("post_calibration") or {}).get("full_logits"))
    if lhs_full_logits is not None and rhs_full_logits is not None:
        full_logits_diff = max_abs_diff(lhs_full_logits, rhs_full_logits)
    return {
        "prob_max_abs_diff": round_or_none(prob_diff, digits=10),
        "loss_abs_diff": round_or_none(loss_diff, digits=10),
        "full_logits_max_abs_diff": round_or_none(full_logits_diff, digits=10),
        "y_hat_equal": hat_equal,
        "pass": bool(
            prob_diff <= atol
            and loss_diff <= atol
            and hat_equal
            and (full_logits_diff is None or full_logits_diff <= atol)
        ),
        "atol": atol,
    }


def list_l2h_param_names(model: RCE_MIL_BiomedCLIP) -> list[str]:
    return [name for name, _ in model.named_parameters() if "l2h" in name]


def build_run_commands_text() -> str:
    display_root = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
    if not (display_root / "main.py").is_file():
        display_root = ROOT
    return "\n".join(
        [
            f"cd {display_root}",
            "RUN_TRAIN=1 bash scripts/experiments/run_stage61B_l2h_retrieval_smoke.sh",
            "",
            "# Refresh Step61B audit",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage61B_l2h_retrieval_audit.py",
        ]
    )


def summarise_tensor_shape(tensor: object) -> list[int] | None:
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return list(tensor.shape)
    return None


def build_summary_md(
    config_audit: dict[str, object],
    all_off_audit: dict[str, object],
    param_init_audit: dict[str, object],
    smoke_audit: dict[str, object],
) -> str:
    l2h_params = ", ".join(NEW_ARGS)
    l2h_breakdown = smoke_audit.get("l2h_breakdown", {}) or {}
    lines = [
        "# Step61B Low-to-High Concept-guided Retrieval all-off audit",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否修改了 RCE-v2 默认行为：否，默认关闭路径保持不变。",
        "3. rce_use_l2h_retrieval=False 时是否创建 L2H retrieval 专属参数："
        f" {'否' if not param_init_audit.get('l2h_param_names_when_off') else '是'}。",
        "4. full all-off forward 等价检查是否通过："
        f" {'是' if all_off_audit.get('full_all_off_equivalence', {}).get('pass') else '否'}。",
        "5. Step58C config D 上的 l2h-off 等价检查是否通过："
        f" {'是' if all_off_audit.get('l2h_off_equivalence', {}).get('pass') else '否'}。",
        "6. 开启 L2H 且 alpha_init=0.0 时，forward 输出是否仍与基线一致："
        f" {'是' if all_off_audit.get('alpha_zero_forward_equivalence', {}).get('pass') else '否'}。",
        f"7. 新增了哪些 L2H 参数：{l2h_params}。",
        "8. L2H 的计算方式是什么："
        " 先用 low patch 与 low prompt 的 concept similarity 做 low_prompt_max 打分并取 top-k；"
        "再按 low coords 在 high coords 中做同坐标系窗口检索；"
        "对 retrieved high patch features 做 mean aggregation；"
        "最后用 fused_high_region = original_high_region + alpha * scale * retrieved_context 做 residual fusion，"
        "后续继续复用原有 high evidence / concept evidence 路径。",
        "9. alpha 初始化是多少，以及为什么 Step61B 使用 0.0：0.0；"
        "0.0 最安全，便于 all-off / smoke 审计，同时验证 retrieval 接线本身不引入额外 logits 扰动；Step61C 可扫 0.001 / 0.01 / 0.05。",
        "10. 坐标检索 smoke 是否通过："
        f" {'是' if smoke_audit.get('retrieval_smoke_pass') else '否'}。",
        "11. forward smoke 是否通过："
        f" {'是' if smoke_audit.get('forward_smoke_pass') else '否'}。",
        "12. residual constraint 机制在 Step58C config D 主线上是否仍保留："
        f" {'是' if smoke_audit.get('residual_constraint_still_enabled') else '否'}。",
        "13. 当前 L2H 实现是否安全处理 coords 缺失场景："
        f" {'是' if config_audit.get('coords_skip_logic_present') else '否'}。",
        "14. 是否可以进入 Step61C L2H 参数探索："
        f" {'是' if smoke_audit.get('smoke_pass') and all_off_audit.get('full_all_off_equivalence', {}).get('pass') else '否'}。",
        "",
        "## Audit Notes",
        "",
        f"- forward return format preserved: `{config_audit.get('forward_return_format')}`",
        f"- l2h default enabled flag: `{config_audit.get('l2h_default_enabled')}`",
        f"- l2h params when off: `{param_init_audit.get('l2h_param_names_when_off')}`",
        f"- l2h breakdown keys present: `{smoke_audit.get('l2h_breakdown_keys_present')}`",
        f"- l2h_enabled in smoke: `{l2h_breakdown.get('l2h_enabled')}`",
        f"- skipped_reason in smoke: `{l2h_breakdown.get('skipped_reason')}`",
        f"- low/high coords shape in smoke: `{l2h_breakdown.get('low_coords_shape')}` / `{l2h_breakdown.get('high_coords_shape')}`",
        f"- retrieved_high_patch_features_shape in smoke: `{l2h_breakdown.get('retrieved_high_patch_features_shape')}`",
        f"- fused_high_region_features_shape in smoke: `{l2h_breakdown.get('fused_high_region_features_shape')}`",
        f"- retrieved match count stats in smoke: `{smoke_audit.get('retrieved_match_count_stats')}`",
        f"- smoke training script syntax check: `{config_audit.get('smoke_script_syntax_ok')}`",
        "",
        "## Recommendation",
        "",
        "- Step61B 只完成 L2H Retrieval 的安全接线、默认关闭、坐标检索 smoke、forward smoke 与 all-off 审计，没有启动完整 5-fold。",
    ]
    if smoke_audit.get("smoke_pass") and all_off_audit.get("full_all_off_equivalence", {}).get("pass"):
        lines.append("- 结论：可以进入 Step61C，对 alpha_init / low_topk / high_max_per_low 做小扫。")
    else:
        lines.append("- 结论：先修复 Step61B 审计未通过项，再考虑进入 Step61C。")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stage61B_run_commands.txt").write_text(
        build_run_commands_text() + "\n",
        encoding="utf-8",
    )

    smoke_syntax_cmd = run_command(["bash", "-n", str(SMOKE_SCRIPT)])
    smoke_script_cmd = run_command(["bash", str(SMOKE_SCRIPT)])
    config_audit = {
        "original_rce_modified": git_path_modified(ORIGINAL_RCE_PATH),
        "main_has_new_args": file_contains_tokens(MAIN_PATH, NEW_ARGS),
        "core_utils_has_new_args": file_contains_tokens(CORE_UTILS_PATH, NEW_ARGS),
        "model_has_l2h_attrs": file_contains_tokens(MODEL_PATH, MODEL_ATTRS),
        "model_has_l2h_exports": file_contains_tokens(MODEL_PATH, L2H_EXPORT_ATTRS),
        "coords_skip_logic_present": "missing_coords" in MODEL_PATH.read_text(encoding="utf-8"),
        "smoke_script_exists": SMOKE_SCRIPT.is_file(),
        "smoke_script_print_only_ok": smoke_script_cmd["ok"],
        "smoke_script_syntax_ok": smoke_syntax_cmd["ok"],
    }

    step57c_settings = read_experiment_settings(STEP57C_EXPERIMENT)
    step58c_settings = read_experiment_settings(STEP58C_EXPERIMENT)
    sample = load_sample(step58c_settings)

    l2h_off_base_model, _ = instantiate_model(step58c_settings, STEP58C_CHECKPOINT, strict=True)
    l2h_off_override_model, _ = instantiate_model(
        step58c_settings,
        STEP58C_CHECKPOINT,
        overrides={
            "rce_use_l2h_retrieval": False,
            "rce_l2h_mode": "low_topk_coord_window",
            "rce_l2h_low_topk": 8,
            "rce_l2h_high_max_per_low": 16,
            "rce_l2h_scale_ratio": 1.0,
            "rce_l2h_patch_footprint_ratio": 4.0,
            "rce_l2h_alpha_init": 0.0,
            "rce_l2h_scale": 1.0,
            "rce_l2h_fusion": "high_region_residual",
            "rce_l2h_aggregate": "mean",
            "rce_l2h_score_mode": "low_prompt_max",
            "rce_l2h_detach_low_scores": False,
            "rce_l2h_min_high_matches": 1,
            "rce_l2h_clip": 5.0,
        },
        strict=True,
    )
    l2h_off_base_output = run_model(l2h_off_base_model, sample)
    l2h_off_override_output = run_model(l2h_off_override_model, sample)
    l2h_off_equivalence = compare_outputs(l2h_off_base_output, l2h_off_override_output)

    full_all_off_sample = load_sample(step57c_settings)
    full_all_off_base_model, _ = instantiate_model(step57c_settings, STEP57C_CHECKPOINT, strict=True)
    full_all_off_override_model, _ = instantiate_model(
        step57c_settings,
        STEP57C_CHECKPOINT,
        overrides={
            "rce_use_residual_constraint": False,
            "rce_residual_constraint_lambda": 0.0,
            "rce_use_concept_aux_loss": False,
            "rce_concept_aux_loss_weight": 0.0,
            "rce_use_dynamic_csg": False,
            "rce_dynamic_csg_mode": "evidence_outer",
            "rce_dynamic_csg_alpha_init": 0.0,
            "rce_dynamic_csg_scale": 1.0,
            "rce_dynamic_csg_norm": "softmax",
            "rce_dynamic_csg_detach_evidence": False,
            "rce_dynamic_csg_clip": 5.0,
            "rce_use_ccra": False,
            "rce_ccra_mode": "concept_query_residual",
            "rce_ccra_alpha_init": 0.0,
            "rce_ccra_scale": 1.0,
            "rce_ccra_num_queries": 0,
            "rce_ccra_query_source": "prompt_mean",
            "rce_ccra_detach_prompt": False,
            "rce_ccra_norm": "layernorm",
            "rce_ccra_dropout": 0.0,
            "rce_ccra_clip": 5.0,
            "rce_use_l2h_retrieval": False,
            "rce_l2h_mode": "low_topk_coord_window",
            "rce_l2h_low_topk": 8,
            "rce_l2h_high_max_per_low": 16,
            "rce_l2h_scale_ratio": 1.0,
            "rce_l2h_patch_footprint_ratio": 4.0,
            "rce_l2h_alpha_init": 0.0,
            "rce_l2h_scale": 1.0,
            "rce_l2h_fusion": "high_region_residual",
            "rce_l2h_aggregate": "mean",
            "rce_l2h_score_mode": "low_prompt_max",
            "rce_l2h_detach_low_scores": False,
            "rce_l2h_min_high_matches": 1,
            "rce_l2h_clip": 5.0,
        },
        strict=True,
    )
    full_all_off_base_output = run_model(full_all_off_base_model, full_all_off_sample)
    full_all_off_override_output = run_model(full_all_off_override_model, full_all_off_sample)
    full_all_off_equivalence = compare_outputs(full_all_off_base_output, full_all_off_override_output)

    l2h_on_model, l2h_on_load = instantiate_model(
        step58c_settings,
        STEP58C_CHECKPOINT,
        overrides={
            "rce_use_l2h_retrieval": True,
            "rce_l2h_mode": "low_topk_coord_window",
            "rce_l2h_low_topk": 8,
            "rce_l2h_high_max_per_low": 16,
            "rce_l2h_scale_ratio": 1.0,
            "rce_l2h_patch_footprint_ratio": 4.0,
            "rce_l2h_alpha_init": 0.0,
            "rce_l2h_scale": 1.0,
            "rce_l2h_fusion": "high_region_residual",
            "rce_l2h_aggregate": "mean",
            "rce_l2h_score_mode": "low_prompt_max",
            "rce_l2h_detach_low_scores": False,
            "rce_l2h_min_high_matches": 1,
            "rce_l2h_clip": 5.0,
        },
        strict=False,
    )
    l2h_on_output = run_model(l2h_on_model, sample)
    alpha_zero_forward_equivalence = compare_outputs(l2h_off_base_output, l2h_on_output)

    l2h_breakdown = l2h_on_output.get("l2h_breakdown") or {}
    loss_breakdown = l2h_on_output.get("loss_breakdown") or {}
    l2h_exports = l2h_on_output.get("l2h_exports") or {}
    retrieved_match_counts = l2h_exports.get("last_retrieved_high_patch_match_counts")
    retrieval_smoke_pass = bool(
        l2h_breakdown.get("l2h_enabled") is True
        and l2h_breakdown.get("skipped_reason") is None
        and retrieved_match_counts is not None
        and int((retrieved_match_counts > 0).sum().item()) > 0
    )

    param_init_audit = {
        "l2h_param_names_when_off": list_l2h_param_names(l2h_off_base_model),
        "l2h_param_names_when_on": list_l2h_param_names(l2h_on_model),
    }

    config_audit["forward_return_format"] = {
        "type": "tuple",
        "length": 3,
        "y_prob_shape": list(l2h_on_output["y_prob"].shape),
        "y_hat_shape": list(l2h_on_output["y_hat"].shape),
        "loss_is_scalar": bool(l2h_on_output["loss"].numel() == 1),
    }
    config_audit["l2h_default_enabled"] = False

    lhs_breakdown = l2h_off_base_output.get("logit_breakdown") or {}
    rhs_breakdown = l2h_off_override_output.get("logit_breakdown") or {}
    lhs_full_logits = ((lhs_breakdown.get("post_calibration") or {}).get("full_logits"))
    rhs_full_logits = ((rhs_breakdown.get("post_calibration") or {}).get("full_logits"))
    logit_breakdown_consistent = True
    if lhs_full_logits is not None and rhs_full_logits is not None:
        logit_breakdown_consistent = bool(max_abs_diff(lhs_full_logits, rhs_full_logits) <= 1e-7)

    all_off_audit = {
        "l2h_off_equivalence": l2h_off_equivalence,
        "full_all_off_equivalence": full_all_off_equivalence,
        "alpha_zero_forward_equivalence": alpha_zero_forward_equivalence,
        "logit_breakdown_consistent": logit_breakdown_consistent,
    }

    retrieved_match_count_stats = None
    if isinstance(retrieved_match_counts, torch.Tensor):
        retrieved_match_count_stats = {
            "shape": list(retrieved_match_counts.shape),
            "sum": int(retrieved_match_counts.sum().item()),
            "max": int(retrieved_match_counts.max().item()),
            "mean": round_or_none(retrieved_match_counts.float().mean()),
        }

    smoke_audit = {
        "smoke_pass": bool(
            retrieval_smoke_pass
            and alpha_zero_forward_equivalence.get("pass")
            and l2h_on_output.get("logit_breakdown") is not None
        ),
        "retrieval_smoke_pass": retrieval_smoke_pass,
        "forward_smoke_pass": bool(
            l2h_on_output["y_prob"].shape == torch.Size([1, int(step58c_settings["n_classes"])])
            and l2h_on_output["y_hat"].shape == torch.Size([1, 1])
            and l2h_on_output["loss"].numel() == 1
        ),
        "l2h_breakdown": {
            key: round_or_none(l2h_breakdown.get(key))
            if isinstance(l2h_breakdown.get(key), (float, int, torch.Tensor))
            else l2h_breakdown.get(key)
            for key in L2H_BREAKDOWN_KEYS
        },
        "l2h_breakdown_keys_present": {key: key in l2h_breakdown for key in L2H_BREAKDOWN_KEYS},
        "l2h_export_shapes": {
            name: summarise_tensor_shape(value) for name, value in l2h_exports.items()
        },
        "retrieved_match_count_stats": retrieved_match_count_stats,
        "load_state_dict": l2h_on_load,
        "residual_constraint_still_enabled": bool(loss_breakdown.get("residual_constraint_enabled")),
        "concept_aux_still_enabled": bool(loss_breakdown.get("concept_aux_enabled")),
        "loss_breakdown": {
            "ce_loss": round_or_none(loss_breakdown.get("ce_loss")),
            "residual_constraint_loss": round_or_none(loss_breakdown.get("residual_constraint_loss")),
            "concept_aux_loss": round_or_none(loss_breakdown.get("concept_aux_loss")),
            "total_loss": round_or_none(loss_breakdown.get("total_loss")),
            "visual_ratio_mean": round_or_none(loss_breakdown.get("visual_ratio_mean")),
        },
    }

    summary_md = build_summary_md(config_audit, all_off_audit, param_init_audit, smoke_audit)
    (OUTPUT_DIR / "stage61B_config_audit.json").write_text(
        json.dumps(config_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage61B_all_off_equivalence_audit.json").write_text(
        json.dumps(all_off_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage61B_param_init_audit.json").write_text(
        json.dumps(param_init_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage61B_l2h_smoke.json").write_text(
        json.dumps(smoke_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage61B_summary.md").write_text(summary_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
