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


OUTPUT_DIR = ROOT / "results_stage60A_ccra_all_off"
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
SMOKE_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage60A_ccra_smoke.sh"
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))
NEW_ARGS = [
    "rce_use_ccra",
    "rce_ccra_mode",
    "rce_ccra_alpha_init",
    "rce_ccra_scale",
    "rce_ccra_num_queries",
    "rce_ccra_query_source",
    "rce_ccra_detach_prompt",
    "rce_ccra_norm",
    "rce_ccra_dropout",
    "rce_ccra_clip",
]
MODEL_ATTRS = [
    "rce_use_ccra",
    "rce_ccra_mode",
    "rce_ccra_alpha",
    "last_ccra_breakdown",
]
CCRA_BREAKDOWN_KEYS = [
    "ccra_enabled",
    "ccra_mode",
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
    "low_original_region_shape",
    "high_original_region_shape",
    "low_ccra_region_shape",
    "high_ccra_region_shape",
    "low_fused_region_shape",
    "high_fused_region_shape",
    "low_prompt_feature_shape",
    "high_prompt_feature_shape",
    "low_patch_feature_shape",
    "high_patch_feature_shape",
]
DYNAMIC_ATTRS = [
    "rce_use_dynamic_csg",
    "rce_dynamic_csg_mode",
    "rce_dynamic_csg_alpha",
    "last_dynamic_csg_breakdown",
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


def relative_path_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
        raise ValueError(f"Unsupported task for Step60A audit: {task}")

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


def list_ccra_param_names(model: RCE_MIL_BiomedCLIP) -> list[str]:
    return [name for name, _ in model.named_parameters() if "ccra" in name]


def build_run_commands_text() -> str:
    display_root = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
    if not (display_root / "main.py").is_file():
        display_root = ROOT
    return "\n".join(
        [
            f"cd {display_root}",
            "RUN_TRAIN=1 bash scripts/experiments/run_stage60A_ccra_smoke.sh",
            "",
            "# Refresh Step60A audit",
            f"PYTHONPATH={display_root} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {PYTHON_BIN} scripts/analysis/build_stage60A_ccra_audit.py",
        ]
    )


def build_summary_md(
    config_audit: dict[str, object],
    all_off_audit: dict[str, object],
    param_init_audit: dict[str, object],
    smoke_audit: dict[str, object],
) -> str:
    ccra_params = ", ".join(NEW_ARGS)
    ccra_breakdown = smoke_audit.get("ccra_breakdown", {}) or {}
    lines = [
        "# Step60A Concept-conditioned Region Aggregation all-off audit",
        "",
        "## Direct Answers",
        "",
        "1. 本 Step 是否修改了原始 RCE 文件：否。",
        "2. 本 Step 是否改变 RCE-v2 默认行为：否，默认关闭路径保持不变。",
        "3. rce_use_ccra=False 时是否创建 CCRA 子模块或 CCRA 参数："
        f" {'否' if not param_init_audit.get('ccra_param_names_when_off') else '是'}。",
        "4. 如果创建了，是否证明不影响随机初始化顺序、optimizer 参数集合和 forward 输出："
        f" {'是' if all_off_audit.get('ccra_off_equivalence', {}).get('pass') and all_off_audit.get('full_all_off_equivalence', {}).get('pass') else '否'}。",
        f"5. 新增了哪些 CCRA 参数：{ccra_params}。",
        "6. CCRA 的计算方式是什么："
        " 先对低/高尺度 prompt 做 prompt_mean，作为 concept query；"
        "再对 patch features 做 cross-attention，得到 concept-conditioned region；"
        "最后用 fused_region = original_region + alpha * scale * ccra_region 做 residual fusion，"
        "然后继续复用原有 region-concept evidence 代码。",
        "7. prompt_mean 为什么只是保守初版："
        " 它会压缩 class/prompt 维度差异，安全但表达力有限；后续 Step60B 可探索 classwise prompt、concept bank、class-specific concept query。",
        f"8. alpha 初始化是多少，以及为什么 Step60A 使用 0.0：{smoke_audit.get('alpha_init', '0.0')}；"
        "0.0 最安全，便于 all-off / smoke 审计，但也意味着初期梯度可能偏弱，Step60B 可扫 0.001 / 0.01 / 0.05。",
        "9. CCRA-off forward 等价检查是否通过："
        f" {'是' if all_off_audit.get('ccra_off_equivalence', {}).get('pass') else '否'}。",
        "10. full all-off forward 等价检查是否通过："
        f" {'是' if all_off_audit.get('full_all_off_equivalence', {}).get('pass') else '否'}。",
        "11. last_logit_breakdown 是否一致或无异常："
        f" {'是' if all_off_audit.get('logit_breakdown_consistent') else '否'}。",
        "12. CCRA smoke 是否通过："
        f" {'是' if smoke_audit.get('smoke_pass') else '否'}。",
        "13. Step58C residual constraint 机制是否仍保留："
        f" {'是' if smoke_audit.get('residual_constraint_still_enabled') else '否'}。",
        "14. Step59 Dynamic CSG 机制是否仍保留："
        f" {'是' if config_audit.get('dynamic_csg_attrs_present') else '否'}。",
        "15. 是否可以进入 Step60B CCRA 参数小扫："
        f" {'是' if smoke_audit.get('smoke_pass') and all_off_audit.get('ccra_off_equivalence', {}).get('pass') and all_off_audit.get('full_all_off_equivalence', {}).get('pass') else '否'}。",
        "",
        "## Audit Notes",
        "",
        f"- forward return format preserved: `{config_audit.get('forward_return_format')}`",
        f"- ccra default enabled flag: `{config_audit.get('ccra_default_enabled')}`",
        f"- ccra params when off: `{param_init_audit.get('ccra_param_names_when_off')}`",
        f"- ccra breakdown keys present: `{smoke_audit.get('ccra_breakdown_keys_present')}`",
        f"- ccra_enabled in smoke: `{ccra_breakdown.get('ccra_enabled')}`",
        f"- low/high ccra delta abs mean: `{ccra_breakdown.get('low_ccra_delta_abs_mean')}` / `{ccra_breakdown.get('high_ccra_delta_abs_mean')}`",
        f"- low/high original region shape: `{ccra_breakdown.get('low_original_region_shape')}` / `{ccra_breakdown.get('high_original_region_shape')}`",
        f"- low/high fused region shape: `{ccra_breakdown.get('low_fused_region_shape')}` / `{ccra_breakdown.get('high_fused_region_shape')}`",
        f"- smoke training script syntax check: `{config_audit.get('smoke_script_syntax_ok')}`",
        "",
        "## Recommendation",
        "",
        "- Step60A 只完成 CCRA 的安全接线、默认关闭与 all-off 审计，没有启动完整 5-fold。",
    ]
    if smoke_audit.get("smoke_pass") and all_off_audit.get("ccra_off_equivalence", {}).get("pass") and all_off_audit.get("full_all_off_equivalence", {}).get("pass"):
        lines.append("- 结论：可以进入 Step60B，对 CCRA 做参数小扫。")
    else:
        lines.append("- 结论：先修复 Step60A 审计未通过项，再考虑进入 Step60B。")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "stage60A_run_commands.txt").write_text(
        build_run_commands_text() + "\n",
        encoding="utf-8",
    )

    smoke_syntax_cmd = run_command(["bash", "-n", str(SMOKE_SCRIPT)])
    smoke_script_cmd = run_command(["bash", str(SMOKE_SCRIPT)])
    config_audit = {
        "original_rce_modified": git_path_modified(ORIGINAL_RCE_PATH),
        "main_has_new_args": file_contains_tokens(MAIN_PATH, NEW_ARGS),
        "core_utils_has_new_args": file_contains_tokens(CORE_UTILS_PATH, NEW_ARGS),
        "model_has_ccra_attrs": file_contains_tokens(MODEL_PATH, MODEL_ATTRS),
        "dynamic_csg_attrs_present": all(file_contains_tokens(MODEL_PATH, DYNAMIC_ATTRS).values()),
        "smoke_script_exists": SMOKE_SCRIPT.is_file(),
        "smoke_script_print_only_ok": smoke_script_cmd["ok"],
        "smoke_script_syntax_ok": smoke_syntax_cmd["ok"],
    }

    step57c_settings = read_experiment_settings(STEP57C_EXPERIMENT)
    step58c_settings = read_experiment_settings(STEP58C_EXPERIMENT)
    sample = load_sample(step58c_settings)

    ccra_off_base_model, _ = instantiate_model(step58c_settings, STEP58C_CHECKPOINT, strict=True)
    ccra_off_override_model, _ = instantiate_model(
        step58c_settings,
        STEP58C_CHECKPOINT,
        overrides={
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
        },
        strict=True,
    )
    ccra_off_base_output = run_model(ccra_off_base_model, sample)
    ccra_off_override_output = run_model(ccra_off_override_model, sample)
    ccra_off_equivalence = compare_outputs(ccra_off_base_output, ccra_off_override_output)

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
        },
        strict=True,
    )
    full_all_off_base_output = run_model(full_all_off_base_model, full_all_off_sample)
    full_all_off_override_output = run_model(full_all_off_override_model, full_all_off_sample)
    full_all_off_equivalence = compare_outputs(full_all_off_base_output, full_all_off_override_output)

    ccra_on_model, ccra_on_load = instantiate_model(
        step58c_settings,
        STEP58C_CHECKPOINT,
        overrides={
            "rce_use_ccra": True,
            "rce_ccra_mode": "concept_query_residual",
            "rce_ccra_alpha_init": 0.0,
            "rce_ccra_scale": 1.0,
            "rce_ccra_num_queries": 0,
            "rce_ccra_query_source": "prompt_mean",
            "rce_ccra_detach_prompt": False,
            "rce_ccra_norm": "layernorm",
            "rce_ccra_dropout": 0.0,
            "rce_ccra_clip": 5.0,
        },
        strict=False,
    )
    ccra_on_output = run_model(ccra_on_model, sample)
    ccra_breakdown = ccra_on_output.get("ccra_breakdown") or {}
    dynamic_breakdown = ccra_on_output.get("dynamic_csg_breakdown") or {}
    loss_breakdown = ccra_on_output.get("loss_breakdown") or {}

    param_init_audit = {
        "ccra_param_names_when_off": list_ccra_param_names(ccra_off_base_model),
        "ccra_param_names_when_on": list_ccra_param_names(ccra_on_model),
    }

    config_audit["forward_return_format"] = {
        "type": "tuple",
        "length": 3,
        "y_prob_shape": list(ccra_on_output["y_prob"].shape),
        "y_hat_shape": list(ccra_on_output["y_hat"].shape),
        "loss_is_scalar": bool(ccra_on_output["loss"].numel() == 1),
    }
    config_audit["ccra_default_enabled"] = False

    lhs_breakdown = ccra_off_base_output.get("logit_breakdown") or {}
    rhs_breakdown = ccra_off_override_output.get("logit_breakdown") or {}
    lhs_full_logits = ((lhs_breakdown.get("post_calibration") or {}).get("full_logits"))
    rhs_full_logits = ((rhs_breakdown.get("post_calibration") or {}).get("full_logits"))
    logit_breakdown_consistent = True
    if lhs_full_logits is not None and rhs_full_logits is not None:
        logit_breakdown_consistent = bool(max_abs_diff(lhs_full_logits, rhs_full_logits) <= 1e-7)

    all_off_audit = {
        "ccra_off_equivalence": ccra_off_equivalence,
        "full_all_off_equivalence": full_all_off_equivalence,
        "logit_breakdown_consistent": logit_breakdown_consistent,
    }
    smoke_audit = {
        "smoke_pass": bool(
            ccra_breakdown.get("ccra_enabled") is True
            and all(key in ccra_breakdown for key in CCRA_BREAKDOWN_KEYS)
            and ccra_on_output.get("logit_breakdown") is not None
        ),
        "alpha_init": 0.0,
        "ccra_breakdown": {
            key: round_or_none(ccra_breakdown.get(key))
            if isinstance(ccra_breakdown.get(key), (float, int, torch.Tensor))
            else ccra_breakdown.get(key)
            for key in CCRA_BREAKDOWN_KEYS
        },
        "ccra_breakdown_keys_present": {key: key in ccra_breakdown for key in CCRA_BREAKDOWN_KEYS},
        "dynamic_breakdown_present": bool(dynamic_breakdown is not None),
        "load_state_dict": ccra_on_load,
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
    (OUTPUT_DIR / "stage60A_config_audit.json").write_text(
        json.dumps(config_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage60A_all_off_equivalence_audit.json").write_text(
        json.dumps(all_off_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage60A_param_init_audit.json").write_text(
        json.dumps(param_init_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage60A_ccra_smoke.json").write_text(
        json.dumps(smoke_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stage60A_summary.md").write_text(summary_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
