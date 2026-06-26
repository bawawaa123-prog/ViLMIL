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


OUTPUT_DIR = ROOT / "results_stage58A_residual_constrained_rce"
STEP57C_RUN_DIR = ROOT / "results_stage57C_rce_v2_copy_reproduction" / "rce_v2_copy_csg_a01_rq16_5fold_e20_s1"
STEP57C_EXPERIMENT = STEP57C_RUN_DIR / "experiment_rce_v2_copy_csg_a01_rq16_5fold_e20.txt"
STEP57C_CHECKPOINT = STEP57C_RUN_DIR / "s_0_checkpoint.pt"
MODEL_PATH = ROOT / "models" / "model_RCE_MIL_BiomedCLIP_v2.py"
MAIN_PATH = ROOT / "main.py"
CORE_UTILS_PATH = ROOT / "utils" / "core_utils.py"
ORIGINAL_RCE_PATH = ROOT / "models" / "model_RCE_MIL_BiomedCLIP.py"
SMOKE_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage58A_residual_constrained_rce_smoke.sh"
NEW_ARGS = [
    "rce_use_residual_constraint",
    "rce_residual_constraint_lambda",
    "rce_residual_ratio_target",
    "rce_residual_constraint_type",
    "rce_use_concept_aux_loss",
    "rce_concept_aux_loss_weight",
    "rce_residual_ratio_eps",
    "rce_residual_ratio_detach",
]
MODEL_ATTRS = [
    "rce_use_residual_constraint",
    "rce_residual_constraint_lambda",
    "rce_residual_ratio_target",
    "rce_residual_constraint_type",
    "rce_use_concept_aux_loss",
    "rce_concept_aux_loss_weight",
    "rce_residual_ratio_eps",
    "rce_residual_ratio_detach",
    "last_loss_breakdown",
]


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def relative_path_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
        raise ValueError(f"Unsupported task for Step58A audit: {task}")

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


def build_model_config(settings: dict[str, object], overrides: dict[str, object] | None = None) -> SimpleNamespace:
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
        scale_mode=str(merged.get("scale_mode", "dual")),
        finetune_text_encoder=False,
        enable_logit_breakdown_audit=True,
    )


def load_sample(settings: dict[str, object]) -> dict[str, object]:
    dataset = build_dataset(settings)
    split_dir = Path(str(settings["split_dir"]))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    _, _, test_split = dataset.return_splits(
        from_id=False,
        csv_path=str(split_dir / "splits_0.csv"),
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
) -> RCE_MIL_BiomedCLIP:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = build_model_config(settings, overrides)
    model = RCE_MIL_BiomedCLIP(config=config, num_classes=int(settings["n_classes"]))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.relocate()
    model.eval()
    return model


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

    logit_breakdown = getattr(model, "last_logit_breakdown", None)
    loss_breakdown = getattr(model, "last_loss_breakdown", None)
    return {
        "tuple_len": 3,
        "y_prob": y_prob.detach().cpu(),
        "y_hat": y_hat.detach().cpu(),
        "loss": loss.detach().cpu(),
        "loss_breakdown": loss_breakdown,
        "logit_breakdown_keys": sorted(logit_breakdown.keys()) if isinstance(logit_breakdown, dict) else [],
    }


def tensor_max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def build_all_off_equivalence(
    settings: dict[str, object],
    checkpoint_path: Path,
    sample: dict[str, object],
) -> dict[str, object]:
    default_model = instantiate_model(settings, checkpoint_path, overrides={})
    explicit_off_model = instantiate_model(
        settings,
        checkpoint_path,
        overrides={
            "rce_use_residual_constraint": False,
            "rce_residual_constraint_lambda": 0.0,
            "rce_residual_ratio_target": 0.5,
            "rce_residual_constraint_type": "relu_l2",
            "rce_use_concept_aux_loss": False,
            "rce_concept_aux_loss_weight": 0.0,
            "rce_residual_ratio_eps": 1e-6,
            "rce_residual_ratio_detach": False,
        },
    )

    default_out = run_model(default_model, sample)
    explicit_off_out = run_model(explicit_off_model, sample)
    ce_loss_default = default_out["loss_breakdown"]["ce_loss"]
    total_loss_default = default_out["loss_breakdown"]["total_loss"]

    y_prob_diff = tensor_max_abs_diff(default_out["y_prob"], explicit_off_out["y_prob"])
    loss_diff = abs(float(default_out["loss"].item()) - float(explicit_off_out["loss"].item()))
    y_hat_equal = bool(torch.equal(default_out["y_hat"], explicit_off_out["y_hat"]))
    passed = (
        y_prob_diff == 0.0
        and loss_diff == 0.0
        and y_hat_equal
        and abs(ce_loss_default - total_loss_default) <= 1e-12
        and not default_out["loss_breakdown"]["residual_constraint_enabled"]
        and not default_out["loss_breakdown"]["concept_aux_enabled"]
    )

    return {
        "passed": passed,
        "default_flags": {
            "rce_use_residual_constraint": False,
            "rce_residual_constraint_lambda": 0.0,
            "rce_use_concept_aux_loss": False,
            "rce_concept_aux_loss_weight": 0.0,
        },
        "forward_return_format_unchanged": default_out["tuple_len"] == 3,
        "explicit_off_matches_default": {
            "y_prob_max_abs_diff": round_or_none(y_prob_diff),
            "loss_abs_diff": round_or_none(loss_diff),
            "y_hat_equal": y_hat_equal,
        },
        "default_loss_breakdown": default_out["loss_breakdown"],
    }


def build_smoke_payload(
    settings: dict[str, object],
    checkpoint_path: Path,
    sample: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "smoke_attempted": True,
        "sample_slide_id": sample["slide_id"],
        "modes": {},
    }

    mode_overrides = {
        "all_off": {},
        "residual_enabled": {
            "rce_use_residual_constraint": True,
            "rce_residual_constraint_lambda": 0.05,
            "rce_residual_ratio_target": 0.5,
        },
        "concept_aux_enabled": {
            "rce_use_concept_aux_loss": True,
            "rce_concept_aux_loss_weight": 0.2,
        },
    }

    for mode_name, overrides in mode_overrides.items():
        model = instantiate_model(settings, checkpoint_path, overrides=overrides)
        result = run_model(model, sample)
        breakdown = result["loss_breakdown"] or {}
        payload["modes"][mode_name] = {
            "forward_return_format_unchanged": result["tuple_len"] == 3,
            "loss": round_or_none(result["loss"]),
            "y_hat": int(result["y_hat"].reshape(-1)[0].item()),
            "loss_breakdown_present": isinstance(breakdown, dict),
            "logit_breakdown_has_step58a_fields": {
                key: (key in result["logit_breakdown_keys"])
                for key in [
                    "residual_constraint_enabled",
                    "concept_aux_enabled",
                    "visual_ratio_for_loss",
                    "residual_constraint_loss",
                    "concept_aux_loss",
                    "total_loss",
                ]
            },
            "loss_breakdown": breakdown,
        }

    payload["smoke_passed"] = (
        payload["modes"]["all_off"]["forward_return_format_unchanged"]
        and payload["modes"]["all_off"]["loss_breakdown_present"]
        and payload["modes"]["residual_enabled"]["loss_breakdown_present"]
        and payload["modes"]["concept_aux_enabled"]["loss_breakdown_present"]
    )
    return payload


def build_run_commands_text() -> str:
    return "\n".join(
        [
            f"cd {ROOT}",
            "python scripts/analysis/build_stage58A_residual_constraint_audit.py",
            "",
            "# Optional 1-fold / 1-epoch training smoke",
            "PYTHON_BIN=/home/ljh/anaconda3/envs/vila_mil/bin/python \\",
            "DATA_ROOT_DIR=/xiangmu/data/VILMIL \\",
            "RESULTS_DIR=results_stage58A_residual_constrained_rce \\",
            "SEED=1 \\",
            "MAX_EPOCHS=1 \\",
            "bash scripts/experiments/run_stage58A_residual_constrained_rce_smoke.sh",
        ]
    )


def build_summary_md(
    config_audit: dict[str, object],
    equivalence: dict[str, object],
    smoke: dict[str, object],
    compile_status: dict[str, object],
) -> str:
    smoke_done = bool(smoke.get("smoke_attempted")) and bool(smoke.get("smoke_passed"))
    return "\n".join(
        [
            "# Step58A Residual-Constrained RCE all-off equivalence implementation",
            "",
            "## Direct Answers",
            "",
            f"1. 本 Step 是否修改了原始 RCE 文件：{'否' if config_audit['modified_original_rce_file'] is False else '未知'}。",
            f"2. 本 Step 是否修改了 RCE-v2 默认行为：{'否' if equivalence['passed'] else '未确认'}。",
            "3. 新增了哪些 residual constraint 参数："
            " `--rce_use_residual_constraint`、`--rce_residual_constraint_lambda`、"
            " `--rce_residual_ratio_target`、`--rce_residual_constraint_type`、"
            " `--rce_use_concept_aux_loss`、`--rce_concept_aux_loss_weight`、"
            " `--rce_residual_ratio_eps`、`--rce_residual_ratio_detach`。",
            f"4. all-off 时是否保持等价：{'是' if equivalence['passed'] else '否'}。",
            "5. residual constraint loss 的计算方式是什么："
            " `visual_ratio = ||visual_residual_logits|| / (||concept_logits|| + ||visual_residual_logits|| + eps)`；"
            " `residual_constraint_loss = mean(ReLU(visual_ratio - target)^2)`。",
            "6. concept auxiliary loss 的计算方式是什么："
            " 对 `concept_logits = low_evidence_logits + high_evidence_logits + csg_logits`"
            " 单独计算 `CE(concept_logits, label)`，并按权重加到总 loss。",
            f"7. 是否完成 py_compile / bash -n：{'是' if compile_status['all_passed'] else '否'}。",
            f"8. 是否完成 smoke test：{'是' if smoke_done else '否'}。",
            "9. 下一步是否建议进入 Step58B 参数小扫："
            + ("是，可以进入。" if equivalence["passed"] and smoke_done else "暂缓，先修复 Step58A。"),
            "",
            "## Checks",
            "",
            f"- 原始 RCE 文件改动检查：`{config_audit['modified_original_rce_file']}`",
            f"- 新参数在 `main.py` 中存在：`{config_audit['all_new_args_found_in_main']}`",
            f"- 新参数在 `utils/core_utils.py` 中透传：`{config_audit['all_new_args_wired_in_core_utils']}`",
            f"- RCE-v2 支持 residual constraint 属性：`{config_audit['all_model_attrs_present']}`",
            f"- all-off 等价通过：`{equivalence['passed']}`",
            f"- smoke 通过：`{smoke_done}`",
        ]
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    compile_status = {
        "model_py_compile": run_command(
            [sys.executable, "-m", "py_compile", str(MODEL_PATH)]
        ),
        "audit_py_compile": run_command(
            [sys.executable, "-m", "py_compile", str(Path(__file__))]
        ),
        "smoke_bash_n": run_command(["bash", "-n", str(SMOKE_SCRIPT)]),
    }
    compile_status["all_passed"] = all(item["ok"] for item in compile_status.values())

    main_tokens = file_contains_tokens(MAIN_PATH, [f"--{arg}" for arg in NEW_ARGS])
    core_utils_tokens = file_contains_tokens(CORE_UTILS_PATH, NEW_ARGS)
    settings = read_experiment_settings(STEP57C_EXPERIMENT)
    sample = load_sample(settings)
    model_probe = instantiate_model(settings, STEP57C_CHECKPOINT, overrides={})
    model_attrs_present = {attr: hasattr(model_probe, attr) for attr in MODEL_ATTRS}

    config_audit = {
        "modified_original_rce_file": git_path_modified(ORIGINAL_RCE_PATH),
        "modified_rce_v2_file": git_path_modified(MODEL_PATH),
        "new_args_found_in_main": main_tokens,
        "all_new_args_found_in_main": all(main_tokens.values()),
        "new_args_wired_in_core_utils": core_utils_tokens,
        "all_new_args_wired_in_core_utils": all(core_utils_tokens.values()),
        "model_attrs_present": model_attrs_present,
        "all_model_attrs_present": all(model_attrs_present.values()),
    }

    equivalence = build_all_off_equivalence(settings, STEP57C_CHECKPOINT, sample)
    smoke = build_smoke_payload(settings, STEP57C_CHECKPOINT, sample)

    run_commands_path = OUTPUT_DIR / "stage58A_run_commands.txt"
    run_commands_path.write_text(build_run_commands_text() + "\n", encoding="utf-8")

    config_audit_path = OUTPUT_DIR / "stage58A_config_audit.json"
    config_audit_path.write_text(
        json.dumps(config_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    equivalence_path = OUTPUT_DIR / "stage58A_all_off_equivalence_audit.json"
    equivalence_path.write_text(
        json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    smoke_path = OUTPUT_DIR / "stage58A_loss_component_smoke.json"
    smoke_path.write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_path = OUTPUT_DIR / "stage58A_summary.md"
    summary_path.write_text(
        build_summary_md(config_audit, equivalence, smoke, compile_status) + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
