#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_DEG_MIL_BiomedCLIP import DEG_MIL_BiomedCLIP
from models.model_RCE_MIL_BiomedCLIP import RCE_MIL_BiomedCLIP


TASK = "task_adenocarcinoma"
DATA_ROOT_DIR = Path(os.environ.get("DATA_ROOT_DIR", "/xiangmu/data/VILMIL"))
DATA_FOLDER_S = "features_biomedclip_5x"
DATA_FOLDER_L = "features_biomedclip_20x"
CONCEPT_PROMPT_PATH = ROOT_DIR / "dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json"
TEXT_PROMPT_PATH = ROOT_DIR / "text_prompt/adenocarcinoma_dual_scale_prompt.csv"
SPLIT_DIR = ROOT_DIR / "splits/adenocarcinoma/task_adenocarcinoma_strictcv_100"
DATASET_CSV = ROOT_DIR / "dataset_csv/all_data.csv"
DOC_MD = ROOT_DIR / "docs/stage51b_deg_skeleton_equivalence_audit.md"
DOC_CSV = ROOT_DIR / "docs/stage51b_deg_skeleton_equivalence_audit.csv"
STAGE51_SCRIPT = ROOT_DIR / "scripts/experiments/run_stage51_reproduce_rce_and_deg_skeleton.sh"
STAGE23_SCRIPT = ROOT_DIR / "scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh"
MAIN_PY = ROOT_DIR / "main.py"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_config() -> SimpleNamespace:
    return SimpleNamespace(
        input_size=512,
        hidden_size=192,
        class_names=["Adenocarcinoma", "NonAdenocarcinoma"],
        use_concept_prompt_pool=True,
        concept_prompt_path=str(CONCEPT_PROMPT_PATH),
        peps_tau=0.1,
        prototype_number=16,
        rce_use_logit_calibration=True,
        rce_use_concept_prior=True,
        rce_logit_scale_init=10.0,
        rce_concept_prior_strength=1.0,
        rce_use_visual_residual=True,
        rce_visual_residual_init=0.05,
        rce_use_visual_evidence_gate=False,
        rce_visual_gate_init=1.0,
        rce_use_prarc_gate=False,
        rce_prarc_gate_version="v1",
        rce_prarc_gate_hidden_dim=16,
        rce_prarc_gate_init=0.8,
        rce_prarc_gate_dropout=0.0,
        rce_prarc_gate_gain=1.0,
        rce_prarc_gate_last_weight_init=0.01,
        rce_prarc_gate_feature_set="v1",
        rce_prarc_detach_features=False,
        rce_prarc_include_optional_features=False,
        rce_prarc_feature_clip=10.0,
        rce_prarc_export_debug=False,
        rce_prarc_use_conflict_prior=False,
        rce_prarc_conflict_prior_strength=0.2,
        rce_prarc_use_gate_entropy_reg=False,
        rce_prarc_gate_entropy_lambda=0.0,
        rce_prarc_use_gate_variance_reg=False,
        rce_prarc_gate_variance_lambda=0.0,
        rce_use_low_high_consistency_loss=False,
        rce_lh_consistency_lambda=0.0,
        rce_lh_consistency_margin=0.0,
        rce_use_cross_scale_graph=True,
        rce_cross_scale_graph_init=0.1,
        rce_cross_scale_graph_norm="sqrt",
        rce_use_hcrc=False,
        rce_hcrc_alpha_init=0.05,
        rce_hcrc_num_anchors=16,
        rce_hcrc_num_high_children=16,
        rce_hcrc_proposal_radius=4096.0,
        rce_hcrc_nms_radius=512.0,
        rce_hcrc_bbox_expand=8.0,
        rce_hcrc_coord_mode="top_left",
        rce_hcrc_scale_ratio=1.0,
        rce_hcrc_child_strategy="bbox_containment",
        rce_hcrc_candidate_top_l=64,
        rce_hcrc_top_g_concepts=8,
        rce_hcrc_per_concept_top_m=4,
        rce_hcrc_prompt_topk=3,
        rce_hcrc_margin_weight=0.5,
        rce_hcrc_prompt_scale="high",
        rce_hcrc_min_child_count=1,
        rce_hcrc_export_debug=False,
        deg_use_region_graph=False,
        deg_region_graph_k=4,
        deg_region_graph_alpha=0.1,
        deg_use_concept_graph=False,
        deg_concept_graph_topk=4,
        deg_concept_graph_alpha=0.05,
        scale_mode="dual",
        scale_fusion_mode="sum",
        scale_gate_hidden_dim=128,
        scale_gate_dropout=0.25,
        scale_residual_gamma=0.25,
        allow_legacy_scale_fusion_ckpt=False,
        finetune_text_encoder=False,
    )


def parse_shell_assignments(script_text: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line in script_text.splitlines():
        stripped = line.strip()
        match = re.match(r"^([A-Za-z0-9_]+)=(.+)$", stripped)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip()
        if value.startswith("${") or "$(" in value:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        assignments[key] = value
    return assignments


def parse_stage23_variant_assignments(script_text: str, variant: str = "rq16") -> dict[str, str]:
    pattern = re.compile(
        rf"{re.escape(variant)}\)\s*(.*?)\s*;;",
        re.DOTALL,
    )
    match = pattern.search(script_text)
    if not match:
        return {}
    return parse_shell_assignments(match.group(1))


def extract_flag_value(script_path: Path, flag: str) -> str | None:
    text = script_path.read_text(encoding="utf-8")
    assignments = parse_shell_assignments(text)
    if script_path == STAGE23_SCRIPT:
        assignments.update(parse_stage23_variant_assignments(text, variant="rq16"))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or f"--{flag}" not in line:
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        flag_match = re.search(rf"(^|\s)--{re.escape(flag)}(\s|$)", line)
        if not flag_match:
            continue
        remainder = line[flag_match.end():].strip()
        if not remainder or remainder.startswith("--"):
            return "true"
        value = remainder.split()[0].strip().strip('"').strip("'")
        var_match = re.fullmatch(r"\$\{?([A-Za-z0-9_]+)(?::-([^}]+))?\}?", value)
        if var_match:
            var_name = var_match.group(1)
            default_value = var_match.group(2)
            return assignments.get(var_name, default_value if default_value is not None else value)
        return value
    return None


def normalize_flag_value(value: str | None, assignments: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().strip('"').strip("'")
    if assignments is None:
        assignments = {}

    for _ in range(8):
        var_match = re.fullmatch(r"\$\{?([A-Za-z0-9_]+)(?::-([^}]+))?\}?", normalized)
        if not var_match:
            break
        var_name = var_match.group(1)
        default_value = var_match.group(2)
        replacement = assignments.get(var_name, default_value if default_value is not None else normalized)
        if replacement == normalized:
            break
        normalized = str(replacement).strip().strip('"').strip("'")
    return normalized


def compare_script_flags(rows: list[dict[str, str]]) -> None:
    stage51_text = STAGE51_SCRIPT.read_text(encoding="utf-8")
    stage23_text = STAGE23_SCRIPT.read_text(encoding="utf-8")
    stage51_assignments = parse_shell_assignments(stage51_text)
    stage23_assignments = parse_shell_assignments(stage23_text)
    stage23_assignments.update(parse_stage23_variant_assignments(stage23_text, variant="rq16"))
    expected = {
        "prototype_number": "16",
        "rce_use_logit_calibration": "true",
        "rce_use_concept_prior": "true",
        "rce_use_visual_residual": "true",
        "rce_use_cross_scale_graph": "true",
        "rce_cross_scale_graph_init": "0.1",
        "rce_cross_scale_graph_norm": "sqrt",
        "rce_logit_scale_init": "10.0",
        "rce_concept_prior_strength": "1.0",
        "rce_visual_residual_init": "0.05",
        "concept_prompt_path": "${ROOT_DIR}/dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json",
        "split_dir": "splits/adenocarcinoma/task_adenocarcinoma_strictcv_100",
        "text_prompt_path": "${ROOT_DIR}/text_prompt/adenocarcinoma_dual_scale_prompt.csv",
        "data_folder_s": "features_biomedclip_5x",
        "data_folder_l": "features_biomedclip_20x",
        "scale_mode": "dual",
        "prompt_ensemble_mode": "embedding_mean",
        "k": "${K_FOLDS}",
        "k_start": "${K_START}",
        "k_end": "${K_END}",
        "max_epochs": "${MAX_EPOCHS}",
        "seed": "${SEED}",
    }
    for flag, expected_value in expected.items():
        stage51_value = normalize_flag_value(
            extract_flag_value(STAGE51_SCRIPT, flag),
            assignments=stage51_assignments,
        )
        stage23_value = normalize_flag_value(
            extract_flag_value(STAGE23_SCRIPT, flag),
            assignments=stage23_assignments,
        )
        expected_norm = normalize_flag_value(expected_value, assignments=stage23_assignments)
        status = (
            "PASS"
            if stage51_value is not None and stage51_value == stage23_value
            else "FAIL"
        )
        details = f"stage51={stage51_value} | stage23={stage23_value} | expected_ref={expected_value}"
        rows.append(
            {
                "category": "static_script_flag",
                "check": flag,
                "status": status,
                "details": details,
            }
        )

    disabled_flags = [
        "deg_use_region_graph",
        "deg_use_concept_graph",
        "rce_use_visual_evidence_gate",
        "rce_use_prarc_gate",
        "rce_use_hcrc",
        "rce_use_low_high_consistency_loss",
    ]
    main_text = MAIN_PY.read_text(encoding="utf-8")
    for flag in disabled_flags:
        default_false = re.search(
            rf'add_argument\("--{re.escape(flag)}".*default=False',
            main_text,
        ) is not None
        stage51_present = extract_flag_value(STAGE51_SCRIPT, flag)
        status = "PASS" if default_false and stage51_present is None else "FAIL"
        rows.append(
            {
                "category": "static_disabled_module",
                "check": flag,
                "status": status,
                "details": f"stage51_flag_present={stage51_present is not None} | cli_default_false={default_false}",
            }
        )


def ensure_required_paths() -> None:
    required_paths = [
        DATA_ROOT_DIR,
        DATA_ROOT_DIR / DATA_FOLDER_S,
        DATA_ROOT_DIR / DATA_FOLDER_L,
        CONCEPT_PROMPT_PATH,
        TEXT_PROMPT_PATH,
        SPLIT_DIR,
        DATASET_CSV,
        SPLIT_DIR / "splits_0.csv",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))


def load_real_batch() -> dict[str, object]:
    ensure_required_paths()
    dataset = Generic_MIL_Dataset(
        csv_path=str(DATASET_CSV),
        mode="transformer",
        data_dir_s=str(DATA_ROOT_DIR / DATA_FOLDER_S),
        data_dir_l=str(DATA_ROOT_DIR / DATA_FOLDER_L),
        shuffle=False,
        print_info=False,
        label_dict={"Adenocarcinoma": 0, "NonAdenocarcinoma": 1},
        patient_strat=False,
        ignore=[],
    )
    train_split, _, _ = dataset.return_splits(
        from_id=False,
        csv_path=str(SPLIT_DIR / "splits_0.csv"),
    )
    if train_split is None or len(train_split) == 0:
        raise RuntimeError("Split file exists but train split is empty.")
    features_s, coords_s, features_l, coords_l, label, slide_id = train_split[0]
    return {
        "x_s": features_s.float(),
        "coord_s": coords_s,
        "x_l": features_l.float(),
        "coords_l": coords_l,
        "label": torch.tensor([int(label)], dtype=torch.long),
        "slide_id": slide_id,
    }


def run_model(model: torch.nn.Module, batch: dict[str, object]) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        y_prob, y_hat, loss = model(
            batch["x_s"],
            batch["coord_s"],
            batch["x_l"],
            batch["coords_l"],
            batch["label"],
            slide_id=batch["slide_id"],
        )
    outputs: dict[str, object] = {
        "y_prob": y_prob.detach().cpu(),
        "y_hat": y_hat.detach().cpu(),
        "loss": loss.detach().cpu(),
        "final_logits": getattr(model, "last_final_logits", None),
        "visual_logits": getattr(model, "last_visual_logits", None),
        "cross_scale_logits": getattr(model, "last_cross_scale_logits", None),
        "low_scale_logits": getattr(model, "last_low_scale_logits", None),
        "high_scale_logits": getattr(model, "last_high_scale_logits", None),
        "lh_consistency_loss": getattr(model, "last_lh_consistency_loss", None),
        "total_loss": getattr(model, "last_total_loss", None),
    }
    return outputs


def tensor_close(a: torch.Tensor | None, b: torch.Tensor | None, atol: float = 1e-6) -> tuple[str, str]:
    if a is None and b is None:
        return "N/A", "both unavailable"
    if a is None or b is None:
        return "FAIL", f"availability mismatch: a_is_none={a is None}, b_is_none={b is None}"
    a_cpu = a.detach().cpu()
    b_cpu = b.detach().cpu()
    same_shape = tuple(a_cpu.shape) == tuple(b_cpu.shape)
    if not same_shape:
        return "FAIL", f"shape mismatch: {tuple(a_cpu.shape)} vs {tuple(b_cpu.shape)}"
    max_abs_diff = float((a_cpu - b_cpu).abs().max().item()) if a_cpu.numel() > 0 else 0.0
    status = "PASS" if torch.allclose(a_cpu, b_cpu, atol=atol, rtol=0.0) else "FAIL"
    return status, f"shape={tuple(a_cpu.shape)} | max_abs_diff={max_abs_diff:.8g}"


def add_output_compare_rows(
    rows: list[dict[str, str]],
    mode_name: str,
    rce_outputs: dict[str, object],
    deg_outputs: dict[str, object],
) -> None:
    for key in [
        "y_prob",
        "y_hat",
        "loss",
        "final_logits",
        "visual_logits",
        "cross_scale_logits",
        "low_scale_logits",
        "high_scale_logits",
        "lh_consistency_loss",
        "total_loss",
    ]:
        status, details = tensor_close(rce_outputs.get(key), deg_outputs.get(key))
        rows.append(
            {
                "category": f"dynamic_output_{mode_name}",
                "check": key,
                "status": status,
                "details": details,
            }
        )


def add_param_compare_rows(
    rows: list[dict[str, str]],
    rce_model: torch.nn.Module,
    deg_model: torch.nn.Module,
) -> dict[str, object]:
    rce_named = dict(rce_model.named_parameters())
    deg_named = dict(deg_model.named_parameters())
    common_names = sorted(set(rce_named) & set(deg_named))
    extra_deg_names = sorted(set(deg_named) - set(rce_named))
    extra_deg_trainable = [name for name in extra_deg_names if deg_named[name].requires_grad]

    shape_mismatches = []
    value_mismatches = []
    for name in common_names:
        if tuple(rce_named[name].shape) != tuple(deg_named[name].shape):
            shape_mismatches.append(name)
        elif not torch.equal(rce_named[name].detach().cpu(), deg_named[name].detach().cpu()):
            value_mismatches.append(name)

    rows.append(
        {
            "category": "dynamic_native_params",
            "check": "common_parameter_names",
            "status": "PASS",
            "details": f"common_count={len(common_names)}",
        }
    )
    rows.append(
        {
            "category": "dynamic_native_params",
            "check": "common_parameter_shapes",
            "status": "PASS" if not shape_mismatches else "FAIL",
            "details": "none" if not shape_mismatches else ", ".join(shape_mismatches[:20]),
        }
    )
    rows.append(
        {
            "category": "dynamic_native_params",
            "check": "common_parameter_initial_values",
            "status": "PASS" if not value_mismatches else "FAIL",
            "details": "none" if not value_mismatches else ", ".join(value_mismatches[:20]),
        }
    )
    rows.append(
        {
            "category": "dynamic_native_params",
            "check": "deg_extra_trainable_parameters",
            "status": "PASS" if not extra_deg_trainable else "FAIL",
            "details": "none" if not extra_deg_trainable else ", ".join(extra_deg_trainable[:50]),
        }
    )
    return {
        "common_names": common_names,
        "extra_deg_names": extra_deg_names,
        "extra_deg_trainable": extra_deg_trainable,
    }


def add_extra_grad_rows(
    rows: list[dict[str, str]],
    deg_model: torch.nn.Module,
    batch: dict[str, object],
    extra_deg_names: list[str],
) -> None:
    deg_model.zero_grad(set_to_none=True)
    deg_model.train()
    _, _, loss = deg_model(
        batch["x_s"],
        batch["coord_s"],
        batch["x_l"],
        batch["coords_l"],
        batch["label"],
        slide_id=batch["slide_id"],
    )
    loss.backward()

    named = dict(deg_model.named_parameters())
    extra_with_grad = []
    for name in extra_deg_names:
        grad = named[name].grad
        if grad is not None and torch.count_nonzero(grad).item() > 0:
            extra_with_grad.append(name)
    rows.append(
        {
            "category": "dynamic_native_grads",
            "check": "deg_extra_parameter_grad_flow",
            "status": "PASS" if not extra_with_grad else "FAIL",
            "details": "none" if not extra_with_grad else ", ".join(extra_with_grad[:50]),
        }
    )


def write_reports(rows: list[dict[str, str]], summary_lines: list[str]) -> None:
    DOC_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DOC_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "check", "status", "details"])
        writer.writeheader()
        writer.writerows(rows)

    status_counts = pd.DataFrame(rows)["status"].value_counts().to_dict() if rows else {}
    lines = [
        "# Stage51B DEG Skeleton Equivalence Audit",
        "",
        "## Summary",
        "",
    ]
    lines.extend([f"- {line}" for line in summary_lines])
    lines.extend(
        [
            "",
            "## Status Counts",
            "",
            f"- PASS: {status_counts.get('PASS', 0)}",
            f"- FAIL: {status_counts.get('FAIL', 0)}",
            f"- N/A: {status_counts.get('N/A', 0)}",
            "",
            "## Detailed Checks",
            "",
            "| Category | Check | Status | Details |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['check']} | {row['status']} | {row['details']} |"
        )
    DOC_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict[str, str]] = []
    summary_lines: list[str] = []
    try:
        compare_script_flags(rows)
        batch = load_real_batch()

        seed_everything(1)
        rce_model = RCE_MIL_BiomedCLIP(config=default_config(), num_classes=2)
        seed_everything(1)
        deg_model = DEG_MIL_BiomedCLIP(config=default_config(), num_classes=2)

        param_info = add_param_compare_rows(rows, rce_model, deg_model)
        native_rce_outputs = run_model(rce_model, batch)
        native_deg_outputs = run_model(deg_model, batch)
        add_output_compare_rows(rows, "native_init", native_rce_outputs, native_deg_outputs)
        add_extra_grad_rows(rows, deg_model, batch, param_info["extra_deg_names"])

        seed_everything(1)
        deg_shared = DEG_MIL_BiomedCLIP(config=default_config(), num_classes=2)
        load_result = deg_shared.load_state_dict(rce_model.state_dict(), strict=False)
        rows.append(
            {
                "category": "dynamic_shared_weights",
                "check": "load_state_dict_missing_keys",
                "status": "PASS" if not load_result.missing_keys else "FAIL",
                "details": "none" if not load_result.missing_keys else ", ".join(load_result.missing_keys[:50]),
            }
        )
        rows.append(
            {
                "category": "dynamic_shared_weights",
                "check": "load_state_dict_unexpected_keys",
                "status": "PASS" if not load_result.unexpected_keys else "FAIL",
                "details": "none" if not load_result.unexpected_keys else ", ".join(load_result.unexpected_keys[:50]),
            }
        )
        shared_deg_outputs = run_model(deg_shared, batch)
        add_output_compare_rows(rows, "shared_weights", native_rce_outputs, shared_deg_outputs)

        deg_skeleton_passthrough = bool(getattr(deg_model, "deg_skeleton_passthrough", False))
        summary_lines.append(f"Static Stage51 vs Stage23 flag audit completed for {STAGE51_SCRIPT.name} and {STAGE23_SCRIPT.name}.")
        summary_lines.append(f"Real batch source: split=0 train sample | slide_id={batch['slide_id']}.")
        summary_lines.append(f"DEG skeleton passthrough active: {deg_skeleton_passthrough}.")
        summary_lines.append(
            f"Extra DEG trainable parameters in skeleton mode: {len(param_info['extra_deg_trainable'])}."
        )

        fail_count = sum(1 for row in rows if row["status"] == "FAIL")
        summary_lines.append(f"Total failed checks: {fail_count}.")
        write_reports(rows, summary_lines)
        print(f"[Audit] Wrote markdown report to {DOC_MD}")
        print(f"[Audit] Wrote csv report to {DOC_CSV}")
        return 1 if fail_count else 0
    except Exception as exc:
        rows.append(
            {
                "category": "audit_runtime",
                "check": "script_execution",
                "status": "FAIL",
                "details": f"{type(exc).__name__}: {exc}",
            }
        )
        summary_lines.append(f"Audit failed: {type(exc).__name__}: {exc}")
        write_reports(rows, summary_lines)
        print(f"[Audit] Failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
