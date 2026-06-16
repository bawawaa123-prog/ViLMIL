from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_DEG_MIL_BiomedCLIP import DEG_MIL_BiomedCLIP


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results_stage46" / "stage46_prarc_gate_smoke_summary"
RUN_PREFIX = "stage46_prarc_gate_smoke"


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_run_dir() -> Path:
    env_run_dir = os.environ.get("RUN_DIR", "").strip()
    if env_run_dir:
        path = Path(env_run_dir)
        return path if path.is_absolute() else ROOT / path

    results_dir = ROOT / "results_stage46"
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(RUN_PREFIX) and "summary" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No Step46 smoke run directory found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_settings(run_dir: Path) -> dict[str, object]:
    experiment_files = sorted(run_dir.glob("experiment_*.txt"))
    if not experiment_files:
        raise FileNotFoundError(f"No experiment_*.txt found in {run_dir}")
    return ast.literal_eval(experiment_files[0].read_text(encoding="utf-8"))


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
        raise ValueError(f"Unsupported task for Stage46 summary: {task}")

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


def build_model(settings: dict[str, object], num_classes: int) -> DEG_MIL_BiomedCLIP:
    config = SimpleNamespace(
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
        rce_use_visual_evidence_gate=to_bool(settings.get("rce_use_visual_evidence_gate", False)),
        rce_visual_gate_init=float(settings.get("rce_visual_gate_init", 1.0)),
        rce_use_prarc_gate=to_bool(settings.get("rce_use_prarc_gate", False)),
        rce_prarc_gate_hidden_dim=int(settings.get("rce_prarc_gate_hidden_dim", 16)),
        rce_prarc_gate_init=float(settings.get("rce_prarc_gate_init", 0.8)),
        rce_prarc_gate_dropout=float(settings.get("rce_prarc_gate_dropout", 0.0)),
        rce_prarc_gate_feature_set=str(settings.get("rce_prarc_gate_feature_set", "v1")),
        rce_prarc_detach_features=to_bool(settings.get("rce_prarc_detach_features", False)),
        rce_prarc_include_optional_features=to_bool(settings.get("rce_prarc_include_optional_features", False)),
        rce_prarc_feature_clip=float(settings.get("rce_prarc_feature_clip", 10.0)),
        rce_prarc_export_debug=to_bool(settings.get("rce_prarc_export_debug", False)),
        rce_use_low_high_consistency_loss=to_bool(settings.get("rce_use_low_high_consistency_loss", False)),
        rce_lh_consistency_lambda=float(settings.get("rce_lh_consistency_lambda", 0.0)),
        rce_lh_consistency_margin=float(settings.get("rce_lh_consistency_margin", 0.0)),
        rce_use_cross_scale_graph=to_bool(settings.get("rce_use_cross_scale_graph", False)),
        rce_cross_scale_graph_init=float(settings.get("rce_cross_scale_graph_init", 0.05)),
        rce_cross_scale_graph_norm=str(settings.get("rce_cross_scale_graph_norm", "sqrt")),
        rce_use_hcrc=to_bool(settings.get("rce_use_hcrc", False)),
        rce_hcrc_alpha_init=float(settings.get("rce_hcrc_alpha_init", 0.05)),
        rce_hcrc_num_anchors=int(settings.get("rce_hcrc_num_anchors", 16)),
        rce_hcrc_num_high_children=int(settings.get("rce_hcrc_num_high_children", 16)),
        rce_hcrc_proposal_radius=float(settings.get("rce_hcrc_proposal_radius", 4096.0)),
        rce_hcrc_nms_radius=float(settings.get("rce_hcrc_nms_radius", 512.0)),
        rce_hcrc_bbox_expand=float(settings.get("rce_hcrc_bbox_expand", 8.0)),
        rce_hcrc_coord_mode=str(settings.get("rce_hcrc_coord_mode", "top_left")),
        rce_hcrc_scale_ratio=float(settings.get("rce_hcrc_scale_ratio", 1.0)),
        rce_hcrc_child_strategy=str(settings.get("rce_hcrc_child_strategy", "bbox_containment")),
        rce_hcrc_candidate_top_l=int(settings.get("rce_hcrc_candidate_top_l", 64)),
        rce_hcrc_top_g_concepts=int(settings.get("rce_hcrc_top_g_concepts", 8)),
        rce_hcrc_per_concept_top_m=int(settings.get("rce_hcrc_per_concept_top_m", 4)),
        rce_hcrc_prompt_topk=int(settings.get("rce_hcrc_prompt_topk", 3)),
        rce_hcrc_margin_weight=float(settings.get("rce_hcrc_margin_weight", 0.5)),
        rce_hcrc_prompt_scale=str(settings.get("rce_hcrc_prompt_scale", "high")),
        rce_hcrc_min_child_count=int(settings.get("rce_hcrc_min_child_count", 1)),
        rce_hcrc_export_debug=to_bool(settings.get("rce_hcrc_export_debug", False)),
        deg_use_region_graph=to_bool(settings.get("deg_use_region_graph", False)),
        deg_region_graph_k=int(settings.get("deg_region_graph_k", 4)),
        deg_region_graph_alpha=float(settings.get("deg_region_graph_alpha", 0.1)),
        deg_use_concept_graph=to_bool(settings.get("deg_use_concept_graph", False)),
        deg_concept_graph_topk=int(settings.get("deg_concept_graph_topk", 4)),
        deg_concept_graph_alpha=float(settings.get("deg_concept_graph_alpha", 0.05)),
        scale_mode=str(settings.get("scale_mode", "dual")),
        finetune_text_encoder=False,
    )
    return DEG_MIL_BiomedCLIP(config=config, num_classes=num_classes)


def parse_fold_metrics(run_dir: Path) -> dict[str, object]:
    fold_summary_path = run_dir / "fold_summary.csv"
    if not fold_summary_path.is_file():
        return {"available": False, "path": str(fold_summary_path)}
    df = pd.read_csv(fold_summary_path)
    if df.empty:
        return {"available": False, "path": str(fold_summary_path)}
    row = df.iloc[0].to_dict()
    row["available"] = True
    row["path"] = str(fold_summary_path)
    return row


def inspect_log(log_path: Path) -> dict[str, object]:
    if not log_path.is_file():
        return {"available": False, "path": str(log_path)}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "available": True,
        "path": str(log_path),
        "has_traceback": "Traceback" in text,
        "has_nan_or_inf": bool(re.search(r"(?i)\\b(?:nan|inf)\\b", text)),
    }


def probe_prarc_debug(run_dir: Path, settings: dict[str, object]) -> dict[str, object]:
    checkpoint_path = run_dir / "s_0_checkpoint.pt"
    result = {
        "probe_attempted": False,
        "probe_success": False,
        "checkpoint_path": str(checkpoint_path),
        "failure_reason": None,
    }
    if not checkpoint_path.is_file():
        result["failure_reason"] = "checkpoint_missing"
        return result

    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        dataset = build_dataset(settings)
        split_dir = Path(str(settings["split_dir"]))
        if not split_dir.is_absolute():
            split_dir = ROOT / split_dir
        fold = int(settings.get("k_start", 0))
        _, _, test_split = dataset.return_splits(
            from_id=False,
            csv_path=str(split_dir / f"splits_{fold}.csv"),
        )

        model = build_model(settings, num_classes=int(settings["n_classes"]))
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.relocate()
        device = next(model.parameters()).device
        model.eval()
        result["probe_attempted"] = True

        max_probe = min(len(test_split), 8)
        collected_gates = []
        collected_feature_names = None
        visual_adjusted_flags = []
        for idx in range(max_probe):
            features_s, coords_s, features_l, coords_l, label, slide_id = test_split[idx]
            with torch.no_grad():
                _ = model(
                    features_s.to(device),
                    coords_s.to(device),
                    features_l.to(device),
                    coords_l.to(device),
                    torch.tensor([label], device=device, dtype=torch.long),
                    slide_id=slide_id,
                )
            if model.last_prarc_gate is not None:
                collected_gates.append(model.last_prarc_gate.float().reshape(-1))
                if collected_feature_names is None:
                    collected_feature_names = model.last_prarc_gate_feature_names
                if (
                    model.last_prarc_visual_residual_contribution is not None
                    and model.last_prarc_visual_gated_contribution is not None
                ):
                    visual_adjusted_flags.append(
                        not torch.allclose(
                            model.last_prarc_visual_residual_contribution,
                            model.last_prarc_visual_gated_contribution,
                        )
                    )

        gate_tensor = torch.cat(collected_gates, dim=0) if collected_gates else model.last_prarc_gate
        feature_tensor = model.last_prarc_gate_features
        gate_nonconstant = None
        if gate_tensor is not None:
            gate_nonconstant = bool((gate_tensor.max() - gate_tensor.min()).abs().item() > 1e-6)

        visual_adjusted = None
        if visual_adjusted_flags:
            visual_adjusted = bool(any(visual_adjusted_flags))

        result.update(
            {
                "probe_success": True,
                "slide_id": model.last_slide_id,
                "prarc_enabled": bool(model.last_prarc_enabled.item()) if model.last_prarc_enabled is not None else False,
                "prarc_gate_init": float(settings.get("rce_prarc_gate_init", 0.8)),
                "gate_mean": float(gate_tensor.mean().item()) if gate_tensor is not None else model.last_prarc_gate_mean,
                "gate_min": float(gate_tensor.min().item()) if gate_tensor is not None else model.last_prarc_gate_min,
                "gate_max": float(gate_tensor.max().item()) if gate_tensor is not None else model.last_prarc_gate_max,
                "gate_nonconstant": gate_nonconstant,
                "gate_feature_names": collected_feature_names or model.last_prarc_gate_feature_names,
                "gate_features_present": feature_tensor is not None,
                "gate_feature_shape": list(feature_tensor.shape) if feature_tensor is not None else None,
                "visual_residual_adjusted": visual_adjusted,
                "skip_reason": model.last_prarc_skip_reason,
                "last_visual_residual_alpha": float(model.last_visual_residual_alpha.item()) if model.last_visual_residual_alpha is not None else None,
                "checkpoint_params_finite": all(
                    bool(torch.isfinite(param).all().item()) for param in state_dict.values() if torch.is_tensor(param)
                ),
            }
        )
        return result
    except Exception as exc:
        result["failure_reason"] = str(exc)
        return result


def build_report(
    run_dir: Path,
    settings: dict[str, object],
    metrics: dict[str, object],
    log_info: dict[str, object],
    probe: dict[str, object],
) -> str:
    smoke_completed = all(
        [
            (run_dir / "s_0_checkpoint.pt").is_file(),
            (run_dir / "split_0_results.pkl").is_file(),
            metrics.get("available", False),
        ]
    )
    recommend_step47 = bool(
        smoke_completed
        and probe.get("probe_success")
        and probe.get("prarc_enabled")
        and not log_info.get("has_traceback", False)
        and not log_info.get("has_nan_or_inf", False)
        and probe.get("gate_features_present")
    )

    lines = [
        "# Stage46 PRARC Gate Smoke Report",
        "",
        "## Smoke Status",
        f"- run_dir: `{run_dir}`",
        f"- smoke_completed: `{smoke_completed}`",
        f"- checkpoint_exists: `{(run_dir / 's_0_checkpoint.pt').is_file()}`",
        f"- split_0_results_exists: `{(run_dir / 'split_0_results.pkl').is_file()}`",
        f"- fold_summary_exists: `{metrics.get('available', False)}`",
        "",
        "## Fold0 / 1 Epoch Metrics",
        f"- test_auc: `{metrics.get('test_auc', 'N/A')}`",
        f"- test_acc: `{metrics.get('test_acc', 'N/A')}`",
        f"- test_f1: `{metrics.get('test_f1', 'N/A')}`",
        f"- balanced_acc: `{metrics.get('balanced_acc', 'N/A')}`",
        f"- sensitivity: `{metrics.get('sensitivity', 'N/A')}`",
        f"- specificity: `{metrics.get('specificity', 'N/A')}`",
        f"- pr_auc: `{metrics.get('pr_auc', 'N/A')}`",
        "",
        "## PRARC Debug Probe",
        f"- probe_success: `{probe.get('probe_success')}`",
        f"- prarc_enabled: `{probe.get('prarc_enabled')}`",
        f"- prarc_gate_init: `{probe.get('prarc_gate_init')}`",
        f"- gate_mean: `{probe.get('gate_mean')}`",
        f"- gate_min: `{probe.get('gate_min')}`",
        f"- gate_max: `{probe.get('gate_max')}`",
        f"- gate_nonconstant: `{probe.get('gate_nonconstant')}`",
        f"- gate_features_present: `{probe.get('gate_features_present')}`",
        f"- gate_feature_names: `{probe.get('gate_feature_names')}`",
        f"- visual_residual_adjusted: `{probe.get('visual_residual_adjusted')}`",
        f"- skip_reason: `{probe.get('skip_reason')}`",
        "",
        "## Runtime Safety",
        f"- log_has_traceback: `{log_info.get('has_traceback')}`",
        f"- log_has_nan_or_inf: `{log_info.get('has_nan_or_inf')}`",
        f"- checkpoint_params_finite: `{probe.get('checkpoint_params_finite')}`",
        "",
        "## Recommendation",
        f"- recommend_enter_step47_prarc_5fold: `{recommend_step47}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = resolve_run_dir()
    settings = read_settings(run_dir)
    metrics = parse_fold_metrics(run_dir)
    log_path = run_dir.parent / "logs" / f"{run_dir.name}.log"
    log_info = inspect_log(log_path)
    probe = probe_prarc_debug(run_dir, settings)
    report = build_report(run_dir, settings, metrics, log_info, probe)

    manifest = {
        "step": 46,
        "name": "PRARC Gate Smoke Summary",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "settings": settings,
        "metrics": metrics,
        "log_info": log_info,
        "probe": probe,
        "output_paths": {
            "report_md": str(output_dir / "stage46_prarc_smoke_report.md"),
            "manifest_json": str(output_dir / "stage46_prarc_smoke_manifest.json"),
        },
    }

    (output_dir / "stage46_prarc_smoke_report.md").write_text(report, encoding="utf-8")
    (output_dir / "stage46_prarc_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Stage46] Wrote smoke summary to: {output_dir}")


if __name__ == "__main__":
    main()
