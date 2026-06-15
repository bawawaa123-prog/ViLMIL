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
OUTPUT_DIR = ROOT / "results_stage43" / "stage43_hcrc_light_smoke_summary"
RUN_PREFIX = "stage43_hcrc_light_smoke"


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

    results_dir = ROOT / "results_stage43"
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(RUN_PREFIX) and "summary" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No Step43 smoke run directory found under {results_dir}")
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
        raise ValueError(f"Unsupported task for Stage43 summary: {task}")

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
        "has_nan_or_inf": bool(re.search(r"(?i)\b(?:nan|inf)\b", text)),
    }


def probe_hcrc_debug(run_dir: Path, settings: dict[str, object]) -> dict[str, object]:
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
            if model.last_hcrc_anchor_valid_mask is not None:
                break

        hcrc_enabled = bool(model.last_hcrc_enabled.item()) if model.last_hcrc_enabled is not None else False
        hcrc_logits_nonzero = False
        if model.last_hcrc_logits is not None:
            hcrc_logits_nonzero = bool(model.last_hcrc_logits.abs().sum().item() > 1e-8)

        child_used_mean = None
        empty_anchor_ratio_mean = None
        if model.last_hcrc_child_used_counts is not None and model.last_hcrc_anchor_valid_mask is not None:
            valid_mask = model.last_hcrc_anchor_valid_mask.bool()
            used_counts = model.last_hcrc_child_used_counts.float()
            if bool(valid_mask.any().item()):
                child_used_mean = float(used_counts[valid_mask].mean().item())
        if model.last_hcrc_empty_anchor_ratio is not None:
            empty_anchor_ratio_mean = float(model.last_hcrc_empty_anchor_ratio.float().mean().item())

        finite_flags = []
        for tensor in [
            model.last_hcrc_logits,
            model.last_hcrc_prompt_weights,
            model.last_hcrc_prompt_evidence,
            model.last_hcrc_region_concept_sim,
            model.last_hcrc_anchor_scores,
            model.last_hcrc_empty_anchor_ratio,
        ]:
            if tensor is not None:
                finite_flags.append(bool(torch.isfinite(tensor).all().item()))
        all_params_finite = all(
            bool(torch.isfinite(param).all().item()) for param in state_dict.values() if torch.is_tensor(param)
        )

        result.update(
            {
                "probe_success": True,
                "slide_id": model.last_slide_id,
                "hcrc_enabled": hcrc_enabled,
                "hcrc_logits_nonzero": hcrc_logits_nonzero,
                "hcrc_alpha": float(model.last_hcrc_alpha.item()) if model.last_hcrc_alpha is not None else None,
                "empty_anchor_ratio_mean": empty_anchor_ratio_mean,
                "child_used_count_mean": child_used_mean,
                "child_counts": model.last_hcrc_child_counts.tolist() if model.last_hcrc_child_counts is not None else None,
                "child_used_counts": model.last_hcrc_child_used_counts.tolist() if model.last_hcrc_child_used_counts is not None else None,
                "anchor_valid_count": int(model.last_hcrc_anchor_valid_mask.sum().item())
                if model.last_hcrc_anchor_valid_mask is not None
                else None,
                "skip_reason": model.last_hcrc_skip_reason,
                "has_nan_or_inf": (not all(finite_flags)) if finite_flags else None,
                "all_checkpoint_params_finite": all_params_finite,
            }
        )
        return result
    except Exception as exc:
        result["failure_reason"] = str(exc)
        return result


def build_report(run_dir: Path, settings: dict[str, object], metrics: dict[str, object], log_info: dict[str, object], probe: dict[str, object]) -> str:
    smoke_completed = all(
        [
            (run_dir / "s_0_checkpoint.pt").is_file(),
            (run_dir / "split_0_results.pkl").is_file(),
            metrics.get("available", False),
        ]
    )
    recommend_step44 = bool(
        smoke_completed
        and probe.get("probe_success")
        and probe.get("hcrc_enabled")
        and probe.get("hcrc_logits_nonzero")
        and (probe.get("empty_anchor_ratio_mean") is None or probe["empty_anchor_ratio_mean"] <= 0.5)
        and not log_info.get("has_traceback", False)
    )

    lines = [
        "# Stage43 HCRC-Light Smoke Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Smoke completed: `{smoke_completed}`",
        f"- Checkpoint generated: `{(run_dir / 's_0_checkpoint.pt').is_file()}`",
        f"- HCRC enabled in config: `{to_bool(settings.get('rce_use_hcrc', False))}`",
        f"- HCRC alpha init: `{settings.get('rce_hcrc_alpha_init')}`",
        f"- Recommended Step42b params: proposal_radius={settings.get('rce_hcrc_proposal_radius')}, nms_radius={settings.get('rce_hcrc_nms_radius')}, bbox_expand={settings.get('rce_hcrc_bbox_expand')}, num_anchors={settings.get('rce_hcrc_num_anchors')}, num_high_children={settings.get('rce_hcrc_num_high_children')}, child_strategy={settings.get('rce_hcrc_child_strategy')}, prompt_scale={settings.get('rce_hcrc_prompt_scale')}",
        "",
        "## Fold0 Metrics",
    ]

    if metrics.get("available"):
        lines.extend(
            [
                f"- test_auc: `{metrics.get('test_auc')}`",
                f"- test_acc: `{metrics.get('test_acc')}`",
                f"- test_f1: `{metrics.get('test_f1')}`",
                f"- val_auc: `{metrics.get('val_auc')}`",
                f"- balanced_acc: `{metrics.get('balanced_acc')}`",
                f"- sensitivity: `{metrics.get('sensitivity')}`",
                f"- specificity: `{metrics.get('specificity')}`",
                f"- pr_auc: `{metrics.get('pr_auc')}`",
            ]
        )
    else:
        lines.append("- Fold metrics unavailable.")

    lines.extend(
        [
            "",
            "## HCRC Probe",
            f"- Probe attempted: `{probe.get('probe_attempted')}`",
            f"- Probe success: `{probe.get('probe_success')}`",
            f"- HCRC enabled at forward: `{probe.get('hcrc_enabled')}`",
            f"- hcrc_logits non-zero: `{probe.get('hcrc_logits_nonzero')}`",
            f"- hcrc alpha (post-sigmoid): `{probe.get('hcrc_alpha')}`",
            f"- empty anchor ratio mean: `{probe.get('empty_anchor_ratio_mean')}`",
            f"- child used count mean: `{probe.get('child_used_count_mean')}`",
            f"- anchor valid count: `{probe.get('anchor_valid_count')}`",
            f"- skip reason: `{probe.get('skip_reason')}`",
            f"- Probe failure reason: `{probe.get('failure_reason')}`",
            "",
            "## Stability",
            f"- Log has Traceback: `{log_info.get('has_traceback')}`",
            f"- Log has NaN/Inf token: `{log_info.get('has_nan_or_inf')}`",
            f"- Probe tensors have NaN/Inf: `{probe.get('has_nan_or_inf')}`",
            f"- Checkpoint params all finite: `{probe.get('all_checkpoint_params_finite')}`",
            "",
            "## Recommendation",
            f"- Enter Step44 HCRC-Light 5-fold: `{recommend_step44}`",
        ]
    )

    if not probe.get("probe_success"):
        lines.append(
            "- Limitation: checkpoint does not store forward debug buffers directly, and the post-run probe could not recover them."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    run_dir = resolve_run_dir()
    settings = read_settings(run_dir)
    metrics = parse_fold_metrics(run_dir)
    log_info = inspect_log(run_dir.parent / "logs" / f"{run_dir.name}.log")
    probe = probe_hcrc_debug(run_dir, settings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(run_dir, settings, metrics, log_info, probe)
    report_path = OUTPUT_DIR / "stage43_hcrc_smoke_report.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "run_dir": str(run_dir),
        "settings": settings,
        "metrics": metrics,
        "log_info": log_info,
        "probe": probe,
    }
    manifest_path = OUTPUT_DIR / "stage43_hcrc_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Done] Report: {report_path}")
    print(f"[Done] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
