from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_DEG_MIL_BiomedCLIP import DEG_MIL_BiomedCLIP


DEFAULT_VARIANTS = [
    "v2_gain2_g08",
    "v2_gain4_g08",
    "v2_gain2_g05",
    "v2_opt_gain2_g08",
    "v2_confprior_g08",
    "v2_varreg_g08",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step48 PRARC-v2 smoke summary.")
    parser.add_argument("--results_root", default="results_stage48")
    parser.add_argument("--baseline_diagnostics", default="results_stage47/stage47_prarc_gate_diagnostics/stage47_prarc_gate_distribution_summary.csv")
    parser.add_argument("--output_dir", default="results_stage48/stage48_prarc_v2_smoke_summary")
    parser.add_argument("--variants", default="v2_gain2_g08")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_root_dir", default="/xiangmu/data/VILMIL")
    parser.add_argument("--data_folder_s", default="features_biomedclip_5x")
    parser.add_argument("--data_folder_l", default="features_biomedclip_20x")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_slides", type=int, default=0)
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def parse_variants(raw: str) -> list[str]:
    items = [item.strip() for item in str(raw).split(",")]
    return [item for item in items if item]


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def find_run_dir(results_root: Path, variant: str, seed: int) -> Path | None:
    exact = results_root / f"stage48_{variant}_s{seed}"
    if exact.is_dir():
        return exact
    candidates = [path for path in results_root.glob(f"stage48_{variant}_s{seed}*") if path.is_dir()]
    if candidates:
        return sorted(candidates)[-1]
    return None


def read_experiment_settings(run_dir: Path) -> dict[str, object]:
    files = sorted(run_dir.glob("experiment_*.txt"))
    if not files:
        return {}
    try:
        return ast.literal_eval(files[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


def inspect_log(log_path: Path) -> dict[str, object]:
    if not log_path.is_file():
        return {"log_exists": False, "has_traceback": None, "has_nan_or_inf": None, "log_path": str(log_path)}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "log_exists": True,
        "has_traceback": "Traceback" in text,
        "has_nan_or_inf": bool(re.search(r"(?i)\b(?:nan|inf)\b", text)),
        "log_path": str(log_path),
    }


def build_dataset(settings: dict[str, object], fallback_args: argparse.Namespace) -> Generic_MIL_Dataset:
    task = str(settings.get("task") or "task_adenocarcinoma")
    data_root_dir = Path(str(settings.get("data_root_dir") or fallback_args.data_root_dir))
    data_folder_s = str(settings.get("data_folder_s") or fallback_args.data_folder_s)
    data_folder_l = str(settings.get("data_folder_l") or fallback_args.data_folder_l)

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
        raise ValueError(f"Unsupported task for Step48 summary: {task}")

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
        rce_prarc_gate_version=str(settings.get("rce_prarc_gate_version", "v1")),
        rce_prarc_gate_hidden_dim=int(settings.get("rce_prarc_gate_hidden_dim", 16)),
        rce_prarc_gate_init=float(settings.get("rce_prarc_gate_init", 0.8)),
        rce_prarc_gate_dropout=float(settings.get("rce_prarc_gate_dropout", 0.0)),
        rce_prarc_gate_gain=float(settings.get("rce_prarc_gate_gain", 1.0)),
        rce_prarc_gate_last_weight_init=float(settings.get("rce_prarc_gate_last_weight_init", 0.01)),
        rce_prarc_gate_feature_set=str(settings.get("rce_prarc_gate_feature_set", "v1")),
        rce_prarc_detach_features=to_bool(settings.get("rce_prarc_detach_features", False)),
        rce_prarc_include_optional_features=to_bool(settings.get("rce_prarc_include_optional_features", False)),
        rce_prarc_feature_clip=float(settings.get("rce_prarc_feature_clip", 10.0)),
        rce_prarc_export_debug=to_bool(settings.get("rce_prarc_export_debug", False)),
        rce_prarc_use_conflict_prior=to_bool(settings.get("rce_prarc_use_conflict_prior", False)),
        rce_prarc_conflict_prior_strength=float(settings.get("rce_prarc_conflict_prior_strength", 0.2)),
        rce_prarc_use_gate_entropy_reg=to_bool(settings.get("rce_prarc_use_gate_entropy_reg", False)),
        rce_prarc_gate_entropy_lambda=float(settings.get("rce_prarc_gate_entropy_lambda", 0.0)),
        rce_prarc_use_gate_variance_reg=to_bool(settings.get("rce_prarc_use_gate_variance_reg", False)),
        rce_prarc_gate_variance_lambda=float(settings.get("rce_prarc_gate_variance_lambda", 0.0)),
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


def get_split_dataset(dataset: Generic_MIL_Dataset, split_dir: Path, fold: int, split_name: str):
    train_split, val_split, test_split = dataset.return_splits(
        from_id=False,
        csv_path=str(split_dir / f"splits_{fold}.csv"),
    )
    mapping = {"train": train_split, "val": val_split, "test": test_split}
    return mapping[split_name]


def load_baseline_step47_distribution(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None
    subset = df[df["variant"].astype(str) == "prarc_v1_g08"].copy()
    if subset.empty:
        return None, None
    return (
        float(pd.to_numeric(subset["gate_std"], errors="coerce").mean()),
        float(pd.to_numeric(subset["gate_range"], errors="coerce").mean()),
    )


def probe_run(
    variant: str,
    run_dir: Path,
    settings: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoint_path = run_dir / "s_0_checkpoint.pt"
    if not checkpoint_path.is_file():
        return [], {"status": "checkpoint_missing", "checkpoint_path": str(checkpoint_path)}

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dataset = build_dataset(settings, args)
    split_dir = Path(str(settings.get("split_dir") or "splits/adenocarcinoma/task_adenocarcinoma_strictcv_100"))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    split_dataset = get_split_dataset(dataset, split_dir, 0, args.split)

    model = build_model(settings, num_classes=int(settings.get("n_classes", 2)))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.relocate()
    device = next(model.parameters()).device
    model.eval()

    rows: list[dict[str, object]] = []
    max_slides = len(split_dataset)
    if args.max_slides > 0:
        max_slides = min(max_slides, args.max_slides)

    for idx in range(max_slides):
        features_s, coords_s, features_l, coords_l, label, slide_id = split_dataset[idx]
        label_tensor = torch.tensor([int(label)], device=device, dtype=torch.long)
        with torch.no_grad():
            y_prob, y_hat, _ = model(
                features_s.to(device),
                coords_s.to(device),
                features_l.to(device),
                coords_l.to(device),
                label_tensor,
                slide_id=slide_id,
            )
        feature_dict = model.last_prarc_gate_feature_dict or {}
        row = {
            "variant": variant,
            "slide_id": str(slide_id),
            "label": int(label),
            "pred": int(y_hat.detach().cpu().reshape(-1)[0].item()),
            "prarc_enabled": bool(model.last_prarc_enabled.item()) if model.last_prarc_enabled is not None else False,
            "gate_version": model.last_prarc_gate_version,
            "gate_gain": model.last_prarc_gate_gain,
            "prarc_gate": safe_float(model.last_prarc_gate),
            "gate_logit": safe_float(model.last_prarc_gate_logits),
            "gate_entropy": safe_float(model.last_prarc_gate_entropy),
            "gate_variance": safe_float(model.last_prarc_gate_variance),
            "gate_reg_loss": safe_float(model.last_prarc_gate_reg_loss),
            "conflict_prior": safe_float(model.last_prarc_conflict_prior),
            "visual_concept_conflict": safe_float(feature_dict.get("visual_concept_conflict")),
            "concept_pred_margin_abs": safe_float(feature_dict.get("concept_pred_margin_abs")),
            "prediction_confidence_margin": safe_float(feature_dict.get("prediction_confidence_margin")),
            "dominant_source_ratio": safe_float(feature_dict.get("dominant_source_ratio")),
        }
        for name, value in feature_dict.items():
            row[name] = safe_float(value)
        rows.append(row)

    return rows, {"status": "ok", "checkpoint_path": str(checkpoint_path), "slides": len(rows)}


def summarize_distribution(variant: str, gate_df: pd.DataFrame) -> dict[str, object]:
    gate_series = pd.to_numeric(gate_df["prarc_gate"], errors="coerce").dropna()
    if gate_series.empty:
        return {"variant": variant}
    out = {
        "variant": variant,
        "n": int(len(gate_series)),
        "gate_mean": float(gate_series.mean()),
        "gate_std": float(gate_series.std(ddof=0)),
        "gate_min": float(gate_series.min()),
        "gate_max": float(gate_series.max()),
        "gate_range": float(gate_series.max() - gate_series.min()),
    }
    conflict_series = gate_df[gate_df["visual_concept_conflict"] == 1]["prarc_gate"]
    non_conflict_series = gate_df[gate_df["visual_concept_conflict"] == 0]["prarc_gate"]
    if len(conflict_series) > 0:
        out["conflict_gate_mean"] = float(pd.to_numeric(conflict_series, errors="coerce").mean())
    if len(non_conflict_series) > 0:
        out["non_conflict_gate_mean"] = float(pd.to_numeric(non_conflict_series, errors="coerce").mean())
    if len(conflict_series) > 0 and len(non_conflict_series) > 0:
        out["conflict_minus_nonconflict"] = float(
            pd.to_numeric(conflict_series, errors="coerce").mean()
            - pd.to_numeric(non_conflict_series, errors="coerce").mean()
        )
    return out


def main() -> int:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    output_dir = resolve_path(args.output_dir)
    baseline_diag_path = resolve_path(args.baseline_diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants) or ["v2_gain2_g08"]
    baseline_gate_std, baseline_gate_range = load_baseline_step47_distribution(baseline_diag_path)

    report_lines = ["# Step48 PRARC-v2 Smoke Report", ""]
    manifest: dict[str, object] = {
        "step": "Step48 PRARC-v2 Gate Dynamics Repair Smoke",
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "variants": variants,
        "baseline_step47_gate_std": baseline_gate_std,
        "baseline_step47_gate_range": baseline_gate_range,
        "variant_results": [],
    }

    probe_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for variant in variants:
        run_dir = find_run_dir(results_root, variant, args.seed)
        variant_result: dict[str, object] = {"variant": variant, "run_dir": str(run_dir) if run_dir else None}
        if run_dir is None:
            variant_result["status"] = "missing"
            manifest["variant_results"].append(variant_result)
            report_lines.append(f"- {variant}: run directory missing")
            continue

        settings = read_experiment_settings(run_dir)
        fold_summary_path = run_dir / "fold_summary.csv"
        metrics = {}
        if fold_summary_path.is_file():
            try:
                df_metrics = pd.read_csv(fold_summary_path)
                if not df_metrics.empty:
                    metrics = df_metrics.iloc[0].to_dict()
            except Exception:
                metrics = {}

        log_info = inspect_log(results_root / "logs" / f"stage48_{variant}_s{args.seed}.log")
        checkpoint_exists = (run_dir / "s_0_checkpoint.pt").is_file()
        rows, probe_status = probe_run(variant, run_dir, settings, args)
        probe_rows.extend(rows)

        gate_df = pd.DataFrame([row for row in rows if row["variant"] == variant])
        dist_row = summarize_distribution(variant, gate_df) if not gate_df.empty else {"variant": variant}
        distribution_rows.append(dist_row)

        gate_std = dist_row.get("gate_std")
        gate_range = dist_row.get("gate_range")
        variance_better = False
        if baseline_gate_std is not None and gate_std is not None:
            variance_better = gate_std > baseline_gate_std
        range_better = False
        if baseline_gate_range is not None and gate_range is not None:
            range_better = gate_range > baseline_gate_range
        recommend_step49 = bool(
            probe_status.get("status") == "ok"
            and checkpoint_exists
            and not log_info.get("has_traceback")
            and not log_info.get("has_nan_or_inf")
            and gate_std is not None and gate_std > 1e-3
            and gate_range is not None and gate_range > max(1e-2, baseline_gate_range or 0.0)
        )

        variant_result.update(
            {
                "status": probe_status.get("status"),
                "checkpoint_exists": checkpoint_exists,
                "fold0_metrics": metrics,
                "log_info": log_info,
                "probe_status": probe_status,
                "gate_distribution": dist_row,
                "variance_better_than_step47_v1_g08": variance_better,
                "range_better_than_step47_v1_g08": range_better,
                "recommend_step49": recommend_step49,
            }
        )
        manifest["variant_results"].append(variant_result)

        report_lines.extend(
            [
                f"## {variant}",
                f"- smoke_completed: `{probe_status.get('status') == 'ok'}`",
                f"- checkpoint_exists: `{checkpoint_exists}`",
                f"- test_auc: `{metrics.get('test_auc')}`",
                f"- test_acc: `{metrics.get('test_acc')}`",
                f"- test_f1: `{metrics.get('test_f1')}`",
                f"- balanced_acc: `{metrics.get('balanced_acc')}`",
                f"- pr_auc: `{metrics.get('pr_auc')}`",
                f"- prarc_enabled: `{bool(rows and rows[0].get('prarc_enabled'))}`",
                f"- gate_mean: `{dist_row.get('gate_mean')}`",
                f"- gate_std: `{dist_row.get('gate_std')}`",
                f"- gate_min: `{dist_row.get('gate_min')}`",
                f"- gate_max: `{dist_row.get('gate_max')}`",
                f"- gate_range: `{dist_row.get('gate_range')}`",
                f"- gate_std_gt_step47_v1_g08: `{variance_better}`",
                f"- gate_range_gt_step47_v1_g08: `{range_better}`",
                f"- conflict_gate_mean: `{dist_row.get('conflict_gate_mean')}`",
                f"- non_conflict_gate_mean: `{dist_row.get('non_conflict_gate_mean')}`",
                f"- conflict_minus_nonconflict: `{dist_row.get('conflict_minus_nonconflict')}`",
                f"- log_has_traceback: `{log_info.get('has_traceback')}`",
                f"- log_has_nan_or_inf: `{log_info.get('has_nan_or_inf')}`",
                f"- recommend_enter_step49: `{recommend_step49}`",
                "",
            ]
        )

    probe_csv = output_dir / "stage48_prarc_v2_gate_probe.csv"
    dist_csv = output_dir / "stage48_prarc_v2_gate_distribution.csv"
    report_md = output_dir / "stage48_prarc_v2_smoke_report.md"
    manifest_json = output_dir / "stage48_prarc_v2_smoke_manifest.json"

    pd.DataFrame(probe_rows).to_csv(probe_csv, index=False)
    pd.DataFrame(distribution_rows).to_csv(dist_csv, index=False)
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[Done] Wrote Step48 smoke summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
