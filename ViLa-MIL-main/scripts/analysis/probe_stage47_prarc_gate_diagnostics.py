from __future__ import annotations

import argparse
import ast
import json
import math
import os
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


DEFAULT_VARIANTS = ["prarc_v1_g05", "prarc_v1_g08", "prarc_v1_g10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Step47 PRARC gate diagnostics.")
    parser.add_argument("--results_root", default="results_stage47")
    parser.add_argument("--variants", default="prarc_v1_g05,prarc_v1_g08,prarc_v1_g10")
    parser.add_argument("--data_root_dir", default="/xiangmu/data/VILMIL")
    parser.add_argument("--data_folder_s", default="features_biomedclip_5x")
    parser.add_argument("--data_folder_l", default="features_biomedclip_20x")
    parser.add_argument("--split_dir", default="splits/adenocarcinoma/task_adenocarcinoma_strictcv_100")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_slides_per_fold", type=int, default=0)
    parser.add_argument("--output_dir", default="results_stage47/stage47_prarc_gate_diagnostics")
    parser.add_argument("--seed", type=int, default=1)
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


def find_run_dir(results_root: Path, variant: str, seed: int) -> Path | None:
    exact = results_root / f"stage47_{variant}_s{seed}"
    if exact.is_dir():
        return exact
    candidates = [
        path
        for path in results_root.glob(f"stage47_{variant}_s{seed}*")
        if path.is_dir()
    ]
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
        raise ValueError(f"Unsupported task for Step47 diagnostics: {task}")

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


def normalize_slide_id(slide_id) -> str:
    if isinstance(slide_id, str):
        return slide_id
    if torch.is_tensor(slide_id):
        if slide_id.numel() == 1:
            return str(slide_id.detach().cpu().item())
        return str(slide_id.detach().cpu().tolist())
    if isinstance(slide_id, (list, tuple)):
        return "|".join(normalize_slide_id(item) for item in slide_id)
    return str(slide_id)


def expected_fold_indices(settings: dict[str, object]) -> list[int]:
    k = int(settings.get("num_splits", settings.get("k", 5)) or 5)
    k_start = int(settings.get("k_start", 0) or 0)
    k_end = int(settings.get("k_end", k - 1) or (k - 1))
    if k_start == -1:
        k_start = 0
    if k_end == -1:
        k_end = k - 1
    return list(range(k_start, k_end + 1))


def get_split_dataset(dataset: Generic_MIL_Dataset, split_dir: Path, fold: int, split_name: str):
    train_split, val_split, test_split = dataset.return_splits(
        from_id=False,
        csv_path=str(split_dir / f"splits_{fold}.csv"),
    )
    mapping = {"train": train_split, "val": val_split, "test": test_split}
    return mapping[split_name]


def collect_condition_rows(slide_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if slide_df.empty:
        return rows
    for condition_name in ["correct", "visual_concept_conflict", "low_high_sign_agreement"]:
        if condition_name not in slide_df.columns:
            continue
        grouped = slide_df.groupby(condition_name, dropna=False)
        for condition_value, group in grouped:
            gate_series = pd.to_numeric(group["prarc_gate"], errors="coerce").dropna()
            rows.append(
                {
                    "variant": group["variant"].iloc[0],
                    "fold": group["fold"].iloc[0],
                    "condition_name": condition_name,
                    "condition_value": condition_value,
                    "n": int(len(group)),
                    "gate_mean": float(gate_series.mean()) if not gate_series.empty else None,
                    "gate_std": float(gate_series.std(ddof=0)) if len(gate_series) > 0 else None,
                }
            )
    return rows


def collect_correlation_rows(slide_df: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if slide_df.empty or "prarc_gate" not in slide_df.columns:
        return rows
    gate_series = pd.to_numeric(slide_df["prarc_gate"], errors="coerce")
    for feature_name in feature_columns:
        feature_series = pd.to_numeric(slide_df[feature_name], errors="coerce")
        valid = pd.DataFrame({"gate": gate_series, "feature": feature_series}).dropna()
        corr = None
        if len(valid) >= 3 and valid["gate"].nunique() > 1 and valid["feature"].nunique() > 1:
            corr = float(valid["gate"].corr(valid["feature"]))
        rows.append(
            {
                "variant": slide_df["variant"].iloc[0],
                "fold": slide_df["fold"].iloc[0],
                "feature_name": feature_name,
                "correlation": corr,
                "n": int(len(valid)),
            }
        )
    return rows


def collect_distribution_row(slide_df: pd.DataFrame) -> dict[str, object]:
    gate_series = pd.to_numeric(slide_df["prarc_gate"], errors="coerce").dropna()
    if gate_series.empty:
        return {
            "variant": slide_df["variant"].iloc[0],
            "fold": slide_df["fold"].iloc[0],
            "n": int(len(slide_df)),
        }
    return {
        "variant": slide_df["variant"].iloc[0],
        "fold": slide_df["fold"].iloc[0],
        "n": int(len(gate_series)),
        "gate_mean": float(gate_series.mean()),
        "gate_std": float(gate_series.std(ddof=0)),
        "gate_min": float(gate_series.min()),
        "gate_max": float(gate_series.max()),
        "gate_p10": float(gate_series.quantile(0.10)),
        "gate_p25": float(gate_series.quantile(0.25)),
        "gate_p50": float(gate_series.quantile(0.50)),
        "gate_p75": float(gate_series.quantile(0.75)),
        "gate_p90": float(gate_series.quantile(0.90)),
        "gate_range": float(gate_series.max() - gate_series.min()),
        "fraction_gate_below_0.5": float((gate_series < 0.5).mean()),
        "fraction_gate_above_0.9": float((gate_series > 0.9).mean()),
    }


def probe_variant_fold(
    variant: str,
    fold: int,
    run_dir: Path,
    settings: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoint_path = run_dir / f"s_{fold}_checkpoint.pt"
    if not checkpoint_path.is_file():
        return [], {
            "variant": variant,
            "fold": fold,
            "status": "checkpoint_missing",
            "warning": f"Missing checkpoint: {checkpoint_path}",
        }

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dataset = build_dataset(settings, args)
    split_dir = Path(str(settings.get("split_dir") or args.split_dir))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    split_dataset = get_split_dataset(dataset, split_dir, fold, args.split)
    if split_dataset is None:
        return [], {
            "variant": variant,
            "fold": fold,
            "status": "split_missing",
            "warning": f"Split `{args.split}` missing for fold {fold}",
        }

    model = build_model(settings, num_classes=int(settings.get("n_classes", 2)))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.relocate()
    device = next(model.parameters()).device
    model.eval()

    rows: list[dict[str, object]] = []
    num_slides = len(split_dataset)
    if args.max_slides_per_fold > 0:
        num_slides = min(num_slides, args.max_slides_per_fold)

    for idx in range(num_slides):
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

        pred = int(y_hat.detach().cpu().reshape(-1)[0].item())
        feature_dict = model.last_prarc_gate_feature_dict or {}
        feature_names = list(model.last_prarc_gate_feature_names or [])
        feature_row = {name: safe_float(feature_dict.get(name)) for name in feature_names}
        gate_value = safe_float(model.last_prarc_gate)
        row = {
            "variant": variant,
            "fold": fold,
            "slide_id": normalize_slide_id(slide_id),
            "label": int(label),
            "pred": pred,
            "correct": int(pred == int(label)),
            "prarc_gate": gate_value,
            "prarc_gate_features": json.dumps(feature_row, ensure_ascii=False, sort_keys=True),
            "prarc_skip_reason": model.last_prarc_skip_reason,
            "visual_concept_conflict": feature_row.get("visual_concept_conflict"),
            "low_high_sign_agreement": feature_row.get("low_high_sign_agreement"),
            "concept_pred_margin_abs": feature_row.get("concept_pred_margin_abs"),
            "prediction_confidence_margin": feature_row.get("prediction_confidence_margin"),
            "dominant_source_ratio": feature_row.get("dominant_source_ratio"),
        }
        for name, value in feature_row.items():
            row[name] = value
        rows.append(row)

    return rows, {
        "variant": variant,
        "fold": fold,
        "status": "ok",
        "slides_probed": len(rows),
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> int:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants) or list(DEFAULT_VARIANTS)
    slide_rows: list[dict[str, object]] = []
    fold_status_rows: list[dict[str, object]] = []

    for variant in variants:
        run_dir = find_run_dir(results_root, variant, args.seed)
        if run_dir is None:
            fold_status_rows.append(
                {
                    "variant": variant,
                    "fold": None,
                    "status": "run_dir_missing",
                    "warning": f"Run directory missing for variant {variant}",
                }
            )
            continue

        settings = read_experiment_settings(run_dir)
        if not settings:
            fold_status_rows.append(
                {
                    "variant": variant,
                    "fold": None,
                    "status": "settings_missing",
                    "warning": f"experiment_*.txt missing or unreadable in {run_dir}",
                }
            )
            continue

        for fold in expected_fold_indices(settings):
            try:
                rows, status = probe_variant_fold(variant, fold, run_dir, settings, args)
            except Exception as exc:
                rows = []
                status = {
                    "variant": variant,
                    "fold": fold,
                    "status": "probe_failed",
                    "warning": str(exc),
                }
            slide_rows.extend(rows)
            fold_status_rows.append(status)

    slide_df = pd.DataFrame(slide_rows)
    feature_columns = []
    base_columns = {
        "variant",
        "fold",
        "slide_id",
        "label",
        "pred",
        "correct",
        "prarc_gate",
        "prarc_gate_features",
        "prarc_skip_reason",
        "visual_concept_conflict",
        "low_high_sign_agreement",
        "concept_pred_margin_abs",
        "prediction_confidence_margin",
        "dominant_source_ratio",
    }
    if not slide_df.empty:
        feature_columns = [col for col in slide_df.columns if col not in base_columns]

    distribution_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    variant_diagnostics: list[dict[str, object]] = []

    if not slide_df.empty:
        for (variant, fold), group in slide_df.groupby(["variant", "fold"]):
            distribution_rows.append(collect_distribution_row(group))
            condition_rows.extend(collect_condition_rows(group))
            correlation_rows.extend(collect_correlation_rows(group, feature_columns))

        dist_df = pd.DataFrame(distribution_rows)
        cond_df = pd.DataFrame(condition_rows)
        corr_df = pd.DataFrame(correlation_rows)

        for variant, group in dist_df.groupby("variant"):
            entry: dict[str, object] = {"variant": variant}
            for col in ["gate_mean", "gate_std", "gate_range", "fraction_gate_below_0.5", "fraction_gate_above_0.9"]:
                if col in group.columns:
                    entry[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            for condition_name, positive_value, negative_value, out_name in [
                ("correct", 0, 1, "error_vs_correct_delta"),
                ("visual_concept_conflict", 1, 0, "conflict_vs_nonconflict_delta"),
                ("low_high_sign_agreement", 0, 1, "disagree_vs_agree_delta"),
            ]:
                subset = cond_df[cond_df["variant"] == variant]
                subset = subset[subset["condition_name"] == condition_name]
                pos = subset[subset["condition_value"] == positive_value]["gate_mean"]
                neg = subset[subset["condition_value"] == negative_value]["gate_mean"]
                if not pos.empty and not neg.empty:
                    entry[out_name] = float(pos.mean() - neg.mean())
            entry["sample_adaptive_flag"] = bool(
                (entry.get("gate_std_mean") is not None and entry.get("gate_std_mean", 0.0) > 1e-3)
                and (entry.get("gate_range_mean") is not None and entry.get("gate_range_mean", 0.0) > 1e-2)
                or (entry.get("error_vs_correct_delta") is not None and entry.get("error_vs_correct_delta", 0.0) < -0.01)
                or (entry.get("conflict_vs_nonconflict_delta") is not None and entry.get("conflict_vs_nonconflict_delta", 0.0) < -0.01)
            )
            variant_diagnostics.append(entry)
    else:
        dist_df = pd.DataFrame(columns=["variant", "fold"])
        cond_df = pd.DataFrame(columns=["variant", "fold", "condition_name", "condition_value"])
        corr_df = pd.DataFrame(columns=["variant", "fold", "feature_name", "correlation", "n"])

    slide_csv = output_dir / "stage47_prarc_gate_probe_slide_level.csv"
    dist_csv = output_dir / "stage47_prarc_gate_distribution_summary.csv"
    cond_csv = output_dir / "stage47_prarc_gate_by_condition.csv"
    corr_csv = output_dir / "stage47_prarc_gate_feature_correlation.csv"
    report_md = output_dir / "stage47_prarc_gate_diagnostics_report.md"
    manifest_json = output_dir / "stage47_prarc_gate_diagnostics_manifest.json"

    slide_df.to_csv(slide_csv, index=False)
    dist_df.to_csv(dist_csv, index=False)
    cond_df.to_csv(cond_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)

    report_lines = [
        "# Step47 PRARC Gate Diagnostics Report",
        "",
        "## Purpose",
        "- Probe per-slide PRARC gates on the requested split using trained Step47 checkpoints.",
        "- Check whether the gate has meaningful sample-adaptive spread instead of Step46 smoke-level near-constant behavior.",
        "",
        "## Probe Status",
        f"- requested_variants: `{', '.join(variants)}`",
        f"- split: `{args.split}`",
        f"- max_slides_per_fold: `{args.max_slides_per_fold}`",
        f"- probed_rows: `{len(slide_df)}`",
    ]

    if variant_diagnostics:
        report_lines.extend(["", "## Variant Diagnostics"])
        for item in variant_diagnostics:
            report_lines.extend(
                [
                    f"- {item['variant']}: gate_std_mean=`{item.get('gate_std_mean')}` gate_range_mean=`{item.get('gate_range_mean')}` "
                    f"error_minus_correct=`{item.get('error_vs_correct_delta')}` "
                    f"conflict_minus_nonconflict=`{item.get('conflict_vs_nonconflict_delta')}` "
                    f"sample_adaptive_flag=`{item.get('sample_adaptive_flag')}`",
                ]
            )
    else:
        report_lines.extend(["", "## Variant Diagnostics", "- No valid slide-level probe rows were produced."])

    warnings = [row for row in fold_status_rows if row.get("status") != "ok"]
    if warnings:
        report_lines.extend(["", "## Warnings"])
        for row in warnings:
            report_lines.append(f"- {row.get('variant')} fold={row.get('fold')}: {row.get('status')} | {row.get('warning')}")

    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "step": "Step47 PRARC Gate Diagnostics",
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "requested_variants": variants,
        "split": args.split,
        "max_slides_per_fold": args.max_slides_per_fold,
        "variant_diagnostics": variant_diagnostics,
        "fold_status": fold_status_rows,
        "outputs": {
            "slide_csv": str(slide_csv),
            "distribution_csv": str(dist_csv),
            "condition_csv": str(cond_csv),
            "correlation_csv": str(corr_csv),
            "report_md": str(report_md),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[Done] Wrote Step47 diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
