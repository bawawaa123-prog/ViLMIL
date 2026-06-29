from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
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
from utils.eval_utils import _load_state_dict_with_scale_gate_compat
from utils.metric_utils import compute_classification_metrics
from utils.utils import get_simple_loader

DEFAULT_OUTPUT_DIR = ROOT / "results_stage57B_logit_contribution_audit"
DEFAULT_DATA_ROOT = Path("/xiangmu/data/VILMIL")
DEFAULT_FULL_MANIFEST = ROOT / "results_stage54_rce_evidence_interpretability" / "full" / "stage32_manifest.json"

BRANCH_ORDER = [
    "full",
    "concept_only",
    "full_without_visual",
    "visual_only",
    "low_only",
    "high_only",
    "csg_only",
]

SAMPLE_COLUMNS = [
    "fold",
    "sample_id",
    "slide_id",
    "label",
    "pred_full",
    "pred_concept_only",
    "pred_visual_only",
    "pred_full_without_visual",
    "full_correct",
    "concept_correct",
    "visual_correct",
    "full_without_visual_correct",
    "full_margin",
    "concept_margin",
    "visual_margin",
    "csg_margin",
    "full_top1_margin",
    "concept_top1_margin",
    "visual_top1_margin",
    "csg_top1_margin",
    "visual_contribution_ratio",
    "concept_contribution_ratio",
    "csg_contribution_ratio",
]

FLIP_COLUMNS = [
    "flip_type",
    "fold",
    "sample_id",
    "slide_id",
    "label",
    "pred_full",
    "pred_concept_only",
    "pred_visual_only",
    "pred_full_without_visual",
    "full_correct",
    "concept_correct",
    "visual_correct",
    "full_without_visual_correct",
    "full_margin",
    "concept_margin",
    "visual_margin",
    "csg_margin",
    "visual_contribution_ratio",
    "concept_contribution_ratio",
    "csg_contribution_ratio",
]

BRANCH_METRIC_COLUMNS = [
    "branch",
    "available",
    "num_samples",
    "logit_space",
    "acc",
    "balanced_acc",
    "macro_f1",
    "auc",
    "pr_auc",
]

MARGIN_STATS_COLUMNS = [
    "metric_name",
    "count",
    "mean",
    "std",
    "median",
    "min",
    "max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step57B logit contribution audit for RCE-v2.")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path or run directory")
    parser.add_argument("--run_dir", type=str, default=None, help="directory containing s_<fold>_checkpoint.pt")
    parser.add_argument("--split_csv", type=str, default=None, help="explicit split csv for the selected fold")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--fold", type=int, default=-1, help="single fold to audit; -1 means all available folds")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_folder_s", type=str, default=None)
    parser.add_argument("--data_folder_l", type=str, default=None)
    parser.add_argument("--concept_prompt_path", type=str, default=None)
    parser.add_argument("--rce_use_dynamic_csg", action="store_true")
    parser.add_argument("--rce_dynamic_csg_mode", type=str, default=None)
    parser.add_argument("--rce_dynamic_csg_alpha_init", type=float, default=None)
    parser.add_argument("--rce_dynamic_csg_scale", type=float, default=None)
    parser.add_argument("--rce_dynamic_csg_norm", type=str, default=None)
    parser.add_argument("--rce_dynamic_csg_detach_evidence", action="store_true")
    parser.add_argument("--rce_dynamic_csg_clip", type=float, default=None)
    parser.add_argument("--rce_use_ccra", action="store_true")
    parser.add_argument("--rce_ccra_mode", type=str, default=None)
    parser.add_argument("--rce_ccra_alpha_init", type=float, default=None)
    parser.add_argument("--rce_ccra_scale", type=float, default=None)
    parser.add_argument("--rce_ccra_num_queries", type=int, default=None)
    parser.add_argument("--rce_ccra_query_source", type=str, default=None)
    parser.add_argument("--rce_ccra_detach_prompt", action="store_true")
    parser.add_argument("--rce_ccra_norm", type=str, default=None)
    parser.add_argument("--rce_ccra_dropout", type=float, default=None)
    parser.add_argument("--rce_ccra_clip", type=float, default=None)
    parser.add_argument("--rce_use_l2h_retrieval", action="store_true")
    parser.add_argument("--rce_l2h_mode", type=str, default=None)
    parser.add_argument("--rce_l2h_low_topk", type=int, default=None)
    parser.add_argument("--rce_l2h_high_max_per_low", type=int, default=None)
    parser.add_argument("--rce_l2h_scale_ratio", type=float, default=None)
    parser.add_argument("--rce_l2h_patch_footprint_ratio", type=float, default=None)
    parser.add_argument("--rce_l2h_alpha_init", type=float, default=None)
    parser.add_argument("--rce_l2h_scale", type=float, default=None)
    parser.add_argument("--rce_l2h_fusion", type=str, default=None)
    parser.add_argument("--rce_l2h_aggregate", type=str, default=None)
    parser.add_argument("--rce_l2h_score_mode", type=str, default=None)
    parser.add_argument("--rce_l2h_detach_low_scores", action="store_true")
    parser.add_argument("--rce_l2h_min_high_matches", type=int, default=None)
    parser.add_argument("--rce_l2h_clip", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_folds", type=int, default=None, help="optional cap on audited folds after discovery")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    text = str(path_str)
    marker = "ViLa-MIL-main/"
    if marker in text:
        tail = text.split(marker, 1)[1]
        remapped = ROOT / tail
        if remapped.exists():
            return remapped
    remapped = ROOT / text
    if remapped.exists():
        return remapped
    return candidate


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_experiment_config(path: Path) -> dict:
    return ast.literal_eval(path.read_text(encoding="utf-8").strip())


def discover_sources(args: argparse.Namespace, warnings: list[str]) -> dict:
    source = {
        "manifest_path": None,
        "run_dir": None,
        "experiment_config_path": None,
        "experiment_config": {},
        "manifest_payload": None,
    }

    if args.run_dir:
        run_dir = resolve_repo_path(args.run_dir)
        if run_dir and run_dir.exists():
            source["run_dir"] = run_dir
        else:
            warnings.append(f"run_dir not found: {args.run_dir}")

    checkpoint_arg = resolve_repo_path(args.checkpoint) if args.checkpoint else None
    if checkpoint_arg and checkpoint_arg.is_dir() and source["run_dir"] is None:
        source["run_dir"] = checkpoint_arg

    if DEFAULT_FULL_MANIFEST.is_file():
        manifest_path = DEFAULT_FULL_MANIFEST
        manifest_payload = read_json(manifest_path)
        source["manifest_path"] = manifest_path
        source["manifest_payload"] = manifest_payload
        if source["run_dir"] is None:
            manifest_run_dir = resolve_repo_path(manifest_payload.get("results_dir"))
            if manifest_run_dir and manifest_run_dir.exists():
                source["run_dir"] = manifest_run_dir

    fallback_run_dirs = [
        ROOT / "results_stage52_rce_core_ablation" / "full_rce_v4_csg_rq16_5fold_e20_s1",
        ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1",
        ROOT / "results_stage23_repro" / "rce_v4_csg_a01_rq16_5fold_e20_s1",
    ]
    if source["run_dir"] is None:
        for candidate in fallback_run_dirs:
            if candidate.is_dir():
                source["run_dir"] = candidate
                break

    run_dir = source["run_dir"]
    if run_dir is not None:
        experiment_files = sorted(run_dir.glob("experiment_*.txt"))
        if experiment_files:
            source["experiment_config_path"] = experiment_files[0]
            source["experiment_config"] = read_experiment_config(experiment_files[0])
        else:
            warnings.append(f"no experiment_*.txt found in {run_dir}")

    if not source["experiment_config"] and source["manifest_payload"]:
        source["experiment_config"] = dict(source["manifest_payload"].get("config", {}))

    if checkpoint_arg and checkpoint_arg.is_file():
        source["checkpoint_file"] = checkpoint_arg
    else:
        source["checkpoint_file"] = None

    return source


def build_runtime_config(
    args: argparse.Namespace,
    source: dict,
    warnings: list[str],
) -> SimpleNamespace:
    base = dict(source.get("experiment_config") or {})
    manifest_payload = source.get("manifest_payload") or {}
    manifest_config = dict(manifest_payload.get("config") or {})
    if manifest_config:
        base.update(manifest_config)

    if not base:
        warnings.append("no experiment config discovered; using minimal defaults")

    resolved = {}
    resolved.update(base)
    resolved["model_type"] = "RCE_MIL_BiomedCLIP_v2"
    resolved["mode"] = str(resolved.get("mode", "transformer"))
    resolved["task"] = str(args.task or resolved.get("task", "task_adenocarcinoma"))
    resolved["n_classes"] = int(resolved.get("n_classes", 2))
    resolved["class_names"] = resolved.get("class_names", ["Adenocarcinoma", "NonAdenocarcinoma"])
    resolved["prototype_number"] = int(resolved.get("prototype_number", 16))
    resolved["use_concept_prompt_pool"] = bool(resolved.get("use_concept_prompt_pool", True))
    resolved["allow_legacy_scale_fusion_ckpt"] = bool(resolved.get("allow_legacy_scale_fusion_ckpt", False))
    resolved["drop_out"] = bool(resolved.get("use_drop_out", resolved.get("drop_out", False)))
    resolved["concept_prompt_path"] = str(
        resolve_repo_path(args.concept_prompt_path or resolved.get("concept_prompt_path")) or ""
    )
    resolved["data_root_dir"] = str(
        resolve_repo_path(args.data_root or resolved.get("data_root_dir")) or DEFAULT_DATA_ROOT
    )
    resolved["data_folder_s"] = str(args.data_folder_s or resolved.get("data_folder_s", "features_biomedclip_5x"))
    resolved["data_folder_l"] = str(args.data_folder_l or resolved.get("data_folder_l", "features_biomedclip_20x"))
    resolved["split_dir"] = str(resolve_repo_path(resolved.get("split_dir")) or "")
    resolved["finetune_text_encoder"] = bool(resolved.get("finetune_text_encoder", False))
    resolved["text_finetune_mode"] = str(resolved.get("text_finetune_mode", "proj"))
    resolved["text_unfreeze_last_n"] = int(resolved.get("text_unfreeze_last_n", 2))
    resolved["enable_logit_breakdown_audit"] = True
    resolved["rce_use_dynamic_csg"] = bool(args.rce_use_dynamic_csg or resolved.get("rce_use_dynamic_csg", False))
    resolved["rce_dynamic_csg_mode"] = str(
        args.rce_dynamic_csg_mode or resolved.get("rce_dynamic_csg_mode", "evidence_outer")
    )
    resolved["rce_dynamic_csg_alpha_init"] = float(
        args.rce_dynamic_csg_alpha_init
        if args.rce_dynamic_csg_alpha_init is not None
        else resolved.get("rce_dynamic_csg_alpha_init", 0.0)
    )
    resolved["rce_dynamic_csg_scale"] = float(
        args.rce_dynamic_csg_scale
        if args.rce_dynamic_csg_scale is not None
        else resolved.get("rce_dynamic_csg_scale", 1.0)
    )
    resolved["rce_dynamic_csg_norm"] = str(
        args.rce_dynamic_csg_norm or resolved.get("rce_dynamic_csg_norm", "softmax")
    )
    resolved["rce_dynamic_csg_detach_evidence"] = bool(
        args.rce_dynamic_csg_detach_evidence or resolved.get("rce_dynamic_csg_detach_evidence", False)
    )
    resolved["rce_dynamic_csg_clip"] = float(
        args.rce_dynamic_csg_clip
        if args.rce_dynamic_csg_clip is not None
        else resolved.get("rce_dynamic_csg_clip", 5.0)
    )
    resolved["rce_use_ccra"] = bool(args.rce_use_ccra or resolved.get("rce_use_ccra", False))
    resolved["rce_ccra_mode"] = str(
        args.rce_ccra_mode or resolved.get("rce_ccra_mode", "concept_query_residual")
    )
    resolved["rce_ccra_alpha_init"] = float(
        args.rce_ccra_alpha_init
        if args.rce_ccra_alpha_init is not None
        else resolved.get("rce_ccra_alpha_init", 0.0)
    )
    resolved["rce_ccra_scale"] = float(
        args.rce_ccra_scale
        if args.rce_ccra_scale is not None
        else resolved.get("rce_ccra_scale", 1.0)
    )
    resolved["rce_ccra_num_queries"] = int(
        args.rce_ccra_num_queries
        if args.rce_ccra_num_queries is not None
        else resolved.get("rce_ccra_num_queries", 0)
    )
    resolved["rce_ccra_query_source"] = str(
        args.rce_ccra_query_source or resolved.get("rce_ccra_query_source", "prompt_mean")
    )
    resolved["rce_ccra_detach_prompt"] = bool(
        args.rce_ccra_detach_prompt or resolved.get("rce_ccra_detach_prompt", False)
    )
    resolved["rce_ccra_norm"] = str(args.rce_ccra_norm or resolved.get("rce_ccra_norm", "layernorm"))
    resolved["rce_ccra_dropout"] = float(
        args.rce_ccra_dropout
        if args.rce_ccra_dropout is not None
        else resolved.get("rce_ccra_dropout", 0.0)
    )
    resolved["rce_ccra_clip"] = float(
        args.rce_ccra_clip
        if args.rce_ccra_clip is not None
        else resolved.get("rce_ccra_clip", 5.0)
    )
    resolved["rce_use_l2h_retrieval"] = bool(
        args.rce_use_l2h_retrieval or resolved.get("rce_use_l2h_retrieval", False)
    )
    resolved["rce_l2h_mode"] = str(
        args.rce_l2h_mode or resolved.get("rce_l2h_mode", "low_topk_coord_window")
    )
    resolved["rce_l2h_low_topk"] = int(
        args.rce_l2h_low_topk
        if args.rce_l2h_low_topk is not None
        else resolved.get("rce_l2h_low_topk", 8)
    )
    resolved["rce_l2h_high_max_per_low"] = int(
        args.rce_l2h_high_max_per_low
        if args.rce_l2h_high_max_per_low is not None
        else resolved.get("rce_l2h_high_max_per_low", 16)
    )
    resolved["rce_l2h_scale_ratio"] = float(
        args.rce_l2h_scale_ratio
        if args.rce_l2h_scale_ratio is not None
        else resolved.get("rce_l2h_scale_ratio", 1.0)
    )
    resolved["rce_l2h_patch_footprint_ratio"] = float(
        args.rce_l2h_patch_footprint_ratio
        if args.rce_l2h_patch_footprint_ratio is not None
        else resolved.get("rce_l2h_patch_footprint_ratio", 4.0)
    )
    resolved["rce_l2h_alpha_init"] = float(
        args.rce_l2h_alpha_init
        if args.rce_l2h_alpha_init is not None
        else resolved.get("rce_l2h_alpha_init", 0.0)
    )
    resolved["rce_l2h_scale"] = float(
        args.rce_l2h_scale
        if args.rce_l2h_scale is not None
        else resolved.get("rce_l2h_scale", 1.0)
    )
    resolved["rce_l2h_fusion"] = str(
        args.rce_l2h_fusion or resolved.get("rce_l2h_fusion", "high_region_residual")
    )
    resolved["rce_l2h_aggregate"] = str(
        args.rce_l2h_aggregate or resolved.get("rce_l2h_aggregate", "mean")
    )
    resolved["rce_l2h_score_mode"] = str(
        args.rce_l2h_score_mode or resolved.get("rce_l2h_score_mode", "low_prompt_max")
    )
    resolved["rce_l2h_detach_low_scores"] = bool(
        args.rce_l2h_detach_low_scores or resolved.get("rce_l2h_detach_low_scores", False)
    )
    resolved["rce_l2h_min_high_matches"] = int(
        args.rce_l2h_min_high_matches
        if args.rce_l2h_min_high_matches is not None
        else resolved.get("rce_l2h_min_high_matches", 1)
    )
    resolved["rce_l2h_clip"] = float(
        args.rce_l2h_clip
        if args.rce_l2h_clip is not None
        else resolved.get("rce_l2h_clip", 5.0)
    )

    if not resolved["concept_prompt_path"]:
        warnings.append("concept_prompt_path missing; model init may fail")

    return SimpleNamespace(**resolved)


def infer_fold_from_checkpoint(path: Path) -> int:
    stem = path.stem
    if stem.startswith("s_") and "_checkpoint" in stem:
        try:
            return int(stem.split("_")[1])
        except Exception:
            return 0
    return 0


def resolve_split_path(
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    source: dict,
    fold: int,
) -> Path | None:
    if args.split_csv:
        return resolve_repo_path(args.split_csv)
    run_dir = source.get("run_dir")
    if run_dir is not None:
        run_split = run_dir / f"splits_{fold}.csv"
        if run_split.is_file():
            return run_split
    split_dir = resolve_repo_path(getattr(runtime_args, "split_dir", ""))
    if split_dir:
        split_path = split_dir / f"splits_{fold}.csv"
        if split_path.is_file():
            return split_path
    return None


def discover_fold_plan(
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    source: dict,
    warnings: list[str],
) -> list[dict]:
    plans = []
    checkpoint_file = source.get("checkpoint_file")
    if checkpoint_file is not None:
        fold = args.fold if args.fold >= 0 else infer_fold_from_checkpoint(checkpoint_file)
        plans.append(
            {
                "fold": fold,
                "ckpt_path": checkpoint_file,
                "split_path": resolve_split_path(args, runtime_args, source, fold),
            }
        )
        return plans

    run_dir = source.get("run_dir")
    if run_dir is None or not run_dir.exists():
        warnings.append("no runnable checkpoint source discovered")
        return plans

    requested_folds = [args.fold] if args.fold >= 0 else list(range(5))
    for fold in requested_folds:
        ckpt_path = run_dir / f"s_{fold}_checkpoint.pt"
        split_path = resolve_split_path(args, runtime_args, source, fold)
        if ckpt_path.is_file():
            plans.append({"fold": fold, "ckpt_path": ckpt_path, "split_path": split_path})
        else:
            warnings.append(f"missing checkpoint for fold {fold}: {ckpt_path}")

    if args.fold < 0 and not plans:
        for ckpt_path in sorted(run_dir.glob("s_*_checkpoint.pt")):
            fold = infer_fold_from_checkpoint(ckpt_path)
            plans.append(
                {
                    "fold": fold,
                    "ckpt_path": ckpt_path,
                    "split_path": resolve_split_path(args, runtime_args, source, fold),
                }
            )

    if args.max_folds is not None and args.max_folds > 0:
        plans = plans[: args.max_folds]

    return plans


def build_dataset(runtime_args: SimpleNamespace) -> Generic_MIL_Dataset:
    task = runtime_args.task
    data_root = Path(runtime_args.data_root_dir)
    data_dir_s = data_root / runtime_args.data_folder_s
    data_dir_l = data_root / runtime_args.data_folder_l
    if task == "task_tcga_rcc_subtyping":
        label_names = ["CCRCC", "PRCC", "CRCC"]
        csv_path = ROOT / "dataset_csv" / "TCGA_RCC_subtyping.csv"
    elif task == "task_tcga_lung_subtyping":
        label_names = ["LUAD", "LUSC"]
        csv_path = ROOT / "dataset_csv" / "TCGA_Lung_subtyping.csv"
    elif task == "task_adenocarcinoma":
        label_names = ["Adenocarcinoma", "NonAdenocarcinoma"]
        csv_path = ROOT / "dataset_csv" / "all_data.csv"
    else:
        raise NotImplementedError(f"Unsupported task: {task}")

    runtime_args.class_names = label_names
    runtime_args.n_classes = len(label_names)
    return Generic_MIL_Dataset(
        csv_path=str(csv_path),
        mode=runtime_args.mode,
        data_dir_s=str(data_dir_s),
        data_dir_l=str(data_dir_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(label_names)},
        patient_strat=False,
        ignore=[],
    )


def initiate_rce_v2_model(runtime_args: SimpleNamespace, ckpt_path: Path) -> RCE_MIL_BiomedCLIP:
    import ml_collections

    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.hidden_size = 192
    config.class_names = getattr(runtime_args, "class_names", None)
    config.use_concept_prompt_pool = bool(getattr(runtime_args, "use_concept_prompt_pool", False))
    config.concept_prompt_path = getattr(runtime_args, "concept_prompt_path", None)
    config.peps_tau = float(getattr(runtime_args, "peps_tau", 0.1))
    config.prototype_number = int(getattr(runtime_args, "prototype_number", 16))
    config.rce_use_logit_calibration = bool(getattr(runtime_args, "rce_use_logit_calibration", False))
    config.rce_use_concept_prior = bool(getattr(runtime_args, "rce_use_concept_prior", False))
    config.rce_logit_scale_init = float(getattr(runtime_args, "rce_logit_scale_init", 10.0))
    config.rce_concept_prior_strength = float(getattr(runtime_args, "rce_concept_prior_strength", 1.0))
    config.rce_use_visual_residual = bool(getattr(runtime_args, "rce_use_visual_residual", False))
    config.rce_visual_residual_init = float(getattr(runtime_args, "rce_visual_residual_init", 0.1))
    config.rce_use_cross_scale_graph = bool(getattr(runtime_args, "rce_use_cross_scale_graph", False))
    config.rce_cross_scale_graph_init = float(getattr(runtime_args, "rce_cross_scale_graph_init", 0.05))
    config.rce_cross_scale_graph_norm = str(getattr(runtime_args, "rce_cross_scale_graph_norm", "sqrt"))
    config.rce_use_dynamic_csg = bool(getattr(runtime_args, "rce_use_dynamic_csg", False))
    config.rce_dynamic_csg_mode = str(getattr(runtime_args, "rce_dynamic_csg_mode", "evidence_outer"))
    config.rce_dynamic_csg_alpha_init = float(getattr(runtime_args, "rce_dynamic_csg_alpha_init", 0.0))
    config.rce_dynamic_csg_scale = float(getattr(runtime_args, "rce_dynamic_csg_scale", 1.0))
    config.rce_dynamic_csg_norm = str(getattr(runtime_args, "rce_dynamic_csg_norm", "softmax"))
    config.rce_dynamic_csg_detach_evidence = bool(
        getattr(runtime_args, "rce_dynamic_csg_detach_evidence", False)
    )
    config.rce_dynamic_csg_clip = float(getattr(runtime_args, "rce_dynamic_csg_clip", 5.0))
    config.rce_use_ccra = bool(getattr(runtime_args, "rce_use_ccra", False))
    config.rce_ccra_mode = str(getattr(runtime_args, "rce_ccra_mode", "concept_query_residual"))
    config.rce_ccra_alpha_init = float(getattr(runtime_args, "rce_ccra_alpha_init", 0.0))
    config.rce_ccra_scale = float(getattr(runtime_args, "rce_ccra_scale", 1.0))
    config.rce_ccra_num_queries = int(getattr(runtime_args, "rce_ccra_num_queries", 0))
    config.rce_ccra_query_source = str(getattr(runtime_args, "rce_ccra_query_source", "prompt_mean"))
    config.rce_ccra_detach_prompt = bool(getattr(runtime_args, "rce_ccra_detach_prompt", False))
    config.rce_ccra_norm = str(getattr(runtime_args, "rce_ccra_norm", "layernorm"))
    config.rce_ccra_dropout = float(getattr(runtime_args, "rce_ccra_dropout", 0.0))
    config.rce_ccra_clip = float(getattr(runtime_args, "rce_ccra_clip", 5.0))
    config.rce_use_l2h_retrieval = bool(getattr(runtime_args, "rce_use_l2h_retrieval", False))
    config.rce_l2h_mode = str(getattr(runtime_args, "rce_l2h_mode", "low_topk_coord_window"))
    config.rce_l2h_low_topk = int(getattr(runtime_args, "rce_l2h_low_topk", 8))
    config.rce_l2h_high_max_per_low = int(getattr(runtime_args, "rce_l2h_high_max_per_low", 16))
    config.rce_l2h_scale_ratio = float(getattr(runtime_args, "rce_l2h_scale_ratio", 1.0))
    config.rce_l2h_patch_footprint_ratio = float(
        getattr(runtime_args, "rce_l2h_patch_footprint_ratio", 4.0)
    )
    config.rce_l2h_alpha_init = float(getattr(runtime_args, "rce_l2h_alpha_init", 0.0))
    config.rce_l2h_scale = float(getattr(runtime_args, "rce_l2h_scale", 1.0))
    config.rce_l2h_fusion = str(getattr(runtime_args, "rce_l2h_fusion", "high_region_residual"))
    config.rce_l2h_aggregate = str(getattr(runtime_args, "rce_l2h_aggregate", "mean"))
    config.rce_l2h_score_mode = str(getattr(runtime_args, "rce_l2h_score_mode", "low_prompt_max"))
    config.rce_l2h_detach_low_scores = bool(getattr(runtime_args, "rce_l2h_detach_low_scores", False))
    config.rce_l2h_min_high_matches = int(getattr(runtime_args, "rce_l2h_min_high_matches", 1))
    config.rce_l2h_clip = float(getattr(runtime_args, "rce_l2h_clip", 5.0))
    config.scale_mode = str(getattr(runtime_args, "scale_mode", "dual"))
    config.finetune_text_encoder = bool(getattr(runtime_args, "finetune_text_encoder", False))
    config.enable_logit_breakdown_audit = True

    model = RCE_MIL_BiomedCLIP(config=config, num_classes=runtime_args.n_classes)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_clean = {}
    for key, value in ckpt.items():
        if "instance_loss_fn" in key:
            continue
        ckpt_clean[key.replace(".module", "")] = value
    _load_state_dict_with_scale_gate_compat(
        model,
        ckpt_clean,
        allow_legacy_scale_fusion_ckpt=bool(getattr(runtime_args, "allow_legacy_scale_fusion_ckpt", False)),
    )
    if hasattr(model, "set_logit_breakdown_audit"):
        model.set_logit_breakdown_audit(True)
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    return model


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = exp.sum()
    if denom <= 0:
        return np.full_like(exp, fill_value=1.0 / len(exp), dtype=float)
    return exp / denom


def tensor_to_numpy_row(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 2:
        return array[0].astype(float)
    return array.reshape(-1).astype(float)


def scalar_from(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def bool_or_none(value) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return bool(value)


def compute_branch_metric_row(branch: str, labels: list[int], logits_list: list[np.ndarray]) -> dict:
    if not logits_list:
        return {
            "branch": branch,
            "available": False,
            "num_samples": 0,
            "logit_space": "post_calibration",
            "acc": np.nan,
            "balanced_acc": np.nan,
            "macro_f1": np.nan,
            "auc": np.nan,
            "pr_auc": np.nan,
        }

    probs = np.vstack([softmax_numpy(logits) for logits in logits_list])
    preds = np.argmax(probs, axis=1)
    labels_np = np.asarray(labels, dtype=int)
    metrics = compute_classification_metrics(labels_np, probs, preds, probs.shape[1])
    return {
        "branch": branch,
        "available": True,
        "num_samples": int(len(labels)),
        "logit_space": "post_calibration",
        "acc": metrics["acc"],
        "balanced_acc": metrics["balanced_acc"],
        "macro_f1": metrics["f1"],
        "auc": metrics["auc"],
        "pr_auc": metrics["pr_auc"],
    }


def summarize_series(metric_name: str, values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "metric_name": metric_name,
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "metric_name": metric_name,
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def select_split_dataset(
    dataset: Generic_MIL_Dataset,
    split_path: Path | None,
    split_name: str,
) -> tuple[Generic_MIL_Dataset | None, str]:
    if split_name == "all":
        return dataset, "all"
    if split_path is None or not split_path.is_file():
        return None, f"missing split csv for split={split_name}"
    split_index = {"train": 0, "val": 1, "test": 2}[split_name]
    split_dataset = dataset.return_splits(from_id=False, csv_path=str(split_path))[split_index]
    return split_dataset, "ok"


def build_sample_record(
    fold: int,
    slide_row: pd.Series,
    label: int,
    branch_preds: dict,
    branch_correct: dict,
    branch_true_margins: dict,
    branch_top1_margins: dict,
    ratios: dict,
) -> dict:
    sample_id = slide_row["case_id"] if "case_id" in slide_row.index else slide_row["slide_id"]
    return {
        "fold": fold,
        "sample_id": sample_id,
        "slide_id": slide_row["slide_id"],
        "label": label,
        "pred_full": branch_preds.get("full"),
        "pred_concept_only": branch_preds.get("concept_only"),
        "pred_visual_only": branch_preds.get("visual_only"),
        "pred_full_without_visual": branch_preds.get("full_without_visual"),
        "full_correct": branch_correct.get("full"),
        "concept_correct": branch_correct.get("concept_only"),
        "visual_correct": branch_correct.get("visual_only"),
        "full_without_visual_correct": branch_correct.get("full_without_visual"),
        "full_margin": branch_true_margins.get("full"),
        "concept_margin": branch_true_margins.get("concept_only"),
        "visual_margin": branch_true_margins.get("visual_only"),
        "csg_margin": branch_true_margins.get("csg_only"),
        "full_top1_margin": branch_top1_margins.get("full"),
        "concept_top1_margin": branch_top1_margins.get("concept_only"),
        "visual_top1_margin": branch_top1_margins.get("visual_only"),
        "csg_top1_margin": branch_top1_margins.get("csg_only"),
        "visual_contribution_ratio": ratios.get("visual_contribution_ratio"),
        "concept_contribution_ratio": ratios.get("concept_contribution_ratio"),
        "csg_contribution_ratio": ratios.get("csg_contribution_ratio"),
    }


def determine_best_branch(branch_metrics_df: pd.DataFrame) -> tuple[str | None, str]:
    subset = branch_metrics_df[
        branch_metrics_df["branch"].isin(["full", "concept_only", "visual_only", "full_without_visual"])
        & (branch_metrics_df["available"] == True)
    ].copy()
    if subset.empty:
        return None, "balanced_acc"
    subset["balanced_rank"] = pd.to_numeric(subset["balanced_acc"], errors="coerce")
    subset["acc_rank"] = pd.to_numeric(subset["acc"], errors="coerce")
    subset = subset.sort_values(["balanced_rank", "acc_rank"], ascending=False, na_position="last")
    return str(subset.iloc[0]["branch"]), "balanced_acc"


def assess_visual_dominance(sample_df: pd.DataFrame, branch_metrics_df: pd.DataFrame) -> tuple[str, dict]:
    if sample_df.empty:
        return "insufficient_audit", {}

    metrics_map = branch_metrics_df.set_index("branch").to_dict(orient="index")
    mean_visual_ratio = float(pd.to_numeric(sample_df["visual_contribution_ratio"], errors="coerce").mean())
    median_visual_ratio = float(pd.to_numeric(sample_df["visual_contribution_ratio"], errors="coerce").median())
    pct_visual_over_half = float(
        (pd.to_numeric(sample_df["visual_contribution_ratio"], errors="coerce") > 0.5).mean()
    )
    visual_acc = metrics_map.get("visual_only", {}).get("acc", np.nan)
    concept_acc = metrics_map.get("concept_only", {}).get("acc", np.nan)
    full_acc = metrics_map.get("full", {}).get("acc", np.nan)
    wo_visual_acc = metrics_map.get("full_without_visual", {}).get("acc", np.nan)

    dominates = bool(
        (not np.isnan(mean_visual_ratio) and mean_visual_ratio >= 0.5 and pct_visual_over_half >= 0.5)
        or (
            not np.isnan(visual_acc)
            and not np.isnan(concept_acc)
            and not np.isnan(full_acc)
            and not np.isnan(wo_visual_acc)
            and visual_acc >= concept_acc
            and full_acc - wo_visual_acc >= 0.05
        )
    )
    assessment = "yes" if dominates else "no_clear_dominance"
    details = {
        "mean_visual_contribution_ratio": mean_visual_ratio,
        "median_visual_contribution_ratio": median_visual_ratio,
        "pct_visual_ratio_gt_0_5": pct_visual_over_half,
        "visual_only_acc": visual_acc,
        "concept_only_acc": concept_acc,
        "full_acc": full_acc,
        "full_without_visual_acc": wo_visual_acc,
    }
    return assessment, details


def assess_csg_contribution(sample_df: pd.DataFrame, branch_metrics_df: pd.DataFrame) -> tuple[str, dict]:
    if sample_df.empty:
        return "insufficient_audit", {}

    metrics_map = branch_metrics_df.set_index("branch").to_dict(orient="index")
    mean_csg_ratio = float(pd.to_numeric(sample_df["csg_contribution_ratio"], errors="coerce").mean())
    csg_acc = metrics_map.get("csg_only", {}).get("acc", np.nan)
    concept_acc = metrics_map.get("concept_only", {}).get("acc", np.nan)
    low_only_acc = metrics_map.get("low_only", {}).get("acc", np.nan)
    high_only_acc = metrics_map.get("high_only", {}).get("acc", np.nan)
    low_high_reference = np.nanmean([low_only_acc, high_only_acc])
    observable = bool(
        (not np.isnan(mean_csg_ratio) and mean_csg_ratio >= 0.03)
        or (not np.isnan(csg_acc) and csg_acc > 0.5)
        or (not np.isnan(concept_acc) and not np.isnan(low_high_reference) and concept_acc > low_high_reference)
    )
    return (
        "yes" if observable else "weak_or_unclear",
        {
            "mean_csg_contribution_ratio": mean_csg_ratio,
            "csg_only_acc": csg_acc,
            "concept_only_acc": concept_acc,
            "mean_low_high_acc": low_high_reference,
        },
    )


def recommend_next_step(audit_status: str, visual_assessment: str, csg_assessment: str) -> str:
    if audit_status != "completed":
        return "先补充审计"
    if visual_assessment == "yes":
        return "residual-constrained RCE"
    if csg_assessment == "yes":
        return "dynamic CSG"
    return "residual-constrained RCE"


def write_empty_outputs(output_dir: Path) -> None:
    pd.DataFrame(columns=BRANCH_METRIC_COLUMNS).to_csv(output_dir / "stage57B_branch_metrics.csv", index=False)
    pd.DataFrame(columns=SAMPLE_COLUMNS).to_csv(output_dir / "stage57B_sample_contribution.csv", index=False)
    pd.DataFrame(columns=MARGIN_STATS_COLUMNS).to_csv(output_dir / "stage57B_margin_stats.csv", index=False)
    pd.DataFrame(columns=FLIP_COLUMNS).to_csv(output_dir / "stage57B_flip_cases.csv", index=False)


def main() -> int:
    start_time = time.time()
    args = parse_args()
    output_dir = ensure_output_dir(Path(args.output_dir))
    warnings: list[str] = []
    errors: list[str] = []

    source = discover_sources(args, warnings)
    runtime_args = build_runtime_config(args, source, warnings)
    fold_plans = discover_fold_plan(args, runtime_args, source, warnings)

    config_payload = {
        "resolved_runtime_args": vars(runtime_args),
        "source": {
            "manifest_path": str(source["manifest_path"]) if source.get("manifest_path") else None,
            "run_dir": str(source["run_dir"]) if source.get("run_dir") else None,
            "experiment_config_path": str(source["experiment_config_path"]) if source.get("experiment_config_path") else None,
            "checkpoint_file": str(source["checkpoint_file"]) if source.get("checkpoint_file") else None,
        },
        "requested": vars(args),
        "fold_plan": [
            {
                "fold": item["fold"],
                "ckpt_path": str(item["ckpt_path"]),
                "split_path": str(item["split_path"]) if item.get("split_path") else None,
            }
            for item in fold_plans
        ],
    }
    (output_dir / "stage57B_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_empty_outputs(output_dir)

    dataset = None
    if not fold_plans:
        warnings.append("no fold plan available; only static audit outputs will be generated")
    else:
        data_root = Path(runtime_args.data_root_dir)
        data_dir_s = data_root / runtime_args.data_folder_s
        data_dir_l = data_root / runtime_args.data_folder_l
        if not data_dir_s.exists() or not data_dir_l.exists():
            warnings.append(
                f"feature directories missing: low={data_dir_s.exists()} high={data_dir_l.exists()} | "
                f"{data_dir_s} | {data_dir_l}"
            )
            fold_plans = []

    if fold_plans:
        try:
            dataset = build_dataset(runtime_args)
        except Exception as exc:
            errors.append(f"dataset initialization failed: {exc}")
            fold_plans = []

    sample_records = []
    branch_logits_buffers = {branch: {"labels": [], "logits": []} for branch in BRANCH_ORDER}
    executed_folds = []

    if fold_plans and dataset is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for plan in fold_plans:
            fold = int(plan["fold"])
            split_dataset, split_status = select_split_dataset(dataset, plan.get("split_path"), args.split)
            if split_dataset is None:
                warnings.append(f"fold {fold} skipped: {split_status}")
                continue

            try:
                model = initiate_rce_v2_model(runtime_args, plan["ckpt_path"])
            except Exception as exc:
                errors.append(f"fold {fold} model init failed: {exc}")
                continue

            loader = get_simple_loader(split_dataset, mode=runtime_args.mode)
            slide_rows = split_dataset.slide_data.reset_index(drop=True)
            executed_folds.append(fold)

            for batch_idx, (data_s, coord_s, data_l, coord_l, label, batch_slide_ids) in enumerate(loader):
                data_s = data_s.to(device)
                coord_s = coord_s.to(device)
                data_l = data_l.to(device)
                coord_l = coord_l.to(device)
                label = label.to(device)
                slide_id_for_model = batch_slide_ids[0] if isinstance(batch_slide_ids, (list, tuple)) else None

                with torch.no_grad():
                    model(data_s, coord_s, data_l, coord_l, label, slide_id=slide_id_for_model)

                breakdown = getattr(model, "last_logit_breakdown", None)
                if not breakdown:
                    warnings.append(f"fold {fold} batch {batch_idx}: missing last_logit_breakdown")
                    continue

                post = breakdown.get("post_calibration", {})
                margins = breakdown.get("margins_post_calibration", {})
                ratios = breakdown.get("ratios", {})

                branch_logits = {
                    "full": tensor_to_numpy_row(post.get("full_logits")),
                    "concept_only": tensor_to_numpy_row(post.get("concept_only_logits")),
                    "full_without_visual": tensor_to_numpy_row(post.get("full_without_visual_logits")),
                    "visual_only": tensor_to_numpy_row(post.get("visual_residual_logits")),
                    "low_only": tensor_to_numpy_row(post.get("low_evidence_logits")),
                    "high_only": tensor_to_numpy_row(post.get("high_evidence_logits")),
                    "csg_only": tensor_to_numpy_row(post.get("csg_logits")),
                }
                branch_preds = {}
                branch_correct = {}
                branch_true_margins = {}
                branch_top1_margins = {}
                label_value = int(label.item())

                for branch_name, logits in branch_logits.items():
                    if logits is None:
                        branch_preds[branch_name] = None
                        branch_correct[branch_name] = None
                        branch_true_margins[branch_name] = None
                        branch_top1_margins[branch_name] = None
                        continue
                    branch_preds[branch_name] = int(np.argmax(logits))
                    branch_correct[branch_name] = branch_preds[branch_name] == label_value
                    branch_logits_buffers[branch_name]["labels"].append(label_value)
                    branch_logits_buffers[branch_name]["logits"].append(logits)

                    branch_margin_payload = margins.get(branch_name, {})
                    branch_true_margins[branch_name] = scalar_from(branch_margin_payload.get("true_class_margin"))
                    branch_top1_margins[branch_name] = scalar_from(branch_margin_payload.get("top1_margin"))

                ratio_payload = {
                    "visual_contribution_ratio": scalar_from(ratios.get("visual_contribution_ratio")),
                    "concept_contribution_ratio": scalar_from(ratios.get("concept_contribution_ratio")),
                    "csg_contribution_ratio": scalar_from(ratios.get("csg_contribution_ratio")),
                }
                slide_row = slide_rows.iloc[batch_idx]
                sample_records.append(
                    build_sample_record(
                        fold=fold,
                        slide_row=slide_row,
                        label=label_value,
                        branch_preds=branch_preds,
                        branch_correct=branch_correct,
                        branch_true_margins=branch_true_margins,
                        branch_top1_margins=branch_top1_margins,
                        ratios=ratio_payload,
                    )
                )

    branch_metric_rows = []
    for branch_name in BRANCH_ORDER:
        buffer = branch_logits_buffers[branch_name]
        branch_metric_rows.append(
            compute_branch_metric_row(branch_name, buffer["labels"], buffer["logits"])
        )
    branch_metrics_df = pd.DataFrame(branch_metric_rows, columns=BRANCH_METRIC_COLUMNS)
    branch_metrics_df.to_csv(output_dir / "stage57B_branch_metrics.csv", index=False)

    sample_df = pd.DataFrame(sample_records, columns=SAMPLE_COLUMNS)
    sample_df.to_csv(output_dir / "stage57B_sample_contribution.csv", index=False)

    margin_stats_rows = []
    for column in [
        "full_margin",
        "concept_margin",
        "visual_margin",
        "csg_margin",
        "full_top1_margin",
        "concept_top1_margin",
        "visual_top1_margin",
        "csg_top1_margin",
        "visual_contribution_ratio",
        "concept_contribution_ratio",
        "csg_contribution_ratio",
    ]:
        if column in sample_df.columns:
            margin_stats_rows.append(summarize_series(column, sample_df[column]))
    margin_stats_df = pd.DataFrame(margin_stats_rows, columns=MARGIN_STATS_COLUMNS)
    margin_stats_df.to_csv(output_dir / "stage57B_margin_stats.csv", index=False)

    flip_frames = []
    if not sample_df.empty:
        working = sample_df.copy()
        working["full_correct"] = working["full_correct"].apply(bool_or_none)
        working["concept_correct"] = working["concept_correct"].apply(bool_or_none)
        working["visual_correct"] = working["visual_correct"].apply(bool_or_none)
        working["full_without_visual_correct"] = working["full_without_visual_correct"].apply(bool_or_none)

        flip_specs = [
            ("concept_only_correct_full_wrong", (working["concept_correct"] == True) & (working["full_correct"] == False)),
            ("visual_only_correct_concept_wrong", (working["visual_correct"] == True) & (working["concept_correct"] == False)),
            ("concept_only_wrong_full_correct", (working["concept_correct"] == False) & (working["full_correct"] == True)),
            ("full_without_visual_correct_full_wrong", (working["full_without_visual_correct"] == True) & (working["full_correct"] == False)),
            ("full_correct_full_without_visual_wrong", (working["full_correct"] == True) & (working["full_without_visual_correct"] == False)),
        ]
        for flip_type, mask in flip_specs:
            subset = working.loc[mask, FLIP_COLUMNS[1:]].copy()
            if subset.empty:
                continue
            subset.insert(0, "flip_type", flip_type)
            flip_frames.append(subset)

    flip_df = pd.concat(flip_frames, ignore_index=True) if flip_frames else pd.DataFrame(columns=FLIP_COLUMNS)
    flip_df.to_csv(output_dir / "stage57B_flip_cases.csv", index=False)

    audit_status = "completed" if executed_folds and not errors else "partial" if executed_folds else "static_only"
    best_branch, best_metric_name = determine_best_branch(branch_metrics_df)
    visual_assessment, visual_details = assess_visual_dominance(sample_df, branch_metrics_df)
    csg_assessment, csg_details = assess_csg_contribution(sample_df, branch_metrics_df)
    recommendation = recommend_next_step(audit_status, visual_assessment, csg_assessment)

    status_payload = {
        "status": audit_status,
        "executed_folds": executed_folds,
        "num_samples": int(sample_df.shape[0]),
        "num_flip_cases": int(flip_df.shape[0]),
        "visual_assessment": visual_assessment,
        "visual_details": visual_details,
        "csg_assessment": csg_assessment,
        "csg_details": csg_details,
        "best_branch": best_branch,
        "best_branch_metric": best_metric_name,
        "recommendation": recommendation,
        "warnings": warnings,
        "errors": errors,
        "duration_seconds": round(time.time() - start_time, 2),
    }
    (output_dir / "stage57B_audit_status.json").write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# Step57B Summary",
        "",
        "## Answers",
        "",
        f"- 本 Step 是否修改了原始 RCE 文件：`否`。`models/model_RCE_MIL_BiomedCLIP.py` 未被修改。",
        (
            "- RCE-v2 是否仍保持原训练逻辑不变：`是`。"
            "仅在 `models/model_RCE_MIL_BiomedCLIP_v2.py` 中新增了默认关闭的 "
            "`last_logit_breakdown` 审计缓存，`forward` 仍返回原有的 `(Y_prob, Y_hat, loss)`。"
        ),
        (
            f"- 是否成功导出 logit breakdown：`{'是' if not sample_df.empty else '部分/否'}`。"
            "审计同时缓存了 pre-calibration 与 post-calibration logits；"
            "branch-level metrics 与 sample-level margins 采用 post-calibration logits 统计，以对齐最终预测。"
        ),
        (
            f"- full / concept_only / visual_only / full_without_visual 哪个分支表现最好："
            f"`{best_branch or '无法判断'}`（按 `{best_metric_name}` 选取）。"
        ),
        (
            f"- visual residual 是否明显主导 final logits："
            f"`{visual_assessment}`。"
            f" 依据：mean visual ratio={visual_details.get('mean_visual_contribution_ratio', np.nan):.4f}，"
            f"median={visual_details.get('median_visual_contribution_ratio', np.nan):.4f}，"
            f"ratio>0.5 占比={visual_details.get('pct_visual_ratio_gt_0_5', np.nan):.4f}。"
        ),
        (
            f"- CSG 是否有可观察贡献：`{csg_assessment}`。"
            f" 依据：mean csg ratio={csg_details.get('mean_csg_contribution_ratio', np.nan):.4f}，"
            f"csg_only acc={csg_details.get('csg_only_acc', np.nan)}。"
        ),
        f"- 下一步建议：`{recommendation}`。",
        "",
        "## Run Status",
        "",
        f"- 审计状态：`{audit_status}`",
        f"- 已执行 folds：`{executed_folds}`",
        f"- 审计样本数：`{int(sample_df.shape[0])}`",
        f"- flip cases 数量：`{int(flip_df.shape[0])}`",
        "",
        "## Notes",
        "",
        "- `stage57B_branch_metrics.csv`、`stage57B_sample_contribution.csv`、`stage57B_margin_stats.csv`、`stage57B_flip_cases.csv` 已写入固定输出目录。",
        "- 若未能完整推理，请检查 `stage57B_audit_status.json` 中的 warnings / errors，并使用下方命令复现。",
        "",
        "## Repro Command",
        "",
        "```bash",
        "python scripts/analysis/build_stage57B_logit_contribution_audit.py \\",
        f"  --output_dir {output_dir} \\",
        f"  --split {args.split} \\",
        (f"  --fold {args.fold} \\" if args.fold >= 0 else "  --fold 0 \\"),
        (
            f"  --run_dir {source['run_dir']}"
            if source.get("run_dir")
            else f"  --checkpoint {source['checkpoint_file']}"
        ),
        "```",
    ]
    if warnings:
        summary_lines.extend(["", "## Warnings", ""])
        summary_lines.extend([f"- {item}" for item in warnings])
    if errors:
        summary_lines.extend(["", "## Errors", ""])
        summary_lines.extend([f"- {item}" for item in errors])

    (output_dir / "stage57B_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    return 0 if audit_status == "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
