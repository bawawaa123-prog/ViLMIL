from __future__ import print_function
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from datasets.dataset_generic import Generic_MIL_Dataset
from utils.core_utils import train
from utils.file_utils import save_pkl
from utils.metric_utils import summarize_metric_list
from utils.prompt_utils import print_and_save_concept_prompt_class_mapping
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Configurations for WSI Training")
parser.add_argument("--data_root_dir", type=str, default=None, help="data directory")
parser.add_argument("--data_folder_s", type=str, default=None, help="dir under data directory")
parser.add_argument("--data_folder_l", type=str, default=None, help="dir under data directory")
parser.add_argument("--max_epochs", type=int, default=80, help="maximum number of epochs to train (default: 80)")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 0.0001)")
parser.add_argument("--label_frac", type=float, default=1.0, help="fraction of training labels (default: 1.0)")
parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
parser.add_argument("--k", type=int, default=10, help="number of folds (default: 10)")
parser.add_argument("--k_start", type=int, default=-1, help="start fold (default: -1, last fold)")
parser.add_argument("--k_end", type=int, default=-1, help="end fold (default: -1, first fold)")
parser.add_argument("--results_dir", default="./results", help="results directory (default: ./results)")
parser.add_argument("--split_dir", type=str, default=None)
parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
parser.add_argument("--testing", action="store_true", default=False, help="debugging tool")
parser.add_argument("--early_stopping", action="store_true", default=False, help="enable early stopping")
parser.add_argument("--patience", type=int, default=15, help="early stopping patience (fixed: 15)")
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--drop_out", action="store_true", default=False, help="enable dropout (p=0.25)")
parser.add_argument(
    "--model_type",
    type=str,
    choices=[
        "ViLa_MIL",
        "ViLa_MIL_BiomedCLIP",
        "RCE_MIL_BiomedCLIP",
        "RCE_MIL_BiomedCLIP_v2",
        "DEG_MIL_BiomedCLIP",
    ],
    default="ViLa_MIL_BiomedCLIP",
    help="type of model",
)
parser.add_argument("--mode", type=str, choices=["transformer"], default="transformer")
parser.add_argument("--exp_code", type=str, help="experiment code for saving results")
parser.add_argument("--weighted_sample", action="store_true", default=False, help="enable weighted sampling")
parser.add_argument("--reg", type=float, default=1e-5, help="weight decay (default: 1e-5)")
parser.add_argument("--bag_loss", type=str, choices=["svm", "ce", "focal"], default="ce")
parser.add_argument("--task", type=str)
parser.add_argument("--text_prompt", type=str, default=None)
parser.add_argument("--text_prompt_path", type=str, default=None)
parser.add_argument("--concept_prompt_path", type=str, default=None)
parser.add_argument("--use_concept_prompt_pool", action="store_true", default=False)
parser.add_argument(
    "--prompt_ensemble_mode",
    type=str,
    choices=["embedding_mean", "logit_mean", "dynamic_gate", "peps", "sap_peps"],
    default="embedding_mean",
)
parser.add_argument("--use_dynamic_prompt_gate", action="store_true", default=False)
parser.add_argument("--dynamic_gate_hidden_dim", type=int, default=256)
parser.add_argument("--dynamic_gate_residual_mean", action="store_true", default=False)
parser.add_argument("--prompt_dropout", type=float, default=0.0)
parser.add_argument("--peps_topk", type=int, default=3)
parser.add_argument("--peps_tau", type=float, default=0.1)
parser.add_argument("--save_peps_weights", action="store_true", default=False)
parser.add_argument("--save_sap_peps_weights", action="store_true", default=False)
parser.add_argument("--spatial_lambda", type=float, default=1.0)
parser.add_argument("--spatial_sigma", type=float, default=1.0)
parser.add_argument(
    "--spatial_score_type",
    type=str,
    choices=["centroid_mean_dist"],
    default="centroid_mean_dist",
)
parser.add_argument("--prototype_number", type=int, default=16)
parser.add_argument("--rce_use_logit_calibration", action="store_true", default=False)
parser.add_argument("--rce_use_concept_prior", action="store_true", default=False)
parser.add_argument("--rce_logit_scale_init", type=float, default=10.0)
parser.add_argument("--rce_concept_prior_strength", type=float, default=1.0)
parser.add_argument("--rce_use_visual_residual", action="store_true", default=False)
parser.add_argument("--rce_visual_residual_init", type=float, default=0.1)
parser.add_argument("--rce_use_visual_evidence_gate", action="store_true", default=False)
parser.add_argument("--rce_visual_gate_init", type=float, default=1.0)
parser.add_argument("--rce_use_prarc_gate", action="store_true", default=False)
parser.add_argument("--rce_prarc_gate_version", type=str, choices=["v1", "v2"], default="v1")
parser.add_argument("--rce_prarc_gate_hidden_dim", type=int, default=16)
parser.add_argument("--rce_prarc_gate_init", type=float, default=0.8)
parser.add_argument("--rce_prarc_gate_dropout", type=float, default=0.0)
parser.add_argument("--rce_prarc_gate_gain", type=float, default=1.0)
parser.add_argument("--rce_prarc_gate_last_weight_init", type=float, default=0.01)
parser.add_argument("--rce_prarc_gate_feature_set", type=str, default="v1")
parser.add_argument("--rce_prarc_detach_features", action="store_true", default=False)
parser.add_argument("--rce_prarc_include_optional_features", action="store_true", default=False)
parser.add_argument("--rce_prarc_feature_clip", type=float, default=10.0)
parser.add_argument("--rce_prarc_export_debug", action="store_true", default=False)
parser.add_argument("--rce_prarc_use_conflict_prior", action="store_true", default=False)
parser.add_argument("--rce_prarc_conflict_prior_strength", type=float, default=0.2)
parser.add_argument("--rce_prarc_use_gate_entropy_reg", action="store_true", default=False)
parser.add_argument("--rce_prarc_gate_entropy_lambda", type=float, default=0.0)
parser.add_argument("--rce_prarc_use_gate_variance_reg", action="store_true", default=False)
parser.add_argument("--rce_prarc_gate_variance_lambda", type=float, default=0.0)
parser.add_argument("--rce_use_low_high_consistency_loss", action="store_true", default=False)
parser.add_argument("--rce_lh_consistency_lambda", type=float, default=0.0)
parser.add_argument("--rce_lh_consistency_margin", type=float, default=0.0)
parser.add_argument("--rce_use_cross_scale_graph", action="store_true", default=False)
parser.add_argument("--rce_cross_scale_graph_init", type=float, default=0.05)
parser.add_argument(
    "--rce_cross_scale_graph_norm",
    type=str,
    choices=["sqrt", "none"],
    default="sqrt",
    help="normalization for the learnable cross-scale graph residual.",
)
parser.add_argument("--rce_use_hcrc", action="store_true", default=False)
parser.add_argument("--rce_hcrc_alpha_init", type=float, default=0.05)
parser.add_argument("--rce_hcrc_num_anchors", type=int, default=16)
parser.add_argument("--rce_hcrc_num_high_children", type=int, default=16)
parser.add_argument("--rce_hcrc_proposal_radius", type=float, default=4096.0)
parser.add_argument("--rce_hcrc_nms_radius", type=float, default=512.0)
parser.add_argument("--rce_hcrc_bbox_expand", type=float, default=8.0)
parser.add_argument("--rce_hcrc_coord_mode", type=str, default="top_left")
parser.add_argument("--rce_hcrc_scale_ratio", type=float, default=1.0)
parser.add_argument("--rce_hcrc_child_strategy", type=str, default="bbox_containment")
parser.add_argument("--rce_hcrc_candidate_top_l", type=int, default=64)
parser.add_argument("--rce_hcrc_top_g_concepts", type=int, default=8)
parser.add_argument("--rce_hcrc_per_concept_top_m", type=int, default=4)
parser.add_argument("--rce_hcrc_prompt_topk", type=int, default=3)
parser.add_argument("--rce_hcrc_margin_weight", type=float, default=0.5)
parser.add_argument(
    "--rce_hcrc_prompt_scale",
    type=str,
    choices=["low", "high", "avg"],
    default="high",
)
parser.add_argument("--rce_hcrc_min_child_count", type=int, default=1)
parser.add_argument("--rce_hcrc_export_debug", action="store_true", default=False)
parser.add_argument("--deg_use_region_graph", action="store_true", default=False)
parser.add_argument("--deg_region_graph_k", type=int, default=4)
parser.add_argument("--deg_region_graph_alpha", type=float, default=0.1)
parser.add_argument("--deg_use_concept_graph", action="store_true", default=False)
parser.add_argument("--deg_concept_graph_topk", type=int, default=4)
parser.add_argument("--deg_concept_graph_alpha", type=float, default=0.05)
parser.add_argument(
    "--scale_mode",
    type=str,
    choices=["dual", "low_only", "high_only"],
    default="dual",
    help="scale fusion mode: dual keeps the original low+high fusion; low_only/high_only are ablations.",
)
parser.add_argument(
    "--scale_fusion_mode",
    type=str,
    choices=["sum", "learned_gate", "residual_gate"],
    default="sum",
    help="dual-scale fusion strategy: sum keeps the legacy logits_low + logits_high behavior.",
)
parser.add_argument("--scale_gate_hidden_dim", type=int, default=128)
parser.add_argument("--scale_gate_dropout", type=float, default=0.25)
parser.add_argument("--scale_residual_gamma", type=float, default=0.25)
parser.add_argument(
    "--allow_legacy_scale_fusion_ckpt",
    action="store_true",
    default=False,
    help="allow loading old checkpoints that do not contain SAF-PEPS scale gate parameters",
)
parser.add_argument(
    "--finetune_text_encoder",
    action="store_true",
    default=False,
    help="(BiomedCLIP) finetune text encoder parameters (default: False / frozen)",
)
parser.add_argument(
    "--prompt_lr",
    type=float,
    default=None,
    help="(BiomedCLIP) learning rate for prompt_learner. If omitted, auto-derive from --lr.",
)
parser.add_argument(
    "--text_lr",
    type=float,
    default=None,
    help="(BiomedCLIP) learning rate for text encoder trainable parameters.",
)
parser.add_argument(
    "--text_finetune_mode",
    type=str,
    choices=["proj", "last", "full"],
    default="proj",
    help="(BiomedCLIP) finetune projection only, projection + last N layers, or full text tower.",
)
parser.add_argument(
    "--text_unfreeze_last_n",
    type=int,
    default=2,
    help="(BiomedCLIP) used when --text_finetune_mode=last: unfreeze last N transformer layers.",
)

args = parser.parse_args()


def _resolve_text_prompt_path(path):
    if not path:
        return path
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join("text_prompt", path),
        os.path.join(os.path.dirname(__file__), "text_prompt", path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return path


def _resolve_concept_prompt_path(path):
    if not path:
        return path
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join("dataset_csv", path),
        os.path.join(os.path.dirname(__file__), "dataset_csv", path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return path


def _load_text_prompts(path):
    if not path:
        return None

    try:
        df_tp = pd.read_csv(path)
        cols = [c.strip().lower() for c in df_tp.columns]

        def to_str_list(series):
            return series.astype(str).fillna("").tolist()

        low_prompts = []
        high_prompts = []

        if "low_resolution_description" in cols and "high_resolution_description" in cols:
            low_idx = cols.index("low_resolution_description")
            high_idx = cols.index("high_resolution_description")
            low_prompts = to_str_list(df_tp.iloc[:, low_idx])
            high_prompts = to_str_list(df_tp.iloc[:, high_idx])
        elif len(df_tp.columns) >= 2:
            low_prompts = to_str_list(df_tp.iloc[:, -2])
            high_prompts = to_str_list(df_tp.iloc[:, -1])
        elif len(df_tp.columns) == 1:
            low_prompts = to_str_list(df_tp.iloc[:, 0])

        return list(map(str, low_prompts)) + list(map(str, high_prompts))
    except Exception:
        arr = pd.read_csv(path, header=None).values
        return [str(x) for x in arr.reshape(-1).tolist()]


args.text_prompt_path = _resolve_text_prompt_path(args.text_prompt_path)
args.concept_prompt_path = _resolve_concept_prompt_path(args.concept_prompt_path)
args.text_prompt = _load_text_prompts(args.text_prompt_path)

if args.max_epochs > 80:
    print(f"[Info] 检测到 max_epochs={args.max_epochs}，已强制限制为 80")
args.max_epochs = min(args.max_epochs, 80)
args.patience = 10


def seed_torch(seed=7):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

settings = {
    "num_splits": args.k,
    "k_start": args.k_start,
    "k_end": args.k_end,
    "task": args.task,
    "data_root_dir": args.data_root_dir,
    "data_folder_s": args.data_folder_s,
    "data_folder_l": args.data_folder_l,
    "max_epochs": args.max_epochs,
    "results_dir": args.results_dir,
    "lr": args.lr,
    "experiment": args.exp_code,
    "label_frac": args.label_frac,
    "seed": args.seed,
    "model_type": args.model_type,
    "mode": args.mode,
    "use_drop_out": args.drop_out,
    "weighted_sample": args.weighted_sample,
    "opt": args.opt,
    "patience": args.patience,
    "use_concept_prompt_pool": args.use_concept_prompt_pool,
    "concept_prompt_path": args.concept_prompt_path,
    "prompt_ensemble_mode": args.prompt_ensemble_mode,
    "use_dynamic_prompt_gate": args.use_dynamic_prompt_gate,
    "dynamic_gate_hidden_dim": args.dynamic_gate_hidden_dim,
    "dynamic_gate_residual_mean": args.dynamic_gate_residual_mean,
    "prompt_dropout": args.prompt_dropout,
    "peps_topk": args.peps_topk,
    "peps_tau": args.peps_tau,
    "save_peps_weights": args.save_peps_weights,
    "save_sap_peps_weights": args.save_sap_peps_weights,
    "spatial_lambda": args.spatial_lambda,
    "spatial_sigma": args.spatial_sigma,
    "spatial_score_type": args.spatial_score_type,
    "rce_use_logit_calibration": args.rce_use_logit_calibration,
    "rce_use_concept_prior": args.rce_use_concept_prior,
    "rce_logit_scale_init": args.rce_logit_scale_init,
    "rce_concept_prior_strength": args.rce_concept_prior_strength,
    "rce_use_visual_residual": args.rce_use_visual_residual,
    "rce_visual_residual_init": args.rce_visual_residual_init,
    "rce_use_visual_evidence_gate": args.rce_use_visual_evidence_gate,
    "rce_visual_gate_init": args.rce_visual_gate_init,
    "rce_use_prarc_gate": args.rce_use_prarc_gate,
    "rce_prarc_gate_version": args.rce_prarc_gate_version,
    "rce_prarc_gate_hidden_dim": args.rce_prarc_gate_hidden_dim,
    "rce_prarc_gate_init": args.rce_prarc_gate_init,
    "rce_prarc_gate_dropout": args.rce_prarc_gate_dropout,
    "rce_prarc_gate_gain": args.rce_prarc_gate_gain,
    "rce_prarc_gate_last_weight_init": args.rce_prarc_gate_last_weight_init,
    "rce_prarc_gate_feature_set": args.rce_prarc_gate_feature_set,
    "rce_prarc_detach_features": args.rce_prarc_detach_features,
    "rce_prarc_include_optional_features": args.rce_prarc_include_optional_features,
    "rce_prarc_feature_clip": args.rce_prarc_feature_clip,
    "rce_prarc_export_debug": args.rce_prarc_export_debug,
    "rce_prarc_use_conflict_prior": args.rce_prarc_use_conflict_prior,
    "rce_prarc_conflict_prior_strength": args.rce_prarc_conflict_prior_strength,
    "rce_prarc_use_gate_entropy_reg": args.rce_prarc_use_gate_entropy_reg,
    "rce_prarc_gate_entropy_lambda": args.rce_prarc_gate_entropy_lambda,
    "rce_prarc_use_gate_variance_reg": args.rce_prarc_use_gate_variance_reg,
    "rce_prarc_gate_variance_lambda": args.rce_prarc_gate_variance_lambda,
    "rce_use_low_high_consistency_loss": args.rce_use_low_high_consistency_loss,
    "rce_lh_consistency_lambda": args.rce_lh_consistency_lambda,
    "rce_lh_consistency_margin": args.rce_lh_consistency_margin,
    "rce_use_cross_scale_graph": args.rce_use_cross_scale_graph,
    "rce_cross_scale_graph_init": args.rce_cross_scale_graph_init,
    "rce_cross_scale_graph_norm": args.rce_cross_scale_graph_norm,
    "rce_use_hcrc": args.rce_use_hcrc,
    "rce_hcrc_alpha_init": args.rce_hcrc_alpha_init,
    "rce_hcrc_num_anchors": args.rce_hcrc_num_anchors,
    "rce_hcrc_num_high_children": args.rce_hcrc_num_high_children,
    "rce_hcrc_proposal_radius": args.rce_hcrc_proposal_radius,
    "rce_hcrc_nms_radius": args.rce_hcrc_nms_radius,
    "rce_hcrc_bbox_expand": args.rce_hcrc_bbox_expand,
    "rce_hcrc_coord_mode": args.rce_hcrc_coord_mode,
    "rce_hcrc_scale_ratio": args.rce_hcrc_scale_ratio,
    "rce_hcrc_child_strategy": args.rce_hcrc_child_strategy,
    "rce_hcrc_candidate_top_l": args.rce_hcrc_candidate_top_l,
    "rce_hcrc_top_g_concepts": args.rce_hcrc_top_g_concepts,
    "rce_hcrc_per_concept_top_m": args.rce_hcrc_per_concept_top_m,
    "rce_hcrc_prompt_topk": args.rce_hcrc_prompt_topk,
    "rce_hcrc_margin_weight": args.rce_hcrc_margin_weight,
    "rce_hcrc_prompt_scale": args.rce_hcrc_prompt_scale,
    "rce_hcrc_min_child_count": args.rce_hcrc_min_child_count,
    "rce_hcrc_export_debug": args.rce_hcrc_export_debug,
    "deg_use_region_graph": args.deg_use_region_graph,
    "deg_region_graph_k": args.deg_region_graph_k,
    "deg_region_graph_alpha": args.deg_region_graph_alpha,
    "deg_use_concept_graph": args.deg_use_concept_graph,
    "deg_concept_graph_topk": args.deg_concept_graph_topk,
    "deg_concept_graph_alpha": args.deg_concept_graph_alpha,
    "scale_mode": args.scale_mode,
    "scale_fusion_mode": args.scale_fusion_mode,
    "scale_gate_hidden_dim": args.scale_gate_hidden_dim,
    "scale_gate_dropout": args.scale_gate_dropout,
    "scale_residual_gamma": args.scale_residual_gamma,
    "allow_legacy_scale_fusion_ckpt": args.allow_legacy_scale_fusion_ckpt,
}

print("\nLoad Dataset")

if args.task == "task_tcga_rcc_subtyping":
    args.n_classes = 3
    args.class_names = ["CCRCC", "PRCC", "CRCC"]
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/TCGA_RCC_subtyping.csv",
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(args.class_names)},
        patient_strat=False,
        ignore=[],
    )
elif args.task == "task_tcga_lung_subtyping":
    args.n_classes = 2
    args.class_names = ["LUAD", "LUSC"]
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/TCGA_Lung_subtyping.csv",
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(args.class_names)},
        patient_strat=False,
        ignore=[],
    )
elif args.task == "task_adenocarcinoma":
    args.n_classes = 2
    args.class_names = ["Adenocarcinoma", "NonAdenocarcinoma"]
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/all_data.csv",
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={name: idx for idx, name in enumerate(args.class_names)},
        patient_strat=False,
        ignore=[],
    )

    if isinstance(args.text_prompt, list):
        args.text_prompt = [str(x) for x in args.text_prompt]
        expected = args.n_classes * 2
        print(f"Text prompts loaded: {len(args.text_prompt)} items (expected {expected} = 2 x n_classes)")
        if len(args.text_prompt) != expected:
            print(
                "[Warning] The number of text prompts does not match 2 x n_classes.\n"
                "          Ensure your CSV has both low_resolution_description and high_resolution_description columns."
            )
    else:
        print("[Warning] args.text_prompt is not a list. Please check --text_prompt_path parsing.")
else:
    raise NotImplementedError

settings.update(
    {
        "n_classes": args.n_classes,
        "class_names": args.class_names,
    }
)

if args.use_concept_prompt_pool and not args.concept_prompt_path:
    raise ValueError("--use_concept_prompt_pool is set but --concept_prompt_path is missing.")
if args.prompt_ensemble_mode == "dynamic_gate" and not args.use_concept_prompt_pool:
    raise ValueError("--prompt_ensemble_mode dynamic_gate requires --use_concept_prompt_pool.")
if args.use_dynamic_prompt_gate and args.prompt_ensemble_mode != "dynamic_gate":
    raise ValueError("--use_dynamic_prompt_gate requires --prompt_ensemble_mode dynamic_gate.")
if args.prompt_ensemble_mode == "peps" and not args.use_concept_prompt_pool:
    raise ValueError("--prompt_ensemble_mode peps requires --use_concept_prompt_pool.")
if args.prompt_ensemble_mode == "sap_peps" and not args.use_concept_prompt_pool:
    raise ValueError("--prompt_ensemble_mode sap_peps requires --use_concept_prompt_pool.")

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.use_concept_prompt_pool:
    print_and_save_concept_prompt_class_mapping(
        prompt_json_path=args.concept_prompt_path,
        output_dir=args.results_dir,
        num_classes=args.n_classes,
        class_names=args.class_names,
    )

if args.split_dir is None:
    args.split_dir = os.path.join("splits", f"{args.task}_{int(args.label_frac * 100)}")
else:
    if not (args.split_dir.startswith("splits/") or args.split_dir.startswith("splits\\")):
        args.split_dir = os.path.join("splits", args.split_dir)

print("split_dir: ", args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({"split_dir": args.split_dir})

with open(os.path.join(args.results_dir, f"experiment_{args.exp_code}.txt"), "w") as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print(f"{key}:  {val}")


def main(args):
    total_start_time = time.time()
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end + 1

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    all_test_balanced_acc = []
    all_test_sensitivity = []
    all_test_specificity = []
    all_test_pr_auc = []
    all_epoch_details = []

    folds = np.arange(start, end)
    if len(folds) == 0:
        raise ValueError(
            f"Empty fold range computed from --k_start={args.k_start} --k_end={args.k_end} with --k={args.k}. "
            "Note: --k_end is inclusive. Example single-fold run: --k_start 0 --k_end 0"
        )

    print(f"\n{'=' * 100}")
    print(f"🎯 开始 {len(folds)} 折交叉验证训练")
    print(f"{'=' * 100}")

    for i in folds:
        fold_start_time = time.time()
        print(f"\n{'🔥' * 50}")
        print(f"🚀 开始训练 FOLD {i + 1}/{len(folds)}")
        print(f"{'🔥' * 50}")

        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path=f"{args.split_dir}/splits_{i}.csv",
        )
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_metrics, val_metrics, _, epoch_details = train(datasets, i, args)

        all_test_auc.append(test_metrics["auc"])
        all_val_auc.append(val_metrics["auc"])
        all_test_f1.append(test_metrics["f1"])
        all_test_acc.append(test_metrics["acc"])
        all_val_acc.append(val_metrics["acc"])
        all_test_balanced_acc.append(test_metrics["balanced_acc"])
        all_test_sensitivity.append(test_metrics["sensitivity"])
        all_test_specificity.append(test_metrics["specificity"])
        all_test_pr_auc.append(test_metrics["pr_auc"])
        all_epoch_details.extend(epoch_details)

        filename = os.path.join(args.results_dir, f"split_{i}_results.pkl")
        save_pkl(filename, results)

        fold_duration = time.time() - fold_start_time
        print(f"\n✅ FOLD {i + 1} 完成! (用时: {fold_duration / 60:.2f} 分钟)")
        print(f"   Final Test AUC: {test_metrics['auc']:.4f}")
        print(f"   Final Test ACC: {test_metrics['acc']:.4f}")
        print(f"   Final Test F1:  {test_metrics['f1']:.4f}")
        print(f"   Balanced ACC:   {test_metrics['balanced_acc']:.4f}")
        print(f"   Sensitivity:    {test_metrics['sensitivity']:.4f}")
        print(f"   Specificity:    {test_metrics['specificity']:.4f}")
        print(f"   PR-AUC:         {test_metrics['pr_auc']:.4f}")

    if all_epoch_details:
        epoch_df = pd.DataFrame(all_epoch_details)
        epoch_csv_path = os.path.join(args.results_dir, "epoch_details.csv")
        epoch_df.to_csv(epoch_csv_path, index=False)
        print(f"\n📁 已保存epoch详情到: {epoch_csv_path}")

    fold_summary_data = []
    for i, fold in enumerate(folds):
        fold_summary_data.append(
            {
                "fold": fold + 1,
                "test_auc": all_test_auc[i],
                "test_acc": all_test_acc[i],
                "test_f1": all_test_f1[i],
                "val_auc": all_val_auc[i],
                "val_acc": all_val_acc[i],
                "balanced_acc": all_test_balanced_acc[i],
                "sensitivity": all_test_sensitivity[i],
                "specificity": all_test_specificity[i],
                "pr_auc": all_test_pr_auc[i],
            }
        )

    fold_summary_df = pd.DataFrame(fold_summary_data)
    fold_summary_csv = os.path.join(args.results_dir, "fold_summary.csv")
    fold_summary_df.to_csv(fold_summary_csv, index=False)
    print(f"📁 已保存折汇总到: {fold_summary_csv}")

    print(f"\n{'=' * 100}")
    print(f"📊 {len(folds)} 折交叉验证 - 详细总结报告")
    print(f"{'=' * 100}")

    print(f"\n📋 各折详细结果:")
    print(
        f"{'Fold':<6}{'Test_AUC':<10}{'Test_ACC':<10}{'Test_F1':<10}{'Val_AUC':<10}"
        f"{'Bal_ACC':<10}{'Sens':<10}{'Spec':<10}{'PR_AUC':<10}"
    )
    print(f"{'-' * 90}")
    for i, fold in enumerate(folds):
        print(
            f"{fold + 1:<6}{all_test_auc[i]:<10.4f}{all_test_acc[i]:<10.4f}"
            f"{all_test_f1[i]:<10.4f}{all_val_auc[i]:<10.4f}{all_test_balanced_acc[i]:<10.4f}"
            f"{all_test_sensitivity[i]:<10.4f}{all_test_specificity[i]:<10.4f}{all_test_pr_auc[i]:<10.4f}"
        )

    print(f"\n📈 统计总结:")
    for label, values in [
        ("Test AUC", all_test_auc),
        ("Test ACC", all_test_acc),
        ("Test F1", all_test_f1),
        ("Val AUC", all_val_auc),
        ("Balanced ACC", all_test_balanced_acc),
        ("Sensitivity", all_test_sensitivity),
        ("Specificity", all_test_specificity),
        ("PR-AUC", all_test_pr_auc),
    ]:
        mean_value, std_value = summarize_metric_list(values)
        print(f"{label}:  Mean={mean_value:.4f}, Std={std_value:.4f}")

    total_duration = time.time() - total_start_time
    print(f"\n{'=' * 100}")
    print(f"🎉 训练全部完成! 总用时: {total_duration / 3600:.2f} 小时 ({total_duration / 60:.2f} 分钟)")
    print(f"{'=' * 100}")

    final_df = pd.DataFrame(
        {
            "folds": folds,
            "test_auc": all_test_auc,
            "test_acc": all_test_acc,
            "test_f1": all_test_f1,
            "val_auc": all_val_auc,
            "balanced_acc": all_test_balanced_acc,
            "sensitivity": all_test_sensitivity,
            "specificity": all_test_specificity,
            "pr_auc": all_test_pr_auc,
        }
    )
    test_auc_mean, test_auc_std = summarize_metric_list(all_test_auc)
    test_f1_mean, test_f1_std = summarize_metric_list(all_test_f1)
    test_acc_mean, test_acc_std = summarize_metric_list(all_test_acc)
    val_auc_mean, val_auc_std = summarize_metric_list(all_val_auc)
    bal_acc_mean, bal_acc_std = summarize_metric_list(all_test_balanced_acc)
    sens_mean, sens_std = summarize_metric_list(all_test_sensitivity)
    spec_mean, spec_std = summarize_metric_list(all_test_specificity)
    pr_auc_mean, pr_auc_std = summarize_metric_list(all_test_pr_auc)
    result_df = pd.DataFrame(
        {
            "metric": ["mean", "std"],
            "test_auc": [test_auc_mean, test_auc_std],
            "test_f1": [test_f1_mean, test_f1_std],
            "test_acc": [test_acc_mean, test_acc_std],
            "val_auc": [val_auc_mean, val_auc_std],
            "balanced_acc": [bal_acc_mean, bal_acc_std],
            "sensitivity": [sens_mean, sens_std],
            "specificity": [spec_mean, spec_std],
            "pr_auc": [pr_auc_mean, pr_auc_std],
        }
    )

    if len(folds) != args.k:
        save_name = f"summary_partial_{folds[0]}_{folds[-1]}.csv"
        result_name = f"result_partial_{folds[0]}_{folds[-1]}.csv"
    else:
        save_name = "summary.csv"
        result_name = "result.csv"

    result_df.to_csv(os.path.join(args.results_dir, result_name), index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    main(args)
    print("finished!")
    print("end script")
