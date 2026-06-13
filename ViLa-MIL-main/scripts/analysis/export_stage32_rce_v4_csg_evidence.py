from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

DEFAULT_RESULTS_PRIMARY = Path("results_stage30/deg_skeleton_cg_ablation_5fold_e20_s1")
DEFAULT_RESULTS_FALLBACK = Path("results_stage22/rce_v4_csg_a01_5fold_e20_s1")
DEFAULT_OUTPUT_DIR = Path("results_stage32/stage32_rce_v4_csg_evidence_export")
DEFAULT_CONCEPT_PATH = Path("dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json")
DEFAULT_SPLIT_DIR = Path("splits/adenocarcinoma/task_adenocarcinoma_strictcv_100")
DEFAULT_DATA_ROOT = Path("/xiangmu/data/VILMIL")
DEFAULT_DATA_FOLDER_S = "features_biomedclip_5x"
DEFAULT_DATA_FOLDER_L = "features_biomedclip_20x"
DEFAULT_MODEL_PATH = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLASS_NAMES = ["Adenocarcinoma", "NonAdenocarcinoma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Step32 evidence for the current RCE-v4-CSG main model.")
    parser.add_argument("--data_root_dir", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_PRIMARY))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--topk_concepts", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model_type",
        type=str,
        default="DEG_MIL_BiomedCLIP",
        choices=["DEG_MIL_BiomedCLIP", "RCE_MIL_BiomedCLIP"],
    )
    parser.add_argument("--concept_prompt_path", type=str, default=str(DEFAULT_CONCEPT_PATH))
    parser.add_argument("--split_dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--data_folder_s", type=str, default=DEFAULT_DATA_FOLDER_S)
    parser.add_argument("--data_folder_l", type=str, default=DEFAULT_DATA_FOLDER_L)
    return parser.parse_args()


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def resolve_results_dir(root: Path, results_dir_arg: str, warning_log: list[str]) -> Path:
    requested = resolve_path(root, results_dir_arg)
    if requested is not None and requested.is_dir():
        return requested

    primary = resolve_path(root, DEFAULT_RESULTS_PRIMARY)
    fallback = resolve_path(root, DEFAULT_RESULTS_FALLBACK)
    if requested == primary and fallback is not None and fallback.is_dir():
        warn_message(
            f"Primary results directory not found: {requested}. Falling back to {fallback}.",
            warning_log,
        )
        return fallback

    if requested is None:
        return primary

    warn_message(f"Results directory not found: {requested}", warning_log)
    return requested


def load_experiment_args(results_dir: Path, warning_log: list[str]) -> dict:
    candidates = sorted(results_dir.glob("experiment_*.txt"))
    if not candidates:
        warn_message(f"No experiment_*.txt found under {results_dir}", warning_log)
        return {}

    experiment_path = candidates[0]
    try:
        return ast.literal_eval(experiment_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn_message(f"Failed to parse {experiment_path}: {exc}", warning_log)
        return {}


def build_mainline_config(root: Path, args: argparse.Namespace, experiment_args: dict) -> SimpleNamespace:
    config = SimpleNamespace()
    for key, value in experiment_args.items():
        setattr(config, key, value)

    config.task = "task_adenocarcinoma"
    config.n_classes = 2
    config.class_names = list(CLASS_NAMES)
    config.model_type = args.model_type
    config.mode = "transformer"
    config.input_size = 512
    config.hidden_size = 192
    config.prototype_number = 16
    config.scale_mode = "dual"
    config.use_concept_prompt_pool = True
    config.prompt_ensemble_mode = "embedding_mean"
    config.use_dynamic_prompt_gate = False
    config.dynamic_gate_hidden_dim = 256
    config.dynamic_gate_residual_mean = False
    config.prompt_dropout = 0.0
    config.peps_topk = int(getattr(config, "peps_topk", 3))
    config.peps_tau = float(getattr(config, "peps_tau", 0.1))
    config.save_peps_weights = False
    config.save_sap_peps_weights = False
    config.spatial_lambda = 1.0
    config.spatial_sigma = 1.0
    config.spatial_score_type = "centroid_mean_dist"
    config.scale_fusion_mode = "sum"
    config.scale_gate_hidden_dim = 128
    config.scale_gate_dropout = 0.25
    config.scale_residual_gamma = 0.25
    config.allow_legacy_scale_fusion_ckpt = False
    config.finetune_text_encoder = False
    config.text_finetune_mode = "proj"
    config.text_unfreeze_last_n = 2
    config.rce_use_concept_prior = True
    config.rce_use_logit_calibration = True
    config.rce_logit_scale_init = 10.0
    config.rce_concept_prior_strength = 1.0
    config.rce_use_visual_residual = True
    config.rce_visual_residual_init = 0.05
    config.rce_use_cross_scale_graph = True
    config.rce_cross_scale_graph_init = 0.1
    config.rce_cross_scale_graph_norm = "sqrt"
    config.deg_use_region_graph = False
    config.deg_region_graph_k = int(getattr(config, "deg_region_graph_k", 4))
    config.deg_region_graph_alpha = float(getattr(config, "deg_region_graph_alpha", 0.1))
    config.deg_use_concept_graph = False
    config.deg_concept_graph_topk = int(getattr(config, "deg_concept_graph_topk", 4))
    config.deg_concept_graph_alpha = float(getattr(config, "deg_concept_graph_alpha", 0.05))
    config.data_root_dir = str(args.data_root_dir)
    config.data_folder_s = str(args.data_folder_s)
    config.data_folder_l = str(args.data_folder_l)
    config.concept_prompt_path = str(resolve_path(root, args.concept_prompt_path))
    config.split_dir = str(resolve_path(root, args.split_dir))
    return config


def load_concept_prompt_metadata(prompt_json_path: Path, warning_log: list[str]) -> dict[str, dict[int, list[dict]]]:
    grouped = {"low": {0: [], 1: []}, "high": {0: [], 1: []}}
    if not prompt_json_path.is_file():
        warn_message(f"Concept prompt JSON not found: {prompt_json_path}", warning_log)
        return grouped

    try:
        payload = json.loads(prompt_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn_message(f"Failed to read concept prompt JSON {prompt_json_path}: {exc}", warning_log)
        return grouped

    prompt_items = payload.get("prompts", []) if isinstance(payload, dict) else payload
    if not isinstance(prompt_items, list):
        warn_message(f"Unexpected concept prompt structure in {prompt_json_path}", warning_log)
        return grouped

    for item in prompt_items:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("use_in_stage2", True)):
            continue
        scale = str(item.get("scale", "")).strip().lower()
        if scale not in grouped:
            continue
        try:
            class_id = int(item.get("class_id"))
        except Exception:
            continue
        if class_id not in grouped[scale]:
            continue
        grouped[scale][class_id].append(
            {
                "concept_id": str(item.get("concept_id", "")).strip() or None,
                "concept_text": (
                    str(item.get("concept_en", "")).strip()
                    or str(item.get("prompt", "")).strip()
                    or str(item.get("concept_id", "")).strip()
                    or f"class_{class_id}_{scale}_concept_{len(grouped[scale][class_id])}"
                ),
                "prompt_text": str(item.get("prompt", "")).strip() or None,
                "class_name": str(item.get("class_name", "")).strip() or CLASS_NAMES[class_id],
                "scale": scale,
            }
        )
    return grouped


def build_dataset(config: SimpleNamespace, root: Path, warning_log: list[str]):
    try:
        from datasets.dataset_generic import Generic_MIL_Dataset
    except Exception as exc:
        warn_message(f"Failed to import Generic_MIL_Dataset: {exc}", warning_log)
        return None

    csv_path = root / "dataset_csv" / "all_data.csv"
    data_dir_s = Path(config.data_root_dir) / config.data_folder_s
    data_dir_l = Path(config.data_root_dir) / config.data_folder_l
    if not csv_path.is_file():
        warn_message(f"Dataset CSV not found: {csv_path}", warning_log)
        return None
    if not data_dir_s.is_dir():
        warn_message(f"Low-scale feature directory not found: {data_dir_s}", warning_log)
        return None
    if not data_dir_l.is_dir():
        warn_message(f"High-scale feature directory not found: {data_dir_l}", warning_log)
        return None

    try:
        return Generic_MIL_Dataset(
            csv_path=str(csv_path),
            mode="transformer",
            data_dir_s=str(data_dir_s),
            data_dir_l=str(data_dir_l),
            shuffle=False,
            print_info=False,
            label_dict={name: idx for idx, name in enumerate(CLASS_NAMES)},
            patient_strat=False,
            ignore=[],
        )
    except Exception as exc:
        warn_message(f"Failed to build dataset: {exc}", warning_log)
        return None


def resolve_split_csv(results_dir: Path, split_dir: Path, fold: int, warning_log: list[str]) -> Path | None:
    candidates = [
        results_dir / f"splits_{fold}.csv",
        split_dir / f"splits_{fold}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    warn_message(
        f"Split CSV not found for fold={fold}. Checked: {', '.join(str(path) for path in candidates)}",
        warning_log,
    )
    return None


def get_split_dataset(dataset, split_csv: Path, split_name: str, warning_log: list[str]):
    try:
        all_splits = pd.read_csv(split_csv, dtype=dataset.slide_data["slide_id"].dtype)
    except Exception as exc:
        warn_message(f"Failed to read split CSV {split_csv}: {exc}", warning_log)
        return None

    if split_name not in all_splits.columns:
        warn_message(f"Split column '{split_name}' not found in {split_csv}", warning_log)
        return None

    split_dataset = dataset.get_split_from_df(all_splits, split_name)
    if split_dataset is None:
        warn_message(f"Requested split '{split_name}' is empty in {split_csv}", warning_log)
    return split_dataset


def resolve_checkpoint_path(
    root: Path,
    results_dir: Path,
    fold: int,
    ckpt_path_arg: str | None,
    warning_log: list[str],
) -> Path | None:
    if ckpt_path_arg:
        requested = Path(ckpt_path_arg)
        candidates = [requested]
        if not requested.is_absolute():
            candidates.extend([results_dir / requested, root / requested])
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        warn_message(f"Checkpoint not found from --ckpt_path={ckpt_path_arg}", warning_log)
        return None

    patterns = [
        f"s_{fold}_checkpoint.pt",
        f"fold_{fold}_checkpoint.pt",
        f"fold{fold}_checkpoint.pt",
        f"s_{fold}_*.pt",
        f"*fold{fold}*.pt",
    ]
    for pattern in patterns:
        matches = sorted(results_dir.glob(pattern))
        if matches:
            return matches[0]

    warn_message(f"No checkpoint found under {results_dir} for fold={fold}", warning_log)
    return None


def instantiate_model(config: SimpleNamespace, warning_log: list[str]):
    model_type = config.model_type
    try:
        if model_type == "DEG_MIL_BiomedCLIP":
            from models.model_DEG_MIL_BiomedCLIP import DEG_MIL_BiomedCLIP

            model_cls = DEG_MIL_BiomedCLIP
        else:
            from models.model_RCE_MIL_BiomedCLIP import RCE_MIL_BiomedCLIP

            model_cls = RCE_MIL_BiomedCLIP
        return model_cls(config=config, num_classes=config.n_classes, model_path=DEFAULT_MODEL_PATH)
    except Exception as exc:
        warn_message(f"Failed to instantiate {model_type}: {exc}", warning_log)
        return None


def load_model_checkpoint(model, ckpt_path: Path, device: torch.device, warning_log: list[str]) -> bool:
    try:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        warn_message(f"Failed to load checkpoint {ckpt_path}: {exc}", warning_log)
        return False

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

    try:
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return True
    except Exception as exc:
        warn_message(f"Failed to restore checkpoint {ckpt_path}: {exc}", warning_log)
        return False


def tensor_to_numpy(tensor) -> np.ndarray | None:
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        try:
            tensor = torch.as_tensor(tensor)
        except Exception:
            return None
    return tensor.detach().cpu().numpy()


def scalarize(value):
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, np.generic):
        return scalarize(value.item())
    if torch.is_tensor(value):
        if value.numel() == 1:
            return scalarize(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [scalarize(item) for item in value]
    return value


def safe_float(value) -> float | None:
    value = scalarize(value)
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def compute_scale_logits(prompt_weights, prompt_evidence) -> np.ndarray | None:
    weights = tensor_to_numpy(prompt_weights)
    evidence = tensor_to_numpy(prompt_evidence)
    if weights is None or evidence is None:
        return None
    logits = (weights * evidence).sum(axis=-1)
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    return logits


def compute_pred_margin(final_logits: np.ndarray, pred: int) -> float | None:
    if final_logits is None or final_logits.ndim != 1 or final_logits.size == 0:
        return None
    if final_logits.size == 1:
        return 0.0
    order = np.argsort(final_logits)[::-1]
    if pred == int(order[0]):
        return safe_float(final_logits[order[0]] - final_logits[order[1]])
    return safe_float(final_logits[pred] - final_logits[order[0]])


def compute_metrics(labels: list[int], preds: list[int], prob_class_1: list[float]) -> dict[str, float | None]:
    metrics = {
        "test_auc": None,
        "test_acc": None,
        "test_f1": None,
        "balanced_acc": None,
        "pr_auc": None,
    }
    if not labels:
        return metrics

    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    y_score = np.asarray(prob_class_1)

    try:
        metrics["test_acc"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        pass
    try:
        metrics["test_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        pass
    try:
        metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        pass
    try:
        if len(np.unique(y_true)) > 1:
            metrics["test_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        pass
    try:
        if len(np.unique(y_true)) > 1:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        pass
    return metrics


def build_top_concept_rows(
    slide_id: str,
    fold: int,
    split_name: str,
    label: int,
    pred: int,
    correct: bool,
    scale: str,
    class_id: int,
    class_type: str,
    prompt_weights,
    prompt_evidence,
    region_concept_sim,
    region_coords,
    prompt_metadata: dict[str, dict[int, list[dict]]],
    topk: int,
    warning_log: list[str],
) -> tuple[list[dict], dict | None]:
    weights = tensor_to_numpy(prompt_weights)
    evidence = tensor_to_numpy(prompt_evidence)
    sim = tensor_to_numpy(region_concept_sim)
    coords = tensor_to_numpy(region_coords)

    if weights is None or evidence is None:
        warn_message(f"Missing prompt weights/evidence for slide {slide_id}, scale={scale}", warning_log)
        return [], None

    if weights.ndim == 3:
        weights = weights[0]
    if evidence.ndim == 3:
        evidence = evidence[0]
    if sim is not None and sim.ndim == 4:
        sim = sim[0]
    if coords is not None and coords.ndim == 3:
        coords = coords[0]

    if class_id >= weights.shape[0] or class_id >= evidence.shape[0]:
        return [], None

    concept_items = prompt_metadata.get(scale, {}).get(class_id, [])
    num_prompts = min(len(concept_items), evidence.shape[-1], weights.shape[-1])
    if num_prompts <= 0:
        return [], None

    contributions = weights[class_id, :num_prompts] * evidence[class_id, :num_prompts]
    top_indices = np.argsort(contributions)[::-1][: max(int(topk), 1)]

    rows: list[dict] = []
    first_row: dict | None = None
    for rank, concept_idx in enumerate(top_indices, start=1):
        metadata = concept_items[concept_idx] if concept_idx < len(concept_items) else {}
        region_idx = None
        region_score = None
        region_x = None
        region_y = None
        if sim is not None and class_id < sim.shape[0] and concept_idx < sim.shape[-1]:
            region_scores = sim[class_id, :, concept_idx]
            if region_scores.ndim == 1 and region_scores.size > 0:
                region_idx = int(np.argmax(region_scores))
                region_score = safe_float(region_scores[region_idx])
                if coords is not None and region_idx < coords.shape[0] and coords.shape[-1] >= 2:
                    region_x = safe_float(coords[region_idx, 0])
                    region_y = safe_float(coords[region_idx, 1])

        row = {
            "slide_id": slide_id,
            "fold": fold,
            "split": split_name,
            "label": label,
            "pred": pred,
            "correct": correct,
            "scale": scale,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
            "class_type": class_type,
            "concept_rank": rank,
            "concept_idx": int(concept_idx),
            "concept_id": metadata.get("concept_id"),
            "concept_text": metadata.get("concept_text"),
            "prompt_text": metadata.get("prompt_text"),
            "evidence": safe_float(evidence[class_id, concept_idx]),
            "weight": safe_float(weights[class_id, concept_idx]),
            "contribution": safe_float(contributions[concept_idx]),
            "support_region_idx": region_idx,
            "support_region_score": region_score,
            "support_region_x": region_x,
            "support_region_y": region_y,
        }
        rows.append(row)
        if first_row is None:
            first_row = row
    return rows, first_row


def summarize_concepts(rows: list[dict]) -> str | None:
    if not rows:
        return None
    parts = []
    for row in rows:
        label = row.get("concept_id") or row.get("concept_text") or f"concept_{row.get('concept_idx')}"
        parts.append(f"{label}:{format_float(row.get('contribution'))}")
    return " | ".join(parts)


def build_top_csg_records(
    slide_id: str,
    fold: int,
    split_name: str,
    label: int,
    pred: int,
    correct: bool,
    low_prompt_evidence,
    high_prompt_evidence,
    cross_scale_adj,
    cross_scale_alpha,
    prompt_metadata: dict[str, dict[int, list[dict]]],
    norm_mode: str,
) -> list[dict]:
    low_ev = tensor_to_numpy(low_prompt_evidence)
    high_ev = tensor_to_numpy(high_prompt_evidence)
    adj = tensor_to_numpy(cross_scale_adj)
    alpha = safe_float(cross_scale_alpha)
    if low_ev is None or high_ev is None or adj is None:
        return []

    if low_ev.ndim == 3:
        low_ev = low_ev[0]
    if high_ev.ndim == 3:
        high_ev = high_ev[0]
    if adj.ndim == 3:
        pass
    else:
        return []

    records: list[dict] = []
    for class_id in range(min(low_ev.shape[0], high_ev.shape[0], adj.shape[0], len(CLASS_NAMES))):
        low_items = prompt_metadata.get("low", {}).get(class_id, [])
        high_items = prompt_metadata.get("high", {}).get(class_id, [])
        low_count = min(len(low_items), low_ev.shape[-1], adj.shape[1])
        high_count = min(len(high_items), high_ev.shape[-1], adj.shape[2])
        if low_count <= 0 or high_count <= 0:
            continue

        pair_matrix = (
            low_ev[class_id, :low_count].reshape(low_count, 1)
            * adj[class_id, :low_count, :high_count]
            * high_ev[class_id, :high_count].reshape(1, high_count)
        )
        if norm_mode == "sqrt":
            pair_matrix = pair_matrix / max(math.sqrt(float(low_count * high_count)), 1.0)
        flat_idx = int(np.argmax(pair_matrix))
        low_idx, high_idx = np.unravel_index(flat_idx, pair_matrix.shape)
        low_meta = low_items[low_idx]
        high_meta = high_items[high_idx]
        records.append(
            {
                "slide_id": slide_id,
                "fold": fold,
                "split": split_name,
                "label": label,
                "pred": pred,
                "correct": correct,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "csg_alpha": alpha,
                "low_concept_idx": low_idx,
                "high_concept_idx": high_idx,
                "low_concept_id": low_meta.get("concept_id"),
                "high_concept_id": high_meta.get("concept_id"),
                "low_concept_text": low_meta.get("concept_text"),
                "high_concept_text": high_meta.get("concept_text"),
                "pair_score": safe_float(pair_matrix[low_idx, high_idx]),
                "low_evidence": safe_float(low_ev[class_id, low_idx]),
                "high_evidence": safe_float(high_ev[class_id, high_idx]),
                "adj_weight": safe_float(adj[class_id, low_idx, high_idx]),
                "pair_label": f"{low_meta.get('concept_id') or low_meta.get('concept_text')} -> "
                f"{high_meta.get('concept_id') or high_meta.get('concept_text')}",
            }
        )
    return records


def aggregate_top_items(df: pd.DataFrame, group_cols: list[str], score_col: str, topn: int = 8) -> list[str]:
    if df.empty:
        return ["- none"]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(count=("slide_id", "count"), mean_score=(score_col, "mean"))
        .reset_index()
        .sort_values(["mean_score", "count"], ascending=[False, False])
        .head(topn)
    )
    lines = []
    for row in grouped.itertuples(index=False):
        prefix = " | ".join(str(getattr(row, col)) for col in group_cols)
        lines.append(f"- {prefix}: mean_{score_col}={format_float(getattr(row, 'mean_score'))}, count={int(row.count)}")
    return lines


def build_report(
    output_dir: Path,
    slide_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    csg_df: pd.DataFrame,
    metrics: dict[str, float | None],
    manifest: dict,
    warning_log: list[str],
) -> None:
    report_path = output_dir / "stage32_evidence_export_report.md"
    correct_count = int(slide_df["correct"].sum()) if not slide_df.empty else 0
    error_count = int((~slide_df["correct"].astype(bool)).sum()) if not slide_df.empty else 0

    pred_concept_df = concept_df[concept_df["class_type"] == "pred"] if not concept_df.empty else concept_df
    low_df = pred_concept_df[pred_concept_df["scale"] == "low"] if not pred_concept_df.empty else pred_concept_df
    high_df = pred_concept_df[pred_concept_df["scale"] == "high"] if not pred_concept_df.empty else pred_concept_df

    correct_pred_df = pred_concept_df[pred_concept_df["correct"] == True] if not pred_concept_df.empty else pred_concept_df
    error_pred_df = pred_concept_df[pred_concept_df["correct"] == False] if not pred_concept_df.empty else pred_concept_df

    low_concept_lines = aggregate_top_items(low_df, ["class_name", "concept_text"], "contribution")
    high_concept_lines = aggregate_top_items(high_df, ["class_name", "concept_text"], "contribution")
    csg_lines = aggregate_top_items(csg_df, ["class_name", "pair_label"], "pair_score")
    correct_lines = aggregate_top_items(correct_pred_df, ["scale", "class_name", "concept_text"], "contribution", topn=6)
    error_lines = aggregate_top_items(error_pred_df, ["scale", "class_name", "concept_text"], "contribution", topn=6)

    concept_source = {
        "mean_abs_low_logit": safe_float(slide_df["low_logit_class_0"].abs().add(slide_df["low_logit_class_1"].abs()).mean() / 2.0)
        if not slide_df.empty and {"low_logit_class_0", "low_logit_class_1"}.issubset(slide_df.columns)
        else None,
        "mean_abs_high_logit": safe_float(slide_df["high_logit_class_0"].abs().add(slide_df["high_logit_class_1"].abs()).mean() / 2.0)
        if not slide_df.empty and {"high_logit_class_0", "high_logit_class_1"}.issubset(slide_df.columns)
        else None,
        "mean_abs_visual_logit": safe_float(slide_df["visual_logit_class_0"].abs().add(slide_df["visual_logit_class_1"].abs()).mean() / 2.0)
        if not slide_df.empty and {"visual_logit_class_0", "visual_logit_class_1"}.issubset(slide_df.columns)
        else None,
        "mean_abs_csg_logit": safe_float(slide_df["csg_logit_class_0"].abs().add(slide_df["csg_logit_class_1"].abs()).mean() / 2.0)
        if not slide_df.empty and {"csg_logit_class_0", "csg_logit_class_1"}.issubset(slide_df.columns)
        else None,
        "mean_visual_alpha": safe_float(slide_df["visual_alpha"].dropna().mean()) if not slide_df.empty and "visual_alpha" in slide_df else None,
        "mean_csg_alpha": safe_float(slide_df["csg_alpha"].dropna().mean()) if not slide_df.empty and "csg_alpha" in slide_df else None,
    }

    lines = [
        "# Step32 Evidence Export Report",
        "",
        "## Scope",
        "- This step does not train the model.",
        "- This step does not modify model computation logic.",
        "- This step only exports the evidence chain of the current main model.",
        "",
        "## Run Summary",
        f"- Results directory: `{manifest.get('results_dir')}`",
        f"- Checkpoint: `{manifest.get('ckpt_path')}`",
        f"- Fold: `{manifest.get('fold')}`",
        f"- Split: `{manifest.get('split')}`",
        f"- Model type: `{manifest.get('model_type')}`",
        f"- Exported slides: `{len(slide_df)}`",
        f"- Correct slides: `{correct_count}`",
        f"- Error slides: `{error_count}`",
        "",
        "## Metrics",
        f"- test AUC: `{format_float(metrics.get('test_auc'))}`",
        f"- test ACC: `{format_float(metrics.get('test_acc'))}`",
        f"- test F1: `{format_float(metrics.get('test_f1'))}`",
        f"- Balanced ACC: `{format_float(metrics.get('balanced_acc'))}`",
        f"- PR-AUC: `{format_float(metrics.get('pr_auc'))}`",
        "",
        "## Global Top Low Concepts",
        *low_concept_lines,
        "",
        "## Global Top High Concepts",
        *high_concept_lines,
        "",
        "## Global Top CSG Low-High Pairs",
        *csg_lines,
        "",
        "## Correct vs Error Concept Difference",
        "### Correct samples",
        *correct_lines,
        "",
        "### Error samples",
        *error_lines,
        "",
        "## Evidence Source Overview",
        f"- concept evidence / low mean abs logit: `{format_float(concept_source['mean_abs_low_logit'])}`",
        f"- concept evidence / high mean abs logit: `{format_float(concept_source['mean_abs_high_logit'])}`",
        f"- visual residual mean abs logit: `{format_float(concept_source['mean_abs_visual_logit'])}`",
        f"- visual residual alpha: `{format_float(concept_source['mean_visual_alpha'])}`",
        f"- CSG mean abs logit: `{format_float(concept_source['mean_abs_csg_logit'])}`",
        f"- CSG alpha: `{format_float(concept_source['mean_csg_alpha'])}`",
        "",
        "## Outputs",
        "- `stage32_slide_evidence_summary.csv`",
        "- `stage32_top_concepts_long.csv`",
        "- `stage32_top_csg_pairs.csv`",
        "- `stage32_error_cases.csv`",
        "- `stage32_manifest.json`",
        "",
        "## Warnings",
    ]

    if warning_log:
        lines.extend([f"- {message}" for message in warning_log])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Next Suggested Step",
            "- Step33: Evidence failure / conflict analysis",
            "- Step34: Evidence-level gated residual",
            "- Step35: Evidence consistency / margin loss",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def export_evidence(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    warning_log: list[str] = []
    root = DEFAULT_ROOT
    output_dir = resolve_path(root, args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = resolve_results_dir(root, args.results_dir, warning_log)
    experiment_args = load_experiment_args(results_dir, warning_log) if results_dir.is_dir() else {}
    config = build_mainline_config(root, args, experiment_args)
    split_dir = Path(config.split_dir)
    prompt_metadata = load_concept_prompt_metadata(Path(config.concept_prompt_path), warning_log)

    manifest = {
        "root_dir": str(root),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "fold": int(args.fold),
        "split": args.split,
        "requested_model_type": args.model_type,
        "model_type": config.model_type,
        "ckpt_path": None,
        "config": {key: scalarize(value) for key, value in vars(config).items()},
        "warnings": warning_log,
    }

    dataset = build_dataset(config, root, warning_log)
    split_csv = resolve_split_csv(results_dir, split_dir, args.fold, warning_log)
    split_dataset = get_split_dataset(dataset, split_csv, args.split, warning_log) if dataset is not None and split_csv else None
    ckpt_path = resolve_checkpoint_path(root, results_dir, args.fold, args.ckpt_path, warning_log) if results_dir.is_dir() else None
    manifest["ckpt_path"] = str(ckpt_path) if ckpt_path is not None else None

    slide_rows: list[dict] = []
    concept_rows: list[dict] = []
    csg_rows: list[dict] = []

    if split_dataset is None or ckpt_path is None:
        build_report(output_dir, pd.DataFrame(slide_rows), pd.DataFrame(concept_rows), pd.DataFrame(csg_rows), compute_metrics([], [], []), manifest, warning_log)
        (output_dir / "stage32_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    device = torch.device(args.device)
    model = instantiate_model(config, warning_log)
    if model is None or not load_model_checkpoint(model, ckpt_path, device, warning_log):
        build_report(output_dir, pd.DataFrame(slide_rows), pd.DataFrame(concept_rows), pd.DataFrame(csg_rows), compute_metrics([], [], []), manifest, warning_log)
        (output_dir / "stage32_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    with torch.no_grad():
        for index in range(len(split_dataset)):
            try:
                features_s, coords_s, features_l, coords_l, label, slide_id = split_dataset[index]
            except Exception as exc:
                warn_message(f"Failed to read split item idx={index}: {exc}", warning_log)
                continue

            try:
                label_int = int(label)
                slide_id_str = str(slide_id)
                label_tensor = torch.tensor([label_int], dtype=torch.long, device=device)
                y_prob, y_hat, _ = model(
                    features_s.to(device),
                    coords_s.to(device),
                    features_l.to(device),
                    coords_l.to(device),
                    label_tensor,
                    slide_id=slide_id_str,
                )
            except Exception as exc:
                warn_message(f"Forward failed for slide {slide_id}: {exc}", warning_log)
                continue

            prob = tensor_to_numpy(y_prob)
            pred = int(tensor_to_numpy(y_hat).reshape(-1)[0]) if tensor_to_numpy(y_hat) is not None else 0
            final_logits = tensor_to_numpy(getattr(model, "last_final_logits", None))
            if final_logits is not None and final_logits.ndim == 2:
                final_logits = final_logits[0]
            low_logits = compute_scale_logits(getattr(model, "last_low_prompt_weights", None), getattr(model, "last_low_prompt_evidence", None))
            high_logits = compute_scale_logits(getattr(model, "last_high_prompt_weights", None), getattr(model, "last_high_prompt_evidence", None))
            visual_logits = tensor_to_numpy(getattr(model, "last_visual_logits", None))
            if visual_logits is not None and visual_logits.ndim == 2:
                visual_logits = visual_logits[0]
            csg_logits = tensor_to_numpy(getattr(model, "last_cross_scale_logits", None))
            if csg_logits is not None and csg_logits.ndim == 2:
                csg_logits = csg_logits[0]

            low_pred_rows, low_pred_top = build_top_concept_rows(
                slide_id=slide_id_str,
                fold=args.fold,
                split_name=args.split,
                label=label_int,
                pred=pred,
                correct=pred == label_int,
                scale="low",
                class_id=pred,
                class_type="pred",
                prompt_weights=getattr(model, "last_low_prompt_weights", None),
                prompt_evidence=getattr(model, "last_low_prompt_evidence", None),
                region_concept_sim=getattr(model, "last_low_region_concept_sim", None),
                region_coords=getattr(model, "last_low_region_coords", None),
                prompt_metadata=prompt_metadata,
                topk=args.topk_concepts,
                warning_log=warning_log,
            )
            high_pred_rows, high_pred_top = build_top_concept_rows(
                slide_id=slide_id_str,
                fold=args.fold,
                split_name=args.split,
                label=label_int,
                pred=pred,
                correct=pred == label_int,
                scale="high",
                class_id=pred,
                class_type="pred",
                prompt_weights=getattr(model, "last_high_prompt_weights", None),
                prompt_evidence=getattr(model, "last_high_prompt_evidence", None),
                region_concept_sim=getattr(model, "last_high_region_concept_sim", None),
                region_coords=getattr(model, "last_high_region_coords", None),
                prompt_metadata=prompt_metadata,
                topk=args.topk_concepts,
                warning_log=warning_log,
            )
            low_true_rows, _ = build_top_concept_rows(
                slide_id=slide_id_str,
                fold=args.fold,
                split_name=args.split,
                label=label_int,
                pred=pred,
                correct=pred == label_int,
                scale="low",
                class_id=label_int,
                class_type="true",
                prompt_weights=getattr(model, "last_low_prompt_weights", None),
                prompt_evidence=getattr(model, "last_low_prompt_evidence", None),
                region_concept_sim=getattr(model, "last_low_region_concept_sim", None),
                region_coords=getattr(model, "last_low_region_coords", None),
                prompt_metadata=prompt_metadata,
                topk=args.topk_concepts,
                warning_log=warning_log,
            )
            high_true_rows, _ = build_top_concept_rows(
                slide_id=slide_id_str,
                fold=args.fold,
                split_name=args.split,
                label=label_int,
                pred=pred,
                correct=pred == label_int,
                scale="high",
                class_id=label_int,
                class_type="true",
                prompt_weights=getattr(model, "last_high_prompt_weights", None),
                prompt_evidence=getattr(model, "last_high_prompt_evidence", None),
                region_concept_sim=getattr(model, "last_high_region_concept_sim", None),
                region_coords=getattr(model, "last_high_region_coords", None),
                prompt_metadata=prompt_metadata,
                topk=args.topk_concepts,
                warning_log=warning_log,
            )
            concept_rows.extend(low_pred_rows + high_pred_rows + low_true_rows + high_true_rows)

            slide_csg_rows = build_top_csg_records(
                slide_id=slide_id_str,
                fold=args.fold,
                split_name=args.split,
                label=label_int,
                pred=pred,
                correct=pred == label_int,
                low_prompt_evidence=getattr(model, "last_low_prompt_evidence", None),
                high_prompt_evidence=getattr(model, "last_high_prompt_evidence", None),
                cross_scale_adj=getattr(model, "last_cross_scale_adj", None),
                cross_scale_alpha=getattr(model, "last_cross_scale_alpha", None),
                prompt_metadata=prompt_metadata,
                norm_mode=config.rce_cross_scale_graph_norm,
            )
            csg_rows.extend(slide_csg_rows)
            csg_by_class = {row["class_id"]: row for row in slide_csg_rows}

            slide_rows.append(
                {
                    "slide_id": slide_id_str,
                    "fold": args.fold,
                    "split": args.split,
                    "label": label_int,
                    "pred": pred,
                    "correct": pred == label_int,
                    "prob_class_0": safe_float(prob[0, 0]) if prob is not None and prob.ndim == 2 else None,
                    "prob_class_1": safe_float(prob[0, 1]) if prob is not None and prob.ndim == 2 and prob.shape[1] > 1 else None,
                    "final_logit_class_0": safe_float(final_logits[0]) if final_logits is not None and final_logits.size > 0 else None,
                    "final_logit_class_1": safe_float(final_logits[1]) if final_logits is not None and final_logits.size > 1 else None,
                    "pred_margin": compute_pred_margin(final_logits, pred) if final_logits is not None else None,
                    "low_logit_class_0": safe_float(low_logits[0]) if low_logits is not None and low_logits.size > 0 else None,
                    "low_logit_class_1": safe_float(low_logits[1]) if low_logits is not None and low_logits.size > 1 else None,
                    "high_logit_class_0": safe_float(high_logits[0]) if high_logits is not None and high_logits.size > 0 else None,
                    "high_logit_class_1": safe_float(high_logits[1]) if high_logits is not None and high_logits.size > 1 else None,
                    "visual_logit_class_0": safe_float(visual_logits[0]) if visual_logits is not None and visual_logits.size > 0 else None,
                    "visual_logit_class_1": safe_float(visual_logits[1]) if visual_logits is not None and visual_logits.size > 1 else None,
                    "visual_alpha": safe_float(getattr(model, "last_visual_residual_alpha", None)),
                    "csg_logit_class_0": safe_float(csg_logits[0]) if csg_logits is not None and csg_logits.size > 0 else None,
                    "csg_logit_class_1": safe_float(csg_logits[1]) if csg_logits is not None and csg_logits.size > 1 else None,
                    "csg_alpha": safe_float(getattr(model, "last_cross_scale_alpha", None)),
                    "top_csg_pair_class_0": csg_by_class.get(0, {}).get("pair_label"),
                    "top_csg_pair_class_1": csg_by_class.get(1, {}).get("pair_label"),
                    "top_csg_pair_score_class_0": csg_by_class.get(0, {}).get("pair_score"),
                    "top_csg_pair_score_class_1": csg_by_class.get(1, {}).get("pair_score"),
                    "top_low_region_idx_for_pred": low_pred_top.get("support_region_idx") if low_pred_top else None,
                    "top_high_region_idx_for_pred": high_pred_top.get("support_region_idx") if high_pred_top else None,
                    "top_low_region_score_for_pred": low_pred_top.get("support_region_score") if low_pred_top else None,
                    "top_high_region_score_for_pred": high_pred_top.get("support_region_score") if high_pred_top else None,
                    "top_low_region_x": low_pred_top.get("support_region_x") if low_pred_top else None,
                    "top_low_region_y": low_pred_top.get("support_region_y") if low_pred_top else None,
                    "top_high_region_x": high_pred_top.get("support_region_x") if high_pred_top else None,
                    "top_high_region_y": high_pred_top.get("support_region_y") if high_pred_top else None,
                    "top_low_concepts_for_pred": summarize_concepts(low_pred_rows),
                    "top_high_concepts_for_pred": summarize_concepts(high_pred_rows),
                    "top_low_concepts_for_true": summarize_concepts(low_true_rows),
                    "top_high_concepts_for_true": summarize_concepts(high_true_rows),
                }
            )

    slide_df = pd.DataFrame(slide_rows)
    concept_df = pd.DataFrame(concept_rows)
    csg_df = pd.DataFrame(csg_rows)
    error_df = slide_df[slide_df["correct"] == False].copy() if not slide_df.empty else pd.DataFrame(slide_rows)

    slide_df.to_csv(output_dir / "stage32_slide_evidence_summary.csv", index=False, encoding="utf-8")
    concept_df.to_csv(output_dir / "stage32_top_concepts_long.csv", index=False, encoding="utf-8")
    csg_df.to_csv(output_dir / "stage32_top_csg_pairs.csv", index=False, encoding="utf-8")
    error_df.to_csv(output_dir / "stage32_error_cases.csv", index=False, encoding="utf-8")

    metrics = compute_metrics(
        labels=slide_df["label"].astype(int).tolist() if not slide_df.empty else [],
        preds=slide_df["pred"].astype(int).tolist() if not slide_df.empty else [],
        prob_class_1=slide_df["prob_class_1"].fillna(0.0).astype(float).tolist() if not slide_df.empty else [],
    )
    manifest["metrics"] = metrics
    manifest["num_exported_slides"] = int(len(slide_df))
    manifest["num_error_slides"] = int(len(error_df))
    manifest["warnings"] = warning_log

    (output_dir / "stage32_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(output_dir, slide_df, concept_df, csg_df, metrics, manifest, warning_log)
    return 0


def main() -> int:
    args = parse_args()
    return export_evidence(args)


if __name__ == "__main__":
    raise SystemExit(main())
