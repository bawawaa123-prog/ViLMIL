from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COORD_KEYS = ("coords", "coord", "coordinates", "patch_coords")
FALLBACK_CLASS_NAMES = ["Adenocarcinoma", "NonAdenocarcinoma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step42: Concept-Guided Spatial Proposal anchor selection audit.")
    parser.add_argument("--data_root_dir", default="/xiangmu/data/VILMIL")
    parser.add_argument("--data_folder_s", default="features_biomedclip_5x")
    parser.add_argument("--data_folder_l", default="features_biomedclip_20x")
    parser.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    parser.add_argument("--split_dir", default="splits/adenocarcinoma/task_adenocarcinoma_strictcv_100")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output_dir", default="results_stage42/cgsp_anchor_selection_audit")
    parser.add_argument("--stage41_manifest", default="results_stage41/low_high_coordinate_audit/stage41_manifest.json")
    parser.add_argument("--concept_prompt_path", default="dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--max_slides", type=int, default=0)
    parser.add_argument("--low_patch_size", type=float, default=256.0)
    parser.add_argument("--high_patch_size", type=float, default=256.0)
    parser.add_argument("--coord_mode", default="", choices=["", "top_left", "center"])
    parser.add_argument("--scale_ratio", type=float, default=None)
    parser.add_argument("--bbox_expand_values", default="1.0,1.5,2.0,3.0")
    parser.add_argument("--proposal_radius", type=float, default=512.0)
    parser.add_argument("--nms_radius", type=float, default=512.0)
    parser.add_argument("--num_anchors", type=int, default=16)
    parser.add_argument("--candidate_top_l", type=int, default=64)
    parser.add_argument("--top_g_concepts", type=int, default=8)
    parser.add_argument("--per_concept_top_m", type=int, default=4)
    parser.add_argument("--prompt_topk", type=int, default=3)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--high_child_top_m", type=int, default=16)
    parser.add_argument("--num_examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_float_list(raw: str) -> list[float]:
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0:
            raise ValueError(f"Expected positive value, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Expected at least one positive float value.")
    return values


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def warn(message: str, warning_log: list[str]) -> None:
    print(f"[Warning] {message}")
    warning_log.append(message)


def format_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("N/A").astype(str)
    header = "| " + " | ".join(safe_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(safe_df.columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in safe_df.columns) + " |" for _, row in safe_df.iterrows()]
    return "\n".join([header, separator] + rows)


def load_stage41_settings(stage41_manifest: Path, args: argparse.Namespace, warning_log: list[str]) -> dict[str, object]:
    coord_mode = args.coord_mode or "top_left"
    scale_ratio = args.scale_ratio if args.scale_ratio is not None else 1.0
    loaded = False
    if stage41_manifest.is_file():
        try:
            obj = json.loads(stage41_manifest.read_text(encoding="utf-8"))
            if not args.coord_mode:
                coord_mode = str(obj.get("recommended_coord_mode") or "top_left")
            if args.scale_ratio is None:
                scale_ratio = float(obj.get("recommended_scale_ratio") or 1.0)
            loaded = True
        except Exception as exc:
            warn(f"Failed to read Step41 manifest {stage41_manifest}: {exc}", warning_log)
    else:
        warn(f"Step41 manifest not found: {stage41_manifest}; using coord_mode=top_left, scale_ratio=1.0", warning_log)
    if coord_mode not in {"top_left", "center"}:
        warn(f"Invalid coord_mode from Step41/args: {coord_mode}; using top_left", warning_log)
        coord_mode = "top_left"
    return {"coord_mode": coord_mode, "scale_ratio": float(scale_ratio), "loaded_from_stage41": loaded}


def load_slide_ids(
    csv_path: Path,
    split_dir: Path,
    fold: int,
    split: str,
    max_slides: int,
    warning_log: list[str],
) -> tuple[list[str], dict[str, object]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV path does not exist: {csv_path}")
    df_csv = pd.read_csv(csv_path)
    if "slide_id" not in df_csv.columns:
        raise ValueError(f"Expected slide_id column in {csv_path}")
    csv_slide_ids = dedupe_preserve_order(df_csv["slide_id"].dropna().astype(str).tolist())
    selected = list(csv_slide_ids)
    source = "csv"
    source_path = csv_path
    fallback = False
    if split != "all":
        split_path = split_dir / f"splits_{fold}.csv"
        if split_path.is_file():
            df_split = pd.read_csv(split_path)
            if split not in df_split.columns:
                warn(f"Split file {split_path} does not contain column {split}; falling back to CSV.", warning_log)
                fallback = True
            else:
                selected = dedupe_preserve_order(df_split[split].dropna().astype(str).tolist())
                source = "split"
                source_path = split_path
        else:
            warn(f"Split file {split_path} not found; falling back to CSV.", warning_log)
            fallback = True
    if max_slides > 0:
        selected = selected[:max_slides]
    return selected, {
        "source": source,
        "source_path": str(source_path),
        "csv_slide_count": len(csv_slide_ids),
        "selected_slide_count": len(selected),
        "split": split,
        "fold": fold,
        "fallback_to_csv": fallback,
    }


def read_h5_features_coords(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        coord_key = next((key for key in COORD_KEYS if key in handle), None)
        if coord_key is None:
            raise KeyError(f"No coord dataset found in {path}; tried {COORD_KEYS}")
        if "features" not in handle:
            raise KeyError(f"No features dataset found in {path}")
        coords = np.asarray(handle[coord_key])[:, :2]
        features = np.asarray(handle["features"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coords shape in {path}: {coords.shape}")
    if features.ndim != 2:
        raise ValueError(f"Invalid features shape in {path}: {features.shape}")
    if features.shape[0] != coords.shape[0]:
        raise ValueError(f"features/coords row mismatch in {path}: {features.shape[0]} vs {coords.shape[0]}")
    return features, coords.astype(np.float64), coord_key


def normalize_rows(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


def topmean(array: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    array = np.asarray(array)
    size = array.shape[axis]
    if size == 0:
        return np.zeros(array.shape[:axis] + array.shape[axis + 1 :], dtype=np.float32)
    k = min(max(int(k), 1), size)
    partitioned = np.partition(array, size - k, axis=axis)
    top_values = np.take(partitioned, indices=range(size - k, size), axis=axis)
    return top_values.mean(axis=axis)


def load_prompt_metadata(prompt_path: Path, warning_log: list[str]) -> dict[str, dict[int, list[dict[str, object]]]]:
    try:
        import sys

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from utils.prompt_utils import _group_concept_prompt_items

        _, grouped = _group_concept_prompt_items(
            prompt_json_path=str(prompt_path),
            num_classes=2,
            class_names=None,
        )
        return grouped
    except Exception as exc:
        warn(f"Failed to parse concept prompt metadata {prompt_path}: {exc}", warning_log)
        return {"low": {0: [], 1: []}, "high": {0: [], 1: []}}


def load_prompt_features_from_checkpoint(checkpoint_path: Path, warning_log: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    if not checkpoint_path.is_file():
        warn(f"Checkpoint not found: {checkpoint_path}", warning_log)
        return None
    try:
        import torch

        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        low = state.get("low_prompt_features")
        high = state.get("high_prompt_features")
        if low is None or high is None:
            warn(f"Checkpoint does not contain low_prompt_features/high_prompt_features: {checkpoint_path}", warning_log)
            return None
        low_np = low.detach().cpu().float().numpy()
        high_np = high.detach().cpu().float().numpy()
        if low_np.ndim != 3 or high_np.ndim != 3:
            warn(f"Prompt feature buffers have invalid ranks: low={low_np.shape}, high={high_np.shape}", warning_log)
            return None
        return low_np, high_np
    except Exception as exc:
        warn(f"Failed to load prompt features from checkpoint {checkpoint_path}: {exc}", warning_log)
        return None


def build_prompt_features_with_biomedclip(prompt_path: Path, warning_log: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import sys
        import torch
        import ml_collections

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from models.model_RCE_MIL_BiomedCLIP import RCE_MIL_BiomedCLIP

        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.prototype_number = 16
        config.peps_tau = 0.1
        config.scale_mode = "dual"
        config.use_concept_prompt_pool = True
        config.concept_prompt_path = str(prompt_path)
        config.rce_use_logit_calibration = False
        config.rce_use_concept_prior = False
        config.rce_use_visual_residual = False
        config.rce_use_cross_scale_graph = False
        config.rce_logit_scale_init = 10.0
        config.class_names = None
        config.finetune_text_encoder = False
        model = RCE_MIL_BiomedCLIP(config=config, num_classes=2)
        low_np = model.low_prompt_features.detach().cpu().float().numpy()
        high_np = model.high_prompt_features.detach().cpu().float().numpy()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return low_np, high_np
    except Exception as exc:
        warn(f"Failed to build prompt features with BiomedCLIP/text encoder: {exc}", warning_log)
        return None


def load_prompt_features(
    checkpoint_path_raw: str,
    prompt_path: Path,
    warning_log: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
    source_info: dict[str, object] = {"source": None, "checkpoint_path": checkpoint_path_raw, "prompt_path": str(prompt_path)}
    if checkpoint_path_raw:
        checkpoint_path = resolve_path(checkpoint_path_raw)
        loaded = load_prompt_features_from_checkpoint(checkpoint_path, warning_log)
        if loaded is not None:
            source_info.update({"source": "checkpoint", "resolved_checkpoint_path": str(checkpoint_path)})
            return loaded[0], loaded[1], source_info

    loaded = build_prompt_features_with_biomedclip(prompt_path, warning_log)
    if loaded is not None:
        source_info.update({"source": "biomedclip_text_encoder"})
        return loaded[0], loaded[1], source_info

    source_info.update({"source": "unavailable"})
    return None, None, source_info


def low_patch_bbox(coord: np.ndarray, patch_size: float, coord_mode: str, scale_ratio: float) -> np.ndarray:
    x, y = float(coord[0]), float(coord[1])
    if coord_mode == "top_left":
        bbox = np.array([x, y, x + patch_size, y + patch_size], dtype=np.float64)
    else:
        half = patch_size / 2.0
        bbox = np.array([x - half, y - half, x + half, y + half], dtype=np.float64)
    return bbox * float(scale_ratio)


def coords_to_centers(coords: np.ndarray, patch_size: float, coord_mode: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    if coord_mode == "top_left":
        return coords + patch_size / 2.0
    return coords


def expand_bbox(bbox: np.ndarray, expand: float) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float64)
    center = (bbox[:2] + bbox[2:]) / 2.0
    half = (bbox[2:] - bbox[:2]) * float(expand) / 2.0
    return np.concatenate([center - half, center + half])


def compute_patch_scores(low_features: np.ndarray, low_prompt_features: np.ndarray, prompt_topk: int, margin_weight: float) -> dict[str, np.ndarray]:
    low_features = normalize_rows(low_features)
    prompt_features = normalize_rows(low_prompt_features.reshape(-1, low_prompt_features.shape[-1])).reshape(low_prompt_features.shape)
    sim = np.einsum("nd,cpd->ncp", low_features, prompt_features).astype(np.float32)

    concept_relevance = topmean(sim.reshape(sim.shape[0], -1), prompt_topk, axis=1)
    class_scores = topmean(sim, prompt_topk, axis=2)
    class_order = np.argsort(-class_scores, axis=1)
    top_class = class_order[:, 0]
    second_class = class_order[:, 1] if class_scores.shape[1] > 1 else class_order[:, 0]
    top_class_score = class_scores[np.arange(class_scores.shape[0]), top_class]
    second_class_score = class_scores[np.arange(class_scores.shape[0]), second_class]
    class_margin = top_class_score - second_class_score
    patch_score = concept_relevance + float(margin_weight) * class_margin

    flat_idx = np.argmax(sim.reshape(sim.shape[0], -1), axis=1)
    top_prompt_class = flat_idx // sim.shape[2]
    top_prompt_index = flat_idx % sim.shape[2]
    top_prompt_score = sim.reshape(sim.shape[0], -1)[np.arange(sim.shape[0]), flat_idx]

    return {
        "sim": sim,
        "low_features_norm": low_features,
        "concept_relevance": concept_relevance,
        "class_scores": class_scores,
        "top_class": top_class,
        "second_class": second_class,
        "top_class_score": top_class_score,
        "second_class_score": second_class_score,
        "class_margin": class_margin,
        "patch_score": patch_score,
        "top_prompt_class": top_prompt_class,
        "top_prompt_index": top_prompt_index,
        "top_prompt_score": top_prompt_score,
    }


def get_prompt_meta(prompt_metadata: dict[str, dict[int, list[dict[str, object]]]], class_id: int, prompt_index: int) -> dict[str, object]:
    items = prompt_metadata.get("low", {}).get(int(class_id), [])
    if 0 <= int(prompt_index) < len(items):
        return items[int(prompt_index)]
    return {
        "class_id": int(class_id),
        "class_name": FALLBACK_CLASS_NAMES[int(class_id)] if int(class_id) < len(FALLBACK_CLASS_NAMES) else str(class_id),
        "scale": "low",
        "concept_id": f"class_{class_id}_prompt_{prompt_index}",
        "concept_en": f"class_{class_id}_prompt_{prompt_index}",
        "prompt": "",
    }


def get_class_name(prompt_metadata: dict[str, dict[int, list[dict[str, object]]]], class_id: int) -> str:
    items = prompt_metadata.get("low", {}).get(int(class_id), []) or prompt_metadata.get("high", {}).get(int(class_id), [])
    for item in items:
        class_name = str(item.get("class_name", "")).strip()
        if class_name:
            return class_name
    if int(class_id) < len(FALLBACK_CLASS_NAMES):
        return FALLBACK_CLASS_NAMES[int(class_id)]
    return str(class_id)


def build_candidate_indices(
    sim: np.ndarray,
    patch_score: np.ndarray,
    candidate_top_l: int,
    top_g_concepts: int,
    per_concept_top_m: int,
    prompt_topk: int,
) -> dict[int, dict[str, object]]:
    n_patches = sim.shape[0]
    candidates: dict[int, dict[str, object]] = {}
    top_l = min(max(candidate_top_l, 0), n_patches)
    if top_l > 0:
        for rank, patch_idx in enumerate(np.argsort(-patch_score)[:top_l], start=1):
            record = candidates.setdefault(int(patch_idx), {"sources": [], "concept_hits": []})
            record["sources"].append(f"global_rank_{rank}")

    evidence = topmean(sim, prompt_topk, axis=0)
    flat_evidence = evidence.reshape(-1)
    top_g = min(max(top_g_concepts, 0), flat_evidence.size)
    for concept_rank, flat_idx in enumerate(np.argsort(-flat_evidence)[:top_g], start=1):
        class_id = int(flat_idx // sim.shape[2])
        prompt_idx = int(flat_idx % sim.shape[2])
        prompt_scores = sim[:, class_id, prompt_idx]
        top_m = min(max(per_concept_top_m, 0), n_patches)
        for local_rank, patch_idx in enumerate(np.argsort(-prompt_scores)[:top_m], start=1):
            record = candidates.setdefault(int(patch_idx), {"sources": [], "concept_hits": []})
            record["sources"].append(f"concept_rank_{concept_rank}_local_{local_rank}")
            record["concept_hits"].append(
                {
                    "class_id": class_id,
                    "prompt_index": prompt_idx,
                    "score": float(prompt_scores[patch_idx]),
                    "evidence": float(flat_evidence[flat_idx]),
                }
            )
    return candidates


def build_candidate_proposals(
    slide_id: str,
    candidates: dict[int, dict[str, object]],
    low_coords: np.ndarray,
    low_centers: np.ndarray,
    low_features_norm: np.ndarray,
    score_info: dict[str, np.ndarray],
    prompt_metadata: dict[str, dict[int, list[dict[str, object]]]],
    args: argparse.Namespace,
    coord_mode: str,
    scale_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    proposal_rows: list[dict[str, object]] = []
    proposals: list[dict[str, object]] = []
    patch_score = score_info["patch_score"]
    for proposal_id, patch_idx in enumerate(sorted(candidates.keys()), start=1):
        center = low_centers[patch_idx]
        distances = np.linalg.norm(low_centers - center.reshape(1, 2), axis=1)
        neighbor_indices = np.where(distances <= args.proposal_radius)[0]
        if neighbor_indices.size == 0:
            neighbor_indices = np.array([patch_idx], dtype=np.int64)
        neighbor_scores = patch_score[neighbor_indices]
        logits = neighbor_scores - np.max(neighbor_scores)
        weights = np.exp(logits)
        weights = weights / np.maximum(weights.sum(), 1e-8)

        proposal_feature = np.sum(low_features_norm[neighbor_indices] * weights[:, None], axis=0)
        proposal_coord = np.sum(low_coords[neighbor_indices] * weights[:, None], axis=0)
        proposal_center = np.sum(low_centers[neighbor_indices] * weights[:, None], axis=0)
        proposal_score = float(np.sum(neighbor_scores * weights))

        neighbor_bboxes = np.stack(
            [low_patch_bbox(low_coords[idx], args.low_patch_size, coord_mode, scale_ratio) for idx in neighbor_indices],
            axis=0,
        )
        proposal_bbox = np.array(
            [
                neighbor_bboxes[:, 0].min(),
                neighbor_bboxes[:, 1].min(),
                neighbor_bboxes[:, 2].max(),
                neighbor_bboxes[:, 3].max(),
            ],
            dtype=np.float64,
        )

        top_class = int(score_info["top_class"][patch_idx])
        top_prompt_index = int(score_info["top_prompt_index"][patch_idx])
        top_meta = get_prompt_meta(prompt_metadata, top_class, top_prompt_index)
        concept_hits = candidates[patch_idx].get("concept_hits", [])
        if concept_hits:
            best_hit = max(concept_hits, key=lambda item: item["score"])
            proposal_class = int(best_hit["class_id"])
            proposal_prompt_index = int(best_hit["prompt_index"])
            proposal_meta = get_prompt_meta(prompt_metadata, proposal_class, proposal_prompt_index)
            proposal_concept = str(proposal_meta.get("concept_en") or proposal_meta.get("concept_id"))
        else:
            proposal_class = top_class
            proposal_prompt_index = top_prompt_index
            proposal_concept = str(top_meta.get("concept_en") or top_meta.get("concept_id"))

        row = {
            "slide_id": slide_id,
            "proposal_id": proposal_id,
            "source_type": "|".join(candidates[patch_idx].get("sources", [])),
            "center_patch_index": int(patch_idx),
            "center_coord_x": float(low_coords[patch_idx, 0]),
            "center_coord_y": float(low_coords[patch_idx, 1]),
            "proposal_coord_x": float(proposal_coord[0]),
            "proposal_coord_y": float(proposal_coord[1]),
            "proposal_center_x": float(proposal_center[0]),
            "proposal_center_y": float(proposal_center[1]),
            "proposal_bbox_x0": float(proposal_bbox[0]),
            "proposal_bbox_y0": float(proposal_bbox[1]),
            "proposal_bbox_x1": float(proposal_bbox[2]),
            "proposal_bbox_y1": float(proposal_bbox[3]),
            "proposal_score": proposal_score,
            "center_patch_score": float(patch_score[patch_idx]),
            "neighbor_count": int(neighbor_indices.size),
            "proposal_class": proposal_class,
            "proposal_class_name": get_class_name(prompt_metadata, proposal_class),
            "proposal_prompt_index": proposal_prompt_index,
            "proposal_concept": proposal_concept,
            "proposal_prompt": str(get_prompt_meta(prompt_metadata, proposal_class, proposal_prompt_index).get("prompt", "")),
        }
        proposal_rows.append(row)
        proposals.append(
            {
                **row,
                "proposal_feature": proposal_feature,
                "proposal_bbox": proposal_bbox,
                "proposal_center": proposal_center,
            }
        )
    return proposal_rows, proposals


def select_anchors_nms(proposals: list[dict[str, object]], num_anchors: int, nms_radius: float) -> list[dict[str, object]]:
    ordered = sorted(proposals, key=lambda item: float(item["proposal_score"]), reverse=True)
    selected: list[dict[str, object]] = []
    for proposal in ordered:
        center = np.asarray(proposal["proposal_center"], dtype=np.float64)
        suppressed_by = None
        for selected_anchor in selected:
            dist = float(np.linalg.norm(center - np.asarray(selected_anchor["proposal_center"], dtype=np.float64)))
            if dist < nms_radius:
                suppressed_by = selected_anchor["anchor_rank"]
                break
        if suppressed_by is not None:
            continue
        anchor = dict(proposal)
        anchor["anchor_rank"] = len(selected) + 1
        anchor["nms_suppressed_by_rank"] = None
        anchor["nms_radius"] = float(nms_radius)
        selected.append(anchor)
        if len(selected) >= num_anchors:
            break
    return selected


def match_high_children(
    anchor: dict[str, object],
    high_coords: np.ndarray,
    high_centers: np.ndarray,
    bbox_expand: float,
    high_child_top_m: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bbox = expand_bbox(np.asarray(anchor["proposal_bbox"], dtype=np.float64), bbox_expand)
    inside = (
        (high_centers[:, 0] >= bbox[0])
        & (high_centers[:, 0] <= bbox[2])
        & (high_centers[:, 1] >= bbox[1])
        & (high_centers[:, 1] <= bbox[3])
    )
    child_indices = np.where(inside)[0]
    anchor_center = np.asarray(anchor["proposal_center"], dtype=np.float64)
    if child_indices.size:
        distances = np.linalg.norm(high_centers[child_indices] - anchor_center.reshape(1, 2), axis=1)
        order = np.argsort(distances)
        sorted_child_indices = child_indices[order]
        sorted_distances = distances[order]
        used_child_indices = sorted_child_indices[:high_child_top_m]
        used_distances = sorted_distances[:high_child_top_m]
    else:
        distances = np.zeros((0,), dtype=np.float64)
        sorted_child_indices = np.zeros((0,), dtype=np.int64)
        used_child_indices = np.zeros((0,), dtype=np.int64)
        used_distances = np.zeros((0,), dtype=np.float64)

    stats = {
        "slide_id": anchor["slide_id"],
        "anchor_rank": anchor["anchor_rank"],
        "bbox_expand": float(bbox_expand),
        "raw_child_count": int(child_indices.size),
        "used_child_count": int(used_child_indices.size),
        "empty_child_flag": bool(child_indices.size == 0),
        "mean_child_distance": float(distances.mean()) if distances.size else None,
        "min_child_distance": float(distances.min()) if distances.size else None,
        "max_child_distance": float(distances.max()) if distances.size else None,
        "used_mean_child_distance": float(used_distances.mean()) if used_distances.size else None,
        "anchor_bbox_expand_x0": float(bbox[0]),
        "anchor_bbox_expand_y0": float(bbox[1]),
        "anchor_bbox_expand_x1": float(bbox[2]),
        "anchor_bbox_expand_y1": float(bbox[3]),
    }
    pair_rows = []
    for rank, child_idx in enumerate(used_child_indices, start=1):
        child_center = high_centers[child_idx]
        child_coord = high_coords[child_idx]
        pair_rows.append(
            {
                "slide_id": anchor["slide_id"],
                "anchor_rank": anchor["anchor_rank"],
                "bbox_expand": float(bbox_expand),
                "anchor_score": float(anchor["proposal_score"]),
                "anchor_coord_x": float(anchor["proposal_coord_x"]),
                "anchor_coord_y": float(anchor["proposal_coord_y"]),
                "anchor_center_x": float(anchor_center[0]),
                "anchor_center_y": float(anchor_center[1]),
                "anchor_bbox_x0": float(anchor["proposal_bbox_x0"]),
                "anchor_bbox_y0": float(anchor["proposal_bbox_y0"]),
                "anchor_bbox_x1": float(anchor["proposal_bbox_x1"]),
                "anchor_bbox_y1": float(anchor["proposal_bbox_y1"]),
                "high_child_rank": rank,
                "high_child_index": int(child_idx),
                "high_child_coord_x": float(child_coord[0]),
                "high_child_coord_y": float(child_coord[1]),
                "high_child_center_x": float(child_center[0]),
                "high_child_center_y": float(child_center[1]),
                "distance_to_anchor": float(np.linalg.norm(child_center - anchor_center)),
            }
        )
    return stats, pair_rows


def summarize_anchor_distribution(slide_id: str, anchors: list[dict[str, object]], low_coords: np.ndarray) -> dict[str, object]:
    if not anchors:
        return {
            "slide_id": slide_id,
            "selected_anchor_count": 0,
            "anchor_pair_distance_mean": None,
            "anchor_pair_distance_min": None,
            "anchor_pair_distance_max": None,
            "anchor_coverage_span_x": None,
            "anchor_coverage_span_y": None,
            "slide_span_x": float(low_coords[:, 0].max() - low_coords[:, 0].min()) if len(low_coords) else None,
            "slide_span_y": float(low_coords[:, 1].max() - low_coords[:, 1].min()) if len(low_coords) else None,
            "coverage_ratio_x": None,
            "coverage_ratio_y": None,
            "anchor_class_distribution": "{}",
            "anchor_concept_distribution": "{}",
            "selection_note": "no anchors selected",
        }
    centers = np.stack([np.asarray(anchor["proposal_center"], dtype=np.float64) for anchor in anchors], axis=0)
    pair_distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pair_distances.append(float(np.linalg.norm(centers[i] - centers[j])))
    pair_distances_np = np.asarray(pair_distances, dtype=np.float64)
    anchor_span = centers.max(axis=0) - centers.min(axis=0)
    slide_span = low_coords.max(axis=0) - low_coords.min(axis=0) if len(low_coords) else np.array([np.nan, np.nan])
    class_counter = Counter(str(anchor["proposal_class_name"]) for anchor in anchors)
    concept_counter = Counter(str(anchor["proposal_concept"]) for anchor in anchors)
    return {
        "slide_id": slide_id,
        "selected_anchor_count": int(len(anchors)),
        "anchor_pair_distance_mean": float(pair_distances_np.mean()) if pair_distances_np.size else None,
        "anchor_pair_distance_min": float(pair_distances_np.min()) if pair_distances_np.size else None,
        "anchor_pair_distance_max": float(pair_distances_np.max()) if pair_distances_np.size else None,
        "anchor_coverage_span_x": float(anchor_span[0]),
        "anchor_coverage_span_y": float(anchor_span[1]),
        "slide_span_x": float(slide_span[0]) if np.isfinite(slide_span[0]) else None,
        "slide_span_y": float(slide_span[1]) if np.isfinite(slide_span[1]) else None,
        "coverage_ratio_x": float(anchor_span[0] / max(slide_span[0], 1e-6)) if np.isfinite(slide_span[0]) else None,
        "coverage_ratio_y": float(anchor_span[1] / max(slide_span[1], 1e-6)) if np.isfinite(slide_span[1]) else None,
        "anchor_class_distribution": json.dumps(dict(class_counter), ensure_ascii=False),
        "anchor_concept_distribution": json.dumps(dict(concept_counter), ensure_ascii=False),
        "unique_anchor_classes": int(len(class_counter)),
        "unique_anchor_concepts": int(len(concept_counter)),
        "selection_note": "ok" if len(anchors) >= 16 else f"selected fewer than requested: {len(anchors)}",
    }


def decide_recommendation(
    anchor_df: pd.DataFrame,
    child_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    bbox_values: list[float],
    args: argparse.Namespace,
) -> dict[str, object]:
    if anchor_df.empty or child_df.empty:
        return {
            "recommend_bbox_expand": None,
            "recommend_num_anchors": args.num_anchors,
            "recommend_num_high_children": args.high_child_top_m,
            "enter_step43": False,
            "reason": "No anchors or child matches were produced.",
        }

    bbox_rows = []
    for bbox_expand in bbox_values:
        group = child_df[child_df["bbox_expand"] == bbox_expand]
        if group.empty:
            continue
        bbox_rows.append(
            {
                "bbox_expand": float(bbox_expand),
                "empty_anchor_ratio": float(group["empty_child_flag"].astype(float).mean()),
                "median_raw_child_count": float(group["raw_child_count"].median()),
                "median_used_child_count": float(group["used_child_count"].median()),
                "mean_raw_child_count": float(group["raw_child_count"].mean()),
                "mean_used_child_count": float(group["used_child_count"].mean()),
            }
        )
    bbox_df = pd.DataFrame(bbox_rows)
    if bbox_df.empty:
        return {
            "recommend_bbox_expand": None,
            "recommend_num_anchors": args.num_anchors,
            "recommend_num_high_children": args.high_child_top_m,
            "enter_step43": False,
            "reason": "No bbox_expand child summary was available.",
        }
    rank_df = bbox_df.assign(
        score=lambda df: (
            df["empty_anchor_ratio"] * 10.0
            - np.minimum(df["median_used_child_count"], args.high_child_top_m) / max(args.high_child_top_m, 1)
            + np.abs(df["bbox_expand"] - 2.0) * 0.05
        )
    ).sort_values("score", kind="mergesort")
    best = rank_df.iloc[0].to_dict()

    selected_counts = dist_df["selected_anchor_count"].dropna().astype(float)
    median_anchor_count = float(selected_counts.median()) if not selected_counts.empty else 0.0
    count_ok = median_anchor_count >= max(1, args.num_anchors * 0.75)
    diversity_ok = bool((dist_df.get("unique_anchor_concepts", pd.Series(dtype=float)).fillna(0).astype(float).median() >= 2))
    spread_ok = bool(
        (
            dist_df.get("coverage_ratio_x", pd.Series(dtype=float)).fillna(0).astype(float).median()
            + dist_df.get("coverage_ratio_y", pd.Series(dtype=float)).fillna(0).astype(float).median()
        )
        / 2.0
        >= 0.1
    )
    child_ok = bool(best["empty_anchor_ratio"] <= 0.35 and (best["median_used_child_count"] >= 4 or best["median_raw_child_count"] >= 4))
    enter_step43 = bool(count_ok and diversity_ok and spread_ok and child_ok)
    reason = (
        f"median_anchor_count={median_anchor_count:.2f}, "
        f"best_empty_anchor_ratio={best['empty_anchor_ratio']:.4f}, "
        f"best_median_used_child_count={best['median_used_child_count']:.2f}, "
        f"diversity_ok={diversity_ok}, spread_ok={spread_ok}."
    )
    return {
        "recommend_bbox_expand": float(best["bbox_expand"]),
        "recommend_num_anchors": int(args.num_anchors),
        "recommend_num_high_children": int(args.high_child_top_m),
        "enter_step43": enter_step43,
        "reason": reason,
        "bbox_summary": bbox_rows,
    }


def write_failure_outputs(output_dir: Path, args: argparse.Namespace, manifest: dict[str, object], reason: str, warning_log: list[str]) -> None:
    columns_by_file = {
        "cgsp_patch_scores.csv": ["slide_id", "patch_index", "coord_x", "coord_y", "patch_score"],
        "cgsp_candidate_proposals.csv": ["slide_id", "proposal_id", "proposal_score"],
        "cgsp_selected_anchors.csv": ["slide_id", "anchor_rank", "anchor_score"],
        "cgsp_anchor_child_match_stats.csv": ["slide_id", "anchor_rank", "bbox_expand", "raw_child_count"],
        "cgsp_anchor_distribution_summary.csv": ["slide_id", "selected_anchor_count"],
        "cgsp_example_anchor_child_pairs.csv": ["slide_id", "anchor_rank", "high_child_rank"],
    }
    for filename, columns in columns_by_file.items():
        pd.DataFrame(columns=columns).to_csv(output_dir / filename, index=False)
    lines = [
        "# Step42 CGSP Anchor Selection Audit",
        "",
        "## Status",
        "- Step42 did not complete.",
        f"- Reason: {reason}",
        "",
        "## Prompt Feature Source",
        f"- checkpoint_path: `{args.checkpoint_path}`",
        f"- concept_prompt_path: `{args.concept_prompt_path}`",
        "",
        "## Warnings",
    ]
    lines.extend([f"- {msg}" for msg in warning_log] if warning_log else ["- None"])
    (output_dir / "cgsp_anchor_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.update(
        {
            "status": "failed",
            "failure_reason": reason,
            "warning_count": len(warning_log),
            "warnings": warning_log,
            "processed_slide_count": 0,
            "failed_slide_count": 0,
            "recommend_bbox_expand": None,
            "enter_step43": False,
        }
    )
    (output_dir / "stage42_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    stage41_settings: dict[str, object],
    prompt_source_info: dict[str, object],
    slide_source_info: dict[str, object],
    manifest: dict[str, object],
    patch_df: pd.DataFrame,
    proposal_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    child_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    recommendation: dict[str, object],
    warning_log: list[str],
) -> None:
    patch_summary = pd.DataFrame(
        [
            {
                "metric": col,
                "mean": format_float(patch_df[col].mean()) if col in patch_df else "N/A",
                "median": format_float(patch_df[col].median()) if col in patch_df else "N/A",
                "p90": format_float(patch_df[col].quantile(0.9)) if col in patch_df else "N/A",
                "p95": format_float(patch_df[col].quantile(0.95)) if col in patch_df else "N/A",
            }
            for col in ["patch_score", "concept_relevance", "class_margin", "top_prompt_score"]
        ]
    )
    proposal_summary = proposal_df.groupby("slide_id").size().describe().reset_index()
    proposal_summary.columns = ["stat", "candidate_proposals"]
    anchor_summary = anchor_df.groupby("slide_id").size().describe().reset_index()
    anchor_summary.columns = ["stat", "selected_anchors"]

    class_dist = anchor_df["top_class_name"].value_counts().reset_index()
    class_dist.columns = ["top_class", "count"]
    concept_dist = anchor_df["top_concept"].value_counts().head(12).reset_index()
    concept_dist.columns = ["top_concept", "count"]

    bbox_summary = pd.DataFrame(recommendation.get("bbox_summary", []))
    bbox_report = bbox_summary.copy()
    for col in ["empty_anchor_ratio", "median_raw_child_count", "median_used_child_count", "mean_raw_child_count", "mean_used_child_count"]:
        if col in bbox_report:
            bbox_report[col] = bbox_report[col].map(format_float)

    spread_report = pd.DataFrame(
        [
            {
                "metric": "selected_anchor_count",
                "mean": format_float(dist_df["selected_anchor_count"].mean()),
                "median": format_float(dist_df["selected_anchor_count"].median()),
                "min": format_float(dist_df["selected_anchor_count"].min()),
                "max": format_float(dist_df["selected_anchor_count"].max()),
            },
            {
                "metric": "anchor_pair_distance_mean",
                "mean": format_float(dist_df["anchor_pair_distance_mean"].mean()),
                "median": format_float(dist_df["anchor_pair_distance_mean"].median()),
                "min": format_float(dist_df["anchor_pair_distance_min"].min()),
                "max": format_float(dist_df["anchor_pair_distance_max"].max()),
            },
            {
                "metric": "coverage_ratio_x",
                "mean": format_float(dist_df["coverage_ratio_x"].mean()),
                "median": format_float(dist_df["coverage_ratio_x"].median()),
                "min": format_float(dist_df["coverage_ratio_x"].min()),
                "max": format_float(dist_df["coverage_ratio_x"].max()),
            },
            {
                "metric": "coverage_ratio_y",
                "mean": format_float(dist_df["coverage_ratio_y"].mean()),
                "median": format_float(dist_df["coverage_ratio_y"].median()),
                "min": format_float(dist_df["coverage_ratio_y"].min()),
                "max": format_float(dist_df["coverage_ratio_y"].max()),
            },
        ]
    )

    enter_text = "建议进入 Step43 HCRC-Light Smoke。" if recommendation.get("enter_step43") else "暂不建议直接进入 Step43，建议先调参。"
    risk_lines = []
    best_bbox = recommendation.get("recommend_bbox_expand")
    if best_bbox is not None:
        best_child = child_df[child_df["bbox_expand"] == best_bbox]
        empty_ratio = float(best_child["empty_child_flag"].astype(float).mean()) if not best_child.empty else 1.0
        median_used = float(best_child["used_child_count"].median()) if not best_child.empty else 0.0
        if empty_ratio > 0.35:
            risk_lines.append(f"empty-anchor ratio 偏高：推荐 bbox_expand={best_bbox} 时为 {empty_ratio:.4f}。")
        if median_used < 4:
            risk_lines.append(f"median used_child_count 偏低：推荐 bbox_expand={best_bbox} 时为 {median_used:.2f}。")
    if dist_df["selected_anchor_count"].median() < args.num_anchors:
        risk_lines.append("部分 slide 选不满目标 anchor 数，需要关注 candidate_top_l / nms_radius。")
    if dist_df["coverage_ratio_x"].median() < 0.1 and dist_df["coverage_ratio_y"].median() < 0.1:
        risk_lines.append("anchors 可能过度集中，建议调小 proposal_radius 或 nms_radius / 增强 concept coverage。")
    if not risk_lines:
        risk_lines.append("当前审计未发现阻断 Step43 的 anchor 覆盖问题，但 Step43 仍需做小规模 smoke。")

    lines = [
        "# Step42 CGSP Anchor Selection Audit",
        "",
        "## Step42 目的",
        "- 验证 Concept-Guided Spatial Proposal 是否能从 low patch 层面选出合理 spatial anchors。",
        "- 检查 selected anchors 在不同 bbox_expand 下能否稳定匹配 high child patches，为 Step43 HCRC-Light 做准备。",
        "- 本步骤不训练模型，不修改模型前向逻辑。",
        "",
        "## Step41 推荐设置读取情况",
        f"- loaded_from_stage41: `{stage41_settings['loaded_from_stage41']}`",
        f"- coord_mode: `{stage41_settings['coord_mode']}`",
        f"- scale_ratio: `{stage41_settings['scale_ratio']}`",
        "",
        "## 输入路径",
        f"- low directory: `{Path(args.data_root_dir) / args.data_folder_s}`",
        f"- high directory: `{Path(args.data_root_dir) / args.data_folder_l}`",
        f"- slide source: `{slide_source_info['source']}` from `{slide_source_info['source_path']}`",
        f"- concept/checkpoint source: `{prompt_source_info.get('source')}`",
        f"- checkpoint_path: `{prompt_source_info.get('resolved_checkpoint_path') or args.checkpoint_path}`",
        f"- concept_prompt_path: `{prompt_source_info.get('prompt_path')}`",
        "",
        "## 处理概况",
        f"- requested slides: `{slide_source_info['selected_slide_count']}`",
        f"- processed slides: `{manifest['processed_slide_count']}`",
        f"- failed slides: `{manifest['failed_slide_count']}`",
        f"- warning count: `{len(warning_log)}`",
        "",
        "## patch score 分布",
        markdown_table(patch_summary),
        "",
        "## proposal 生成数量统计",
        markdown_table(proposal_summary),
        "",
        "## selected anchors 数量统计",
        markdown_table(anchor_summary),
        "",
        "## anchor 空间分散性统计",
        markdown_table(spread_report),
        "",
        "## anchor top_class 分布",
        markdown_table(class_dist),
        "",
        "## anchor top_concept 分布",
        markdown_table(concept_dist),
        "",
        "## 不同 bbox_expand 下 high child count 分布",
        markdown_table(bbox_report),
        "",
        "## 推荐给 HCRC-Light 的设置",
        f"- bbox_expand: `{recommendation.get('recommend_bbox_expand')}`",
        f"- num_anchors: `{recommendation.get('recommend_num_anchors')}`",
        f"- num_high_children: `{recommendation.get('recommend_num_high_children')}`",
        f"- decision: {enter_text}",
        f"- decision reason: {recommendation.get('reason')}",
        "",
        "## 风险说明",
    ]
    lines.extend([f"- {line}" for line in risk_lines])
    if warning_log:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {msg}" for msg in warning_log[:80]])
    (output_dir / "cgsp_anchor_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warning_log: list[str] = []
    bbox_expand_values = parse_float_list(args.bbox_expand_values)
    stage41_manifest = resolve_path(args.stage41_manifest)
    stage41_settings = load_stage41_settings(stage41_manifest, args, warning_log)
    coord_mode = str(stage41_settings["coord_mode"])
    scale_ratio = float(stage41_settings["scale_ratio"])

    low_dir = resolve_path(Path(args.data_root_dir) / args.data_folder_s)
    high_dir = resolve_path(Path(args.data_root_dir) / args.data_folder_l)
    csv_path = resolve_path(args.csv_path)
    split_dir = resolve_path(args.split_dir)
    concept_prompt_path = resolve_path(args.concept_prompt_path)

    manifest: dict[str, object] = {
        "step": "Step42 CGSP Anchor Selection Audit",
        "status": "started",
        "parameters": vars(args),
        "input_paths": {
            "low_dir": str(low_dir),
            "high_dir": str(high_dir),
            "csv_path": str(csv_path),
            "split_dir": str(split_dir),
            "stage41_manifest": str(stage41_manifest),
            "concept_prompt_path": str(concept_prompt_path),
            "checkpoint_path": args.checkpoint_path,
        },
        "output_paths": {
            "cgsp_patch_scores_csv": str(output_dir / "cgsp_patch_scores.csv"),
            "cgsp_candidate_proposals_csv": str(output_dir / "cgsp_candidate_proposals.csv"),
            "cgsp_selected_anchors_csv": str(output_dir / "cgsp_selected_anchors.csv"),
            "cgsp_anchor_child_match_stats_csv": str(output_dir / "cgsp_anchor_child_match_stats.csv"),
            "cgsp_anchor_distribution_summary_csv": str(output_dir / "cgsp_anchor_distribution_summary.csv"),
            "cgsp_example_anchor_child_pairs_csv": str(output_dir / "cgsp_example_anchor_child_pairs.csv"),
            "cgsp_anchor_selection_report_md": str(output_dir / "cgsp_anchor_selection_report.md"),
            "stage42_manifest_json": str(output_dir / "stage42_manifest.json"),
        },
        "stage41_settings": stage41_settings,
    }

    prompt_metadata = load_prompt_metadata(concept_prompt_path, warning_log)
    low_prompt_features, high_prompt_features, prompt_source_info = load_prompt_features(
        args.checkpoint_path,
        concept_prompt_path,
        warning_log,
    )
    manifest["prompt_source"] = prompt_source_info
    if low_prompt_features is None or high_prompt_features is None:
        reason = "Prompt features unavailable; checkpoint buffers missing and BiomedCLIP/text encoder fallback failed."
        write_failure_outputs(output_dir, args, manifest, reason, warning_log)
        print(f"Step42 did not complete: {reason}")
        return

    slide_ids, slide_source_info = load_slide_ids(csv_path, split_dir, args.fold, args.split, args.max_slides, warning_log)
    manifest["slide_source"] = slide_source_info

    patch_rows: list[dict[str, object]] = []
    proposal_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    processed_slide_ids: list[str] = []

    for slide_id in slide_ids:
        low_path = low_dir / f"{slide_id}.h5"
        high_path = high_dir / f"{slide_id}.h5"
        if not low_path.is_file() or not high_path.is_file():
            failed_rows.append({"slide_id": slide_id, "error": "missing low/high h5"})
            warn(f"Missing low/high h5 for slide {slide_id}", warning_log)
            continue
        try:
            low_features, low_coords, _ = read_h5_features_coords(low_path)
            high_features, high_coords, _ = read_h5_features_coords(high_path)
            if low_features.shape[1] != low_prompt_features.shape[-1]:
                raise ValueError(
                    f"low feature dim mismatch: features={low_features.shape[1]}, prompts={low_prompt_features.shape[-1]}"
                )
            score_info = compute_patch_scores(
                low_features=low_features,
                low_prompt_features=low_prompt_features,
                prompt_topk=args.prompt_topk,
                margin_weight=args.margin_weight,
            )
            low_centers = coords_to_centers(low_coords, args.low_patch_size, coord_mode) * scale_ratio
            high_centers = coords_to_centers(high_coords, args.high_patch_size, coord_mode)

            for patch_idx in range(low_coords.shape[0]):
                top_class = int(score_info["top_class"][patch_idx])
                top_prompt_index = int(score_info["top_prompt_index"][patch_idx])
                meta = get_prompt_meta(prompt_metadata, top_class, top_prompt_index)
                patch_rows.append(
                    {
                        "slide_id": slide_id,
                        "patch_index": patch_idx,
                        "coord_x": float(low_coords[patch_idx, 0]),
                        "coord_y": float(low_coords[patch_idx, 1]),
                        "center_x": float(low_centers[patch_idx, 0]),
                        "center_y": float(low_centers[patch_idx, 1]),
                        "patch_score": float(score_info["patch_score"][patch_idx]),
                        "concept_relevance": float(score_info["concept_relevance"][patch_idx]),
                        "class_margin": float(score_info["class_margin"][patch_idx]),
                        "top_class": top_class,
                        "top_class_name": get_class_name(prompt_metadata, top_class),
                        "top_prompt_index": top_prompt_index,
                        "top_prompt_score": float(score_info["top_prompt_score"][patch_idx]),
                        "top_class_score": float(score_info["top_class_score"][patch_idx]),
                        "second_class_score": float(score_info["second_class_score"][patch_idx]),
                        "top_concept_id": str(meta.get("concept_id", "")),
                        "top_concept": str(meta.get("concept_en", "")),
                        "top_prompt": str(meta.get("prompt", "")),
                    }
                )

            candidates = build_candidate_indices(
                sim=score_info["sim"],
                patch_score=score_info["patch_score"],
                candidate_top_l=args.candidate_top_l,
                top_g_concepts=args.top_g_concepts,
                per_concept_top_m=args.per_concept_top_m,
                prompt_topk=args.prompt_topk,
            )
            slide_proposal_rows, proposals = build_candidate_proposals(
                slide_id=slide_id,
                candidates=candidates,
                low_coords=low_coords,
                low_centers=low_centers,
                low_features_norm=score_info["low_features_norm"],
                score_info=score_info,
                prompt_metadata=prompt_metadata,
                args=args,
                coord_mode=coord_mode,
                scale_ratio=scale_ratio,
            )
            proposal_rows.extend(slide_proposal_rows)
            selected = select_anchors_nms(proposals, args.num_anchors, args.nms_radius)
            for anchor in selected:
                top_class = int(anchor["proposal_class"])
                anchor_rows.append(
                    {
                        "slide_id": slide_id,
                        "anchor_rank": int(anchor["anchor_rank"]),
                        "anchor_score": float(anchor["proposal_score"]),
                        "anchor_coord_x": float(anchor["proposal_coord_x"]),
                        "anchor_coord_y": float(anchor["proposal_coord_y"]),
                        "anchor_center_x": float(anchor["proposal_center_x"]),
                        "anchor_center_y": float(anchor["proposal_center_y"]),
                        "anchor_bbox_x0": float(anchor["proposal_bbox_x0"]),
                        "anchor_bbox_y0": float(anchor["proposal_bbox_y0"]),
                        "anchor_bbox_x1": float(anchor["proposal_bbox_x1"]),
                        "anchor_bbox_y1": float(anchor["proposal_bbox_y1"]),
                        "neighbor_count": int(anchor["neighbor_count"]),
                        "top_class": top_class,
                        "top_class_name": get_class_name(prompt_metadata, top_class),
                        "top_concept": str(anchor["proposal_concept"]),
                        "top_prompt_index": int(anchor["proposal_prompt_index"]),
                        "center_patch_index": int(anchor["center_patch_index"]),
                        "nms_radius": float(args.nms_radius),
                        "nms_suppressed_by_rank": anchor.get("nms_suppressed_by_rank"),
                        "selection_note": "ok" if len(selected) >= args.num_anchors else "fewer_than_requested",
                    }
                )
                for bbox_expand in bbox_expand_values:
                    stats, pairs = match_high_children(anchor, high_coords, high_centers, bbox_expand, args.high_child_top_m)
                    child_rows.append(stats)
                    if len(example_rows) < args.num_examples:
                        example_rows.extend(pairs[: max(0, args.num_examples - len(example_rows))])
            dist_rows.append(summarize_anchor_distribution(slide_id, selected, low_coords))
            processed_slide_ids.append(slide_id)
        except Exception as exc:
            failed_rows.append({"slide_id": slide_id, "error": str(exc)})
            warn(f"Failed to process slide {slide_id}: {exc}", warning_log)

    patch_df = pd.DataFrame(patch_rows)
    proposal_df = pd.DataFrame(proposal_rows)
    anchor_df = pd.DataFrame(anchor_rows)
    child_df = pd.DataFrame(child_rows)
    dist_df = pd.DataFrame(dist_rows)
    example_df = pd.DataFrame(example_rows)

    recommendation = decide_recommendation(anchor_df, child_df, dist_df, bbox_expand_values, args)

    patch_df.to_csv(output_dir / "cgsp_patch_scores.csv", index=False)
    proposal_df.to_csv(output_dir / "cgsp_candidate_proposals.csv", index=False)
    anchor_df.to_csv(output_dir / "cgsp_selected_anchors.csv", index=False)
    child_df.to_csv(output_dir / "cgsp_anchor_child_match_stats.csv", index=False)
    dist_df.to_csv(output_dir / "cgsp_anchor_distribution_summary.csv", index=False)
    example_df.to_csv(output_dir / "cgsp_example_anchor_child_pairs.csv", index=False)

    manifest.update(
        {
            "status": "ok",
            "processed_slide_count": len(processed_slide_ids),
            "failed_slide_count": len(failed_rows),
            "processed_slide_ids": processed_slide_ids,
            "failed_slides": failed_rows,
            "warning_count": len(warning_log),
            "warnings": warning_log,
            "recommend_bbox_expand": recommendation.get("recommend_bbox_expand"),
            "recommend_num_anchors": recommendation.get("recommend_num_anchors"),
            "recommend_num_high_children": recommendation.get("recommend_num_high_children"),
            "enter_step43": recommendation.get("enter_step43"),
            "recommendation_reason": recommendation.get("reason"),
            "bbox_summary": recommendation.get("bbox_summary", []),
        }
    )
    write_report(
        output_dir=output_dir,
        args=args,
        stage41_settings=stage41_settings,
        prompt_source_info=prompt_source_info,
        slide_source_info=slide_source_info,
        manifest=manifest,
        patch_df=patch_df,
        proposal_df=proposal_df,
        anchor_df=anchor_df,
        child_df=child_df,
        dist_df=dist_df,
        recommendation=recommendation,
        warning_log=warning_log,
    )
    (output_dir / "stage42_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Step42 CGSP anchor selection audit complete.")
    print(f"Processed slides: {len(processed_slide_ids)}")
    print(f"Failed slides: {len(failed_rows)}")
    print(f"Recommended bbox_expand: {recommendation.get('recommend_bbox_expand')}")
    print(f"Enter Step43: {recommendation.get('enter_step43')}")


if __name__ == "__main__":
    main()
