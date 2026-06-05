#!/usr/bin/env python3
"""Inspect Step19 CSG smoke artifacts without modifying training/model code."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_STAGE18_DIR = "results_stage9/stage18_cross_scale_evidence_graph_fold0"
DEFAULT_CONCEPT_PROMPT_PATH = "dataset_csv/private_lung_concept_prompt_pool_stage2_core12.json"
DEFAULT_OUT_DIR = "results_stage20/stage20_csg_smoke_inspection"
DEFAULT_CKPT_CANDIDATES = [
    "results_stage19/rce_v4_csg_smoke_s1/s_0_checkpoint.pt",
    "results_stage19/rce_v4_csg_smoke_s1/s_1_checkpoint.pt",
    "results_stage19/rce_v4_csg_smoke_s1/checkpoint.pt",
]


def parse_args():
    root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Inspect Step19 CSG smoke checkpoint artifacts.")
    parser.add_argument("--root", type=str, default=str(root_default))
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--stage18_dir", type=str, default=DEFAULT_STAGE18_DIR)
    parser.add_argument("--concept_prompt_path", type=str, default=DEFAULT_CONCEPT_PROMPT_PATH)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top_k_pairs", type=int, default=20)
    return parser.parse_args()


def resolve_path(root, path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return root / path


def append_warning(warnings, message):
    text = f"[Warning] {message}"
    print(text)
    warnings.append(message)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_state_dict(root, ckpt_path_arg, warnings):
    tried_paths = []
    candidate_paths = []
    if ckpt_path_arg:
        candidate_paths.append(resolve_path(root, ckpt_path_arg))
    else:
        for rel_path in DEFAULT_CKPT_CANDIDATES:
            candidate_paths.append(resolve_path(root, rel_path))

    for path in candidate_paths:
        tried_paths.append(str(path))
        if not path.exists():
            continue
        try:
            checkpoint_obj = safe_torch_load(path)
        except Exception as exc:
            append_warning(warnings, f"Failed to load checkpoint: {path} ({exc})")
            return None, None, tried_paths

        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj and isinstance(
            checkpoint_obj["state_dict"], dict
        ):
            return checkpoint_obj["state_dict"], path, tried_paths
        if hasattr(checkpoint_obj, "keys"):
            return checkpoint_obj, path, tried_paths

        append_warning(warnings, f"Unsupported checkpoint object type at {path}: {type(checkpoint_obj).__name__}")
        return None, path, tried_paths

    append_warning(warnings, f"No checkpoint found in candidates: {tried_paths}")
    return None, None, tried_paths


def find_state_value(state_dict, candidate_keys):
    if state_dict is None:
        return None, None
    for key in candidate_keys:
        if key in state_dict:
            return state_dict[key], key
    return None, None


def load_concept_prompt_metadata(path, warnings):
    grouped = {"low": {}, "high": {}}
    class_name_by_id = {}
    if not path.exists():
        append_warning(warnings, f"Concept prompt JSON not found: {path}")
        return grouped, class_name_by_id

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        append_warning(warnings, f"Failed to parse concept prompt JSON: {path} ({exc})")
        return grouped, class_name_by_id

    if isinstance(obj, dict):
        prompt_items = obj.get("prompts", [])
    elif isinstance(obj, list):
        prompt_items = obj
    else:
        append_warning(warnings, f"Unexpected concept prompt JSON type: {type(obj).__name__}")
        return grouped, class_name_by_id

    for item in prompt_items:
        if not bool(item.get("use_in_stage2", True)):
            continue

        try:
            class_id = int(item.get("class_id"))
        except Exception:
            continue

        scale = str(item.get("scale", "")).strip().lower()
        if scale not in grouped:
            continue

        if class_id not in grouped[scale]:
            grouped[scale][class_id] = []

        class_name = str(item.get("class_name", "")).strip()
        if class_name:
            class_name_by_id[class_id] = class_name

        grouped[scale][class_id].append(
            {
                "concept_id": str(item.get("concept_id", "")).strip(),
                "concept_en": str(item.get("concept_en", "")).strip(),
                "prompt": str(item.get("prompt", "")).strip(),
            }
        )

    return grouped, class_name_by_id


def load_step18_pairs(path, warnings):
    expected_columns = [
        "low_concept_id",
        "low_concept_text",
        "high_concept_id",
        "high_concept_text",
        "co_occurrence_count",
        "co_occurrence_rate",
        "mean_joint_evidence",
        "correct_count",
        "incorrect_count",
        "dominant_conflict_type",
        "high_scale_override_rate",
        "wrong_class_drift_rate",
        "example_slides",
    ]
    if not path.exists():
        append_warning(warnings, f"Step18 low/high concept pairs CSV not found: {path}")
        return pd.DataFrame(columns=expected_columns)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        append_warning(warnings, f"Failed to read Step18 low/high concept pairs CSV: {path} ({exc})")
        return pd.DataFrame(columns=expected_columns)


def load_step18_summary(path, warnings):
    expected_columns = ["section", "metric", "value", "note"]
    if not path.exists():
        append_warning(warnings, f"Step18 summary CSV not found: {path}")
        return pd.DataFrame(columns=expected_columns)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        append_warning(warnings, f"Failed to read Step18 summary CSV: {path} ({exc})")
        return pd.DataFrame(columns=expected_columns)


def to_float(value):
    if value is None:
        return np.nan
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return np.nan
        return float(value.detach().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except Exception:
        return np.nan


def compute_tensor_stats(tensor):
    array = tensor.detach().cpu().float().numpy()
    flat = array.reshape(-1)
    finite_mask = np.isfinite(flat)
    finite_values = flat[finite_mask]
    stats = {
        "has_nan": bool(np.isnan(flat).any()),
        "has_inf": bool(np.isinf(flat).any()),
        "element_count": int(flat.size),
        "finite_count": int(finite_values.size),
        "nonzero_count": int(np.count_nonzero(flat)),
    }
    if finite_values.size == 0:
        for key in ["min", "max", "mean", "std", "abs_max", "l1", "l2"]:
            stats[key] = np.nan
        return stats

    stats["min"] = float(finite_values.min())
    stats["max"] = float(finite_values.max())
    stats["mean"] = float(finite_values.mean())
    stats["std"] = float(finite_values.std())
    stats["abs_max"] = float(np.abs(finite_values).max())
    stats["l1"] = float(np.abs(finite_values).sum())
    stats["l2"] = float(np.sqrt(np.square(finite_values).sum()))
    return stats


def get_expected_shape_info(grouped_prompts):
    class_ids = sorted(set(grouped_prompts["low"].keys()) | set(grouped_prompts["high"].keys()))
    low_counts = [len(grouped_prompts["low"].get(class_id, [])) for class_id in class_ids]
    high_counts = [len(grouped_prompts["high"].get(class_id, [])) for class_id in class_ids]
    num_classes = len(class_ids)
    low_balanced = len(set(low_counts)) <= 1 if low_counts else False
    high_balanced = len(set(high_counts)) <= 1 if high_counts else False
    num_low = low_counts[0] if low_balanced and low_counts else None
    num_high = high_counts[0] if high_balanced and high_counts else None
    return {
        "class_ids": class_ids,
        "num_classes": num_classes,
        "num_low": num_low,
        "num_high": num_high,
        "low_counts": low_counts,
        "high_counts": high_counts,
        "low_balanced": low_balanced,
        "high_balanced": high_balanced,
    }


def make_adj_stats_rows(
    adj_tensor,
    alpha_tensor,
    grouped_prompts,
    class_name_by_id,
    adj_key,
    alpha_key,
    warnings,
):
    rows = []
    expected = get_expected_shape_info(grouped_prompts)
    expected_shape = None
    if expected["num_low"] is not None and expected["num_high"] is not None:
        expected_shape = (expected["num_classes"], expected["num_low"], expected["num_high"])

    shape_matches = False
    actual_shape = None
    if adj_tensor is not None and hasattr(adj_tensor, "shape"):
        actual_shape = tuple(int(x) for x in adj_tensor.shape)
        shape_matches = (
            len(actual_shape) == 3
            and expected_shape is not None
            and actual_shape == expected_shape
        )

    for tensor_name, tensor_value in [("adj_raw", adj_tensor), ("adj_tanh", None)]:
        if tensor_name == "adj_tanh" and adj_tensor is not None:
            tensor_value = torch.tanh(adj_tensor.detach().cpu().float())

        if tensor_value is None:
            rows.append(
                {
                    "tensor_name": tensor_name,
                    "class_id": "all",
                    "class_name": "all",
                    "state_key": adj_key if "adj" in tensor_name else "",
                    "found": False,
                    "shape": "",
                    "expected_shape": str(expected_shape),
                    "shape_matches_expected": False,
                    "has_nan": np.nan,
                    "has_inf": np.nan,
                    "element_count": 0,
                    "finite_count": 0,
                    "nonzero_count": 0,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "abs_max": np.nan,
                    "l1": np.nan,
                    "l2": np.nan,
                    "expected_num_classes": expected["num_classes"],
                    "expected_num_low_concepts": expected["num_low"],
                    "expected_num_high_concepts": expected["num_high"],
                    "low_counts": json.dumps(expected["low_counts"]),
                    "high_counts": json.dumps(expected["high_counts"]),
                }
            )
            continue

        global_stats = compute_tensor_stats(tensor_value)
        rows.append(
            {
                "tensor_name": tensor_name,
                "class_id": "all",
                "class_name": "all",
                "state_key": adj_key,
                "found": True,
                "shape": str(tuple(int(x) for x in tensor_value.shape)),
                "expected_shape": str(expected_shape),
                "shape_matches_expected": shape_matches,
                "expected_num_classes": expected["num_classes"],
                "expected_num_low_concepts": expected["num_low"],
                "expected_num_high_concepts": expected["num_high"],
                "low_counts": json.dumps(expected["low_counts"]),
                "high_counts": json.dumps(expected["high_counts"]),
                **global_stats,
            }
        )

        if tensor_value.ndim == 3:
            for class_id in range(tensor_value.shape[0]):
                class_stats = compute_tensor_stats(tensor_value[class_id])
                rows.append(
                    {
                        "tensor_name": tensor_name,
                        "class_id": class_id,
                        "class_name": class_name_by_id.get(class_id, f"class_{class_id}"),
                        "state_key": adj_key,
                        "found": True,
                        "shape": str(tuple(int(x) for x in tensor_value[class_id].shape)),
                        "expected_shape": str(
                            (expected["num_low"], expected["num_high"])
                            if expected["num_low"] is not None and expected["num_high"] is not None
                            else None
                        ),
                        "shape_matches_expected": shape_matches,
                        "expected_num_classes": expected["num_classes"],
                        "expected_num_low_concepts": expected["num_low"],
                        "expected_num_high_concepts": expected["num_high"],
                        "low_counts": json.dumps(expected["low_counts"]),
                        "high_counts": json.dumps(expected["high_counts"]),
                        **class_stats,
                    }
                )

    alpha_value = to_float(alpha_tensor)
    rows.append(
        {
            "tensor_name": "alpha",
            "class_id": "all",
            "class_name": "all",
            "state_key": alpha_key,
            "found": alpha_tensor is not None,
            "shape": str(tuple(int(x) for x in alpha_tensor.shape)) if isinstance(alpha_tensor, torch.Tensor) else "",
            "expected_shape": "",
            "shape_matches_expected": np.nan,
            "has_nan": bool(np.isnan(alpha_value)) if alpha_tensor is not None else np.nan,
            "has_inf": bool(np.isinf(alpha_value)) if alpha_tensor is not None else np.nan,
            "element_count": int(alpha_tensor.numel()) if isinstance(alpha_tensor, torch.Tensor) else 0,
            "finite_count": int(np.isfinite(alpha_value)) if alpha_tensor is not None else 0,
            "nonzero_count": int(alpha_value != 0.0) if alpha_tensor is not None and np.isfinite(alpha_value) else 0,
            "min": alpha_value,
            "max": alpha_value,
            "mean": alpha_value,
            "std": 0.0 if alpha_tensor is not None and np.isfinite(alpha_value) else np.nan,
            "abs_max": abs(alpha_value) if alpha_tensor is not None and np.isfinite(alpha_value) else np.nan,
            "l1": abs(alpha_value) if alpha_tensor is not None and np.isfinite(alpha_value) else np.nan,
            "l2": abs(alpha_value) if alpha_tensor is not None and np.isfinite(alpha_value) else np.nan,
            "expected_num_classes": expected["num_classes"],
            "expected_num_low_concepts": expected["num_low"],
            "expected_num_high_concepts": expected["num_high"],
            "low_counts": json.dumps(expected["low_counts"]),
            "high_counts": json.dumps(expected["high_counts"]),
        }
    )

    if adj_tensor is not None and adj_tensor.ndim != 3:
        append_warning(warnings, f"Adjacency tensor has unexpected rank: shape={tuple(adj_tensor.shape)}")

    return pd.DataFrame(rows)


def get_concept_entry(grouped_prompts, scale_name, class_id, concept_index):
    entries = grouped_prompts.get(scale_name, {}).get(class_id, [])
    if 0 <= concept_index < len(entries):
        return entries[concept_index]
    return {"concept_id": f"{scale_name}_{concept_index}", "concept_en": "", "prompt": ""}


def build_top_learned_pairs_df(adj_tensor, grouped_prompts, class_name_by_id, top_k_pairs):
    columns = [
        "class_id",
        "class_name",
        "low_concept_index",
        "high_concept_index",
        "low_concept_id",
        "high_concept_id",
        "low_concept_text",
        "high_concept_text",
        "raw_weight",
        "tanh_weight",
        "abs_tanh_weight",
        "rank",
    ]
    if adj_tensor is None or adj_tensor.ndim != 3:
        return pd.DataFrame(columns=columns)

    adj_raw = adj_tensor.detach().cpu().float()
    adj_tanh = torch.tanh(adj_raw)
    rows = []

    for class_id in range(adj_raw.shape[0]):
        raw_matrix = adj_raw[class_id].numpy()
        tanh_matrix = adj_tanh[class_id].numpy()
        abs_tanh = np.abs(tanh_matrix).reshape(-1)
        if abs_tanh.size == 0:
            continue

        top_indices = np.argsort(-abs_tanh)[: max(int(top_k_pairs), 0)]
        for rank_idx, flat_index in enumerate(top_indices, start=1):
            low_index, high_index = np.unravel_index(int(flat_index), tanh_matrix.shape)
            low_entry = get_concept_entry(grouped_prompts, "low", class_id, low_index)
            high_entry = get_concept_entry(grouped_prompts, "high", class_id, high_index)
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name_by_id.get(class_id, f"class_{class_id}"),
                    "low_concept_index": int(low_index),
                    "high_concept_index": int(high_index),
                    "low_concept_id": low_entry.get("concept_id", ""),
                    "high_concept_id": high_entry.get("concept_id", ""),
                    "low_concept_text": low_entry.get("prompt", ""),
                    "high_concept_text": high_entry.get("prompt", ""),
                    "raw_weight": float(raw_matrix[low_index, high_index]),
                    "tanh_weight": float(tanh_matrix[low_index, high_index]),
                    "abs_tanh_weight": float(abs_tanh[int(flat_index)]),
                    "rank": rank_idx,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def normalize_pair_text_columns(df):
    if df.empty:
        return df.copy()
    out = df.copy()
    out["low_concept_text_norm"] = out["low_concept_text"].fillna("").astype(str).str.strip()
    out["high_concept_text_norm"] = out["high_concept_text"].fillna("").astype(str).str.strip()
    return out


def build_step18_overlap_df(learned_pairs_df, step18_pairs_df, top_k_pairs):
    columns = [
        "comparison_name",
        "class_id",
        "class_name",
        "learned_pair_count",
        "step18_pair_count",
        "overlap_count",
        "overlap_rate_vs_learned",
        "overlap_rate_vs_step18",
        "overlap_pairs",
    ]
    if learned_pairs_df.empty or step18_pairs_df.empty:
        return pd.DataFrame(columns=columns)

    learned_norm = normalize_pair_text_columns(learned_pairs_df)
    step18_norm = normalize_pair_text_columns(step18_pairs_df)

    subset_defs = []
    subset_defs.append(
        (
            "step18_top_mean_joint_evidence",
            step18_norm.sort_values(
                by=["mean_joint_evidence", "co_occurrence_count"],
                ascending=[False, False],
            ).head(top_k_pairs),
        )
    )

    override_subset = step18_norm[step18_norm["high_scale_override_rate"].fillna(0) > 0].copy()
    override_subset = override_subset.sort_values(
        by=["high_scale_override_rate", "mean_joint_evidence", "co_occurrence_count"],
        ascending=[False, False, False],
    ).head(top_k_pairs)
    subset_defs.append(("step18_high_scale_override", override_subset))

    drift_subset = step18_norm[step18_norm["wrong_class_drift_rate"].fillna(0) > 0].copy()
    drift_subset = drift_subset.sort_values(
        by=["wrong_class_drift_rate", "mean_joint_evidence", "co_occurrence_count"],
        ascending=[False, False, False],
    ).head(top_k_pairs)
    subset_defs.append(("step18_wrong_class_drift", drift_subset))

    rows = []
    class_ids = sorted(learned_norm["class_id"].dropna().unique().tolist())
    compare_targets = [("all", "all")] + [(int(class_id), None) for class_id in class_ids]

    for comparison_name, subset_df in subset_defs:
        subset_pairs = subset_df[["low_concept_text_norm", "high_concept_text_norm"]].drop_duplicates()
        subset_count = int(len(subset_pairs))

        for class_token, _ in compare_targets:
            if class_token == "all":
                learned_subset = learned_norm
                class_name = "all"
            else:
                learned_subset = learned_norm[learned_norm["class_id"] == class_token]
                class_name = (
                    learned_subset["class_name"].iloc[0]
                    if not learned_subset.empty
                    else f"class_{class_token}"
                )

            learned_pairs = learned_subset[["low_concept_text_norm", "high_concept_text_norm"]].drop_duplicates()
            learned_count = int(len(learned_pairs))
            if learned_count == 0 or subset_count == 0:
                overlap_pairs = pd.DataFrame(columns=["low_concept_text_norm", "high_concept_text_norm"])
            else:
                overlap_pairs = learned_pairs.merge(
                    subset_pairs,
                    on=["low_concept_text_norm", "high_concept_text_norm"],
                    how="inner",
                ).drop_duplicates()

            overlap_count = int(len(overlap_pairs))
            overlap_labels = []
            for _, row in overlap_pairs.iterrows():
                overlap_labels.append(
                    f"{row['low_concept_text_norm']} || {row['high_concept_text_norm']}"
                )

            rows.append(
                {
                    "comparison_name": comparison_name,
                    "class_id": class_token,
                    "class_name": class_name,
                    "learned_pair_count": learned_count,
                    "step18_pair_count": subset_count,
                    "overlap_count": overlap_count,
                    "overlap_rate_vs_learned": (
                        float(overlap_count / learned_count) if learned_count > 0 else np.nan
                    ),
                    "overlap_rate_vs_step18": (
                        float(overlap_count / subset_count) if subset_count > 0 else np.nan
                    ),
                    "overlap_pairs": json.dumps(overlap_labels, ensure_ascii=False),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def markdown_table_from_df(df, columns):
    if df.empty:
        return "_No rows._"
    table_df = df.loc[:, columns].copy().fillna("")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in table_df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            text = str(value).replace("\n", " ").replace("|", "/")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_summary_df(
    ckpt_path,
    tried_paths,
    adj_tensor,
    alpha_tensor,
    adj_key,
    alpha_key,
    adj_stats_df,
    overlap_df,
    warnings,
    next_step,
):
    raw_row = adj_stats_df[adj_stats_df["tensor_name"] == "adj_raw"]
    raw_row = raw_row[raw_row["class_id"].astype(str) == "all"]
    tanh_row = adj_stats_df[adj_stats_df["tensor_name"] == "adj_tanh"]
    tanh_row = tanh_row[tanh_row["class_id"].astype(str) == "all"]

    overlap_all = overlap_df[overlap_df["class_id"].astype(str) == "all"].copy()
    overlap_lookup = {row["comparison_name"]: int(row["overlap_count"]) for _, row in overlap_all.iterrows()}

    summary = {
        "checkpoint_path": str(ckpt_path) if ckpt_path is not None else "",
        "tried_checkpoint_paths": json.dumps(tried_paths, ensure_ascii=False),
        "adj_found": bool(adj_tensor is not None),
        "alpha_found": bool(alpha_tensor is not None),
        "adj_state_key": adj_key or "",
        "alpha_state_key": alpha_key or "",
        "adj_shape": (
            str(tuple(int(x) for x in adj_tensor.shape))
            if isinstance(adj_tensor, torch.Tensor)
            else ""
        ),
        "shape_matches_expected": (
            bool(raw_row["shape_matches_expected"].iloc[0]) if not raw_row.empty else False
        ),
        "raw_min": float(raw_row["min"].iloc[0]) if not raw_row.empty else np.nan,
        "raw_max": float(raw_row["max"].iloc[0]) if not raw_row.empty else np.nan,
        "raw_abs_max": float(raw_row["abs_max"].iloc[0]) if not raw_row.empty else np.nan,
        "tanh_min": float(tanh_row["min"].iloc[0]) if not tanh_row.empty else np.nan,
        "tanh_max": float(tanh_row["max"].iloc[0]) if not tanh_row.empty else np.nan,
        "tanh_abs_max": float(tanh_row["abs_max"].iloc[0]) if not tanh_row.empty else np.nan,
        "alpha_value": to_float(alpha_tensor),
        "overlap_top_mean_joint_evidence": overlap_lookup.get("step18_top_mean_joint_evidence", 0),
        "overlap_high_scale_override": overlap_lookup.get("step18_high_scale_override", 0),
        "overlap_wrong_class_drift": overlap_lookup.get("step18_wrong_class_drift", 0),
        "warning_count": len(warnings),
        "next_step": next_step,
    }
    return pd.DataFrame([summary])


def choose_next_step(adj_tensor, adj_stats_df):
    raw_row = adj_stats_df[adj_stats_df["tensor_name"] == "adj_raw"]
    raw_row = raw_row[raw_row["class_id"].astype(str) == "all"]
    if adj_tensor is None or raw_row.empty:
        return "Step19 fix first: cross-scale graph parameter missing in smoke checkpoint."
    if not bool(raw_row["shape_matches_expected"].iloc[0]):
        return "Step19 fix first: cross-scale graph shape does not match expected concept dimensions."

    abs_max = float(raw_row["abs_max"].iloc[0])
    if np.isfinite(abs_max) and abs_max <= 1e-6:
        return "Step21 non-testing fold0 pilot: smoke is too short to learn meaningful adjacency."

    return "Step21 non-testing fold0 pilot: adjacency exists and basic stats look normal for smoke inspection."


def write_report(
    out_path,
    ckpt_path,
    tried_paths,
    stage18_pairs_path,
    stage18_summary_path,
    concept_prompt_path,
    adj_tensor,
    alpha_tensor,
    adj_key,
    alpha_key,
    adj_stats_df,
    top_pairs_df,
    overlap_df,
    warnings,
    next_step,
):
    raw_global = adj_stats_df[
        (adj_stats_df["tensor_name"] == "adj_raw") & (adj_stats_df["class_id"].astype(str) == "all")
    ]
    tanh_global = adj_stats_df[
        (adj_stats_df["tensor_name"] == "adj_tanh") & (adj_stats_df["class_id"].astype(str) == "all")
    ]

    lines = []
    lines.append("# Step20 CSG Smoke Artifact Inspection")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Step20 only performs checkpoint artifact inspection.")
    lines.append("- Step20 does not retrain the model, does not modify model code, and does not judge final model performance.")
    lines.append("- The inspected checkpoint comes from a smoke run with `--testing` and `1 epoch`, so adjacency values may stay very small or unstable.")
    lines.append("- If adjacency is near zero after smoke, that is not a failure by itself; a non-testing fold0 pilot is needed to judge learnable patterns.")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Selected checkpoint: `{ckpt_path}`" if ckpt_path is not None else "- Selected checkpoint: not found")
    lines.append(f"- Tried checkpoint paths: `{json.dumps(tried_paths, ensure_ascii=False)}`")
    lines.append(f"- Step18 pair CSV: `{stage18_pairs_path}`")
    lines.append(f"- Step18 summary CSV: `{stage18_summary_path}`")
    lines.append(f"- Concept prompt JSON: `{concept_prompt_path}`")
    lines.append("")
    lines.append("## Parameter Presence")
    lines.append(f"- `rce_cross_scale_graph_adj` found: `{adj_tensor is not None}`")
    lines.append(f"- `rce_cross_scale_graph_alpha` found: `{alpha_tensor is not None}`")
    lines.append(f"- adjacency key: `{adj_key}`")
    lines.append(f"- alpha key: `{alpha_key}`")
    lines.append("")

    if not raw_global.empty:
        raw_row = raw_global.iloc[0]
        lines.append("## Adjacency Shape And Range")
        lines.append(f"- adjacency shape: `{raw_row['shape']}`")
        lines.append(f"- expected shape: `{raw_row['expected_shape']}`")
        lines.append(f"- shape matches expected `num_classes x num_low_concepts x num_high_concepts`: `{bool(raw_row['shape_matches_expected'])}`")
        lines.append(
            "- raw adjacency stats: "
            f"`min={raw_row['min']:.6f}`, `max={raw_row['max']:.6f}`, `mean={raw_row['mean']:.6f}`, "
            f"`std={raw_row['std']:.6f}`, `abs_max={raw_row['abs_max']:.6f}`, `l1={raw_row['l1']:.6f}`, `l2={raw_row['l2']:.6f}`"
        )
        lines.append(
            "- raw adjacency finite checks: "
            f"`has_nan={bool(raw_row['has_nan'])}`, `has_inf={bool(raw_row['has_inf'])}`, "
            f"`nonzero_count={int(raw_row['nonzero_count'])}`"
        )
        if not tanh_global.empty:
            tanh_row = tanh_global.iloc[0]
            lines.append(
                "- `tanh(adj)` stats: "
                f"`min={tanh_row['min']:.6f}`, `max={tanh_row['max']:.6f}`, `mean={tanh_row['mean']:.6f}`, "
                f"`std={tanh_row['std']:.6f}`, `abs_max={tanh_row['abs_max']:.6f}`, `l1={tanh_row['l1']:.6f}`, `l2={tanh_row['l2']:.6f}`"
            )
    else:
        lines.append("## Adjacency Shape And Range")
        lines.append("- adjacency tensor is missing, so shape/range stats were not computed.")

    lines.append("")
    lines.append("## Alpha")
    if alpha_tensor is not None:
        alpha_value = to_float(alpha_tensor)
        lines.append(f"- learned alpha value: `{alpha_value:.6f}`")
    else:
        lines.append("- alpha tensor is missing.")

    lines.append("")
    lines.append("## Top Learned Low-High Pairs")
    if top_pairs_df.empty:
        lines.append("- No learned pair table could be generated.")
    else:
        for class_id in sorted(top_pairs_df["class_id"].unique().tolist()):
            class_df = top_pairs_df[top_pairs_df["class_id"] == class_id].copy()
            class_name = class_df["class_name"].iloc[0]
            lines.append(f"### Class {class_id} - {class_name}")
            lines.append(
                markdown_table_from_df(
                    class_df,
                    [
                        "rank",
                        "low_concept_id",
                        "high_concept_id",
                        "raw_weight",
                        "tanh_weight",
                        "abs_tanh_weight",
                    ],
                )
            )
            lines.append("")

    lines.append("## Step18 Overlap")
    if overlap_df.empty:
        lines.append("- No overlap rows were produced. This usually means learned pairs or Step18 pair CSV could not be aligned.")
    else:
        lines.append(
            markdown_table_from_df(
                overlap_df,
                [
                    "comparison_name",
                    "class_id",
                    "overlap_count",
                    "learned_pair_count",
                    "step18_pair_count",
                    "overlap_rate_vs_learned",
                    "overlap_rate_vs_step18",
                ],
            )
        )
        lines.append("")
        lines.append("- This overlap is a lightweight exact-text comparison against Step18 post-hoc pairs.")
        lines.append("- Pay special attention to `step18_high_scale_override` and `step18_wrong_class_drift` rows; any overlap suggests the learnable CSG path is already pointing toward known post-hoc patterns.")

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for warning_text in warnings:
            lines.append(f"- {warning_text}")

    lines.append("")
    lines.append("## Conclusion")
    lines.append("- Step20 only inspects artifacts from the existing Step19 smoke checkpoint.")
    lines.append("- Step20 does not re-run smoke, does not train, does not run 5-fold, and does not extract features.")
    lines.append(f"- Next suggested step: `{next_step}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    stage18_dir = resolve_path(root, args.stage18_dir)
    concept_prompt_path = resolve_path(root, args.concept_prompt_path)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []

    state_dict, ckpt_path, tried_paths = load_checkpoint_state_dict(root, args.ckpt_path, warnings)
    adj_tensor, adj_key = find_state_value(
        state_dict,
        ["rce_cross_scale_graph_adj", "module.rce_cross_scale_graph_adj"],
    )
    alpha_tensor, alpha_key = find_state_value(
        state_dict,
        ["rce_cross_scale_graph_alpha", "module.rce_cross_scale_graph_alpha"],
    )

    if adj_tensor is None:
        append_warning(warnings, "Cross-scale graph adjacency parameter was not found in the checkpoint state_dict.")
    if alpha_tensor is None:
        append_warning(warnings, "Cross-scale graph alpha parameter was not found in the checkpoint state_dict.")

    grouped_prompts, class_name_by_id = load_concept_prompt_metadata(concept_prompt_path, warnings)
    stage18_pairs_path = stage18_dir / "stage18_low_high_concept_pairs.csv"
    stage18_summary_path = stage18_dir / "stage18_cross_scale_graph_summary.csv"
    step18_pairs_df = load_step18_pairs(stage18_pairs_path, warnings)
    _ = load_step18_summary(stage18_summary_path, warnings)

    adj_stats_df = make_adj_stats_rows(
        adj_tensor=adj_tensor,
        alpha_tensor=alpha_tensor,
        grouped_prompts=grouped_prompts,
        class_name_by_id=class_name_by_id,
        adj_key=adj_key or "",
        alpha_key=alpha_key or "",
        warnings=warnings,
    )
    top_pairs_df = build_top_learned_pairs_df(
        adj_tensor=adj_tensor,
        grouped_prompts=grouped_prompts,
        class_name_by_id=class_name_by_id,
        top_k_pairs=args.top_k_pairs,
    )
    overlap_df = build_step18_overlap_df(
        learned_pairs_df=top_pairs_df,
        step18_pairs_df=step18_pairs_df,
        top_k_pairs=args.top_k_pairs,
    )
    next_step = choose_next_step(adj_tensor, adj_stats_df)
    summary_df = build_summary_df(
        ckpt_path=ckpt_path,
        tried_paths=tried_paths,
        adj_tensor=adj_tensor,
        alpha_tensor=alpha_tensor,
        adj_key=adj_key,
        alpha_key=alpha_key,
        adj_stats_df=adj_stats_df,
        overlap_df=overlap_df,
        warnings=warnings,
        next_step=next_step,
    )

    adj_stats_df.to_csv(out_dir / "stage20_csg_adj_stats.csv", index=False)
    top_pairs_df.to_csv(out_dir / "stage20_csg_top_learned_pairs.csv", index=False)
    overlap_df.to_csv(out_dir / "stage20_csg_step18_overlap.csv", index=False)
    summary_df.to_csv(out_dir / "stage20_csg_smoke_inspection_summary.csv", index=False)
    write_report(
        out_path=out_dir / "stage20_csg_smoke_inspection_report.md",
        ckpt_path=ckpt_path,
        tried_paths=tried_paths,
        stage18_pairs_path=stage18_pairs_path,
        stage18_summary_path=stage18_summary_path,
        concept_prompt_path=concept_prompt_path,
        adj_tensor=adj_tensor,
        alpha_tensor=alpha_tensor,
        adj_key=adj_key or "",
        alpha_key=alpha_key or "",
        adj_stats_df=adj_stats_df,
        top_pairs_df=top_pairs_df,
        overlap_df=overlap_df,
        warnings=warnings,
        next_step=next_step,
    )

    print(f"[Done] Wrote Step20 inspection artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
