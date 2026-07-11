#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
TEXT_REPO = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
BIOMEDCLIP_MODEL_PATH = f"hf-hub:{BIOMEDCLIP_REPO}"

METRIC_NAME_CANDIDATES = {
    "test_auc": ["test_auc", "auc"],
    "test_acc": ["test_acc", "acc", "test_accuracy", "accuracy"],
    "test_f1": ["test_f1", "f1"],
    "val_auc": ["val_auc"],
    "balanced_acc": ["balanced_acc", "test_balanced_acc"],
    "sensitivity": ["sensitivity", "test_sensitivity", "recall"],
    "specificity": ["specificity", "test_specificity"],
    "pr_auc": ["pr_auc", "test_pr_auc"],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_scalar(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)
    return value


def _numeric_stats(array: np.ndarray) -> dict[str, object]:
    if array.size == 0:
        return {
            "numel": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "abs_mean": None,
            "abs_max": None,
        }
    cast = array.astype(np.float64, copy=False)
    return {
        "numel": int(array.size),
        "mean": float(cast.mean()),
        "std": float(cast.std()),
        "min": float(cast.min()),
        "max": float(cast.max()),
        "abs_mean": float(np.abs(cast).mean()),
        "abs_max": float(np.abs(cast).max()),
    }


def tensor_payload(value, *, include_values: bool = False, max_values: int | None = None):
    array = tensor_to_numpy(value)
    if array is None:
        return {"available": False, "reason": "unavailable"}
    payload = {
        "available": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
        "stats": _numeric_stats(array),
    }
    if include_values:
        if max_values is not None and array.size > max_values:
            payload["values"] = np.asarray(array).reshape(-1)[:max_values].tolist()
            payload["values_truncated"] = True
        else:
            payload["values"] = array.tolist()
    return payload


def flatten_numeric_payloads(payload, prefix=""):
    items = {}
    if isinstance(payload, dict):
        if payload.get("available") is False:
            return items
        if payload.get("available") is True and "values" in payload:
            items[prefix.rstrip(".")] = tensor_to_numpy(payload["values"])
            return items
        for key, value in payload.items():
            next_prefix = f"{prefix}{key}."
            items.update(flatten_numeric_payloads(value, next_prefix))
    elif isinstance(payload, list):
        try:
            items[prefix.rstrip(".")] = np.asarray(payload)
        except Exception:
            return items
    return items


def stable_state_dict_hash(state_dict) -> tuple[list[dict[str, object]], str]:
    digest = hashlib.sha256()
    rows = []
    for name, tensor in state_dict.items():
        array = tensor_to_numpy(tensor)
        array = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
        rows.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "numel": int(array.size),
                "sha256": sha256_bytes(array.tobytes()),
            }
        )
    return rows, digest.hexdigest()


def load_text_prompts(path: os.PathLike[str] | str | None):
    if not path:
        return None
    try:
        df = pd.read_csv(path)
        columns = [str(col).strip().lower() for col in df.columns]

        def to_list(series):
            return [str(item) for item in series.fillna("").astype(str).tolist()]

        low = []
        high = []
        if "low_resolution_description" in columns and "high_resolution_description" in columns:
            low = to_list(df.iloc[:, columns.index("low_resolution_description")])
            high = to_list(df.iloc[:, columns.index("high_resolution_description")])
        elif len(df.columns) >= 2:
            low = to_list(df.iloc[:, -2])
            high = to_list(df.iloc[:, -1])
        elif len(df.columns) == 1:
            low = to_list(df.iloc[:, 0])
        return {"low": low, "high": high, "combined": low + high}
    except Exception:
        values = pd.read_csv(path, header=None).values.reshape(-1).tolist()
        values = [str(item) for item in values]
        half = len(values) // 2
        return {"low": values[:half], "high": values[half:], "combined": values}


def load_concept_prompt_items(path: os.PathLike[str] | str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("prompts", [])
    else:
        items = data
    grouped = {"low": {}, "high": {}}
    for item in items:
        if not bool(item.get("use_in_stage2", True)):
            continue
        scale = str(item.get("scale", "")).strip().lower()
        if scale not in grouped:
            continue
        class_id = int(item["class_id"])
        grouped[scale].setdefault(class_id, []).append(dict(item))
    for scale in grouped:
        grouped[scale] = {int(key): grouped[scale][key] for key in sorted(grouped[scale])}
    return grouped


def tokenize_prompt_groups(tokenizer, prompt_groups: dict[str, list[str]]):
    output = {}
    for name, prompts in prompt_groups.items():
        tokens = tokenizer(prompts)
        output[name] = {
            "texts": list(prompts),
            "token_ids": tensor_payload(tokens, include_values=True),
            "count": len(prompts),
        }
    return output


def tokenize_concept_groups(tokenizer, grouped_items: dict[str, dict[int, list[dict[str, object]]]]):
    output = {}
    for scale in ("low", "high"):
        scale_payload = {}
        aggregate = hashlib.sha256()
        for class_id, items in grouped_items[scale].items():
            prompts = [str(item.get("prompt", "")).strip() for item in items]
            tokens = tokenizer(prompts)
            token_array = tensor_to_numpy(tokens)
            aggregate.update(np.ascontiguousarray(token_array).tobytes())
            scale_payload[str(class_id)] = {
                "prompts": prompts,
                "token_ids": tensor_payload(token_array, include_values=True),
                "metadata": items,
            }
        output[scale] = {
            "classes": scale_payload,
            "aggregate_sha256": aggregate.hexdigest(),
        }
    return output


def find_snapshot_dir(cache_dir: os.PathLike[str] | str | None, repo_id: str):
    if not cache_dir:
        return None
    snapshots_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not candidates:
        return None
    return candidates[-1]


def read_snapshot_revision(cache_dir: os.PathLike[str] | str | None, repo_id: str):
    if not cache_dir:
        return None
    refs_main = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    if refs_main.is_file():
        return refs_main.read_text(encoding="utf-8").strip()
    snapshot_dir = find_snapshot_dir(cache_dir, repo_id)
    return snapshot_dir.name if snapshot_dir else None


def load_split_csv(split_dir: os.PathLike[str] | str, fold: int) -> pd.DataFrame:
    split_path = Path(split_dir) / f"splits_{int(fold)}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return pd.read_csv(split_path)


def select_probe_slide(split_dir: os.PathLike[str] | str, fold: int, selection_column: str = "test"):
    df = load_split_csv(split_dir, fold)
    if selection_column not in df.columns:
        raise KeyError(f"Column '{selection_column}' missing in split file for fold={fold}")
    values = [str(item) for item in df[selection_column].dropna().tolist() if str(item)]
    if not values:
        raise ValueError(f"No slide ids found in column '{selection_column}' for fold={fold}")
    return values[0]


def load_label_for_slide(dataset_csv: os.PathLike[str] | str, slide_id: str):
    df = pd.read_csv(dataset_csv)
    rows = df[df["slide_id"].astype(str) == str(slide_id)]
    if rows.empty:
        raise KeyError(f"slide_id not found in dataset CSV: {slide_id}")
    return normalize_scalar(rows.iloc[0]["label"])


def load_h5_payload(h5_path: os.PathLike[str] | str):
    with h5py.File(h5_path, "r") as handle:
        features = np.asarray(handle["features"])
        coords = np.asarray(handle["coords"])
    return {
        "path": str(h5_path),
        "sha256": sha256_file(h5_path),
        "features": features,
        "coords": coords,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "coords_shape": list(coords.shape),
        "coords_dtype": str(coords.dtype),
    }


def find_run_dir(path: os.PathLike[str] | str) -> Path:
    root = Path(path)
    if root.is_file():
        root = root.parent
    target_names = {
        "summary.csv",
        "summary_partial_1_1.csv",
        "result.csv",
        "result_partial_1_1.csv",
        "fold_summary.csv",
        "epoch_details.csv",
    }
    candidates = []
    for candidate in [root] + sorted(p for p in root.rglob("*") if p.is_dir()):
        names = {child.name for child in candidate.iterdir() if child.is_file()}
        score = len(target_names & names)
        if score:
            candidates.append((score, len(candidate.parts), candidate))
    if not candidates:
        raise FileNotFoundError(f"No result files found under {path}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def discover_result_files(run_dir: os.PathLike[str] | str) -> dict[str, Path]:
    run_dir = Path(run_dir)
    files = {}
    patterns = {
        "summary": ["summary.csv", "summary_partial_*.csv"],
        "result": ["result.csv", "result_partial_*.csv"],
        "fold_summary": ["fold_summary.csv"],
        "epoch_details": ["epoch_details.csv"],
        "split_csvs": ["splits_*.csv", "split_*.csv"],
        "pickles": ["*.pkl"],
        "checkpoints": ["*.pt"],
    }
    for key, pattern_list in patterns.items():
        matches = []
        for pattern in pattern_list:
            matches.extend(sorted(run_dir.glob(pattern)))
        if key in {"split_csvs", "pickles", "checkpoints"}:
            files[key] = matches
        elif matches:
            files[key] = matches[0]
    return files


def read_csv_records(path: os.PathLike[str] | str) -> list[dict[str, object]]:
    return pd.read_csv(path).to_dict(orient="records")


def choose_fold_row(records: list[dict[str, object]], fold_index: int, *, human_fold: int | None = None):
    for record in records:
        if "fold" in record and normalize_scalar(record["fold"]) == human_fold:
            return record
    for record in records:
        if "folds" in record and normalize_scalar(record["folds"]) == fold_index:
            return record
    for record in records:
        if "fold" in record and normalize_scalar(record["fold"]) == fold_index:
            return record
    return records[0] if records else {}


def extract_metrics(row: dict[str, object]) -> dict[str, object]:
    metrics = {}
    for canonical_name, candidates in METRIC_NAME_CANDIDATES.items():
        for key in candidates:
            if key in row and pd.notna(row[key]):
                metrics[canonical_name] = normalize_scalar(row[key])
                break
    return metrics


def load_pickle_predictions(path: os.PathLike[str] | str):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_prediction_pickle(pickle_paths: list[Path], fold_index: int):
    suffix = f"split_{fold_index}_results.pkl"
    for path in pickle_paths:
        if path.name == suffix:
            return path
    return pickle_paths[0] if pickle_paths else None


def prediction_summary(predictions: dict[str, dict[str, object]], positive_index: int = 1):
    y_true = []
    y_prob = []
    for slide_id in sorted(predictions):
        payload = predictions[slide_id]
        label = payload.get("label")
        prob = payload.get("prob")
        if isinstance(label, np.ndarray):
            label = label.item()
        if isinstance(prob, np.ndarray):
            prob = np.asarray(prob).reshape(-1).tolist()
        elif isinstance(prob, (list, tuple)):
            prob = np.asarray(prob).reshape(-1).tolist()
        if label is None or prob is None:
            continue
        if len(prob) <= positive_index:
            continue
        y_true.append(int(label))
        y_prob.append(float(prob[positive_index]))
    if not y_true:
        return {}
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    quantiles = np.quantile(y_prob, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "sample_count": int(y_true.size),
        "positive_rate_pred": float(y_pred.mean()),
        "positive_rate_true": float(y_true.mean()),
        "prob_mean": float(y_prob.mean()),
        "prob_std": float(y_prob.std()),
        "prob_min": float(y_prob.min()),
        "prob_max": float(y_prob.max()),
        "prob_quantiles": {
            "p10": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def best_epoch_summary(epoch_records: list[dict[str, object]]):
    if not epoch_records:
        return {}
    df = pd.DataFrame(epoch_records)
    result = {}
    if "val_auc" in df:
        idx = int(df["val_auc"].astype(float).idxmax())
        row = df.iloc[idx]
        result["best_val_auc_epoch"] = int(row["epoch"])
        result["best_val_auc"] = float(row["val_auc"])
    if "val_loss" in df:
        idx = int(df["val_loss"].astype(float).idxmin())
        row = df.iloc[idx]
        result["best_val_loss_epoch"] = int(row["epoch"])
        result["best_val_loss"] = float(row["val_loss"])
    if "val_error" in df:
        idx = int(df["val_error"].astype(float).idxmin())
        row = df.iloc[idx]
        result["best_val_error_epoch"] = int(row["epoch"])
        result["best_val_error"] = float(row["val_error"])
    last_row = df.iloc[-1]
    result["last_epoch"] = int(last_row["epoch"])
    for key in ("val_auc", "val_acc", "val_f1", "val_loss", "val_error", "train_loss"):
        if key in last_row and pd.notna(last_row[key]):
            result[f"last_{key}"] = normalize_scalar(last_row[key])
    if "best_val_auc_epoch" in result and "last_val_auc" in result:
        result["best_vs_last_val_auc_delta"] = float(result["best_val_auc"] - result["last_val_auc"])
    return result


def dump_json(path: os.PathLike[str] | str, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[object]]):
    parts = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        parts.append("| " + " | ".join("" if item is None else str(item) for item in row) + " |")
    return "\n".join(parts)


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric
