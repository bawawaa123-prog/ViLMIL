#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from stage64k_common import (
    BIOMEDCLIP_MODEL_PATH,
    BIOMEDCLIP_REPO,
    TEXT_REPO,
    dump_json,
    find_snapshot_dir,
    load_concept_prompt_items,
    load_h5_payload,
    load_label_for_slide,
    load_text_prompts,
    read_snapshot_revision,
    select_probe_slide,
    stable_state_dict_hash,
    tensor_payload,
    tokenize_concept_groups,
    tokenize_prompt_groups,
)


PYTHON_BIN = "/opt/conda/envs/vila_mil_overlay_rt/bin/python"
CLASS_NAMES = ["Adenocarcinoma", "NonAdenocarcinoma"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--concept-path", type=Path, required=True)
    parser.add_argument("--hf-cache-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def build_worker_command(args):
    return [
        PYTHON_BIN,
        str(Path(__file__).resolve()),
        "--worker",
        "--project-root",
        str(args.project_root),
        "--data-root",
        str(args.data_root),
        "--split-dir",
        str(args.split_dir),
        "--prompt-path",
        str(args.prompt_path),
        "--concept-path",
        str(args.concept_path),
        "--hf-cache-dir",
        str(args.hf_cache_dir),
        "--fold",
        str(args.fold),
        "--seed",
        str(args.seed),
        "--output",
        str(args.output),
    ]


def run_launcher(args):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": str(args.hf_cache_dir),
            "HF_HUB_CACHE": str(args.hf_cache_dir),
            "HUGGINGFACE_HUB_CACHE": str(args.hf_cache_dir),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONPATH": str(args.project_root),
        }
    )
    completed = subprocess.run(
        build_worker_command(args),
        cwd=str(args.project_root),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        error_payload = {
            "project_root": str(args.project_root),
            "returncode": completed.returncode,
        }
        args.output.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(completed.returncode)


def worker_import_project(project_root: Path):
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)


def load_project_versions():
    import torch
    import open_clip
    import transformers
    import huggingface_hub

    try:
        import ml_collections

        ml_collections_version = getattr(ml_collections, "__version__", "unknown")
    except Exception:
        ml_collections_version = "unavailable"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "huggingface_hub_version": getattr(huggingface_hub, "__version__", "unknown"),
        "ml_collections_version": ml_collections_version,
    }


def current_commit(project_root: Path):
    output = subprocess.run(
        ["git", "-C", str(project_root.parent), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return output.stdout.strip() if output.returncode == 0 else None


def resolve_model_metadata(cache_dir: Path):
    clip_snapshot = find_snapshot_dir(cache_dir, BIOMEDCLIP_REPO)
    text_snapshot = find_snapshot_dir(cache_dir, TEXT_REPO)
    return {
        "requested_model_path": BIOMEDCLIP_MODEL_PATH,
        "resolved_snapshot_path": str(clip_snapshot) if clip_snapshot else None,
        "snapshot_revision": read_snapshot_revision(cache_dir, BIOMEDCLIP_REPO),
        "text_snapshot_path": str(text_snapshot) if text_snapshot else None,
        "text_snapshot_revision": read_snapshot_revision(cache_dir, TEXT_REPO),
        "hf_cache_dir": str(cache_dir),
        "offline_env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }


def build_model():
    import ml_collections
    import torch

    from models.model_RCE_MIL_BiomedCLIP_v2 import RCE_MIL_BiomedCLIP

    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.hidden_size = 192
    config.class_names = CLASS_NAMES
    config.use_concept_prompt_pool = True
    config.concept_prompt_path = str(ARGS.concept_path)
    config.peps_tau = 0.1
    config.prototype_number = 16
    config.rce_use_logit_calibration = True
    config.rce_use_concept_prior = True
    config.rce_logit_scale_init = 10.0
    config.rce_concept_prior_strength = 1.0
    config.rce_use_visual_residual = True
    config.rce_visual_residual_init = 0.05
    config.rce_use_residual_constraint = True
    config.rce_residual_constraint_lambda = 0.03
    config.rce_residual_ratio_target = 0.50
    config.rce_residual_constraint_type = "relu_l2"
    config.rce_use_concept_aux_loss = True
    config.rce_concept_aux_loss_weight = 0.20
    config.rce_residual_ratio_eps = 1e-6
    config.rce_residual_ratio_detach = False
    config.rce_use_cross_scale_graph = True
    config.rce_cross_scale_graph_init = 0.1
    config.rce_cross_scale_graph_norm = "sqrt"
    config.scale_mode = "dual"
    config.finetune_text_encoder = False
    config.enable_logit_breakdown_audit = True

    model = RCE_MIL_BiomedCLIP(config=config, num_classes=2, model_path=BIOMEDCLIP_MODEL_PATH)
    if hasattr(model, "set_logit_breakdown_audit"):
        model.set_logit_breakdown_audit(True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def read_prompt_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def encode_text_prompts(model, prompt_path: Path):
    import torch

    prompt_groups = load_text_prompts(prompt_path)
    token_payload = tokenize_prompt_groups(model.tokenizer, {"low": prompt_groups["low"], "high": prompt_groups["high"]})
    embeddings = {}
    for scale in ("low", "high"):
        prompts = token_payload[scale]["texts"]
        tokens = model.tokenizer(prompts).to(next(model.parameters()).device)
        with torch.no_grad():
            features = model.text_encoder(tokens)
        embeddings[scale] = tensor_payload(features, include_values=True)
    return {
        "prompt_csv_path": str(prompt_path),
        "prompt_csv_rows": read_prompt_csv_rows(prompt_path),
        "token_ids": token_payload,
        "embeddings": embeddings,
    }


def encode_concept_prompts(model, concept_path: Path):
    concept_groups = load_concept_prompt_items(concept_path)
    token_payload = tokenize_concept_groups(model.tokenizer, concept_groups)
    return {
        "concept_prompt_path": str(concept_path),
        "token_ids": token_payload,
        "model_low_prompt_features": tensor_payload(model.low_prompt_features, include_values=True),
        "model_high_prompt_features": tensor_payload(model.high_prompt_features, include_values=True),
    }


def summarize_trainable_parameters(model):
    rows = []
    total = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = int(parameter.numel())
        total += count
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "numel": count,
            }
        )
    return {"parameters": rows, "total_numel": total}


def forward_probe(model, slide_id: str, label_id: int, data_root: Path):
    import torch
    from torch.nn import functional as F

    low_h5 = load_h5_payload(data_root / "features_biomedclip_5x" / f"{slide_id}.h5")
    high_h5 = load_h5_payload(data_root / "features_biomedclip_20x" / f"{slide_id}.h5")

    device = next(model.parameters()).device
    x_s = torch.from_numpy(low_h5["features"]).to(device)
    coord_s = torch.from_numpy(low_h5["coords"]).to(device)
    x_l = torch.from_numpy(high_h5["features"]).to(device)
    coord_l = torch.from_numpy(high_h5["coords"]).to(device)
    label = torch.tensor([label_id], dtype=torch.long, device=device)

    with torch.no_grad():
        y_prob, y_hat, total_loss = model(x_s, coord_s, x_l, coord_l, label, slide_id=slide_id)

    final_logits = getattr(model, "last_final_logits", None)
    low_logits = None
    high_logits = None
    concept_logits = None
    if getattr(model, "last_logit_breakdown", None):
        breakdown = model.last_logit_breakdown
        low_logits = breakdown.get("post_calibration", {}).get("low_evidence_logits")
        high_logits = breakdown.get("post_calibration", {}).get("high_evidence_logits")
        concept_logits = breakdown.get("post_calibration", {}).get("concept_only_logits")
    else:
        breakdown = None

    forward_tensors = {
        "probabilities": tensor_payload(y_prob, include_values=True),
        "predictions": tensor_payload(y_hat, include_values=True),
        "total_loss": tensor_payload(total_loss, include_values=True),
        "final_logits": tensor_payload(final_logits, include_values=True),
        "low_logits": tensor_payload(low_logits, include_values=True) if low_logits is not None else {"available": False, "reason": "unavailable"},
        "high_logits": tensor_payload(high_logits, include_values=True) if high_logits is not None else {"available": False, "reason": "unavailable"},
        "concept_logits": tensor_payload(concept_logits, include_values=True) if concept_logits is not None else {"available": False, "reason": "unavailable"},
        "low_visual_logits": tensor_payload(getattr(model, "last_low_visual_logits", None), include_values=True),
        "high_visual_logits": tensor_payload(getattr(model, "last_high_visual_logits", None), include_values=True),
        "visual_logits": tensor_payload(getattr(model, "last_visual_logits", None), include_values=True),
        "cross_scale_logits": tensor_payload(getattr(model, "last_cross_scale_logits", None), include_values=True),
    }
    return {
        "selected_slide": {
            "selection_rule": "first test slide in splits_{fold}.csv",
            "slide_id": slide_id,
            "label_id": label_id,
            "label_name": CLASS_NAMES[label_id],
            "low_h5": {
                "path": low_h5["path"],
                "sha256": low_h5["sha256"],
                "feature_shape": low_h5["feature_shape"],
                "feature_dtype": low_h5["feature_dtype"],
                "coords_shape": low_h5["coords_shape"],
                "coords_dtype": low_h5["coords_dtype"],
            },
            "high_h5": {
                "path": high_h5["path"],
                "sha256": high_h5["sha256"],
                "feature_shape": high_h5["feature_shape"],
                "feature_dtype": high_h5["feature_dtype"],
                "coords_shape": high_h5["coords_shape"],
                "coords_dtype": high_h5["coords_dtype"],
            },
        },
        "forward": {
            "outputs": forward_tensors,
            "probabilities": tensor_payload(F.softmax(final_logits, dim=1) if final_logits is not None else None, include_values=True),
            "logit_breakdown": convert_breakdown_payload(breakdown),
            "loss_breakdown": getattr(model, "last_loss_breakdown", None),
            "l2h_debug": convert_breakdown_payload(getattr(model, "last_l2h_retrieval_debug", None)),
            "ccra_breakdown": convert_breakdown_payload(getattr(model, "last_ccra_breakdown", None)),
            "dynamic_csg_breakdown": convert_breakdown_payload(getattr(model, "last_dynamic_csg_breakdown", None)),
        },
    }


def convert_breakdown_payload(payload):
    if payload is None:
        return {"available": False, "reason": "unavailable"}
    if isinstance(payload, dict):
        converted = {}
        for key, value in payload.items():
            if hasattr(value, "detach") or isinstance(value, np.ndarray):
                converted[key] = tensor_payload(value, include_values=True)
            elif isinstance(value, dict):
                converted[key] = convert_breakdown_payload(value)
            elif isinstance(value, (list, tuple)) and value and hasattr(value[0], "detach"):
                converted[key] = tensor_payload(np.asarray([item.detach().cpu().numpy() for item in value]), include_values=True)
            else:
                converted[key] = value
        return converted
    if hasattr(payload, "detach") or isinstance(payload, np.ndarray):
        return tensor_payload(payload, include_values=True)
    return payload


def worker_main(args):
    worker_import_project(args.project_root)

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_csv = args.project_root / "dataset_csv" / "all_data.csv"
    slide_id = select_probe_slide(args.split_dir, args.fold)
    label_name = load_label_for_slide(dataset_csv, slide_id)
    label_id = LABEL_TO_ID[str(label_name)]

    model = build_model()
    state_rows, state_hash = stable_state_dict_hash(model.state_dict())
    text_rows, text_hash = stable_state_dict_hash(model.text_encoder.state_dict())

    payload = {
        "probe_spec": {
            "project_root": str(args.project_root),
            "data_root": str(args.data_root),
            "split_dir": str(args.split_dir),
            "prompt_path": str(args.prompt_path),
            "concept_path": str(args.concept_path),
            "hf_cache_dir": str(args.hf_cache_dir),
            "fold_index": int(args.fold),
            "human_fold": int(args.fold) + 1,
            "seed": int(args.seed),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git_commit": current_commit(args.project_root),
        "versions": load_project_versions(),
        "biomedclip": resolve_model_metadata(args.hf_cache_dir),
        "state_dict": {
            "parameters": state_rows,
            "aggregate_sha256": state_hash,
        },
        "trainable_parameters": summarize_trainable_parameters(model),
        "text_encoder_state_dict": {
            "parameters": text_rows,
            "aggregate_sha256": text_hash,
        },
        "text_prompts": encode_text_prompts(model, args.prompt_path),
        "concept_prompts": encode_concept_prompts(model, args.concept_path),
    }
    payload.update(forward_probe(model, slide_id, label_id, args.data_root))
    dump_json(args.output, payload)


ARGS = parse_args()

if __name__ == "__main__":
    if ARGS.worker:
        worker_main(ARGS)
    else:
        run_launcher(ARGS)
