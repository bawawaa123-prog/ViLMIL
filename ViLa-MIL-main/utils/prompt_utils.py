import json
import os

import torch
from torch.nn import functional as F


def _resolve_concept_prompt_path(path):
    if not path:
        return path
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join("dataset_csv", path),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset_csv", path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return path


def _load_prompt_items(prompt_json_path):
    resolved_path = _resolve_concept_prompt_path(prompt_json_path)
    if not resolved_path or not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Concept prompt JSON not found: {prompt_json_path}")

    with open(resolved_path, encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        prompt_items = obj.get("prompts", [])
    elif isinstance(obj, list):
        prompt_items = obj
    else:
        raise ValueError("Concept prompt JSON must be a dict with 'prompts' or a top-level list.")

    return resolved_path, prompt_items


def _group_concept_prompts(prompt_json_path, num_classes, class_names=None):
    resolved_path, prompt_items = _load_prompt_items(prompt_json_path)

    grouped_prompts = {
        "low": {class_id: [] for class_id in range(num_classes)},
        "high": {class_id: [] for class_id in range(num_classes)},
    }

    expected_name_by_id = {}
    if class_names is not None:
        expected_name_by_id = {idx: str(name) for idx, name in enumerate(class_names)}

    for item in prompt_items:
        if not bool(item.get("use_in_stage2", True)):
            continue

        scale = str(item.get("scale", "")).strip().lower()
        class_id = int(item.get("class_id"))
        prompt = str(item.get("prompt", "")).strip()
        class_name = str(item.get("class_name", "")).strip()

        if scale not in grouped_prompts:
            continue
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"Invalid class_id={class_id} for num_classes={num_classes}")
        if not prompt:
            continue

        if expected_name_by_id:
            expected_name = expected_name_by_id[class_id]
            if class_name and class_name != expected_name:
                raise ValueError(
                    f"Concept prompt class mismatch at class_id={class_id}: "
                    f"json class_name='{class_name}' vs expected '{expected_name}'"
                )

        grouped_prompts[scale][class_id].append(prompt)

    return resolved_path, grouped_prompts


def _encode_prompt_list(prompts, text_encoder, tokenizer, device):
    prev_mode = text_encoder.training
    text_encoder.eval()
    try:
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = text_encoder(text_tokens)
        return F.normalize(text_features.float(), dim=-1)
    finally:
        text_encoder.train(prev_mode)


def _build_group_features_and_texts(
    grouped_prompts,
    text_encoder,
    tokenizer,
    device,
    num_classes,
    dtype=None,
):
    def _encode_group(scale_name):
        prompt_counts = [len(grouped_prompts[scale_name][class_id]) for class_id in range(num_classes)]
        if any(count == 0 for count in prompt_counts):
            raise ValueError(
                f"Missing concept prompts for scale='{scale_name}': counts={prompt_counts}"
            )
        if len(set(prompt_counts)) != 1:
            raise ValueError(
                f"Dynamic prompt modes require balanced prompt counts per class for scale='{scale_name}', "
                f"but got counts={prompt_counts}"
            )

        class_prompt_features = []
        class_prompt_texts = []
        for class_id in range(num_classes):
            prompts = grouped_prompts[scale_name][class_id]
            text_features = _encode_prompt_list(prompts, text_encoder, tokenizer, device)
            class_prompt_features.append(text_features)
            class_prompt_texts.append(list(prompts))

        features = torch.stack(class_prompt_features, dim=0)
        if dtype is not None:
            features = features.to(dtype=dtype)
        return features, class_prompt_texts

    low_prompt_tensor, low_prompt_texts = _encode_group("low")
    high_prompt_tensor, high_prompt_texts = _encode_group("high")
    return low_prompt_tensor, high_prompt_tensor, low_prompt_texts, high_prompt_texts


def build_concept_text_features(
    prompt_json_path,
    text_encoder,
    tokenizer,
    device,
    num_classes,
    dtype=None,
    class_names=None,
):
    _, grouped_prompts = _group_concept_prompts(
        prompt_json_path=prompt_json_path,
        num_classes=num_classes,
        class_names=class_names,
    )

    def _encode_group(scale_name):
        class_embeddings = []
        for class_id in range(num_classes):
            prompts = grouped_prompts[scale_name][class_id]
            if len(prompts) == 0:
                raise ValueError(f"No concept prompts found for scale='{scale_name}', class_id={class_id}")

            text_features = _encode_prompt_list(prompts, text_encoder, tokenizer, device)
            mean_feature = text_features.mean(dim=0, keepdim=True)
            mean_feature = F.normalize(mean_feature, dim=-1)
            class_embeddings.append(mean_feature)

        features = torch.cat(class_embeddings, dim=0)
        if dtype is not None:
            features = features.to(dtype=dtype)
        return features

    low_text_features = _encode_group("low")
    high_text_features = _encode_group("high")
    return low_text_features, high_text_features


def build_concept_prompt_tensors(
    prompt_json_path,
    text_encoder,
    tokenizer,
    device,
    num_classes,
    dtype=None,
    class_names=None,
):
    _, grouped_prompts = _group_concept_prompts(
        prompt_json_path=prompt_json_path,
        num_classes=num_classes,
        class_names=class_names,
    )
    low_prompt_tensor, high_prompt_tensor, _, _ = _build_group_features_and_texts(
        grouped_prompts=grouped_prompts,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        num_classes=num_classes,
        dtype=dtype,
    )
    return low_prompt_tensor, high_prompt_tensor


def build_concept_prompt_bundle(
    prompt_json_path,
    text_encoder,
    tokenizer,
    device,
    num_classes,
    dtype=None,
    class_names=None,
):
    _, grouped_prompts = _group_concept_prompts(
        prompt_json_path=prompt_json_path,
        num_classes=num_classes,
        class_names=class_names,
    )
    return _build_group_features_and_texts(
        grouped_prompts=grouped_prompts,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        num_classes=num_classes,
        dtype=dtype,
    )
