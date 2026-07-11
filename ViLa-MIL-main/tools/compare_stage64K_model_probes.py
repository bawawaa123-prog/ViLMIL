#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from stage64k_common import dump_json, flatten_numeric_payloads, markdown_table, tensor_to_numpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-probe", type=Path, required=True)
    parser.add_argument("--current-probe", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_scalars(old_value, current_value):
    if old_value == current_value:
        return {"equal": True, "old": old_value, "current": current_value}
    return {"equal": False, "old": old_value, "current": current_value}


def compare_arrays(name, old_value, current_value):
    if isinstance(old_value, dict) and old_value.get("available") is True and "values" in old_value:
        old_array = tensor_to_numpy(old_value["values"])
    else:
        old_array = tensor_to_numpy(old_value)
    if isinstance(current_value, dict) and current_value.get("available") is True and "values" in current_value:
        current_array = tensor_to_numpy(current_value["values"])
    else:
        current_array = tensor_to_numpy(current_value)
    if old_array is None or current_array is None:
        return {
            "available_both": False,
            "old_available": old_array is not None,
            "current_available": current_array is not None,
        }
    old_cast = old_array.astype(np.float64, copy=False)
    current_cast = current_array.astype(np.float64, copy=False)
    diff = np.abs(old_cast - current_cast)
    return {
        "available_both": True,
        "shape_equal": list(old_array.shape) == list(current_array.shape),
        "old_shape": list(old_array.shape),
        "current_shape": list(current_array.shape),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "allclose": bool(np.allclose(old_cast, current_cast, atol=1e-6, rtol=1e-5)),
        "old_sha256": old_value.get("sha256") if isinstance(old_value, dict) else None,
        "current_sha256": current_value.get("sha256") if isinstance(current_value, dict) else None,
        "field": name,
    }


def compare_trainable_parameters(old_probe, current_probe):
    old_rows = old_probe["trainable_parameters"]["parameters"]
    current_rows = current_probe["trainable_parameters"]["parameters"]
    return {
        "equal": old_rows == current_rows,
        "old_total_numel": old_probe["trainable_parameters"]["total_numel"],
        "current_total_numel": current_probe["trainable_parameters"]["total_numel"],
        "old_parameters": old_rows,
        "current_parameters": current_rows,
    }


def extract_payloads(probe):
    return {
        "text_prompt_low_token_ids": probe["text_prompts"]["token_ids"]["low"]["token_ids"],
        "text_prompt_high_token_ids": probe["text_prompts"]["token_ids"]["high"]["token_ids"],
        "text_prompt_low_embeddings": probe["text_prompts"]["embeddings"]["low"],
        "text_prompt_high_embeddings": probe["text_prompts"]["embeddings"]["high"],
        "concept_low_embeddings": probe["concept_prompts"]["model_low_prompt_features"],
        "concept_high_embeddings": probe["concept_prompts"]["model_high_prompt_features"],
        "final_logits": probe["forward"]["outputs"]["final_logits"],
        "probabilities": probe["forward"]["outputs"]["probabilities"],
        "low_logits": probe["forward"]["outputs"]["low_logits"],
        "high_logits": probe["forward"]["outputs"]["high_logits"],
        "visual_logits": probe["forward"]["outputs"]["visual_logits"],
        "cross_scale_logits": probe["forward"]["outputs"]["cross_scale_logits"],
    }


def compare_token_groups(old_probe, current_probe):
    fields = {
        "text_prompt_low_token_ids": (
            old_probe["text_prompts"]["token_ids"]["low"]["token_ids"],
            current_probe["text_prompts"]["token_ids"]["low"]["token_ids"],
        ),
        "text_prompt_high_token_ids": (
            old_probe["text_prompts"]["token_ids"]["high"]["token_ids"],
            current_probe["text_prompts"]["token_ids"]["high"]["token_ids"],
        ),
        "concept_low_class_aggregate_sha256": (
            old_probe["concept_prompts"]["token_ids"]["low"]["aggregate_sha256"],
            current_probe["concept_prompts"]["token_ids"]["low"]["aggregate_sha256"],
        ),
        "concept_high_class_aggregate_sha256": (
            old_probe["concept_prompts"]["token_ids"]["high"]["aggregate_sha256"],
            current_probe["concept_prompts"]["token_ids"]["high"]["aggregate_sha256"],
        ),
    }
    result = {}
    for name, values in fields.items():
        if isinstance(values[0], dict):
            result[name] = compare_arrays(name, values[0], values[1])
        else:
            result[name] = compare_scalars(values[0], values[1])
    return result


def compare_field_group(old_probe, current_probe):
    old_payloads = extract_payloads(old_probe)
    current_payloads = extract_payloads(current_probe)
    results = {}
    for name in sorted(old_payloads):
        results[name] = compare_arrays(name, old_payloads[name], current_payloads[name])
    return results


def overall_conclusion(comparison):
    numeric_fields = [
        comparison["numeric_fields"]["final_logits"],
        comparison["numeric_fields"]["probabilities"],
        comparison["numeric_fields"]["text_prompt_low_embeddings"],
        comparison["numeric_fields"]["text_prompt_high_embeddings"],
        comparison["numeric_fields"]["concept_low_embeddings"],
        comparison["numeric_fields"]["concept_high_embeddings"],
    ]
    all_numeric_close = all(item.get("available_both") and item.get("allclose") for item in numeric_fields)
    return {
        "biomedclip_weights_consistent": (
            comparison["biomedclip"]["snapshot_revision"]["equal"]
            and comparison["text_encoder_state_dict"]["aggregate_sha256"]["equal"]
        ),
        "trainable_parameters_consistent": comparison["trainable_parameters"]["equal"],
        "token_ids_consistent": all(
            item.get("equal", item.get("allclose", False))
            for item in comparison["token_ids"].values()
        ),
        "text_embeddings_consistent": (
            comparison["numeric_fields"]["text_prompt_low_embeddings"]["allclose"]
            and comparison["numeric_fields"]["text_prompt_high_embeddings"]["allclose"]
        ),
        "concept_embeddings_consistent": (
            comparison["numeric_fields"]["concept_low_embeddings"]["allclose"]
            and comparison["numeric_fields"]["concept_high_embeddings"]["allclose"]
        ),
        "initial_logits_consistent": (
            comparison["numeric_fields"]["final_logits"]["allclose"]
            and comparison["numeric_fields"]["low_logits"].get("allclose", False)
            and comparison["numeric_fields"]["high_logits"].get("allclose", False)
        ),
        "probabilities_consistent": comparison["numeric_fields"]["probabilities"]["allclose"],
        "training_pre_behavior_equivalent": all_numeric_close,
    }


def markdown_report(comparison):
    rows = []
    for name, payload in comparison["numeric_fields"].items():
        rows.append(
            [
                name,
                payload.get("available_both"),
                payload.get("shape_equal"),
                payload.get("max_abs_diff"),
                payload.get("mean_abs_diff"),
                payload.get("allclose"),
            ]
        )
    summary_rows = [
        ["biomedclip_weights_consistent", comparison["conclusion"]["biomedclip_weights_consistent"]],
        ["trainable_parameters_consistent", comparison["conclusion"]["trainable_parameters_consistent"]],
        ["token_ids_consistent", comparison["conclusion"]["token_ids_consistent"]],
        ["text_embeddings_consistent", comparison["conclusion"]["text_embeddings_consistent"]],
        ["concept_embeddings_consistent", comparison["conclusion"]["concept_embeddings_consistent"]],
        ["initial_logits_consistent", comparison["conclusion"]["initial_logits_consistent"]],
        ["probabilities_consistent", comparison["conclusion"]["probabilities_consistent"]],
        ["training_pre_behavior_equivalent", comparison["conclusion"]["training_pre_behavior_equivalent"]],
    ]
    return "\n".join(
        [
            "# Step64K Probe Comparison",
            "",
            "## Summary",
            "",
            markdown_table(["check", "result"], summary_rows),
            "",
            "## Numeric Comparisons",
            "",
            markdown_table(
                ["field", "available_both", "shape_equal", "max_abs_diff", "mean_abs_diff", "allclose"],
                rows,
            ),
            "",
        ]
    )


def main():
    args = parse_args()
    old_probe = load_json(args.old_probe)
    current_probe = load_json(args.current_probe)

    comparison = {
        "old_probe": str(args.old_probe),
        "current_probe": str(args.current_probe),
        "biomedclip": {
            "resolved_snapshot_path": compare_scalars(
                old_probe["biomedclip"]["resolved_snapshot_path"],
                current_probe["biomedclip"]["resolved_snapshot_path"],
            ),
            "snapshot_revision": compare_scalars(
                old_probe["biomedclip"]["snapshot_revision"],
                current_probe["biomedclip"]["snapshot_revision"],
            ),
        },
        "state_dict": {
            "aggregate_sha256": compare_scalars(
                old_probe["state_dict"]["aggregate_sha256"],
                current_probe["state_dict"]["aggregate_sha256"],
            ),
        },
        "text_encoder_state_dict": {
            "aggregate_sha256": compare_scalars(
                old_probe["text_encoder_state_dict"]["aggregate_sha256"],
                current_probe["text_encoder_state_dict"]["aggregate_sha256"],
            ),
        },
        "trainable_parameters": compare_trainable_parameters(old_probe, current_probe),
        "token_ids": compare_token_groups(old_probe, current_probe),
        "numeric_fields": compare_field_group(old_probe, current_probe),
    }
    comparison["conclusion"] = overall_conclusion(comparison)
    dump_json(args.output_json, comparison)
    args.output_md.write_text(markdown_report(comparison), encoding="utf-8")


if __name__ == "__main__":
    main()
