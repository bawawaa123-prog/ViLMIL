from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, failures: list[str], label: str) -> None:
    if needle not in text:
        failures.append(f"Missing `{needle}` in {label}")


def main() -> int:
    failures: list[str] = []

    main_py = ROOT / "main.py"
    core_utils = ROOT / "utils" / "core_utils.py"
    deg_model = ROOT / "models" / "model_DEG_MIL_BiomedCLIP.py"

    main_text = read_text(main_py)
    core_text = read_text(core_utils)
    model_text = read_text(deg_model)

    cli_args = [
        "--rce_use_hcrc",
        "--rce_hcrc_alpha_init",
        "--rce_hcrc_num_anchors",
        "--rce_hcrc_num_high_children",
        "--rce_hcrc_proposal_radius",
        "--rce_hcrc_nms_radius",
        "--rce_hcrc_bbox_expand",
        "--rce_hcrc_coord_mode",
        "--rce_hcrc_scale_ratio",
        "--rce_hcrc_child_strategy",
        "--rce_hcrc_candidate_top_l",
        "--rce_hcrc_top_g_concepts",
        "--rce_hcrc_per_concept_top_m",
        "--rce_hcrc_prompt_topk",
        "--rce_hcrc_margin_weight",
        "--rce_hcrc_prompt_scale",
        "--rce_hcrc_min_child_count",
        "--rce_hcrc_export_debug",
    ]
    for arg in cli_args:
        require(main_text, arg, failures, "main.py")

    config_fields = [
        "config.rce_use_hcrc",
        "config.rce_hcrc_alpha_init",
        "config.rce_hcrc_num_anchors",
        "config.rce_hcrc_num_high_children",
        "config.rce_hcrc_proposal_radius",
        "config.rce_hcrc_nms_radius",
        "config.rce_hcrc_bbox_expand",
        "config.rce_hcrc_coord_mode",
        "config.rce_hcrc_scale_ratio",
        "config.rce_hcrc_child_strategy",
        "config.rce_hcrc_candidate_top_l",
        "config.rce_hcrc_top_g_concepts",
        "config.rce_hcrc_per_concept_top_m",
        "config.rce_hcrc_prompt_topk",
        "config.rce_hcrc_margin_weight",
        "config.rce_hcrc_prompt_scale",
        "config.rce_hcrc_min_child_count",
        "config.rce_hcrc_export_debug",
    ]
    for field in config_fields:
        require(core_text, field, failures, "utils/core_utils.py")

    model_needles = [
        "self.rce_use_hcrc",
        "self.hcrc_fusion_gate",
        "self.last_hcrc_logits",
        "self.last_hcrc_empty_anchor_ratio",
        "final_logits = final_logits + hcrc_alpha * hcrc_result[\"hcrc_logits\"]",
        "getattr(config, \"rce_use_hcrc\", False)",
    ]
    for needle in model_needles:
        require(model_text, needle, failures, "models/model_DEG_MIL_BiomedCLIP.py")

    if failures:
        print("[FAIL] Step43 HCRC integrity check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[PASS] Step43 HCRC integrity check passed.")
    print(f"[Checked] {main_py}")
    print(f"[Checked] {core_utils}")
    print(f"[Checked] {deg_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
