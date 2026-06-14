from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage37_lh_consistency_5fold.sh"
SUMMARY_SCRIPT = ROOT / "scripts" / "analysis" / "build_stage37_lh_consistency_summary.py"

VARIANTS = [
    "skeleton",
    "lh_l0001_m0",
    "lh_l0005_m0",
    "lh_l001_m0",
    "lh_l0005_m002",
    "lh_l001_m002",
    "lh_l001_m005",
    "lh_l005_m0",
    "lh_l005_m005",
    "all",
]
CONSISTENCY_VARIANTS = [variant for variant in VARIANTS if variant not in {"skeleton", "all"}]


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed reading {path}: {exc}") from exc


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    require(errors, RUN_SCRIPT.is_file(), f"missing run script: {RUN_SCRIPT}")
    require(errors, SUMMARY_SCRIPT.is_file(), f"missing summary script: {SUMMARY_SCRIPT}")

    if errors:
        for error in errors:
            print(f"[Error] {error}")
        sys.exit(1)

    try:
        run_text = read(RUN_SCRIPT)
        summary_text = read(SUMMARY_SCRIPT)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    for variant in VARIANTS:
        require(errors, variant in run_text, f"run script does not support variant `{variant}`")

    skeleton_block_start = run_text.find("skeleton)")
    next_variant_start = run_text.find("lh_l0001_m0)", skeleton_block_start)
    skeleton_block = run_text[skeleton_block_start:next_variant_start]
    require(
        errors,
        "--rce_use_low_high_consistency_loss" not in skeleton_block,
        "skeleton block must not pass --rce_use_low_high_consistency_loss",
    )

    for variant in CONSISTENCY_VARIANTS:
        require(errors, f"{variant})" in run_text, f"missing case arm for `{variant}`")
    for fragment in [
        "--rce_use_low_high_consistency_loss",
        "--rce_lh_consistency_lambda",
        "--rce_lh_consistency_margin",
    ]:
        require(errors, fragment in run_text, f"run script missing consistency arg `{fragment}`")

    for forbidden in [
        "--deg_use_region_graph",
        "--deg_use_concept_graph",
        "--rce_use_visual_evidence_gate",
    ]:
        require(errors, forbidden not in run_text, f"run script must not pass `{forbidden}`")

    for fragment in [
        "results_stage37",
        "lh_consistency_*_5fold_e*_s*",
        "stage37_lh_consistency_summary.csv",
        "stage37_lh_consistency_metric_deltas.csv",
        "stage37_lh_consistency_rankings.csv",
        "stage37_lh_consistency_report.md",
        "stage37_recommendations.json",
        "MAX_EPOCHS_FILTER",
        "SEED_FILTER",
    ]:
        require(errors, fragment in summary_text, f"summary script missing `{fragment}`")

    if errors:
        for error in errors:
            print(f"[Error] {error}")
        sys.exit(1)

    print("[OK] Stage37 low-high consistency 5-fold integrity check passed.")


if __name__ == "__main__":
    main()
