from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_stage54_rce_evidence_interpretability"
DOCS_DIR = ROOT / "docs" / "stage54b_rce_evidence_figure_polish"

FULL_RUN_DIR = ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1"
WO_CSG_RUN_DIR = ROOT / "results_stage52_rce_core_ablation" / "wo_csg_5fold_e20_s1"

FULL_DIRECT_DIR = RESULTS_DIR / "full"
WO_CSG_DIRECT_DIR = RESULTS_DIR / "wo_csg"
FULL_STAGE32_FALLBACK_DIR = ROOT / "results_stage32" / "stage32_rce_v4_csg_evidence_export"

BASE_FIELDS = ["slide_id", "fold", "split", "label", "pred"]
PROBABILITY_FIELDS = ["prob_class_0", "prob_class_1"]
VISUAL_FIELDS = [
    "visual_logit_class_0",
    "visual_logit_class_1",
    "visual_alpha",
]
CSG_FIELDS = [
    "csg_logit_class_0",
    "csg_logit_class_1",
    "csg_alpha",
    "top_csg_pair_class_0",
    "top_csg_pair_class_1",
]
CONCEPT_SUMMARY_FIELDS = [
    "top_low_concepts_for_pred",
    "top_high_concepts_for_pred",
    "top_low_concepts_for_true",
    "top_high_concepts_for_true",
]
CONCEPT_LONG_FIELDS = [
    "slide_id",
    "fold",
    "scale",
    "class_type",
    "concept_rank",
    "concept_id",
    "contribution",
]
FALLBACK_CONSUMERS = [
    "stage54b_case_level_metadata.csv",
    "stage54b_region_concept_heatmap.png",
    "stage54b_low_high_concept_comparison.png",
    "stage54b_full_vs_wo_csg_concept_ranking_delta.png(full side only)",
    "stage54b_correct_case_evidence_components.png",
    "stage54b_failure_case_evidence_components.png",
    "stage54b_csg_benefit_case_comparison.png(full side only)",
    "stage54b_aggregate_top_concept_frequency.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Step54B evidence sources without running export.")
    parser.add_argument("--docs_dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_path_text(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\\", "/").rstrip("/")


def path_matches(manifest_path: str | None, expected_path: Path) -> bool:
    expected = normalize_path_text(str(expected_path))
    actual = normalize_path_text(manifest_path)
    return bool(actual) and (actual == expected or actual.endswith(rel(expected_path)))


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def present_and_non_null(df: pd.DataFrame | None, columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    if df is None:
        return [], columns[:], []
    present = [column for column in columns if column in df.columns]
    missing = [column for column in columns if column not in df.columns]
    non_null = [column for column in present if df[column].notna().any()]
    return present, missing, non_null


def join_items(items: list[str]) -> str:
    return "; ".join(items) if items else "none"


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def inspect_source(
    *,
    source_id: str,
    variant: str,
    source_role: str,
    evidence_dir: Path,
    expected_results_dir: Path,
    expected_checkpoint: Path,
    expected_model_type: str,
    allow_fallback: bool,
    fallback_consumers: list[str],
) -> dict[str, object]:
    manifest_path = evidence_dir / "stage32_manifest.json"
    slide_summary_path = evidence_dir / "stage32_slide_evidence_summary.csv"
    top_concepts_path = evidence_dir / "stage32_top_concepts_long.csv"
    top_csg_pairs_path = evidence_dir / "stage32_top_csg_pairs.csv"
    export_report_path = evidence_dir / "stage32_evidence_export_report.md"

    manifest = safe_read_json(manifest_path)
    slide_df = safe_read_csv(slide_summary_path)
    concept_df = safe_read_csv(top_concepts_path)
    csg_df = safe_read_csv(top_csg_pairs_path)

    results_dir_text = None if manifest is None else str(manifest.get("results_dir", ""))
    ckpt_path_text = None if manifest is None else str(manifest.get("ckpt_path", ""))
    model_type = None if manifest is None else str(manifest.get("model_type", ""))
    requested_model_type = None if manifest is None else str(manifest.get("requested_model_type", ""))

    matched_results_dir = path_matches(results_dir_text, expected_results_dir)
    matched_checkpoint = path_matches(ckpt_path_text, expected_checkpoint)
    matched_model_type = model_type == expected_model_type if model_type else False

    base_present, base_missing, _ = present_and_non_null(slide_df, BASE_FIELDS)
    prob_present, prob_missing, prob_non_null = present_and_non_null(slide_df, PROBABILITY_FIELDS)
    visual_present, visual_missing, visual_non_null = present_and_non_null(slide_df, VISUAL_FIELDS)
    csg_present, csg_missing, csg_non_null = present_and_non_null(slide_df, CSG_FIELDS)
    concept_summary_present, concept_summary_missing, concept_summary_non_null = present_and_non_null(
        slide_df, CONCEPT_SUMMARY_FIELDS
    )
    concept_long_present, concept_long_missing, concept_long_non_null = present_and_non_null(concept_df, CONCEPT_LONG_FIELDS)

    notes: list[str] = []
    if not evidence_dir.exists():
        notes.append("evidence directory missing")
    if manifest is None:
        notes.append("manifest missing or unreadable")
    if slide_df is None:
        notes.append("slide summary missing or unreadable")
    if concept_df is None:
        notes.append("top concept table missing or unreadable")
    if csg_df is None:
        notes.append("top csg pair table missing or unreadable")
    if allow_fallback:
        notes.append("fallback source only; not a direct Step54 full export")
    if manifest is not None and not matched_results_dir:
        notes.append("manifest results_dir does not match expected variant directory")
    if manifest is not None and not matched_checkpoint:
        notes.append("manifest ckpt_path does not match expected checkpoint")
    if manifest is not None and not matched_model_type:
        notes.append("manifest model_type does not match expected model type")
    if prob_present and not prob_non_null:
        notes.append("probability columns exist but contain no non-null values")
    if csg_present and not csg_non_null:
        notes.append("CSG-related columns exist but contain no non-null values")

    if slide_df is not None and manifest is not None and matched_results_dir and matched_checkpoint and matched_model_type:
        provenance_status = "direct_export_verified"
    elif slide_df is not None and allow_fallback:
        provenance_status = "fallback_reuse"
    elif slide_df is not None:
        provenance_status = "historical_reuse_or_mismatch"
    else:
        provenance_status = "missing"

    return {
        "source_id": source_id,
        "variant": variant,
        "source_role": source_role,
        "evidence_dir": rel(evidence_dir),
        "provenance_status": provenance_status,
        "manifest_present": manifest is not None,
        "slide_summary_present": slide_df is not None,
        "top_concepts_present": concept_df is not None,
        "top_csg_pairs_present": csg_df is not None,
        "export_report_present": export_report_path.is_file(),
        "expected_results_dir": rel(expected_results_dir),
        "manifest_results_dir": results_dir_text or "",
        "expected_checkpoint": rel(expected_checkpoint),
        "manifest_ckpt_path": ckpt_path_text or "",
        "expected_model_type": expected_model_type,
        "manifest_model_type": model_type or "",
        "manifest_requested_model_type": requested_model_type or "",
        "matches_expected_results_dir": matched_results_dir,
        "matches_expected_checkpoint": matched_checkpoint,
        "matches_expected_model_type": matched_model_type,
        "num_slides": 0 if slide_df is None else int(len(slide_df.index)),
        "base_fields_present": join_items(base_present),
        "base_fields_missing": join_items(base_missing),
        "probability_fields_present": join_items(prob_present),
        "probability_fields_missing": join_items(prob_missing),
        "probability_fields_non_null": join_items(prob_non_null),
        "concept_summary_fields_present": join_items(concept_summary_present),
        "concept_summary_fields_missing": join_items(concept_summary_missing),
        "concept_summary_fields_non_null": join_items(concept_summary_non_null),
        "concept_long_fields_present": join_items(concept_long_present),
        "concept_long_fields_missing": join_items(concept_long_missing),
        "concept_long_fields_non_null": join_items(concept_long_non_null),
        "visual_fields_present": join_items(visual_present),
        "visual_fields_missing": join_items(visual_missing),
        "visual_fields_non_null": join_items(visual_non_null),
        "csg_fields_present": join_items(csg_present),
        "csg_fields_missing": join_items(csg_missing),
        "csg_fields_non_null": join_items(csg_non_null),
        "historical_reuse_only": provenance_status in {"fallback_reuse", "historical_reuse_or_mismatch"},
        "uses_stage32_fallback": allow_fallback,
        "fallback_consumers": join_items(fallback_consumers if allow_fallback else []),
        "manual_export_command": (
            f"MODE={variant} bash scripts/experiments/run_stage54_export_rce_evidence.sh"
            if variant in {"full", "wo_csg"}
            else "MODE=both bash scripts/experiments/run_stage54_export_rce_evidence.sh"
        ),
        "notes": join_items(notes),
    }


def build_markdown(rows: list[dict[str, object]]) -> str:
    row_map = {str(row["source_id"]): row for row in rows}
    full_direct = row_map.get("full_direct_expected", {})
    full_fallback = row_map.get("full_stage32_fallback", {})
    wo_csg_direct = row_map.get("wo_csg_direct_export", {})

    full_direct_ready = full_direct.get("provenance_status") == "direct_export_verified"
    wo_csg_ready = wo_csg_direct.get("provenance_status") == "direct_export_verified"

    key_lines = [
        (
            "- Full direct export under `results_stage54_rce_evidence_interpretability/full/` is present and "
            "verified against `RCE-v4-CSG-a01-rq16`."
            if full_direct_ready
            else "- Full direct export under `results_stage54_rce_evidence_interpretability/full/` is still missing or unverified."
        ),
        (
            "- `wo_csg` direct export is present and points to `results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1`."
            if wo_csg_ready
            else "- `wo_csg` direct export is missing or unverified."
        ),
        (
            "- The Stage32 legacy fallback remains available only as backup and should no longer be the preferred full-side source."
            if full_direct_ready
            else "- The currently reused full-side fallback under `results_stage32/stage32_rce_v4_csg_evidence_export/` is not a direct export of `RCE-v4-CSG-a01-rq16`."
        ),
        (
            "- Full vs `wo_csg` evidence-level figures can now be refreshed from Step54 direct exports on both sides."
            if full_direct_ready and wo_csg_ready
            else "- Any full vs `wo_csg` evidence-level figure in Step54B must disclose the full-side provenance mismatch when fallback is used."
        ),
    ]

    table_columns = [
        "source_id",
        "variant",
        "source_role",
        "evidence_dir",
        "provenance_status",
        "matches_expected_results_dir",
        "matches_expected_checkpoint",
        "matches_expected_model_type",
        "num_slides",
        "uses_stage32_fallback",
        "fallback_consumers",
        "notes",
    ]
    table_rows = pd.DataFrame(rows)[table_columns].to_dict("records")

    markdown_lines = [
        "# Stage54B Evidence Source Audit",
        "",
        "## Key Findings",
        *key_lines,
        "",
        "## Manual Export Commands",
        "- `MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh`",
        "- `MODE=wo_csg bash scripts/experiments/run_stage54_export_rce_evidence.sh`",
        "- `MODE=both bash scripts/experiments/run_stage54_export_rce_evidence.sh`",
        "- These commands were audited only; Step54B did not execute them.",
        "",
        "## Audit Table",
        markdown_table(table_rows, table_columns),
        "",
        "## Interpretation Guardrails",
        (
            "- Preferred full-side provenance is now the Step54 direct export."
            if full_direct_ready
            else "- Do not describe the current full-side fallback as a direct Step54 export of the main model."
        ),
        (
            "- Full vs `wo_csg` evidence comparisons can be described as matched Step54 direct-export illustrations, but not as training-significance evidence."
            if full_direct_ready and wo_csg_ready
            else "- Do not call the current full vs `wo_csg` evidence comparison a fully same-source 5-fold comparison."
        ),
        "- Single-case evidence illustrations from the fallback may still be used for interpretability examples if provenance is stated explicitly.",
        "- CSG-related fields in the `wo_csg` export are structurally present but expected to be empty because the module is disabled.",
    ]
    return "\n".join(markdown_lines) + "\n"


def main() -> None:
    args = parse_args()
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        inspect_source(
            source_id="full_direct_expected",
            variant="full",
            source_role="expected_step54_direct_export",
            evidence_dir=FULL_DIRECT_DIR,
            expected_results_dir=FULL_RUN_DIR,
            expected_checkpoint=FULL_RUN_DIR / "s_0_checkpoint.pt",
            expected_model_type="RCE_MIL_BiomedCLIP",
            allow_fallback=False,
            fallback_consumers=[],
        ),
        inspect_source(
            source_id="full_stage32_fallback",
            variant="full",
            source_role="legacy_fallback_candidate",
            evidence_dir=FULL_STAGE32_FALLBACK_DIR,
            expected_results_dir=FULL_RUN_DIR,
            expected_checkpoint=FULL_RUN_DIR / "s_0_checkpoint.pt",
            expected_model_type="RCE_MIL_BiomedCLIP",
            allow_fallback=True,
            fallback_consumers=FALLBACK_CONSUMERS,
        ),
        inspect_source(
            source_id="wo_csg_direct_export",
            variant="wo_csg",
            source_role="step54_direct_export",
            evidence_dir=WO_CSG_DIRECT_DIR,
            expected_results_dir=WO_CSG_RUN_DIR,
            expected_checkpoint=WO_CSG_RUN_DIR / "s_0_checkpoint.pt",
            expected_model_type="RCE_MIL_BiomedCLIP",
            allow_fallback=False,
            fallback_consumers=[],
        ),
    ]

    audit_df = pd.DataFrame(rows)
    docs_csv_path = args.docs_dir / "stage54b_evidence_source_audit.csv"
    results_csv_path = args.results_dir / "stage54b_evidence_source_audit.csv"
    md_path = args.docs_dir / "stage54b_evidence_source_audit.md"

    audit_df.to_csv(docs_csv_path, index=False, encoding="utf-8")
    audit_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    md_path.write_text(build_markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
