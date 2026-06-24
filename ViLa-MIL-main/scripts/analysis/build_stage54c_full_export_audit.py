from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_stage54_rce_evidence_interpretability"
DOCS_DIR = ROOT / "docs" / "stage54c_rce_full_direct_export_refresh"

FULL_DIR = RESULTS_DIR / "full"
WO_CSG_DIR = RESULTS_DIR / "wo_csg"
FULL_STAGE23_DIR = ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1"
FULL_STAGE52_DIR = ROOT / "results_stage52_rce_core_ablation" / "full_rce_v4_csg_rq16_5fold_e20_s1"
LEGACY_STAGE32_DIR = ROOT / "results_stage32" / "stage32_rce_v4_csg_evidence_export"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Step54C full direct export and its provenance refresh readiness.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--docs_dir", type=Path, default=DOCS_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_path_text(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\\", "/").rstrip("/")


def path_matches(manifest_path: str | None, expected_path: Path) -> bool:
    expected = normalize_path_text(str(expected_path))
    actual = normalize_path_text(manifest_path)
    return bool(actual) and (actual == expected or actual.endswith(rel(expected_path)))


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = safe_read_json(FULL_DIR / "stage32_manifest.json")
    full_slide_df = safe_read_csv(FULL_DIR / "stage32_slide_evidence_summary.csv")
    full_concept_df = safe_read_csv(FULL_DIR / "stage32_top_concepts_long.csv")
    full_csg_df = safe_read_csv(FULL_DIR / "stage32_top_csg_pairs.csv")

    wo_manifest = safe_read_json(WO_CSG_DIR / "stage32_manifest.json")
    wo_slide_df = safe_read_csv(WO_CSG_DIR / "stage32_slide_evidence_summary.csv")
    wo_concept_df = safe_read_csv(WO_CSG_DIR / "stage32_top_concepts_long.csv")
    wo_csg_df = safe_read_csv(WO_CSG_DIR / "stage32_top_csg_pairs.csv")

    full_results_dir = None if full_manifest is None else str(full_manifest.get("results_dir", ""))
    full_ckpt_path = None if full_manifest is None else str(full_manifest.get("ckpt_path", ""))
    full_model_type = None if full_manifest is None else str(full_manifest.get("model_type", ""))
    full_requested_model_type = None if full_manifest is None else str(full_manifest.get("requested_model_type", ""))

    full_matches_stage23 = path_matches(full_results_dir, FULL_STAGE23_DIR) and path_matches(
        full_ckpt_path, FULL_STAGE23_DIR / "s_0_checkpoint.pt"
    )
    full_matches_stage52 = path_matches(full_results_dir, FULL_STAGE52_DIR) and path_matches(
        full_ckpt_path, FULL_STAGE52_DIR / "s_0_checkpoint.pt"
    )
    full_verified = (
        full_manifest is not None
        and full_slide_df is not None
        and full_concept_df is not None
        and full_csg_df is not None
        and full_model_type == "RCE_MIL_BiomedCLIP"
        and full_requested_model_type == "RCE_MIL_BiomedCLIP"
        and (full_matches_stage23 or full_matches_stage52)
    )

    slide_field_alignment = (
        full_slide_df is not None
        and wo_slide_df is not None
        and list(full_slide_df.columns) == list(wo_slide_df.columns)
    )
    concept_field_alignment = (
        full_concept_df is not None
        and wo_concept_df is not None
        and list(full_concept_df.columns) == list(wo_concept_df.columns)
    )
    csg_field_alignment = full_csg_df is not None
    wo_csg_pairs_empty = wo_csg_df is None

    rows = [
        {
            "check_name": "full_export_directory_exists",
            "status": FULL_DIR.is_dir(),
            "details": rel(FULL_DIR),
        },
        {
            "check_name": "full_manifest_exists",
            "status": full_manifest is not None,
            "details": rel(FULL_DIR / "stage32_manifest.json"),
        },
        {
            "check_name": "full_slide_summary_exists",
            "status": full_slide_df is not None,
            "details": rel(FULL_DIR / "stage32_slide_evidence_summary.csv"),
        },
        {
            "check_name": "full_top_concepts_exists",
            "status": full_concept_df is not None,
            "details": rel(FULL_DIR / "stage32_top_concepts_long.csv"),
        },
        {
            "check_name": "full_top_csg_pairs_exists",
            "status": full_csg_df is not None,
            "details": rel(FULL_DIR / "stage32_top_csg_pairs.csv"),
        },
        {
            "check_name": "full_num_slides_reasonable",
            "status": full_slide_df is not None and len(full_slide_df.index) > 0,
            "details": 0 if full_slide_df is None else int(len(full_slide_df.index)),
        },
        {
            "check_name": "full_model_type_rce",
            "status": full_model_type == "RCE_MIL_BiomedCLIP" and full_requested_model_type == "RCE_MIL_BiomedCLIP",
            "details": f"requested={full_requested_model_type}; model={full_model_type}",
        },
        {
            "check_name": "full_points_to_stage23_or_stage52_full",
            "status": full_matches_stage23 or full_matches_stage52,
            "details": f"results_dir={full_results_dir}; ckpt_path={full_ckpt_path}",
        },
        {
            "check_name": "full_still_relies_on_stage32_fallback",
            "status": False if full_verified else True,
            "details": "false means Step54 full direct export is usable as preferred source",
        },
        {
            "check_name": "slide_fields_align_with_wo_csg",
            "status": slide_field_alignment,
            "details": "full and wo_csg slide summaries have identical columns",
        },
        {
            "check_name": "concept_fields_align_with_wo_csg",
            "status": concept_field_alignment,
            "details": "full and wo_csg concept tables have identical columns",
        },
        {
            "check_name": "csg_table_context",
            "status": csg_field_alignment,
            "details": "full has non-empty csg pair table; wo_csg csg table may be empty because the module is disabled",
        },
        {
            "check_name": "full_direct_export_verified",
            "status": full_verified,
            "details": "verified if manifest/model/checkpoint/results_dir all match an accepted full-RCE source",
        },
    ]

    audit_df = pd.DataFrame(rows)
    audit_csv_path = args.docs_dir / "stage54c_full_export_audit.csv"
    audit_df.to_csv(audit_csv_path, index=False, encoding="utf-8")

    md_lines = [
        "# Stage54C Full Export Audit",
        "",
        "## Key Findings",
        (
            "- Full direct export is present and verified against the full RCE main-model lineage."
            if full_verified
            else "- Full direct export is missing or failed verification."
        ),
        f"- Full export results_dir: `{full_results_dir}`",
        f"- Full export checkpoint: `{full_ckpt_path}`",
        f"- Full export model_type/requested_model_type: `{full_model_type}` / `{full_requested_model_type}`",
        (
            "- Full and wo_csg slide/concept tables are field-aligned for matched interpretability comparisons."
            if slide_field_alignment and concept_field_alignment
            else "- Full and wo_csg exports still need field-alignment review."
        ),
        (
            "- wo_csg top CSG pair table is empty/unreadable, which is expected because cross-scale graph is disabled."
            if wo_csg_pairs_empty
            else "- wo_csg top CSG pair table is present."
        ),
        "",
        "## Audit Table",
        markdown_table(rows, ["check_name", "status", "details"]),
        "",
        "## Interpretation",
        (
            "- Stage32 legacy fallback is no longer required as the preferred full-side source."
            if full_verified
            else "- Stage32 legacy fallback must remain available because the preferred full-side direct export is not yet fully verified."
        ),
        "- This audit checks export provenance and field readiness only; it does not claim any new training-time statistical result.",
        f"- Legacy fallback directory retained for reference: `{rel(LEGACY_STAGE32_DIR)}`",
    ]
    (args.docs_dir / "stage54c_full_export_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
