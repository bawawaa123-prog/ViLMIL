#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-results_stage39/final_evidence_package}"
STAGE24_DIR="${STAGE24_DIR:-results_stage24/stage24_rce_v4_csg_summary}"
STAGE28_DIR="${STAGE28_DIR:-results_stage28/stage28_deg_region_graph_summary}"
STAGE31_DIR="${STAGE31_DIR:-results_stage31/stage31_deg_concept_graph_summary}"
STAGE35_DIR="${STAGE35_DIR:-results_stage35}"
STAGE37_DIR="${STAGE37_DIR:-results_stage37/stage37_lh_consistency_summary}"
STAGE38_DIR="${STAGE38_DIR:-results_stage38/stage38_lh_consistency_failure_comparison}"
SKELETON_EVIDENCE_DIR="${SKELETON_EVIDENCE_DIR:-results_stage38/evidence_export_skeleton_fold0_test}"
LH_EVIDENCE_DIR="${LH_EVIDENCE_DIR:-results_stage38/evidence_export_lh_l001_m0_fold0_test}"

CMD=(
    "${PYTHON_BIN}" "scripts/analysis/build_stage39_final_evidence_package.py"
    "--output_dir" "${OUTPUT_DIR}"
    "--stage24_dir" "${STAGE24_DIR}"
    "--stage28_dir" "${STAGE28_DIR}"
    "--stage31_dir" "${STAGE31_DIR}"
    "--stage35_dir" "${STAGE35_DIR}"
    "--stage37_dir" "${STAGE37_DIR}"
    "--stage38_dir" "${STAGE38_DIR}"
    "--skeleton_evidence_dir" "${SKELETON_EVIDENCE_DIR}"
    "--lh_evidence_dir" "${LH_EVIDENCE_DIR}"
)

printf 'Running Step39 final evidence package builder:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
