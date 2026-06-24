from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "stage55_rce_frozen_release"
RESULTS_DIR = ROOT / "results_stage55_rce_frozen_release"

STAGE23_DIR = ROOT / "results_stage23" / "rce_v4_csg_a01_rq16_5fold_e20_s1"
STAGE52_DIR = ROOT / "results_stage52_rce_core_ablation"
STAGE52_FULL_DIR = STAGE52_DIR / "full_rce_v4_csg_rq16_5fold_e20_s1"
STAGE52B_MD = ROOT / "docs" / "stage52b_rce_ablation_table_paper.md"
STAGE52B_CSV = ROOT / "docs" / "stage52b_rce_ablation_table_paper.csv"
STAGE52B_INTERPRETATION = ROOT / "docs" / "stage52b_rce_ablation_interpretation.md"
STAGE53_DIR = ROOT / "docs" / "stage53_rce_paper_ready_package"
STAGE54_DIR = ROOT / "docs" / "stage54_rce_evidence_interpretability_package"
STAGE54B_DIR = ROOT / "docs" / "stage54b_rce_evidence_figure_polish"
STAGE54C_DIR = ROOT / "docs" / "stage54c_rce_full_direct_export_refresh"
STAGE54_RESULTS_DIR = ROOT / "results_stage54_rce_evidence_interpretability"
FULL_EVIDENCE_DIR = STAGE54_RESULTS_DIR / "full"
WO_CSG_EVIDENCE_DIR = STAGE54_RESULTS_DIR / "wo_csg"
STAGE54B_CASE_METADATA = STAGE54_RESULTS_DIR / "stage54b_case_level_metadata.csv"
STAGE54B_AUDIT_RESULTS_CSV = STAGE54_RESULTS_DIR / "stage54b_evidence_source_audit.csv"
STAGE54B_AUDIT_DOC_CSV = STAGE54B_DIR / "stage54b_evidence_source_audit.csv"
STAGE54B_FIGURE_INDEX_CSV = STAGE54B_DIR / "stage54b_figure_index.csv"
STAGE54B_FIGURE_INDEX_MD = STAGE54B_DIR / "stage54b_figure_index.md"
STAGE54B_CAPTIONS_MD = STAGE54B_DIR / "stage54b_paper_figure_caption_drafts.md"
STAGE54C_AUDIT_CSV = STAGE54C_DIR / "stage54c_full_export_audit.csv"

MAIN_MODEL_FILE = ROOT / "models" / "model_RCE_MIL_BiomedCLIP.py"
MAIN_TRAIN_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage23_rce_v4_csg_region_queries_5fold.sh"
ABLATION_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage52_rce_core_ablation_5fold.sh"
EVIDENCE_SCRIPT = ROOT / "scripts" / "experiments" / "run_stage54_export_rce_evidence.sh"
BUILD_SCRIPT = ROOT / "scripts" / "analysis" / "build_stage55_rce_frozen_release.py"
MAIN_PY = ROOT / "main.py"
CORE_UTILS_PY = ROOT / "utils" / "core_utils.py"

METRIC_MAP = {
    "AUC": "test_auc",
    "ACC": "test_acc",
    "F1": "test_f1",
    "BACC": "balanced_acc",
    "PR-AUC": "pr_auc",
}
NEGATIVE_SECTION_HINTS = ("avoid", "not supported", "limitation", "guardrail")
NEGATIVE_LINE_HINTS = (
    "none of",
    "do not",
    "should not",
    "cannot",
    "can't",
    "is not",
    "are not",
    "not a",
    "not the",
    "not fully",
    "not support",
    "unsupported",
    "avoid",
)


@dataclass
class CheckResult:
    check_id: str
    status: str
    severity: str
    details: str
    evidence: str
    action: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Step55 RCE frozen release package without training or export.")
    parser.add_argument("--docs_dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--check", action="store_true", help="Only validate and print a summary without rewriting files.")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number):
        return None
    return number


def fmt(value: float | None, digits: int = 4, missing: str = "missing") -> str:
    if value is None:
        return missing
    return f"{value:.{digits}f}"


def fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def read_text_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def read_json_optional(path: Path) -> dict[str, Any] | None:
    text = read_text_optional(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def markdown_table_from_df(df: pd.DataFrame) -> str:
    safe_df = df.fillna("").astype(str)
    return markdown_table(safe_df.to_dict(orient="records"), list(safe_df.columns))


def shell_output(args: list[str]) -> str:
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def get_git_state() -> dict[str, Any]:
    commit_hash = shell_output(["git", "rev-parse", "HEAD"])
    branch = shell_output(["git", "branch", "--show-current"])
    status_text = shell_output(["git", "status", "--short"])
    status_lines = [line for line in status_text.splitlines() if line.strip()]
    return {
        "commit_hash": commit_hash,
        "branch": branch,
        "status_short": status_lines,
        "has_uncommitted_changes": bool(status_lines),
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def get_main_result_rows() -> tuple[pd.DataFrame, dict[str, float | None], dict[str, float | None]]:
    result_csv = read_csv_optional(STAGE23_DIR / "result.csv")
    if result_csv is None or "metric" not in result_csv.columns:
        rows = []
        for label in METRIC_MAP:
            rows.append(
                {
                    "metric": label,
                    "mean": None,
                    "std": None,
                    "source_dir": rel(STAGE23_DIR),
                    "source_file": rel(STAGE23_DIR / "result.csv"),
                }
            )
        return pd.DataFrame(rows), {}, {}

    mean_row = result_csv[result_csv["metric"] == "mean"]
    std_row = result_csv[result_csv["metric"] == "std"]
    mean_series = mean_row.iloc[0] if not mean_row.empty else pd.Series(dtype=object)
    std_series = std_row.iloc[0] if not std_row.empty else pd.Series(dtype=object)

    mean_metrics: dict[str, float | None] = {}
    std_metrics: dict[str, float | None] = {}
    rows = []
    for label, column in METRIC_MAP.items():
        mean_value = safe_float(mean_series.get(column))
        std_value = safe_float(std_series.get(column))
        mean_metrics[label] = mean_value
        std_metrics[label] = std_value
        rows.append(
            {
                "metric": label,
                "mean": fmt(mean_value),
                "std": fmt(std_value),
                "source_dir": rel(STAGE23_DIR),
                "source_file": rel(STAGE23_DIR / "result.csv"),
            }
        )
    return pd.DataFrame(rows), mean_metrics, std_metrics


def parse_stage52b_table() -> pd.DataFrame:
    source_df = read_csv_optional(STAGE52B_CSV)
    columns = [
        "variant",
        "paper_label",
        "status",
        "AUC_mean",
        "AUC_std",
        "ACC_mean",
        "ACC_std",
        "F1_mean",
        "F1_std",
        "BACC_mean",
        "BACC_std",
        "PR_AUC_mean",
        "PR_AUC_std",
        "delta_AUC_vs_full",
        "delta_ACC_vs_full",
        "delta_F1_vs_full",
        "delta_BACC_vs_full",
        "delta_PR_AUC_vs_full",
        "p_AUC_vs_full",
        "p_ACC_vs_full",
        "p_F1_vs_full",
        "p_BACC_vs_full",
        "p_PR_AUC_vs_full",
        "source_dir",
        "source_file",
    ]
    if source_df is None:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for _, row in source_df.iterrows():
        rows.append(
            {
                "variant": row.get("variant", ""),
                "paper_label": row.get("paper_label", ""),
                "status": row.get("status", ""),
                "AUC_mean": fmt(safe_float(row.get("test_auc_mean"))),
                "AUC_std": fmt(safe_float(row.get("test_auc_std"))),
                "ACC_mean": fmt(safe_float(row.get("test_acc_mean"))),
                "ACC_std": fmt(safe_float(row.get("test_acc_std"))),
                "F1_mean": fmt(safe_float(row.get("test_f1_mean"))),
                "F1_std": fmt(safe_float(row.get("test_f1_std"))),
                "BACC_mean": fmt(safe_float(row.get("balanced_acc_mean"))),
                "BACC_std": fmt(safe_float(row.get("balanced_acc_std"))),
                "PR_AUC_mean": fmt(safe_float(row.get("pr_auc_mean"))),
                "PR_AUC_std": fmt(safe_float(row.get("pr_auc_std"))),
                "delta_AUC_vs_full": fmt(safe_float(row.get("delta_test_auc_vs_full"))),
                "delta_ACC_vs_full": fmt(safe_float(row.get("delta_test_acc_vs_full"))),
                "delta_F1_vs_full": fmt(safe_float(row.get("delta_test_f1_vs_full"))),
                "delta_BACC_vs_full": fmt(safe_float(row.get("delta_balanced_acc_vs_full"))),
                "delta_PR_AUC_vs_full": fmt(safe_float(row.get("delta_pr_auc_vs_full"))),
                "p_AUC_vs_full": fmt(safe_float(row.get("pvalue_test_auc_vs_full"))),
                "p_ACC_vs_full": fmt(safe_float(row.get("pvalue_test_acc_vs_full"))),
                "p_F1_vs_full": fmt(safe_float(row.get("pvalue_test_f1_vs_full"))),
                "p_BACC_vs_full": fmt(safe_float(row.get("pvalue_balanced_acc_vs_full"))),
                "p_PR_AUC_vs_full": fmt(safe_float(row.get("pvalue_pr_auc_vs_full"))),
                "source_dir": str(row.get("path", "")),
                "source_file": rel(STAGE52B_CSV),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def path_matches_text(value: str | None, expected_path: Path) -> bool:
    if not value:
        return False
    normalized_value = value.replace("\\", "/").rstrip("/")
    normalized_expected = str(expected_path).replace("\\", "/").rstrip("/")
    return normalized_value == normalized_expected or normalized_value.endswith(rel(expected_path))


def collect_evidence_inventory() -> tuple[pd.DataFrame, dict[str, Any]]:
    full_manifest = read_json_optional(FULL_EVIDENCE_DIR / "stage32_manifest.json")
    wo_manifest = read_json_optional(WO_CSG_EVIDENCE_DIR / "stage32_manifest.json")
    full_slide_df = read_csv_optional(FULL_EVIDENCE_DIR / "stage32_slide_evidence_summary.csv")
    full_concepts_df = read_csv_optional(FULL_EVIDENCE_DIR / "stage32_top_concepts_long.csv")
    full_csg_df = read_csv_optional(FULL_EVIDENCE_DIR / "stage32_top_csg_pairs.csv")
    wo_slide_df = read_csv_optional(WO_CSG_EVIDENCE_DIR / "stage32_slide_evidence_summary.csv")
    wo_concepts_df = read_csv_optional(WO_CSG_EVIDENCE_DIR / "stage32_top_concepts_long.csv")
    wo_csg_df = read_csv_optional(WO_CSG_EVIDENCE_DIR / "stage32_top_csg_pairs.csv")
    case_df = read_csv_optional(STAGE54B_CASE_METADATA)
    audit_results_df = read_csv_optional(STAGE54B_AUDIT_RESULTS_CSV)
    full_results_dir = "" if full_manifest is None else str(full_manifest.get("results_dir", ""))
    full_ckpt_path = "" if full_manifest is None else str(full_manifest.get("ckpt_path", ""))
    full_model_type = "" if full_manifest is None else str(full_manifest.get("model_type", ""))
    full_requested_model_type = "" if full_manifest is None else str(full_manifest.get("requested_model_type", ""))

    summary = {
        "full_manifest_points_to_stage23": path_matches_text(full_results_dir, STAGE23_DIR)
        and path_matches_text(full_ckpt_path, STAGE23_DIR / "s_0_checkpoint.pt"),
        "full_model_type_is_rce": full_model_type == "RCE_MIL_BiomedCLIP"
        and full_requested_model_type == "RCE_MIL_BiomedCLIP",
        "full_results_dir": full_results_dir,
        "full_ckpt_path": full_ckpt_path,
        "full_model_type": full_model_type,
        "full_requested_model_type": full_requested_model_type,
        "full_num_slides": 0 if full_slide_df is None else int(len(full_slide_df.index)),
        "wo_csg_num_slides": 0 if wo_slide_df is None else int(len(wo_slide_df.index)),
    }

    rows = [
        {
            "artifact_id": "full_evidence_dir",
            "variant": "full",
            "artifact_type": "directory",
            "path": rel(FULL_EVIDENCE_DIR),
            "exists": FULL_EVIDENCE_DIR.is_dir(),
            "source_stage": "Step54C preferred source",
            "results_dir": full_results_dir,
            "ckpt_path": full_ckpt_path,
            "model_type": full_model_type,
            "requested_model_type": full_requested_model_type,
            "matches_stage23_main": summary["full_manifest_points_to_stage23"],
            "uses_stage32_fallback": False,
            "row_count": "",
            "notes": "Preferred full evidence source after Step54C.",
        },
        {
            "artifact_id": "full_manifest",
            "variant": "full",
            "artifact_type": "json",
            "path": rel(FULL_EVIDENCE_DIR / "stage32_manifest.json"),
            "exists": (FULL_EVIDENCE_DIR / "stage32_manifest.json").is_file(),
            "source_stage": "Step54C preferred source",
            "results_dir": full_results_dir,
            "ckpt_path": full_ckpt_path,
            "model_type": full_model_type,
            "requested_model_type": full_requested_model_type,
            "matches_stage23_main": summary["full_manifest_points_to_stage23"],
            "uses_stage32_fallback": False,
            "row_count": "",
            "notes": "Manifest should point to Stage23 main run and fold0 checkpoint.",
        },
        {
            "artifact_id": "full_slide_summary",
            "variant": "full",
            "artifact_type": "csv",
            "path": rel(FULL_EVIDENCE_DIR / "stage32_slide_evidence_summary.csv"),
            "exists": (FULL_EVIDENCE_DIR / "stage32_slide_evidence_summary.csv").is_file(),
            "source_stage": "Step54C preferred source",
            "results_dir": full_results_dir,
            "ckpt_path": full_ckpt_path,
            "model_type": full_model_type,
            "requested_model_type": full_requested_model_type,
            "matches_stage23_main": summary["full_manifest_points_to_stage23"],
            "uses_stage32_fallback": False,
            "row_count": 0 if full_slide_df is None else int(len(full_slide_df.index)),
            "notes": "Fold0/test slide evidence summary.",
        },
        {
            "artifact_id": "full_top_concepts",
            "variant": "full",
            "artifact_type": "csv",
            "path": rel(FULL_EVIDENCE_DIR / "stage32_top_concepts_long.csv"),
            "exists": (FULL_EVIDENCE_DIR / "stage32_top_concepts_long.csv").is_file(),
            "source_stage": "Step54C preferred source",
            "results_dir": full_results_dir,
            "ckpt_path": full_ckpt_path,
            "model_type": full_model_type,
            "requested_model_type": full_requested_model_type,
            "matches_stage23_main": summary["full_manifest_points_to_stage23"],
            "uses_stage32_fallback": False,
            "row_count": 0 if full_concepts_df is None else int(len(full_concepts_df.index)),
            "notes": "Concept-level evidence export.",
        },
        {
            "artifact_id": "full_top_csg_pairs",
            "variant": "full",
            "artifact_type": "csv",
            "path": rel(FULL_EVIDENCE_DIR / "stage32_top_csg_pairs.csv"),
            "exists": (FULL_EVIDENCE_DIR / "stage32_top_csg_pairs.csv").is_file(),
            "source_stage": "Step54C preferred source",
            "results_dir": full_results_dir,
            "ckpt_path": full_ckpt_path,
            "model_type": full_model_type,
            "requested_model_type": full_requested_model_type,
            "matches_stage23_main": summary["full_manifest_points_to_stage23"],
            "uses_stage32_fallback": False,
            "row_count": 0 if full_csg_df is None else int(len(full_csg_df.index)),
            "notes": "CSG pair table for full model.",
        },
        {
            "artifact_id": "wo_csg_evidence_dir",
            "variant": "wo_csg",
            "artifact_type": "directory",
            "path": rel(WO_CSG_EVIDENCE_DIR),
            "exists": WO_CSG_EVIDENCE_DIR.is_dir(),
            "source_stage": "Step54 direct export",
            "results_dir": "" if wo_manifest is None else str(wo_manifest.get("results_dir", "")),
            "ckpt_path": "" if wo_manifest is None else str(wo_manifest.get("ckpt_path", "")),
            "model_type": "" if wo_manifest is None else str(wo_manifest.get("model_type", "")),
            "requested_model_type": "" if wo_manifest is None else str(wo_manifest.get("requested_model_type", "")),
            "matches_stage23_main": False,
            "uses_stage32_fallback": False,
            "row_count": "",
            "notes": "Preferred wo_csg evidence source.",
        },
        {
            "artifact_id": "wo_csg_manifest",
            "variant": "wo_csg",
            "artifact_type": "json",
            "path": rel(WO_CSG_EVIDENCE_DIR / "stage32_manifest.json"),
            "exists": (WO_CSG_EVIDENCE_DIR / "stage32_manifest.json").is_file(),
            "source_stage": "Step54 direct export",
            "results_dir": "" if wo_manifest is None else str(wo_manifest.get("results_dir", "")),
            "ckpt_path": "" if wo_manifest is None else str(wo_manifest.get("ckpt_path", "")),
            "model_type": "" if wo_manifest is None else str(wo_manifest.get("model_type", "")),
            "requested_model_type": "" if wo_manifest is None else str(wo_manifest.get("requested_model_type", "")),
            "matches_stage23_main": False,
            "uses_stage32_fallback": False,
            "row_count": "",
            "notes": "Manifest for wo_csg comparison export.",
        },
        {
            "artifact_id": "wo_csg_slide_summary",
            "variant": "wo_csg",
            "artifact_type": "csv",
            "path": rel(WO_CSG_EVIDENCE_DIR / "stage32_slide_evidence_summary.csv"),
            "exists": (WO_CSG_EVIDENCE_DIR / "stage32_slide_evidence_summary.csv").is_file(),
            "source_stage": "Step54 direct export",
            "results_dir": "" if wo_manifest is None else str(wo_manifest.get("results_dir", "")),
            "ckpt_path": "" if wo_manifest is None else str(wo_manifest.get("ckpt_path", "")),
            "model_type": "" if wo_manifest is None else str(wo_manifest.get("model_type", "")),
            "requested_model_type": "" if wo_manifest is None else str(wo_manifest.get("requested_model_type", "")),
            "matches_stage23_main": False,
            "uses_stage32_fallback": False,
            "row_count": 0 if wo_slide_df is None else int(len(wo_slide_df.index)),
            "notes": "Fold0/test wo_csg slide evidence summary.",
        },
        {
            "artifact_id": "wo_csg_top_concepts",
            "variant": "wo_csg",
            "artifact_type": "csv",
            "path": rel(WO_CSG_EVIDENCE_DIR / "stage32_top_concepts_long.csv"),
            "exists": (WO_CSG_EVIDENCE_DIR / "stage32_top_concepts_long.csv").is_file(),
            "source_stage": "Step54 direct export",
            "results_dir": "" if wo_manifest is None else str(wo_manifest.get("results_dir", "")),
            "ckpt_path": "" if wo_manifest is None else str(wo_manifest.get("ckpt_path", "")),
            "model_type": "" if wo_manifest is None else str(wo_manifest.get("model_type", "")),
            "requested_model_type": "" if wo_manifest is None else str(wo_manifest.get("requested_model_type", "")),
            "matches_stage23_main": False,
            "uses_stage32_fallback": False,
            "row_count": 0 if wo_concepts_df is None else int(len(wo_concepts_df.index)),
            "notes": "wo_csg concept-level evidence export.",
        },
        {
            "artifact_id": "wo_csg_top_csg_pairs",
            "variant": "wo_csg",
            "artifact_type": "csv",
            "path": rel(WO_CSG_EVIDENCE_DIR / "stage32_top_csg_pairs.csv"),
            "exists": (WO_CSG_EVIDENCE_DIR / "stage32_top_csg_pairs.csv").is_file(),
            "source_stage": "Step54 direct export",
            "results_dir": "" if wo_manifest is None else str(wo_manifest.get("results_dir", "")),
            "ckpt_path": "" if wo_manifest is None else str(wo_manifest.get("ckpt_path", "")),
            "model_type": "" if wo_manifest is None else str(wo_manifest.get("model_type", "")),
            "requested_model_type": "" if wo_manifest is None else str(wo_manifest.get("requested_model_type", "")),
            "matches_stage23_main": False,
            "uses_stage32_fallback": False,
            "row_count": 0 if wo_csg_df is None else int(len(wo_csg_df.index)),
            "notes": "Expected to be empty or sparse because CSG is disabled.",
        },
        {
            "artifact_id": "stage54b_case_level_metadata",
            "variant": "shared",
            "artifact_type": "csv",
            "path": rel(STAGE54B_CASE_METADATA),
            "exists": STAGE54B_CASE_METADATA.is_file(),
            "source_stage": "Step54B/54C refreshed metadata",
            "results_dir": "",
            "ckpt_path": "",
            "model_type": "",
            "requested_model_type": "",
            "matches_stage23_main": "",
            "uses_stage32_fallback": False if case_df is None or "uses_stage32_fallback" not in case_df.columns else bool(case_df["uses_stage32_fallback"].astype(bool).any()),
            "row_count": 0 if case_df is None else int(len(case_df.index)),
            "notes": "Matched-case metadata for full vs wo_csg evidence illustrations.",
        },
        {
            "artifact_id": "stage54b_evidence_source_audit",
            "variant": "shared",
            "artifact_type": "csv",
            "path": rel(STAGE54B_AUDIT_RESULTS_CSV),
            "exists": STAGE54B_AUDIT_RESULTS_CSV.is_file(),
            "source_stage": "Step54B/54C refreshed audit",
            "results_dir": "",
            "ckpt_path": "",
            "model_type": "",
            "requested_model_type": "",
            "matches_stage23_main": "",
            "uses_stage32_fallback": False,
            "row_count": 0 if audit_results_df is None else int(len(audit_results_df.index)),
            "notes": "Machine-readable evidence provenance audit.",
        },
    ]
    return pd.DataFrame(rows), summary


def collect_figure_inventory() -> tuple[pd.DataFrame, dict[str, Any]]:
    index_df = read_csv_optional(STAGE54B_FIGURE_INDEX_CSV)
    columns = [
        "figure_filename",
        "exists",
        "figure_type",
        "evidence_source",
        "uses_stage32_fallback",
        "has_provenance",
        "recommended_location",
        "supports_claim",
        "cannot_support_claim",
        "provenance_note",
        "source_index_file",
    ]
    rows: list[dict[str, Any]] = []
    if index_df is None:
        return pd.DataFrame(columns=columns), {"num_figures": 0, "all_exist": False, "all_have_provenance": False}

    for _, row in index_df.iterrows():
        figure_path = ROOT / str(row.get("figure_filename", ""))
        suggest_main = str(row.get("suggest_main_text", "")).strip().lower() == "yes"
        suggest_supp = str(row.get("suggest_supplementary", "")).strip().lower() == "yes"
        if suggest_main and suggest_supp:
            placement = "main_text_or_supplementary"
        elif suggest_main:
            placement = "main_text"
        elif suggest_supp:
            placement = "supplementary"
        else:
            placement = "not_recommended"
        provenance_note = str(row.get("provenance_note", "")).strip()
        evidence_source = str(row.get("evidence_source", "")).strip()
        rows.append(
            {
                "figure_filename": str(row.get("figure_filename", "")),
                "exists": figure_path.is_file(),
                "figure_type": str(row.get("figure_type", "")),
                "evidence_source": evidence_source,
                "uses_stage32_fallback": str(row.get("uses_stage32_fallback", "")).strip(),
                "has_provenance": bool(evidence_source and provenance_note),
                "recommended_location": placement,
                "supports_claim": str(row.get("supports_claim", "")),
                "cannot_support_claim": str(row.get("cannot_support_claim", "")),
                "provenance_note": provenance_note,
                "source_index_file": rel(STAGE54B_FIGURE_INDEX_CSV),
            }
        )
    figure_df = pd.DataFrame(rows, columns=columns)
    summary = {
        "num_figures": int(len(figure_df.index)),
        "all_exist": bool(figure_df["exists"].all()) if not figure_df.empty else False,
        "all_have_provenance": bool(figure_df["has_provenance"].all()) if not figure_df.empty else False,
    }
    return figure_df, summary


def collect_results_inventory() -> pd.DataFrame:
    rows = [
        ("stage23_main_results_dir", "Step23", "directory", STAGE23_DIR, True, "Main frozen result directory."),
        ("stage23_main_result_csv", "Step23", "csv", STAGE23_DIR / "result.csv", True, "Main metrics source."),
        ("stage23_main_fold_summary", "Step23", "csv", STAGE23_DIR / "fold_summary.csv", True, "Fold-level test metrics."),
        ("stage23_main_checkpoint_s0", "Step23", "checkpoint", STAGE23_DIR / "s_0_checkpoint.pt", True, "Full evidence manifest target."),
        ("stage52_ablation_dir", "Step52", "directory", STAGE52_DIR, True, "Core ablation result directory."),
        ("stage52b_table_md", "Step52B", "markdown", STAGE52B_MD, True, "Human-readable ablation table."),
        ("stage52b_table_csv", "Step52B", "csv", STAGE52B_CSV, True, "Machine-readable ablation table."),
        ("stage52b_interpretation_md", "Step52B", "markdown", STAGE52B_INTERPRETATION, True, "Interpretation guardrails."),
        ("stage53_package_dir", "Step53", "directory", STAGE53_DIR, True, "Paper-ready package."),
        ("stage54_package_dir", "Step54", "directory", STAGE54_DIR, True, "Initial evidence package."),
        ("stage54b_package_dir", "Step54B", "directory", STAGE54B_DIR, True, "Figure polish package."),
        ("stage54c_package_dir", "Step54C", "directory", STAGE54C_DIR, True, "Full direct export refresh package."),
        ("stage54_full_evidence_dir", "Step54C", "directory", FULL_EVIDENCE_DIR, True, "Preferred full evidence source."),
        ("stage54_wo_csg_evidence_dir", "Step54", "directory", WO_CSG_EVIDENCE_DIR, True, "Preferred wo_csg evidence source."),
        ("stage54b_case_metadata", "Step54B/54C", "csv", STAGE54B_CASE_METADATA, True, "Matched-case metadata."),
        ("stage54b_figure_index", "Step54B", "csv", STAGE54B_FIGURE_INDEX_CSV, True, "Figure provenance index."),
        ("main_model_file", "Code", "python", MAIN_MODEL_FILE, True, "Frozen main model definition."),
        ("main_training_script", "Code", "shell", MAIN_TRAIN_SCRIPT, True, "Main 5-fold training command."),
        ("ablation_training_script", "Code", "shell", ABLATION_SCRIPT, True, "Core ablation training command."),
        ("evidence_export_script", "Code", "shell", EVIDENCE_SCRIPT, True, "Evidence export command."),
        ("main_py", "Code", "python", MAIN_PY, True, "Training entrypoint."),
        ("core_utils_py", "Code", "python", CORE_UTILS_PY, True, "Training orchestration."),
        ("stage55_build_script", "Step55", "python", BUILD_SCRIPT, True, "Frozen release builder."),
    ]
    records = []
    for artifact_id, stage, artifact_type, path, lock_recommended, role in rows:
        records.append(
            {
                "artifact_id": artifact_id,
                "stage": stage,
                "artifact_type": artifact_type,
                "path": rel(path),
                "exists": path.exists(),
                "lock_recommended": lock_recommended,
                "role": role,
            }
        )
    return pd.DataFrame(records)


def scan_markdown_phrase(files: list[Path], phrase: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    needle = phrase.lower()
    for path in files:
        text = read_text_optional(path)
        if text is None:
            continue
        current_heading = ""
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                current_heading = stripped.lstrip("#").strip()
            if needle in stripped.lower():
                matches.append(
                    {
                        "file": rel(path),
                        "line": line_number,
                        "heading": current_heading,
                        "text": stripped,
                    }
                )
    return matches


def classify_claim_matches(matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive: list[dict[str, Any]] = []
    guarded: list[dict[str, Any]] = []
    for match in matches:
        line = str(match["text"]).lower()
        heading = str(match["heading"]).lower()
        negative_context = any(hint in heading for hint in NEGATIVE_SECTION_HINTS) or any(
            hint in line for hint in NEGATIVE_LINE_HINTS
        )
        if negative_context:
            guarded.append(match)
        else:
            positive.append(match)
    return positive, guarded


def make_check(
    check_id: str,
    ok: bool,
    pass_details: str,
    fail_details: str,
    evidence: str,
    action: str,
    fail_severity: str = "blocker",
) -> CheckResult:
    if ok:
        return CheckResult(check_id, "pass", "info", pass_details, evidence, "none")
    return CheckResult(check_id, fail_severity, fail_severity, fail_details, evidence, action)


def build_quality_audit(
    main_results_df: pd.DataFrame,
    main_means: dict[str, float | None],
    ablation_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    evidence_summary: dict[str, Any],
    figure_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    checks: list[CheckResult] = []

    stage23_result_exists = (STAGE23_DIR / "result.csv").is_file() and not main_results_df.empty
    checks.append(
        make_check(
            "stage23_main_results_exist",
            stage23_result_exists,
            "Stage23 main result.csv exists and was parsed.",
            "Stage23 main result.csv is missing or unreadable.",
            rel(STAGE23_DIR / "result.csv"),
            "Restore or regenerate the Stage23 main result before freezing.",
        )
    )

    stage52_full_df = read_csv_optional(STAGE52_FULL_DIR / "result.csv")
    stage52_full_ok = stage52_full_df is not None and "metric" in stage52_full_df.columns
    metric_match = True
    if stage52_full_ok:
        mean_row = stage52_full_df[stage52_full_df["metric"] == "mean"]
        mean_row = mean_row.iloc[0] if not mean_row.empty else pd.Series(dtype=object)
        for label, column in METRIC_MAP.items():
            if safe_float(mean_row.get(column)) != main_means.get(label):
                metric_match = False
                break
    else:
        metric_match = False
    checks.append(
        make_check(
            "step52_full_matches_stage23",
            metric_match,
            "Step52 full metrics match Stage23 exactly across AUC/ACC/F1/BACC/PR-AUC.",
            "Step52 full metrics do not exactly match Stage23 or Step52 full result.csv is unreadable.",
            f"{rel(STAGE52_FULL_DIR / 'result.csv')} vs {rel(STAGE23_DIR / 'result.csv')}",
            "Do not freeze the release until the Stage23 and Step52 full mismatch is explained.",
        )
    )

    checks.append(
        make_check(
            "step52b_ablation_table_exists",
            STAGE52B_MD.is_file() and STAGE52B_CSV.is_file() and not ablation_df.empty,
            "Step52B ablation markdown and CSV exist.",
            "Step52B ablation markdown or CSV is missing/unreadable.",
            f"{rel(STAGE52B_MD)}; {rel(STAGE52B_CSV)}",
            "Restore Step52B paper tables before freezing.",
        )
    )
    checks.append(
        make_check(
            "step53_package_exists",
            STAGE53_DIR.is_dir(),
            "Step53 package exists.",
            "Step53 package is missing.",
            rel(STAGE53_DIR),
            "Restore Step53 paper-ready package before freezing.",
        )
    )
    checks.append(
        make_check(
            "step54_package_exists",
            STAGE54_DIR.is_dir(),
            "Step54 package exists.",
            "Step54 package is missing.",
            rel(STAGE54_DIR),
            "Restore Step54 evidence package before freezing.",
        )
    )
    checks.append(
        make_check(
            "step54b_package_exists",
            STAGE54B_DIR.is_dir(),
            "Step54B package exists.",
            "Step54B package is missing.",
            rel(STAGE54B_DIR),
            "Restore Step54B figure polish package before freezing.",
        )
    )
    checks.append(
        make_check(
            "step54c_package_exists",
            STAGE54C_DIR.is_dir(),
            "Step54C package exists.",
            "Step54C package is missing.",
            rel(STAGE54C_DIR),
            "Restore Step54C direct-export refresh package before freezing.",
        )
    )
    checks.append(
        make_check(
            "full_evidence_dir_exists",
            FULL_EVIDENCE_DIR.is_dir(),
            "Full evidence directory exists.",
            "Full evidence directory is missing.",
            rel(FULL_EVIDENCE_DIR),
            "Restore Step54C full evidence export directory.",
        )
    )
    checks.append(
        make_check(
            "full_evidence_points_to_stage23_main",
            bool(evidence_summary.get("full_manifest_points_to_stage23")),
            "Full evidence manifest points to the Stage23 main run and s_0 checkpoint.",
            "Full evidence manifest does not point to the Stage23 main run and expected checkpoint.",
            rel(FULL_EVIDENCE_DIR / "stage32_manifest.json"),
            "Refresh or repair the full evidence manifest before freezing.",
        )
    )
    checks.append(
        make_check(
            "full_evidence_model_type_rce",
            bool(evidence_summary.get("full_model_type_is_rce")),
            "Full evidence manifest model_type/requested_model_type are both RCE_MIL_BiomedCLIP.",
            "Full evidence manifest model_type/requested_model_type are not both RCE_MIL_BiomedCLIP.",
            rel(FULL_EVIDENCE_DIR / "stage32_manifest.json"),
            "Fix the manifest before freezing the release.",
        )
    )
    checks.append(
        make_check(
            "wo_csg_evidence_exists",
            WO_CSG_EVIDENCE_DIR.is_dir(),
            "wo_csg evidence directory exists.",
            "wo_csg evidence directory is missing.",
            rel(WO_CSG_EVIDENCE_DIR),
            "Restore the wo_csg evidence export before freezing.",
        )
    )
    checks.append(
        make_check(
            "stage54b_case_level_metadata_exists",
            STAGE54B_CASE_METADATA.is_file(),
            "stage54b_case_level_metadata.csv exists.",
            "stage54b_case_level_metadata.csv is missing.",
            rel(STAGE54B_CASE_METADATA),
            "Regenerate or restore the case-level metadata.",
        )
    )
    checks.append(
        make_check(
            "stage54b_figure_index_exists",
            STAGE54B_FIGURE_INDEX_CSV.is_file() and STAGE54B_FIGURE_INDEX_MD.is_file(),
            "stage54b figure index CSV and markdown exist.",
            "stage54b figure index CSV or markdown is missing.",
            f"{rel(STAGE54B_FIGURE_INDEX_CSV)}; {rel(STAGE54B_FIGURE_INDEX_MD)}",
            "Restore the Step54B figure index files.",
        )
    )
    checks.append(
        make_check(
            "stage54b_seven_figures_exist",
            not figure_df.empty and len(figure_df.index) == 7 and bool(figure_df["exists"].all()),
            "All 7 Step54B figures exist.",
            "One or more Step54B figures are missing, or the figure index does not contain 7 rows.",
            rel(STAGE54B_DIR / "figures"),
            "Restore the missing figures before freezing.",
        )
    )

    case_df = read_csv_optional(STAGE54B_CASE_METADATA)
    figure_uses_fallback = False
    if not figure_df.empty and "uses_stage32_fallback" in figure_df.columns:
        figure_uses_fallback = any(str(value).strip().lower() == "true" for value in figure_df["uses_stage32_fallback"].tolist())
    case_uses_fallback = False
    if case_df is not None and "uses_stage32_fallback" in case_df.columns:
        case_uses_fallback = bool(case_df["uses_stage32_fallback"].astype(bool).any())
    checks.append(
        make_check(
            "uses_stage32_fallback_false",
            not figure_uses_fallback and not case_uses_fallback,
            "Step54B figure index and case metadata both record uses_stage32_fallback=False.",
            "Some Step54B metadata still records uses_stage32_fallback=True.",
            f"{rel(STAGE54B_FIGURE_INDEX_CSV)}; {rel(STAGE54B_CASE_METADATA)}",
            "Refresh the Step54B package so the preferred source is consistently direct export.",
        )
    )

    stale_stage32_matches = scan_markdown_phrase(
        [
            STAGE54_DIR / "stage54_evidence_resource_audit.md",
            STAGE54_DIR / "stage54_rce_evidence_report.md",
        ],
        "current full-model-equivalent evidence source",
    )
    if stale_stage32_matches:
        checks.append(
            CheckResult(
                "historical_docs_still_reference_stage32_as_current_source",
                "warning",
                "warning",
                "Legacy Step54 docs still describe Stage32 fallback as the current full-equivalent source.",
                "; ".join(f"{item['file']}:{item['line']}" for item in stale_stage32_matches),
                "Do not use the old Step54 package as the paper-facing preferred source; prefer Step54B/Step54C/Step55 docs.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "historical_docs_still_reference_stage32_as_current_source",
                "pass",
                "info",
                "No Step54 legacy doc still describes Stage32 fallback as the current preferred source.",
                rel(STAGE54_DIR),
                "none",
            )
        )

    paper_facing_files = [
        STAGE53_DIR / "stage53_rce_method_claim_draft.md",
        STAGE53_DIR / "stage53_rce_main_results.md",
        STAGE53_DIR / "stage53_rce_paper_ready_summary.md",
        STAGE54B_DIR / "stage54b_summary.md",
        STAGE54B_DIR / "stage54b_paper_figure_caption_drafts.md",
        STAGE54B_DIR / "stage54b_figure_index.md",
        STAGE54C_DIR / "stage54c_summary.md",
        STAGE54C_DIR / "stage54c_provenance_change_report.md",
        STAGE54C_DIR / "stage54c_claims_update.md",
    ]

    stat_positive, _ = classify_claim_matches(scan_markdown_phrase(paper_facing_files, "statistically significant"))
    if stat_positive:
        checks.append(
            CheckResult(
                "positive_statistically_significant_claims_present",
                "warning",
                "warning",
                "Some paper-facing docs still contain positive-looking statistically significant wording.",
                "; ".join(f"{item['file']}:{item['line']}" for item in stat_positive),
                "Rewrite those lines as descriptive trends only.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "positive_statistically_significant_claims_present",
                "pass",
                "info",
                "No positive statistically significant claim was found in the paper-facing docs that were audited.",
                "; ".join(rel(path) for path in paper_facing_files),
                "none",
            )
        )

    loc_positive, _ = classify_claim_matches(scan_markdown_phrase(paper_facing_files, "localization benchmark"))
    pathologist_positive, _ = classify_claim_matches(scan_markdown_phrase(paper_facing_files, "pathologist-level"))
    loc_hits = loc_positive + pathologist_positive
    if loc_hits:
        checks.append(
            CheckResult(
                "positive_localization_claims_present",
                "warning",
                "warning",
                "Some paper-facing docs still contain positive-looking localization benchmark or pathologist-level wording.",
                "; ".join(f"{item['file']}:{item['line']}" for item in loc_hits),
                "Keep these figures strictly in the interpretability/illustration scope.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "positive_localization_claims_present",
                "pass",
                "info",
                "No positive localization benchmark or pathologist-level claim was found in the paper-facing docs that were audited.",
                "; ".join(rel(path) for path in paper_facing_files),
                "none",
            )
        )

    component_phrase_files = [
        STAGE53_DIR / "stage53_rce_method_claim_draft.md",
        STAGE53_DIR / "stage53_rce_main_results.md",
        STAGE53_DIR / "stage53_rce_paper_ready_summary.md",
    ]
    deg_positive, _ = classify_claim_matches(scan_markdown_phrase(component_phrase_files, "final main model"))
    deg_component_issues = []
    for match in deg_positive:
        text = str(match["text"]).lower()
        if any(token in text for token in ("deg", "hcrc", "prarc")):
            deg_component_issues.append(match)
    if deg_component_issues:
        checks.append(
            CheckResult(
                "deg_hcrc_prarc_main_model_claim_present",
                "warning",
                "warning",
                "Some paper-facing docs still appear to place DEG/HCRC/PRARC in the final main-model slot.",
                "; ".join(f"{item['file']}:{item['line']}" for item in deg_component_issues),
                "Keep DEG/HCRC/PRARC outside the Step55 frozen main-model definition.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "deg_hcrc_prarc_main_model_claim_present",
                "pass",
                "info",
                "No audited paper-facing doc places DEG/HCRC/PRARC into the frozen main-model definition.",
                "; ".join(rel(path) for path in component_phrase_files),
                "none",
            )
        )

    old_caption_hits = scan_markdown_phrase(
        [STAGE54B_FIGURE_INDEX_MD, STAGE54B_CAPTIONS_MD],
        "fully same-source evidence comparison",
    ) + scan_markdown_phrase([STAGE54B_FIGURE_INDEX_MD, STAGE54B_CAPTIONS_MD], "cross-source example")
    if old_caption_hits:
        checks.append(
            CheckResult(
                "outdated_same_source_caption_wording_present",
                "warning",
                "warning",
                "Some Step54B caption/index text still uses the pre-Step54C same-source mismatch wording.",
                "; ".join(f"{item['file']}:{item['line']}" for item in old_caption_hits),
                "If these old captions are reused, revise them to direct-export matched-comparison wording.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "outdated_same_source_caption_wording_present",
                "pass",
                "info",
                "No outdated pre-Step54C same-source mismatch wording was found in the Step54B caption/index files.",
                f"{rel(STAGE54B_FIGURE_INDEX_MD)}; {rel(STAGE54B_CAPTIONS_MD)}",
                "none",
            )
        )

    audit_df = pd.DataFrame([check.__dict__ for check in checks])
    warnings = audit_df[audit_df["status"] == "warning"]["details"].tolist()
    blockers = audit_df[audit_df["status"] == "blocker"]["details"].tolist()
    return audit_df, warnings, blockers


def build_main_model_definition(git_state: dict[str, Any]) -> str:
    lines = [
        "# Step55 Main Model Definition",
        "",
        "## Frozen Main Model",
        "- Main model name: `RCE-v4-CSG-a01-rq16`",
        "- model_type: `RCE_MIL_BiomedCLIP`",
        f"- Main model file: `{rel(MAIN_MODEL_FILE)}`",
        f"- Main training script: `{rel(MAIN_TRAIN_SCRIPT)}`",
        f"- Main result directory: `{rel(STAGE23_DIR)}`",
        "- Runtime call chain: `main.py -> utils/core_utils.py -> models/model_RCE_MIL_BiomedCLIP.py`",
        "",
        "## Frozen Configuration Summary",
        "- Feature backbone: BiomedCLIP features (`features_biomedclip_5x` + `features_biomedclip_20x`).",
        "- Scale mode: dual scale.",
        "- Region queries: `rq16`.",
        "- Active RCE components: concept prior, visual residual, cross-scale graph (CSG), logit calibration.",
        "- CSG normalization: `sqrt`.",
        "- Text encoder mode: BiomedCLIP text tower kept frozen in the main release run.",
        "",
        "## Explicit Scope Boundary",
        "- `DEG`, `HCRC`, and `PRARC` are not components of the current Step55 frozen main model.",
        "- The frozen release is anchored on the direct RCE main line, not a DEG wrapper line.",
        "",
        "## Git Snapshot",
        f"- Commit hash: `{git_state['commit_hash']}`",
        f"- Branch: `{git_state['branch']}`",
        f"- Captured at: `{git_state['captured_at']}`",
        f"- Uncommitted changes present: `{fmt_bool(git_state['has_uncommitted_changes'])}`",
    ]
    return "\n".join(lines) + "\n"


def build_repro_commands() -> str:
    lines = [
        "# Step55 Reproducibility Commands",
        "",
        "These commands are recorded for reproducibility only. They were not executed in Step55.",
        "",
        "## Main Model 5-Fold Reproduction",
        "```bash",
        "VARIANT=rq16 bash scripts/experiments/run_stage23_rce_v4_csg_region_queries_5fold.sh",
        "```",
        "",
        "## Step52 Core Ablation",
        "```bash",
        "VARIANT=all bash scripts/experiments/run_stage52_rce_core_ablation_5fold.sh",
        "```",
        "",
        "## Step52B Paper Table Build",
        "```bash",
        "python scripts/analysis/build_stage52b_rce_ablation_paper_tables.py",
        "```",
        "",
        "## Step53 Paper-Ready Package Build",
        "```bash",
        "python scripts/analysis/build_stage53_rce_paper_ready_package.py",
        "```",
        "",
        "## Step54 Evidence Package Refresh",
        "```bash",
        "python scripts/analysis/build_stage54_rce_evidence_interpretability_package.py",
        "python scripts/analysis/plot_stage54_rce_evidence_figures.py",
        "```",
        "",
        "## Step54B Evidence Audit / Metadata / Figures",
        "```bash",
        "python scripts/analysis/build_stage54b_evidence_source_audit.py",
        "python scripts/analysis/build_stage54b_case_metadata.py",
        "python scripts/analysis/plot_stage54b_rce_evidence_figures.py",
        "```",
        "",
        "## Step54 / Step54C Evidence Export Refresh",
        "```bash",
        "MODE=full bash scripts/experiments/run_stage54_export_rce_evidence.sh",
        "MODE=wo_csg bash scripts/experiments/run_stage54_export_rce_evidence.sh",
        "MODE=both bash scripts/experiments/run_stage54_export_rce_evidence.sh",
        "python scripts/analysis/build_stage54c_full_export_audit.py",
        "```",
        "",
        "## Step55 Frozen Release Build",
        "```bash",
        "python scripts/analysis/build_stage55_rce_frozen_release.py",
        "python scripts/analysis/build_stage55_rce_frozen_release.py --check",
        "```",
        "",
        "## Suggested Manual Git Tag",
        "```bash",
        "git tag rce-paper-ready-v1",
        "git push origin rce-paper-ready-v1",
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_claims_doc() -> str:
    rows = [
        {
            "Can Make": "The frozen main model is `RCE-v4-CSG-a01-rq16` with `model_type=RCE_MIL_BiomedCLIP`.",
            "Avoid": "DEG/HCRC/PRARC is the current frozen main model.",
        },
        {
            "Can Make": "RCE extends ViLa-MIL from slide-level vision-language alignment toward region-concept evidence reasoning.",
            "Avoid": "The current release already proves broad external validation or multi-cancer generalization.",
        },
        {
            "Can Make": "Concept prior is the strongest single contributor among the audited core ablations.",
            "Avoid": "All modules are statistically significant.",
        },
        {
            "Can Make": "Visual residual provides important complementary visual evidence.",
            "Avoid": "Heatmaps or evidence figures provide pathologist-level localization.",
        },
        {
            "Can Make": "CSG mainly improves AUC / PR-AUC / evidence ranking or confidence behavior and should not be oversold on ACC.",
            "Avoid": "CSG significantly and substantially improves ACC.",
        },
        {
            "Can Make": "Step54C makes the full and wo_csg evidence sources matched direct-export interpretability illustrations.",
            "Avoid": "Evidence figures are training-significance proof or a strict localization benchmark.",
        },
        {
            "Can Make": "Evidence visualization can support interpretability discussion of region-concept reasoning.",
            "Avoid": "Evidence visualization is equivalent to expert annotation or formal localization evaluation.",
        },
    ]
    lines = [
        "# Step55 Claims To Make And Avoid",
        "",
        markdown_table(rows, ["Can Make", "Avoid"]),
        "",
        "Paired t-test values from Step52B should be cited only as descriptive references in this release. No conventional `p < 0.05` conclusion should be claimed for the audited core ablations.",
    ]
    return "\n".join(lines) + "\n"


def build_known_limitations() -> str:
    lines = [
        "# Step55 Known Limitations",
        "",
        "- CSG improves ranking-oriented metrics more clearly than ACC; ACC gains are small and should not be oversold.",
        "- The current paired t-test results should not be written as traditional statistical significance claims.",
        "- Evidence export is mainly a fold0/test interpretability package rather than a complete localization benchmark.",
        "- No expert-annotated region localization validation is included in this frozen release.",
        "- External validation and multi-dataset generalization remain insufficient for strong deployment-style claims.",
        "- A broader, uniformly curated comparison against more strong baselines still needs follow-up packaging.",
        "- DEG any-on purity audit is still unfinished.",
        "- HCRC and PRARC are not part of the current frozen main-model release.",
    ]
    return "\n".join(lines) + "\n"


def build_file_locklist() -> str:
    lock_items = [
        "results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/",
        "results_stage52_rce_core_ablation/",
        "docs/stage52b_rce_ablation_table_paper.md",
        "docs/stage52b_rce_ablation_table_paper.csv",
        "docs/stage53_rce_paper_ready_package/",
        "docs/stage54_rce_evidence_interpretability_package/",
        "docs/stage54b_rce_evidence_figure_polish/",
        "docs/stage54c_rce_full_direct_export_refresh/",
        "results_stage54_rce_evidence_interpretability/full/",
        "results_stage54_rce_evidence_interpretability/wo_csg/",
    ]
    lines = [
        "# Step55 File Locklist",
        "",
        "The following directories/files should be treated as locked historical inputs for later steps unless there is an explicit reason to rebuild them:",
        "",
        *[f"- `{item}`" for item in lock_items],
    ]
    return "\n".join(lines) + "\n"


def build_git_tag_instructions() -> str:
    lines = [
        "# Step55 Git Tag Instructions",
        "",
        "Step55 did not execute any git tagging or pushing.",
        "",
        "Suggested manual commands after review:",
        "",
        "```bash",
        "git tag rce-paper-ready-v1",
        "git push origin rce-paper-ready-v1",
        "```",
        "",
        "Use the tag only after reviewing the Step55 warnings and confirming that the frozen package is the intended fallback release state.",
    ]
    return "\n".join(lines) + "\n"


def build_next_steps(blockers: list[str], warnings: list[str]) -> str:
    lines = [
        "# Step55 Next Steps",
        "",
        "- Use the Step55 package as the paper-facing frozen fallback release index.",
        "- If Step54B captions or figure-index text is reused directly, update the residual pre-Step54C same-source wording first.",
        "- Keep Step54 legacy package references clearly historical when writing the final manuscript text.",
        "- Complete the separate DEG any-on purity audit before making any later DEG-based claim.",
        "- Add external validation / additional baselines only in later steps, not in this frozen release.",
    ]
    if blockers:
        lines.extend(["", "Current blockers:", *[f"- {item}" for item in blockers]])
    elif warnings:
        lines.extend(["", "Current warnings:", *[f"- {item}" for item in warnings]])
    else:
        lines.extend(["", "- No blocker or warning remains inside the Step55 audit."])
    return "\n".join(lines) + "\n"


def build_results_inventory_md(results_df: pd.DataFrame) -> str:
    lines = [
        "# Step55 Results Inventory",
        "",
        "This inventory records the source artifacts used to define the Step55 frozen release.",
        "",
        markdown_table_from_df(results_df),
    ]
    return "\n".join(lines) + "\n"


def build_main_results_md(main_results_df: pd.DataFrame) -> str:
    lines = [
        "# Step55 Main Results Table",
        "",
        "Main paper metrics are copied from the Stage23 direct-RCE run without recomputation.",
        "",
        markdown_table_from_df(main_results_df),
    ]
    return "\n".join(lines) + "\n"


def build_ablation_md(ablation_df: pd.DataFrame) -> str:
    lines = [
        "# Step55 Ablation Table",
        "",
        "This table mirrors the Step52B paper table values and does not recompute or overwrite the original ablation results.",
        "",
        markdown_table_from_df(ablation_df),
        "",
        "Interpretation boundary: use these values as descriptive trends. The paired t-test values from Step52B should not be promoted to a conventional `p < 0.05` significance claim in the frozen release.",
    ]
    return "\n".join(lines) + "\n"


def build_evidence_inventory_md(evidence_df: pd.DataFrame, evidence_summary: dict[str, Any]) -> str:
    lines = [
        "# Step55 Evidence Inventory",
        "",
        f"- Full evidence source: `{rel(FULL_EVIDENCE_DIR)}`",
        f"- wo_csg evidence source: `{rel(WO_CSG_EVIDENCE_DIR)}`",
        f"- Full manifest points to Stage23 main model: `{fmt_bool(evidence_summary['full_manifest_points_to_stage23'])}`",
        f"- Full model_type/requested_model_type are RCE_MIL_BiomedCLIP: `{fmt_bool(evidence_summary['full_model_type_is_rce'])}`",
        "",
        markdown_table_from_df(evidence_df),
    ]
    return "\n".join(lines) + "\n"


def build_figure_inventory_md(figure_df: pd.DataFrame, figure_summary: dict[str, Any]) -> str:
    lines = [
        "# Step55 Figure Inventory",
        "",
        f"- Number of indexed Step54B figures: `{figure_summary['num_figures']}`",
        f"- All 7 figures exist: `{fmt_bool(figure_summary['all_exist'])}`",
        f"- All figures carry provenance fields: `{fmt_bool(figure_summary['all_have_provenance'])}`",
        "",
        markdown_table_from_df(figure_df),
        "",
        "All figure interpretations remain qualitative or semi-structured. None of these figures should be used to claim localization benchmarking, pathologist-level localization, or training-time statistical significance.",
    ]
    return "\n".join(lines) + "\n"


def build_quality_audit_md(audit_df: pd.DataFrame, warnings: list[str], blockers: list[str]) -> str:
    lines = [
        "# Step55 Quality Audit",
        "",
        "## Status Summary",
        f"- Blocker count: `{len(blockers)}`",
        f"- Warning count: `{len(warnings)}`",
        "",
    ]
    if blockers:
        lines.extend(["## Blockers", *[f"- {item}" for item in blockers], ""])
    if warnings:
        lines.extend(["## Warnings", *[f"- {item}" for item in warnings], ""])
    if not blockers and not warnings:
        lines.extend(["- No blocker or warning was produced by the audited Step55 checks.", ""])
    lines.extend(["## Audit Table", "", markdown_table_from_df(audit_df)])
    return "\n".join(lines) + "\n"


def build_release_manifest_md(
    manifest: dict[str, Any],
    main_results_df: pd.DataFrame,
    warnings: list[str],
    blockers: list[str],
) -> str:
    git_state = manifest["git"]
    main_rows = []
    for _, row in main_results_df.iterrows():
        main_rows.append(
            {
                "metric": row["metric"],
                "mean": row["mean"],
                "std": row["std"],
            }
        )
    lines = [
        "# Step55 Release Manifest",
        "",
        "## Release Identity",
        f"- Release step: `{manifest['release_step']}`",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Main model: `{manifest['main_model']['name']}`",
        f"- model_type: `{manifest['main_model']['model_type']}`",
        "",
        "## Git State",
        f"- Commit hash: `{git_state['commit_hash']}`",
        f"- Branch: `{git_state['branch']}`",
        f"- Uncommitted changes present: `{fmt_bool(git_state['has_uncommitted_changes'])}`",
        "",
        "```text",
        *git_state["status_short"],
        "```",
        "",
        "## Main Results",
        "",
        markdown_table(main_rows, ["metric", "mean", "std"]),
        "",
        "## Audit Outcome",
        f"- Blockers: `{len(blockers)}`",
        f"- Warnings: `{len(warnings)}`",
        f"- Suggested tag readiness: `{manifest['release_readiness']['tag_recommendation']}`",
    ]
    return "\n".join(lines) + "\n"


def build_summary_md(
    manifest: dict[str, Any],
    warnings: list[str],
    blockers: list[str],
    evidence_summary: dict[str, Any],
    figure_summary: dict[str, Any],
    fallback_false: str,
) -> str:
    lines = [
        "# Step55 Frozen Release Summary",
        "",
        "Step55 freezes the current RCE paper-ready line as a read-only release package. No training, no evidence export, and no main-model logic changes were performed.",
        "",
        "## Frozen Main Line",
        "- Main model: `RCE-v4-CSG-a01-rq16`",
        "- model_type: `RCE_MIL_BiomedCLIP`",
        f"- Main result directory: `{rel(STAGE23_DIR)}`",
        f"- Main training script: `{rel(MAIN_TRAIN_SCRIPT)}`",
        "",
        "## Provenance Snapshot",
        f"- Full evidence source: `{rel(FULL_EVIDENCE_DIR)}`",
        f"- wo_csg evidence source: `{rel(WO_CSG_EVIDENCE_DIR)}`",
        f"- Full manifest points to Stage23 main model: `{fmt_bool(evidence_summary['full_manifest_points_to_stage23'])}`",
        f"- Full model_type/requested_model_type are RCE_MIL_BiomedCLIP: `{fmt_bool(evidence_summary['full_model_type_is_rce'])}`",
        f"- uses_stage32_fallback=False maintained in refreshed Step54B metadata: `{fallback_false}`",
        f"- Indexed Step54B figures: `{figure_summary['num_figures']}`",
        "",
        "## Audit Outcome",
        f"- Blocker count: `{len(blockers)}`",
        f"- Warning count: `{len(warnings)}`",
        f"- Tag recommendation: `{manifest['release_readiness']['tag_recommendation']}`",
    ]
    if warnings:
        lines.extend(["", "Warnings carried into the frozen release:", *[f"- {item}" for item in warnings]])
    if blockers:
        lines.extend(["", "Blockers:", *[f"- {item}" for item in blockers]])
    lines.extend(
        [
            "",
            "## Scope Reminder",
            "- This release is a conservative fallback package for reproducibility, handoff, and submission support.",
            "- Claims stay within descriptive classification trends and interpretability illustrations.",
            "- DEG/HCRC/PRARC remain outside the frozen main-model definition.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest_payload(
    git_state: dict[str, Any],
    main_results_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    evidence_summary: dict[str, Any],
    figure_summary: dict[str, Any],
    warnings: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    if blockers:
        tag_recommendation = "no_blockers_unresolved"
    elif warnings:
        tag_recommendation = "yes_with_warnings"
    else:
        tag_recommendation = "yes_ready_for_manual_tag"
    main_results = {}
    for _, row in main_results_df.iterrows():
        main_results[str(row["metric"])] = {
            "mean": row["mean"],
            "std": row["std"],
            "source_dir": row["source_dir"],
            "source_file": row["source_file"],
        }
    return {
        "release_step": "Step55",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_state,
        "main_model": {
            "name": "RCE-v4-CSG-a01-rq16",
            "model_type": "RCE_MIL_BiomedCLIP",
            "model_file": rel(MAIN_MODEL_FILE),
            "training_script": rel(MAIN_TRAIN_SCRIPT),
            "result_dir": rel(STAGE23_DIR),
            "main_components": [
                "BiomedCLIP features",
                "dual scale",
                "rq16",
                "concept prior",
                "visual residual",
                "CSG",
                "logit calibration",
            ],
            "excluded_components": ["DEG", "HCRC", "PRARC"],
        },
        "main_results": main_results,
        "ablation_variants": ablation_df["variant"].tolist() if not ablation_df.empty else [],
        "evidence": {
            "full_source": rel(FULL_EVIDENCE_DIR),
            "wo_csg_source": rel(WO_CSG_EVIDENCE_DIR),
            "full_manifest_points_to_stage23_main": evidence_summary["full_manifest_points_to_stage23"],
            "full_model_type_is_rce": evidence_summary["full_model_type_is_rce"],
            "full_num_slides": evidence_summary["full_num_slides"],
            "wo_csg_num_slides": evidence_summary["wo_csg_num_slides"],
        },
        "figures": figure_summary,
        "release_readiness": {
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "tag_recommendation": tag_recommendation,
        },
        "warnings": warnings,
        "blockers": blockers,
        "generated_files": {
            "docs_dir": rel(DOCS_DIR),
            "results_dir": rel(RESULTS_DIR),
        },
    }


def write_placeholder_manifests(docs_dir: Path, results_dir: Path) -> None:
    write_text(docs_dir / "stage55_release_manifest.md", "# placeholder\n")
    write_json(docs_dir / "stage55_release_manifest.json", {"placeholder": True})
    write_json(results_dir / "stage55_release_manifest.json", {"placeholder": True})


def build_release(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    ensure_dir(args.docs_dir)
    ensure_dir(args.results_dir)

    main_results_df, main_means, _ = get_main_result_rows()
    ablation_df = parse_stage52b_table()
    evidence_df, evidence_summary = collect_evidence_inventory()
    figure_df, figure_summary = collect_figure_inventory()
    results_inventory_df = collect_results_inventory()
    quality_audit_df, warnings, blockers = build_quality_audit(
        main_results_df,
        main_means,
        ablation_df,
        evidence_df,
        evidence_summary,
        figure_df,
    )

    if args.check:
        print(f"[Step55][check] blockers={len(blockers)} warnings={len(warnings)}")
        print(f"[Step55][check] docs_dir={rel(args.docs_dir)} results_dir={rel(args.results_dir)}")
        return warnings, blockers

    write_placeholder_manifests(args.docs_dir, args.results_dir)

    write_csv(args.docs_dir / "stage55_results_inventory.csv", results_inventory_df)
    write_text(args.docs_dir / "stage55_results_inventory.md", build_results_inventory_md(results_inventory_df))

    write_csv(args.docs_dir / "stage55_main_results_table.csv", main_results_df)
    write_text(args.docs_dir / "stage55_main_results_table.md", build_main_results_md(main_results_df))

    write_csv(args.docs_dir / "stage55_ablation_table.csv", ablation_df)
    write_text(args.docs_dir / "stage55_ablation_table.md", build_ablation_md(ablation_df))

    write_csv(args.docs_dir / "stage55_evidence_inventory.csv", evidence_df)
    write_text(args.docs_dir / "stage55_evidence_inventory.md", build_evidence_inventory_md(evidence_df, evidence_summary))

    write_csv(args.docs_dir / "stage55_figure_inventory.csv", figure_df)
    write_text(args.docs_dir / "stage55_figure_inventory.md", build_figure_inventory_md(figure_df, figure_summary))

    write_csv(args.docs_dir / "stage55_quality_audit.csv", quality_audit_df)
    write_text(args.docs_dir / "stage55_quality_audit.md", build_quality_audit_md(quality_audit_df, warnings, blockers))

    write_text(args.docs_dir / "stage55_main_model_definition.md", build_main_model_definition(get_git_state()))
    write_text(args.docs_dir / "stage55_reproducibility_commands.md", build_repro_commands())
    write_text(args.docs_dir / "stage55_claims_to_make_and_avoid.md", build_claims_doc())
    write_text(args.docs_dir / "stage55_known_limitations.md", build_known_limitations())
    write_text(args.docs_dir / "stage55_file_locklist.md", build_file_locklist())
    write_text(args.docs_dir / "stage55_git_tag_instructions.md", build_git_tag_instructions())
    write_text(args.docs_dir / "stage55_next_steps.md", build_next_steps(blockers, warnings))

    # Mirror the machine-readable files into the results package.
    write_csv(args.results_dir / "stage55_results_inventory.csv", results_inventory_df)
    write_csv(args.results_dir / "stage55_evidence_inventory.csv", evidence_df)
    write_csv(args.results_dir / "stage55_figure_inventory.csv", figure_df)
    write_csv(args.results_dir / "stage55_quality_audit.csv", quality_audit_df)

    git_state = get_git_state()
    manifest = build_manifest_payload(
        git_state=git_state,
        main_results_df=main_results_df,
        ablation_df=ablation_df,
        evidence_summary=evidence_summary,
        figure_summary=figure_summary,
        warnings=warnings,
        blockers=blockers,
    )
    fallback_false = "unknown"
    if not figure_df.empty and "uses_stage32_fallback" in figure_df.columns:
        fallback_false = fmt_bool(
            not any(str(value).strip().lower() == "true" for value in figure_df["uses_stage32_fallback"].tolist())
        )
    write_json(args.docs_dir / "stage55_release_manifest.json", manifest)
    write_json(args.results_dir / "stage55_release_manifest.json", manifest)
    write_text(args.docs_dir / "stage55_release_manifest.md", build_release_manifest_md(manifest, main_results_df, warnings, blockers))
    write_text(
        args.docs_dir / "stage55_frozen_release_summary.md",
        build_summary_md(manifest, warnings, blockers, evidence_summary, figure_summary, fallback_false),
    )

    return warnings, blockers


def main() -> None:
    args = parse_args()
    warnings, blockers = build_release(args)
    if args.check:
        if blockers:
            raise SystemExit(1)
        return
    print(f"[Step55] wrote docs to {rel(args.docs_dir)}")
    print(f"[Step55] wrote machine-readable outputs to {rel(args.results_dir)}")
    print(f"[Step55] blockers={len(blockers)} warnings={len(warnings)}")


if __name__ == "__main__":
    main()
