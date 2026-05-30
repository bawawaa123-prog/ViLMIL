from __future__ import annotations

import argparse
import math
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = Path("results_stage9/stage13_rce_evidence_export")
CORE_FILES = {
    "prediction_csv": "slide_prediction_evidence.csv",
    "top_concepts_csv": "slide_top_concepts.csv",
    "evidence_pkl": "region_concept_evidence.pkl",
    "export_report_md": "stage13_rce_evidence_export_report.md",
}


def warn_message(message: str, warning_log: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    warning_log.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage13 RCE evidence export quality.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to ViLa-MIL-main root.")
    parser.add_argument(
        "--evidence_dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing Stage13 evidence export files.",
    )
    parser.add_argument("--top_n_print", type=int, default=10, help="Top-N concept frequency rows highlighted in report.")
    return parser.parse_args()


def resolve_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return root / value


def safe_read_csv(path: Path, warning_log: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        warn_message(f"Missing CSV: {path}", warning_log)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warn_message(f"Failed to read CSV {path}: {exc}", warning_log)
        return None


def safe_read_text(path: Path, warning_log: list[str]) -> str:
    if not path.is_file():
        warn_message(f"Missing Markdown report: {path}", warning_log)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warn_message(f"Failed to read text {path}: {exc}", warning_log)
        return ""


def safe_load_pickle(path: Path, warning_log: list[str]):
    if not path.is_file():
        warn_message(f"Missing pickle: {path}", warning_log)
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        warn_message(f"Failed to load pickle {path}: {exc}", warning_log)
        return None


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
        return np.asarray(value)
    try:
        return np.asarray(value)
    except Exception:
        return None


def shape_string(value) -> str:
    arr = to_numpy(value)
    if arr is None:
        return "missing"
    return str(tuple(arr.shape))


def has_nan_or_inf(value) -> tuple[bool, bool]:
    arr = to_numpy(value)
    if arr is None or arr.size == 0:
        return False, False
    try:
        return bool(np.isnan(arr).any()), bool(np.isinf(arr).any())
    except Exception:
        return False, False


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    safe_df = df.fillna("NA").astype(str)
    columns = list(safe_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in safe_df.iterrows()
    ]
    return "\n".join([header, separator] + rows)


def check_prediction_table(df: pd.DataFrame | None, warning_log: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    summary = {
        "check": "slide_prediction_evidence",
        "status": "missing" if df is None else "ok",
        "row_count": 0 if df is None else len(df),
        "slide_id_unique": False,
        "missing_required_columns": "",
        "correct_matches_pred_eq_label": False,
        "prob_sum_close_to_one_ratio": math.nan,
        "alpha_in_range_ratio": math.nan,
        "accuracy": math.nan,
    }
    if df is None:
        return summary, pd.DataFrame()

    required_cols = [
        "slide_id",
        "label",
        "pred",
        "correct",
        "prob_0",
        "prob_1",
        "final_logit_0",
        "final_logit_1",
        "visual_logit_0",
        "visual_logit_1",
        "visual_residual_alpha",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    summary["missing_required_columns"] = ",".join(missing_cols)
    if missing_cols:
        summary["status"] = "partial"
        warn_message(f"Prediction CSV missing columns: {missing_cols}", warning_log)
        return summary, df

    summary["slide_id_unique"] = bool(df["slide_id"].is_unique)
    computed_correct = (pd.to_numeric(df["pred"], errors="coerce") == pd.to_numeric(df["label"], errors="coerce")).astype(float)
    reported_correct = pd.to_numeric(df["correct"], errors="coerce")
    summary["correct_matches_pred_eq_label"] = bool((computed_correct == reported_correct).all())

    prob_sum = pd.to_numeric(df["prob_0"], errors="coerce") + pd.to_numeric(df["prob_1"], errors="coerce")
    summary["prob_sum_close_to_one_ratio"] = float(np.mean(np.isclose(prob_sum, 1.0, atol=1e-4, rtol=1e-4)))

    alpha = pd.to_numeric(df["visual_residual_alpha"], errors="coerce")
    summary["alpha_in_range_ratio"] = float(np.mean((alpha >= 0.0) & (alpha <= 1.0)))
    summary["accuracy"] = float(computed_correct.mean())

    if not summary["slide_id_unique"]:
        summary["status"] = "partial"
        warn_message("Prediction CSV has non-unique slide_id values.", warning_log)
    if not summary["correct_matches_pred_eq_label"]:
        summary["status"] = "partial"
        warn_message("Prediction CSV 'correct' column does not match pred == label.", warning_log)
    return summary, df


def check_top_concepts_table(
    df: pd.DataFrame | None,
    prediction_df: pd.DataFrame | None,
    warning_log: list[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    summary = {
        "check": "slide_top_concepts",
        "status": "missing" if df is None else "ok",
        "row_count": 0 if df is None else len(df),
        "aligned_slide_ratio": math.nan,
        "scale_values": "",
        "class_values": "",
        "rank_min": math.nan,
        "rank_max": math.nan,
        "concept_text_nonempty_ratio": math.nan,
    }
    frequency_df = pd.DataFrame(columns=["scale", "class_id", "concept_id", "concept_text", "count"])
    if df is None:
        return summary, pd.DataFrame(), frequency_df

    required_cols = ["slide_id", "scale", "class_id", "rank", "evidence_score", "prompt_weight"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        summary["status"] = "partial"
        warn_message(f"Top concepts CSV missing columns: {missing_cols}", warning_log)
        return summary, df, frequency_df

    summary["scale_values"] = ",".join(sorted(df["scale"].dropna().astype(str).unique().tolist()))
    class_values = sorted(pd.to_numeric(df["class_id"], errors="coerce").dropna().astype(int).unique().tolist())
    summary["class_values"] = ",".join(str(v) for v in class_values)
    rank_values = pd.to_numeric(df["rank"], errors="coerce").dropna()
    if not rank_values.empty:
        summary["rank_min"] = int(rank_values.min())
        summary["rank_max"] = int(rank_values.max())

    if "concept_text" in df.columns:
        concept_text = df["concept_text"].fillna("").astype(str).str.strip()
        summary["concept_text_nonempty_ratio"] = float(np.mean(concept_text != ""))

    if prediction_df is not None and "slide_id" in prediction_df.columns:
        pred_slides = set(prediction_df["slide_id"].astype(str))
        concept_slides = df["slide_id"].astype(str)
        summary["aligned_slide_ratio"] = float(np.mean(concept_slides.isin(pred_slides)))
        if summary["aligned_slide_ratio"] < 1.0:
            summary["status"] = "partial"
            warn_message("Some top-concept rows do not align to prediction slide_id values.", warning_log)

    invalid_scale = set(df["scale"].dropna().astype(str)) - {"low", "high"}
    if invalid_scale:
        summary["status"] = "partial"
        warn_message(f"Unexpected scale values in top concepts CSV: {sorted(invalid_scale)}", warning_log)

    invalid_class = set(class_values) - {0, 1}
    if invalid_class:
        summary["status"] = "partial"
        warn_message(f"Unexpected class_id values in top concepts CSV: {sorted(invalid_class)}", warning_log)

    numeric_cols = ["evidence_score", "prompt_weight", "rank"]
    for col in numeric_cols:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.isna().any():
            summary["status"] = "partial"
            warn_message(f"Non-numeric values found in top concepts column '{col}'.", warning_log)

    group_cols = ["scale", "class_id", "concept_id"] if "concept_id" in df.columns else ["scale", "class_id"]
    if "concept_text" in df.columns:
        frequency_df = (
            df.groupby(group_cols + ["concept_text"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "scale", "class_id"], ascending=[False, True, True])
            .reset_index(drop=True)
        )
    else:
        frequency_df = (
            df.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "scale", "class_id"], ascending=[False, True, True])
            .reset_index(drop=True)
        )
    return summary, df, frequency_df


def check_evidence_pickle(
    payload,
    prediction_df: pd.DataFrame | None,
    warning_log: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    summary = {
        "check": "region_concept_evidence",
        "status": "missing" if payload is None else "ok",
        "entry_count": 0 if payload is None or not hasattr(payload, "__len__") else len(payload),
        "aligned_slide_ratio": math.nan,
        "has_required_keys_ratio": math.nan,
        "low_shape_summary": "",
        "high_shape_summary": "",
        "has_nan": False,
        "has_inf": False,
    }
    shape_rows: list[dict[str, object]] = []
    if payload is None:
        return summary, pd.DataFrame(shape_rows)
    if not isinstance(payload, list):
        summary["status"] = "partial"
        warn_message(f"Evidence pickle root object is not a list: {type(payload).__name__}", warning_log)
        return summary, pd.DataFrame(shape_rows)

    required_keys = {
        "low_region_concept_sim",
        "high_region_concept_sim",
        "low_prompt_evidence",
        "high_prompt_evidence",
        "low_prompt_weights",
        "high_prompt_weights",
        "final_logits",
        "visual_logits",
    }
    pred_slide_ids = set(prediction_df["slide_id"].astype(str)) if prediction_df is not None and "slide_id" in prediction_df.columns else set()
    aligned_flags = []
    required_flags = []
    low_shapes = Counter()
    high_shapes = Counter()
    any_nan = False
    any_inf = False

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            required_flags.append(False)
            warn_message(f"Evidence pickle entry {idx} is not a dict: {type(item).__name__}", warning_log)
            continue
        slide_id = str(item.get("slide_id", ""))
        aligned_flags.append(slide_id in pred_slide_ids if pred_slide_ids else False)
        required_flags.append(required_keys.issubset(item.keys()))
        if not required_flags[-1]:
            missing = sorted(required_keys - set(item.keys()))
            warn_message(f"Evidence pickle entry {idx} missing keys: {missing}", warning_log)

        low_shape = shape_string(item.get("low_region_concept_sim"))
        high_shape = shape_string(item.get("high_region_concept_sim"))
        low_shapes[low_shape] += 1
        high_shapes[high_shape] += 1
        shape_rows.append(
            {
                "slide_id": slide_id,
                "low_region_concept_shape": low_shape,
                "high_region_concept_shape": high_shape,
                "final_logits_shape": shape_string(item.get("final_logits")),
                "visual_logits_shape": shape_string(item.get("visual_logits")),
            }
        )
        for key in required_keys:
            has_nan, has_inf = has_nan_or_inf(item.get(key))
            any_nan = any_nan or has_nan
            any_inf = any_inf or has_inf

    if aligned_flags:
        summary["aligned_slide_ratio"] = float(np.mean(aligned_flags))
    if required_flags:
        summary["has_required_keys_ratio"] = float(np.mean(required_flags))
    summary["low_shape_summary"] = "; ".join(f"{shape}:{count}" for shape, count in sorted(low_shapes.items()))
    summary["high_shape_summary"] = "; ".join(f"{shape}:{count}" for shape, count in sorted(high_shapes.items()))
    summary["has_nan"] = any_nan
    summary["has_inf"] = any_inf

    if any_nan:
        summary["status"] = "partial"
        warn_message("NaN values detected in evidence pickle tensors.", warning_log)
    if any_inf:
        summary["status"] = "partial"
        warn_message("Inf values detected in evidence pickle tensors.", warning_log)
    if not math.isnan(summary["aligned_slide_ratio"]) and summary["aligned_slide_ratio"] < 1.0:
        summary["status"] = "partial"
        warn_message("Evidence pickle slide_id values do not fully align with prediction CSV.", warning_log)
    if not math.isnan(summary["has_required_keys_ratio"]) and summary["has_required_keys_ratio"] < 1.0:
        summary["status"] = "partial"
    return summary, pd.DataFrame(shape_rows)


def extract_exported_slides(report_text: str) -> int | None:
    for line in report_text.splitlines():
        if "exported_slides" in line:
            try:
                return int(line.split("`")[1])
            except Exception:
                return None
    return None


def build_quality_report(
    evidence_dir: Path,
    summary_df: pd.DataFrame,
    prediction_df: pd.DataFrame | None,
    frequency_df: pd.DataFrame,
    shape_df: pd.DataFrame,
    report_text: str,
    top_n_print: int,
    warning_log: list[str],
) -> str:
    exported_slides = extract_exported_slides(report_text)
    accuracy = math.nan
    if prediction_df is not None and "correct" in prediction_df.columns:
        accuracy = float(pd.to_numeric(prediction_df["correct"], errors="coerce").mean())

    top_freq_preview = frequency_df.head(top_n_print).copy() if not frequency_df.empty else pd.DataFrame()
    shape_preview = shape_df.drop_duplicates().head(top_n_print).copy() if not shape_df.empty else pd.DataFrame()

    lines = [
        "# Stage13 RCE Evidence Quality Check",
        "",
        "This is a lightweight small-sample evidence quality check, not a training-result evaluation.",
        "",
        "## Input Status",
        "",
        f"- evidence_dir: `{evidence_dir}`",
        f"- exported_slides_from_report: `{exported_slides}`" if exported_slides is not None else "- exported_slides_from_report: `NA`",
        "",
        markdown_table(summary_df.fillna("NA")),
        "",
        "## Quality Summary",
        "",
        f"- prediction accuracy on exported slides: `{accuracy:.4f}`" if not math.isnan(accuracy) else "- prediction accuracy on exported slides: `NA`",
        f"- warnings_found: `{len(warning_log)}`",
        "",
    ]

    if not top_freq_preview.empty:
        lines.extend(
            [
                "## Top Concept Frequency",
                "",
                markdown_table(top_freq_preview),
                "",
            ]
        )

    if not shape_preview.empty:
        lines.extend(
            [
                "## Evidence Shape Preview",
                "",
                markdown_table(shape_preview),
                "",
            ]
        )

    if warning_log:
        lines.extend(["## Warnings", ""])
        for warning in warning_log:
            lines.append(f"- {warning}")
        lines.append("")
    else:
        lines.extend(["## Warnings", "", "- None", ""])

    lines.extend(
        [
            "## Recommendation",
            "",
            "If file alignment, probability checks, tensor shapes, and NaN/Inf checks all look healthy, the evidence export is ready for larger fold0 export or Step14 visualization.",
            "",
            "Step14: concept-class graph or evidence visualization.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    evidence_dir = resolve_path(root, args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    warning_log: list[str] = []

    prediction_df = safe_read_csv(evidence_dir / CORE_FILES["prediction_csv"], warning_log)
    concepts_df = safe_read_csv(evidence_dir / CORE_FILES["top_concepts_csv"], warning_log)
    evidence_payload = safe_load_pickle(evidence_dir / CORE_FILES["evidence_pkl"], warning_log)
    export_report_text = safe_read_text(evidence_dir / CORE_FILES["export_report_md"], warning_log)

    prediction_summary, prediction_df = check_prediction_table(prediction_df, warning_log)
    concept_summary, concepts_df, frequency_df = check_top_concepts_table(concepts_df, prediction_df, warning_log)
    evidence_summary, shape_df = check_evidence_pickle(evidence_payload, prediction_df, warning_log)

    summary_df = pd.DataFrame([prediction_summary, concept_summary, evidence_summary])
    summary_csv = evidence_dir / "stage13_rce_evidence_quality_summary.csv"
    report_md = evidence_dir / "stage13_rce_evidence_quality_report.md"
    frequency_csv = evidence_dir / "stage13_top_concept_frequency.csv"

    summary_df.to_csv(summary_csv, index=False)
    frequency_df.to_csv(frequency_csv, index=False)

    report_text = build_quality_report(
        evidence_dir=evidence_dir,
        summary_df=summary_df,
        prediction_df=prediction_df,
        frequency_df=frequency_df,
        shape_df=shape_df,
        report_text=export_report_text,
        top_n_print=args.top_n_print,
        warning_log=warning_log,
    )
    report_md.write_text(report_text, encoding="utf-8")

    print(f"Saved quality summary CSV to: {summary_csv}")
    print(f"Saved top concept frequency CSV to: {frequency_csv}")
    print(f"Saved quality report Markdown to: {report_md}")


if __name__ == "__main__":
    main()
