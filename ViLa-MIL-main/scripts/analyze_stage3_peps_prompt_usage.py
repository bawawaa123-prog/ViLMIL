from __future__ import annotations

import ast
import math
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
EVAL_ROOT = ROOT / "eval_results"
OUT_DIR = ROOT / "trained_models" / "stage3_peps_comparison"

PEPS_EVALS = {
    "Concept-12 PEPS topk=1 tau=0.1": EVAL_ROOT / "EVAL_adeno_concept12_peps_topk1_tau0.1",
    "Concept-12 PEPS topk=3 tau=0.1": EVAL_ROOT / "EVAL_adeno_concept12_peps_topk3_tau0.1",
    "Concept-12 PEPS topk=5 tau=0.1": EVAL_ROOT / "EVAL_adeno_concept12_peps_topk5_tau0.1",
    "Concept-12 PEPS topk=3 tau=0.07": EVAL_ROOT / "EVAL_adeno_concept12_peps_topk3_tau0.07",
    "Concept-12 PEPS topk=3 tau=0.15": EVAL_ROOT / "EVAL_adeno_concept12_peps_topk3_tau0.15",
    "Concept-12 PEPS topk=5 tau=0.07": EVAL_ROOT / "EVAL_adeno_concept12_peps_topk5_tau0.07",
    "Concept-12 PEPS topk=5 tau=0.15": EVAL_ROOT / "EVAL_adeno_concept12_peps_topk5_tau0.15",
}

DYNAMIC_GATE_EVAL = {
    "Concept-12 MLP dynamic_gate": EVAL_ROOT / "EVAL_adeno_concept12_dynamic_gate_eval",
}


def _shannon_entropy(items: list[str]) -> float:
    if not items:
        return math.nan
    counts = Counter(items)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def _collapse_rate(items: list[str]) -> float:
    if not items:
        return math.nan
    counts = Counter(items)
    return max(counts.values()) / len(items)


def _safe_literal_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _fold_from_path(path: Path) -> int:
    stem = path.stem
    fold_str = stem.split("fold")[-1]
    return int(fold_str) + 1


def _analyze_peps_experiment(experiment: str, eval_dir: Path):
    summary_rows = []
    top_prompt_rows = []

    if not eval_dir.is_dir():
        summary_rows.append({"experiment": experiment, "status": "missing_eval_dir"})
        return summary_rows, top_prompt_rows

    for csv_path in sorted(eval_dir.glob("peps_prompt_analysis_fold*.csv")):
        fold = _fold_from_path(csv_path)
        df = pd.read_csv(csv_path)

        pred_df = df[df["class_id"] == df["pred_label"]].copy()
        low_items = pred_df[pred_df["scale"] == "low"]["top1_prompt_text"].tolist()
        high_items = pred_df[pred_df["scale"] == "high"]["top1_prompt_text"].tolist()

        summary_rows.append(
            {
                "experiment": experiment,
                "method_family": "peps",
                "fold": fold,
                "status": "ok",
                "low_top1_unique_prompts": len(set(low_items)),
                "high_top1_unique_prompts": len(set(high_items)),
                "low_prompt_entropy": _shannon_entropy(low_items),
                "high_prompt_entropy": _shannon_entropy(high_items),
                "low_prompt_collapse_rate": _collapse_rate(low_items),
                "high_prompt_collapse_rate": _collapse_rate(high_items),
                "num_low_samples": len(low_items),
                "num_high_samples": len(high_items),
            }
        )

        for true_label, class_name in [(0, "Adenocarcinoma"), (1, "NonAdenocarcinoma")]:
            true_class_df = df[(df["true_label"] == true_label) & (df["class_id"] == true_label)].copy()
            counts = (
                true_class_df.groupby(["top1_prompt_text", "top1_prompt_concept"])
                .size()
                .reset_index(name="count")
                .sort_values(["count", "top1_prompt_text"], ascending=[False, True])
                .head(5)
            )
            total = len(true_class_df)
            for rank, (_, row) in enumerate(counts.iterrows(), start=1):
                top_prompt_rows.append(
                    {
                        "experiment": experiment,
                        "fold": fold,
                        "true_class_name": class_name,
                        "rank": rank,
                        "prompt_text": row["top1_prompt_text"],
                        "prompt_concept": row["top1_prompt_concept"],
                        "count": int(row["count"]),
                        "frequency": float(row["count"] / total) if total else math.nan,
                    }
                )

    return summary_rows, top_prompt_rows


def _analyze_dynamic_gate_experiment(experiment: str, eval_dir: Path):
    summary_rows = []
    if not eval_dir.is_dir():
        summary_rows.append({"experiment": experiment, "status": "missing_eval_dir"})
        return summary_rows

    for csv_path in sorted(eval_dir.glob("prompt_weight_analysis_fold*.csv")):
        fold = _fold_from_path(csv_path)
        df = pd.read_csv(csv_path)
        low_items = [_safe_literal_list(v)[0] for v in df["top3_pred_low_prompts"] if _safe_literal_list(v)]
        high_items = [_safe_literal_list(v)[0] for v in df["top3_pred_high_prompts"] if _safe_literal_list(v)]
        summary_rows.append(
            {
                "experiment": experiment,
                "method_family": "dynamic_gate",
                "fold": fold,
                "status": "ok",
                "low_top1_unique_prompts": len(set(low_items)),
                "high_top1_unique_prompts": len(set(high_items)),
                "low_prompt_entropy": _shannon_entropy(low_items),
                "high_prompt_entropy": _shannon_entropy(high_items),
                "low_prompt_collapse_rate": _collapse_rate(low_items),
                "high_prompt_collapse_rate": _collapse_rate(high_items),
                "num_low_samples": len(low_items),
                "num_high_samples": len(high_items),
            }
        )
    return summary_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    top_prompt_rows = []

    for experiment, eval_dir in PEPS_EVALS.items():
        peps_summary, peps_top_prompts = _analyze_peps_experiment(experiment, eval_dir)
        summary_rows.extend(peps_summary)
        top_prompt_rows.extend(peps_top_prompts)

    for experiment, eval_dir in DYNAMIC_GATE_EVAL.items():
        summary_rows.extend(_analyze_dynamic_gate_experiment(experiment, eval_dir))

    summary_df = pd.DataFrame(summary_rows)
    top_prompt_df = pd.DataFrame(top_prompt_rows)
    overall_top_prompt_rows = []

    overall_rows = []
    if not summary_df.empty:
        ok_df = summary_df[summary_df["status"] == "ok"].copy()
        metric_cols = [
            "low_top1_unique_prompts",
            "high_top1_unique_prompts",
            "low_prompt_entropy",
            "high_prompt_entropy",
            "low_prompt_collapse_rate",
            "high_prompt_collapse_rate",
            "num_low_samples",
            "num_high_samples",
        ]
        for experiment, group in ok_df.groupby("experiment"):
            row = {"experiment": experiment, "num_folds": int(len(group))}
            for col in metric_cols:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
            overall_rows.append(row)

    if not top_prompt_df.empty:
        grouped = (
            top_prompt_df.groupby(["experiment", "true_class_name", "prompt_text", "prompt_concept"])["count"]
            .sum()
            .reset_index()
        )
        for (experiment, true_class_name), group in grouped.groupby(["experiment", "true_class_name"]):
            ranked = group.sort_values(["count", "prompt_text"], ascending=[False, True]).head(5)
            total = int(group["count"].sum())
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                overall_top_prompt_rows.append(
                    {
                        "experiment": experiment,
                        "true_class_name": true_class_name,
                        "rank": rank,
                        "prompt_text": row["prompt_text"],
                        "prompt_concept": row["prompt_concept"],
                        "count": int(row["count"]),
                        "frequency": float(row["count"] / total) if total else math.nan,
                    }
                )

    overall_df = pd.DataFrame(overall_rows)
    overall_top_prompt_df = pd.DataFrame(overall_top_prompt_rows)

    summary_csv = OUT_DIR / "peps_prompt_usage_summary.csv"
    overall_csv = OUT_DIR / "peps_prompt_usage_overall.csv"
    top_prompt_csv = OUT_DIR / "peps_prompt_usage_top_prompts.csv"
    overall_top_prompt_csv = OUT_DIR / "peps_prompt_usage_top_prompts_overall.csv"

    summary_df.to_csv(summary_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)
    top_prompt_df.to_csv(top_prompt_csv, index=False)
    overall_top_prompt_df.to_csv(overall_top_prompt_csv, index=False)

    print(f"Saved prompt usage per-fold summary to: {summary_csv}")
    print(f"Saved prompt usage overall summary to: {overall_csv}")
    print(f"Saved prompt usage top-prompt table to: {top_prompt_csv}")
    print(f"Saved overall top-prompt table to: {overall_top_prompt_csv}")


if __name__ == "__main__":
    main()
