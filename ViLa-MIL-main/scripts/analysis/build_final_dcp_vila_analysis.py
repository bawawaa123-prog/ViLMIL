from __future__ import annotations

import ast
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
TRAINED_MODELS = ROOT / "trained_models"
EVAL_RESULTS = ROOT / "eval_results"
OUT_DIR = TRAINED_MODELS / "final_dcp_vila_analysis"

METRICS = [
    "test_auc",
    "test_acc",
    "test_f1",
    "val_auc",
    "balanced_acc",
    "sensitivity",
    "specificity",
    "pr_auc",
]

FINAL_METHODS = [
    "BiomedCLIP static prompt baseline",
    "Concept-12 embedding_mean",
    "Concept-12 MLP dynamic_gate",
    "Concept-12 PEPS topk=5 tau=0.07",
    "Concept-12 PEPS topk=5 tau=0.1",
]

MAIN_MODEL = "Concept-12 PEPS topk=5 tau=0.07"
AUC_FIRST_MODEL = "Concept-12 PEPS topk=5 tau=0.1"

TRAIN_DIRS = {
    "BiomedCLIP static prompt baseline": TRAINED_MODELS / "adenocarcinoma_biomedclip_dual_strict5_s1",
    "Concept-12 embedding_mean": TRAINED_MODELS / "adeno_concept12_embedding_mean_s1",
    "Concept-12 MLP dynamic_gate": TRAINED_MODELS / "adeno_concept12_dynamic_gate_s1",
    "Concept-12 PEPS topk=5 tau=0.07": TRAINED_MODELS / "adeno_concept12_peps_topk5_tau0.07_s1",
    "Concept-12 PEPS topk=5 tau=0.1": TRAINED_MODELS / "adeno_concept12_peps_topk5_tau0.1_s1",
    "PEPS low-only": TRAINED_MODELS / "adeno_concept12_peps_topk5_tau0.07_lowonly_s1",
    "PEPS high-only": TRAINED_MODELS / "adeno_concept12_peps_topk5_tau0.07_highonly_s1",
}

BASELINE_SUPPLEMENT_DIR = EVAL_RESULTS / "EVAL_adeno_baseline_metrics_supplement"
PEPS_MAIN_EVAL_DIR = EVAL_RESULTS / "EVAL_adeno_concept12_peps_topk5_tau0.07"
DYNAMIC_GATE_EVAL_DIR = EVAL_RESULTS / "EVAL_adeno_concept12_dynamic_gate_eval"


def _fmt_pm(mean_value, std_value):
    if pd.isna(mean_value) or pd.isna(std_value):
        return "NA"
    return f"{float(mean_value):.6f} ± {float(std_value):.6f}"


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def _read_result_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path).set_index("metric")


def _read_train_result_and_folds(train_dir: Path):
    result_df = _read_result_csv(train_dir / "result.csv")
    fold_df = pd.read_csv(train_dir / "fold_summary.csv")
    if "fold" in fold_df.columns:
        fold_df["fold"] = fold_df["fold"].astype(int)
    return result_df, fold_df


def _empty_summary_and_folds():
    summary = {metric: (math.nan, math.nan) for metric in METRICS}
    fold_df = pd.DataFrame(columns=["fold"] + METRICS)
    return summary, fold_df


def _build_baseline_tables():
    train_result_df, train_fold_df = _read_train_result_and_folds(TRAIN_DIRS["BiomedCLIP static prompt baseline"])
    supplement_result_df = _read_result_csv(BASELINE_SUPPLEMENT_DIR / "result.csv")
    supplement_fold_df = pd.read_csv(BASELINE_SUPPLEMENT_DIR / "fold_metrics.csv")
    supplement_fold_df["fold"] = supplement_fold_df["fold"].astype(int) + 1

    merged_fold_df = train_fold_df.merge(
        supplement_fold_df,
        on="fold",
        how="left",
        suffixes=("_train", ""),
    )
    merged_fold_df["val_auc"] = merged_fold_df["val_auc"]
    merged_fold_df["test_auc"] = merged_fold_df["test_auc"]
    merged_fold_df["test_acc"] = merged_fold_df["test_acc"]
    merged_fold_df["test_f1"] = merged_fold_df["test_f1"]

    summary = {}
    for metric in METRICS:
        if metric == "val_auc":
            summary[metric] = (
                _safe_float(train_result_df.loc["mean", metric]),
                _safe_float(train_result_df.loc["std", metric]),
            )
        else:
            summary[metric] = (
                _safe_float(supplement_result_df.loc["mean", metric]),
                _safe_float(supplement_result_df.loc["std", metric]),
            )

    fold_cols = ["fold"] + METRICS
    return summary, merged_fold_df[fold_cols].copy()


def _build_standard_tables(train_dir: Path):
    if not (train_dir / "result.csv").is_file() or not (train_dir / "fold_summary.csv").is_file():
        return _empty_summary_and_folds()
    result_df, fold_df = _read_train_result_and_folds(train_dir)
    summary = {}
    for metric in METRICS:
        summary[metric] = (
            _safe_float(result_df.loc["mean", metric]),
            _safe_float(result_df.loc["std", metric]),
        )
    fold_cols = ["fold"] + [metric for metric in METRICS if metric in fold_df.columns]
    return summary, fold_df[fold_cols].copy()


def _collect_method_tables():
    summary_map = {}
    fold_map = {}
    for method in FINAL_METHODS:
        if method == "BiomedCLIP static prompt baseline":
            summary_map[method], fold_map[method] = _build_baseline_tables()
        else:
            summary_map[method], fold_map[method] = _build_standard_tables(TRAIN_DIRS[method])

    for method in ["PEPS low-only", "PEPS high-only"]:
        summary_map[method], fold_map[method] = _build_standard_tables(TRAIN_DIRS[method])
    return summary_map, fold_map


def _bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, seed: int = 20260428):
    rng = np.random.default_rng(seed)
    samples = []
    n = len(deltas)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(deltas[idx])))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _outcome_label(value: float, tol: float = 1e-12):
    if value > tol:
        return "W"
    if value < -tol:
        return "L"
    return "T"


def _build_main_table(summary_map):
    rows = []
    for method in FINAL_METHODS:
        row = {"method": method}
        for metric in METRICS:
            mean_value, std_value = summary_map[method][metric]
            row[metric] = _fmt_pm(mean_value, std_value)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_significance_table(fold_map):
    comparisons = [
        "BiomedCLIP static prompt baseline",
        "Concept-12 embedding_mean",
        "Concept-12 MLP dynamic_gate",
    ]
    rows = []
    candidate_df = fold_map[MAIN_MODEL].sort_values("fold").reset_index(drop=True)

    for reference in comparisons:
        reference_df = fold_map[reference].sort_values("fold").reset_index(drop=True)
        merged = candidate_df.merge(reference_df, on="fold", suffixes=("_cand", "_ref"))
        for metric in METRICS:
            cand = merged[f"{metric}_cand"].to_numpy(dtype=float)
            ref = merged[f"{metric}_ref"].to_numpy(dtype=float)
            deltas = cand - ref
            mean_delta = float(np.mean(deltas))
            ci_low, ci_high = _bootstrap_ci(deltas)
            ttest_p = float(ttest_rel(cand, ref).pvalue)
            try:
                wilcoxon_p = float(wilcoxon(deltas).pvalue)
            except Exception:
                wilcoxon_p = math.nan
            outcomes = [_outcome_label(v) for v in deltas]
            rows.append(
                {
                    "candidate_method": MAIN_MODEL,
                    "reference_method": reference,
                    "metric": metric,
                    "mean_delta": mean_delta,
                    "paired_ttest_pvalue": ttest_p,
                    "wilcoxon_pvalue": wilcoxon_p,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wins": int(sum(o == "W" for o in outcomes)),
                    "losses": int(sum(o == "L" for o in outcomes)),
                    "ties": int(sum(o == "T" for o in outcomes)),
                    "fold_outcomes": ",".join(outcomes),
                    "fold_deltas": ",".join(f"{v:.6f}" for v in deltas),
                }
            )
    return pd.DataFrame(rows)


def _build_scale_ablation_table(summary_map):
    methods = [
        ("PEPS low-only", "PEPS low-only"),
        ("PEPS high-only", "PEPS high-only"),
        ("PEPS dual-scale", MAIN_MODEL),
    ]
    rows = []
    for display_name, method in methods:
        row = {"method": display_name}
        for metric in METRICS:
            mean_value, std_value = summary_map[method][metric]
            row[metric] = _fmt_pm(mean_value, std_value)
        rows.append(row)
    return pd.DataFrame(rows)


def _shannon_entropy(items):
    if not items:
        return math.nan
    counts = Counter(items)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def _collapse_rate(items):
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


def _analyze_peps_main_interpretability():
    per_fold_rows = []
    top_prompt_rows = []
    all_prompt_rows = []

    for csv_path in sorted(PEPS_MAIN_EVAL_DIR.glob("peps_prompt_analysis_fold*.csv")):
        fold = int(csv_path.stem.split("fold")[-1]) + 1
        df = pd.read_csv(csv_path)
        pred_df = df[df["class_id"] == df["pred_label"]].copy()
        low_items = pred_df[pred_df["scale"] == "low"]["top1_prompt_text"].tolist()
        high_items = pred_df[pred_df["scale"] == "high"]["top1_prompt_text"].tolist()
        per_fold_rows.append(
            {
                "section": "diversity_per_fold",
                "experiment": MAIN_MODEL,
                "fold": fold,
                "low_top1_unique_prompts": len(set(low_items)),
                "high_top1_unique_prompts": len(set(high_items)),
                "low_prompt_entropy": _shannon_entropy(low_items),
                "high_prompt_entropy": _shannon_entropy(high_items),
                "low_prompt_collapse_rate": _collapse_rate(low_items),
                "high_prompt_collapse_rate": _collapse_rate(high_items),
            }
        )
        all_prompt_rows.append(df)

    all_df = pd.concat(all_prompt_rows, ignore_index=True)
    for true_label, class_name in [(0, "Adenocarcinoma"), (1, "NonAdenocarcinoma")]:
        class_df = all_df[(all_df["true_label"] == true_label) & (all_df["class_id"] == true_label)].copy()
        counts = (
            class_df.groupby(["top1_prompt_text", "top1_prompt_concept"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "top1_prompt_text"], ascending=[False, True])
            .head(10)
        )
        total = len(class_df)
        for rank, (_, row) in enumerate(counts.iterrows(), start=1):
            top_prompt_rows.append(
                {
                    "section": "top_prompts",
                    "experiment": MAIN_MODEL,
                    "true_class_name": class_name,
                    "rank": rank,
                    "prompt_text": row["top1_prompt_text"],
                    "prompt_concept": row["top1_prompt_concept"],
                    "count": int(row["count"]),
                    "frequency": float(row["count"] / total) if total else math.nan,
                }
            )

    per_fold_df = pd.DataFrame(per_fold_rows)
    diversity_overall_rows = []
    for experiment, eval_dir, family in [
        (MAIN_MODEL, PEPS_MAIN_EVAL_DIR, "peps"),
        ("Concept-12 MLP dynamic_gate", DYNAMIC_GATE_EVAL_DIR, "dynamic_gate"),
    ]:
        if family == "peps":
            fold_rows = per_fold_df[per_fold_df["experiment"] == experiment].copy()
        else:
            dg_rows = []
            for csv_path in sorted(eval_dir.glob("prompt_weight_analysis_fold*.csv")):
                fold = int(csv_path.stem.split("fold")[-1]) + 1
                df = pd.read_csv(csv_path)
                low_items = [_safe_literal_list(v)[0] for v in df["top3_pred_low_prompts"] if _safe_literal_list(v)]
                high_items = [_safe_literal_list(v)[0] for v in df["top3_pred_high_prompts"] if _safe_literal_list(v)]
                dg_rows.append(
                    {
                        "section": "diversity_per_fold",
                        "experiment": experiment,
                        "fold": fold,
                        "low_top1_unique_prompts": len(set(low_items)),
                        "high_top1_unique_prompts": len(set(high_items)),
                        "low_prompt_entropy": _shannon_entropy(low_items),
                        "high_prompt_entropy": _shannon_entropy(high_items),
                        "low_prompt_collapse_rate": _collapse_rate(low_items),
                        "high_prompt_collapse_rate": _collapse_rate(high_items),
                    }
                )
            fold_rows = pd.DataFrame(dg_rows)

        if fold_rows.empty:
            continue
        diversity_overall_rows.append(
            {
                "section": "diversity_overall",
                "experiment": experiment,
                "low_top1_unique_prompts_mean": float(fold_rows["low_top1_unique_prompts"].mean()),
                "high_top1_unique_prompts_mean": float(fold_rows["high_top1_unique_prompts"].mean()),
                "low_prompt_entropy_mean": float(fold_rows["low_prompt_entropy"].mean()),
                "high_prompt_entropy_mean": float(fold_rows["high_prompt_entropy"].mean()),
                "low_prompt_collapse_rate_mean": float(fold_rows["low_prompt_collapse_rate"].mean()),
                "high_prompt_collapse_rate_mean": float(fold_rows["high_prompt_collapse_rate"].mean()),
            }
        )
        if family == "dynamic_gate":
            per_fold_df = pd.concat([per_fold_df, fold_rows], ignore_index=True)

    return pd.concat(
        [
            pd.DataFrame(diversity_overall_rows),
            per_fold_df,
            pd.DataFrame(top_prompt_rows),
        ],
        ignore_index=True,
        sort=False,
    )


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.6f}" if isinstance(x, float) and not pd.isna(x) else ("NA" if pd.isna(x) else str(x))
        )
    headers = list(display_df.columns)
    rows = display_df.values.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def _build_report(main_table_df, significance_df, scale_df, interpret_df):
    overall_div = interpret_df[interpret_df["section"] == "diversity_overall"].copy()
    top_prompts = interpret_df[interpret_df["section"] == "top_prompts"].copy()

    lines = []
    lines.append("# Final DCP-ViLa-MIL Analysis")
    lines.append("")
    lines.append("## Fixed Models")
    lines.append("")
    lines.append(f"- Primary model: `{MAIN_MODEL}`")
    lines.append(f"- AUC-first reference: `{AUC_FIRST_MODEL}`")
    lines.append("")
    lines.append("## Final Main Table")
    lines.append("")
    lines.append(_markdown_table(main_table_df))
    lines.append("")
    lines.append("## Statistical Significance")
    lines.append("")
    lines.append(_markdown_table(significance_df))
    lines.append("")
    lines.append("## Scale Ablation")
    lines.append("")
    lines.append(_markdown_table(scale_df))
    lines.append("")
    lines.append("## Interpretability Summary")
    lines.append("")
    lines.append("### Diversity Overview")
    lines.append("")
    lines.append(_markdown_table(overall_div))
    lines.append("")
    lines.append("### Top-10 Prompts for the Main Model")
    lines.append("")
    lines.append(_markdown_table(top_prompts))
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_map, fold_map = _collect_method_tables()
    main_table_df = _build_main_table(summary_map)
    significance_df = _build_significance_table(fold_map)
    scale_df = _build_scale_ablation_table(summary_map)
    interpret_df = _analyze_peps_main_interpretability()

    final_main_csv = OUT_DIR / "final_main_table.csv"
    significance_csv = OUT_DIR / "significance_tests.csv"
    scale_csv = OUT_DIR / "scale_ablation.csv"
    interpret_csv = OUT_DIR / "interpretability_summary.csv"
    report_md = OUT_DIR / "final_report.md"

    main_table_df.to_csv(final_main_csv, index=False)
    significance_df.to_csv(significance_csv, index=False)
    scale_df.to_csv(scale_csv, index=False)
    interpret_df.to_csv(interpret_csv, index=False)
    report_md.write_text(_build_report(main_table_df, significance_df, scale_df, interpret_df), encoding="utf-8")

    print(f"Saved final main table to: {final_main_csv}")
    print(f"Saved significance tests to: {significance_csv}")
    print(f"Saved scale ablation to: {scale_csv}")
    print(f"Saved interpretability summary to: {interpret_csv}")
    print(f"Saved final report to: {report_md}")


if __name__ == "__main__":
    main()
