#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from stage64k_common import (
    best_epoch_summary,
    discover_result_files,
    dump_json,
    extract_metrics,
    find_prediction_pickle,
    find_run_dir,
    load_pickle_predictions,
    markdown_table,
    prediction_summary,
    read_csv_records,
    safe_float,
)


HISTORICAL_STEP58C_DEFAULT = Path(
    "/private/ljh-data/shared/Linux_school/ViLMIL_step58C_c1253e8/ViLa-MIL-main/results_stage58C_residual_constrained_configD_5fold/rce_v2_rcD_l003_t050_aux020_5fold_e20_s1"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-step58c-root", type=Path, default=HISTORICAL_STEP58C_DEFAULT)
    parser.add_argument("--oldcode-fold2", type=Path, required=True)
    parser.add_argument("--currentcode-fold2", type=Path, required=True)
    parser.add_argument("--oldcode-fold5", type=Path, required=True)
    parser.add_argument("--currentcode-fold5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_run_payload(path: Path, *, fold_index: int, human_fold: int, historical: bool):
    run_dir = find_run_dir(path)
    files = discover_result_files(run_dir)

    summary_row = {}
    fold_summary_row = {}
    result_rows = []
    epoch_rows = []

    if "summary" in files:
        summary_records = read_csv_records(files["summary"])
        if summary_records:
            if historical:
                summary_row = next(
                    (row for row in summary_records if int(float(row.get("folds", -1))) == fold_index),
                    summary_records[0],
                )
            else:
                summary_row = summary_records[0]
    if "fold_summary" in files:
        fold_records = read_csv_records(files["fold_summary"])
        if fold_records:
            fold_summary_row = next(
                (row for row in fold_records if int(float(row.get("fold", -1))) == human_fold),
                fold_records[0],
            )
    if "result" in files:
        result_rows = read_csv_records(files["result"])
    if "epoch_details" in files:
        epoch_rows = read_csv_records(files["epoch_details"])
        filtered_epoch_rows = [
            row for row in epoch_rows if int(float(row.get("fold", -1))) == human_fold
        ]
        if filtered_epoch_rows:
            epoch_rows = filtered_epoch_rows

    pickle_path = find_prediction_pickle(files.get("pickles", []), fold_index)
    predictions = load_pickle_predictions(pickle_path) if pickle_path else {}

    metrics_row = fold_summary_row or summary_row
    metrics = extract_metrics(metrics_row)
    epoch_summary = best_epoch_summary(epoch_rows)
    return {
        "input_path": str(path),
        "run_dir": str(run_dir),
        "files": {key: ([str(item) for item in value] if isinstance(value, list) else str(value)) for key, value in files.items()},
        "metrics": metrics,
        "summary_row": summary_row,
        "fold_summary_row": fold_summary_row,
        "epoch_summary": epoch_summary,
        "prediction_summary": prediction_summary(predictions),
        "prediction_pickle": str(pickle_path) if pickle_path else None,
    }


def diff_metrics(left: dict[str, object], right: dict[str, object]):
    metrics = {}
    for key in sorted(set(left) | set(right)):
        left_value = safe_float(left.get(key))
        right_value = safe_float(right.get(key))
        metrics[key] = {
            "left": left.get(key),
            "right": right.get(key),
            "abs_diff": None
            if left_value is None or right_value is None
            else abs(right_value - left_value),
            "signed_diff": None
            if left_value is None or right_value is None
            else right_value - left_value,
        }
    return metrics


def fold_report(name: str, historical_payload, old_payload, current_payload):
    deltas = {
        "historical_vs_old": diff_metrics(historical_payload["metrics"], old_payload["metrics"]),
        "historical_vs_current": diff_metrics(historical_payload["metrics"], current_payload["metrics"]),
        "old_vs_current": diff_metrics(old_payload["metrics"], current_payload["metrics"]),
    }
    return {
        "fold_name": name,
        "historical": historical_payload,
        "oldcode": old_payload,
        "currentcode": current_payload,
        "metric_deltas": deltas,
    }


def heuristic_decision(fold2, fold5):
    def collect_auc_acc(payload):
        auc = safe_float(payload["metrics"].get("test_auc"))
        acc = safe_float(payload["metrics"].get("test_acc"))
        return auc, acc

    historical_vs_old = []
    old_vs_current = []
    for fold_payload in (fold2, fold5):
        h_auc, h_acc = collect_auc_acc(fold_payload["historical"])
        o_auc, o_acc = collect_auc_acc(fold_payload["oldcode"])
        c_auc, c_acc = collect_auc_acc(fold_payload["currentcode"])
        if None not in (h_auc, o_auc):
            historical_vs_old.append(abs(h_auc - o_auc))
        if None not in (o_auc, c_auc):
            old_vs_current.append(abs(o_auc - c_auc))
        if None not in (h_acc, o_acc):
            historical_vs_old.append(abs(h_acc - o_acc))
        if None not in (o_acc, c_acc):
            old_vs_current.append(abs(o_acc - c_acc))

    old_close_to_historical = historical_vs_old and max(historical_vs_old) <= 0.01
    current_close_to_old = old_vs_current and max(old_vs_current) <= 0.01
    if old_close_to_historical and not current_close_to_old:
        root_cause = "情况1：当前代码行为或 BiomedCLIP 新加载逻辑更可疑。"
    elif current_close_to_old and not old_close_to_historical:
        root_cause = "情况2：服务器环境、H5、BiomedCLIP cache provenance 或非确定性更可疑。"
    elif current_close_to_old and old_close_to_historical:
        root_cause = "情况6：旧代码和当前代码都接近历史 Step58C，更像单次训练轨迹波动。"
    else:
        root_cause = "三方指标仍有混合差异，需要结合 probe 和 epoch 行为进一步判断。"
    return {
        "old_close_to_historical": old_close_to_historical,
        "current_close_to_old": current_close_to_old,
        "root_cause_hint": root_cause,
        "threshold_note": "辅助阈值参考：AUC <= 0.005, ACC <= 0.01。",
    }


def fold_markdown(title: str, payload):
    rows = []
    for label, run_payload in [
        ("historical_step58C", payload["historical"]),
        ("new_server_oldcode", payload["oldcode"]),
        ("new_server_currentcode", payload["currentcode"]),
    ]:
        metrics = run_payload["metrics"]
        epoch_summary = run_payload["epoch_summary"]
        rows.append(
            [
                label,
                metrics.get("test_auc"),
                metrics.get("test_acc"),
                metrics.get("test_f1"),
                metrics.get("balanced_acc"),
                metrics.get("sensitivity"),
                metrics.get("specificity"),
                metrics.get("pr_auc"),
                metrics.get("val_auc"),
                epoch_summary.get("best_val_auc_epoch"),
                epoch_summary.get("last_epoch"),
            ]
        )
    return "\n".join(
        [
            f"# {title}",
            "",
            markdown_table(
                [
                    "run",
                    "test_auc",
                    "test_acc",
                    "test_f1",
                    "balanced_acc",
                    "sensitivity",
                    "specificity",
                    "pr_auc",
                    "val_auc",
                    "best_val_auc_epoch",
                    "last_epoch",
                ],
                rows,
            ),
            "",
        ]
    )


def build_metrics_csv(fold2, fold5, output_path: Path):
    rows = []
    for fold_label, payload in [("fold2", fold2), ("fold5", fold5)]:
        for run_label, run_payload in [
            ("historical_step58C", payload["historical"]),
            ("new_server_oldcode", payload["oldcode"]),
            ("new_server_currentcode", payload["currentcode"]),
        ]:
            row = {"fold": fold_label, "run": run_label}
            row.update(run_payload["metrics"])
            row.update({f"epoch_{key}": value for key, value in run_payload["epoch_summary"].items()})
            row.update({f"pred_{key}": value for key, value in run_payload["prediction_summary"].items() if not isinstance(value, dict)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    historical_fold2 = load_run_payload(args.historical_step58c_root, fold_index=1, human_fold=2, historical=True)
    old_fold2 = load_run_payload(args.oldcode_fold2, fold_index=1, human_fold=2, historical=False)
    current_fold2 = load_run_payload(args.currentcode_fold2, fold_index=1, human_fold=2, historical=False)

    historical_fold5 = load_run_payload(args.historical_step58c_root, fold_index=4, human_fold=5, historical=True)
    old_fold5 = load_run_payload(args.oldcode_fold5, fold_index=4, human_fold=5, historical=False)
    current_fold5 = load_run_payload(args.currentcode_fold5, fold_index=4, human_fold=5, historical=False)

    fold2 = fold_report("Fold 2", historical_fold2, old_fold2, current_fold2)
    fold5 = fold_report("Fold 5", historical_fold5, old_fold5, current_fold5)
    decision = heuristic_decision(fold2, fold5)

    build_metrics_csv(fold2, fold5, args.output_dir / "three_way_metrics.csv")
    (args.output_dir / "fold2_compare.md").write_text(fold_markdown("Fold 2", fold2), encoding="utf-8")
    (args.output_dir / "fold5_compare.md").write_text(fold_markdown("Fold 5", fold5), encoding="utf-8")
    (args.output_dir / "decision.md").write_text(
        "\n".join(
            [
                "# Step64K Three-Way Decision",
                "",
                f"- old_close_to_historical: {decision['old_close_to_historical']}",
                f"- current_close_to_old: {decision['current_close_to_old']}",
                f"- root_cause_hint: {decision['root_cause_hint']}",
                f"- threshold_note: {decision['threshold_note']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    dump_json(
        args.output_dir / "three_way_compare.json",
        {
            "fold2": fold2,
            "fold5": fold5,
            "decision": decision,
        },
    )


if __name__ == "__main__":
    main()
