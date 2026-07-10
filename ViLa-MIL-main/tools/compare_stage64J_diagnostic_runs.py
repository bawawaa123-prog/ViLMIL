#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "test_auc",
    "test_acc",
    "test_f1",
    "test_balanced_acc",
    "test_sensitivity",
    "test_specificity",
    "test_pr_auc",
    "val_auc",
]


def load_result_dir(path: Path) -> dict:
    payload: dict[str, object] = {"path": str(path)}
    summary = path / "summary.csv"
    result = path / "result.csv"
    epoch_details = path / "epoch_details.csv"
    if summary.exists():
        payload["summary"] = pd.read_csv(summary).to_dict(orient="records")
    if result.exists():
        payload["result"] = pd.read_csv(result).to_dict(orient="records")
    if epoch_details.exists():
        payload["epoch_details"] = pd.read_csv(epoch_details).to_dict(orient="records")
    return payload


def first_metrics(payload: dict) -> dict[str, object]:
    for key in ("summary", "result"):
        rows = payload.get(key)
        if rows:
            row = rows[0]
            return {metric: row.get(metric) for metric in METRIC_COLUMNS if metric in row}
    return {}


def heuristics(old_payload: dict, current_payload: dict) -> dict[str, object]:
    old_metrics = first_metrics(old_payload)
    current_metrics = first_metrics(current_payload)
    deltas = {}
    for key in sorted(set(old_metrics) & set(current_metrics)):
        try:
            deltas[key] = float(current_metrics[key]) - float(old_metrics[key])
        except Exception:
            pass
    auc_delta = abs(deltas.get("test_auc", 0.0))
    acc_delta = abs(deltas.get("test_acc", 0.0))
    return {
        "metric_deltas_current_minus_old": deltas,
        "old_code_on_new_server_close_to_old_step58c": auc_delta < 0.005 and acc_delta < 0.02,
        "current_vs_old_code_has_visible_gap": auc_delta >= 0.005 or acc_delta >= 0.02,
        "difference_looks_more_like_threshold_shift_than_ranking_shift": abs(deltas.get("test_acc", 0.0)) > abs(deltas.get("test_auc", 0.0)),
        "recommend_deterministic_audit": auc_delta >= 0.002 or acc_delta >= 0.01,
        "recommend_best_checkpoint_audit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-fold2", type=Path, required=False)
    parser.add_argument("--current-fold2", type=Path, required=False)
    parser.add_argument("--old-fold5", type=Path, required=False)
    parser.add_argument("--current-fold5", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args()

    report = {}
    if args.old_fold2 and args.current_fold2:
        old_payload = load_result_dir(args.old_fold2)
        current_payload = load_result_dir(args.current_fold2)
        report["fold2"] = {
            "old": old_payload,
            "current": current_payload,
            "heuristics": heuristics(old_payload, current_payload),
        }
    if args.old_fold5 and args.current_fold5:
        old_payload = load_result_dir(args.old_fold5)
        current_payload = load_result_dir(args.current_fold5)
        report["fold5"] = {
            "old": old_payload,
            "current": current_payload,
            "heuristics": heuristics(old_payload, current_payload),
        }

    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
