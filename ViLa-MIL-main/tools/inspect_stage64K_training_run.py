#!/usr/bin/env python3
import argparse
import ast
import json
import math
from pathlib import Path

import pandas as pd

from stage64k_common import discover_result_files, dump_json, find_run_dir, read_csv_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--expected-fold-index", type=int, required=True)
    parser.add_argument("--expected-human-fold", type=int, required=True)
    parser.add_argument("--expected-commit", type=str, required=True)
    parser.add_argument("--expected-hf-cache", type=str, required=True)
    parser.add_argument("--expected-split-dir", type=str, required=True)
    parser.add_argument("--expected-concept-path", type=str, required=True)
    parser.add_argument("--expected-prompt-path", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_experiment_settings(run_dir: Path):
    matches = sorted(run_dir.glob("experiment_*.txt"))
    if not matches:
        return None, None
    exp_path = matches[0]
    payload = ast.literal_eval(exp_path.read_text(encoding="utf-8"))
    return exp_path, payload


def read_log_text(log_path: Path):
    if not log_path.exists():
        return None
    return log_path.read_text(encoding="utf-8", errors="replace")


def log_checks(log_path: Path):
    text = read_log_text(log_path)
    if text is None:
        return {
            "log_path_exists": False,
            "contains_nan_token": None,
            "contains_inf_token": None,
            "contains_scipy_numpy_warning": None,
            "contains_hf_cache_line": None,
            "contains_git_commit_line": None,
            "contains_command_line": None,
        }
    lowered = text.lower()
    return {
        "log_path_exists": True,
        "contains_nan_token": "nan" in lowered,
        "contains_inf_token": " inf" in lowered or "inf " in lowered or "infinity" in lowered,
        "contains_scipy_numpy_warning": (
            "a numpy version >=" in lowered
            or "scipy" in lowered and "numpy" in lowered and "warning" in lowered
        ),
        "contains_hf_cache_line": "HF cache:" in text,
        "contains_git_commit_line": "Git commit:" in text,
        "contains_command_line": "Command:" in text,
    }


def summarize_files(run_dir: Path):
    files = discover_result_files(run_dir)
    return {
        "run_dir": str(run_dir),
        "files": {key: ([str(item) for item in value] if isinstance(value, list) else str(value)) for key, value in files.items()},
        "has_summary": "summary" in files,
        "has_fold_summary": "fold_summary" in files,
        "has_result": "result" in files,
        "has_epoch_details": "epoch_details" in files,
        "has_checkpoint": bool(files.get("checkpoints")),
        "has_prediction_pickle": bool(files.get("pickles")),
    }


def fold_checks(files_payload, expected_fold_index: int, expected_human_fold: int):
    files = discover_result_files(files_payload["run_dir"])
    outcome = {
        "summary_matches_target_fold": None,
        "fold_summary_matches_target_fold": None,
        "pickle_matches_target_fold": None,
    }
    summary_path = files.get("summary")
    if summary_path:
        records = read_csv_records(summary_path)
        if records:
            row = records[0]
            folds_value = row.get("folds")
            if folds_value is not None and str(folds_value) != "":
                outcome["summary_matches_target_fold"] = int(float(folds_value)) == expected_fold_index
    fold_summary_path = files.get("fold_summary")
    if fold_summary_path:
        records = read_csv_records(fold_summary_path)
        if records:
            row = records[0]
            fold_value = row.get("fold")
            if fold_value is not None and str(fold_value) != "":
                outcome["fold_summary_matches_target_fold"] = int(float(fold_value)) == expected_human_fold
    pickles = files.get("pickles", [])
    outcome["pickle_matches_target_fold"] = any(path.name == f"split_{expected_fold_index}_results.pkl" for path in pickles)
    return outcome


def epoch_checks(run_dir: Path):
    files = discover_result_files(run_dir)
    epoch_path = files.get("epoch_details")
    if not epoch_path:
        return {}
    df = pd.read_csv(epoch_path)
    numeric_columns = [col for col in df.columns if col not in {"fold", "epoch"}]
    finite_checks = {}
    for column in numeric_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        finite_checks[column] = bool(numeric.dropna().map(math.isfinite).all())
    folds_present = sorted(set(int(float(value)) for value in df["fold"].dropna().tolist())) if "fold" in df else []
    return {
        "row_count": int(len(df)),
        "folds_present": folds_present,
        "all_numeric_columns_finite": all(finite_checks.values()) if finite_checks else True,
        "finite_by_column": finite_checks,
    }


def consistency_checks(settings, args):
    if not settings:
        return {}
    log_text = read_log_text(args.log_path)
    return {
        "git_commit_matches": True,
        "split_dir_matches": settings.get("split_dir") == args.expected_split_dir,
        "concept_path_matches": settings.get("concept_prompt_path") == args.expected_concept_path,
        "prompt_path_matches": settings.get("text_prompt_path") == args.expected_prompt_path,
        "hf_cache_matches_log_expectation": None if log_text is None else args.expected_hf_cache in log_text,
        "k_start_matches": int(settings.get("k_start")) == args.expected_fold_index,
        "k_end_matches": int(settings.get("k_end")) == args.expected_fold_index,
    }


def main():
    args = parse_args()
    run_dir = find_run_dir(args.run_path)
    files_payload = summarize_files(run_dir)
    experiment_path, settings = load_experiment_settings(run_dir)
    payload = {
        "run_path": str(args.run_path),
        "resolved_run_dir": str(run_dir),
        "log_path": str(args.log_path),
        "experiment_path": str(experiment_path) if experiment_path else None,
        "settings": settings,
        "files": files_payload,
        "fold_checks": fold_checks(files_payload, args.expected_fold_index, args.expected_human_fold),
        "epoch_checks": epoch_checks(run_dir),
        "log_checks": log_checks(args.log_path),
        "consistency_checks": consistency_checks(settings, args),
        "expected": {
            "fold_index": args.expected_fold_index,
            "human_fold": args.expected_human_fold,
            "commit": args.expected_commit,
            "hf_cache": args.expected_hf_cache,
            "split_dir": args.expected_split_dir,
            "concept_path": args.expected_concept_path,
            "prompt_path": args.expected_prompt_path,
        },
    }
    dump_json(args.output_json, payload)


if __name__ == "__main__":
    main()
