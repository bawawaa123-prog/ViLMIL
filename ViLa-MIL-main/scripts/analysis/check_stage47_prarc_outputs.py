from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step47 PRARC summary/diagnostic outputs.")
    parser.add_argument("--summary_dir", default="results_stage47/stage47_prarc_gate_summary")
    parser.add_argument("--diagnostics_dir", default="results_stage47/stage47_prarc_gate_diagnostics")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    summary_dir = resolve_path(args.summary_dir)
    diagnostics_dir = resolve_path(args.diagnostics_dir)

    required_summary = [
        "stage47_prarc_5fold_summary.csv",
        "stage47_prarc_fold_metrics.csv",
        "stage47_prarc_vs_baseline.csv",
        "stage47_prarc_stability_check.csv",
        "stage47_prarc_gate_report.md",
        "stage47_manifest.json",
    ]
    required_diagnostics = [
        "stage47_prarc_gate_probe_slide_level.csv",
        "stage47_prarc_gate_distribution_summary.csv",
        "stage47_prarc_gate_by_condition.csv",
        "stage47_prarc_gate_feature_correlation.csv",
        "stage47_prarc_gate_diagnostics_report.md",
        "stage47_prarc_gate_diagnostics_manifest.json",
    ]

    failures: list[str] = []
    for name in required_summary:
        if not (summary_dir / name).is_file():
            failures.append(f"Missing summary output: {summary_dir / name}")
    for name in required_diagnostics:
        if not (diagnostics_dir / name).is_file():
            failures.append(f"Missing diagnostics output: {diagnostics_dir / name}")

    if failures:
        print("[FAIL] Step47 output check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[PASS] Step47 output check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
