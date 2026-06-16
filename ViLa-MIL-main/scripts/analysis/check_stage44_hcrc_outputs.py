from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step44 HCRC output completeness.")
    parser.add_argument("--results_root", default="results_stage44")
    parser.add_argument("--variants", default="hcrc_a002_b8,hcrc_a005_b8,hcrc_a01_b8")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def parse_variants(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    failures: list[str] = []

    for variant in parse_variants(args.variants):
        run_dir = results_root / f"stage44_{variant}_s{args.seed}"
        log_path = results_root / "logs" / f"stage44_{variant}_s{args.seed}.log"
        fold_summary = run_dir / "fold_summary.csv"

        if not run_dir.is_dir():
            failures.append(f"Missing run directory: {run_dir}")
            continue
        if not log_path.is_file():
            failures.append(f"Missing log: {log_path}")
        if not fold_summary.is_file():
            failures.append(f"Missing fold_summary.csv: {fold_summary}")
        checkpoints = sorted(run_dir.glob("s_*_checkpoint.pt"))
        if len(checkpoints) < 5:
            failures.append(f"Expected 5 checkpoints, found {len(checkpoints)} in {run_dir}")
        if fold_summary.is_file():
            try:
                df = pd.read_csv(fold_summary)
                if len(df) < 5:
                    failures.append(f"Expected >=5 fold rows, found {len(df)} in {fold_summary}")
            except Exception as exc:
                failures.append(f"Failed to read {fold_summary}: {exc}")

    if failures:
        print("[FAIL] Step44 outputs check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[PASS] Step44 outputs check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
