#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT_DIR / "experiment_outputs" / "stage_results"
CATEGORY_ORDER = [
    "stage57_to_63_rce_innovation",
    "stage51_to_56_rce_repro_ablation",
    "stage41_to_50_prarc_hcrc_final_package",
    "stage19_to_40_legacy_exploration",
    "other",
]


def infer_stage_id(directory_name: str) -> str:
    suffix = directory_name[len("results_stage") :]
    match = re.match(r"([0-9]+[A-Za-z]?)", suffix)
    if match:
        return match.group(1)
    return suffix or "unknown"


def categorize_stage(stage_id: str) -> str:
    match = re.match(r"(\d+)", stage_id)
    if not match:
        return "other"

    stage_num = int(match.group(1))
    if 57 <= stage_num <= 63:
        return "stage57_to_63_rce_innovation"
    if 51 <= stage_num <= 56:
        return "stage51_to_56_rce_repro_ablation"
    if 41 <= stage_num <= 50:
        return "stage41_to_50_prarc_hcrc_final_package"
    if 19 <= stage_num <= 40:
        return "stage19_to_40_legacy_exploration"
    return "other"


def collect_stage_dirs(base_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in base_dir.glob("results_stage*")
        if path.is_dir()
    )


def main() -> int:
    root_stage_dirs = collect_stage_dirs(ROOT_DIR)
    archived_stage_dirs = collect_stage_dirs(ARCHIVE_DIR) if ARCHIVE_DIR.exists() else []

    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"ARCHIVE_DIR: {ARCHIVE_DIR}")
    print()

    print(f"Root-level results_stage* directories: {len(root_stage_dirs)}")
    if root_stage_dirs:
        for path in root_stage_dirs:
            print(f"  - {path.name}")
    else:
        print("  - none")
    print()

    print(f"Archived results_stage* directories: {len(archived_stage_dirs)}")
    if archived_stage_dirs:
        print(f"  - first: {archived_stage_dirs[0].name}")
        print(f"  - last: {archived_stage_dirs[-1].name}")
    else:
        print("  - none")
    print()

    category_counts: Counter[str] = Counter()
    for path in archived_stage_dirs:
        stage_id = infer_stage_id(path.name)
        category_counts[categorize_stage(stage_id)] += 1

    print("stage_results_index candidate statistics:")
    print(f"  - total rows: {len(archived_stage_dirs)}")
    for category in CATEGORY_ORDER:
        print(f"  - {category}: {category_counts.get(category, 0)}")
    print()

    if root_stage_dirs:
        print("Layout check failed: root-level results_stage* directories are still present.")
        return 1
    if not ARCHIVE_DIR.exists():
        print("Layout check failed: archive directory does not exist.")
        return 1

    print("Layout check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
