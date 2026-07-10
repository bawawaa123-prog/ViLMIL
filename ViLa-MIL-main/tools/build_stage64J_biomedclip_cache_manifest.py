#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


MODEL_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
TEXT_REPO = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def candidate_cache_dirs(project_root: Path) -> list[Path]:
    candidates = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        project_root / "hf_cache",
        project_root.parent / "hf_cache",
        project_root / "model_cache",
    ]
    try:
        shared_root = project_root.parents[2]
        candidates.extend([
            shared_root / "ViLMIL" / "hf_cache",
            shared_root / "ViLMIL" / "ViLa-MIL-main" / "hf_cache",
        ])
    except IndexError:
        pass
    resolved = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if str(path) in seen:
            continue
        seen.add(str(path))
        resolved.append(path)
    return resolved


def snapshot_dirs(cache_dir: Path, repo_id: str) -> list[Path]:
    base = cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = args.project_root.resolve()

    selected_cache = None
    selected_snapshot = None
    rows = []
    all_snapshots = []
    for cache_dir in candidate_cache_dirs(project_root):
        if not cache_dir.exists():
            continue
        snapshots = snapshot_dirs(cache_dir, MODEL_REPO)
        text_snapshots = snapshot_dirs(cache_dir, TEXT_REPO)
        all_snapshots.extend([str(p) for p in snapshots])
        if snapshots and selected_snapshot is None:
            selected_cache = cache_dir
            selected_snapshot = snapshots[-1]
        for snapshot in snapshots:
            for file_path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
                rows.append(
                    {
                        "cache_dir": str(cache_dir),
                        "repo_id": MODEL_REPO,
                        "snapshot_dir": str(snapshot),
                        "relative_path": file_path.relative_to(snapshot).as_posix(),
                        "size_bytes": file_path.stat().st_size,
                        "sha256": sha256_file(file_path),
                    }
                )
        for snapshot in text_snapshots:
            for file_path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
                rows.append(
                    {
                        "cache_dir": str(cache_dir),
                        "repo_id": TEXT_REPO,
                        "snapshot_dir": str(snapshot),
                        "relative_path": file_path.relative_to(snapshot).as_posix(),
                        "size_bytes": file_path.stat().st_size,
                        "sha256": sha256_file(file_path),
                    }
                )

    manifest = {
        "model_name": f"hf-hub:{MODEL_REPO}",
        "text_repo": TEXT_REPO,
        "candidate_cache_dirs": [str(p) for p in candidate_cache_dirs(project_root)],
        "selected_cache_dir": str(selected_cache) if selected_cache else None,
        "selected_snapshot_dir": str(selected_snapshot) if selected_snapshot else None,
        "all_model_snapshots": all_snapshots,
        "multiple_model_snapshots_present": len(all_snapshots) > 1,
        "selection_rule": "last lexicographically sorted snapshot under the first cache dir containing the model snapshot",
    }

    (args.output_dir / "biomedclip_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "biomedclip_cache_files.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cache_dir", "repo_id", "snapshot_dir", "relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
