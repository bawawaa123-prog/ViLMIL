from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_file_clean(repo_root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet", "--", relative_path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def write_hash_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    original_rel = "models/model_RCE_MIL_BiomedCLIP.py"
    v2_rel = "models/model_RCE_MIL_BiomedCLIP_v2.py"
    main_rel = "main.py"
    core_rel = "utils/core_utils.py"

    original_path = repo_root / original_rel
    v2_path = repo_root / v2_rel
    main_path = repo_root / main_rel
    core_path = repo_root / core_rel

    output_dir = repo_root / "results_stage57A_rce_v2_copy_equivalence"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_sha = sha256sum(original_path)
    v2_sha = sha256sum(v2_path)
    original_bytes = original_path.read_bytes()
    v2_bytes = v2_path.read_bytes()

    main_text = main_path.read_text(encoding="utf-8")
    core_text = core_path.read_text(encoding="utf-8")

    checks = {
        "original_rce_file_exists": original_path.exists(),
        "v2_rce_file_exists": v2_path.exists(),
        "original_rce_file_unmodified_in_git_diff": git_file_clean(repo_root, original_rel),
        "v2_is_complete_copy_of_original": original_bytes == v2_bytes,
        "main_supports_rce_v2_model_type": '"RCE_MIL_BiomedCLIP_v2"' in main_text,
        "core_utils_supports_rce_v2_model_type": "args.model_type == 'RCE_MIL_BiomedCLIP_v2'" in core_text,
        "core_utils_imports_rce_v2_alias": (
            "from models.model_RCE_MIL_BiomedCLIP_v2 import RCE_MIL_BiomedCLIP as RCE_MIL_BiomedCLIP_v2"
            in core_text
        ),
    }
    checks["all_passed"] = all(checks.values())

    hash_rows = [
        {"file": original_rel, "sha256": original_sha},
        {"file": v2_rel, "sha256": v2_sha},
    ]
    write_hash_csv(output_dir / "stage57A_file_hashes.csv", hash_rows)

    audit_payload = {
        "step": "Step57A",
        "goal": "RCE-v2 copy equivalence test",
        "files": {
            "original_rce_model": original_rel,
            "rce_v2_model": v2_rel,
            "main": main_rel,
            "core_utils": core_rel,
        },
        "sha256": {
            original_rel: original_sha,
            v2_rel: v2_sha,
        },
        "checks": checks,
        "conclusion": (
            "No new model logic was introduced in Step57A. "
            "The v2 model file is a safe copy entrypoint for future experimentation."
        ),
    }
    (output_dir / "stage57A_equivalence_audit.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# Step57A Summary",
        "",
        "## Scope",
        "",
        "- Step57A did not introduce any new model logic.",
        "- `models/model_RCE_MIL_BiomedCLIP.py` remained untouched.",
        "- `models/model_RCE_MIL_BiomedCLIP_v2.py` is intended as a safe copy entrypoint for later innovation.",
        "",
        "## File Hashes",
        "",
        f"- Original RCE sha256: `{original_sha}`",
        f"- RCE v2 sha256: `{v2_sha}`",
        "",
        "## Static Checks",
        "",
        f"- Original RCE file exists: `{checks['original_rce_file_exists']}`",
        f"- RCE v2 file exists: `{checks['v2_rce_file_exists']}`",
        f"- Original RCE file unmodified in git diff: `{checks['original_rce_file_unmodified_in_git_diff']}`",
        f"- RCE v2 is a complete copy of original: `{checks['v2_is_complete_copy_of_original']}`",
        f"- `main.py` supports `RCE_MIL_BiomedCLIP_v2`: `{checks['main_supports_rce_v2_model_type']}`",
        f"- `utils/core_utils.py` supports `RCE_MIL_BiomedCLIP_v2`: `{checks['core_utils_supports_rce_v2_model_type']}`",
        f"- `utils/core_utils.py` imports the v2 alias: `{checks['core_utils_imports_rce_v2_alias']}`",
        f"- Overall audit passed: `{checks['all_passed']}`",
        "",
        "## Conclusion",
        "",
        "- Step57A only adds a safe RCE-v2 copy and training entrypoint.",
        "- Future innovation should happen in `models/model_RCE_MIL_BiomedCLIP_v2.py`, not in the locked original RCE file.",
    ]
    (output_dir / "stage57A_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
