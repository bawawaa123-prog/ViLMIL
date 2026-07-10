#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "data_root_dir",
    "results_dir",
    "concept_prompt_path",
    "text_prompt_path",
    "split_dir",
    "data_folder_s",
    "data_folder_l",
}


def load_literal_dict(path: Path) -> dict[str, Any]:
    data = ast.literal_eval(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a dict literal")
    return data


def ast_to_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return "<non_literal>"


def extract_arg_defaults(main_py: Path) -> dict[str, Any]:
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        option = first.value
        if not option.startswith("--"):
            continue
        key = option[2:]
        default = "<no_default>"
        for kw in node.keywords:
            if kw.arg == "default":
                default = ast_to_value(kw.value)
                break
        defaults[key] = default
    return defaults


def extract_explicit_args_from_stage58c_script(script_path: Path) -> set[str]:
    explicit: set[str] = set()
    for line in script_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("--"):
            continue
        token = line.split()[0]
        explicit.add(token[2:])
    return explicit


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dir_signature(path: Path) -> str:
    hasher = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for file_path in files:
        rel = file_path.relative_to(path).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(file_path.stat().st_size).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(file_sha256(file_path).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def resolve_candidate(project_root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()
    candidate = (project_root / raw).resolve()
    if candidate.exists():
        return candidate
    return None


def path_semantic_note(key: str, old_value: Any, new_value: Any, old_project: Path, new_project: Path) -> str:
    if key not in PATH_KEYS:
        return ""
    old_path = resolve_candidate(old_project, old_value)
    new_path = resolve_candidate(new_project, new_value)
    if old_path is None or new_path is None:
        return "unresolved_path"
    if old_path.is_file() and new_path.is_file():
        return "same_content" if file_sha256(old_path) == file_sha256(new_path) else "different_content"
    if old_path.is_dir() and new_path.is_dir():
        return "same_tree" if dir_signature(old_path) == dir_signature(new_path) else "different_tree"
    return "different_kinds"


def compare(
    old_cfg: dict[str, Any],
    new_cfg: dict[str, Any],
    old_defaults: dict[str, Any],
    new_defaults: dict[str, Any],
    old_explicit: set[str],
    old_project: Path,
    new_project: Path,
) -> list[dict[str, Any]]:
    keys = sorted(set(old_cfg) | set(new_cfg))
    rows: list[dict[str, Any]] = []
    for key in keys:
        old_present = key in old_cfg
        new_present = key in new_cfg
        old_value = old_cfg.get(key)
        new_value = new_cfg.get(key)
        if old_present and new_present:
            status = "equal" if old_value == new_value else "different"
        elif old_present:
            status = "only_old"
        else:
            status = "only_new"
        note = path_semantic_note(key, old_value, new_value, old_project, new_project)
        row = {
            "key": key,
            "status": status,
            "old_value": stable_json(old_value) if old_present else "",
            "new_value": stable_json(new_value) if new_present else "",
            "old_default": stable_json(old_defaults.get(key, "<missing>")),
            "new_default": stable_json(new_defaults.get(key, "<missing>")),
            "old_explicit_in_stage58c_script": key in old_explicit,
            "old_matches_old_default": old_present and old_value == old_defaults.get(key, object()),
            "new_matches_new_default": new_present and new_value == new_defaults.get(key, object()),
            "path_semantic_note": note,
        }
        rows.append(row)
    return rows


def write_markdown(rows: list[dict[str, Any]], output_md: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {
        "equal": [],
        "different": [],
        "only_old": [],
        "only_new": [],
    }
    path_semantic = []
    inferred_defaults = []
    for row in rows:
        grouped.setdefault(row["status"], []).append(row)
        if row["path_semantic_note"] in {"same_content", "same_tree", "different_content", "different_tree"}:
            path_semantic.append(row)
        if row["new_matches_new_default"]:
            inferred_defaults.append(row)

    lines = [
        "# Stage64J Config Diff",
        "",
        "## Equal Parameters",
    ]
    lines += [f"- `{row['key']}` = {row['old_value']}" for row in grouped["equal"]] or ["- None"]
    lines += ["", "## Different Parameters"]
    lines += [
        f"- `{row['key']}`: old={row['old_value']} | new={row['new_value']} | path_note={row['path_semantic_note'] or 'n/a'}"
        for row in grouped["different"]
    ] or ["- None"]
    lines += ["", "## Only In Old Config"]
    lines += [f"- `{row['key']}` = {row['old_value']}" for row in grouped["only_old"]] or ["- None"]
    lines += ["", "## Only In New Config"]
    lines += [f"- `{row['key']}` = {row['new_value']}" for row in grouped["only_new"]] or ["- None"]
    lines += ["", "## Path Parameters With Content Check"]
    lines += [
        f"- `{row['key']}`: old={row['old_value']} | new={row['new_value']} | content={row['path_semantic_note']}"
        for row in path_semantic
    ] or ["- None"]
    lines += ["", "## Args Matching argparse Defaults"]
    lines += [
        f"- `{row['key']}`: old_matches_default={row['old_matches_old_default']} | new_matches_default={row['new_matches_new_default']} | old_explicit_stage58c={row['old_explicit_in_stage58c_script']}"
        for row in inferred_defaults
    ] or ["- None"]
    lines += [
        "",
        "## Notes",
        "- `old_explicit_in_stage58c_script` is derived from `scripts/experiments/run_stage58C_residual_constrained_configD_5fold.sh`.",
        "- No authoritative Step64I launch script was found in the result directory, so `new_matches_new_default` is an inference based on `main.py` defaults, not proof that the CLI omitted that argument.",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-config", required=True, type=Path)
    parser.add_argument("--new-config", required=True, type=Path)
    parser.add_argument("--old-main", required=True, type=Path)
    parser.add_argument("--new-main", required=True, type=Path)
    parser.add_argument("--old-stage58c-script", required=True, type=Path)
    parser.add_argument("--old-project", required=True, type=Path)
    parser.add_argument("--new-project", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_cfg = load_literal_dict(args.old_config)
    new_cfg = load_literal_dict(args.new_config)
    old_defaults = extract_arg_defaults(args.old_main)
    new_defaults = extract_arg_defaults(args.new_main)
    old_explicit = extract_explicit_args_from_stage58c_script(args.old_stage58c_script)
    rows = compare(old_cfg, new_cfg, old_defaults, new_defaults, old_explicit, args.old_project, args.new_project)

    (args.output_dir / "old_config.json").write_text(json.dumps(old_cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "new_config.json").write_text(json.dumps(new_cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "config_diff.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "status",
                "old_value",
                "new_value",
                "old_default",
                "new_default",
                "old_explicit_in_stage58c_script",
                "old_matches_old_default",
                "new_matches_new_default",
                "path_semantic_note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.output_dir / "config_diff.md")


if __name__ == "__main__":
    main()
