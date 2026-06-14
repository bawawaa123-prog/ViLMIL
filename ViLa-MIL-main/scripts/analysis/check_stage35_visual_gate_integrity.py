from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"
CORE_UTILS_PY = ROOT / "utils" / "core_utils.py"
MODEL_PY = ROOT / "models" / "model_DEG_MIL_BiomedCLIP.py"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def collect_add_argument_flags(path: Path) -> set[str]:
    tree = ast.parse(read_text(path), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flags.add(arg.value)
    return flags


def require_substring(errors: list[str], text: str, needle: str, description: str) -> None:
    if needle not in text:
        errors.append(f"{description}: missing `{needle}`")


def require_visual_gate_flow(errors: list[str], model_text: str) -> None:
    required_fragments = [
        "alpha = torch.sigmoid(self.rce_visual_residual_alpha)",
        "visual_residual_contribution = alpha * visual_logits",
        "if self.rce_use_visual_evidence_gate:",
        "gate = torch.sigmoid(self.rce_visual_evidence_gate)",
        "visual_gated_contribution = gate * visual_residual_contribution",
        "final_logits = final_logits + visual_gated_contribution",
        "visual_gated_contribution = visual_residual_contribution",
    ]
    for fragment in required_fragments:
        require_substring(errors, model_text, fragment, "forward visual gate flow")


def main() -> None:
    errors: list[str] = []

    try:
        main_flags = collect_add_argument_flags(MAIN_PY)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    for flag in ["--rce_use_visual_evidence_gate", "--rce_visual_gate_init"]:
        if flag not in main_flags:
            errors.append(f"main.py: missing CLI arg `{flag}`")

    try:
        core_utils_text = read_text(CORE_UTILS_PY)
        model_text = read_text(MODEL_PY)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    require_substring(
        errors,
        core_utils_text,
        "config.rce_use_visual_evidence_gate = bool(getattr(args, 'rce_use_visual_evidence_gate', False))",
        "utils/core_utils.py",
    )
    require_substring(
        errors,
        core_utils_text,
        "config.rce_visual_gate_init = float(getattr(args, 'rce_visual_gate_init', 1.0))",
        "utils/core_utils.py",
    )

    for fragment in [
        "self.rce_use_visual_evidence_gate = bool(getattr(config, \"rce_use_visual_evidence_gate\", False))",
        "self.rce_visual_evidence_gate = nn.Parameter(",
        "self.last_visual_evidence_gate = None",
        "self.last_visual_residual_contribution = None",
        "self.last_visual_gated_contribution = None",
        "self.last_visual_evidence_gate = gate.detach().cpu()",
        "self.last_visual_residual_contribution = visual_residual_contribution.detach().cpu()",
        "self.last_visual_gated_contribution = visual_gated_contribution.detach().cpu()",
    ]:
        require_substring(errors, model_text, fragment, "models/model_DEG_MIL_BiomedCLIP.py")

    require_visual_gate_flow(errors, model_text)

    if errors:
        for error in errors:
            print(f"[Error] {error}")
        sys.exit(1)

    print("[OK] Stage35 visual gate integrity check passed.")


if __name__ == "__main__":
    main()
