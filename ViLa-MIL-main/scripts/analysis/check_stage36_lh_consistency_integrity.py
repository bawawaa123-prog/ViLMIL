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


def collect_add_argument_defaults(path: Path) -> dict[str, object]:
    tree = ast.parse(read_text(path), filename=str(path))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue

        flag = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flag = arg.value
                break
        if flag is None:
            continue

        default_value = None
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            try:
                default_value = ast.literal_eval(kw.value)
            except Exception:
                default_value = None
            break
        defaults[flag] = default_value
    return defaults


def require_substring(errors: list[str], text: str, needle: str, description: str) -> None:
    if needle not in text:
        errors.append(f"{description}: missing `{needle}`")


def main() -> None:
    errors: list[str] = []

    try:
        main_defaults = collect_add_argument_defaults(MAIN_PY)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    expected_main_defaults = {
        "--rce_use_low_high_consistency_loss": False,
        "--rce_lh_consistency_lambda": 0.0,
        "--rce_lh_consistency_margin": 0.0,
    }
    for flag, expected in expected_main_defaults.items():
        if flag not in main_defaults:
            errors.append(f"main.py: missing CLI arg `{flag}`")
        elif main_defaults[flag] != expected:
            errors.append(
                f"main.py: `{flag}` default expected {expected!r}, got {main_defaults[flag]!r}"
            )

    try:
        core_utils_text = read_text(CORE_UTILS_PY)
        model_text = read_text(MODEL_PY)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    for fragment in [
        "config.rce_use_low_high_consistency_loss = bool(getattr(args, 'rce_use_low_high_consistency_loss', False))",
        "config.rce_lh_consistency_lambda = float(getattr(args, 'rce_lh_consistency_lambda', 0.0))",
        "config.rce_lh_consistency_margin = float(getattr(args, 'rce_lh_consistency_margin', 0.0))",
    ]:
        require_substring(errors, core_utils_text, fragment, "utils/core_utils.py")

    for fragment in [
        "self.rce_use_low_high_consistency_loss = bool(",
        "getattr(config, \"rce_use_low_high_consistency_loss\", False)",
        "self.rce_lh_consistency_lambda = float(getattr(config, \"rce_lh_consistency_lambda\", 0.0))",
        "self.rce_lh_consistency_margin = float(getattr(config, \"rce_lh_consistency_margin\", 0.0))",
        "self.last_low_scale_logits = None",
        "self.last_high_scale_logits = None",
        "self.last_low_true_wrong_margin = None",
        "self.last_high_true_wrong_margin = None",
        "self.last_lh_margin_gap = None",
        "self.last_lh_consistency_loss = None",
        "self.last_total_loss = None",
        "def _true_vs_wrong_margin(logits, label):",
        "true_logit = logits.gather(1, label.view(-1, 1)).squeeze(1)",
        "wrong_logits = logits.masked_fill(",
        "max_wrong_logit = wrong_logits.max(dim=1).values",
        "return true_logit - max_wrong_logit",
        "self.last_low_scale_logits = logits_low.detach().cpu()",
        "self.last_high_scale_logits = logits_high.detach().cpu()",
        "if self.scale_mode == \"dual\" and self.rce_use_low_high_consistency_loss:",
        "low_margin = self._true_vs_wrong_margin(logits_low, label)",
        "high_margin = self._true_vs_wrong_margin(logits_high, label)",
        "low_loss = F.relu(margin - low_margin)",
        "high_loss = F.relu(margin - high_margin)",
        "lh_consistency_loss = (low_loss + high_loss).mean()",
        "loss = ce_loss + self.rce_lh_consistency_lambda * lh_consistency_loss",
        "lh_consistency_loss = ce_loss.new_zeros(())",
        "loss = ce_loss",
        "self.last_low_true_wrong_margin = low_margin.detach().cpu() if low_margin is not None else None",
        "self.last_high_true_wrong_margin = high_margin.detach().cpu() if high_margin is not None else None",
        "self.last_lh_margin_gap = lh_margin_gap.detach().cpu() if lh_margin_gap is not None else None",
        "self.last_lh_consistency_loss = lh_consistency_loss.detach().cpu()",
        "self.last_total_loss = loss.detach().cpu()",
    ]:
        require_substring(errors, model_text, fragment, "models/model_DEG_MIL_BiomedCLIP.py")

    if errors:
        for error in errors:
            print(f"[Error] {error}")
        sys.exit(1)

    print("[OK] Stage36 low-high consistency integrity check passed.")


if __name__ == "__main__":
    main()
