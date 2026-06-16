from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require_substring(failures: list[str], text: str, needle: str, label: str) -> None:
    if needle not in text:
        failures.append(f"Missing {label}: {needle}")


def main() -> int:
    failures: list[str] = []

    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    core_text = (ROOT / "utils" / "core_utils.py").read_text(encoding="utf-8")
    model_text = (ROOT / "models" / "model_DEG_MIL_BiomedCLIP.py").read_text(encoding="utf-8")

    for needle in [
        "--rce_use_prarc_gate",
        "--rce_prarc_gate_hidden_dim",
        "--rce_prarc_gate_init",
        "--rce_prarc_gate_dropout",
        "--rce_prarc_gate_feature_set",
        "--rce_prarc_detach_features",
        "--rce_prarc_include_optional_features",
        "--rce_prarc_feature_clip",
        "--rce_prarc_export_debug",
    ]:
        require_substring(failures, main_text, needle, "main.py CLI arg")

    for needle in [
        "config.rce_use_prarc_gate",
        "config.rce_prarc_gate_hidden_dim",
        "config.rce_prarc_gate_init",
        "config.rce_prarc_gate_dropout",
        "config.rce_prarc_gate_feature_set",
        "config.rce_prarc_detach_features",
        "config.rce_prarc_include_optional_features",
        "config.rce_prarc_feature_clip",
        "config.rce_prarc_export_debug",
    ]:
        require_substring(failures, core_text, needle, "utils/core_utils.py PRARC config pass-through")

    for needle, label in [
        ("self.rce_use_prarc_gate = bool(getattr(config, \"rce_use_prarc_gate\", False))", "PRARC default-off config"),
        ("self.prarc_gate_mlp = nn.Sequential(", "PRARC gate MLP"),
        ("def _compute_prarc_gate_features(", "PRARC feature helper"),
        ("def _apply_prarc_gate(", "PRARC gate application helper"),
        ("self.last_prarc_gate = None", "PRARC debug buffer"),
        ("self.last_prarc_gate_features = None", "PRARC feature debug buffer"),
        ("prarc_gate * visual_residual_contribution", "PRARC gated visual contribution"),
    ]:
        require_substring(failures, model_text, needle, label)

    if failures:
        print("[FAIL] Step46 PRARC integrity check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[PASS] Step46 PRARC integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
