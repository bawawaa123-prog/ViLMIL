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
        "--rce_prarc_gate_version",
        "--rce_prarc_gate_gain",
        "--rce_prarc_gate_last_weight_init",
        "--rce_prarc_use_conflict_prior",
        "--rce_prarc_conflict_prior_strength",
        "--rce_prarc_use_gate_entropy_reg",
        "--rce_prarc_gate_entropy_lambda",
        "--rce_prarc_use_gate_variance_reg",
        "--rce_prarc_gate_variance_lambda",
    ]:
        require_substring(failures, main_text, needle, "main.py CLI arg")

    for needle in [
        "config.rce_prarc_gate_version",
        "config.rce_prarc_gate_gain",
        "config.rce_prarc_gate_last_weight_init",
        "config.rce_prarc_use_conflict_prior",
        "config.rce_prarc_conflict_prior_strength",
        "config.rce_prarc_use_gate_entropy_reg",
        "config.rce_prarc_gate_entropy_lambda",
        "config.rce_prarc_use_gate_variance_reg",
        "config.rce_prarc_gate_variance_lambda",
    ]:
        require_substring(failures, core_text, needle, "utils/core_utils.py PRARC-v2 config pass-through")

    for needle, label in [
        ('self.rce_prarc_gate_version = str(getattr(config, "rce_prarc_gate_version", "v1")).strip().lower()', "PRARC-v2 version config"),
        ('self.rce_prarc_gate_gain = float(getattr(config, "rce_prarc_gate_gain", 1.0))', "PRARC-v2 gain config"),
        ('self.rce_prarc_gate_last_weight_init = float(getattr(config, "rce_prarc_gate_last_weight_init", 0.01))', "PRARC-v2 last-weight init config"),
        ("self.last_prarc_gate_variance = None", "PRARC variance debug buffer"),
        ("self.last_prarc_gate_reg_loss = None", "PRARC regularization debug buffer"),
        ("def _compute_prarc_conflict_prior(", "PRARC conflict prior helper"),
        ("def _compute_prarc_gate_regularization(", "PRARC gate regularization helper"),
        ('if self.rce_prarc_gate_version == "v2":', "PRARC-v2 conditional path"),
        ("prarc_gate = torch.sigmoid(float(self.rce_prarc_gate_gain) * gate_logits)", "PRARC-v2 gain-scaled gate"),
        ("nn.init.normal_(", "PRARC-v2 nonzero last-layer init"),
    ]:
        require_substring(failures, model_text, needle, label)

    if failures:
        print("[FAIL] Step48 PRARC-v2 integrity check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("[PASS] Step48 PRARC-v2 integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
