from __future__ import annotations

import csv
import json
import subprocess
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results_stage63_freeze_step57_to_step62"

STEP62_DIR = ROOT / "results_stage62_final_innovation_consolidation"
STEP57C_DIR = ROOT / "results_stage57C_rce_v2_copy_reproduction"
STEP58C_DIR = ROOT / "results_stage58C_residual_constrained_configD_5fold"
STEP59C_DIR = ROOT / "results_stage59C_dynamic_csg_configA_5fold"
STEP60D_DIR = ROOT / "results_stage60D_ccra_configC_formal"
STEP61D_DIR = ROOT / "results_stage61D_l2h_configG_5fold"

DISPLAY_ROOT_CANDIDATES = [
    Path("/xiangmu/ViLMIL/ViLa-MIL-main"),
    ROOT,
]

PROTECTED_FILES = [
    "main.py",
    "utils/core_utils.py",
    "models/model_RCE_MIL_BiomedCLIP.py",
    "models/model_RCE_MIL_BiomedCLIP_v2.py",
]

RESULT_ROWS_FALLBACK = [
    {
        "stage_id": "Stage57C",
        "display_name": "Stage57C RCE-v2 baseline",
        "source_dir": "results_stage57C_rce_v2_copy_reproduction",
        "model_role": "baseline_reference",
        "AUC": 0.970213,
        "ACC": 0.922509,
        "F1": 0.914456,
        "BACC": 0.917143,
        "PR_AUC": 0.944399,
        "selected_primary": False,
        "selected_secondary": False,
        "rejected": False,
        "notes": "Reproduced formal 5-fold baseline reference.",
    },
    {
        "stage_id": "Step58C",
        "display_name": "Step58C Residual-Constrained RCE",
        "source_dir": "results_stage58C_residual_constrained_configD_5fold",
        "model_role": "final_primary_model",
        "AUC": 0.964842,
        "ACC": 0.927691,
        "F1": 0.919625,
        "BACC": 0.919700,
        "PR_AUC": 0.931551,
        "selected_primary": True,
        "selected_secondary": False,
        "rejected": False,
        "notes": "Selected primary model.",
    },
    {
        "stage_id": "Step59C",
        "display_name": "Step59C Dynamic CSG",
        "source_dir": "results_stage59C_dynamic_csg_configA_5fold",
        "model_role": "secondary_variant",
        "AUC": 0.969323,
        "ACC": 0.922504,
        "F1": 0.913797,
        "BACC": 0.913596,
        "PR_AUC": 0.941877,
        "selected_primary": False,
        "selected_secondary": True,
        "rejected": False,
        "notes": "Secondary variant only.",
    },
    {
        "stage_id": "Step60D",
        "display_name": "Step60D CCRA config C",
        "source_dir": "results_stage60D_ccra_configC_formal",
        "model_role": "secondary_variant",
        "AUC": 0.968601,
        "ACC": 0.922493,
        "F1": 0.915115,
        "BACC": 0.920087,
        "PR_AUC": 0.937097,
        "selected_primary": False,
        "selected_secondary": True,
        "rejected": False,
        "notes": "Secondary variant only.",
    },
    {
        "stage_id": "Step61D",
        "display_name": "Step61D L2H Retrieval",
        "source_dir": "results_stage61D_l2h_configG_5fold",
        "model_role": "not_selected",
        "AUC": 0.967921,
        "ACC": 0.921479,
        "F1": 0.911256,
        "BACC": 0.904859,
        "PR_AUC": 0.938282,
        "selected_primary": False,
        "selected_secondary": False,
        "rejected": True,
        "notes": "Rejected for final main model selection.",
    },
]

INTERPRETABILITY_FALLBACK = {
    "Stage57C": {
        "visual_ratio_mean": 0.719580,
        "concept_ratio_mean": 0.280420,
        "full_minus_concept_acc": 0.000000,
        "full_minus_concept_auc": 0.015152,
    },
    "Step58C": {
        "visual_ratio_mean": 0.320575,
        "concept_ratio_mean": 0.679425,
        "full_minus_concept_acc": 0.010325,
        "full_minus_concept_auc": 0.003213,
    },
    "Step59C": {
        "visual_ratio_mean": 0.337585,
        "concept_ratio_mean": 0.662415,
        "full_minus_concept_acc": 0.001031,
        "full_minus_concept_auc": 0.003895,
    },
    "Step60D": {
        "visual_ratio_mean": 0.357331,
        "concept_ratio_mean": 0.642669,
        "full_minus_concept_acc": -0.003093,
        "full_minus_concept_auc": -0.002586,
    },
    "Step61D": {
        "visual_ratio_mean": 0.281393,
        "concept_ratio_mean": 0.718607,
        "full_minus_concept_acc": 0.002067,
        "full_minus_concept_auc": 0.002182,
    },
}

DECISION_FALLBACK = {
    "final_primary_model": "Step58C Residual-Constrained RCE config D",
    "secondary_variants": [
        "Step59C Dynamic CSG config A",
        "Step60D CCRA config C",
    ],
    "rejected_variants": [
        "Step61D L2H Retrieval config G",
    ],
}

KEY_SCRIPTS = [
    "scripts/analysis/build_stage57C_rce_v2_reproduction_summary.py",
    "scripts/analysis/build_stage58C_residual_constrained_5fold_summary.py",
    "scripts/analysis/build_stage59C_dynamic_csg_5fold_summary.py",
    "scripts/analysis/build_stage60D_ccra_configC_formal_summary.py",
    "scripts/analysis/build_stage61D_l2h_5fold_summary.py",
    "scripts/analysis/build_stage62_final_innovation_consolidation.py",
    "scripts/analysis/build_stage63_freeze_step57_to_step62.py",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def display_root() -> Path:
    for candidate in DISPLAY_ROOT_CANDIDATES:
        if (candidate / "main.py").is_file() and (candidate / "scripts").is_dir():
            return candidate
    return ROOT


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_cmd(args: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def git_output(args: list[str]) -> str:
    code, stdout, _stderr = run_cmd(args)
    if code != 0:
        return ""
    return stdout.rstrip()


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_rows_optional(path: Path) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_metric(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "NA"
    return f"{number:.6f}"


def fmt_delta(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "NA"
    return f"{number:+.6f}"


def load_stage62_results() -> list[dict[str, Any]]:
    path = STEP62_DIR / "stage62_main_results_table.csv"
    rows = read_csv_rows_optional(path)
    if not rows:
        return [dict(item) for item in RESULT_ROWS_FALLBACK]

    mapping = OrderedDict(
        [
            ("stage57C_rce_v2_baseline", "Stage57C"),
            ("step58C_residual_constrained_primary", "Step58C"),
            ("step59C_dynamic_csg_variant", "Step59C"),
            ("step60D_ccra_variant", "Step60D"),
            ("step61D_l2h_not_selected", "Step61D"),
        ]
    )
    display_names = {
        "Stage57C": "Stage57C RCE-v2 baseline",
        "Step58C": "Step58C Residual-Constrained RCE",
        "Step59C": "Step59C Dynamic CSG",
        "Step60D": "Step60D CCRA config C",
        "Step61D": "Step61D L2H Retrieval",
    }
    parsed: list[dict[str, Any]] = []
    for row in rows:
        stage_id = mapping.get(row.get("model_id", ""))
        if stage_id is None:
            continue
        parsed.append(
            {
                "stage_id": stage_id,
                "display_name": display_names[stage_id],
                "source_dir": row.get("source_dir") or "",
                "model_role": row.get("model_role") or "",
                "AUC": as_float(row.get("AUC")),
                "ACC": as_float(row.get("ACC")),
                "F1": as_float(row.get("F1")),
                "BACC": as_float(row.get("BACC")),
                "PR_AUC": as_float(row.get("PR_AUC")),
                "selected_primary": str(row.get("selected_as_primary")).lower() == "true",
                "selected_secondary": str(row.get("selected_as_variant")).lower() == "true",
                "rejected": stage_id == "Step61D",
                "notes": row.get("not_selected_reason") or "",
                "delta_auc_vs_stage57C": as_float(row.get("delta_auc_vs_stage57C")),
                "delta_acc_vs_stage57C": as_float(row.get("delta_acc_vs_stage57C")),
                "delta_f1_vs_stage57C": as_float(row.get("delta_f1_vs_stage57C")),
                "delta_bacc_vs_stage57C": as_float(row.get("delta_bacc_vs_stage57C")),
                "delta_pr_auc_vs_stage57C": as_float(row.get("delta_pr_auc_vs_stage57C")),
            }
        )
    if len(parsed) == len(RESULT_ROWS_FALLBACK):
        return parsed
    return [dict(item) for item in RESULT_ROWS_FALLBACK]


def load_interpretability() -> dict[str, dict[str, float | None]]:
    path = STEP62_DIR / "stage62_branch_contribution_summary.csv"
    rows = read_csv_rows_optional(path)
    if not rows:
        return dict(INTERPRETABILITY_FALLBACK)

    mapping = {
        "stage57C_rce_v2_baseline": "Stage57C",
        "step58C_residual_constrained_primary": "Step58C",
        "step59C_dynamic_csg_variant": "Step59C",
        "step60D_ccra_variant": "Step60D",
        "step61D_l2h_not_selected": "Step61D",
    }
    parsed: dict[str, dict[str, float | None]] = {}
    for row in rows:
        stage_id = mapping.get(row.get("model_id", ""))
        if stage_id is None:
            continue
        parsed[stage_id] = {
            "visual_ratio_mean": as_float(row.get("visual_ratio_mean")),
            "concept_ratio_mean": as_float(row.get("concept_ratio_mean")),
            "full_minus_concept_acc": as_float(row.get("full_minus_concept_acc")),
            "full_minus_concept_auc": as_float(row.get("full_minus_concept_auc")),
        }
    for key, fallback in INTERPRETABILITY_FALLBACK.items():
        parsed.setdefault(key, fallback)
    return parsed


def load_decision_payload() -> dict[str, Any]:
    decision = read_json_optional(STEP62_DIR / "stage62_final_model_decision.json")
    if not decision:
        return dict(DECISION_FALLBACK)

    return {
        "final_primary_model": "Step58C Residual-Constrained RCE config D",
        "secondary_variants": [
            "Step59C Dynamic CSG config A",
            "Step60D CCRA config C",
        ],
        "rejected_variants": [
            "Step61D L2H Retrieval config G",
        ],
        "final_primary_reason": decision.get("final_primary_reason", []),
        "paper_main_claim": decision.get("paper_main_claim", ""),
        "paper_secondary_claims": decision.get("paper_secondary_claims", []),
    }


def build_csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            OrderedDict(
                [
                    ("stage_id", item["stage_id"]),
                    ("model_name", item["display_name"]),
                    ("source_dir", item["source_dir"]),
                    ("model_role", item["model_role"]),
                    ("AUC", fmt_metric(item["AUC"])),
                    ("ACC", fmt_metric(item["ACC"])),
                    ("F1", fmt_metric(item["F1"])),
                    ("BACC", fmt_metric(item["BACC"])),
                    ("PR_AUC", fmt_metric(item["PR_AUC"])),
                    ("selected_primary", str(bool(item["selected_primary"]))),
                    ("selected_secondary", str(bool(item["selected_secondary"]))),
                    ("rejected", str(bool(item["rejected"]))),
                    ("notes", item["notes"]),
                ]
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        write_text(path, "")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_freeze_summary(
    results: list[dict[str, Any]],
    interpretability: dict[str, dict[str, float | None]],
    decision_payload: dict[str, Any],
) -> str:
    step57 = next(item for item in results if item["stage_id"] == "Stage57C")
    step58 = next(item for item in results if item["stage_id"] == "Step58C")
    step59 = next(item for item in results if item["stage_id"] == "Step59C")
    step60 = next(item for item in results if item["stage_id"] == "Step60D")
    step61 = next(item for item in results if item["stage_id"] == "Step61D")
    step58_interp = interpretability["Step58C"]

    return "\n".join(
        [
            "# Step63 Freeze Summary",
            "",
            "## Frozen conclusion",
            "",
            f"- final primary model = {decision_payload['final_primary_model']}",
            f"- secondary variants = {', '.join(decision_payload['secondary_variants'])}",
            f"- rejected variant = {', '.join(decision_payload['rejected_variants'])}",
            "- next step should start from a new innovation branch, not overwrite the current frozen version",
            "",
            "## Final reading of Step57-Step62",
            "",
            (
                f"- Stage57C remains the reproduced RCE-v2 baseline reference with "
                f"AUC={fmt_metric(step57['AUC'])}, ACC={fmt_metric(step57['ACC'])}, "
                f"F1={fmt_metric(step57['F1'])}, BACC={fmt_metric(step57['BACC'])}, "
                f"PR-AUC={fmt_metric(step57['PR_AUC'])}."
            ),
            (
                f"- Step58C is frozen as the selected main model because it improves hard classification "
                f"metrics over Stage57C: ACC {fmt_metric(step57['ACC'])} -> {fmt_metric(step58['ACC'])}, "
                f"F1 {fmt_metric(step57['F1'])} -> {fmt_metric(step58['F1'])}, "
                f"BACC {fmt_metric(step57['BACC'])} -> {fmt_metric(step58['BACC'])}, while accepting a "
                f"trade-off in AUC {fmt_metric(step57['AUC'])} -> {fmt_metric(step58['AUC'])} and "
                f"PR-AUC {fmt_metric(step57['PR_AUC'])} -> {fmt_metric(step58['PR_AUC'])}."
            ),
            (
                f"- Step58C also gives the key evidence shift: visual_ratio_mean "
                f"{fmt_metric(interpretability['Stage57C']['visual_ratio_mean'])} -> "
                f"{fmt_metric(step58_interp['visual_ratio_mean'])}, concept_ratio_mean "
                f"{fmt_metric(interpretability['Stage57C']['concept_ratio_mean'])} -> "
                f"{fmt_metric(step58_interp['concept_ratio_mean'])}."
            ),
            (
                f"- Step59C stays as a secondary variant: it preserves strong AUC={fmt_metric(step59['AUC'])} "
                f"and PR-AUC={fmt_metric(step59['PR_AUC'])}, but ACC/F1/BACC are weaker than Step58C."
            ),
            (
                f"- Step60D stays as a secondary variant: it is performance-competitive with "
                f"AUC={fmt_metric(step60['AUC'])}, ACC={fmt_metric(step60['ACC'])}, F1={fmt_metric(step60['F1'])}, "
                f"BACC={fmt_metric(step60['BACC'])}, but it is not the chosen primary checkpoint."
            ),
            (
                f"- Step61D is rejected as a final main model: AUC={fmt_metric(step61['AUC'])}, "
                f"ACC={fmt_metric(step61['ACC'])}, F1={fmt_metric(step61['F1'])}, "
                f"BACC={fmt_metric(step61['BACC'])}, PR-AUC={fmt_metric(step61['PR_AUC'])}."
            ),
            "",
            "## Freeze policy",
            "",
            "- Step63 performs no training.",
            "- Step63 does not modify any model forward logic.",
            "- Step63 is a checkpoint-preparation step for frozen results, paper assets, and GitHub bookkeeping only.",
        ]
    ) + "\n"


def build_latex_table(results: list[dict[str, Any]]) -> str:
    def latex_metric(stage_id: str, metric: str) -> str:
        item = next(row for row in results if row["stage_id"] == stage_id)
        value = fmt_metric(item[metric])
        if stage_id == "Step58C" and metric in {"ACC", "F1", "BACC"}:
            return f"\\textbf{{{value}}}"
        return value

    lines = [
        "% Step63 frozen main results table",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Frozen Step57--Step61 formal 5-fold results. Step58C is the selected primary model.}",
        "\\label{tab:stage63-main-results}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Model & AUC & ACC & F1 & BACC & PR-AUC \\\\",
        "\\hline",
        (
            "Stage57C RCE-v2 baseline & "
            f"{latex_metric('Stage57C', 'AUC')} & {latex_metric('Stage57C', 'ACC')} & "
            f"{latex_metric('Stage57C', 'F1')} & {latex_metric('Stage57C', 'BACC')} & "
            f"{latex_metric('Stage57C', 'PR_AUC')} \\\\"
        ),
        (
            "Step58C Residual-Constrained RCE (selected primary) & "
            f"{latex_metric('Step58C', 'AUC')} & {latex_metric('Step58C', 'ACC')} & "
            f"{latex_metric('Step58C', 'F1')} & {latex_metric('Step58C', 'BACC')} & "
            f"{latex_metric('Step58C', 'PR_AUC')} \\\\"
        ),
        (
            "Step59C Dynamic CSG & "
            f"{latex_metric('Step59C', 'AUC')} & {latex_metric('Step59C', 'ACC')} & "
            f"{latex_metric('Step59C', 'F1')} & {latex_metric('Step59C', 'BACC')} & "
            f"{latex_metric('Step59C', 'PR_AUC')} \\\\"
        ),
        (
            "Step60D CCRA config C & "
            f"{latex_metric('Step60D', 'AUC')} & {latex_metric('Step60D', 'ACC')} & "
            f"{latex_metric('Step60D', 'F1')} & {latex_metric('Step60D', 'BACC')} & "
            f"{latex_metric('Step60D', 'PR_AUC')} \\\\"
        ),
        (
            "Step61D L2H Retrieval & "
            f"{latex_metric('Step61D', 'AUC')} & {latex_metric('Step61D', 'ACC')} & "
            f"{latex_metric('Step61D', 'F1')} & {latex_metric('Step61D', 'BACC')} & "
            f"{latex_metric('Step61D', 'PR_AUC')} \\\\"
        ),
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_model_decision(results: list[dict[str, Any]], interpretability: dict[str, dict[str, float | None]]) -> str:
    stage57 = next(item for item in results if item["stage_id"] == "Stage57C")
    step58 = next(item for item in results if item["stage_id"] == "Step58C")
    step59 = next(item for item in results if item["stage_id"] == "Step59C")
    step60 = next(item for item in results if item["stage_id"] == "Step60D")
    step61 = next(item for item in results if item["stage_id"] == "Step61D")
    base_interp = interpretability["Stage57C"]
    step58_interp = interpretability["Step58C"]

    return "\n".join(
        [
            "# Step63 Model Decision",
            "",
            "## Selected primary model",
            "",
            "- Step58C Residual-Constrained RCE config D is the final primary model.",
            "",
            "## Why Step58C is selected",
            "",
            (
                f"- Compared with Stage57C, Step58C improves ACC ({fmt_metric(stage57['ACC'])} -> {fmt_metric(step58['ACC'])}), "
                f"F1 ({fmt_metric(stage57['F1'])} -> {fmt_metric(step58['F1'])}), and "
                f"BACC ({fmt_metric(stage57['BACC'])} -> {fmt_metric(step58['BACC'])})."
            ),
            (
                f"- visual_ratio_mean drops from about {fmt_metric(base_interp['visual_ratio_mean'])} "
                f"to {fmt_metric(step58_interp['visual_ratio_mean'])}."
            ),
            (
                f"- concept_ratio_mean rises from about {fmt_metric(base_interp['concept_ratio_mean'])} "
                f"to {fmt_metric(step58_interp['concept_ratio_mean'])}."
            ),
            (
                f"- The full branch remains stronger than concept_only: full_minus_concept_acc="
                f"{fmt_delta(step58_interp['full_minus_concept_acc'])}, full_minus_concept_auc="
                f"{fmt_delta(step58_interp['full_minus_concept_auc'])}."
            ),
            (
                f"- The AUC / PR-AUC trade-off is explicit rather than hidden: AUC {fmt_metric(stage57['AUC'])} -> "
                f"{fmt_metric(step58['AUC'])}, PR-AUC {fmt_metric(stage57['PR_AUC'])} -> {fmt_metric(step58['PR_AUC'])}."
            ),
            "",
            "## Why the others are not the primary checkpoint",
            "",
            (
                f"- Step59C Dynamic CSG stays as a secondary variant because its AUC={fmt_metric(step59['AUC'])} "
                f"and PR-AUC={fmt_metric(step59['PR_AUC'])} are good, but ACC/F1/BACC are weaker than Step58C."
            ),
            (
                f"- Step60D CCRA config C stays as a secondary variant because it is balanced and interpretable, "
                f"but it is not the final selected primary checkpoint."
            ),
            (
                f"- Step61D L2H Retrieval does not enter the final primary model because ACC/F1/BACC remain weaker "
                f"than Step58C, especially BACC={fmt_metric(step61['BACC'])}."
            ),
        ]
    ) + "\n"


def build_claims_doc() -> str:
    return "\n".join(
        [
            "# Step63 Claims To Make And Avoid",
            "",
            "## Claims to make",
            "",
            "1. Step58C Residual-Constrained RCE is the selected primary model for the frozen Step57-Step62 checkpoint.",
            "2. Step58C improves ACC / F1 / BACC over the reproduced Stage57C RCE-v2 baseline.",
            "3. Step58C shifts evidence allocation away from visual residual and toward concept evidence.",
            "4. Dynamic CSG and CCRA should be reported as secondary variants rather than the final main checkpoint.",
            "5. L2H retrieval can be described as an explored but not selected direction.",
            "6. The frozen Step63 package is intended to preserve the current result state before any new innovation branch begins.",
            "",
            "## Claims to avoid",
            "",
            "1. 不要说 Dynamic CSG 是主要分类贡献来源。",
            "2. 不要说 CCRA 明确超过所有 baseline。",
            "3. 不要说 L2H retrieval 是最终有效主模型。",
            "4. 不要说所有创新模块已经组合成统一最优模型。",
            "5. 不要只强调单一指标而忽略 trade-off。",
            "6. 不要说 Step58C 全面超过 Stage57C，因为 AUC / PR-AUC 有下降。",
            "7. 不要把当前冻结版本表述成仍适合继续直接叠加未验证改动的工作分支。",
        ]
    ) + "\n"


def build_next_step_plan() -> str:
    return "\n".join(
        [
            "# Step63 Next Step Plan",
            "",
            "## Recommended workflow",
            "",
            "- First submit the frozen Step63 checkpoint and create a tag such as `rce-step62-freeze-v1` or `rce-step63-freeze-v1`.",
            "- Then open a new branch for Step64 instead of continuing to modify the current frozen version.",
            "- Step64 is recommended to focus on Concept Reliability or Concept Selection Guided Residual-Constrained RCE.",
            "- Do not continue making ad hoc changes directly on top of the frozen checkpoint branch.",
            "",
            "## Rationale",
            "",
            "- The current branch now has a clear frozen primary model, secondary variants, and a rejected direction.",
            "- Starting Step64 from a fresh branch keeps the frozen result package reproducible and easy to reference in GitHub and in the paper draft.",
        ]
    ) + "\n"


def build_run_commands(display_root_path: Path) -> str:
    display_root_str = str(display_root_path)
    return "\n".join(
        [
            f"cd {display_root_str}",
            "python -m py_compile scripts/analysis/build_stage63_freeze_step57_to_step62.py",
            "python scripts/analysis/build_stage63_freeze_step57_to_step62.py",
            "",
            "# Suggested next git commands",
            "git add scripts/analysis/build_stage63_freeze_step57_to_step62.py results_stage63_freeze_step57_to_step62",
            "git commit -m \"Step63: freeze Step57-Step62 RCE innovation results\"",
            "git tag -a rce-step63-freeze-v1 -m \"Freeze Step57-Step62 RCE innovation results\"",
            "git push origin dev-rce-innovation",
            "git push origin rce-step63-freeze-v1",
        ]
    ) + "\n"


def protected_file_status() -> list[dict[str, Any]]:
    diff_output = git_output(["git", "diff", "--name-only", "--", *PROTECTED_FILES])
    changed = {line.strip() for line in diff_output.splitlines() if line.strip()}
    rows = []
    for path in PROTECTED_FILES:
        rows.append(
            {
                "path": path,
                "modified_in_worktree": path in changed,
            }
        )
    return rows


def build_manifest(
    branch: str,
    commit_hash: str,
    generated_at: str,
    display_root_path: Path,
    protected_status: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "current_git_branch": branch,
        "current_git_commit_hash": commit_hash,
        "generated_timestamp": generated_at,
        "project_root": str(display_root_path),
        "source_result_directories": [
            "results_stage62_final_innovation_consolidation",
            "results_stage57C_rce_v2_copy_reproduction",
            "results_stage58C_residual_constrained_configD_5fold",
            "results_stage59C_dynamic_csg_configA_5fold",
            "results_stage60D_ccra_configC_formal",
            "results_stage61D_l2h_configG_5fold",
        ],
        "final_primary_model": "Step58C Residual-Constrained RCE config D",
        "secondary_variants": [
            "Step59C Dynamic CSG config A",
            "Step60D CCRA config C",
        ],
        "rejected_variants": [
            "Step61D L2H Retrieval config G",
        ],
        "key_scripts": KEY_SCRIPTS,
        "no_training_run": True,
        "protected_files_not_modified": protected_status,
    }


def build_git_status_summary(status_short: str) -> str:
    lines = [line for line in status_short.splitlines() if line.strip()]
    if not lines:
        return "clean"
    return f"{len(lines)} changed path(s)"


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    display_root_path = display_root()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    branch = git_output(["git", "branch", "--show-current"])
    commit_hash = git_output(["git", "rev-parse", "HEAD"])
    git_log = git_output(["git", "log", "--oneline", "-10"])

    results = load_stage62_results()
    interpretability = load_interpretability()
    decision_payload = load_decision_payload()
    csv_rows = build_csv_rows(results)
    protected_status = protected_file_status()

    generated_files: list[str] = []

    summary_path = OUTPUT_DIR / "stage63_freeze_summary.md"
    write_text(summary_path, build_freeze_summary(results, interpretability, decision_payload))
    generated_files.append(summary_path.name)

    csv_path = OUTPUT_DIR / "stage63_main_results_table.csv"
    write_csv(csv_path, csv_rows)
    generated_files.append(csv_path.name)

    tex_path = OUTPUT_DIR / "stage63_main_results_table.tex"
    write_text(tex_path, build_latex_table(results))
    generated_files.append(tex_path.name)

    decision_path = OUTPUT_DIR / "stage63_model_decision.md"
    write_text(decision_path, build_model_decision(results, interpretability))
    generated_files.append(decision_path.name)

    claims_path = OUTPUT_DIR / "stage63_claims_to_make_and_avoid.md"
    write_text(claims_path, build_claims_doc())
    generated_files.append(claims_path.name)

    manifest_path = OUTPUT_DIR / "stage63_reproducibility_manifest.json"
    write_json(
        manifest_path,
        build_manifest(branch, commit_hash, generated_at, display_root_path, protected_status),
    )
    generated_files.append(manifest_path.name)

    git_status_short = git_output(["git", "status", "--short"])
    git_status_full = git_output(["git", "status"])

    git_status_path = OUTPUT_DIR / "stage63_git_status.txt"
    write_text(git_status_path, (git_status_full or "git status unavailable") + "\n")
    generated_files.append(git_status_path.name)

    git_log_path = OUTPUT_DIR / "stage63_git_log.txt"
    write_text(git_log_path, (git_log or "git log unavailable") + "\n")
    generated_files.append(git_log_path.name)

    run_commands_path = OUTPUT_DIR / "stage63_run_commands.txt"
    run_commands_text = build_run_commands(display_root_path)
    write_text(run_commands_path, run_commands_text)
    generated_files.append(run_commands_path.name)

    next_step_path = OUTPUT_DIR / "stage63_next_step_plan.md"
    write_text(next_step_path, build_next_step_plan())
    generated_files.append(next_step_path.name)

    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated files:")
    for name in generated_files:
        print(f"- {name}")
    print(f"Git status summary: {build_git_status_summary(git_status_short)}")
    print("Suggested git commands:")
    for line in run_commands_text.splitlines():
        if line.startswith("git "):
            print(line)


if __name__ == "__main__":
    main()
