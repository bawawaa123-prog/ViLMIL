from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE39_DIR = "results_stage39/final_evidence_package"
DEFAULT_OUTPUT_DIR = "results_stage40/paper_ready_assets"


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step40 paper-ready assets from the Step39 final evidence package.")
    parser.add_argument("--stage39_dir", default=env_default("STAGE39_DIR", DEFAULT_STAGE39_DIR))
    parser.add_argument("--output_dir", default=env_default("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def read_json_required(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_required(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required text file: {path}")
    return path.read_text(encoding="utf-8")


def format_delta(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def load_stage39(stage39_dir: Path) -> dict[str, object]:
    return {
        "recommendation": read_json_required(stage39_dir / "stage39_final_model_recommendation.json"),
        "performance_summary": read_csv_required(stage39_dir / "stage39_final_performance_summary.csv"),
        "ablation_summary": read_csv_required(stage39_dir / "stage39_ablation_summary.csv"),
        "negative_ablation_summary": read_csv_required(stage39_dir / "stage39_negative_ablation_summary.csv"),
        "evidence_calibration_summary": read_csv_required(stage39_dir / "stage39_evidence_calibration_summary.csv"),
        "failure_comparison_summary": read_csv_required(stage39_dir / "stage39_failure_comparison_summary.csv"),
        "failure_cases": read_csv_required(stage39_dir / "stage39_fixed_regressed_persistent_cases.csv"),
        "top_concepts": read_csv_required(stage39_dir / "stage39_top_concepts_for_examples.csv"),
        "top_csg_pairs": read_csv_required(stage39_dir / "stage39_top_csg_pairs_for_examples.csv"),
        "paper_ready_summary": read_text_required(stage39_dir / "stage39_paper_ready_summary.md"),
        "innovation_points": read_text_required(stage39_dir / "stage39_final_innovation_points.md"),
        "next_steps": read_text_required(stage39_dir / "stage39_final_next_steps.md"),
    }


def build_method_overview_md(rec: dict[str, object]) -> str:
    lines = [
        "# Step40 Method Overview",
        "",
        "最终方法主线：",
        "",
        "`ViLa-MIL slide-level vision-language alignment`",
        "→ `Region-Concept Evidence Learning`",
        "→ `Cross-Scale Concept Evidence Reasoning`",
        "→ `Evidence Source Decomposition`",
        "→ `Evidence Path Interpretability`",
        "",
        "## 最终默认模型",
        f"- `{rec['recommended_default_model']}`",
        "",
        "## 方法主线说明",
        "- 基线仍然是 ViLa-MIL 的 slide-level vision-language alignment，但最终工作重点已经从单纯的 slide-level 对齐推进到 diagnostic evidence modeling。",
        "- `Region-Concept Evidence Learning` 对应区域证据 token 与病理概念的显式对齐，使模型不仅给出分类，还给出 region-to-concept 的证据路径。",
        "- `Cross-Scale Concept Evidence Reasoning` 保留低倍与高倍两条概念证据链，并通过 cross-scale concept reasoning 解释它们如何共同支持最终判断。",
        "- `Evidence Source Decomposition` 将 final logits 拆分为 low-scale concept、high-scale concept、CSG cross-scale pair、visual residual 等来源，支撑 failure diagnosis。",
        "- `Evidence Path Interpretability` 使 evidence export、failure typing、fixed/regressed/persistent case analysis 能成为最终论文叙事的一部分，而不是训练后的附加观察。",
        "",
        "## 结论边界",
        f"- `{rec['secondary_tradeoff_variant']}` 是 secondary trade-off variant，不是最终默认模型。",
        "- Step40 只整理最终论文/报告/答辩资产，不引入新训练、不修改模型主体逻辑、不新增新模块。",
        "",
    ]
    return "\n".join(lines)


def build_main_figure_mermaid_md() -> str:
    lines = [
        "# Step40 Main Figure Mermaid",
        "",
        "```mermaid",
        "flowchart LR",
        '    A[WSI patches] --> B[Low / High BiomedCLIP features]',
        '    B --> C[16 region evidence queries]',
        '    C --> D[Region-concept similarity]',
        '    D --> E[Low-scale concept evidence]',
        '    D --> F[High-scale concept evidence]',
        '    E --> G[CSG cross-scale concept reasoning]',
        '    F --> G',
        '    B --> H[Visual residual]',
        '    G --> I[Final logits]',
        '    H --> I',
        '    I --> J[Evidence export]',
        '    J --> K[Failure diagnosis]',
        '    L[Explored modules only in ablation\\nRegion graph / Concept graph / Scalar gate] -.not final model.-> I',
        "```",
        "",
        "说明：region graph / concept graph / scalar gate 仅作为 explored modules / negative ablation 出现，不属于最终默认主模型主线。",
        "",
    ]
    return "\n".join(lines)


def build_evidence_pipeline_mermaid_md() -> str:
    lines = [
        "# Step40 Evidence Pipeline Mermaid",
        "",
        "```mermaid",
        "flowchart TD",
        '    A[Slide] --> B[Top region evidence token]',
        '    B --> C[Top pathological concepts]',
        '    C --> D[Low-high concept pair]',
        '    D --> E[Final class]',
        '    E --> F[Failure type attribution]',
        '    C --> G[Evidence source attribution]',
        '    D --> G',
        '    G --> F',
        "```",
        "",
        "说明：这条路径强调最终工作不是只输出类别，而是输出 evidence path、failure type 与 evidence source attribution。",
        "",
    ]
    return "\n".join(lines)


def build_experiment_tables_plan_md(stage39_dir: Path) -> str:
    lines = [
        "# Step40 Experiment Tables Plan",
        "",
        "## Table 1：主性能对比表",
        "- 指标：`AUC`、`ACC`、`F1`、`Balanced ACC`、`PR-AUC`。",
        f"- 数据来源：`{relative(stage39_dir / 'stage39_final_performance_summary.csv')}`，并可结合 Stage24/28/31/35/37 的主干行。",
        "- 主要结论：`RCE-v4-CSG-a01-rq16 / DEG skeleton` 是当前最稳默认主模型。",
        "- 摆放建议：正文主表。",
        "",
        "## Table 2：RCE / CSG / rq 消融表",
        "- 指标：`delta_auc`、`delta_acc`、`delta_f1`、`delta_balanced_acc`、`delta_pr_auc`。",
        f"- 数据来源：`{relative(stage39_dir / 'stage39_ablation_summary.csv')}`。",
        "- 主要结论：`CSG a01 > a005`，`rq16 > rq8/rq32`，最终主干收敛到 `RCE-v4-CSG-a01-rq16`。",
        "- 摆放建议：正文。",
        "",
        "## Table 3：negative ablation 表",
        "- 指标：可不强调数值完整性，重点放 tested module、failure/trade-off reason、final decision。",
        f"- 数据来源：`{relative(stage39_dir / 'stage39_negative_ablation_summary.csv')}`。",
        "- 主要结论：region graph、concept graph、scalar gate 都有研究价值，但不进入最终默认主模型。",
        "- 摆放建议：正文或正文+附录扩展。",
        "",
        "## Table 4：skeleton vs lh_l001_m0 evidence calibration trade-off",
        "- 指标：`fixed`、`regressed`、`persistent`、`low_high_conflict`、`both_support_wrong`、`visual_residual_override`、`AUC delta`、`PR-AUC delta`。",
        f"- 数据来源：`{relative(stage39_dir / 'stage39_evidence_calibration_summary.csv')}` 与 `stage39_failure_comparison_summary.csv`。",
        "- 主要结论：low-high consistency 可减少 conflict，但不是最终默认主模型。",
        "- 摆放建议：正文。",
        "",
        "## Table 5：fixed / regressed / persistent case examples",
        "- 指标：案例组别、真实标签、两模型预测、primary failure type、margin、low-high joint state、selection note。",
        f"- 数据来源：`{relative(stage39_dir / 'stage39_fixed_regressed_persistent_cases.csv')}`，必要时配合 top concepts / top CSG pairs。",
        "- 主要结论：可以直观看到哪些错误被修复、哪些错误被引入、哪些错误持续存在。",
        "- 摆放建议：正文展示少量代表案例，完整表放附录。",
        "",
    ]
    return "\n".join(lines)


def build_ablation_table_paper_ready(ablation_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    module_map = {
        "CSG strength": "CSG strength",
        "Region query count": "Region evidence query count",
        "Spatial Region Graph": "Spatial Region Graph",
        "Concept Prompt Graph": "Concept Prompt Graph",
        "Scalar Visual Gate": "Scalar Visual Gate",
        "Low-High Consistency": "Low-High Consistency",
    }
    for _, row in ablation_df.iterrows():
        rows.append(
            {
                "module": module_map.get(row["topic"], row["topic"]),
                "comparison": row["comparison"],
                "preferred": row["preferred_variant"],
                "delta_auc": row["delta_test_auc"],
                "delta_acc": row["delta_test_acc"],
                "delta_f1": row["delta_test_f1"],
                "delta_balanced_acc": row["delta_balanced_acc"],
                "delta_pr_auc": row["delta_pr_auc"],
                "paper_conclusion": row["paper_ready_conclusion"],
            }
        )
    return pd.DataFrame(rows)


def build_negative_ablation_table_paper_ready(negative_df: pd.DataFrame) -> pd.DataFrame:
    decision_map = {
        "attention-centroid region graph": "not final default model",
        "concept prompt graph": "not final default model",
        "scalar visual gate": "not final default model",
        "low-high consistency": "secondary trade-off variant",
    }
    tested_map = {
        "attention-centroid region graph": "graph over semantic region tokens using attention-centroid style spatial assumptions",
        "concept prompt graph": "concept-to-concept smoothing over prompt-level feature graph",
        "scalar visual gate": "global scalar gate to suppress or rebalance visual residual",
        "low-high consistency": "consistency regularization between low-scale and high-scale evidence paths",
    }
    rows = []
    for _, row in negative_df.iterrows():
        rows.append(
            {
                "module": row["module"],
                "what_was_tested": tested_map.get(row["module"], row["negative_ablation_statement"]),
                "why_it_failed_or_traded_off": row["paper_ready_interpretation"],
                "paper_role": row["role"],
                "final_decision": decision_map.get(row["module"], row["role"]),
            }
        )
    return pd.DataFrame(rows)


def build_failure_case_table_paper_ready(failure_cases_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_group",
        "slide_id",
        "label_name",
        "skeleton_pred_name",
        "lh_pred_name",
        "skeleton_primary_failure_type",
        "lh_primary_failure_type",
        "skeleton_final_margin",
        "lh_final_margin",
        "skeleton_low_high_joint_state",
        "lh_low_high_joint_state",
        "selection_note",
    ]
    return failure_cases_df[columns].copy()


def build_defense_slide_outline_md(rec: dict[str, object]) -> str:
    slides = [
        ("1. 研究背景", "病理切片分类中的 slide-level vision-language MIL 已取得进展，但仍缺少更稳定的 evidence path 建模。", "问题定义图 + WSI 场景图", "强调任务不仅是分类，更是可诊断的证据建模。"),
        ("2. ViLa-MIL baseline 与不足", "回顾 ViLa-MIL 的 slide-level 对齐主线，并指出仅做 slide-level 对齐时的解释性和 failure diagnosis 局限。", "ViLa-MIL baseline 示意图", "不要否定 baseline，而是明确本工作的切入点。"),
        ("3. 研究目标：从 slide-level 对齐到 diagnostic evidence modeling", "提出从 slide-level alignment 走向 region-concept evidence、cross-scale reasoning、evidence decomposition。", "目标示意图", "把问题提升为诊断级证据建模。"),
        ("4. 方法总览", "介绍最终主线：ViLa-MIL alignment → Region-Concept Evidence Learning → Cross-Scale Concept Evidence Reasoning → Evidence Source Decomposition → Evidence Path Interpretability。", "Step40 main figure mermaid / 方法总图", f"明确默认模型是 `{rec['recommended_default_model']}`。"),
        ("5. Region-Concept Evidence Learning", "说明区域证据 token 如何与病理概念建立显式对齐。", "区域-概念证据路径图", "突出它与纯 bag-level 聚合的区别。"),
        ("6. Cross-Scale Concept Evidence Reasoning", "说明低倍/高倍概念证据如何共同参与 final decision。", "low/high evidence path 图", "强调 `rq16` 与 CSG 设计是最终被实验支持的部分。"),
        ("7. 实验设置", "数据划分、BiomedCLIP 特征、strict CV、评价指标和不继续训练新模块的收敛策略。", "实验设置表", "讲清楚结果来自完整流程，而不是后验挑选。"),
        ("8. 主性能结果", "给出最终默认模型及主性能对比。", "Table 1：主性能对比表", "结论收敛到 skeleton。"),
        ("9. 消融实验", "展示 CSG 强度、region query 数、以及最终主线的正向实验支持。", "Table 2：RCE/CSG/rq 消融表", "突出 `CSG a01 > a005`、`rq16 > rq8/rq32`。"),
        ("10. Negative ablation 与为什么不继续堆 graph/gate", "解释 region graph、concept graph、scalar gate 为什么保留为 negative ablation。", "Table 3：negative ablation 表", "强调这是有价值的实验发现，而不是失败就删除。"),
        ("11. Evidence decomposition 与 failure analysis", "展示 evidence source decomposition、failure typing 和 evidence export 能力。", "evidence pipeline mermaid + failure type 表", "把解释性分析作为核心产物而非附录点缀。"),
        ("12. Skeleton vs low-high consistency trade-off", "说明 `lh_l001_m0` 的 fixed/regressed/persistent case trade-off。", "Table 4：evidence calibration trade-off", "强调它是 secondary trade-off variant，不是最终默认模型。"),
        ("13. 可解释案例", "展示 fixed / regressed / persistent 三类案例及对应 concept / CSG pair。", "Table 5：案例表", "让观众看到 evidence path 的具体变化。"),
        ("14. 结论与创新点", "总结 4 个最终创新点和最终主模型收束。", "创新点总结页", "避免夸大所有模块都有效。"),
        ("15. 局限与未来工作", "说明不能声称什么，以及下一步 Prompt Reliability / Refined Prompt Pool 分支。", "claims vs limitations 表", "把未来工作聚焦到 prompt reliability，而不是继续堆 graph/gate。"),
    ]
    lines = ["# Step40 Defense Slide Outline", ""]
    for title, content, figure_table, emphasis in slides:
        lines.extend(
            [
                f"## {title}",
                f"- 主要内容：{content}",
                f"- 应放图/表：{figure_table}",
                f"- 讲述重点：{emphasis}",
                "",
            ]
        )
    return "\n".join(lines)


def build_paper_section_draft_md(rec: dict[str, object], calibration_row: pd.Series) -> str:
    lines = [
        "# Step40 Paper Section Draft",
        "",
        "## 摘要草稿",
        "本文围绕病理全切片分类中的 vision-language MIL 展开研究，重点不是继续堆叠新的图模块或门控模块，而是在 ViLa-MIL 基础上构建可诊断的证据建模框架。我们提出 Region-Concept Evidence Learning 与 Cross-Scale Concept Evidence Reasoning，使模型能够在 slide-level 分类之外显式组织区域证据、病理概念证据以及跨尺度概念关系。同时，我们构建了 Evidence Source Decomposition 与 failure diagnosis 分析链路，用于识别 low-scale/high-scale concept conflict、visual residual override 等关键错误模式。实验表明，`RCE-v4-CSG-a01-rq16 / DEG skeleton` 是当前最稳的默认模型；`CSG a01` 与 `rq16` 得到明确实验支持，而 Spatial Region Graph、Concept Prompt Graph 与 Scalar Visual Gate 未超过 skeleton。进一步的 evidence calibration 分析显示，low-high consistency regularization 能减少 low-high conflict，但会引入 visual residual override trade-off，因此更适合作为 secondary variant 而非最终默认模型。",
        "",
        "## 引言贡献点",
        "- 提出 Region-Concept Evidence Learning，使区域级视觉证据与病理概念证据形成显式对应关系。",
        "- 提出 Cross-Scale Concept Evidence Reasoning，把低倍/高倍概念证据纳入统一证据推理视角。",
        "- 构建 Evidence Source Decomposition and Failure Diagnosis，使错误分析可以定位到 concept conflict、visual override 等具体来源。",
        "- 给出 Evidence Calibration Analysis，证明 low-high consistency 更像 trade-off variant，而不是无条件更优的新主模型。",
        "",
        "## 方法章节结构",
        "1. ViLa-MIL baseline 与问题定义。",
        "2. Region-Concept Evidence Learning。",
        "3. Cross-Scale Concept Evidence Reasoning。",
        "4. Evidence Source Decomposition。",
        "5. Evidence Export / Failure Diagnosis。",
        "",
        "## 实验章节结构",
        "1. 数据集、strict CV 设置与评价指标。",
        "2. 主性能比较。",
        "3. RCE/CSG/rq 消融实验。",
        "4. Negative ablation：region graph、concept graph、scalar gate。",
        "5. Evidence calibration：skeleton vs low-high consistency。",
        "6. 可解释案例与 failure analysis。",
        "",
        "## 消融实验描述",
        "消融实验表明，最终主模型的有效性主要来自 `CSG a01` 与 `rq16` 的选择，而不是来自额外 graph/gate 模块。具体而言，Stage24 显示 `CSG a01 > CSG a005`，且 `rq16 > rq8/rq32`，因此最终 RCE 主干收敛为 `RCE-v4-CSG-a01-rq16`。相比之下，Stage28 的 Spatial Region Graph、Stage31 的 Concept Prompt Graph 以及 Stage35 的 Scalar Visual Gate 均未超过 `DEG skeleton`。这些结果说明，语义 region token 不能简单等价为真实 spatial region，普通 feature-level prompt smoothing 会削弱 evidence discrimination，而 visual residual 也不能被一个全局 scalar gate 稳定替代。",
        "",
        "## 解释性分析描述",
        f"在 Step38 的 evidence calibration 分析中，`lh_l001_m0` 相比 skeleton 带来了 `{int(calibration_row['fixed_cases'])}` 个 fixed cases、`{int(calibration_row['regressed_cases'])}` 个 regressed cases 和 `{int(calibration_row['persistent_errors'])}` 个 persistent errors；同时 `low_high_conflict` 从 `{int(calibration_row['low_high_conflict_skeleton'])}` 降至 `{int(calibration_row['low_high_conflict_lh'])}`，`both_support_wrong` 从 `{int(calibration_row['both_support_wrong_skeleton'])}` 降至 `{int(calibration_row['both_support_wrong_lh'])}`。然而，`visual_residual_override` 从 `{int(calibration_row['visual_residual_override_skeleton'])}` 上升到 `{int(calibration_row['visual_residual_override_lh'])}`，并伴随 `AUC={format_delta(calibration_row['auc_delta'])}`、`PR-AUC={format_delta(calibration_row['pr_auc_delta'])}` 的轻微下降。因此，我们将 low-high consistency 视为 evidence calibration trade-off，而非最终默认主模型。",
        "",
        "## 局限性",
        "- 本工作不能证明所有 graph/gate 模块都有效，反而显示其中若干设计更适合作为 negative ablation。",
        "- low-high consistency 不能被表述为全面超过 skeleton，因为它同时带来了 visual residual override 的代价。",
        "- 当前 example cases 的 top concept / top CSG pair 展示主要来自已有汇总字段，后续若需要更细粒度图示，仍可补充独立可视化素材。",
        "",
        "## 结论",
        f"综合所有阶段结果，当前最稳默认模型为 `{rec['recommended_default_model']}`。`{rec['secondary_tradeoff_variant']}` 作为 secondary trade-off variant 具有一定 evidence calibration 价值，但不替代默认主模型。最终论文叙事应聚焦于 region-concept evidence、cross-scale concept reasoning、evidence source decomposition 以及 failure diagnosis / calibration analysis，而不应再扩展为更多 graph/gate 模块的堆叠。",
        "",
    ]
    return "\n".join(lines)


def build_final_claims_and_limitations_md(rec: dict[str, object]) -> str:
    lines = [
        "# Step40 Final Claims and Limitations",
        "",
        "## 可以安全声称",
        "- `RCE-v4-CSG-a01-rq16` 是当前最稳主模型。",
        "- `CSG a01` 和 `rq16` 得到实验支持。",
        "- `region graph / concept graph / scalar gate` 是有价值的 negative ablation。",
        "- `low-high consistency` 可减少 `low-high conflict`，但不是最终默认主模型。",
        "- 模型支持 `evidence source decomposition` 和 `failure diagnosis`。",
        "",
        "## 不能安全声称",
        "- 不能说完整 `dual graph` 模块全面有效。",
        "- 不能说 `low-high consistency` 全面超过 `skeleton`。",
        "- 不能说 `scalar gate` 解决 `visual residual override`。",
        "- 不能说所有指标都超过 `PEPS / ViLa-MIL`，除非有完整对比表支持。",
        "- 不能把 explored modules 画成最终默认主模型的一部分。",
        "",
        "## 口径建议",
        f"- 最终默认模型只写 `{rec['recommended_default_model']}`。",
        f"- `{rec['secondary_tradeoff_variant']}` 的定位是 secondary trade-off variant。",
        "- 论文和答辩中应把 graph/gate 线作为 negative ablation 结果，而不是未完成的主模型路线。",
        "",
    ]
    return "\n".join(lines)


def build_next_research_branch_md() -> str:
    lines = [
        "# Step40 Next Research Branch",
        "",
        "## 推荐方向：Prompt Reliability / Refined Prompt Pool",
        "",
        "### 为什么它比继续 graph/gate 更合理",
        "- Step28/31/35 已经表明继续堆 region graph、concept graph、scalar gate 的边际回报很低。",
        "- 当前 persistent / regressed cases 中仍然可以看到 prompt confusion、wrong-class concept drift、low-high conflict 等问题，这更像 prompt reliability 问题，而不是图结构容量不足。",
        "- 因此下一轮创新更合理的方向，是提升 prompt pool 的判别性和稳定性，而不是继续增加结构模块。",
        "",
        "### 需要分析哪些 prompt 容易出现在错误样本中",
        "- 哪些高频概念在错误样本中持续支持 wrong class。",
        "- 哪些 low-scale 概念与 high-scale 概念在错误样本中形成 conflict。",
        "- 哪些 prompt 在 fixed/regressed/persistent 三类样本里表现出明显不同的可靠性。",
        "- 哪些 prompt 易与 visual residual override 同时出现，提示其语义边界不够稳定。",
        "",
        "### 如何构建 Concept-10 / Concept-8 refined prompt pool",
        "- 先基于 Step38/39 的 error cases，统计错误高频 prompt 与稳定支持正确类别的 prompt。",
        "- 删去在 wrong-class 样本中高频出现、且跨 fold 稳定性差的 prompt。",
        "- 对语义相近但区分度不足的 prompt 做合并或改写，保留更病理学明确的概念表述。",
        "- 优先构建 `Concept-10`，再进一步压缩成更保守的 `Concept-8`，比较其与当前 core12 prompt pool 的主性能和 failure profile。",
        "",
        "### 如何设计下一轮 Step41 / Step42",
        "- Step41：Prompt Reliability Audit。统计 prompt confusion、wrong-class support、error-prone prompt ranking、prompt stability across folds。",
        "- Step42：Refined Prompt Pool Evaluation。构建 `Concept-10 / Concept-8` refined prompt pool，并复用当前最终主线进行正式比较。",
        "",
        "### 边界说明",
        "- 这属于下一轮研究分支，不影响当前最终模型收束。",
        "- 当前默认模型仍然是 `RCE-v4-CSG-a01-rq16 / DEG skeleton`。",
        "",
    ]
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    stage39_dir = resolve_path(args.stage39_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_stage39(stage39_dir)
    rec = data["recommendation"]
    calibration_row = data["evidence_calibration_summary"].iloc[0]

    method_overview_md = build_method_overview_md(rec)
    main_figure_mermaid_md = build_main_figure_mermaid_md()
    evidence_pipeline_mermaid_md = build_evidence_pipeline_mermaid_md()
    experiment_tables_plan_md = build_experiment_tables_plan_md(stage39_dir)
    ablation_table = build_ablation_table_paper_ready(data["ablation_summary"])
    negative_ablation_table = build_negative_ablation_table_paper_ready(data["negative_ablation_summary"])
    failure_case_table = build_failure_case_table_paper_ready(data["failure_cases"])
    defense_slide_outline_md = build_defense_slide_outline_md(rec)
    paper_section_draft_md = build_paper_section_draft_md(rec, calibration_row)
    final_claims_and_limitations_md = build_final_claims_and_limitations_md(rec)
    next_research_branch_md = build_next_research_branch_md()

    write_text(output_dir / "stage40_method_overview.md", method_overview_md)
    write_text(output_dir / "stage40_main_figure_mermaid.md", main_figure_mermaid_md)
    write_text(output_dir / "stage40_evidence_pipeline_mermaid.md", evidence_pipeline_mermaid_md)
    write_text(output_dir / "stage40_experiment_tables_plan.md", experiment_tables_plan_md)
    ablation_table.to_csv(output_dir / "stage40_ablation_table_paper_ready.csv", index=False, encoding="utf-8")
    negative_ablation_table.to_csv(output_dir / "stage40_negative_ablation_table_paper_ready.csv", index=False, encoding="utf-8")
    failure_case_table.to_csv(output_dir / "stage40_failure_case_table_paper_ready.csv", index=False, encoding="utf-8")
    write_text(output_dir / "stage40_defense_slide_outline.md", defense_slide_outline_md)
    write_text(output_dir / "stage40_paper_section_draft.md", paper_section_draft_md)
    write_text(output_dir / "stage40_final_claims_and_limitations.md", final_claims_and_limitations_md)
    write_text(output_dir / "stage40_next_research_branch.md", next_research_branch_md)

    for name in [
        "stage40_method_overview.md",
        "stage40_main_figure_mermaid.md",
        "stage40_evidence_pipeline_mermaid.md",
        "stage40_experiment_tables_plan.md",
        "stage40_ablation_table_paper_ready.csv",
        "stage40_negative_ablation_table_paper_ready.csv",
        "stage40_failure_case_table_paper_ready.csv",
        "stage40_defense_slide_outline.md",
        "stage40_paper_section_draft.md",
        "stage40_final_claims_and_limitations.md",
        "stage40_next_research_branch.md",
    ]:
        print(f"[Done] Wrote: {output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
