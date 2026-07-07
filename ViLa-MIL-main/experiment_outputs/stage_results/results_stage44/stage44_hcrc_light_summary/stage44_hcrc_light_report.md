# Stage44 HCRC-Light 5-fold Report

## Purpose
- Formally evaluate whether Step43 HCRC-Light improves over the current `RCE-v4-CSG-a01-rq16 / DEG skeleton` baseline at the 5-fold performance level.
- Step44 is performance-first. HCRC debug/evidence export is not forced here if the stored outputs are insufficient; that work belongs to Step45.

## Step43 Inheritance
- Step43 summary source: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage43/stage43_hcrc_light_smoke_summary/stage43_hcrc_smoke_report.md`
- Step42b recommended spatial settings were kept: proposal_radius=4096, nms_radius=512, num_anchors=16, num_high_children=16, coord_mode=top_left, scale_ratio=1.0, child_strategy=bbox_containment.

## Variants
- Requested variants: `hcrc_a002_b8, hcrc_a005_b8, hcrc_a01_b8`

## 5-fold Metrics
### hcrc_a002_b8
- status: `ok`
- checkpoints: `5`
- traceback: `False`
- NaN/Inf in log: `False`
- missing folds: `none`
- alpha_init: `0.02`
- bbox_expand: `8.0`
- test_auc: `0.9613 +/- 0.0159`
- test_acc: `0.9246 +/- 0.0178`
- test_f1: `0.9167 +/- 0.0197`
- balanced_acc: `0.9194 +/- 0.0191`
- sensitivity: `0.9032 +/- 0.0267`
- specificity: `0.9357 +/- 0.0182`
- pr_auc: `0.9209 +/- 0.0401`
- val_auc: `0.9560 +/- 0.0171`

### hcrc_a005_b8
- status: `ok`
- checkpoints: `5`
- traceback: `False`
- NaN/Inf in log: `False`
- missing folds: `none`
- alpha_init: `0.05`
- bbox_expand: `8.0`
- test_auc: `0.9614 +/- 0.0203`
- test_acc: `0.9008 +/- 0.0453`
- test_f1: `0.8856 +/- 0.0601`
- balanced_acc: `0.8846 +/- 0.0689`
- sensitivity: `0.8336 +/- 0.1458`
- specificity: `0.9356 +/- 0.0220`
- pr_auc: `0.9181 +/- 0.0477`
- val_auc: `0.9612 +/- 0.0069`

### hcrc_a01_b8
- status: `ok`
- checkpoints: `5`
- traceback: `False`
- NaN/Inf in log: `False`
- missing folds: `none`
- alpha_init: `0.1`
- bbox_expand: `8.0`
- test_auc: `0.9683 +/- 0.0164`
- test_acc: `0.9194 +/- 0.0165`
- test_f1: `0.9110 +/- 0.0178`
- balanced_acc: `0.9127 +/- 0.0143`
- sensitivity: `0.8912 +/- 0.0177`
- specificity: `0.9341 +/- 0.0245`
- pr_auc: `0.9351 +/- 0.0313`
- val_auc: `0.9648 +/- 0.0155`

## Baseline Comparison
- baseline test_auc: `0.9702`
- baseline test_acc: `0.9225`
- baseline test_f1: `0.9145`
- baseline balanced_acc: `0.9171`
- baseline sensitivity: `0.9002`
- baseline specificity: `0.9341`
- baseline pr_auc: `0.9444`

- most stable alpha among valid runs: `0.02` from `hcrc_a002_b8`
## Stability
- Step44 checks logs, fold summaries, checkpoint counts, missing folds, and NaN/Inf tokens.
- Performance-level validation is the primary goal in Step44; HCRC debug/evidence details should be exported in Step45 if needed.

## Sensitivity / Specificity Trade-off
## Recommendation
- hcrc_a002_b8: sens=0.9032, spec=0.9357 | hcrc_a005_b8: sens=0.8336, spec=0.9356 | hcrc_a01_b8: sens=0.8912, spec=0.9341

- recommended variant: `hcrc_a002_b8`
- recommendation reason: `No variant beat baseline by the Step44 rules; selected strongest internal HCRC variant for reporting.`
- enter Step45 HCRC Evidence Export and Failure Comparison: `False`
- Step45 rationale: `All valid HCRC variants remain below baseline by the Step44 decision rules.`

## Risk
- `proposal_radius=4096` can widen the effective low-anchor support and shift anchor coordinates.
- `bbox_expand=8` can introduce broad high-scale evidence regions.
- Even if Step44 performance is acceptable, Step45 must verify whether the evidence is clinically and spatially reliable.
