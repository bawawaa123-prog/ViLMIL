# Stage43 HCRC-Light Smoke Summary

- Run directory: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage43/stage43_hcrc_light_smoke_a005_s1`
- Smoke completed: `True`
- Checkpoint generated: `True`
- HCRC enabled in config: `True`
- HCRC alpha init: `0.05`
- Recommended Step42b params: proposal_radius=4096.0, nms_radius=512.0, bbox_expand=8.0, num_anchors=16, num_high_children=16, child_strategy=bbox_containment, prompt_scale=high

## Fold0 Metrics
- test_auc: `0.9744318181818182`
- test_acc: `0.8865979381443299`
- test_f1: `0.8810214094558431`
- val_auc: `0.964251893939394`
- balanced_acc: `0.9103929924242424`
- sensitivity: `0.9848484848484848`
- specificity: `0.8359375`
- pr_auc: `0.9495034889377436`

## HCRC Probe
- Probe attempted: `True`
- Probe success: `True`
- HCRC enabled at forward: `True`
- hcrc_logits non-zero: `True`
- hcrc alpha (post-sigmoid): `0.05036712437868118`
- empty anchor ratio mean: `0.0625`
- child used count mean: `13.3125`
- anchor valid count: `16`
- skip reason: `None`
- Probe failure reason: `None`

## Stability
- Log has Traceback: `False`
- Log has NaN/Inf token: `False`
- Probe tensors have NaN/Inf: `False`
- Checkpoint params all finite: `True`

## Recommendation
- Enter Step44 HCRC-Light 5-fold: `True`
