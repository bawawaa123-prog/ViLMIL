# Stage54 Case Selection

## Selection Summary

- Full correct high-confidence cases are selected from Stage23 matched predictions.
- Full wrong failure cases are selected from Stage23 matched predictions.
- CSG-benefit cases are defined as `full correct` and `wo_csg wrong`.
- Same-prediction confidence-shift cases are defined from matched full vs `wo_csg` prediction confidence differences.

## Selected Cases

| selection_type | selection_rank | slide_id | fold | true_label_name | full_pred_name | full_confidence | wo_csg_pred_name | wo_csg_confidence | confidence_delta_full_minus_wo_csg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_correct_high_confidence | 1 | 25034940B12 | 0 | NonAdenocarcinoma | NonAdenocarcinoma | 0.9999998807907104 | NonAdenocarcinoma | 0.9999853372573853 | 1.4543533325195312e-05 |
| full_correct_high_confidence | 2 | 25045829B2 | 0 | NonAdenocarcinoma | NonAdenocarcinoma | 0.9999998807907104 | NonAdenocarcinoma | 0.9999934434890747 | 6.4373016357421875e-06 |
| full_correct_high_confidence | 3 | 25057854B3 | 0 | NonAdenocarcinoma | NonAdenocarcinoma | 0.9999997615814209 | NonAdenocarcinoma | 0.9999967813491821 | 2.9802322387695312e-06 |
| full_wrong_failure | 1 | 2485275-B2 | 0 | Adenocarcinoma | NonAdenocarcinoma | 0.9998760223388672 | NonAdenocarcinoma | 0.9993382096290588 | 0.0005378127098083496 |
| full_wrong_failure | 2 | 25001929B3 | 0 | Adenocarcinoma | NonAdenocarcinoma | 0.9998406171798706 | NonAdenocarcinoma | 0.9998358488082886 | 4.76837158203125e-06 |
| full_wrong_failure | 3 | 25015087B2 | 0 | Adenocarcinoma | NonAdenocarcinoma | 0.9996041655540466 | NonAdenocarcinoma | 0.9986690282821655 | 0.0009351372718811035 |
| csg_benefit_full_correct_wo_csg_wrong | 1 | 25015422B | 0 | Adenocarcinoma | Adenocarcinoma | 0.900865375995636 | NonAdenocarcinoma | 0.7535103559494019 | 0.14735502004623413 |
| csg_benefit_full_correct_wo_csg_wrong | 2 | 25063676B | 1 | NonAdenocarcinoma | NonAdenocarcinoma | 0.9935683608055115 | Adenocarcinoma | 0.5331817865371704 | 0.46038657426834106 |
| csg_benefit_full_correct_wo_csg_wrong | 3 | 25029906B2 | 1 | NonAdenocarcinoma | NonAdenocarcinoma | 0.9510307908058167 | Adenocarcinoma | 0.5446416139602661 | 0.40638917684555054 |
| same_pred_confidence_shift | 1 | 2487558-B2 | 0 | Adenocarcinoma | NonAdenocarcinoma | 0.961268424987793 | NonAdenocarcinoma | 0.5813652873039246 | 0.3799031376838684 |
| same_pred_confidence_shift | 2 | 25024855B4 | 0 | NonAdenocarcinoma | NonAdenocarcinoma | 0.9961656332015991 | NonAdenocarcinoma | 0.7772706747055054 | 0.21889495849609375 |
| same_pred_confidence_shift | 3 | 25032179B3 | 0 | NonAdenocarcinoma | NonAdenocarcinoma | 0.9993712306022644 | NonAdenocarcinoma | 0.7945031523704529 | 0.20486807823181152 |

## Counts

- Full correct cases available: `893`
- Full wrong cases available: `75`
- CSG-benefit cases available: `14`
- Same-prediction confidence-shift candidates available: `62`

## Notes

- Matched case selection is available directly from `split_*_results.pkl` for Stage23 full and Step52 `wo_csg`.
- Evidence-level matched comparison still depends on exporting `wo_csg` evidence files under `results_stage54_rce_evidence_interpretability/wo_csg/`.
