# Step32 Evidence Export Report

## Scope
- This step does not train the model.
- This step does not modify model computation logic.
- This step only exports the evidence chain of the current main model.

## Run Summary
- Results directory: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1`
- Checkpoint: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage37/lh_consistency_lh_l001_m0_5fold_e20_s1/s_0_checkpoint.pt`
- Fold: `0`
- Split: `test`
- Model type: `DEG_MIL_BiomedCLIP`
- Exported slides: `194`
- Correct slides: `180`
- Error slides: `14`

## Metrics
- test AUC: `0.9718`
- test ACC: `0.9278`
- test F1: `0.9196`
- Balanced ACC: `0.9196`
- PR-AUC: `0.9514`

## Global Top Low Concepts
- Adenocarcinoma | alveolar replacement by atypical glands: mean_contribution=0.0050, count=116
- Adenocarcinoma | lepidic growth pattern: mean_contribution=0.0048, count=67
- Adenocarcinoma | cribriform glandular pattern: mean_contribution=0.0043, count=111
- Adenocarcinoma | micropapillary pattern: mean_contribution=0.0042, count=77
- Adenocarcinoma | mucinous features: mean_contribution=0.0040, count=97
- Adenocarcinoma | multifocal irregular glandular proliferation: mean_contribution=0.0011, count=75
- NonAdenocarcinoma | pulmonary hamartoma: mean_contribution=-0.0000, count=43
- Adenocarcinoma | infiltrative glandular adenocarcinoma: mean_contribution=-0.0001, count=32

## Global Top High Concepts
- Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0262, count=128
- Adenocarcinoma | floating micropapillary clusters: mean_contribution=0.0232, count=128
- Adenocarcinoma | atypical gland-forming tumor cells: mean_contribution=0.0225, count=128
- NonAdenocarcinoma | nuclear molding: mean_contribution=0.0225, count=8
- Adenocarcinoma | solid sheets with nuclear atypia: mean_contribution=0.0202, count=9
- Adenocarcinoma | hobnail adenocarcinoma cells: mean_contribution=0.0181, count=123
- NonAdenocarcinoma | salt-and-pepper chromatin: mean_contribution=0.0181, count=9
- Adenocarcinoma | fused back-to-back glands: mean_contribution=0.0173, count=96

## Global Top CSG Low-High Pairs
- Adenocarcinoma | adeno_lepidic -> adeno_mucinous_high: mean_pair_score=0.0000, count=81
- Adenocarcinoma | adeno_micropapillary -> adeno_columnar_cells_high: mean_pair_score=0.0000, count=26
- NonAdenocarcinoma | nonadeno_solid_nests_low -> nonadeno_nuclear_molding_high: mean_pair_score=0.0000, count=22
- Adenocarcinoma | adeno_multifocal_glandular_low -> adeno_micropapillary_high: mean_pair_score=0.0000, count=3
- Adenocarcinoma | adeno_lepidic -> adeno_columnar_cells_high: mean_pair_score=0.0000, count=2
- Adenocarcinoma | adeno_mucinous -> adeno_papillary_high: mean_pair_score=0.0000, count=3
- Adenocarcinoma | adeno_solid -> adeno_lepidic_high: mean_pair_score=0.0000, count=1
- NonAdenocarcinoma | nonadeno_solid_nests_low -> nonadeno_chondroid_matrix_high: mean_pair_score=0.0000, count=28

## Correct vs Error Concept Difference
### Correct samples
- high | Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0264, count=121
- high | Adenocarcinoma | floating micropapillary clusters: mean_contribution=0.0232, count=121
- high | Adenocarcinoma | atypical gland-forming tumor cells: mean_contribution=0.0226, count=121
- high | NonAdenocarcinoma | nuclear molding: mean_contribution=0.0224, count=7
- high | Adenocarcinoma | solid sheets with nuclear atypia: mean_contribution=0.0201, count=6
- high | Adenocarcinoma | hobnail adenocarcinoma cells: mean_contribution=0.0182, count=119

### Error samples
- high | NonAdenocarcinoma | nuclear molding: mean_contribution=0.0230, count=1
- high | Adenocarcinoma | floating micropapillary clusters: mean_contribution=0.0226, count=7
- high | Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0222, count=7
- high | Adenocarcinoma | atypical gland-forming tumor cells: mean_contribution=0.0217, count=7
- high | Adenocarcinoma | solid sheets with nuclear atypia: mean_contribution=0.0205, count=3
- high | NonAdenocarcinoma | salt-and-pepper chromatin: mean_contribution=0.0198, count=1

## Evidence Source Overview
- concept evidence / low mean abs logit: `0.0585`
- concept evidence / high mean abs logit: `0.1113`
- visual residual mean abs logit: `7.2363`
- visual residual alpha: `0.0491`
- CSG mean abs logit: `0.0001`
- CSG alpha: `0.0950`

## Outputs
- `stage32_slide_evidence_summary.csv`
- `stage32_top_concepts_long.csv`
- `stage32_top_csg_pairs.csv`
- `stage32_error_cases.csv`
- `stage32_manifest.json`

## Warnings
- none

## Next Suggested Step
- Step33: Evidence failure / conflict analysis
- Step34: Evidence-level gated residual
- Step35: Evidence consistency / margin loss
