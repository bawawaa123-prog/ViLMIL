# Step32 Evidence Export Report

## Scope
- This step does not train the model.
- This step does not modify model computation logic.
- This step only exports the evidence chain of the current main model.

## Run Summary
- Results directory: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1`
- Checkpoint: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage23/rce_v4_csg_a01_rq16_5fold_e20_s1/s_0_checkpoint.pt`
- Fold: `0`
- Split: `test`
- Model type: `RCE_MIL_BiomedCLIP`
- Exported slides: `194`
- Correct slides: `176`
- Error slides: `18`

## Metrics
- test AUC: `0.9757`
- test ACC: `0.9072`
- test F1: `0.8988`
- Balanced ACC: `0.9077`
- PR-AUC: `0.9548`

## Global Top Low Concepts
- NonAdenocarcinoma | pulmonary hamartoma: mean_contribution=0.0093, count=60
- Adenocarcinoma | mixed acinar papillary architecture: mean_contribution=0.0083, count=5
- Adenocarcinoma | lepidic growth pattern: mean_contribution=0.0073, count=83
- Adenocarcinoma | mucinous features: mean_contribution=0.0071, count=122
- Adenocarcinoma | infiltrative glandular adenocarcinoma: mean_contribution=0.0068, count=11
- Adenocarcinoma | alveolar replacement by atypical glands: mean_contribution=0.0063, count=106
- Adenocarcinoma | cribriform glandular pattern: mean_contribution=0.0043, count=108
- Adenocarcinoma | desmoplastic stromal invasion: mean_contribution=0.0035, count=17

## Global Top High Concepts
- Adenocarcinoma | glandular lumen formation: mean_contribution=0.0230, count=5
- Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0228, count=121
- Adenocarcinoma | mucin-producing tumor cells: mean_contribution=0.0223, count=3
- Adenocarcinoma | intracytoplasmic mucin vacuoles: mean_contribution=0.0205, count=2
- Adenocarcinoma | atypical gland-forming tumor cells: mean_contribution=0.0205, count=119
- Adenocarcinoma | fused back-to-back glands: mean_contribution=0.0197, count=109
- Adenocarcinoma | hobnail adenocarcinoma cells: mean_contribution=0.0176, count=99
- Adenocarcinoma | floating micropapillary clusters: mean_contribution=0.0174, count=73

## Global Top CSG Low-High Pairs
- NonAdenocarcinoma | nonadeno_hamartoma_low -> nonadeno_inflammatory_high: mean_pair_score=0.0000, count=1
- NonAdenocarcinoma | nonadeno_hamartoma_low -> nonadeno_sarcomatoid_high: mean_pair_score=0.0000, count=34
- Adenocarcinoma | adeno_cribriform_low -> adeno_solid_high: mean_pair_score=0.0000, count=5
- NonAdenocarcinoma | nonadeno_hamartoma_low -> nonadeno_spindle_atypia_high: mean_pair_score=0.0000, count=24
- NonAdenocarcinoma | nonadeno_hamartoma_low -> nonadeno_dense_eosinophilic_cytoplasm_high: mean_pair_score=0.0000, count=32
- NonAdenocarcinoma | nonadeno_hamartoma_low -> nonadeno_nuclear_molding_high: mean_pair_score=0.0000, count=28
- Adenocarcinoma | adeno_mucinous -> adeno_solid_high: mean_pair_score=0.0000, count=23
- NonAdenocarcinoma | nonadeno_hamartoma_low -> nonadeno_intercellular_bridges_high: mean_pair_score=0.0000, count=9

## Correct vs Error Concept Difference
### Correct samples
- high | Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0228, count=116
- high | Adenocarcinoma | glandular lumen formation: mean_contribution=0.0223, count=4
- high | Adenocarcinoma | mucin-producing tumor cells: mean_contribution=0.0217, count=2
- high | Adenocarcinoma | atypical gland-forming tumor cells: mean_contribution=0.0205, count=113
- high | Adenocarcinoma | fused back-to-back glands: mean_contribution=0.0196, count=104
- high | Adenocarcinoma | intracytoplasmic mucin vacuoles: mean_contribution=0.0177, count=1

### Error samples
- high | Adenocarcinoma | glandular lumen formation: mean_contribution=0.0259, count=1
- high | Adenocarcinoma | mucin-producing tumor cells: mean_contribution=0.0235, count=1
- high | Adenocarcinoma | intracytoplasmic mucin vacuoles: mean_contribution=0.0233, count=1
- high | Adenocarcinoma | solid sheets with nuclear atypia: mean_contribution=0.0226, count=2
- high | Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0225, count=5
- high | NonAdenocarcinoma | fibrosis and chronic inflammation: mean_contribution=0.0215, count=6

## Evidence Source Overview
- concept evidence / low mean abs logit: `0.0392`
- concept evidence / high mean abs logit: `0.1165`
- visual residual mean abs logit: `8.1997`
- visual residual alpha: `0.0492`
- CSG mean abs logit: `0.0002`
- CSG alpha: `0.0969`

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
