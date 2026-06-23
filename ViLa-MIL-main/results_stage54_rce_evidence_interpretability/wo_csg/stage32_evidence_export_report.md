# Step32 Evidence Export Report

## Scope
- This step does not train the model.
- This step does not modify model computation logic.
- This step only exports the evidence chain of the current main model.

## Run Summary
- Results directory: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1`
- Checkpoint: `/home/ljh/ViLMIL/ViLa-MIL-main/results_stage52_rce_core_ablation/wo_csg_5fold_e20_s1/s_0_checkpoint.pt`
- Fold: `0`
- Split: `test`
- Model type: `RCE_MIL_BiomedCLIP`
- Exported slides: `194`
- Correct slides: `180`
- Error slides: `14`

## Metrics
- test AUC: `0.9718`
- test ACC: `0.9278`
- test F1: `0.9202`
- Balanced ACC: `0.9233`
- PR-AUC: `0.9504`

## Global Top Low Concepts
- NonAdenocarcinoma | pulmonary hamartoma: mean_contribution=0.0183, count=62
- Adenocarcinoma | lepidic growth pattern: mean_contribution=0.0123, count=94
- Adenocarcinoma | mucinous features: mean_contribution=0.0122, count=117
- Adenocarcinoma | alveolar replacement by atypical glands: mean_contribution=0.0112, count=118
- Adenocarcinoma | cribriform glandular pattern: mean_contribution=0.0101, count=102
- Adenocarcinoma | infiltrative glandular adenocarcinoma: mean_contribution=0.0086, count=99
- NonAdenocarcinoma | fibroinflammatory scar-like lesion: mean_contribution=0.0035, count=59
- Adenocarcinoma | multifocal irregular glandular proliferation: mean_contribution=0.0024, count=39

## Global Top High Concepts
- NonAdenocarcinoma | intercellular bridges: mean_contribution=0.0798, count=68
- NonAdenocarcinoma | keratinization: mean_contribution=0.0523, count=68
- NonAdenocarcinoma | dense eosinophilic cytoplasm: mean_contribution=0.0388, count=65
- Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0264, count=126
- Adenocarcinoma | atypical gland-forming tumor cells: mean_contribution=0.0228, count=113
- Adenocarcinoma | floating micropapillary clusters: mean_contribution=0.0222, count=121
- Adenocarcinoma | prominent nucleoli in gland-forming tumor cells: mean_contribution=0.0218, count=16
- Adenocarcinoma | solid sheets with nuclear atypia: mean_contribution=0.0214, count=8

## Global Top CSG Low-High Pairs
- none

## Correct vs Error Concept Difference
### Correct samples
- high | NonAdenocarcinoma | intercellular bridges: mean_contribution=0.0774, count=60
- high | NonAdenocarcinoma | keratinization: mean_contribution=0.0509, count=60
- high | NonAdenocarcinoma | dense eosinophilic cytoplasm: mean_contribution=0.0376, count=57
- high | Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0262, count=120
- high | Adenocarcinoma | atypical gland-forming tumor cells: mean_contribution=0.0228, count=108
- high | Adenocarcinoma | floating micropapillary clusters: mean_contribution=0.0222, count=116

### Error samples
- high | NonAdenocarcinoma | intercellular bridges: mean_contribution=0.0981, count=8
- high | NonAdenocarcinoma | keratinization: mean_contribution=0.0621, count=8
- high | NonAdenocarcinoma | dense eosinophilic cytoplasm: mean_contribution=0.0472, count=8
- high | Adenocarcinoma | tumor cells lining fibrovascular cores: mean_contribution=0.0297, count=6
- high | Adenocarcinoma | solid sheets with nuclear atypia: mean_contribution=0.0230, count=1
- high | Adenocarcinoma | floating micropapillary clusters: mean_contribution=0.0226, count=5

## Evidence Source Overview
- concept evidence / low mean abs logit: `0.0918`
- concept evidence / high mean abs logit: `0.1493`
- visual residual mean abs logit: `6.5932`
- visual residual alpha: `0.0488`
- CSG mean abs logit: `N/A`
- CSG alpha: `N/A`

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
