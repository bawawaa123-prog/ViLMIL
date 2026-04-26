# DCP-ViLa-MIL Stage 2 Implementation Plan for Codex

## Goal
Implement Stage 2 only: replace static class prompts with a diagnostic concept prompt pool and average-ensemble the text embeddings. Do not add dynamic prompt gate yet.

## Input artifacts
- `private_lung_concept_prompt_pool_stage2.json`
- label mapping: `NonAdenocarcinoma = 0`, `Adenocarcinoma = 1`
- each prompt item contains: `class_id`, `class_name`, `scale`, `concept_id`, `concept_zh`, `concept_en`, `prompt`

## Required behavior
1. Add a command-line argument:
   - `--concept_prompt_path`
   - `--use_concept_prompt_pool`
   - `--prompt_ensemble_mode embedding_mean`
2. Load the prompt JSON.
3. Group prompts by `(scale, class_id)`.
4. Encode all prompts with the existing BiomedCLIP text encoder/tokenizer.
5. For each `(scale, class_id)`, do:
   - L2 normalize each prompt embedding.
   - Average embeddings.
   - L2 normalize the averaged class embedding.
6. Return:
   - `low_text_features`: shape `[num_classes, embed_dim]`
   - `high_text_features`: shape `[num_classes, embed_dim]`
7. Replace only the static prompt text features in BiomedCLIP-ViLa-MIL.
8. Keep all other model modules, optimizer, loss, train loop, and data loading unchanged.

## Suggested new file
Create `utils/prompt_utils.py` with a function:

```python
def build_concept_text_features(prompt_json_path, text_encoder, tokenizer, device, num_classes, dtype=None):
    ...
    return low_text_features, high_text_features
```

## Integration point
In the BiomedCLIP-ViLa-MIL model initialization or before training:
- If `args.use_concept_prompt_pool` is True:
  - call `build_concept_text_features(...)`
  - register output tensors as buffers:
    - `self.concept_text_low`
    - `self.concept_text_high`
- In `forward`, use these buffers instead of the original static prompt embeddings.

## Important constraints
- Do not use `冰冻报告` or `石蜡报告` as test-time input.
- Do not use `影像数据` in this stage.
- The prompt pool is class-level and can be used on public datasets if a dataset-specific prompt JSON is provided.
- Ensure class id order matches the dataset label mapping.
- If the current code uses label order `[Adenocarcinoma, NonAdenocarcinoma]`, either remap labels or sort prompt features to match the current label order.

## Experiments to run
1. Current Stage 1 baseline:
   - BiomedCLIP-ViLa-MIL + static prompt
2. Stage 2:
   - BiomedCLIP-ViLa-MIL + concept prompt pool + embedding mean
3. Optional ablation:
   - logit mean instead of embedding mean
   - English concept prompts vs Chinese raw prompts

## Expected model comparison
Report AUC, F1, ACC, Balanced Accuracy, Sensitivity, Specificity, and PR-AUC on the same splits.

## Leakage note
The uploaded reports were used to construct an exploratory concept pool. For final paper experiments, either:
1. use this prompt pool as a fixed expert-designed concept pool without referencing test reports during split creation, or
2. regenerate prompt pools from train reports only for each fold.
