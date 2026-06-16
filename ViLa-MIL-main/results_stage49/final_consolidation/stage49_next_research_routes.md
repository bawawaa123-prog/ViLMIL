# Stage49 Next Research Routes

## Route A: Conservative
- Stop structure stacking and keep the current baseline as the final model.
- Prepare the full ablation package for paper writing, with HCRC and PRARC reported as negative ablations.
- Spend effort on clearer evidence visualizations, failure narratives, and reviewer-facing ablation framing instead of more training.

## Route B: Moderate
- Add an evidence-margin auxiliary loss that encourages concept-consistent decisions without directly gating the visual residual branch.
- Treat reduced visual override as the primary optimization target rather than raw branch fusion complexity.
- Re-run only targeted experiments that directly test whether residual override frequency drops.

## Route C: Ambitious
- Redesign the residual branch so visual residual predicts only the residual error on top of concept logits instead of competing as a broad override signal.
- Explore train-fold reliability distillation or uncertainty-aware residual supervision.
- Avoid any test-derived prompt reliability signal in the learning path.

