# Step63 Next Step Plan

## Recommended workflow

- First submit the frozen Step63 checkpoint and create a tag such as `rce-step62-freeze-v1` or `rce-step63-freeze-v1`.
- Then open a new branch for Step64 instead of continuing to modify the current frozen version.
- Step64 is recommended to focus on Concept Reliability or Concept Selection Guided Residual-Constrained RCE.
- Do not continue making ad hoc changes directly on top of the frozen checkpoint branch.

## Rationale

- The current branch now has a clear frozen primary model, secondary variants, and a rejected direction.
- Starting Step64 from a fresh branch keeps the frozen result package reproducible and easy to reference in GitHub and in the paper draft.
