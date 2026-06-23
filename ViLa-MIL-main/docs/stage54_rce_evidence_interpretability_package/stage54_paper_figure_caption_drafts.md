# Stage54 Paper Figure Caption Drafts

## 1. RCE Pipeline

Overview of the proposed RCE framework built on dual-scale vision-language alignment. The model aggregates low- and high-magnification region evidence into concept-aware slide-level predictions, while concept prior, visual residual evidence, and cross-scale reasoning refine the final decision.

## 2. Region-Concept Evidence Heatmap

Region-concept evidence heatmap for a representative correctly classified slide. Rows denote low- and high-scale concept evidence channels, and columns denote the top concepts supporting the final prediction. The visualization is intended as an interpretability aid rather than a direct localization benchmark.

## 3. Low-High CSG Concept Interaction

Top low-to-high concept interaction pairs under the RCE cross-scale graph for a representative case. The figure illustrates how CSG links concept evidence across magnifications and refines the final ranking of supporting evidence.

## 4. Full vs w/o CSG Evidence Ranking

Matched-case comparison of concept evidence ranking between the full RCE model and its `w/o CSG` counterpart. The comparison highlights that CSG mainly affects evidence ordering and confidence structure, even when the predicted label may remain unchanged.

## 5. Correct / Failure Case Visualization

Representative correct and failure cases from the final RCE model. The panels show prediction outcome, confidence, and the dominant concept evidence at low and high magnification, illustrating both successful evidence alignment and typical error modes.
