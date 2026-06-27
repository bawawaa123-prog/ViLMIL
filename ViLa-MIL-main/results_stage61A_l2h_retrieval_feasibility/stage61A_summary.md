# Step61A Low-to-High Concept-guided Retrieval feasibility audit

## Direct Answers

1. 本 Step 是否修改了原始 RCE 文件：否。
2. 本 Step 是否修改了 RCE-v2 模型逻辑：否。
3. 5x 与 20x features 是否能按 slide_id 配对：是，`968/968`，overlap_rate=`1.000000`。
4. low/high coords 是否存在：是，low coords rate=`1.000000`，high coords rate=`1.000000`，both coords rate=`1.000000`。
5. 坐标单位和 scale ratio 是否能确定：可以基本确定。 当前 feature h5 直接存储原始 patch coords；raw coords h5 attrs 显示 low `patch_level=2, downsample≈16`，high `patch_level=1, downsample≈4`，坐标变换 scale_ratio 估计为 `1.000000`，low/high 原图 footprint 比例约为 `4.000134`。
6. low-to-high patch mapping 是否可行：是，mapping_quality_label=`strong_same_coord_system_mapping`。
7. 每个 low patch 平均能匹配多少 high patches：`13.215211`，median=`14.250000`。
8. 有多少 low patches 找不到 high patch：`0.000000`。
9. 当前模型是否已有 low/high region evidence 可导出： low_region_features=`True`，high_region_features=`True`， low/high concept evidence=`True`，per_region_evidence=`True`。
10. 如果进入 Step61B，需要新增哪些 export fields： last_low_patch_concept_scores, last_low_patch_topk_indices, last_low_patch_topk_scores, last_low_patch_coords, last_retrieved_high_patch_indices, last_retrieved_high_patch_coords, last_retrieved_high_patch_match_counts, last_retrieved_high_patch_mask, last_retrieval_debug。
11. 最终 decision 是什么：`feasible_for_step61B`。
12. 下一步建议：进入 Step61B Low-to-High Retrieval all-off implementation。

## Candidate Positioning

- Step58C: hard-metric primary candidate (`decision=candidate_primary`)
- Step59C: Dynamic CSG reference / AUC-PR variant (`decision=candidate_primary`)
- Step60D: CCRA balanced representative (`decision=secondary_preferred_over_config_d`)

## Evidence Notes

- Step41 prior audit: 已存在 Step41 formal audit，recommended_coord_mode=`top_left`，recommended_scale_ratio=`1.0`。
- retrieval_requires_new_export_fields: `True`
- sampled_mapping_slides: `20`
- sampled_low_patches: `3550`
- sampled_high_patches: `49693`
- feature/raw coord set match rate on sampled slides: low=`1.000000` high=`1.000000`

## Decision Basis

- overlap_rate=1.000000
- low_coord_rate=1.000000
- high_coord_rate=1.000000
- both_coord_rate=1.000000
- avg_high_patches_per_low_patch=13.215211
- percent_low_patches_with_zero_high_match=0.000000
- percent_low_patches_with_5plus_high_matches=0.965634
- retrieval_requires_new_export_fields=True
