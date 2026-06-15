# coding=utf-8
"""
DEG-MIL skeleton built on top of the current RCE-MIL BiomedCLIP implementation.

Step25 keeps the logits path aligned with RCE-v4-CSG-a01-rq16 and adds:
- exported low/high region attention weights
- estimated low/high region coordinates from attention-weighted patch coords

Step26 adds an optional same-scale Spatial Region Graph:
- low region graph
- high region graph

The Spatial Region Graph operates on aggregated region tokens rather than raw patch
graphs. Region coordinates are attention-weighted patch-coordinate centroids derived
from the learned region attention maps.

Step29 adds an optional same-scale Concept Prompt Graph:
- low concept graph
- high concept graph

The Concept Prompt Graph is built independently inside each class and scale over the
existing concept prompt pool. It updates prompt features before region-concept
evidence aggregation and keeps the existing RCE cross-scale graph path unchanged.

Step36 adds an optional low-high evidence consistency auxiliary loss for the dual-scale
DEG skeleton without changing the default logits path.

Step43 adds an optional HCRC-Light residual branch over low/high raw patch features.
"""

from __future__ import absolute_import, division, print_function

import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .model_RCE_MIL_BiomedCLIP import RCE_MIL_BiomedCLIP

logger = logging.getLogger(__name__)


class DEG_MIL_BiomedCLIP(RCE_MIL_BiomedCLIP):
    def __init__(
        self,
        config,
        num_classes=2,
        model_path="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ):
        super().__init__(config=config, num_classes=num_classes, model_path=model_path)
        self.rce_use_visual_evidence_gate = bool(getattr(config, "rce_use_visual_evidence_gate", False))
        self.rce_visual_gate_init = float(getattr(config, "rce_visual_gate_init", 1.0))
        self.rce_use_low_high_consistency_loss = bool(
            getattr(config, "rce_use_low_high_consistency_loss", False)
        )
        self.rce_lh_consistency_lambda = float(getattr(config, "rce_lh_consistency_lambda", 0.0))
        self.rce_lh_consistency_margin = float(getattr(config, "rce_lh_consistency_margin", 0.0))
        self.deg_use_region_graph = bool(getattr(config, "deg_use_region_graph", False))
        self.deg_region_graph_k = int(getattr(config, "deg_region_graph_k", 4))
        self.deg_region_graph_alpha = float(getattr(config, "deg_region_graph_alpha", 0.1))
        self.deg_use_concept_graph = bool(getattr(config, "deg_use_concept_graph", False))
        self.deg_concept_graph_topk = int(getattr(config, "deg_concept_graph_topk", 4))
        self.deg_concept_graph_alpha = float(getattr(config, "deg_concept_graph_alpha", 0.05))
        self.rce_use_hcrc = bool(getattr(config, "rce_use_hcrc", False))
        self.rce_hcrc_alpha_init = float(getattr(config, "rce_hcrc_alpha_init", 0.05))
        self.rce_hcrc_num_anchors = int(getattr(config, "rce_hcrc_num_anchors", 16))
        self.rce_hcrc_num_high_children = int(getattr(config, "rce_hcrc_num_high_children", 16))
        self.rce_hcrc_proposal_radius = float(getattr(config, "rce_hcrc_proposal_radius", 4096.0))
        self.rce_hcrc_nms_radius = float(getattr(config, "rce_hcrc_nms_radius", 512.0))
        self.rce_hcrc_bbox_expand = float(getattr(config, "rce_hcrc_bbox_expand", 8.0))
        self.rce_hcrc_coord_mode = str(getattr(config, "rce_hcrc_coord_mode", "top_left"))
        self.rce_hcrc_scale_ratio = float(getattr(config, "rce_hcrc_scale_ratio", 1.0))
        self.rce_hcrc_child_strategy = str(getattr(config, "rce_hcrc_child_strategy", "bbox_containment"))
        self.rce_hcrc_candidate_top_l = int(getattr(config, "rce_hcrc_candidate_top_l", 64))
        self.rce_hcrc_top_g_concepts = int(getattr(config, "rce_hcrc_top_g_concepts", 8))
        self.rce_hcrc_per_concept_top_m = int(getattr(config, "rce_hcrc_per_concept_top_m", 4))
        self.rce_hcrc_prompt_topk = int(getattr(config, "rce_hcrc_prompt_topk", 3))
        self.rce_hcrc_margin_weight = float(getattr(config, "rce_hcrc_margin_weight", 0.5))
        self.rce_hcrc_prompt_scale = str(getattr(config, "rce_hcrc_prompt_scale", "high"))
        self.rce_hcrc_min_child_count = int(getattr(config, "rce_hcrc_min_child_count", 1))
        self.rce_hcrc_export_debug = bool(getattr(config, "rce_hcrc_export_debug", False))
        self.rce_hcrc_low_patch_size = 256.0
        self.rce_hcrc_high_patch_size = 256.0
        self.rce_hcrc_use_bbox_then_nearest_fallback = False
        self.rce_visual_evidence_gate = nn.Parameter(
            self._sigmoid_init_to_logit(self.rce_visual_gate_init)
        )
        self.rce_hcrc_alpha = nn.Parameter(
            self._sigmoid_init_to_logit(self.rce_hcrc_alpha_init)
        )
        self.hcrc_query_proj = nn.Linear(self.input_size, self.input_size)
        self.hcrc_key_proj = nn.Linear(self.input_size, self.input_size)
        self.hcrc_value_proj = nn.Linear(self.input_size, self.input_size)
        self.hcrc_out_proj = nn.Linear(self.input_size, self.input_size)
        self.hcrc_fusion_gate = nn.Linear(3 * self.input_size, self.input_size)
        self.hcrc_norm = nn.LayerNorm(self.input_size)

        if self.deg_use_region_graph:
            self.low_region_graph_proj = nn.Linear(self.input_size, self.input_size)
            self.high_region_graph_proj = nn.Linear(self.input_size, self.input_size)
            self.low_region_graph_norm = nn.LayerNorm(self.input_size)
            self.high_region_graph_norm = nn.LayerNorm(self.input_size)
        else:
            self.low_region_graph_proj = None
            self.high_region_graph_proj = None
            self.low_region_graph_norm = None
            self.high_region_graph_norm = None

        if self.deg_use_concept_graph:
            self.low_concept_graph_proj = nn.Linear(self.input_size, self.input_size)
            self.high_concept_graph_proj = nn.Linear(self.input_size, self.input_size)
            self.low_concept_graph_norm = nn.LayerNorm(self.input_size)
            self.high_concept_graph_norm = nn.LayerNorm(self.input_size)
        else:
            self.low_concept_graph_proj = None
            self.high_concept_graph_proj = None
            self.low_concept_graph_norm = None
            self.high_concept_graph_norm = None

        self.last_low_region_attn = None
        self.last_high_region_attn = None
        self.last_low_region_coords = None
        self.last_high_region_coords = None
        self.last_low_region_adj = None
        self.last_high_region_adj = None
        self.last_low_region_graph_alpha = None
        self.last_high_region_graph_alpha = None
        self.last_low_region_features_before_graph = None
        self.last_high_region_features_before_graph = None
        self.last_low_concept_adj = None
        self.last_high_concept_adj = None
        self.last_low_prompt_features_before_graph = None
        self.last_high_prompt_features_before_graph = None
        self.last_low_prompt_features_after_graph = None
        self.last_high_prompt_features_after_graph = None
        self.last_low_concept_graph_alpha = None
        self.last_high_concept_graph_alpha = None
        self.last_slide_id = None
        self.last_visual_evidence_gate = None
        self.last_visual_residual_contribution = None
        self.last_visual_gated_contribution = None
        self.last_low_scale_logits = None
        self.last_high_scale_logits = None
        self.last_low_true_wrong_margin = None
        self.last_high_true_wrong_margin = None
        self.last_lh_margin_gap = None
        self.last_lh_consistency_loss = None
        self.last_total_loss = None
        self.last_hcrc_enabled = None
        self.last_hcrc_logits = None
        self.last_hcrc_alpha = None
        self.last_hcrc_prompt_weights = None
        self.last_hcrc_prompt_evidence = None
        self.last_hcrc_region_concept_sim = None
        self.last_hcrc_anchor_coords = None
        self.last_hcrc_anchor_bboxes = None
        self.last_hcrc_anchor_scores = None
        self.last_hcrc_anchor_valid_mask = None
        self.last_hcrc_child_counts = None
        self.last_hcrc_child_used_counts = None
        self.last_hcrc_empty_anchor_ratio = None
        self.last_hcrc_child_valid_mask = None
        self.last_hcrc_child_distance_mean = None
        self.last_hcrc_skip_reason = None

        if self.rce_use_visual_evidence_gate and not self.rce_use_visual_residual:
            logger.warning(
                "rce_use_visual_evidence_gate=True but rce_use_visual_residual=False; "
                "visual evidence gate will be ignored."
            )
        if self.rce_use_low_high_consistency_loss and self.scale_mode != "dual":
            logger.warning(
                "rce_use_low_high_consistency_loss=True but scale_mode=%s; "
                "low-high consistency loss will be skipped.",
                self.scale_mode,
            )
        if self.rce_hcrc_coord_mode not in {"top_left", "center"}:
            logger.warning(
                "Unsupported rce_hcrc_coord_mode=%s; falling back to top_left.",
                self.rce_hcrc_coord_mode,
            )
            self.rce_hcrc_coord_mode = "top_left"
        if self.rce_hcrc_prompt_scale not in {"low", "high", "avg"}:
            logger.warning(
                "Unsupported rce_hcrc_prompt_scale=%s; falling back to high.",
                self.rce_hcrc_prompt_scale,
            )
            self.rce_hcrc_prompt_scale = "high"
        if self.rce_hcrc_child_strategy != "bbox_containment":
            logger.warning(
                "Step43 HCRC-Light only implements bbox_containment; got child_strategy=%s.",
                self.rce_hcrc_child_strategy,
            )

    @staticmethod
    def _sigmoid_init_to_logit(init_value):
        init = min(max(float(init_value), 1e-6), 1.0 - 1e-6)
        return torch.logit(torch.tensor(init, dtype=torch.float32))

    @staticmethod
    def _true_vs_wrong_margin(logits, label):
        true_logit = logits.gather(1, label.view(-1, 1)).squeeze(1)
        wrong_logits = logits.masked_fill(
            F.one_hot(label, num_classes=logits.size(1)).bool(),
            float("-inf"),
        )
        max_wrong_logit = wrong_logits.max(dim=1).values
        return true_logit - max_wrong_logit

    @staticmethod
    def _detach_slide_id(slide_id):
        if slide_id is None:
            return None
        if torch.is_tensor(slide_id):
            return slide_id.detach().cpu()
        if isinstance(slide_id, tuple):
            return tuple(DEG_MIL_BiomedCLIP._detach_slide_id(item) for item in slide_id)
        if isinstance(slide_id, list):
            return [DEG_MIL_BiomedCLIP._detach_slide_id(item) for item in slide_id]
        return slide_id

    def _prepare_patch_features_for_attention(self, patch_features):
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(1)
        elif patch_features.dim() == 3 and patch_features.size(0) == 1 and patch_features.size(1) != 1:
            patch_features = patch_features.transpose(0, 1).contiguous()
        elif patch_features.dim() != 3:
            raise ValueError(f"Expected patch features rank=2/3, got rank={patch_features.dim()}")

        if patch_features.size(-1) != self.input_size:
            raise ValueError(
                f"Expected patch feature dim={self.input_size}, got {patch_features.size(-1)}"
            )
        return patch_features

    def _aggregate_region_features(self, patch_features, region_queries, attention_layer, norm_layer):
        patch_features = self._prepare_patch_features_for_attention(patch_features)
        batch_size = patch_features.size(1)
        query = region_queries.expand(-1, batch_size, -1)
        region_features, attn_weights = attention_layer(
            query,
            patch_features,
            patch_features,
            need_weights=True,
            need_raw=False,
        )
        region_features = norm_layer(region_features + query)
        region_features = region_features.permute(1, 0, 2).contiguous()

        if attn_weights is not None and attn_weights.dim() == 2:
            attn_weights = attn_weights.unsqueeze(0)

        return region_features, attn_weights

    def _compute_region_coords(self, attn_weights, coords):
        if attn_weights is None or coords is None:
            return None

        if not torch.is_tensor(coords):
            try:
                coords = torch.as_tensor(coords)
            except Exception:
                return None

        if coords.numel() == 0 or coords.dim() < 2 or coords.size(-1) < 2:
            return None

        num_batches = attn_weights.size(0)
        num_patches = attn_weights.size(-1)

        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        elif coords.dim() == 3:
            if coords.size(0) == num_patches and coords.size(1) == num_batches:
                coords = coords.permute(1, 0, 2).contiguous()
            elif coords.size(0) == num_patches and coords.size(1) == 1 and num_batches == 1:
                coords = coords.permute(1, 0, 2).contiguous()
        else:
            return None

        if coords.size(0) == 1 and num_batches > 1:
            coords = coords.expand(num_batches, -1, -1)

        if coords.size(0) != num_batches or coords.size(1) != num_patches:
            return None

        coords = coords[..., :2].to(device=attn_weights.device, dtype=attn_weights.dtype)
        weights = attn_weights.to(dtype=torch.float32)
        coords = coords.to(dtype=torch.float32)

        weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        region_coords = torch.einsum("brn,bnd->brd", weights, coords) / weight_sum
        return region_coords

    def _build_region_knn_adj(self, region_coords):
        if region_coords is None or region_coords.dim() != 3 or region_coords.size(-1) < 2:
            return None
        if not torch.isfinite(region_coords).all():
            return None

        _, num_regions, _ = region_coords.shape
        if num_regions <= 1:
            return None

        k = min(max(self.deg_region_graph_k, 0), num_regions - 1)
        if k <= 0:
            return None

        pairwise_distance = torch.cdist(region_coords.float(), region_coords.float(), p=2)
        diag_mask = torch.eye(num_regions, device=pairwise_distance.device, dtype=torch.bool).unsqueeze(0)
        pairwise_distance = pairwise_distance.masked_fill(diag_mask, float("inf"))
        neighbor_indices = torch.topk(pairwise_distance, k=k, dim=-1, largest=False).indices

        adjacency = torch.zeros_like(pairwise_distance)
        adjacency.scatter_(-1, neighbor_indices, 1.0)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adjacency

    def _apply_region_graph(self, region_features, region_coords, graph_proj, graph_norm):
        if not self.deg_use_region_graph or graph_proj is None or graph_norm is None:
            return region_features, None

        adjacency = self._build_region_knn_adj(region_coords)
        if adjacency is None:
            return region_features, None

        neighbor_context = torch.bmm(adjacency.to(region_features.dtype), region_features)
        updated_region_features = region_features + self.deg_region_graph_alpha * graph_proj(neighbor_context)
        updated_region_features = graph_norm(updated_region_features)
        return updated_region_features, adjacency

    def _build_concept_knn_adj(self, prompt_features, topk):
        if prompt_features is None or prompt_features.dim() != 3:
            return None
        if prompt_features.size(-1) != self.input_size:
            return None
        if not torch.isfinite(prompt_features).all():
            return None

        _, num_prompts, _ = prompt_features.shape
        if num_prompts <= 1:
            return None

        k = min(max(int(topk), 0), num_prompts - 1)
        if k <= 0:
            return None

        normalized_prompts = F.normalize(prompt_features.float(), dim=-1)
        similarity = torch.bmm(normalized_prompts, normalized_prompts.transpose(1, 2))
        diag_mask = torch.eye(num_prompts, device=similarity.device, dtype=torch.bool).unsqueeze(0)
        similarity = similarity.masked_fill(diag_mask, float("-inf"))
        neighbor_indices = torch.topk(similarity, k=k, dim=-1, largest=True).indices

        adjacency = torch.zeros_like(similarity)
        adjacency.scatter_(-1, neighbor_indices, 1.0)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adjacency

    def _apply_concept_graph(self, prompt_features, graph_proj, graph_norm, scale_name):
        if not self.deg_use_concept_graph or graph_proj is None or graph_norm is None:
            return prompt_features, None
        if prompt_features is None or prompt_features.dim() != 3:
            logger.warning(
                "deg_use_concept_graph=True but %s prompt features are invalid; skipping concept graph.",
                scale_name,
            )
            return prompt_features, None
        if prompt_features.size(-1) != self.input_size:
            logger.warning(
                "deg_use_concept_graph=True but %s prompt feature dim=%s (expected %s); skipping concept graph.",
                scale_name,
                prompt_features.size(-1),
                self.input_size,
            )
            return prompt_features, None

        adjacency = self._build_concept_knn_adj(prompt_features, self.deg_concept_graph_topk)
        if adjacency is None:
            return prompt_features, None

        concept_context = torch.bmm(adjacency.to(prompt_features.dtype), prompt_features)
        updated_prompt_features = prompt_features + self.deg_concept_graph_alpha * graph_proj(concept_context)
        updated_prompt_features = graph_norm(updated_prompt_features)
        return updated_prompt_features, adjacency

    @staticmethod
    def _hcrc_topmean(values, k, dim=-1):
        size = values.size(dim)
        if size == 0:
            shape = list(values.shape)
            del shape[dim]
            return values.new_zeros(shape)
        k = min(max(int(k), 1), size)
        return torch.topk(values, k=k, dim=dim).values.mean(dim=dim)

    def _hcrc_to_batched_patch_features(self, patch_features):
        if patch_features is None:
            return None
        if not torch.is_tensor(patch_features):
            patch_features = torch.as_tensor(patch_features)
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(0)
        elif patch_features.dim() == 3:
            if patch_features.size(-1) != self.input_size:
                raise ValueError(
                    f"Expected patch feature dim={self.input_size}, got {patch_features.size(-1)}"
                )
            if patch_features.size(1) == 1 and patch_features.size(0) != 1:
                patch_features = patch_features.transpose(0, 1).contiguous()
        else:
            raise ValueError(f"Expected patch features rank=2/3, got rank={patch_features.dim()}")

        if patch_features.size(-1) != self.input_size:
            raise ValueError(
                f"Expected patch feature dim={self.input_size}, got {patch_features.size(-1)}"
            )
        return patch_features.float()

    def _hcrc_to_batched_coords(self, coords, expected_batch, expected_patches):
        if coords is None:
            return None
        if not torch.is_tensor(coords):
            try:
                coords = torch.as_tensor(coords)
            except Exception:
                return None
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        elif coords.dim() == 3:
            if coords.size(0) == expected_patches and coords.size(1) == expected_batch:
                coords = coords.permute(1, 0, 2).contiguous()
            elif coords.size(0) == expected_patches and coords.size(1) == 1 and expected_batch == 1:
                coords = coords.permute(1, 0, 2).contiguous()
            elif coords.size(1) == 1 and coords.size(0) != expected_batch and expected_batch == 1:
                coords = coords.transpose(0, 1).contiguous()
        else:
            return None

        if coords.size(0) == 1 and expected_batch > 1:
            coords = coords.expand(expected_batch, -1, -1)
        if coords.size(0) != expected_batch or coords.size(1) != expected_patches or coords.size(-1) < 2:
            return None
        return coords[..., :2].float()

    def _hcrc_coords_to_centers(self, coords, patch_size, coord_mode):
        if coord_mode == "top_left":
            return coords + float(patch_size) / 2.0
        return coords

    def _hcrc_expand_bbox(self, bbox, expand):
        center = (bbox[..., :2] + bbox[..., 2:]) / 2.0
        half = (bbox[..., 2:] - bbox[..., :2]) * float(expand) / 2.0
        return torch.cat([center - half, center + half], dim=-1)

    def _hcrc_compute_patch_scores(self, low_patch_features, low_prompt_features):
        low_features_norm = F.normalize(low_patch_features.float(), dim=-1)
        prompt_features_norm = F.normalize(low_prompt_features.float(), dim=-1)
        sim = torch.einsum("bnd,cpd->bncp", low_features_norm, prompt_features_norm)
        concept_relevance = self._hcrc_topmean(
            sim.reshape(sim.size(0), sim.size(1), -1),
            self.rce_hcrc_prompt_topk,
            dim=-1,
        )
        class_scores = self._hcrc_topmean(sim, self.rce_hcrc_prompt_topk, dim=-1)
        class_topk = min(2, class_scores.size(-1))
        class_values, class_indices = torch.topk(class_scores, k=class_topk, dim=-1)
        top_class_score = class_values[..., 0]
        second_class_score = class_values[..., 1] if class_topk > 1 else class_values[..., 0]
        top_class = class_indices[..., 0]
        second_class = class_indices[..., 1] if class_topk > 1 else class_indices[..., 0]
        class_margin = top_class_score - second_class_score
        patch_score = concept_relevance + self.rce_hcrc_margin_weight * class_margin

        flat_sim = sim.reshape(sim.size(0), sim.size(1), -1)
        top_prompt_flat = flat_sim.argmax(dim=-1)
        prompt_count = sim.size(-1)
        top_prompt_class = torch.div(top_prompt_flat, prompt_count, rounding_mode="floor")
        top_prompt_index = top_prompt_flat.remainder(prompt_count)
        top_prompt_score = flat_sim.gather(-1, top_prompt_flat.unsqueeze(-1)).squeeze(-1)

        return {
            "sim": sim,
            "low_features_norm": low_features_norm,
            "concept_relevance": concept_relevance,
            "class_scores": class_scores,
            "top_class": top_class,
            "second_class": second_class,
            "top_class_score": top_class_score,
            "second_class_score": second_class_score,
            "class_margin": class_margin,
            "patch_score": patch_score,
            "top_prompt_class": top_prompt_class,
            "top_prompt_index": top_prompt_index,
            "top_prompt_score": top_prompt_score,
        }

    def _hcrc_select_candidate_indices(self, sim, patch_score):
        batch_size, num_patches, _, num_prompts = sim.shape
        candidate_indices = []
        for batch_idx in range(batch_size):
            selected = []
            seen = set()
            top_l = min(max(self.rce_hcrc_candidate_top_l, 0), num_patches)
            if top_l > 0:
                for patch_idx in torch.topk(patch_score[batch_idx], k=top_l, dim=0).indices.tolist():
                    patch_idx = int(patch_idx)
                    if patch_idx not in seen:
                        seen.add(patch_idx)
                        selected.append(patch_idx)

            concept_evidence = self._hcrc_topmean(
                sim[batch_idx],
                self.rce_hcrc_prompt_topk,
                dim=0,
            )
            flat_evidence = concept_evidence.reshape(-1)
            top_g = min(max(self.rce_hcrc_top_g_concepts, 0), flat_evidence.numel())
            if top_g > 0:
                for flat_idx in torch.topk(flat_evidence, k=top_g, dim=0).indices.tolist():
                    flat_idx = int(flat_idx)
                    class_idx = flat_idx // num_prompts
                    prompt_idx = flat_idx % num_prompts
                    top_m = min(max(self.rce_hcrc_per_concept_top_m, 0), num_patches)
                    if top_m <= 0:
                        continue
                    prompt_scores = sim[batch_idx, :, class_idx, prompt_idx]
                    for patch_idx in torch.topk(prompt_scores, k=top_m, dim=0).indices.tolist():
                        patch_idx = int(patch_idx)
                        if patch_idx not in seen:
                            seen.add(patch_idx)
                            selected.append(patch_idx)

            if not selected and num_patches > 0:
                selected.append(int(torch.argmax(patch_score[batch_idx]).item()))

            candidate_indices.append(
                torch.as_tensor(selected, device=sim.device, dtype=torch.long)
            )
        return candidate_indices

    def _hcrc_build_local_proposals(self, low_patch_features, low_coords, patch_score, candidate_indices):
        batch_size, _, feat_dim = low_patch_features.shape
        scaled_low_coords = low_coords * self.rce_hcrc_scale_ratio
        scaled_low_patch_size = self.rce_hcrc_low_patch_size * self.rce_hcrc_scale_ratio
        low_centers = self._hcrc_coords_to_centers(
            scaled_low_coords,
            scaled_low_patch_size,
            self.rce_hcrc_coord_mode,
        )

        proposals = []
        for batch_idx in range(batch_size):
            candidates = candidate_indices[batch_idx]
            if candidates.numel() == 0:
                proposals.append(
                    {
                        "features": low_patch_features.new_zeros((0, feat_dim)),
                        "coords": low_patch_features.new_zeros((0, 2)),
                        "centers": low_patch_features.new_zeros((0, 2)),
                        "bboxes": low_patch_features.new_zeros((0, 4)),
                        "scores": low_patch_features.new_zeros((0,)),
                        "neighbor_counts": torch.zeros((0,), device=low_patch_features.device, dtype=torch.long),
                    }
                )
                continue

            proposal_features = []
            proposal_coords = []
            proposal_centers = []
            proposal_bboxes = []
            proposal_scores = []
            neighbor_counts = []

            for patch_idx in candidates.tolist():
                center = low_centers[batch_idx, patch_idx]
                distances = torch.linalg.norm(
                    low_centers[batch_idx] - center.unsqueeze(0),
                    dim=-1,
                )
                neighbor_indices = torch.nonzero(
                    distances <= self.rce_hcrc_proposal_radius,
                    as_tuple=False,
                ).flatten()
                if neighbor_indices.numel() == 0:
                    neighbor_indices = torch.as_tensor(
                        [patch_idx],
                        device=low_patch_features.device,
                        dtype=torch.long,
                    )

                neighbor_scores = patch_score[batch_idx, neighbor_indices]
                weights = F.softmax(neighbor_scores, dim=0)
                proposal_feature = torch.sum(
                    low_patch_features[batch_idx, neighbor_indices] * weights.unsqueeze(-1),
                    dim=0,
                )
                proposal_coord = torch.sum(
                    scaled_low_coords[batch_idx, neighbor_indices] * weights.unsqueeze(-1),
                    dim=0,
                )
                proposal_center = torch.sum(
                    low_centers[batch_idx, neighbor_indices] * weights.unsqueeze(-1),
                    dim=0,
                )
                proposal_score = torch.sum(neighbor_scores * weights)

                neighbor_coords = scaled_low_coords[batch_idx, neighbor_indices]
                if self.rce_hcrc_coord_mode == "top_left":
                    bbox_x0 = neighbor_coords[:, 0].min()
                    bbox_y0 = neighbor_coords[:, 1].min()
                    bbox_x1 = neighbor_coords[:, 0].max() + scaled_low_patch_size
                    bbox_y1 = neighbor_coords[:, 1].max() + scaled_low_patch_size
                else:
                    half = scaled_low_patch_size / 2.0
                    bbox_x0 = (neighbor_coords[:, 0] - half).min()
                    bbox_y0 = (neighbor_coords[:, 1] - half).min()
                    bbox_x1 = (neighbor_coords[:, 0] + half).max()
                    bbox_y1 = (neighbor_coords[:, 1] + half).max()
                proposal_bbox = torch.stack([bbox_x0, bbox_y0, bbox_x1, bbox_y1], dim=0)

                proposal_features.append(proposal_feature)
                proposal_coords.append(proposal_coord)
                proposal_centers.append(proposal_center)
                proposal_bboxes.append(proposal_bbox)
                proposal_scores.append(proposal_score)
                neighbor_counts.append(int(neighbor_indices.numel()))

            proposals.append(
                {
                    "features": torch.stack(proposal_features, dim=0),
                    "coords": torch.stack(proposal_coords, dim=0),
                    "centers": torch.stack(proposal_centers, dim=0),
                    "bboxes": torch.stack(proposal_bboxes, dim=0),
                    "scores": torch.stack(proposal_scores, dim=0),
                    "neighbor_counts": torch.as_tensor(
                        neighbor_counts,
                        device=low_patch_features.device,
                        dtype=torch.long,
                    ),
                }
            )
        return proposals

    def _hcrc_spatial_nms(self, proposals):
        batch_size = len(proposals)
        device = self.region_queries_low.device
        anchor_features = torch.zeros(
            batch_size,
            self.rce_hcrc_num_anchors,
            self.input_size,
            device=device,
        )
        anchor_coords = torch.zeros(batch_size, self.rce_hcrc_num_anchors, 2, device=device)
        anchor_centers = torch.zeros(batch_size, self.rce_hcrc_num_anchors, 2, device=device)
        anchor_bboxes = torch.zeros(batch_size, self.rce_hcrc_num_anchors, 4, device=device)
        anchor_scores = torch.zeros(batch_size, self.rce_hcrc_num_anchors, device=device)
        anchor_valid_mask = torch.zeros(
            batch_size,
            self.rce_hcrc_num_anchors,
            device=device,
            dtype=torch.bool,
        )
        anchor_neighbor_counts = torch.zeros(
            batch_size,
            self.rce_hcrc_num_anchors,
            device=device,
            dtype=torch.long,
        )

        for batch_idx, proposal in enumerate(proposals):
            scores = proposal["scores"]
            if scores.numel() == 0:
                continue
            ordered = torch.argsort(scores, descending=True)
            selected = []
            for idx_tensor in ordered:
                idx = int(idx_tensor.item())
                center = proposal["centers"][idx]
                suppressed = False
                for selected_idx in selected:
                    dist = torch.linalg.norm(center - proposal["centers"][selected_idx], dim=-1)
                    if float(dist.item()) < self.rce_hcrc_nms_radius:
                        suppressed = True
                        break
                if suppressed:
                    continue
                selected.append(idx)
                if len(selected) >= self.rce_hcrc_num_anchors:
                    break

            if not selected:
                continue

            sel = torch.as_tensor(selected, device=device, dtype=torch.long)
            count = sel.numel()
            anchor_features[batch_idx, :count] = proposal["features"][sel]
            anchor_coords[batch_idx, :count] = proposal["coords"][sel]
            anchor_centers[batch_idx, :count] = proposal["centers"][sel]
            anchor_bboxes[batch_idx, :count] = proposal["bboxes"][sel]
            anchor_scores[batch_idx, :count] = proposal["scores"][sel]
            anchor_valid_mask[batch_idx, :count] = True
            anchor_neighbor_counts[batch_idx, :count] = proposal["neighbor_counts"][sel]

        return {
            "features": anchor_features,
            "coords": anchor_coords,
            "centers": anchor_centers,
            "bboxes": anchor_bboxes,
            "scores": anchor_scores,
            "valid_mask": anchor_valid_mask,
            "neighbor_counts": anchor_neighbor_counts,
        }

    def _hcrc_match_high_children(self, anchors, high_patch_features, high_coords):
        batch_size, _, feat_dim = high_patch_features.shape[0], self.rce_hcrc_num_anchors, high_patch_features.shape[-1]
        child_features = high_patch_features.new_zeros(
            batch_size,
            self.rce_hcrc_num_anchors,
            self.rce_hcrc_num_high_children,
            feat_dim,
        )
        child_valid_mask = torch.zeros(
            batch_size,
            self.rce_hcrc_num_anchors,
            self.rce_hcrc_num_high_children,
            device=high_patch_features.device,
            dtype=torch.bool,
        )
        child_counts = torch.zeros(
            batch_size,
            self.rce_hcrc_num_anchors,
            device=high_patch_features.device,
            dtype=torch.long,
        )
        child_used_counts = torch.zeros_like(child_counts)
        child_distance_mean = high_patch_features.new_zeros(batch_size, self.rce_hcrc_num_anchors)
        empty_anchor_ratio = high_patch_features.new_zeros(batch_size)

        high_centers = self._hcrc_coords_to_centers(
            high_coords,
            self.rce_hcrc_high_patch_size,
            "top_left",
        )

        for batch_idx in range(batch_size):
            empty_count = 0
            valid_anchor_count = int(anchors["valid_mask"][batch_idx].sum().item())
            for anchor_idx in range(self.rce_hcrc_num_anchors):
                if not bool(anchors["valid_mask"][batch_idx, anchor_idx].item()):
                    empty_count += 1
                    continue
                if self.rce_hcrc_child_strategy != "bbox_containment":
                    empty_count += 1
                    continue

                bbox = self._hcrc_expand_bbox(
                    anchors["bboxes"][batch_idx, anchor_idx].unsqueeze(0),
                    self.rce_hcrc_bbox_expand,
                ).squeeze(0)
                inside = (
                    (high_centers[batch_idx, :, 0] >= bbox[0])
                    & (high_centers[batch_idx, :, 0] <= bbox[2])
                    & (high_centers[batch_idx, :, 1] >= bbox[1])
                    & (high_centers[batch_idx, :, 1] <= bbox[3])
                )
                raw_indices = torch.nonzero(inside, as_tuple=False).flatten()
                child_counts[batch_idx, anchor_idx] = raw_indices.numel()

                if raw_indices.numel() > 0:
                    anchor_center = anchors["centers"][batch_idx, anchor_idx]
                    distances = torch.linalg.norm(
                        high_centers[batch_idx, raw_indices] - anchor_center.unsqueeze(0),
                        dim=-1,
                    )
                    top_m = min(self.rce_hcrc_num_high_children, raw_indices.numel())
                    order = torch.argsort(distances)[:top_m]
                    chosen = raw_indices[order]
                    chosen_distances = distances[order]
                    child_features[batch_idx, anchor_idx, :top_m] = high_patch_features[batch_idx, chosen]
                    child_valid_mask[batch_idx, anchor_idx, :top_m] = True
                    child_used_counts[batch_idx, anchor_idx] = top_m
                    child_distance_mean[batch_idx, anchor_idx] = chosen_distances.mean()

                if int(child_used_counts[batch_idx, anchor_idx].item()) < self.rce_hcrc_min_child_count:
                    child_valid_mask[batch_idx, anchor_idx] = False
                    child_distance_mean[batch_idx, anchor_idx] = 0.0
                    empty_count += 1

            denom = max(valid_anchor_count, 1)
            empty_anchor_ratio[batch_idx] = float(empty_count) / float(denom)

        return {
            "features": child_features,
            "valid_mask": child_valid_mask,
            "counts": child_counts,
            "used_counts": child_used_counts,
            "distance_mean": child_distance_mean,
            "empty_anchor_ratio": empty_anchor_ratio,
        }

    def _hcrc_aggregate_high_detail(self, anchor_features, child_features, child_valid_mask):
        query = self.hcrc_query_proj(anchor_features)
        key = self.hcrc_key_proj(child_features)
        value = self.hcrc_value_proj(child_features)

        attn_logits = (query.unsqueeze(2) * key).sum(dim=-1) / math.sqrt(float(self.input_size))
        attn_logits = attn_logits.masked_fill(~child_valid_mask, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights * child_valid_mask.float()
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        high_detail = torch.sum(attn_weights.unsqueeze(-1) * value, dim=2)
        high_detail = self.hcrc_out_proj(high_detail)
        valid_anchor = child_valid_mask.any(dim=-1, keepdim=True)
        high_detail = high_detail * valid_anchor.to(high_detail.dtype)
        return high_detail

    def _hcrc_compute_logits(self, paired_tokens, low_prompt_features, high_prompt_features):
        if self.rce_hcrc_prompt_scale == "low":
            prompt_features = low_prompt_features
        elif self.rce_hcrc_prompt_scale == "avg":
            low_norm = F.normalize(low_prompt_features.float(), dim=-1)
            high_norm = F.normalize(high_prompt_features.float(), dim=-1)
            prompt_features = F.normalize(0.5 * (low_norm + high_norm), dim=-1)
        else:
            prompt_features = high_prompt_features
        return self._compute_scale_logits(paired_tokens, prompt_features)

    def _hcrc_run(
        self,
        low_patch_features,
        low_coords,
        high_patch_features,
        high_coords,
        low_prompt_features,
        high_prompt_features,
    ):
        try:
            low_patch_features = self._hcrc_to_batched_patch_features(low_patch_features)
            high_patch_features = self._hcrc_to_batched_patch_features(high_patch_features)
        except Exception as exc:
            return {"success": False, "skip_reason": f"invalid_patch_features:{exc}"}

        low_coords = self._hcrc_to_batched_coords(
            low_coords,
            expected_batch=low_patch_features.size(0),
            expected_patches=low_patch_features.size(1),
        )
        high_coords = self._hcrc_to_batched_coords(
            high_coords,
            expected_batch=high_patch_features.size(0),
            expected_patches=high_patch_features.size(1),
        )
        if low_coords is None or high_coords is None:
            return {"success": False, "skip_reason": "missing_or_invalid_coords"}
        if self.rce_hcrc_child_strategy != "bbox_containment":
            return {"success": False, "skip_reason": "unsupported_child_strategy"}

        score_info = self._hcrc_compute_patch_scores(low_patch_features, low_prompt_features)
        candidate_indices = self._hcrc_select_candidate_indices(
            score_info["sim"],
            score_info["patch_score"],
        )
        proposals = self._hcrc_build_local_proposals(
            low_patch_features,
            low_coords,
            score_info["patch_score"],
            candidate_indices,
        )
        anchors = self._hcrc_spatial_nms(proposals)
        if not bool(anchors["valid_mask"].any().item()):
            return {"success": False, "skip_reason": "no_valid_anchors"}

        child_info = self._hcrc_match_high_children(
            anchors,
            high_patch_features,
            high_coords,
        )
        high_detail = self._hcrc_aggregate_high_detail(
            anchors["features"],
            child_info["features"],
            child_info["valid_mask"],
        )
        gate_input = torch.cat(
            [
                anchors["features"],
                high_detail,
                anchors["features"] * high_detail,
            ],
            dim=-1,
        )
        fusion_gate = torch.sigmoid(self.hcrc_fusion_gate(gate_input))
        paired_tokens = self.hcrc_norm(anchors["features"] + fusion_gate * high_detail)
        paired_tokens = paired_tokens * anchors["valid_mask"].unsqueeze(-1).to(paired_tokens.dtype)
        hcrc_logits, hcrc_prompt_weights, hcrc_prompt_evidence, hcrc_region_concept_sim = self._hcrc_compute_logits(
            paired_tokens,
            low_prompt_features,
            high_prompt_features,
        )

        if not torch.isfinite(hcrc_logits).all():
            return {"success": False, "skip_reason": "non_finite_hcrc_logits"}

        return {
            "success": True,
            "skip_reason": None,
            "hcrc_logits": hcrc_logits,
            "prompt_weights": hcrc_prompt_weights,
            "prompt_evidence": hcrc_prompt_evidence,
            "region_concept_sim": hcrc_region_concept_sim,
            "anchor_coords": anchors["coords"],
            "anchor_bboxes": anchors["bboxes"],
            "anchor_scores": anchors["scores"],
            "anchor_valid_mask": anchors["valid_mask"],
            "anchor_neighbor_counts": anchors["neighbor_counts"],
            "child_counts": child_info["counts"],
            "child_used_counts": child_info["used_counts"],
            "empty_anchor_ratio": child_info["empty_anchor_ratio"],
            "child_valid_mask": child_info["valid_mask"],
            "child_distance_mean": child_info["distance_mean"],
        }

    def _set_hcrc_debug_defaults(self):
        self.last_hcrc_enabled = torch.tensor(False)
        self.last_hcrc_logits = None
        self.last_hcrc_alpha = (
            torch.sigmoid(self.rce_hcrc_alpha).detach().cpu() if self.rce_use_hcrc else None
        )
        self.last_hcrc_prompt_weights = None
        self.last_hcrc_prompt_evidence = None
        self.last_hcrc_region_concept_sim = None
        self.last_hcrc_anchor_coords = None
        self.last_hcrc_anchor_bboxes = None
        self.last_hcrc_anchor_scores = None
        self.last_hcrc_anchor_valid_mask = None
        self.last_hcrc_child_counts = None
        self.last_hcrc_child_used_counts = None
        self.last_hcrc_empty_anchor_ratio = None
        self.last_hcrc_child_valid_mask = None
        self.last_hcrc_child_distance_mean = None
        self.last_hcrc_skip_reason = None

    def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        low_patches = x_s.float()
        high_patches = x_l.float()
        self._set_hcrc_debug_defaults()

        low_region_features, low_attn_weights = self._aggregate_region_features(
            low_patches,
            self.region_queries_low,
            self.region_attention_low,
            self.norm_low,
        )
        high_region_features, high_attn_weights = self._aggregate_region_features(
            high_patches,
            self.region_queries_high,
            self.region_attention_high,
            self.norm_high,
        )

        low_region_coords = self._compute_region_coords(low_attn_weights, coord_s)
        high_region_coords = self._compute_region_coords(high_attn_weights, coords_l)
        low_region_features_before_graph = low_region_features
        high_region_features_before_graph = high_region_features

        if self.deg_use_region_graph:
            low_region_features, low_region_adj = self._apply_region_graph(
                low_region_features,
                low_region_coords,
                self.low_region_graph_proj,
                self.low_region_graph_norm,
            )
            high_region_features, high_region_adj = self._apply_region_graph(
                high_region_features,
                high_region_coords,
                self.high_region_graph_proj,
                self.high_region_graph_norm,
            )
        else:
            low_region_adj = None
            high_region_adj = None

        low_prompt_features = self.low_prompt_features.to(x_s.device)
        high_prompt_features = self.high_prompt_features.to(x_s.device)
        low_prompt_features_before_graph = low_prompt_features
        high_prompt_features_before_graph = high_prompt_features

        if self.deg_use_concept_graph:
            low_prompt_features, low_concept_adj = self._apply_concept_graph(
                low_prompt_features,
                self.low_concept_graph_proj,
                self.low_concept_graph_norm,
                scale_name="low",
            )
            high_prompt_features, high_concept_adj = self._apply_concept_graph(
                high_prompt_features,
                self.high_concept_graph_proj,
                self.high_concept_graph_norm,
                scale_name="high",
            )
        else:
            low_concept_adj = None
            high_concept_adj = None

        low_concept_prior = self.low_concept_prior if self.rce_use_concept_prior else None
        high_concept_prior = self.high_concept_prior if self.rce_use_concept_prior else None

        logits_low, low_prompt_weights, low_prompt_evidence, low_region_concept_sim = self._compute_scale_logits(
            low_region_features,
            low_prompt_features,
            concept_prior=low_concept_prior,
        )
        logits_high, high_prompt_weights, high_prompt_evidence, high_region_concept_sim = self._compute_scale_logits(
            high_region_features,
            high_prompt_features,
            concept_prior=high_concept_prior,
        )
        self.last_low_scale_logits = logits_low.detach().cpu()
        self.last_high_scale_logits = logits_high.detach().cpu()

        low_visual_logits = None
        high_visual_logits = None
        visual_logits = None

        if self.scale_mode == "low_only":
            final_logits = logits_low
            if self.rce_use_visual_residual:
                low_region_pool = low_region_features.mean(dim=1)
                low_visual_logits = self.low_visual_head(low_region_pool)
                visual_logits = low_visual_logits
        elif self.scale_mode == "high_only":
            final_logits = logits_high
            if self.rce_use_visual_residual:
                high_region_pool = high_region_features.mean(dim=1)
                high_visual_logits = self.high_visual_head(high_region_pool)
                visual_logits = high_visual_logits
        else:
            final_logits = logits_low + logits_high
            if self.rce_use_visual_residual:
                low_region_pool = low_region_features.mean(dim=1)
                high_region_pool = high_region_features.mean(dim=1)
                low_visual_logits = self.low_visual_head(low_region_pool)
                high_visual_logits = self.high_visual_head(high_region_pool)
                visual_logits = low_visual_logits + high_visual_logits

        if self.rce_use_visual_residual:
            alpha = torch.sigmoid(self.rce_visual_residual_alpha)
            visual_residual_contribution = alpha * visual_logits
            if self.rce_use_visual_evidence_gate:
                gate = torch.sigmoid(self.rce_visual_evidence_gate)
                visual_gated_contribution = gate * visual_residual_contribution
            else:
                gate = torch.ones_like(alpha)
                visual_gated_contribution = visual_residual_contribution
            final_logits = final_logits + visual_gated_contribution
            self.last_visual_residual_alpha = alpha.detach().cpu()
            self.last_low_visual_logits = (
                low_visual_logits.detach().cpu() if low_visual_logits is not None else None
            )
            self.last_high_visual_logits = (
                high_visual_logits.detach().cpu() if high_visual_logits is not None else None
            )
            self.last_visual_logits = visual_logits.detach().cpu()
            self.last_visual_evidence_gate = gate.detach().cpu()
            self.last_visual_residual_contribution = visual_residual_contribution.detach().cpu()
            self.last_visual_gated_contribution = visual_gated_contribution.detach().cpu()
        else:
            self.last_visual_residual_alpha = None
            self.last_low_visual_logits = None
            self.last_high_visual_logits = None
            self.last_visual_logits = None
            self.last_visual_evidence_gate = None
            self.last_visual_residual_contribution = None
            self.last_visual_gated_contribution = None

        hcrc_result = None
        if self.rce_use_hcrc and self.scale_mode == "dual":
            hcrc_result = self._hcrc_run(
                low_patches,
                coord_s,
                high_patches,
                coords_l,
                low_prompt_features,
                high_prompt_features,
            )
            if hcrc_result.get("success", False):
                hcrc_alpha = torch.sigmoid(self.rce_hcrc_alpha)
                final_logits = final_logits + hcrc_alpha * hcrc_result["hcrc_logits"]
                self.last_hcrc_enabled = torch.tensor(True)
                self.last_hcrc_logits = hcrc_result["hcrc_logits"].detach().cpu()
                self.last_hcrc_alpha = hcrc_alpha.detach().cpu()
                self.last_hcrc_prompt_weights = hcrc_result["prompt_weights"].detach().cpu()
                self.last_hcrc_prompt_evidence = hcrc_result["prompt_evidence"].detach().cpu()
                self.last_hcrc_region_concept_sim = hcrc_result["region_concept_sim"].detach().cpu()
                self.last_hcrc_anchor_coords = hcrc_result["anchor_coords"].detach().cpu()
                self.last_hcrc_anchor_bboxes = hcrc_result["anchor_bboxes"].detach().cpu()
                self.last_hcrc_anchor_scores = hcrc_result["anchor_scores"].detach().cpu()
                self.last_hcrc_anchor_valid_mask = hcrc_result["anchor_valid_mask"].detach().cpu()
                self.last_hcrc_child_counts = hcrc_result["child_counts"].detach().cpu()
                self.last_hcrc_child_used_counts = hcrc_result["child_used_counts"].detach().cpu()
                self.last_hcrc_empty_anchor_ratio = hcrc_result["empty_anchor_ratio"].detach().cpu()
                self.last_hcrc_child_valid_mask = hcrc_result["child_valid_mask"].detach().cpu()
                self.last_hcrc_child_distance_mean = hcrc_result["child_distance_mean"].detach().cpu()
                self.last_hcrc_skip_reason = None
            else:
                self.last_hcrc_skip_reason = hcrc_result.get("skip_reason")
        elif self.rce_use_hcrc:
            self.last_hcrc_skip_reason = "scale_mode_not_dual"

        if self.rce_use_cross_scale_graph and self.scale_mode == "dual":
            cross_scale_logits, effective_adj = self._compute_cross_scale_logits(
                low_prompt_evidence,
                high_prompt_evidence,
            )
            alpha = self.rce_cross_scale_graph_alpha
            final_logits = final_logits + alpha * cross_scale_logits
            self.last_cross_scale_logits = cross_scale_logits.detach().cpu()
            self.last_cross_scale_alpha = alpha.detach().cpu()
            self.last_cross_scale_adj = effective_adj.detach().cpu()
        else:
            self.last_cross_scale_logits = None
            self.last_cross_scale_alpha = None
            self.last_cross_scale_adj = None

        if self.rce_use_logit_calibration:
            scale = torch.exp(self.rce_logit_scale).clamp(max=100.0)
            final_logits = final_logits * scale + self.rce_class_bias

        ce_loss = self.loss_ce(final_logits, label)
        lh_consistency_loss = ce_loss.new_zeros(())
        low_margin = None
        high_margin = None
        lh_margin_gap = None
        loss = ce_loss
        if self.scale_mode == "dual" and self.rce_use_low_high_consistency_loss:
            low_margin = self._true_vs_wrong_margin(logits_low, label)
            high_margin = self._true_vs_wrong_margin(logits_high, label)
            margin = logits_low.new_tensor(self.rce_lh_consistency_margin)
            low_loss = F.relu(margin - low_margin)
            high_loss = F.relu(margin - high_margin)
            lh_consistency_loss = (low_loss + high_loss).mean()
            lh_margin_gap = torch.abs(low_margin - high_margin)
            loss = ce_loss + self.rce_lh_consistency_lambda * lh_consistency_loss

        self.last_low_prompt_weights = low_prompt_weights.detach().cpu()
        self.last_high_prompt_weights = high_prompt_weights.detach().cpu()
        self.last_low_prompt_evidence = low_prompt_evidence.detach().cpu()
        self.last_high_prompt_evidence = high_prompt_evidence.detach().cpu()
        self.last_low_region_concept_sim = low_region_concept_sim.detach().cpu()
        self.last_high_region_concept_sim = high_region_concept_sim.detach().cpu()
        self.last_low_region_features = low_region_features.detach().cpu()
        self.last_high_region_features = high_region_features.detach().cpu()
        self.last_low_region_features_before_graph = low_region_features_before_graph.detach().cpu()
        self.last_high_region_features_before_graph = high_region_features_before_graph.detach().cpu()
        self.last_low_prompt_features_before_graph = low_prompt_features_before_graph.detach().cpu()
        self.last_high_prompt_features_before_graph = high_prompt_features_before_graph.detach().cpu()
        self.last_low_prompt_features_after_graph = low_prompt_features.detach().cpu()
        self.last_high_prompt_features_after_graph = high_prompt_features.detach().cpu()
        self.last_low_region_attn = (
            low_attn_weights.detach().cpu() if low_attn_weights is not None else None
        )
        self.last_high_region_attn = (
            high_attn_weights.detach().cpu() if high_attn_weights is not None else None
        )
        self.last_low_region_coords = (
            low_region_coords.detach().cpu() if low_region_coords is not None else None
        )
        self.last_high_region_coords = (
            high_region_coords.detach().cpu() if high_region_coords is not None else None
        )
        self.last_low_region_adj = low_region_adj.detach().cpu() if low_region_adj is not None else None
        self.last_high_region_adj = high_region_adj.detach().cpu() if high_region_adj is not None else None
        self.last_low_concept_adj = low_concept_adj.detach().cpu() if low_concept_adj is not None else None
        self.last_high_concept_adj = high_concept_adj.detach().cpu() if high_concept_adj is not None else None
        if self.deg_use_region_graph:
            graph_alpha_cpu = torch.tensor(self.deg_region_graph_alpha, dtype=torch.float32)
            self.last_low_region_graph_alpha = graph_alpha_cpu
            self.last_high_region_graph_alpha = graph_alpha_cpu.clone()
        else:
            self.last_low_region_graph_alpha = None
            self.last_high_region_graph_alpha = None
        if self.deg_use_concept_graph:
            graph_alpha_cpu = torch.tensor(self.deg_concept_graph_alpha, dtype=torch.float32)
            self.last_low_concept_graph_alpha = graph_alpha_cpu
            self.last_high_concept_graph_alpha = graph_alpha_cpu.clone()
        else:
            self.last_low_concept_graph_alpha = None
            self.last_high_concept_graph_alpha = None
        self.last_slide_id = self._detach_slide_id(slide_id)
        self.last_final_logits = final_logits.detach().cpu()
        self.last_low_true_wrong_margin = low_margin.detach().cpu() if low_margin is not None else None
        self.last_high_true_wrong_margin = high_margin.detach().cpu() if high_margin is not None else None
        self.last_lh_margin_gap = lh_margin_gap.detach().cpu() if lh_margin_gap is not None else None
        self.last_lh_consistency_loss = lh_consistency_loss.detach().cpu()
        self.last_total_loss = loss.detach().cpu()
        Y_prob = F.softmax(final_logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        return Y_prob, Y_hat, loss
