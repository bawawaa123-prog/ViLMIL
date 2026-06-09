# coding=utf-8
"""
DEG-MIL skeleton built on top of the current RCE-MIL BiomedCLIP implementation.

Step25 keeps the logits path aligned with RCE-v4-CSG-a01-rq16 and only adds:
- exported low/high region attention weights
- estimated low/high region coordinates from attention-weighted patch coords
"""

from __future__ import absolute_import, division, print_function

import torch
from torch.nn import functional as F

from .model_RCE_MIL_BiomedCLIP import RCE_MIL_BiomedCLIP


class DEG_MIL_BiomedCLIP(RCE_MIL_BiomedCLIP):
    def __init__(
        self,
        config,
        num_classes=2,
        model_path="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ):
        super().__init__(config=config, num_classes=num_classes, model_path=model_path)
        self.last_low_region_attn = None
        self.last_high_region_attn = None
        self.last_low_region_coords = None
        self.last_high_region_coords = None
        self.last_slide_id = None

    def _prepare_patch_features_for_attention(self, patch_features):
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(1)
        elif patch_features.dim() == 3 and patch_features.size(0) == 1 and patch_features.size(1) != 1:
            # Data loader commonly emits bags as [1, num_patches, dim].
            # MultiheadAttention expects [num_patches, batch, dim].
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

        # Expected normalized attention shape after MultiheadAttention:
        # [B, R, N_patches], where B=batch, R=region queries.
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

        # Normalize coords to [B, N_patches, 2].
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

    def _detach_slide_id(self, slide_id):
        if slide_id is None:
            return None
        if torch.is_tensor(slide_id):
            return slide_id.detach().cpu()
        if isinstance(slide_id, tuple):
            return tuple(self._detach_slide_id(item) for item in slide_id)
        if isinstance(slide_id, list):
            return [self._detach_slide_id(item) for item in slide_id]
        return slide_id

    def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        low_patches = x_s.float()
        high_patches = x_l.float()

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

        low_concept_prior = self.low_concept_prior if self.rce_use_concept_prior else None
        high_concept_prior = self.high_concept_prior if self.rce_use_concept_prior else None

        logits_low, low_prompt_weights, low_prompt_evidence, low_region_concept_sim = self._compute_scale_logits(
            low_region_features,
            self.low_prompt_features.to(x_s.device),
            concept_prior=low_concept_prior,
        )
        logits_high, high_prompt_weights, high_prompt_evidence, high_region_concept_sim = self._compute_scale_logits(
            high_region_features,
            self.high_prompt_features.to(x_s.device),
            concept_prior=high_concept_prior,
        )

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
            final_logits = final_logits + alpha * visual_logits
            self.last_visual_residual_alpha = alpha.detach().cpu()
            self.last_low_visual_logits = (
                low_visual_logits.detach().cpu() if low_visual_logits is not None else None
            )
            self.last_high_visual_logits = (
                high_visual_logits.detach().cpu() if high_visual_logits is not None else None
            )
            self.last_visual_logits = visual_logits.detach().cpu()
        else:
            self.last_visual_residual_alpha = None
            self.last_low_visual_logits = None
            self.last_high_visual_logits = None
            self.last_visual_logits = None

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

        self.last_low_prompt_weights = low_prompt_weights.detach().cpu()
        self.last_high_prompt_weights = high_prompt_weights.detach().cpu()
        self.last_low_prompt_evidence = low_prompt_evidence.detach().cpu()
        self.last_high_prompt_evidence = high_prompt_evidence.detach().cpu()
        self.last_low_region_concept_sim = low_region_concept_sim.detach().cpu()
        self.last_high_region_concept_sim = high_region_concept_sim.detach().cpu()
        self.last_low_region_features = low_region_features.detach().cpu()
        self.last_high_region_features = high_region_features.detach().cpu()
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
        self.last_slide_id = self._detach_slide_id(slide_id)
        self.last_final_logits = final_logits.detach().cpu()

        loss = self.loss_ce(final_logits, label)
        Y_prob = F.softmax(final_logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        return Y_prob, Y_hat, loss
