# coding=utf-8
"""
Minimal Region-Concept Evidence MIL with BiomedCLIP text encoder.
"""

from __future__ import absolute_import, division, print_function

import math
import logging
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from .model_utils import MultiheadAttention
from .model_ViLa_MIL_BiomedCLIP import BiomedCLIPTextEncoder
from utils.prompt_utils import build_concept_prompt_bundle

from open_clip import create_model_from_pretrained, get_tokenizer

logger = logging.getLogger(__name__)


class RCE_MIL_BiomedCLIP(nn.Module):
    def __init__(
        self,
        config,
        num_classes=2,
        model_path="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = int(num_classes)
        self.input_size = 512
        self.region_number = int(config.prototype_number)
        self.peps_tau = float(getattr(config, "peps_tau", 0.1))
        self.scale_mode = str(getattr(config, "scale_mode", "dual"))
        self.rce_use_logit_calibration = bool(getattr(config, "rce_use_logit_calibration", False))
        self.rce_use_concept_prior = bool(getattr(config, "rce_use_concept_prior", False))
        self.rce_concept_prior_strength = float(getattr(config, "rce_concept_prior_strength", 1.0))
        self.rce_use_visual_residual = bool(getattr(config, "rce_use_visual_residual", False))
        self.rce_visual_residual_init = float(getattr(config, "rce_visual_residual_init", 0.1))
        self.rce_use_cross_scale_graph = bool(getattr(config, "rce_use_cross_scale_graph", False))
        self.rce_cross_scale_graph_init = float(getattr(config, "rce_cross_scale_graph_init", 0.05))
        self.rce_cross_scale_graph_norm = str(getattr(config, "rce_cross_scale_graph_norm", "sqrt"))
        self.enable_logit_breakdown_audit = bool(
            getattr(config, "enable_logit_breakdown_audit", False)
        )

        if self.scale_mode not in {"dual", "low_only", "high_only"}:
            raise ValueError(f"Unsupported scale_mode: {self.scale_mode}")
        if self.rce_cross_scale_graph_norm not in {"sqrt", "none"}:
            raise ValueError(
                f"Unsupported rce_cross_scale_graph_norm: {self.rce_cross_scale_graph_norm}"
            )

        if int(getattr(config, "input_size", self.input_size)) != self.input_size:
            raise ValueError(f"RCE_MIL_BiomedCLIP expects input_size=512, got {getattr(config, 'input_size', None)}")

        self.use_concept_prompt_pool = bool(getattr(config, "use_concept_prompt_pool", False))
        self.concept_prompt_path = getattr(config, "concept_prompt_path", None)
        if not self.use_concept_prompt_pool:
            raise ValueError("RCE_MIL_BiomedCLIP requires config.use_concept_prompt_pool=True")
        if not self.concept_prompt_path:
            raise ValueError("RCE_MIL_BiomedCLIP requires a non-empty config.concept_prompt_path")

        print(f"🔬 Loading BiomedCLIP from: {model_path}")
        try:
            biomedclip_model, _ = create_model_from_pretrained(model_path)
            tokenizer = get_tokenizer(model_path)
        except Exception as e:
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            msg = (
                "[Error] Failed to load BiomedCLIP from HuggingFace Hub. "
                "This is usually caused by transient network/proxy/SSL issues or missing local cache.\n"
                f"- model_path: {model_path}\n"
                f"- HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')} (offline={offline})\n"
                "Fix options:\n"
                "1) Ensure the model is fully downloaded into cache (run a one-time warmup download).\n"
                "2) If you already downloaded it, re-run with offline cache only: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
                "3) If you use a proxy, make sure HTTPS proxy is stable and supports TLS properly.\n"
                f"Original error: {e}"
            )
            print(msg)
            raise

        self.text_encoder = BiomedCLIPTextEncoder(
            biomedclip_model,
            n_ctx=16,
            finetune=bool(getattr(config, "finetune_text_encoder", False)),
        )
        self.tokenizer = tokenizer

        self.norm_low = nn.LayerNorm(self.input_size)
        self.norm_high = nn.LayerNorm(self.input_size)
        self.region_attention_low = MultiheadAttention(embed_dim=self.input_size, num_heads=1)
        self.region_attention_high = MultiheadAttention(embed_dim=self.input_size, num_heads=1)

        self.region_queries_low = nn.Parameter(torch.empty(self.region_number, 1, self.input_size))
        self.region_queries_high = nn.Parameter(torch.empty(self.region_number, 1, self.input_size))
        nn.init.normal_(self.region_queries_low, std=0.02)
        nn.init.normal_(self.region_queries_high, std=0.02)

        if self.rce_use_logit_calibration:
            self.rce_logit_scale = nn.Parameter(
                torch.log(torch.tensor(float(getattr(config, "rce_logit_scale_init", 10.0))))
            )
            self.rce_class_bias = nn.Parameter(torch.zeros(self.num_classes))

        if self.rce_use_visual_residual:
            self.low_visual_head = nn.Linear(self.input_size, self.num_classes)
            self.high_visual_head = nn.Linear(self.input_size, self.num_classes)
            init = min(max(self.rce_visual_residual_init, 1e-4), 1.0 - 1e-4)
            self.rce_visual_residual_alpha = nn.Parameter(torch.logit(torch.tensor(init, dtype=torch.float32)))

        self.rce_cross_scale_graph_adj = None
        self.rce_cross_scale_graph_alpha = None

        self.last_low_prompt_weights = None
        self.last_high_prompt_weights = None
        self.last_low_prompt_evidence = None
        self.last_high_prompt_evidence = None
        self.last_final_logits = None
        self.last_visual_residual_alpha = None
        self.last_low_visual_logits = None
        self.last_high_visual_logits = None
        self.last_visual_logits = None
        self.last_low_region_concept_sim = None
        self.last_high_region_concept_sim = None
        self.last_low_region_features = None
        self.last_high_region_features = None
        self.last_cross_scale_logits = None
        self.last_cross_scale_alpha = None
        self.last_cross_scale_adj = None
        self.last_logit_breakdown = None

        self._initialize_concept_prompt_pool(config)
        if self.rce_use_cross_scale_graph and self.scale_mode != "dual":
            logger.warning(
                "rce_use_cross_scale_graph=True but scale_mode=%s; cross-scale graph will be skipped.",
                self.scale_mode,
            )

    def _initialize_concept_prompt_pool(self, config):
        low_prompt_features, high_prompt_features, _, _, _, _ = build_concept_prompt_bundle(
            prompt_json_path=self.concept_prompt_path,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            device=next(self.text_encoder.parameters()).device,
            num_classes=self.num_classes,
            dtype=torch.float32,
            class_names=getattr(config, "class_names", None),
        )
        self.register_buffer("low_prompt_features", low_prompt_features.detach().cpu())
        self.register_buffer("high_prompt_features", high_prompt_features.detach().cpu())
        if self.rce_use_concept_prior:
            self.low_concept_prior = nn.Parameter(
                torch.zeros(self.num_classes, low_prompt_features.shape[1], dtype=torch.float32)
            )
            self.high_concept_prior = nn.Parameter(
                torch.zeros(self.num_classes, high_prompt_features.shape[1], dtype=torch.float32)
            )
        if self.rce_use_cross_scale_graph:
            self.rce_cross_scale_graph_adj = nn.Parameter(
                torch.zeros(
                    self.num_classes,
                    low_prompt_features.shape[1],
                    high_prompt_features.shape[1],
                    dtype=torch.float32,
                )
            )
            self.rce_cross_scale_graph_alpha = nn.Parameter(
                torch.tensor(self.rce_cross_scale_graph_init, dtype=torch.float32)
            )
        print(
            f"[ConceptPromptPool] enabled for RCE_MIL_BiomedCLIP | "
            f"path={self.concept_prompt_path} | low_shape={tuple(low_prompt_features.shape)} | "
            f"high_shape={tuple(high_prompt_features.shape)}"
        )

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def _aggregate_region_features(self, patch_features, region_queries, attention_layer, norm_layer):
        if patch_features.dim() == 2:
            patch_features = patch_features.unsqueeze(1)
        elif patch_features.dim() == 3 and patch_features.size(0) == 1 and patch_features.size(1) != 1:
            # The current data pipeline often provides bags as [1, num_patches, dim].
            # Convert to MultiheadAttention's expected [num_patches, batch, dim].
            patch_features = patch_features.transpose(0, 1).contiguous()
        elif patch_features.dim() != 3:
            raise ValueError(f"Expected patch features rank=2/3, got rank={patch_features.dim()}")
        if patch_features.size(-1) != self.input_size:
            raise ValueError(
                f"Expected patch feature dim={self.input_size}, got {patch_features.size(-1)}"
            )

        batch_size = patch_features.size(1)
        query = region_queries.expand(-1, batch_size, -1)
        region_features, _ = attention_layer(query, patch_features, patch_features)
        region_features = norm_layer(region_features + query)
        return region_features.permute(1, 0, 2).contiguous()

    def _compute_scale_logits(self, region_features, prompt_features, concept_prior=None):
        region_features = F.normalize(region_features.float(), dim=-1)
        prompt_features = F.normalize(prompt_features.float(), dim=-1)

        sim = torch.einsum("brd,cpd->bcrp", region_features, prompt_features)
        prompt_evidence = sim.max(dim=2).values
        tau = max(float(self.peps_tau), 1e-6)
        weight_logits = prompt_evidence / tau
        if self.rce_use_concept_prior and concept_prior is not None:
            weight_logits = weight_logits + self.rce_concept_prior_strength * concept_prior.unsqueeze(0)
        prompt_weights = F.softmax(weight_logits, dim=-1)
        logits_scale = torch.sum(prompt_weights * prompt_evidence, dim=-1)
        return logits_scale, prompt_weights, prompt_evidence, sim

    def _compute_cross_scale_logits(self, low_prompt_evidence, high_prompt_evidence):
        effective_adj = torch.tanh(self.rce_cross_scale_graph_adj)
        cross_scale_logits = torch.einsum(
            "bcl,clh,bch->bc",
            low_prompt_evidence.float(),
            effective_adj,
            high_prompt_evidence.float(),
        )
        if self.rce_cross_scale_graph_norm == "sqrt":
            norm = math.sqrt(float(low_prompt_evidence.size(-1) * high_prompt_evidence.size(-1)))
            cross_scale_logits = cross_scale_logits / max(norm, 1.0)
        return cross_scale_logits, effective_adj

    @staticmethod
    def _detach_cpu(tensor):
        return tensor.detach().cpu() if tensor is not None else None

    @staticmethod
    def _compute_margin_dict(logits, label=None):
        topk = min(2, logits.size(1))
        top_values, top_indices = torch.topk(logits, k=topk, dim=1)
        top1_margin = torch.zeros(
            logits.size(0),
            device=logits.device,
            dtype=logits.dtype,
        )
        if topk > 1:
            top1_margin = top_values[:, 0] - top_values[:, 1]

        if label is None:
            true_class_margin = None
        else:
            label = label.view(-1, 1)
            true_logits = logits.gather(1, label).squeeze(1)
            competitor_logits = logits.clone()
            competitor_logits.scatter_(1, label, float("-inf"))
            alt_logits = competitor_logits.max(dim=1).values
            true_class_margin = true_logits - alt_logits

        return {
            "top1_margin": top1_margin,
            "top1_index": top_indices[:, 0],
            "true_class_margin": true_class_margin,
        }

    def _apply_logit_calibration(self, logits):
        if logits is None:
            return None
        if not self.rce_use_logit_calibration:
            return logits
        scale = torch.exp(self.rce_logit_scale).clamp(max=100.0)
        return logits * scale + self.rce_class_bias

    def _cache_logit_breakdown(
        self,
        label,
        logits_low,
        logits_high,
        logits_low_high,
        concept_only_pre,
        weighted_cross_scale_logits,
        raw_cross_scale_logits,
        weighted_visual_logits,
        raw_visual_logits,
        full_pre,
        full_final,
    ):
        if not self.enable_logit_breakdown_audit:
            self.last_logit_breakdown = None
            return

        low_post = self._apply_logit_calibration(logits_low)
        high_post = self._apply_logit_calibration(logits_high)
        low_high_post = self._apply_logit_calibration(logits_low_high)
        csg_post = self._apply_logit_calibration(weighted_cross_scale_logits)
        concept_post = self._apply_logit_calibration(concept_only_pre)
        visual_post = self._apply_logit_calibration(weighted_visual_logits)
        full_without_visual_post = concept_post

        pred_indices = torch.argmax(full_final, dim=1, keepdim=True)
        low_high_abs = torch.zeros_like(pred_indices, dtype=full_pre.dtype).squeeze(1)
        csg_abs = torch.zeros_like(low_high_abs)
        visual_abs = torch.zeros_like(low_high_abs)
        if logits_low_high is not None:
            low_high_abs = logits_low_high.gather(1, pred_indices).abs().squeeze(1)
        if weighted_cross_scale_logits is not None:
            csg_abs = weighted_cross_scale_logits.gather(1, pred_indices).abs().squeeze(1)
        if weighted_visual_logits is not None:
            visual_abs = weighted_visual_logits.gather(1, pred_indices).abs().squeeze(1)

        total_abs = (low_high_abs + csg_abs + visual_abs).clamp_min(1e-8)
        concept_abs = low_high_abs + csg_abs
        visual_ratio = visual_abs / total_abs
        concept_ratio = concept_abs / total_abs
        csg_ratio = csg_abs / total_abs

        branch_logits_pre = {
            "low_only": logits_low,
            "high_only": logits_high,
            "low_high": logits_low_high,
            "csg_only": weighted_cross_scale_logits,
            "concept_only": concept_only_pre,
            "visual_only": weighted_visual_logits,
            "full_without_visual": concept_only_pre,
            "full": full_pre,
        }
        branch_logits_post = {
            "low_only": low_post,
            "high_only": high_post,
            "low_high": low_high_post,
            "csg_only": csg_post,
            "concept_only": concept_post,
            "visual_only": visual_post,
            "full_without_visual": full_without_visual_post,
            "full": full_final,
        }
        margin_pre = {
            name: self._compute_margin_dict(logits, label)
            for name, logits in branch_logits_pre.items()
            if logits is not None
        }
        margin_post = {
            name: self._compute_margin_dict(logits, label)
            for name, logits in branch_logits_post.items()
            if logits is not None
        }

        self.last_logit_breakdown = {
            "audit_enabled": True,
            "uses_logit_calibration": bool(self.rce_use_logit_calibration),
            "calibration_space": {
                "pre_calibration": "additive branch logits before optional global scale+bias calibration",
                "post_calibration": "branch logits after applying the model's final scale+bias calibration",
            },
            "pre_calibration": {
                "low_evidence_logits": self._detach_cpu(logits_low),
                "high_evidence_logits": self._detach_cpu(logits_high),
                "low_high_evidence_logits": self._detach_cpu(logits_low_high),
                "csg_logits": self._detach_cpu(weighted_cross_scale_logits),
                "csg_logits_raw": self._detach_cpu(raw_cross_scale_logits),
                "concept_only_logits": self._detach_cpu(concept_only_pre),
                "visual_residual_logits": self._detach_cpu(weighted_visual_logits),
                "visual_residual_logits_raw": self._detach_cpu(raw_visual_logits),
                "full_without_visual_logits": self._detach_cpu(concept_only_pre),
                "full_logits": self._detach_cpu(full_pre),
            },
            "post_calibration": {
                "low_evidence_logits": self._detach_cpu(low_post),
                "high_evidence_logits": self._detach_cpu(high_post),
                "low_high_evidence_logits": self._detach_cpu(low_high_post),
                "csg_logits": self._detach_cpu(csg_post),
                "concept_only_logits": self._detach_cpu(concept_post),
                "visual_residual_logits": self._detach_cpu(visual_post),
                "full_without_visual_logits": self._detach_cpu(full_without_visual_post),
                "full_logits": self._detach_cpu(full_final),
            },
            "ratios": {
                "visual_contribution_ratio": self._detach_cpu(visual_ratio),
                "concept_contribution_ratio": self._detach_cpu(concept_ratio),
                "csg_contribution_ratio": self._detach_cpu(csg_ratio),
                "reference_pred_class": self._detach_cpu(pred_indices.squeeze(1)),
            },
            "margins_pre_calibration": {
                name: {key: self._detach_cpu(value) for key, value in payload.items()}
                for name, payload in margin_pre.items()
            },
            "margins_post_calibration": {
                name: {key: self._detach_cpu(value) for key, value in payload.items()}
                for name, payload in margin_post.items()
            },
        }

    def set_logit_breakdown_audit(self, enabled=True):
        self.enable_logit_breakdown_audit = bool(enabled)
        if not self.enable_logit_breakdown_audit:
            self.last_logit_breakdown = None

    def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        del coord_s, coords_l, slide_id

        low_patches = x_s.float()
        high_patches = x_l.float()

        low_region_features = self._aggregate_region_features(
            low_patches,
            self.region_queries_low,
            self.region_attention_low,
            self.norm_low,
        )
        high_region_features = self._aggregate_region_features(
            high_patches,
            self.region_queries_high,
            self.region_attention_high,
            self.norm_high,
        )

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
        raw_visual_logits = None
        weighted_visual_logits = None
        raw_cross_scale_logits = None
        weighted_cross_scale_logits = None

        if self.scale_mode == "low_only":
            final_logits = logits_low
            if self.rce_use_visual_residual:
                low_region_pool = low_region_features.mean(dim=1)
                low_visual_logits = self.low_visual_head(low_region_pool)
                raw_visual_logits = low_visual_logits
        elif self.scale_mode == "high_only":
            final_logits = logits_high
            if self.rce_use_visual_residual:
                high_region_pool = high_region_features.mean(dim=1)
                high_visual_logits = self.high_visual_head(high_region_pool)
                raw_visual_logits = high_visual_logits
        else:
            final_logits = logits_low + logits_high
            if self.rce_use_visual_residual:
                low_region_pool = low_region_features.mean(dim=1)
                high_region_pool = high_region_features.mean(dim=1)
                low_visual_logits = self.low_visual_head(low_region_pool)
                high_visual_logits = self.high_visual_head(high_region_pool)
                raw_visual_logits = low_visual_logits + high_visual_logits

        logits_low_high = final_logits

        if self.rce_use_visual_residual:
            alpha = torch.sigmoid(self.rce_visual_residual_alpha)
            weighted_visual_logits = alpha * raw_visual_logits
            final_logits = final_logits + weighted_visual_logits
            self.last_visual_residual_alpha = alpha.detach().cpu()
            self.last_low_visual_logits = (
                low_visual_logits.detach().cpu() if low_visual_logits is not None else None
            )
            self.last_high_visual_logits = (
                high_visual_logits.detach().cpu() if high_visual_logits is not None else None
            )
            self.last_visual_logits = weighted_visual_logits.detach().cpu()
        else:
            self.last_visual_residual_alpha = None
            self.last_low_visual_logits = None
            self.last_high_visual_logits = None
            self.last_visual_logits = None

        if self.rce_use_cross_scale_graph and self.scale_mode == "dual":
            raw_cross_scale_logits, effective_adj = self._compute_cross_scale_logits(
                low_prompt_evidence,
                high_prompt_evidence,
            )
            alpha = self.rce_cross_scale_graph_alpha
            weighted_cross_scale_logits = alpha * raw_cross_scale_logits
            final_logits = final_logits + weighted_cross_scale_logits
            self.last_cross_scale_logits = weighted_cross_scale_logits.detach().cpu()
            self.last_cross_scale_alpha = alpha.detach().cpu()
            self.last_cross_scale_adj = effective_adj.detach().cpu()
        else:
            self.last_cross_scale_logits = None
            self.last_cross_scale_alpha = None
            self.last_cross_scale_adj = None

        full_pre_calibration_logits = final_logits
        if self.rce_use_logit_calibration:
            final_logits = self._apply_logit_calibration(final_logits)

        self.last_low_prompt_weights = low_prompt_weights.detach().cpu()
        self.last_high_prompt_weights = high_prompt_weights.detach().cpu()
        self.last_low_prompt_evidence = low_prompt_evidence.detach().cpu()
        self.last_high_prompt_evidence = high_prompt_evidence.detach().cpu()
        self.last_low_region_concept_sim = low_region_concept_sim.detach().cpu()
        self.last_high_region_concept_sim = high_region_concept_sim.detach().cpu()
        self.last_low_region_features = low_region_features.detach().cpu()
        self.last_high_region_features = high_region_features.detach().cpu()
        self.last_final_logits = final_logits.detach().cpu()
        self._cache_logit_breakdown(
            label=label,
            logits_low=logits_low,
            logits_high=logits_high,
            logits_low_high=logits_low_high,
            concept_only_pre=logits_low_high + (weighted_cross_scale_logits if weighted_cross_scale_logits is not None else 0.0),
            weighted_cross_scale_logits=weighted_cross_scale_logits,
            raw_cross_scale_logits=raw_cross_scale_logits,
            weighted_visual_logits=weighted_visual_logits,
            raw_visual_logits=raw_visual_logits,
            full_pre=full_pre_calibration_logits,
            full_final=final_logits,
        )

        loss = self.loss_ce(final_logits, label)
        Y_prob = F.softmax(final_logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        return Y_prob, Y_hat, loss
