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
        self.rce_use_residual_constraint = bool(
            getattr(config, "rce_use_residual_constraint", False)
        )
        self.rce_residual_constraint_lambda = float(
            getattr(config, "rce_residual_constraint_lambda", 0.0)
        )
        self.rce_residual_ratio_target = float(getattr(config, "rce_residual_ratio_target", 0.5))
        self.rce_residual_constraint_type = str(
            getattr(config, "rce_residual_constraint_type", "relu_l2")
        )
        self.rce_use_concept_aux_loss = bool(getattr(config, "rce_use_concept_aux_loss", False))
        self.rce_concept_aux_loss_weight = float(
            getattr(config, "rce_concept_aux_loss_weight", 0.0)
        )
        self.rce_residual_ratio_eps = float(getattr(config, "rce_residual_ratio_eps", 1e-6))
        self.rce_residual_ratio_detach = bool(getattr(config, "rce_residual_ratio_detach", False))
        self.rce_use_cross_scale_graph = bool(getattr(config, "rce_use_cross_scale_graph", False))
        self.rce_cross_scale_graph_init = float(getattr(config, "rce_cross_scale_graph_init", 0.05))
        self.rce_cross_scale_graph_norm = str(getattr(config, "rce_cross_scale_graph_norm", "sqrt"))
        self.rce_use_dynamic_csg = bool(getattr(config, "rce_use_dynamic_csg", False))
        self.rce_dynamic_csg_mode = str(getattr(config, "rce_dynamic_csg_mode", "evidence_outer"))
        self.rce_dynamic_csg_alpha_init = float(getattr(config, "rce_dynamic_csg_alpha_init", 0.0))
        self.rce_dynamic_csg_scale = float(getattr(config, "rce_dynamic_csg_scale", 1.0))
        self.rce_dynamic_csg_norm = str(getattr(config, "rce_dynamic_csg_norm", "softmax"))
        self.rce_dynamic_csg_detach_evidence = bool(
            getattr(config, "rce_dynamic_csg_detach_evidence", False)
        )
        self.rce_dynamic_csg_clip = float(getattr(config, "rce_dynamic_csg_clip", 5.0))
        self.rce_use_ccra = bool(getattr(config, "rce_use_ccra", False))
        self.rce_ccra_mode = str(getattr(config, "rce_ccra_mode", "concept_query_residual"))
        self.rce_ccra_alpha_init = float(getattr(config, "rce_ccra_alpha_init", 0.0))
        self.rce_ccra_scale = float(getattr(config, "rce_ccra_scale", 1.0))
        self.rce_ccra_num_queries = int(getattr(config, "rce_ccra_num_queries", 0))
        self.rce_ccra_query_source = str(getattr(config, "rce_ccra_query_source", "prompt_mean"))
        self.rce_ccra_detach_prompt = bool(getattr(config, "rce_ccra_detach_prompt", False))
        self.rce_ccra_norm = str(getattr(config, "rce_ccra_norm", "layernorm"))
        self.rce_ccra_dropout = float(getattr(config, "rce_ccra_dropout", 0.0))
        self.rce_ccra_clip = float(getattr(config, "rce_ccra_clip", 5.0))
        self.enable_logit_breakdown_audit = bool(
            getattr(config, "enable_logit_breakdown_audit", False)
        )

        if self.scale_mode not in {"dual", "low_only", "high_only"}:
            raise ValueError(f"Unsupported scale_mode: {self.scale_mode}")
        if self.rce_cross_scale_graph_norm not in {"sqrt", "none"}:
            raise ValueError(
                f"Unsupported rce_cross_scale_graph_norm: {self.rce_cross_scale_graph_norm}"
            )
        if self.rce_dynamic_csg_mode not in {"evidence_outer"}:
            raise ValueError(f"Unsupported rce_dynamic_csg_mode: {self.rce_dynamic_csg_mode}")
        if self.rce_dynamic_csg_norm not in {"softmax", "l1", "none"}:
            raise ValueError(f"Unsupported rce_dynamic_csg_norm: {self.rce_dynamic_csg_norm}")
        if self.rce_ccra_mode not in {"concept_query_residual"}:
            raise ValueError(f"Unsupported rce_ccra_mode: {self.rce_ccra_mode}")
        if self.rce_ccra_query_source not in {"prompt_mean"}:
            raise ValueError(f"Unsupported rce_ccra_query_source: {self.rce_ccra_query_source}")
        if self.rce_ccra_norm not in {"layernorm", "none"}:
            raise ValueError(f"Unsupported rce_ccra_norm: {self.rce_ccra_norm}")
        if self.rce_residual_constraint_type not in {"relu_l2"}:
            raise ValueError(
                f"Unsupported rce_residual_constraint_type: {self.rce_residual_constraint_type}"
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
        self.rce_dynamic_csg_alpha = None
        self.rce_ccra_alpha = None
        self.ccra_prompt_dropout = None
        self.ccra_norm_low = None
        self.ccra_norm_high = None
        self.ccra_attention_low = None
        self.ccra_attention_high = None

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
        self.last_dynamic_csg_breakdown = None
        self.last_ccra_breakdown = None
        self.last_logit_breakdown = None
        self.last_loss_breakdown = None

        self._initialize_concept_prompt_pool(config)
        if self.rce_use_cross_scale_graph and self.scale_mode != "dual":
            logger.warning(
                "rce_use_cross_scale_graph=True but scale_mode=%s; cross-scale graph will be skipped.",
                self.scale_mode,
            )
        if self.rce_use_ccra:
            self._initialize_ccra_modules()

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
            if self.rce_use_dynamic_csg:
                self.rce_dynamic_csg_alpha = nn.Parameter(
                    torch.tensor(self.rce_dynamic_csg_alpha_init, dtype=torch.float32)
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

    def _initialize_ccra_modules(self):
        self.rce_ccra_alpha = nn.Parameter(
            torch.tensor(self.rce_ccra_alpha_init, dtype=torch.float32)
        )
        self.ccra_prompt_dropout = nn.Dropout(p=float(self.rce_ccra_dropout))
        if self.rce_ccra_norm == "layernorm":
            self.ccra_norm_low = nn.LayerNorm(self.input_size)
            self.ccra_norm_high = nn.LayerNorm(self.input_size)
        self.ccra_attention_low = MultiheadAttention(embed_dim=self.input_size, num_heads=1)
        self.ccra_attention_high = MultiheadAttention(embed_dim=self.input_size, num_heads=1)

    def _build_ccra_prompt_query(self, prompt_features, target_queries):
        prompt_features = prompt_features.float()
        if self.rce_ccra_detach_prompt:
            prompt_features = prompt_features.detach()
        if self.rce_ccra_query_source == "prompt_mean":
            prompt_query = prompt_features.mean(dim=(0, 1), keepdim=True)
        else:
            raise ValueError(f"Unsupported rce_ccra_query_source: {self.rce_ccra_query_source}")
        prompt_query = self.ccra_prompt_dropout(prompt_query)
        if target_queries > 1:
            prompt_query = prompt_query.expand(target_queries, -1, -1).contiguous()
        return prompt_query

    def _apply_ccra_to_scale(
        self,
        patch_features,
        original_region_features,
        prompt_features,
        attention_layer,
        norm_layer,
    ):
        if patch_features.dim() == 2:
            patch_tokens = patch_features.unsqueeze(1)
        elif patch_features.dim() == 3 and patch_features.size(0) == 1 and patch_features.size(1) != 1:
            patch_tokens = patch_features.transpose(0, 1).contiguous()
        elif patch_features.dim() == 3:
            patch_tokens = patch_features
        else:
            raise ValueError(f"Expected patch features rank=2/3, got rank={patch_features.dim()}")

        batch_size = patch_tokens.size(1)
        target_queries = original_region_features.size(1)
        if self.rce_ccra_num_queries > 0:
            target_queries = int(self.rce_ccra_num_queries)
        prompt_query = self._build_ccra_prompt_query(prompt_features.to(patch_tokens.device), target_queries)
        prompt_query = prompt_query.expand(-1, batch_size, -1).contiguous()

        ccra_region, _ = attention_layer(prompt_query, patch_tokens, patch_tokens)
        ccra_region = ccra_region.permute(1, 0, 2).contiguous()
        if ccra_region.size(1) != original_region_features.size(1):
            if ccra_region.size(1) > original_region_features.size(1):
                ccra_region = ccra_region[:, : original_region_features.size(1), :]
            else:
                repeat_factor = math.ceil(original_region_features.size(1) / ccra_region.size(1))
                ccra_region = ccra_region.repeat(1, repeat_factor, 1)[:, : original_region_features.size(1), :]

        scale_factor = self.rce_ccra_alpha * float(self.rce_ccra_scale)
        clip_value = max(float(self.rce_ccra_clip), 1e-6)
        ccra_delta = torch.clamp(scale_factor * ccra_region, min=-clip_value, max=clip_value)
        fused_region = original_region_features + ccra_delta
        if norm_layer is not None:
            fused_region = norm_layer(fused_region)
        return {
            "prompt_query": prompt_query,
            "ccra_region": ccra_region,
            "ccra_delta": ccra_delta,
            "fused_region": fused_region,
        }

    def _build_ccra_breakdown(
        self,
        low_patch_features,
        high_patch_features,
        low_prompt_features,
        high_prompt_features,
        low_original_region_features,
        high_original_region_features,
        low_ccra_payload,
        high_ccra_payload,
    ):
        if not self.rce_use_ccra or low_ccra_payload is None or high_ccra_payload is None:
            self.last_ccra_breakdown = {
                "ccra_enabled": False,
                "ccra_mode": self.rce_ccra_mode,
                "ccra_alpha": None if self.rce_ccra_alpha is None else float(self.rce_ccra_alpha.detach().item()),
                "ccra_scale": float(self.rce_ccra_scale),
                "ccra_query_source": self.rce_ccra_query_source,
                "ccra_norm": self.rce_ccra_norm,
                "low_ccra_delta_abs_mean": None,
                "high_ccra_delta_abs_mean": None,
                "low_original_region_norm": None,
                "high_original_region_norm": None,
                "low_fused_region_norm": None,
                "high_fused_region_norm": None,
                "low_ccra_region_norm": None,
                "high_ccra_region_norm": None,
                "low_original_region_shape": None,
                "high_original_region_shape": None,
                "low_ccra_region_shape": None,
                "high_ccra_region_shape": None,
                "low_fused_region_shape": None,
                "high_fused_region_shape": None,
                "low_prompt_feature_shape": None,
                "high_prompt_feature_shape": None,
                "low_patch_feature_shape": None,
                "high_patch_feature_shape": None,
            }
            return

        def _norm_mean(tensor):
            return float(torch.norm(tensor.float(), dim=-1).mean().detach().item())

        def _shape_list(tensor):
            return list(tensor.shape) if tensor is not None else None

        self.last_ccra_breakdown = {
            "ccra_enabled": True,
            "ccra_mode": self.rce_ccra_mode,
            "ccra_alpha": None if self.rce_ccra_alpha is None else float(self.rce_ccra_alpha.detach().item()),
            "ccra_scale": float(self.rce_ccra_scale),
            "ccra_query_source": self.rce_ccra_query_source,
            "ccra_norm": self.rce_ccra_norm,
            "low_ccra_delta_abs_mean": float(low_ccra_payload["ccra_delta"].abs().mean().detach().item()),
            "high_ccra_delta_abs_mean": float(high_ccra_payload["ccra_delta"].abs().mean().detach().item()),
            "low_original_region_norm": _norm_mean(low_original_region_features),
            "high_original_region_norm": _norm_mean(high_original_region_features),
            "low_fused_region_norm": _norm_mean(low_ccra_payload["fused_region"]),
            "high_fused_region_norm": _norm_mean(high_ccra_payload["fused_region"]),
            "low_ccra_region_norm": _norm_mean(low_ccra_payload["ccra_region"]),
            "high_ccra_region_norm": _norm_mean(high_ccra_payload["ccra_region"]),
            "low_original_region_shape": _shape_list(low_original_region_features),
            "high_original_region_shape": _shape_list(high_original_region_features),
            "low_ccra_region_shape": _shape_list(low_ccra_payload["ccra_region"]),
            "high_ccra_region_shape": _shape_list(high_ccra_payload["ccra_region"]),
            "low_fused_region_shape": _shape_list(low_ccra_payload["fused_region"]),
            "high_fused_region_shape": _shape_list(high_ccra_payload["fused_region"]),
            "low_prompt_feature_shape": _shape_list(low_prompt_features),
            "high_prompt_feature_shape": _shape_list(high_prompt_features),
            "low_patch_feature_shape": _shape_list(low_patch_features),
            "high_patch_feature_shape": _shape_list(high_patch_features),
        }

    @staticmethod
    def _tensor_stats(tensor):
        if tensor is None:
            return None, None
        tensor = tensor.float()
        return float(tensor.mean().detach().item()), float(tensor.std(unbiased=False).detach().item())

    def _build_dynamic_csg_delta(self, low_prompt_evidence, high_prompt_evidence):
        low_basis = low_prompt_evidence.float()
        high_basis = high_prompt_evidence.float()
        if self.rce_dynamic_csg_detach_evidence:
            low_basis = low_basis.detach()
            high_basis = high_basis.detach()

        if self.rce_dynamic_csg_mode == "evidence_outer":
            raw_delta = torch.einsum("bcl,bch->bclh", low_basis, high_basis)
        else:
            raise ValueError(f"Unsupported rce_dynamic_csg_mode: {self.rce_dynamic_csg_mode}")

        scaled_delta = raw_delta * float(self.rce_dynamic_csg_scale)
        if self.rce_dynamic_csg_norm == "softmax":
            delta = F.softmax(scaled_delta.flatten(2), dim=-1).view_as(raw_delta)
        elif self.rce_dynamic_csg_norm == "l1":
            denom = scaled_delta.abs().flatten(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            delta = (scaled_delta.flatten(2) / denom).view_as(raw_delta)
        else:
            delta = scaled_delta

        clip_value = max(float(self.rce_dynamic_csg_clip), 1e-6)
        return delta.clamp(min=-clip_value, max=clip_value)

    def _compute_cross_scale_logits(self, low_prompt_evidence, high_prompt_evidence):
        static_adj = torch.tanh(self.rce_cross_scale_graph_adj)
        static_cross_scale_logits = torch.einsum(
            "bcl,clh,bch->bc",
            low_prompt_evidence.float(),
            static_adj,
            high_prompt_evidence.float(),
        )
        effective_adj = static_adj
        dynamic_cross_scale_logits = static_cross_scale_logits
        dynamic_delta = None
        dynamic_enabled = bool(
            self.rce_use_cross_scale_graph
            and self.rce_use_dynamic_csg
            and self.rce_dynamic_csg_alpha is not None
        )

        if dynamic_enabled:
            dynamic_delta = self._build_dynamic_csg_delta(low_prompt_evidence, high_prompt_evidence)
            dynamic_adj = static_adj.unsqueeze(0) + self.rce_dynamic_csg_alpha * dynamic_delta
            clip_value = max(float(self.rce_dynamic_csg_clip), 1e-6)
            dynamic_adj = dynamic_adj.clamp(min=-clip_value, max=clip_value)
            dynamic_cross_scale_logits = torch.einsum(
                "bcl,bclh,bch->bc",
                low_prompt_evidence.float(),
                dynamic_adj,
                high_prompt_evidence.float(),
            )
            effective_adj = dynamic_adj

        if self.rce_cross_scale_graph_norm == "sqrt":
            norm = math.sqrt(float(low_prompt_evidence.size(-1) * high_prompt_evidence.size(-1)))
            static_cross_scale_logits = static_cross_scale_logits / max(norm, 1.0)
            dynamic_cross_scale_logits = dynamic_cross_scale_logits / max(norm, 1.0)

        static_logits_mean, _ = self._tensor_stats(static_cross_scale_logits)
        dynamic_logits_mean, _ = self._tensor_stats(dynamic_cross_scale_logits)
        delta_mean, delta_std = self._tensor_stats(dynamic_delta)
        adj_mean, adj_std = self._tensor_stats(effective_adj)
        logits_delta = dynamic_cross_scale_logits - static_cross_scale_logits
        logits_delta_mean, _ = self._tensor_stats(logits_delta if dynamic_enabled else None)
        logits_delta_abs_mean = (
            float(logits_delta.abs().mean().detach().item()) if dynamic_enabled else None
        )

        breakdown = {
            "dynamic_csg_enabled": dynamic_enabled,
            "dynamic_csg_alpha": None
            if self.rce_dynamic_csg_alpha is None
            else float(self.rce_dynamic_csg_alpha.detach().item()),
            "dynamic_csg_mode": self.rce_dynamic_csg_mode,
            "dynamic_csg_scale": float(self.rce_dynamic_csg_scale),
            "dynamic_csg_norm": self.rce_dynamic_csg_norm,
            "dynamic_csg_detach_evidence": bool(self.rce_dynamic_csg_detach_evidence),
            "dynamic_csg_clip": float(self.rce_dynamic_csg_clip),
            "dynamic_delta_mean": delta_mean,
            "dynamic_delta_std": delta_std,
            "dynamic_adj_mean": adj_mean,
            "dynamic_adj_std": adj_std,
            "static_csg_logits_mean": static_logits_mean,
            "dynamic_csg_logits_mean": dynamic_logits_mean,
            "csg_logits_delta_mean": logits_delta_mean,
            "csg_logits_delta_abs_mean": logits_delta_abs_mean,
        }
        selected_logits = dynamic_cross_scale_logits if dynamic_enabled else static_cross_scale_logits
        return selected_logits, effective_adj, breakdown

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

    def _compute_residual_constraint_loss(self, concept_logits, visual_residual_logits):
        if visual_residual_logits is None:
            return None, None

        concept_norm_input = concept_logits
        visual_norm_input = visual_residual_logits
        if self.rce_residual_ratio_detach:
            concept_norm_input = concept_norm_input.detach()
            visual_norm_input = visual_norm_input.detach()

        concept_norm = torch.norm(concept_norm_input, dim=1)
        visual_norm = torch.norm(visual_norm_input, dim=1)
        ratio = visual_norm / (concept_norm + visual_norm + self.rce_residual_ratio_eps)

        if self.rce_residual_constraint_type == "relu_l2":
            penalty = torch.relu(ratio - self.rce_residual_ratio_target)
            constraint_loss = torch.mean(penalty.pow(2))
        else:
            raise ValueError(
                f"Unsupported rce_residual_constraint_type: {self.rce_residual_constraint_type}"
            )

        return constraint_loss, ratio

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
        loss_breakdown,
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
            "residual_constraint_enabled": bool(
                loss_breakdown.get("residual_constraint_enabled", False)
            ),
            "concept_aux_enabled": bool(loss_breakdown.get("concept_aux_enabled", False)),
            "visual_ratio_for_loss": self._detach_cpu(loss_breakdown.get("visual_ratio")),
            "residual_constraint_loss": self._detach_cpu(
                loss_breakdown.get("residual_constraint_loss_tensor")
            ),
            "concept_aux_loss": self._detach_cpu(loss_breakdown.get("concept_aux_loss_tensor")),
            "total_loss": self._detach_cpu(loss_breakdown.get("total_loss_tensor")),
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
            "dynamic_csg": self.last_dynamic_csg_breakdown,
        }

    def set_logit_breakdown_audit(self, enabled=True):
        self.enable_logit_breakdown_audit = bool(enabled)
        if not self.enable_logit_breakdown_audit:
            self.last_logit_breakdown = None

    def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        del coord_s, coords_l, slide_id

        low_patches = x_s.float()
        high_patches = x_l.float()
        low_prompt_features = self.low_prompt_features.to(x_s.device)
        high_prompt_features = self.high_prompt_features.to(x_s.device)

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
        low_original_region_features = low_region_features
        high_original_region_features = high_region_features

        low_ccra_payload = None
        high_ccra_payload = None
        if self.rce_use_ccra:
            low_ccra_payload = self._apply_ccra_to_scale(
                low_patches,
                low_region_features,
                low_prompt_features,
                self.ccra_attention_low,
                self.ccra_norm_low,
            )
            high_ccra_payload = self._apply_ccra_to_scale(
                high_patches,
                high_region_features,
                high_prompt_features,
                self.ccra_attention_high,
                self.ccra_norm_high,
            )
            low_region_features = low_ccra_payload["fused_region"]
            high_region_features = high_ccra_payload["fused_region"]
        self._build_ccra_breakdown(
            low_patch_features=low_patches,
            high_patch_features=high_patches,
            low_prompt_features=low_prompt_features,
            high_prompt_features=high_prompt_features,
            low_original_region_features=low_original_region_features,
            high_original_region_features=high_original_region_features,
            low_ccra_payload=low_ccra_payload,
            high_ccra_payload=high_ccra_payload,
        )

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
            raw_cross_scale_logits, effective_adj, dynamic_csg_breakdown = self._compute_cross_scale_logits(
                low_prompt_evidence,
                high_prompt_evidence,
            )
            alpha = self.rce_cross_scale_graph_alpha
            weighted_cross_scale_logits = alpha * raw_cross_scale_logits
            final_logits = final_logits + weighted_cross_scale_logits
            self.last_dynamic_csg_breakdown = dynamic_csg_breakdown
            self.last_cross_scale_logits = weighted_cross_scale_logits.detach().cpu()
            self.last_cross_scale_alpha = alpha.detach().cpu()
            self.last_cross_scale_adj = effective_adj.detach().cpu()
        else:
            self.last_dynamic_csg_breakdown = {
                "dynamic_csg_enabled": False,
                "dynamic_csg_alpha": None
                if self.rce_dynamic_csg_alpha is None
                else float(self.rce_dynamic_csg_alpha.detach().item()),
                "dynamic_csg_mode": self.rce_dynamic_csg_mode,
                "dynamic_delta_mean": None,
                "dynamic_delta_std": None,
                "dynamic_adj_mean": None,
                "dynamic_adj_std": None,
                "static_csg_logits_mean": None,
                "dynamic_csg_logits_mean": None,
                "csg_logits_delta_mean": None,
                "csg_logits_delta_abs_mean": None,
            }
            self.last_cross_scale_logits = None
            self.last_cross_scale_alpha = None
            self.last_cross_scale_adj = None

        full_pre_calibration_logits = final_logits
        if self.rce_use_logit_calibration:
            final_logits = self._apply_logit_calibration(final_logits)

        concept_logits_pre = logits_low_high + (
            weighted_cross_scale_logits if weighted_cross_scale_logits is not None else 0.0
        )
        concept_logits_final = self._apply_logit_calibration(concept_logits_pre)

        ce_loss = self.loss_ce(final_logits, label)
        residual_constraint_enabled = bool(
            self.rce_use_residual_constraint
            and self.rce_residual_constraint_lambda > 0.0
            and weighted_visual_logits is not None
            and label is not None
        )
        concept_aux_enabled = bool(
            self.rce_use_concept_aux_loss
            and self.rce_concept_aux_loss_weight > 0.0
            and label is not None
        )

        residual_constraint_loss = final_logits.new_zeros(())
        visual_ratio = None
        if residual_constraint_enabled:
            residual_constraint_loss, visual_ratio = self._compute_residual_constraint_loss(
                concept_logits=concept_logits_pre,
                visual_residual_logits=weighted_visual_logits,
            )

        concept_aux_loss = final_logits.new_zeros(())
        if concept_aux_enabled:
            concept_aux_loss = self.loss_ce(concept_logits_final, label)

        total_loss = ce_loss
        if residual_constraint_enabled:
            total_loss = total_loss + self.rce_residual_constraint_lambda * residual_constraint_loss
        if concept_aux_enabled:
            total_loss = total_loss + self.rce_concept_aux_loss_weight * concept_aux_loss

        visual_ratio_mean = (
            visual_ratio.mean() if visual_ratio is not None else final_logits.new_tensor(math.nan)
        )
        visual_ratio_median = (
            visual_ratio.median() if visual_ratio is not None else final_logits.new_tensor(math.nan)
        )

        self.last_loss_breakdown = {
            "ce_loss": float(ce_loss.detach().item()),
            "residual_constraint_enabled": residual_constraint_enabled,
            "residual_constraint_lambda": float(self.rce_residual_constraint_lambda),
            "residual_constraint_loss": float(residual_constraint_loss.detach().item()),
            "concept_aux_enabled": concept_aux_enabled,
            "concept_aux_loss_weight": float(self.rce_concept_aux_loss_weight),
            "concept_aux_loss": float(concept_aux_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "visual_ratio_mean": None
            if visual_ratio is None
            else float(visual_ratio_mean.detach().item()),
            "visual_ratio_median": None
            if visual_ratio is None
            else float(visual_ratio_median.detach().item()),
        }

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
            concept_only_pre=concept_logits_pre,
            weighted_cross_scale_logits=weighted_cross_scale_logits,
            raw_cross_scale_logits=raw_cross_scale_logits,
            weighted_visual_logits=weighted_visual_logits,
            raw_visual_logits=raw_visual_logits,
            full_pre=full_pre_calibration_logits,
            full_final=final_logits,
            loss_breakdown={
                "residual_constraint_enabled": residual_constraint_enabled,
                "concept_aux_enabled": concept_aux_enabled,
                "visual_ratio": visual_ratio_mean if visual_ratio is not None else None,
                "residual_constraint_loss_tensor": residual_constraint_loss,
                "concept_aux_loss_tensor": concept_aux_loss,
                "total_loss_tensor": total_loss,
            },
        )
        Y_prob = F.softmax(final_logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        return Y_prob, Y_hat, total_loss
