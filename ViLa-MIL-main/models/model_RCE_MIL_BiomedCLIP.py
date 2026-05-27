# coding=utf-8
"""
Minimal Region-Concept Evidence MIL with BiomedCLIP text encoder.
"""

from __future__ import absolute_import, division, print_function

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

        if self.scale_mode not in {"dual", "low_only", "high_only"}:
            raise ValueError(f"Unsupported scale_mode: {self.scale_mode}")

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

        self.last_low_prompt_weights = None
        self.last_high_prompt_weights = None
        self.last_low_prompt_evidence = None
        self.last_high_prompt_evidence = None
        self.last_final_logits = None

        self._initialize_concept_prompt_pool(config)

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
        return logits_scale, prompt_weights, prompt_evidence

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

        logits_low, low_prompt_weights, low_prompt_evidence = self._compute_scale_logits(
            low_region_features,
            self.low_prompt_features.to(x_s.device),
            concept_prior=low_concept_prior,
        )
        logits_high, high_prompt_weights, high_prompt_evidence = self._compute_scale_logits(
            high_region_features,
            self.high_prompt_features.to(x_s.device),
            concept_prior=high_concept_prior,
        )

        if self.scale_mode == "low_only":
            final_logits = logits_low
        elif self.scale_mode == "high_only":
            final_logits = logits_high
        else:
            final_logits = logits_low + logits_high

        if self.rce_use_logit_calibration:
            scale = torch.exp(self.rce_logit_scale).clamp(max=100.0)
            final_logits = final_logits * scale + self.rce_class_bias

        self.last_low_prompt_weights = low_prompt_weights.detach().cpu()
        self.last_high_prompt_weights = high_prompt_weights.detach().cpu()
        self.last_low_prompt_evidence = low_prompt_evidence.detach().cpu()
        self.last_high_prompt_evidence = high_prompt_evidence.detach().cpu()
        self.last_final_logits = final_logits.detach().cpu()

        loss = self.loss_ce(final_logits, label)
        Y_prob = F.softmax(final_logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        return Y_prob, Y_hat, loss
