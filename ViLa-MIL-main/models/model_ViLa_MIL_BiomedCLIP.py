# coding=utf-8
"""
ViLa-MIL with BiomedCLIP
使用BiomedCLIP替换原始CLIP的图像和文本编码器
"""

from __future__ import absolute_import, division, print_function
import inspect
import logging
import math
import os
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from open_clip import create_model_from_pretrained, get_tokenizer

from .model_utils import MultiheadAttention
from utils.prompt_utils import build_concept_prompt_bundle, build_concept_prompt_tensors, build_concept_text_features

logger = logging.getLogger(__name__)


class BiomedCLIPTextEncoder(nn.Module):
    """
    BiomedCLIP文本编码器封装
    使用PubMedBERT作为backbone
    """

    def __init__(self, biomedclip_model, *, n_ctx: int = 16, finetune: bool = False):
        super().__init__()
        self.model = biomedclip_model
        self.n_ctx = int(n_ctx)
        self.finetune = bool(finetune)
        self._warned_embed_fallback = False
        self.fallback_count = 0
        self.fallback_last_error = None

    def forward(
        self,
        text_tokens,
        prompt_embeddings: torch.Tensor | None = None,
        eos_indices: torch.Tensor | None = None,
    ):
        """
        前向传播函数，用于处理输入的文本token并提取文本特征
        参数:
            text_tokens: tokenized text [batch, seq_len]
            prompt_embeddings: learned prompt embeddings [batch, seq_len, dim] (optional)
            eos_indices: indices of end token in the prompt_embeddings sequence [batch] (optional)
        返回:
            text_features: [batch, 512]
        """
        if prompt_embeddings is not None:
            try:
                text_model = self.model.text if hasattr(self.model, "text") else self.model

                if all(
                    hasattr(text_model, attr)
                    for attr in ["positional_embedding", "transformer", "ln_final", "text_projection"]
                ):
                    x = prompt_embeddings
                    x = x + text_model.positional_embedding.to(x.dtype)
                    x = x.permute(1, 0, 2)
                    x = text_model.transformer(x)
                    x = x.permute(1, 0, 2)
                    x = text_model.ln_final(x)

                    if eos_indices is None:
                        attn_mask = text_tokens != 0
                        eos_indices = attn_mask.long().sum(dim=1) - 1
                        eos_indices = torch.clamp(eos_indices + self.n_ctx, min=0, max=x.shape[1] - 1)

                    x = x[torch.arange(x.shape[0], device=x.device), eos_indices]
                    proj = text_model.text_projection
                    if isinstance(proj, (torch.Tensor, nn.Parameter)):
                        return x @ proj
                    if isinstance(proj, nn.Module):
                        return proj(x)
                    raise TypeError(f"Unsupported text_projection type: {type(proj)}")

                transformer = getattr(text_model, "transformer", None)
                if transformer is not None and hasattr(transformer, "forward"):
                    try:
                        sig = inspect.signature(transformer.forward)
                    except Exception:
                        sig = None

                    if sig is not None and "inputs_embeds" in sig.parameters:
                        token_mask = text_tokens != 0
                        bsz, seq_len = text_tokens.shape
                        if prompt_embeddings.shape[1] != seq_len:
                            seq_len = prompt_embeddings.shape[1]
                        prefix_mask = token_mask[:, :1]
                        ctx_mask = torch.ones((bsz, self.n_ctx), device=text_tokens.device, dtype=token_mask.dtype)
                        suffix_keep = max(int(text_tokens.shape[1]) - 1 - self.n_ctx, 0)
                        suffix_mask = token_mask[:, 1 : 1 + suffix_keep]
                        attn_mask = torch.cat([prefix_mask, ctx_mask, suffix_mask], dim=1)
                        attn_mask = attn_mask[:, : prompt_embeddings.shape[1]]

                        out = transformer(
                            inputs_embeds=prompt_embeddings,
                            attention_mask=attn_mask,
                            return_dict=True,
                        )
                        hidden = getattr(out, "last_hidden_state", None)
                        if hidden is None and isinstance(out, (tuple, list)) and len(out) > 0:
                            hidden = out[0]
                        if hidden is None:
                            raise RuntimeError("HF transformer output missing last_hidden_state")

                        pooled = hidden[:, 0]
                        if hasattr(text_model, "ln_final"):
                            pooled = text_model.ln_final(pooled)
                        proj = None
                        for attr in ["proj", "text_projection"]:
                            if hasattr(text_model, attr):
                                proj = getattr(text_model, attr)
                                break
                        if proj is None and hasattr(self.model, "text_projection"):
                            proj = getattr(self.model, "text_projection")
                        if proj is not None:
                            if isinstance(proj, (torch.Tensor, nn.Parameter)):
                                pooled = pooled @ proj
                            elif isinstance(proj, nn.Module):
                                pooled = proj(pooled)
                            else:
                                raise TypeError(f"Unsupported projection type: {type(proj)}")
                        return pooled

                raise RuntimeError("Unsupported BiomedCLIP text tower for prompt embeddings")

            except Exception as e:
                self.fallback_count += 1
                self.fallback_last_error = str(e)
                if not self._warned_embed_fallback:
                    msg = (
                        "[BiomedCLIPTextEncoder] prompt-embedding path failed, "
                        "falling back to encode_text(tokens). "
                        f"Error: {e}"
                    )
                    print(msg)
                    logger.warning(msg)
                    self._warned_embed_fallback = True

        if self.finetune:
            return self.model.encode_text(text_tokens)
        with torch.no_grad():
            return self.model.encode_text(text_tokens)


class BiomedCLIPPromptLearner(nn.Module):
    """
    可学习的提示词模块(适配BiomedCLIP)
    """

    def __init__(self, classnames, biomedclip_model, tokenizer, n_ctx=16):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        self._pad_id = 0

        text_model = biomedclip_model.text if hasattr(biomedclip_model, "text") else biomedclip_model

        if hasattr(text_model, "transformer"):
            token_embedding_layer = text_model.transformer.embeddings.word_embeddings
            ctx_dim = token_embedding_layer.embedding_dim
        elif hasattr(text_model, "token_embedding"):
            with torch.no_grad():
                dummy_tokens = tokenizer(["test"]).to(next(biomedclip_model.parameters()).device)
                dummy_emb = text_model.token_embedding(dummy_tokens)
                ctx_dim = dummy_emb.shape[-1]
        else:
            ctx_dim = 512

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        self.classnames = classnames

        prompts = [f"a histopathology image of {name}" for name in classnames]
        self.tokenized_prompts = tokenizer(prompts)

        with torch.no_grad():
            device = next(biomedclip_model.parameters()).device
            token_ids = self.tokenized_prompts.to(device)

            if hasattr(text_model, "transformer"):
                embedding = text_model.transformer.embeddings.word_embeddings(token_ids)
            elif hasattr(text_model, "token_embedding"):
                embedding = text_model.token_embedding(token_ids)
            else:
                raise AttributeError("Cannot find token embedding layer in BiomedCLIP model")

        seq_len = embedding.shape[1]
        suffix_keep = max(seq_len - 1 - n_ctx, 0)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 : 1 + suffix_keep, :])

        with torch.no_grad():
            non_pad = token_ids != self._pad_id
            last_idx = non_pad.long().sum(dim=1) - 1
            eos_idx = torch.clamp(last_idx + n_ctx, min=0, max=seq_len - 1)
        self.register_buffer("eos_indices", eos_idx)

    def forward(self):
        """
        生成可学习的提示词嵌入
        返回: prompts [n_cls, seq_len, dim]
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """截断正态分布初始化"""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b]", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ViLa_MIL_BiomedCLIP(nn.Module):
    """
    ViLa-MIL模型(BiomedCLIP版本)
    """

    def __init__(
        self,
        config,
        num_classes=2,
        model_path="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

        self.L = 512
        self.D = config.hidden_size
        self.K = 1

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

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

        finetune_text = bool(getattr(config, "finetune_text_encoder", False))
        self.text_encoder = BiomedCLIPTextEncoder(biomedclip_model, n_ctx=16, finetune=finetune_text)
        self.tokenizer = tokenizer

        self.prompt_learner = BiomedCLIPPromptLearner(
            config.text_prompt,
            biomedclip_model,
            tokenizer,
        )
        self.use_concept_prompt_pool = bool(getattr(config, "use_concept_prompt_pool", False))
        self.concept_prompt_path = getattr(config, "concept_prompt_path", None)
        self.prompt_ensemble_mode = str(getattr(config, "prompt_ensemble_mode", "embedding_mean"))
        self.use_dynamic_prompt_gate = bool(getattr(config, "use_dynamic_prompt_gate", False))
        self.use_peps = self.prompt_ensemble_mode == "peps"
        if self.prompt_ensemble_mode == "dynamic_gate":
            self.use_dynamic_prompt_gate = True
        self.dynamic_gate_hidden_dim = int(getattr(config, "dynamic_gate_hidden_dim", 256))
        self.dynamic_gate_residual_mean = bool(getattr(config, "dynamic_gate_residual_mean", False))
        self.prompt_dropout = float(getattr(config, "prompt_dropout", 0.0))
        self.peps_topk = int(getattr(config, "peps_topk", 3))
        self.peps_tau = float(getattr(config, "peps_tau", 0.1))
        self.save_peps_weights = bool(getattr(config, "save_peps_weights", False))

        self.norm = nn.LayerNorm(self.L)
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.dynamic_gate_low = None
        self.dynamic_gate_high = None
        self.dynamic_gate_gamma = None
        self.concept_prompt_texts_low = None
        self.concept_prompt_texts_high = None
        self.concept_prompt_metadata_low = None
        self.concept_prompt_metadata_high = None
        if self.use_dynamic_prompt_gate:
            gate_input_dim = self.L * 4
            self.dynamic_gate_low = nn.Sequential(
                nn.Linear(gate_input_dim, self.dynamic_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.dynamic_gate_hidden_dim, 1),
            )
            self.dynamic_gate_high = nn.Sequential(
                nn.Linear(gate_input_dim, self.dynamic_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.dynamic_gate_hidden_dim, 1),
            )
            self.dynamic_gate_gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.learnable_image_center = nn.Parameter(torch.Tensor(config.prototype_number, 1, self.L))
        trunc_normal_(self.learnable_image_center, std=0.02)

        self._configure_biomedclip_finetune(config)
        self._initialize_concept_prompt_pool(config)

    def _configure_biomedclip_finetune(self, config):
        finetune_text = bool(getattr(config, "finetune_text_encoder", False))
        mode = str(getattr(config, "text_finetune_mode", "proj"))
        last_n = int(getattr(config, "text_unfreeze_last_n", 2))

        for p in self.text_encoder.parameters():
            p.requires_grad = False

        text_clip = self.text_encoder.model
        if hasattr(text_clip, "visual"):
            for p in text_clip.visual.parameters():
                p.requires_grad = False

        if not finetune_text:
            return

        text_model = text_clip.text if hasattr(text_clip, "text") else text_clip

        def _unfreeze_module(module):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = True

        def _unfreeze_obj(obj):
            if obj is None:
                return
            if isinstance(obj, nn.Module):
                _unfreeze_module(obj)
            elif isinstance(obj, (torch.Tensor, nn.Parameter)):
                obj.requires_grad = True

        self.text_encoder.finetune = True

        if mode == "proj":
            for attr in ["proj", "text_projection", "ln_final"]:
                if hasattr(text_model, attr):
                    _unfreeze_obj(getattr(text_model, attr))
            if hasattr(text_clip, "text_projection"):
                _unfreeze_obj(getattr(text_clip, "text_projection"))
            return

        if mode == "last":
            for attr in ["proj", "text_projection", "ln_final"]:
                if hasattr(text_model, attr):
                    _unfreeze_obj(getattr(text_model, attr))

            transformer = getattr(text_model, "transformer", None)
            encoder = getattr(transformer, "encoder", None) if transformer is not None else None
            layers = getattr(encoder, "layer", None) if encoder is not None else None
            if layers is not None and hasattr(layers, "__len__"):
                n_layers = len(layers)
                n = max(0, min(int(last_n), n_layers))
                for i in range(n_layers - n, n_layers):
                    _unfreeze_module(layers[i])
            else:
                mode = "full"

        if mode == "full":
            _unfreeze_module(text_model)
            if hasattr(text_clip, "text_projection"):
                _unfreeze_obj(getattr(text_clip, "text_projection"))

    def _encode_prompt_text_features(self, device):
        if self.use_concept_prompt_pool and hasattr(self, "concept_text_low") and hasattr(self, "concept_text_high"):
            return self.concept_text_low.to(device), self.concept_text_high.to(device)

        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
        prompt_embeddings = self.prompt_learner().to(device)
        eos_indices = self.prompt_learner.eos_indices.to(device)
        text_features = self.text_encoder(
            tokenized_prompts,
            prompt_embeddings=prompt_embeddings,
            eos_indices=eos_indices,
        )
        return text_features[: self.num_classes], text_features[self.num_classes :]

    def _contextualize_text_features(self, text_features, image_context):
        if text_features.dim() == 2:
            text_context_features, _ = self.cross_attention_2(
                text_features.unsqueeze(1),
                image_context,
                image_context,
            )
            return text_context_features.squeeze(1) + text_features

        if text_features.dim() == 3:
            num_classes, num_prompts, dim = text_features.shape
            flat_text_features = text_features.reshape(num_classes * num_prompts, dim)
            text_context_features, _ = self.cross_attention_2(
                flat_text_features.unsqueeze(1),
                image_context,
                image_context,
            )
            text_context_features = text_context_features.squeeze(1).reshape(num_classes, num_prompts, dim)
            return text_context_features + text_features

        raise ValueError(f"Unsupported text feature rank: {text_features.dim()}")

    def _contextualize_dynamic_text_features(self, text_features, image_context):
        if text_features.dim() != 3:
            raise ValueError(f"Dynamic text features must be [batch, classes, dim], got rank={text_features.dim()}")

        batch_size, _, _ = text_features.shape
        if image_context.dim() == 2:
            image_context = image_context.unsqueeze(1)
        if image_context.dim() != 3:
            raise ValueError(f"Image context must be rank-2/3, got rank={image_context.dim()}")
        if image_context.size(1) == 1 and batch_size > 1:
            image_context = image_context.expand(-1, batch_size, -1)
        if image_context.size(1) != batch_size:
            raise ValueError(
                f"Dynamic text/image context batch mismatch: text batch={batch_size}, "
                f"context batch={image_context.size(1)}"
            )

        query = text_features.permute(1, 0, 2)
        text_context_features, _ = self.cross_attention_2(query, image_context, image_context)
        return text_context_features.permute(1, 0, 2) + text_features

    def _compute_scale_logits(self, image_features, text_features):
        if text_features.dim() == 2:
            return image_features @ text_features.T

        if text_features.dim() == 3:
            image_vector = F.normalize(image_features.float(), dim=-1).squeeze(0)
            prompt_features = F.normalize(text_features.float(), dim=-1)
            prompt_logits = torch.einsum("cpd,d->cp", prompt_features, image_vector)
            return prompt_logits.mean(dim=1).unsqueeze(0)

        raise ValueError(f"Unsupported text feature rank for logits: {text_features.dim()}")

    def _compute_dynamic_scale_logits(self, image_features, text_features):
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() != 3:
            raise ValueError(f"Dynamic scale logits expect [batch, classes, dim], got rank={text_features.dim()}")
        if image_features.size(0) != text_features.size(0):
            raise ValueError(
                f"Dynamic logits batch mismatch: image batch={image_features.size(0)}, "
                f"text batch={text_features.size(0)}"
            )
        return torch.einsum("bd,bcd->bc", image_features.float(), text_features.float())

    def _prototype_tensor_from_components(self, components):
        if components.dim() != 3:
            raise ValueError(f"Prototype components must be rank-3, got rank={components.dim()}")
        return components.permute(1, 0, 2).contiguous().float()

    def _apply_prompt_dropout(self, weights):
        if (not self.training) or self.prompt_dropout <= 0.0 or weights.size(-1) <= 1:
            return weights

        keep_mask = (torch.rand_like(weights) >= self.prompt_dropout).to(weights.dtype)
        dropped_weights = weights * keep_mask
        keep_sum = dropped_weights.sum(dim=-1, keepdim=True)
        all_dropped = keep_sum <= 0
        if all_dropped.any():
            fallback_mask = torch.zeros_like(dropped_weights)
            fallback_mask[..., 0] = 1.0
            dropped_weights = torch.where(all_dropped, fallback_mask, dropped_weights)
            keep_sum = dropped_weights.sum(dim=-1, keepdim=True)
        return dropped_weights / keep_sum.clamp_min(1e-6)

    def _build_dynamic_text_features(self, image_features, prompt_features, gate_module, mean_text_features):
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if prompt_features.dim() != 3:
            raise ValueError(f"Prompt features must be [classes, prompts, dim], got rank={prompt_features.dim()}")

        image_features = F.normalize(image_features.float(), dim=-1)
        prompt_features = F.normalize(prompt_features.float(), dim=-1)

        batch_size = image_features.size(0)
        num_classes, num_prompts, dim = prompt_features.shape

        prompt_features_expanded = prompt_features.unsqueeze(0).expand(batch_size, -1, -1, -1)
        image_features_expanded = image_features.unsqueeze(1).unsqueeze(2).expand(-1, num_classes, num_prompts, -1)

        gate_input = torch.cat(
            [
                image_features_expanded,
                prompt_features_expanded,
                image_features_expanded * prompt_features_expanded,
                torch.abs(image_features_expanded - prompt_features_expanded),
            ],
            dim=-1,
        )

        gate_scores = gate_module(gate_input).squeeze(-1)
        prompt_weights = F.softmax(gate_scores, dim=-1)
        prompt_weights = self._apply_prompt_dropout(prompt_weights)

        dynamic_text = torch.sum(prompt_weights.unsqueeze(-1) * prompt_features_expanded, dim=2)
        dynamic_text = F.normalize(dynamic_text, dim=-1)

        mean_text = F.normalize(mean_text_features.float(), dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
        if self.dynamic_gate_residual_mean:
            gamma = torch.clamp(self.dynamic_gate_gamma, min=0.0, max=1.0)
            final_text = F.normalize((1.0 - gamma) * mean_text + gamma * dynamic_text, dim=-1)
        else:
            final_text = dynamic_text

        return final_text, prompt_weights

    def _build_peps_text_features(self, prototype_features, prompt_features):
        if prototype_features.dim() != 3:
            raise ValueError(
                f"Prototype features must be [batch, num_proto, dim], got rank={prototype_features.dim()}"
            )
        if prompt_features.dim() != 3:
            raise ValueError(f"Prompt features must be [classes, prompts, dim], got rank={prompt_features.dim()}")

        prototype_features = F.normalize(prototype_features.float(), dim=-1)
        prompt_features = F.normalize(prompt_features.float(), dim=-1)

        sim = torch.einsum("bpd,cmd->bcmp", prototype_features, prompt_features)
        topk_k = max(1, min(int(self.peps_topk), sim.size(-1)))
        topk_values, topk_indices = torch.topk(sim, k=topk_k, dim=-1)
        evidence = topk_values.mean(dim=-1)
        tau = max(float(self.peps_tau), 1e-6)
        prompt_weights = F.softmax(evidence / tau, dim=-1)
        dynamic_text = torch.einsum("bcm,cmd->bcd", prompt_weights, prompt_features)
        dynamic_text = F.normalize(dynamic_text, dim=-1)

        diagnostics = {
            "prompt_weights": prompt_weights.detach(),
            "prompt_evidence": evidence.detach(),
            "supporting_prototype_index": topk_indices[..., 0].detach(),
        }
        return dynamic_text, diagnostics

    def _compute_text_logits(
        self,
        image_features_low,
        image_context_low,
        image_features_high,
        image_context_high,
        prototype_features_low,
        prototype_features_high,
        device,
        return_diagnostics=False,
    ):
        text_features_low, text_features_high = self._encode_prompt_text_features(device)
        diagnostics = {}

        if self.use_dynamic_prompt_gate:
            mean_low = self.concept_mean_text_low.to(device)
            mean_high = self.concept_mean_text_high.to(device)

            text_features_low, prompt_weights_low = self._build_dynamic_text_features(
                image_features_low,
                text_features_low.to(device),
                self.dynamic_gate_low,
                mean_low,
            )
            text_features_high, prompt_weights_high = self._build_dynamic_text_features(
                image_features_high,
                text_features_high.to(device),
                self.dynamic_gate_high,
                mean_high,
            )

            text_features_low = self._contextualize_dynamic_text_features(text_features_low, image_context_low)
            text_features_high = self._contextualize_dynamic_text_features(text_features_high, image_context_high)

            logits_low = self._compute_dynamic_scale_logits(image_features_low, text_features_low)
            logits_high = self._compute_dynamic_scale_logits(image_features_high, text_features_high)

            if return_diagnostics:
                diagnostics = {
                    "prompt_weights_low": prompt_weights_low.detach(),
                    "prompt_weights_high": prompt_weights_high.detach(),
                    "dynamic_text_low": text_features_low.detach(),
                    "dynamic_text_high": text_features_high.detach(),
                    "gamma": torch.clamp(self.dynamic_gate_gamma.detach(), min=0.0, max=1.0),
                }
        elif self.use_peps:
            text_features_low, peps_diag_low = self._build_peps_text_features(
                prototype_features_low,
                text_features_low.to(device),
            )
            text_features_high, peps_diag_high = self._build_peps_text_features(
                prototype_features_high,
                text_features_high.to(device),
            )

            text_features_low = self._contextualize_dynamic_text_features(text_features_low, image_context_low)
            text_features_high = self._contextualize_dynamic_text_features(text_features_high, image_context_high)

            logits_low = self._compute_dynamic_scale_logits(image_features_low, text_features_low)
            logits_high = self._compute_dynamic_scale_logits(image_features_high, text_features_high)

            if return_diagnostics:
                diagnostics = {
                    "prompt_weights_low": peps_diag_low["prompt_weights"],
                    "prompt_weights_high": peps_diag_high["prompt_weights"],
                    "prompt_evidence_low": peps_diag_low["prompt_evidence"],
                    "prompt_evidence_high": peps_diag_high["prompt_evidence"],
                    "supporting_prototype_index_low": peps_diag_low["supporting_prototype_index"],
                    "supporting_prototype_index_high": peps_diag_high["supporting_prototype_index"],
                    "dynamic_text_low": text_features_low.detach(),
                    "dynamic_text_high": text_features_high.detach(),
                }
        else:
            text_features_low = self._contextualize_text_features(text_features_low, image_context_low)
            text_features_high = self._contextualize_text_features(text_features_high, image_context_high)
            logits_low = self._compute_scale_logits(image_features_low, text_features_low)
            logits_high = self._compute_scale_logits(image_features_high, text_features_high)

        return logits_low, logits_high, diagnostics

    def _initialize_concept_prompt_pool(self, config):
        if not self.use_concept_prompt_pool:
            return

        if self.prompt_ensemble_mode not in {"embedding_mean", "logit_mean", "dynamic_gate", "peps"}:
            raise ValueError(f"Unsupported prompt_ensemble_mode: {self.prompt_ensemble_mode}")
        if not self.concept_prompt_path:
            raise ValueError("use_concept_prompt_pool=True but concept_prompt_path is missing")

        if self.prompt_ensemble_mode == "embedding_mean":
            concept_low, concept_high = build_concept_text_features(
                prompt_json_path=self.concept_prompt_path,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                device=next(self.parameters()).device,
                num_classes=self.num_classes,
                dtype=torch.float32,
                class_names=getattr(config, "class_names", None),
            )
        elif self.prompt_ensemble_mode == "logit_mean":
            concept_low, concept_high = build_concept_prompt_tensors(
                prompt_json_path=self.concept_prompt_path,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                device=next(self.parameters()).device,
                num_classes=self.num_classes,
                dtype=torch.float32,
                class_names=getattr(config, "class_names", None),
            )
        else:
            (
                concept_low,
                concept_high,
                prompt_texts_low,
                prompt_texts_high,
                prompt_metadata_low,
                prompt_metadata_high,
            ) = build_concept_prompt_bundle(
                prompt_json_path=self.concept_prompt_path,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                device=next(self.parameters()).device,
                num_classes=self.num_classes,
                dtype=torch.float32,
                class_names=getattr(config, "class_names", None),
            )
            self.concept_prompt_texts_low = prompt_texts_low
            self.concept_prompt_texts_high = prompt_texts_high
            self.concept_prompt_metadata_low = prompt_metadata_low
            self.concept_prompt_metadata_high = prompt_metadata_high
            if self.prompt_ensemble_mode == "dynamic_gate":
                mean_low = F.normalize(concept_low.mean(dim=1), dim=-1)
                mean_high = F.normalize(concept_high.mean(dim=1), dim=-1)
                self.register_buffer("concept_mean_text_low", mean_low.detach().cpu())
                self.register_buffer("concept_mean_text_high", mean_high.detach().cpu())

        self.register_buffer("concept_text_low", concept_low.detach().cpu())
        self.register_buffer("concept_text_high", concept_high.detach().cpu())
        print(
            f"[ConceptPromptPool] enabled | mode={self.prompt_ensemble_mode} | "
            f"path={self.concept_prompt_path} | low_shape={tuple(concept_low.shape)} | "
            f"high_shape={tuple(concept_high.shape)}"
        )

    def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        """
        前向传播
        """
        del coord_s, coords_l, slide_id

        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)
        prototype_features_low = self._prototype_tensor_from_components(compents)

        H = compents.squeeze().float()
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, H)

        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)
        prototype_features_high = self._prototype_tensor_from_components(compents_high)

        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)

        image_context = torch.cat((compents.squeeze(), M.squeeze(0)), dim=0)
        image_context_high = torch.cat((compents_high.squeeze(), M_high.squeeze(0)), dim=0)
        logits_low, logits_high, _ = self._compute_text_logits(
            image_features_low=image_features_low,
            image_context_low=image_context,
            image_features_high=image_features_high,
            image_context_high=image_context_high,
            prototype_features_low=prototype_features_low,
            prototype_features_high=prototype_features_high,
            device=x_s.device,
            return_diagnostics=False,
        )
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return Y_prob, Y_hat, loss

    def forward_with_prompt_diagnostics(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        del coord_s, coords_l, slide_id

        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)
        prototype_features_low = self._prototype_tensor_from_components(compents)

        H = compents.squeeze().float()
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, H)

        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)
        prototype_features_high = self._prototype_tensor_from_components(compents_high)

        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)

        image_context = torch.cat((compents.squeeze(), M.squeeze(0)), dim=0)
        image_context_high = torch.cat((compents_high.squeeze(), M_high.squeeze(0)), dim=0)
        logits_low, logits_high, diagnostics = self._compute_text_logits(
            image_features_low=image_features_low,
            image_context_low=image_context,
            image_features_high=image_features_high,
            image_context_high=image_context_high,
            prototype_features_low=prototype_features_low,
            prototype_features_high=prototype_features_high,
            device=x_s.device,
            return_diagnostics=True,
        )
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        diagnostics.update(
            {
                "logits_low": logits_low.detach(),
                "logits_high": logits_high.detach(),
                "final_logits": logits.detach(),
                "pred_probs": Y_prob.detach(),
            }
        )
        return Y_prob, Y_hat, loss, diagnostics

    def forward_with_attention(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        """前向传播并返回注意力权重(用于热力图)"""
        del coord_s, coords_l, label, slide_id

        M = x_s.float()
        compents, cross_attn_weights_s = self.cross_attention_1(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)
        prototype_features_low = self._prototype_tensor_from_components(compents)

        M_high = x_l.float()
        compents_high, cross_attn_weights_l = self.cross_attention_1(
            self.learnable_image_center,
            M_high,
            M_high,
        )
        compents_high = self.norm(compents_high + self.learnable_image_center)
        prototype_features_high = self._prototype_tensor_from_components(compents_high)

        H = compents.squeeze().float()
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, H)

        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)

        image_context = torch.cat((compents.squeeze(), M.squeeze(0)), dim=0)
        image_context_high = torch.cat((compents_high.squeeze(), M_high.squeeze(0)), dim=0)
        logits_low, logits_high, _ = self._compute_text_logits(
            image_features_low=image_features_low,
            image_context_low=image_context,
            image_features_high=image_features_high,
            image_context_high=image_context_high,
            prototype_features_low=prototype_features_low,
            prototype_features_high=prototype_features_high,
            device=x_s.device,
            return_diagnostics=False,
        )
        logits = logits_low + logits_high

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        patch_attention_s = cross_attn_weights_s.mean(dim=1).squeeze(0).mean(dim=0)
        patch_attention_l = cross_attn_weights_l.mean(dim=1).squeeze(0).mean(dim=0)

        return logits, Y_prob, Y_hat, patch_attention_s, patch_attention_l
