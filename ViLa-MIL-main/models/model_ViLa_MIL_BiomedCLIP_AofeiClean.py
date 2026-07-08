# coding=utf-8
"""
Clean Aofei ViLa-MIL BiomedCLIP baseline imported into the main project.
"""

from __future__ import absolute_import, division, print_function

import inspect
import logging
import math
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from open_clip import create_model_from_pretrained, get_tokenizer

from .model_utils import MultiheadAttention

DEFAULT_BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_BIOMEDCLIP_MODEL = f"hf-hub:{DEFAULT_BIOMEDCLIP_REPO}"
DEFAULT_TEXT_REPO = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

logger = logging.getLogger(__name__)


def _candidate_cache_dirs(explicit_cache_dir=None):
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        explicit_cache_dir,
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        repo_root / "hf_cache",
        repo_root.parent / "hf_cache",
        repo_root / "model_cache",
    ]

    seen = set()
    resolved = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        candidate_str = str(candidate_path)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        resolved.append(candidate_path)
    return resolved


def _resolve_snapshot_dir(cache_dir, repo_id):
    snapshots_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshot_dirs:
        return None
    return snapshot_dirs[-1]


def _bootstrap_hf_environment():
    for candidate_cache_dir in _candidate_cache_dirs():
        if not candidate_cache_dir.exists():
            continue

        clip_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_BIOMEDCLIP_REPO)
        text_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_TEXT_REPO)
        if not clip_snapshot:
            continue

        os.environ.setdefault("HF_HOME", str(candidate_cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(candidate_cache_dir))
        if text_snapshot:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(candidate_cache_dir)

    return None


def _prepare_biomedclip_loading(model_path, cache_dir=None):
    for candidate_cache_dir in _candidate_cache_dirs(cache_dir):
        if not candidate_cache_dir.exists():
            continue

        clip_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_BIOMEDCLIP_REPO)
        text_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_TEXT_REPO)
        if not clip_snapshot:
            continue

        os.environ.setdefault("HF_HOME", str(candidate_cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(candidate_cache_dir))

        offline_enabled = text_snapshot is not None
        if offline_enabled:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        resolved_model_path = model_path
        if model_path == DEFAULT_BIOMEDCLIP_MODEL:
            resolved_model_path = f"local-dir:{clip_snapshot}"

        return resolved_model_path, str(candidate_cache_dir), offline_enabled

    return model_path, cache_dir, False


_BOOTSTRAP_CACHE_DIR = _bootstrap_hf_environment()


class BiomedCLIPTextEncoder(nn.Module):
    def __init__(self, biomedclip_model, *, n_ctx=16, finetune=False):
        super().__init__()
        self.model = biomedclip_model
        self.n_ctx = int(n_ctx)
        self.finetune = bool(finetune)
        self._warned_embed_fallback = False
        self.fallback_count = 0
        self.fallback_last_error = None

    def forward(self, text_tokens, prompt_embeddings=None, eos_indices=None):
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
            except Exception as exc:
                self.fallback_count += 1
                self.fallback_last_error = str(exc)
                if not self._warned_embed_fallback:
                    msg = (
                        "[BiomedCLIPTextEncoder] prompt-embedding path failed, "
                        "falling back to encode_text(tokens). "
                        f"Error: {exc}"
                    )
                    print(msg)
                    logger.warning(msg)
                    self._warned_embed_fallback = True

        if self.finetune:
            return self.model.encode_text(text_tokens)
        with torch.no_grad():
            return self.model.encode_text(text_tokens)


class BiomedCLIPPromptLearner(nn.Module):
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
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
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


class ViLa_MIL_BiomedCLIP_AofeiClean(nn.Module):
    def __init__(self, config, num_classes=2, model_path=DEFAULT_BIOMEDCLIP_MODEL):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

        env_model_path = os.environ.get("BIOMEDCLIP_MODEL_PATH", "").strip()
        if env_model_path:
            model_path = env_model_path

        self.L = 512
        self.D = config.hidden_size
        self.K = 1

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        resolved_model_path, resolved_cache_dir, offline_enabled = _prepare_biomedclip_loading(model_path)
        print(f"🔬 Loading BiomedCLIP from: {resolved_model_path}")
        if resolved_cache_dir:
            print(f"📦 Using HuggingFace cache: {resolved_cache_dir}")
        if offline_enabled:
            print("📴 Offline cache mode enabled")
        try:
            biomedclip_model, _ = create_model_from_pretrained(
                resolved_model_path,
                cache_dir=resolved_cache_dir,
            )
            tokenizer = get_tokenizer(resolved_model_path)
        except Exception as exc:
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            msg = (
                "[Error] Failed to load BiomedCLIP from HuggingFace Hub. "
                "This is usually caused by transient network/proxy/SSL issues or missing local cache.\n"
                f"- model_path: {resolved_model_path}\n"
                f"- cache_dir: {resolved_cache_dir}\n"
                f"- HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')} (offline={offline})\n"
                "Fix options:\n"
                "1) Ensure the model is fully downloaded into cache (run a one-time warmup download).\n"
                "2) If you already downloaded it, re-run with offline cache only: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
                "3) If you use a proxy, make sure HTTPS proxy is stable and supports TLS properly.\n"
                f"Original error: {exc}"
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

        self.norm = nn.LayerNorm(self.L)
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.learnable_image_center = nn.Parameter(torch.Tensor(config.prototype_number, 1, self.L))
        trunc_normal_(self.learnable_image_center, std=0.02)

        self._configure_biomedclip_finetune(config)

    def _configure_biomedclip_finetune(self, config):
        finetune_text = bool(getattr(config, "finetune_text_encoder", False))
        mode = str(getattr(config, "text_finetune_mode", "proj"))
        last_n = int(getattr(config, "text_unfreeze_last_n", 2))

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        text_clip = self.text_encoder.model
        if hasattr(text_clip, "visual"):
            for param in text_clip.visual.parameters():
                param.requires_grad = False

        if not finetune_text:
            return

        text_model = text_clip.text if hasattr(text_clip, "text") else text_clip

        def _unfreeze_module(module):
            if module is None:
                return
            for param in module.parameters():
                param.requires_grad = True

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
                n_unfreeze = max(0, min(int(last_n), n_layers))
                for idx in range(n_layers - n_unfreeze, n_layers):
                    _unfreeze_module(layers[idx])
            else:
                mode = "full"

        if mode == "full":
            _unfreeze_module(text_model)
            if hasattr(text_clip, "text_projection"):
                _unfreeze_obj(getattr(text_clip, "text_projection"))

    def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None, **kwargs):
        del coord_s, coords_l, slide_id, kwargs

        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        prompt_embeddings = self.prompt_learner().to(x_s.device)
        eos_indices = getattr(self.prompt_learner, "eos_indices", None)
        if eos_indices is not None:
            eos_indices = eos_indices.to(x_s.device)
        text_features = self.text_encoder(
            tokenized_prompts,
            prompt_embeddings=prompt_embeddings,
            eos_indices=eos_indices,
        )

        m_low = x_s.float()
        components_low, _ = self.cross_attention_1(self.learnable_image_center, m_low, m_low)
        components_low = self.norm(components_low + self.learnable_image_center)

        h_low = components_low.squeeze().float()
        a_v_low = self.attention_V(h_low)
        a_u_low = self.attention_U(h_low)
        a_low = self.attention_weights(a_v_low * a_u_low)
        a_low = torch.transpose(a_low, 1, 0)
        a_low = F.softmax(a_low, dim=1)
        image_features_low = torch.mm(a_low, h_low)

        m_high = x_l.float()
        components_high, _ = self.cross_attention_1(self.learnable_image_center, m_high, m_high)
        components_high = self.norm(components_high + self.learnable_image_center)

        h_high = components_high.squeeze().float()
        a_v_high = self.attention_V(h_high)
        a_u_high = self.attention_U(h_high)
        a_high = self.attention_weights(a_v_high * a_u_high)
        a_high = torch.transpose(a_high, 1, 0)
        a_high = F.softmax(a_high, dim=1)
        image_features_high = torch.mm(a_high, h_high)

        text_features_low = text_features[: self.num_classes]
        image_context_low = torch.cat((components_low.squeeze(), m_low.squeeze(0)), dim=0)
        text_context_low, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1),
            image_context_low,
            image_context_low,
        )
        text_features_low = text_context_low.squeeze() + text_features_low

        text_features_high = text_features[self.num_classes :]
        image_context_high = torch.cat((components_high.squeeze(), m_high.squeeze(0)), dim=0)
        text_context_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1),
            image_context_high,
            image_context_high,
        )
        text_features_high = text_context_high.squeeze() + text_features_high

        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(y_prob, 1, dim=1)[1]
        return y_prob, y_hat, loss

    def forward_with_attention(self, x_s, coord_s, x_l, coords_l, label, slide_id=None, **kwargs):
        del coord_s, coords_l, slide_id, kwargs

        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        prompt_embeddings = self.prompt_learner().to(x_s.device)
        eos_indices = getattr(self.prompt_learner, "eos_indices", None)
        if eos_indices is not None:
            eos_indices = eos_indices.to(x_s.device)
        text_features = self.text_encoder(
            tokenized_prompts,
            prompt_embeddings=prompt_embeddings,
            eos_indices=eos_indices,
        )

        m_low = x_s.float()
        components_low, cross_attn_weights_low = self.cross_attention_1(
            self.learnable_image_center,
            m_low,
            m_low,
        )
        components_low = self.norm(components_low + self.learnable_image_center)

        m_high = x_l.float()
        components_high, cross_attn_weights_high = self.cross_attention_1(
            self.learnable_image_center,
            m_high,
            m_high,
        )
        components_high = self.norm(components_high + self.learnable_image_center)

        h_low = components_low.squeeze().float()
        a_v_low = self.attention_V(h_low)
        a_u_low = self.attention_U(h_low)
        a_low = self.attention_weights(a_v_low * a_u_low)
        a_low = torch.transpose(a_low, 1, 0)
        a_low = F.softmax(a_low, dim=1)
        image_features_low = torch.mm(a_low, h_low)

        h_high = components_high.squeeze().float()
        a_v_high = self.attention_V(h_high)
        a_u_high = self.attention_U(h_high)
        a_high = self.attention_weights(a_v_high * a_u_high)
        a_high = torch.transpose(a_high, 1, 0)
        a_high = F.softmax(a_high, dim=1)
        image_features_high = torch.mm(a_high, h_high)

        text_features_low = text_features[: self.num_classes]
        image_context_low = torch.cat((components_low.squeeze(), m_low.squeeze(0)), dim=0)
        text_context_low, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1),
            image_context_low,
            image_context_low,
        )
        text_features_low = text_context_low.squeeze() + text_features_low

        text_features_high = text_features[self.num_classes :]
        image_context_high = torch.cat((components_high.squeeze(), m_high.squeeze(0)), dim=0)
        text_context_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1),
            image_context_high,
            image_context_high,
        )
        text_features_high = text_context_high.squeeze() + text_features_high

        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high

        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(y_prob, 1, dim=1)[1]

        patch_attention_low = cross_attn_weights_low.mean(dim=1).squeeze(0).mean(dim=0)
        patch_attention_high = cross_attn_weights_high.mean(dim=1).squeeze(0).mean(dim=0)
        return logits, y_prob, y_hat, patch_attention_low, patch_attention_high
