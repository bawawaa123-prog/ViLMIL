# coding=utf-8
"""
ViLa-MIL with BiomedCLIP
使用BiomedCLIP替换原始CLIP的图像和文本编码器
"""

from __future__ import absolute_import, division, print_function
import logging
import warnings
import math
import inspect
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# BiomedCLIP依赖(不再需要tokenize函数,使用tokenizer对象)
from open_clip import create_model_from_pretrained, get_tokenizer

from .model_utils import MultiheadAttention
from utils.rag_prompt_rewriter import RAGPromptRewriter

logger = logging.getLogger(__name__)


class DynamicPromptRetriever(nn.Module):
    """Retrieve and fuse class-wise sentence pools conditioned on slide visual features."""

    def __init__(self, num_classes: int, topk: int = 3, temperature: float = 0.1, mix_ratio: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.topk = max(1, int(topk))
        self.temperature = max(1e-6, float(temperature))
        self.mix_ratio = float(min(1.0, max(0.0, mix_ratio)))

    def _retrieve_one_scale(self, query_feat, pool_feats_by_class, static_feats=None, pool_texts_by_class=None, scale_name='low'):
        """
        query_feat: [1, D]
        pool_feats_by_class: list[T_i, D], length == num_classes
        static_feats: [num_classes, D] fallback/mix source
        """
        q = F.normalize(query_feat, dim=-1)
        outputs = []
        debug_items = []

        for c in range(self.num_classes):
            pool = pool_feats_by_class[c]
            has_pool = pool is not None and pool.numel() > 0 and pool.shape[0] > 0
            fallback = static_feats[c] if static_feats is not None else q.squeeze(0)

            if not has_pool:
                outputs.append(fallback)
                debug_items.append({
                    'scale': scale_name,
                    'class_idx': int(c),
                    'top_indices': [],
                    'top_weights': [],
                    'top_scores': [],
                    'top_texts': [],
                })
                continue

            pool_n = F.normalize(pool, dim=-1)
            sim = torch.matmul(q, pool_n.t()).squeeze(0)  # [T_i]

            k = min(self.topk, sim.shape[0])
            top_vals, top_idx = torch.topk(sim, k=k, dim=0)
            top_pool = pool[top_idx]  # [k, D]
            weights = F.softmax(top_vals / self.temperature, dim=0)
            retrieved = torch.sum(weights.unsqueeze(1) * top_pool, dim=0)

            if static_feats is not None and self.mix_ratio < 1.0:
                retrieved = self.mix_ratio * retrieved + (1.0 - self.mix_ratio) * fallback

            outputs.append(retrieved)

            top_texts = []
            if pool_texts_by_class is not None and c < len(pool_texts_by_class):
                c_texts = pool_texts_by_class[c]
                for idx in top_idx.tolist():
                    if 0 <= idx < len(c_texts):
                        top_texts.append(str(c_texts[idx]))
                    else:
                        top_texts.append('')

            debug_items.append({
                'scale': scale_name,
                'class_idx': int(c),
                'top_indices': [int(x) for x in top_idx.tolist()],
                'top_weights': [float(x) for x in weights.detach().cpu().tolist()],
                'top_scores': [float(x) for x in top_vals.detach().cpu().tolist()],
                'top_texts': top_texts,
            })

        return torch.stack(outputs, dim=0), debug_items  # [num_classes, D]

    def forward(self, low_query, high_query, low_pool_feats_by_class, high_pool_feats_by_class,
                static_low=None, static_high=None, low_pool_texts_by_class=None, high_pool_texts_by_class=None):
        low_dyn, low_dbg = self._retrieve_one_scale(
            low_query,
            low_pool_feats_by_class,
            static_feats=static_low,
            pool_texts_by_class=low_pool_texts_by_class,
            scale_name='low',
        )
        high_dyn, high_dbg = self._retrieve_one_scale(
            high_query,
            high_pool_feats_by_class,
            static_feats=static_high,
            pool_texts_by_class=high_pool_texts_by_class,
            scale_name='high',
        )
        return low_dyn, high_dyn, {'low': low_dbg, 'high': high_dbg}


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
        # Debug/observability: track whether we ever had to fall back.
        self.fallback_count = 0
        self.fallback_last_error = None
        
    def forward(self, text_tokens, prompt_embeddings: torch.Tensor | None = None, eos_indices: torch.Tensor | None = None):
        """
        前向传播函数，用于处理输入的文本token并提取文本特征
        参数:
            text_tokens: tokenized text [batch, seq_len]
                - 批量处理文本的token序列
                - batch: 批次大小
                - seq_len: 序列长度
            prompt_embeddings: learned prompt embeddings [batch, seq_len, dim] (optional)
            eos_indices: indices of end token in the *prompt_embeddings* sequence [batch] (optional)
        返回:
            text_features: [batch, 512]
        """
        # Prefer prompt-tuning path (uses learnable ctx) when provided.
        if prompt_embeddings is not None:
            try:
                text_model = self.model.text if hasattr(self.model, 'text') else self.model

                # Case A: CLIP-style text tower (token_embedding/positional_embedding/transformer/ln_final/text_projection)
                if all(hasattr(text_model, a) for a in ['positional_embedding', 'transformer', 'ln_final', 'text_projection']):
                    x = prompt_embeddings
                    x = x + text_model.positional_embedding.to(x.dtype)
                    x = x.permute(1, 0, 2)  # (seq, batch, dim)
                    x = text_model.transformer(x)
                    x = x.permute(1, 0, 2)  # (batch, seq, dim)
                    x = text_model.ln_final(x)

                    if eos_indices is None:
                        # Fallback: use last non-pad token position (token!=0) then shift by n_ctx.
                        attn_mask = (text_tokens != 0)
                        eos_indices = attn_mask.long().sum(dim=1) - 1
                        eos_indices = torch.clamp(eos_indices + self.n_ctx, min=0, max=x.shape[1] - 1)

                    x = x[torch.arange(x.shape[0], device=x.device), eos_indices]
                    proj = text_model.text_projection
                    if isinstance(proj, (torch.Tensor, nn.Parameter)):
                        text_features = x @ proj
                    elif isinstance(proj, nn.Module):
                        text_features = proj(x)
                    else:
                        raise TypeError(f'Unsupported text_projection type: {type(proj)}')
                    return text_features

                # Case B: HF-style transformer (supports inputs_embeds)
                transformer = getattr(text_model, 'transformer', None)
                if transformer is not None and hasattr(transformer, 'forward'):
                    sig = None
                    try:
                        sig = inspect.signature(transformer.forward)
                    except Exception:
                        sig = None

                    if sig is not None and 'inputs_embeds' in sig.parameters:
                        token_mask = (text_tokens != 0)
                        # Build a new attention_mask aligned to prompt_embeddings
                        # prompt = [prefix] + [ctx] + [suffix_trunc]
                        bsz, seq_len = text_tokens.shape
                        if prompt_embeddings.shape[1] != seq_len:
                            # Best-effort: align to prompt_embeddings length
                            seq_len = prompt_embeddings.shape[1]
                        prefix_mask = token_mask[:, :1]
                        ctx_mask = torch.ones((bsz, self.n_ctx), device=text_tokens.device, dtype=token_mask.dtype)
                        suffix_keep = max(int(text_tokens.shape[1]) - 1 - self.n_ctx, 0)
                        suffix_mask = token_mask[:, 1:1 + suffix_keep]
                        attn_mask = torch.cat([prefix_mask, ctx_mask, suffix_mask], dim=1)
                        attn_mask = attn_mask[:, :prompt_embeddings.shape[1]]

                        out = transformer(inputs_embeds=prompt_embeddings, attention_mask=attn_mask, return_dict=True)
                        hidden = getattr(out, 'last_hidden_state', None)
                        if hidden is None and isinstance(out, (tuple, list)) and len(out) > 0:
                            hidden = out[0]
                        if hidden is None:
                            raise RuntimeError('HF transformer output missing last_hidden_state')

                        # Prefer CLS token representation for HF models.
                        pooled = hidden[:, 0]
                        if hasattr(text_model, 'ln_final'):
                            pooled = text_model.ln_final(pooled)
                        proj = None
                        for attr in ['proj', 'text_projection']:
                            if hasattr(text_model, attr):
                                proj = getattr(text_model, attr)
                                break
                        if proj is None and hasattr(self.model, 'text_projection'):
                            proj = getattr(self.model, 'text_projection')
                        if proj is not None:
                            if isinstance(proj, (torch.Tensor, nn.Parameter)):
                                pooled = pooled @ proj
                            elif isinstance(proj, nn.Module):
                                pooled = proj(pooled)
                            else:
                                raise TypeError(f'Unsupported projection type: {type(proj)}')
                        return pooled

                # If we get here, we can't run embeddings through text tower; fall back.
                raise RuntimeError('Unsupported BiomedCLIP text tower for prompt embeddings')

            except Exception as e:
                self.fallback_count += 1
                self.fallback_last_error = str(e)
                if not self._warned_embed_fallback:
                    msg = (
                        "[BiomedCLIPTextEncoder] prompt-embedding path failed, "
                        "falling back to encode_text(tokens). "
                        f"Error: {e}"
                    )
                    # Ensure it's visible in training logs even if logging isn't configured.
                    print(msg)
                    logger.warning(msg)
                    self._warned_embed_fallback = True

        # Fallback: token-id path (no prompt-learning). If finetune=True, allow grads.
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
        
        # 获取文本嵌入维度(BiomedCLIP使用text.transformer结构)
        # BiomedCLIP的文本模型是CustomTextCLIP,通过text属性访问
        text_model = biomedclip_model.text if hasattr(biomedclip_model, 'text') else biomedclip_model
        
        # 获取embedding层(兼容不同结构)
        if hasattr(text_model, 'transformer'):
            # HuggingFace BERT结构
            token_embedding_layer = text_model.transformer.embeddings.word_embeddings
            ctx_dim = token_embedding_layer.embedding_dim
        elif hasattr(text_model, 'token_embedding'):
            # 原始CLIP结构
            with torch.no_grad():
                dummy_tokens = tokenizer(["test"]).to(next(biomedclip_model.parameters()).device)
                dummy_emb = text_model.token_embedding(dummy_tokens)
                ctx_dim = dummy_emb.shape[-1]
        else:
            # 默认使用512(BiomedCLIP标准维度)
            ctx_dim = 512
        
        # 可学习的上下文向量
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 预处理类别名称
        classnames = [name.replace("_", " ") for name in classnames]
        self.classnames = classnames
        
        # Tokenize类别名称(使用tokenizer对象)
        prompts = [f"a histopathology image of {name}" for name in classnames]
        self.tokenized_prompts = tokenizer(prompts)
        
        # 获取token嵌入(使用embedding层)
        with torch.no_grad():
            device = next(biomedclip_model.parameters()).device
            token_ids = self.tokenized_prompts.to(device)
            
            # 根据模型结构选择embedding方法
            if hasattr(text_model, 'transformer'):
                # HuggingFace BERT
                embedding = text_model.transformer.embeddings.word_embeddings(token_ids)
            elif hasattr(text_model, 'token_embedding'):
                # 原始CLIP
                embedding = text_model.token_embedding(token_ids)
            else:
                raise AttributeError("Cannot find token embedding layer in BiomedCLIP model")

        # NOTE: Keep prompt length constant. We insert n_ctx tokens after the first token,
        # so we must drop n_ctx tokens from the tail (usually padding) to preserve seq_len.
        seq_len = embedding.shape[1]
        suffix_keep = max(seq_len - 1 - n_ctx, 0)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", embedding[:, 1:1 + suffix_keep, :])  # class name + [EOS] (+ truncated padding)

        # Track end-token indices in the *prompt_embeddings* sequence for CLIP-style pooling.
        # Use last non-pad token position in original tokens, then shift by n_ctx.
        with torch.no_grad():
            pad_id = self._pad_id
            non_pad = (token_ids != pad_id)
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
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        # 拼接: [SOS] + ctx + class_name + [EOS]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """截断正态分布初始化"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b]", stacklevel=2)
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ViLa_MIL_BiomedCLIP(nn.Module):
    """
    ViLa-MIL模型(BiomedCLIP版本)
    
    主要变化:
    1. 图像特征: 512维 (BiomedCLIP ViT-B/16) vs 1024维 (CLIP RN50)
    2. 文本编码器: PubMedBERT vs OpenAI CLIP Transformer
    3. 医学领域预训练优势
    """
    def __init__(self, config, num_classes=2, 
                 model_path='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        
        # 特征维度适配(BiomedCLIP输出512维)
        self.L = 512  # BiomedCLIP图像特征维度
        self.D = config.hidden_size  # 隐藏层维度(保持原设计)
        self.K = 1
        
        # 注意力模块
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        
        # 加载BiomedCLIP
        print(f"🔬 Loading BiomedCLIP from: {model_path}")
        try:
            biomedclip_model, _ = create_model_from_pretrained(model_path)
            tokenizer = get_tokenizer(model_path)
        except Exception as e:
            offline = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
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
        
        # 文本编码器：可选微调
        finetune_text = bool(getattr(config, 'finetune_text_encoder', False))
        self.text_encoder = BiomedCLIPTextEncoder(biomedclip_model, n_ctx=16, finetune=finetune_text)
        self.tokenizer = tokenizer
        
        # 可学习提示词
        self.prompt_learner = BiomedCLIPPromptLearner(
            config.text_prompt, 
            biomedclip_model,
            tokenizer
        )
        
        # LayerNorm(适配512维)
        self.norm = nn.LayerNorm(self.L)
        
        # 交叉注意力
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1)
        
        # 可学习的图像原型
        self.learnable_image_center = nn.Parameter(
            torch.Tensor(config.prototype_number, 1, self.L)
        )
        trunc_normal_(self.learnable_image_center, std=0.02)
        
        # 文本编码器微调开关：如果关闭则冻结，否则保持可训练
        # Notes:
        # - We NEVER need to finetune BiomedCLIP visual tower here because we consume pre-extracted image features.
        # - For stability, default finetune scope is projection-only (unless user explicitly requests more).
        self._configure_biomedclip_finetune(config)

        # Dynamic prompt retrieval settings
        self.enable_dynamic_prompt = bool(getattr(config, 'enable_dynamic_prompt', False))
        self.dynamic_prompt_topk = int(getattr(config, 'retrieval_topk', 3))
        self.dynamic_prompt_temp = float(getattr(config, 'retrieval_temp', 0.1))
        self.dynamic_prompt_mix = float(getattr(config, 'dynamic_prompt_mix', 1.0))
        self.dynamic_retriever = None
        self._dynamic_low_pool_buffer_names = []
        self._dynamic_high_pool_buffer_names = []
        self._dynamic_low_pool_texts = []
        self._dynamic_high_pool_texts = []
        self.dynamic_prompt_runtime_enabled = bool(self.enable_dynamic_prompt)
        self.last_retrieval_debug = None

        # Visual-conditioned prompt refinement (VCP / HyperPrompt)
        self.enable_vcp = bool(getattr(config, 'enable_vcp', False))
        self.vcp_runtime_enabled = bool(self.enable_vcp)
        vcp_dropout = float(getattr(config, 'vcp_dropout', 0.1))
        vcp_beta_init = float(getattr(config, 'vcp_beta', 0.1))
        self.low_vcp_head = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.GELU(),
            nn.Dropout(vcp_dropout),
            nn.Linear(self.L, self.L),
        )
        self.high_vcp_head = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.GELU(),
            nn.Dropout(vcp_dropout),
            nn.Linear(self.L, self.L),
        )
        self.vcp_ln_low = nn.LayerNorm(self.L)
        self.vcp_ln_high = nn.LayerNorm(self.L)
        self.vcp_beta_low = nn.Parameter(torch.tensor(vcp_beta_init, dtype=torch.float32))
        self.vcp_beta_high = nn.Parameter(torch.tensor(vcp_beta_init, dtype=torch.float32))

        # RAG + LLM rewrite settings
        self.enable_rag_rewrite = bool(getattr(config, 'enable_rag_rewrite', False))
        self.rag_runtime_enabled = bool(self.enable_rag_rewrite)
        self.rag_mode = str(getattr(config, 'rag_mode', 'offline'))
        self.rag_cache_path = str(getattr(config, 'rag_cache_path', 'results/rag_rewrite_cache.jsonl'))
        self.rag_topk = int(getattr(config, 'rag_topk', 3))
        self.rag_ollama_model = str(getattr(config, 'rag_ollama_model', 'qwen2.5:14b-instruct'))
        self.rag_ollama_url = str(getattr(config, 'rag_ollama_url', 'http://localhost:11434/api/generate'))
        self.rag_temperature = float(getattr(config, 'rag_temperature', 0.2))
        self.rag_max_tokens = int(getattr(config, 'rag_max_tokens', 256))
        self.rag_timeout_sec = int(getattr(config, 'rag_timeout_sec', 60))
        self.rag_max_retries = int(getattr(config, 'rag_max_retries', 2))
        self.rag_retry_delay_sec = float(getattr(config, 'rag_retry_delay_sec', 0.5))
        self.rag_failure_log_path = getattr(config, 'rag_failure_log_path', None)
        self.rag_fallback = str(getattr(config, 'rag_fallback', 'dynamic')).lower()
        self.rag_rewriter = None
        self.last_rag_debug = None

        if self.enable_rag_rewrite:
            self.rag_rewriter = RAGPromptRewriter(
                class_names=getattr(config, 'class_names', [str(i) for i in range(self.num_classes)]),
                mode=self.rag_mode,
                cache_path=self.rag_cache_path,
                ollama_model=self.rag_ollama_model,
                ollama_url=self.rag_ollama_url,
                temperature=self.rag_temperature,
                max_tokens=self.rag_max_tokens,
                timeout_sec=self.rag_timeout_sec,
                rag_topk=self.rag_topk,
                max_retries=self.rag_max_retries,
                retry_delay_sec=self.rag_retry_delay_sec,
                failure_log_path=self.rag_failure_log_path,
            )
            print(
                f"[RAG] enabled | mode={self.rag_mode} | model={self.rag_ollama_model} | "
                f"cache={self.rag_cache_path} | retries={self.rag_max_retries} | "
                f"failure_log={self.rag_rewriter.failure_log_path} | fallback={self.rag_fallback}"
            )

        if self.enable_dynamic_prompt:
            self.dynamic_retriever = DynamicPromptRetriever(
                num_classes=self.num_classes,
                topk=self.dynamic_prompt_topk,
                temperature=self.dynamic_prompt_temp,
                mix_ratio=self.dynamic_prompt_mix,
            )
            self._initialize_dynamic_prompt_pool(config)

    def _initialize_dynamic_prompt_pool(self, config):
        prompt_pool = getattr(config, 'prompt_pool', None)
        class_names = getattr(config, 'class_names', None)
        if prompt_pool is None:
            raise ValueError('enable_dynamic_prompt=True but config.prompt_pool is missing')
        if class_names is None:
            raise ValueError('enable_dynamic_prompt=True but config.class_names is missing')

        if len(class_names) != self.num_classes:
            raise ValueError(
                f'class_names length ({len(class_names)}) must match num_classes ({self.num_classes})'
            )

        low_pool = prompt_pool.get('low', None)
        high_pool = prompt_pool.get('high', None)
        if low_pool is None or high_pool is None:
            raise ValueError('prompt_pool must contain both low and high keys')

        for c in range(self.num_classes):
            low_sentences = [str(x) for x in (low_pool[c] if c < len(low_pool) else []) if str(x).strip()]
            high_sentences = [str(x) for x in (high_pool[c] if c < len(high_pool) else []) if str(x).strip()]
            self._dynamic_low_pool_texts.append(low_sentences)
            self._dynamic_high_pool_texts.append(high_sentences)

            low_feats = self._encode_sentence_list(low_sentences)
            high_feats = self._encode_sentence_list(high_sentences)

            low_name = f'_dynamic_low_pool_{c}'
            high_name = f'_dynamic_high_pool_{c}'
            self.register_buffer(low_name, low_feats)
            self.register_buffer(high_name, high_feats)
            self._dynamic_low_pool_buffer_names.append(low_name)
            self._dynamic_high_pool_buffer_names.append(high_name)

        total_low = sum(int(getattr(self, n).shape[0]) for n in self._dynamic_low_pool_buffer_names)
        total_high = sum(int(getattr(self, n).shape[0]) for n in self._dynamic_high_pool_buffer_names)
        print(
            f"[DynamicPrompt] enabled | classes={self.num_classes} | "
            f"low_sentences={total_low} | high_sentences={total_high} | "
            f"topk={self.dynamic_prompt_topk} | temp={self.dynamic_prompt_temp:g} | mix={self.dynamic_prompt_mix:g}"
        )

    def _encode_sentence_list(self, sentences):
        if len(sentences) == 0:
            return torch.empty(0, self.L, dtype=torch.float32)

        device = next(self.parameters()).device
        text_tokens = self.tokenizer(sentences).to(device)
        with torch.no_grad():
            text_features = self.text_encoder(text_tokens)
            text_features = F.normalize(text_features.float(), dim=-1)
        return text_features.detach().cpu()

    def _get_dynamic_pool_features(self, device):
        low_pool_feats = [getattr(self, n).to(device) for n in self._dynamic_low_pool_buffer_names]
        high_pool_feats = [getattr(self, n).to(device) for n in self._dynamic_high_pool_buffer_names]
        return low_pool_feats, high_pool_feats

    def set_dynamic_prompt_runtime(self, enabled: bool):
        self.dynamic_prompt_runtime_enabled = bool(enabled)

    def get_last_retrieval_debug(self):
        return self.last_retrieval_debug

    def get_last_rag_debug(self):
        return self.last_rag_debug

    def set_vcp_runtime(self, enabled: bool):
        self.vcp_runtime_enabled = bool(enabled)

    def set_rag_runtime(self, enabled: bool):
        self.rag_runtime_enabled = bool(enabled)

    def _apply_vcp(self, image_features_low, image_features_high, text_features_low, text_features_high):
        # image_features_*: [1, D], text_features_*: [num_classes, D]
        low_query = F.normalize(image_features_low.float(), dim=-1)
        high_query = F.normalize(image_features_high.float(), dim=-1)

        delta_low = self.low_vcp_head(low_query).expand_as(text_features_low)
        delta_high = self.high_vcp_head(high_query).expand_as(text_features_high)

        text_features_low = self.vcp_ln_low(text_features_low + self.vcp_beta_low * delta_low)
        text_features_high = self.vcp_ln_high(text_features_high + self.vcp_beta_high * delta_high)
        return text_features_low, text_features_high

    def _configure_biomedclip_finetune(self, config):
        finetune_text = bool(getattr(config, 'finetune_text_encoder', False))
        mode = str(getattr(config, 'text_finetune_mode', 'proj'))
        last_n = int(getattr(config, 'text_unfreeze_last_n', 2))

        # Freeze everything in the BiomedCLIP wrapper by default.
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Always freeze visual tower (not used in forward).
        text_clip = self.text_encoder.model
        if hasattr(text_clip, 'visual'):
            for p in text_clip.visual.parameters():
                p.requires_grad = False

        if not finetune_text:
            return

        # Unfreeze text tower according to mode.
        text_model = text_clip.text if hasattr(text_clip, 'text') else text_clip

        def _unfreeze_module(m: nn.Module | None):
            if m is None:
                return
            for p in m.parameters():
                p.requires_grad = True

        def _unfreeze_obj(obj):
            if obj is None:
                return
            if isinstance(obj, nn.Module):
                _unfreeze_module(obj)
            elif isinstance(obj, (torch.Tensor, nn.Parameter)):
                obj.requires_grad = True

        # Always prefer enabling grads in text_encoder token-id fallback path.
        self.text_encoder.finetune = True

        # 1) Projection-only: proj/text_projection/ln_final where available.
        if mode == 'proj':
            for attr in ['proj', 'text_projection', 'ln_final']:
                if hasattr(text_model, attr):
                    _unfreeze_obj(getattr(text_model, attr))
            # Some open_clip wrappers keep projection on the parent model.
            for attr in ['text_projection']:
                if hasattr(text_clip, attr):
                    _unfreeze_obj(getattr(text_clip, attr))
            return

        # 2) Last-N layers: unfreeze proj + last N encoder layers.
        if mode == 'last':
            # Unfreeze proj first
            for attr in ['proj', 'text_projection', 'ln_final']:
                if hasattr(text_model, attr):
                    _unfreeze_obj(getattr(text_model, attr))

            transformer = getattr(text_model, 'transformer', None)
            # HF: BertModel has encoder.layer
            encoder = getattr(transformer, 'encoder', None) if transformer is not None else None
            layers = getattr(encoder, 'layer', None) if encoder is not None else None
            if layers is not None and hasattr(layers, '__len__'):
                n_layers = len(layers)
                n = max(0, min(int(last_n), n_layers))
                for i in range(n_layers - n, n_layers):
                    _unfreeze_module(layers[i])
            else:
                # Best-effort fallback: if we can't locate encoder layers, fall back to full.
                mode = 'full'

        # 3) Full: unfreeze all text-model parameters (still keep visual frozen).
        if mode == 'full':
            _unfreeze_module(text_model)
            # Also unfreeze projections on parent if present
            for attr in ['text_projection']:
                if hasattr(text_clip, attr):
                    _unfreeze_obj(getattr(text_clip, attr))
    
    def forward(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        """
        前向传播
        
        参数:
            x_s: 低分辨率patch特征 [1, N_s, 512]
            coord_s: 低分辨率坐标
            x_l: 高分辨率patch特征 [1, N_l, 512]
            coords_l: 高分辨率坐标
            label: 标签 [1]
        
        返回:
            Y_prob: 预测概率 [1, num_classes]
            Y_hat: 预测类别 [1, 1]
            loss: 交叉熵损失
        """
        # ========== 低分辨率分支 ==========
        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)
        
        H = compents.squeeze().float()
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, H)
        
        # ========== 高分辨率分支 ==========
        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)
        
        H_high = compents_high.squeeze().float()
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)

        # 生成文本特征（Prompt-learning + 可选文本微调）
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        prompt_embeddings = self.prompt_learner().to(x_s.device)
        eos_indices = getattr(self.prompt_learner, 'eos_indices', None)
        if eos_indices is not None:
            eos_indices = eos_indices.to(x_s.device)
        text_features = self.text_encoder(tokenized_prompts, prompt_embeddings=prompt_embeddings, eos_indices=eos_indices)  # [2*num_classes, 512]

        static_text_low = text_features[:self.num_classes]
        static_text_high = text_features[self.num_classes:]

        retrieval_debug = None
        if self.enable_dynamic_prompt and self.dynamic_prompt_runtime_enabled and self.dynamic_retriever is not None:
            low_pool_feats, high_pool_feats = self._get_dynamic_pool_features(x_s.device)
            text_features_low, text_features_high, retrieval_debug = self.dynamic_retriever(
                low_query=image_features_low,
                high_query=image_features_high,
                low_pool_feats_by_class=low_pool_feats,
                high_pool_feats_by_class=high_pool_feats,
                static_low=static_text_low,
                static_high=static_text_high,
                low_pool_texts_by_class=self._dynamic_low_pool_texts,
                high_pool_texts_by_class=self._dynamic_high_pool_texts,
            )
            self.last_retrieval_debug = retrieval_debug
        else:
            text_features_low = static_text_low
            text_features_high = static_text_high
            self.last_retrieval_debug = None

        dynamic_text_low = text_features_low
        dynamic_text_high = text_features_high

        if self.enable_rag_rewrite and self.rag_runtime_enabled and self.rag_rewriter is not None and slide_id is not None:
            sid = str(slide_id)
            rag_item = self.rag_rewriter.get_rewritten_prompts(sid, retrieval_debug=retrieval_debug)
            if rag_item is not None:
                low_texts = rag_item.get('low', [])
                high_texts = rag_item.get('high', [])
                rag_low = self._encode_sentence_list(low_texts).to(x_s.device)
                rag_high = self._encode_sentence_list(high_texts).to(x_s.device)
                if rag_low.shape == dynamic_text_low.shape and rag_high.shape == dynamic_text_high.shape:
                    text_features_low = rag_low
                    text_features_high = rag_high
                    self.last_rag_debug = {'slide_id': sid, 'used': True, 'source': rag_item.get('source', 'unknown')}
                else:
                    text_features_low = dynamic_text_low if self.rag_fallback == 'dynamic' else static_text_low
                    text_features_high = dynamic_text_high if self.rag_fallback == 'dynamic' else static_text_high
                    self.last_rag_debug = {'slide_id': sid, 'used': False, 'reason': 'shape_mismatch', 'fallback': self.rag_fallback}
            else:
                text_features_low = dynamic_text_low if self.rag_fallback == 'dynamic' else static_text_low
                text_features_high = dynamic_text_high if self.rag_fallback == 'dynamic' else static_text_high
                self.last_rag_debug = {'slide_id': sid, 'used': False, 'reason': 'rag_unavailable', 'fallback': self.rag_fallback}
        else:
            self.last_rag_debug = None

        if self.enable_vcp and self.vcp_runtime_enabled:
            text_features_low, text_features_high = self._apply_vcp(
                image_features_low, image_features_high, text_features_low, text_features_high
            )
        
        # ========== 文本-图像对齐 ==========
        image_context = torch.cat((compents.squeeze(), M.squeeze(0)), dim=0)
        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1), image_context, image_context
        )
        text_features_low = text_context_features.squeeze() + text_features_low
        
        image_context_high = torch.cat((compents_high.squeeze(), M_high.squeeze(0)), dim=0)
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high, image_context_high
        )
        text_features_high = text_context_features_high.squeeze() + text_features_high
        
        # ========== 分类 ==========
        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high
        
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        return Y_prob, Y_hat, loss
    
    def forward_with_attention(self, x_s, coord_s, x_l, coords_l, label, slide_id=None):
        """前向传播并返回注意力权重(用于热力图)"""
        M = x_s.float()
        compents, cross_attn_weights_s = self.cross_attention_1(
            self.learnable_image_center, M, M
        )
        compents = self.norm(compents + self.learnable_image_center)
        
        M_high = x_l.float()
        compents_high, cross_attn_weights_l = self.cross_attention_1(
            self.learnable_image_center, M_high, M_high
        )
        compents_high = self.norm(compents_high + self.learnable_image_center)
        
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

        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(x_s.device)
        prompt_embeddings = self.prompt_learner().to(x_s.device)
        eos_indices = getattr(self.prompt_learner, 'eos_indices', None)
        if eos_indices is not None:
            eos_indices = eos_indices.to(x_s.device)
        text_features = self.text_encoder(tokenized_prompts, prompt_embeddings=prompt_embeddings, eos_indices=eos_indices)

        static_text_low = text_features[:self.num_classes]
        static_text_high = text_features[self.num_classes:]

        retrieval_debug = None
        if self.enable_dynamic_prompt and self.dynamic_prompt_runtime_enabled and self.dynamic_retriever is not None:
            low_pool_feats, high_pool_feats = self._get_dynamic_pool_features(x_s.device)
            text_features_low, text_features_high, retrieval_debug = self.dynamic_retriever(
                low_query=image_features_low,
                high_query=image_features_high,
                low_pool_feats_by_class=low_pool_feats,
                high_pool_feats_by_class=high_pool_feats,
                static_low=static_text_low,
                static_high=static_text_high,
                low_pool_texts_by_class=self._dynamic_low_pool_texts,
                high_pool_texts_by_class=self._dynamic_high_pool_texts,
            )
            self.last_retrieval_debug = retrieval_debug
        else:
            text_features_low = static_text_low
            text_features_high = static_text_high
            self.last_retrieval_debug = None

        dynamic_text_low = text_features_low
        dynamic_text_high = text_features_high

        if self.enable_rag_rewrite and self.rag_runtime_enabled and self.rag_rewriter is not None and slide_id is not None:
            sid = str(slide_id)
            rag_item = self.rag_rewriter.get_rewritten_prompts(sid, retrieval_debug=retrieval_debug)
            if rag_item is not None:
                low_texts = rag_item.get('low', [])
                high_texts = rag_item.get('high', [])
                rag_low = self._encode_sentence_list(low_texts).to(x_s.device)
                rag_high = self._encode_sentence_list(high_texts).to(x_s.device)
                if rag_low.shape == dynamic_text_low.shape and rag_high.shape == dynamic_text_high.shape:
                    text_features_low = rag_low
                    text_features_high = rag_high
                    self.last_rag_debug = {'slide_id': sid, 'used': True, 'source': rag_item.get('source', 'unknown')}
                else:
                    text_features_low = dynamic_text_low if self.rag_fallback == 'dynamic' else static_text_low
                    text_features_high = dynamic_text_high if self.rag_fallback == 'dynamic' else static_text_high
                    self.last_rag_debug = {'slide_id': sid, 'used': False, 'reason': 'shape_mismatch', 'fallback': self.rag_fallback}
            else:
                text_features_low = dynamic_text_low if self.rag_fallback == 'dynamic' else static_text_low
                text_features_high = dynamic_text_high if self.rag_fallback == 'dynamic' else static_text_high
                self.last_rag_debug = {'slide_id': sid, 'used': False, 'reason': 'rag_unavailable', 'fallback': self.rag_fallback}
        else:
            self.last_rag_debug = None

        if self.enable_vcp and self.vcp_runtime_enabled:
            text_features_low, text_features_high = self._apply_vcp(
                image_features_low, image_features_high, text_features_low, text_features_high
            )
        
        image_context = torch.cat((compents.squeeze(), M.squeeze(0)), dim=0)
        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1), image_context, image_context
        )
        text_features_low = text_context_features.squeeze() + text_features_low
        
        image_context_high = torch.cat((compents_high.squeeze(), M_high.squeeze(0)), dim=0)
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high, image_context_high
        )
        text_features_high = text_context_features_high.squeeze() + text_features_high
        
        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        
        # 注意力权重
        patch_attention_s = cross_attn_weights_s.mean(dim=1).squeeze(0).mean(dim=0)
        patch_attention_l = cross_attn_weights_l.mean(dim=1).squeeze(0).mean(dim=0)
        
        return logits, Y_prob, Y_hat, patch_attention_s, patch_attention_l
