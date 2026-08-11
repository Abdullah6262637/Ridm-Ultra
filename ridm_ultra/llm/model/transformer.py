"""RoPE + RMSNorm + SwiGLU + GQA kullanan decoder-only Transformer."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ModelConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # Paket importu temel RIDM'i bozmamalı.
    raise RuntimeError("LLM modeli için `pip install .[llm]` ile torch kurulmalıdır.") from exc


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE için head_dim çift olmalıdır.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        angles = torch.einsum("bt,d->btd", positions.float(), self.inv_freq)
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = even * cos - odd * sin
            out[..., 1::2] = even * sin + odd * cos
            return out
        return rotate(q), rotate(k)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None,
                use_cache: bool = False):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, positions)
        if past_key_value is not None:
            previous_k, previous_v = past_key_value
            if previous_k.shape[:2] != (batch, self.n_kv_heads):
                raise ValueError("KV-cache batch boyutu veya head sayısı uyumsuz.")
            k, v = torch.cat((previous_k, k), dim=2), torch.cat((previous_v, v), dim=2)
        cache = (k, v) if use_cache else None
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        # Cache sonrası q'nin mutlak konumu sıfır değildir. Bu nedenle causal
        # maskeyi geçmiş uzunluğunu hesaba katarak açıkça kuruyoruz.
        if past_key_value is None:
            mask, is_causal = None, True
        elif length == 1:
            mask, is_causal = None, False
        else:
            past_length = k.shape[2] - length
            query_positions = torch.arange(length, device=x.device)[:, None] + past_length
            key_positions = torch.arange(k.shape[2], device=x.device)[None, :]
            mask, is_causal = key_positions <= query_positions, False
            
        if attention_mask is not None:
            if mask is None and is_causal:
                query_positions = torch.arange(length, device=x.device)[:, None]
                key_positions = torch.arange(k.shape[2], device=x.device)[None, :]
                mask = key_positions <= query_positions
            
            mask = mask & attention_mask[:, None, None, :] if mask is not None else attention_mask[:, None, None, :]
            is_causal = False

        # PyTorch 2.x CUDA'da FlashAttention/memory-efficient kernel otomatik seçilir.
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
                                                  is_causal=is_causal)
        output = self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, -1))
        return output, cache


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None,
                use_cache: bool = False):
        attended, cache = self.attention(self.attention_norm(x), positions, past_key_value, attention_mask, use_cache)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x)), cache


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            nn.init.normal_(block.ffn.down.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
                attention_mask: torch.Tensor | None = None,
                use_cache: bool = False) -> CausalLMOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input_ids [batch, seq] olmalı ve max_seq_len'i aşmamalıdır.")
        if labels is not None and past_key_values is not None:
            raise ValueError("Kayıplı eğitim ile KV-cache aynı çağrıda kullanılmaz.")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("KV-cache katman sayısı modelle uyuşmuyor.")
        offset = past_key_values[0][0].shape[2] if past_key_values else 0
        if offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("KV-cache ile toplam bağlam max_seq_len'i aşıyor.")
        positions = (torch.arange(input_ids.shape[1], device=input_ids.device) + offset).unsqueeze(0).expand_as(input_ids)
        x = self.token_embedding(input_ids)
        next_cache = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            previous = past_key_values[index] if past_key_values else None
            if self.config.gradient_checkpointing and self.training and not use_cache:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(lambda state: block(state, positions, None, attention_mask, False)[0], x, use_reentrant=False)
            else:
                x, cached = block(x, positions, previous, attention_mask, use_cache)
                if use_cache:
                    next_cache.append(cached)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        return CausalLMOutput(logits=logits, loss=loss, past_key_values=tuple(next_cache) if next_cache is not None else None)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, *, max_new_tokens: int = 128, temperature: float = 0.8,
                 top_k: int | None = 50, top_p: float = 0.95, eos_id: int | None = None) -> torch.Tensor:
        """KV-cache kullanan batched nucleus/top-k örnekleme.

        İlk çağrı tüm prompt'u işler; sonraki her token sadece tek yeni token ve
        cache üzerinden hesaplanır. Bu, uzun üretimde O(T²) tekrarını önler.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("generate için boş olmayan [batch, seq] input gerekir.")
        if not 0 < temperature:
            raise ValueError("temperature pozitif olmalıdır.")
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        generated, finished = input_ids, torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        output = self(input_ids, attention_mask=attention_mask, use_cache=True)
        cache, logits = output.past_key_values, output.logits[:, -1]
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_seq_len:
                break
            scores = logits.float() / temperature
            if top_k is not None and 0 < top_k < scores.shape[-1]:
                threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
                scores = scores.masked_fill(scores < threshold, float("-inf"))
            if 0 < top_p < 1:
                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
                cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
                remove = cumulative - torch.softmax(sorted_scores, dim=-1) > top_p
                sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
                scores = torch.full_like(scores, float("-inf")).scatter(1, sorted_indices, sorted_scores)
            next_token = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)
            if eos_id is not None:
                next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_id), next_token)
                finished |= next_token.squeeze(1).eq(eos_id)
            generated = torch.cat((generated, next_token), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.bool, device=attention_mask.device)], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
            output = self(next_token, past_key_values=cache, attention_mask=attention_mask, use_cache=True)
            cache, logits = output.past_key_values, output.logits[:, -1]
        return generated
