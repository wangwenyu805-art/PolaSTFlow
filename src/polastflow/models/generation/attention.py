from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.layers import RMSNorm
from .rope import apply_rope

class PMFSelfAttention(nn.Module):
    """Apply rotary-positioned multi-head self-attention in Pixel MeanFlow.

    Queries and keys are RMS-normalized per head. Explicit attention
    operations preserve forward-mode automatic differentiation for the
    MeanFlow JVP objective.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be divisible by 4 for 2-D RoPE."
            )

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, rope_ids, rope_cos, rope_sin)
        k = apply_rope(k, rope_ids, rope_cos, rope_sin)

        # Float32 logits/softmax improve stability and are compatible with JVP.
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = torch.softmax(logits, dim=-1).to(dtype=v.dtype)
        if self.attn_drop > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        return self.proj_drop(self.out_proj(out))

class PMFCrossAttention(nn.Module):
    """Inject historical conditions into future trajectory tokens.

    Future-sequence tokens supply queries; historical condition tokens
    supply keys and values. Explicit projected attention is compatible
    with the forward-mode differentiation used by Pixel MeanFlow.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        query_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.query_chunk_size = int(query_chunk_size)

        # Normalize queries and historical keys/values before cross-attention.
        self.norm_q = RMSNorm(hidden_size, eps=1e-6)
        self.norm_kv = RMSNorm(hidden_size, eps=1e-6)

        # Separate Q/K/V projections support explicit multi-head attention
        # compatible with MeanFlow forward-mode automatic differentiation.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or condition.ndim != 3:
            raise ValueError("x and condition must be [B,N,D].")
        if x.shape[0] != condition.shape[0] or x.shape[-1] != condition.shape[-1]:
            raise ValueError(
                f"cross-attention shape mismatch: x={tuple(x.shape)}, "
                f"condition={tuple(condition.shape)}"
            )

        b, nq, d = x.shape
        nk = condition.shape[1]

        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(self.norm_kv(condition))
        v = self.v_proj(self.norm_kv(condition))

        q = q.view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        chunks = []
        step = max(self.query_chunk_size, 1)
        for q_chunk in q.split(step, dim=2):
            logits = torch.matmul(q_chunk.float(), k.float().transpose(-2, -1)) * scale
            attn = torch.softmax(logits, dim=-1).to(dtype=v.dtype)
            if self.attn_drop > 0 and self.training:
                attn = F.dropout(attn, p=self.attn_drop)
            chunks.append(torch.matmul(attn, v))

        out = torch.cat(chunks, dim=2)
        out = out.transpose(1, 2).contiguous().view(b, nq, d)
        return self.proj_drop(self.out_proj(out))
