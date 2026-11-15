import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention for baseline comparison."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Causal lower-triangular mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context output
        context = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(context)

class TransformerBlock(nn.Module):
    """Transformer decoder block with LayerNorm and FeedForward."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BaselineTransformer(nn.Module):
    """
    Matched Causal Transformer Baseline to compare against DALI-1.
    Uses standard self-attention KV caching mechanism.
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return {
            "logits": logits,
            "loss": loss
        }
