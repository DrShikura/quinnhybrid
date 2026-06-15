"""
Baseline GPT-style causal transformer for comparison with SpectralLM.

Standard pre-norm decoder: token embedding + sinusoidal PE → N transformer
blocks → LayerNorm → tied LM head. No spectral structure, no entity memory,
no auxiliary losses — pure LM.

Configs matching SpectralLM's ~63K param budget:
    baseline-1l: d=64, n_layers=1  → ~58K params
    baseline-2l: d=48, n_layers=2  → ~61K params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pe(max_len: int, d: int) -> torch.Tensor:
    pos   = torch.arange(max_len).unsqueeze(1).float()
    dim   = torch.arange(0, d, 2).float()
    angle = pos / (10000 ** (dim / d))
    pe    = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle[:, :d // 2])
    return pe


class _CausalBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                            batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn, attn_mask=mask,
                                need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class BaselineGPT(nn.Module):
    """
    Standard causal GPT at the same parameter budget as SpectralLM.

    Args:
        vocab_size: number of tokens
        d_model:    embedding / hidden dimension
        n_heads:    attention heads
        n_layers:   number of independent transformer blocks
        max_seq_len: maximum sequence length
        dropout:    dropout rate
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int = 64,
        n_heads:     int = 4,
        n_layers:    int = 1,
        max_seq_len: int = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_loops     = n_layers     # trainer compatibility
        self.max_seq_len = max_seq_len  # generate() compatibility

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer('pe', _sinusoidal_pe(max_seq_len, d_model))
        self.drop    = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)

        # LM head tied to token embedding (no extra params)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        tokens:    torch.Tensor,    # (B, T) long
        positions: torch.Tensor,    # (B, T) long  — unused, sinusoidal PE is fixed
        mode:      str = 'lm',
    ) -> dict:
        B, T = tokens.shape
        x = self.drop(self.tok_emb(tokens) + self.pe[:T])

        mask = torch.triu(torch.ones(T, T, device=tokens.device,
                                     dtype=x.dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        return {'lm_logits': self.lm_head(x), 'hidden': x}

    def param_count(self) -> dict:
        # tok_emb and lm_head share weights — count once via model.parameters()
        total       = sum(p.numel() for p in self.parameters())
        embedding   = self.tok_emb.weight.numel()
        transformer = (sum(p.numel() for p in self.layers.parameters())
                       + sum(p.numel() for p in self.norm.parameters()))
        return {
            'total': total, 'embedding': embedding,
            'transformer': transformer, 'memory': 0, 'heads': 0,
        }
