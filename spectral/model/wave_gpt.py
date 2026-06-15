"""
WaveGPT: Phase-aware causal transformer for structural token prediction.

Adds a sinusoidal scope-tracking signal (phase accumulator) to a clean causal
GPT. INDENT / DEDENT tokens increment / decrement a running phase, encoding
the current nesting depth explicitly at every position — no stack memory
required, no auxiliary losses, no Python-loop bottlenecks.

Architecture choices (empirically grounded at sub-100K scale):
  ALiBi positional bias  — beats sinusoidal PE at 28M–125M (Press et al. 2022)
  RMSNorm                — equivalent quality, 7–64% faster (Zhang & Sennrich 2019)
  No linear biases       — "a bit better and faster" (Karpathy, nanoGPT)
  ff_mult = 3 (default)  — 4× is oversized for a 74-token vocabulary
  Tied embedding / head  — helps small models (Press & Wolf 2017)
  std=0.02 weight init   — avoids epoch-1 loss spike from default N(0,1)

Default config (~50K params, 2 independent layers, less than baseline-1l at 55K):
    d_model=48, n_layers=2, n_heads=4, ff_mult=3, max_seq_len=256
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_INDENT_ID = 68   # PythonStructuralTokenizer vocab position — stable
_DEDENT_ID = 69


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


def _make_alibi_bias(n_heads: int, max_seq_len: int) -> torch.Tensor:
    """
    ALiBi + causal mask, shape (n_heads, T, T).
    bias[h, i, j] = -slope_h * (i-j) for j<=i, -inf for j>i.
    """
    slopes = 2.0 ** (
        -(8.0 / n_heads) * torch.arange(1, n_heads + 1, dtype=torch.float)
    )
    T    = max_seq_len
    pos  = torch.arange(T, dtype=torch.float)
    dist = torch.clamp(pos.unsqueeze(1) - pos.unsqueeze(0), min=0)  # (T,T)
    alibi  = -slopes.view(n_heads, 1, 1) * dist.unsqueeze(0)        # (H,T,T)
    causal = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
    return alibi + causal.unsqueeze(0)


class _WaveBlock(nn.Module):
    """
    Pre-norm transformer block with manual multi-head attention.
    Manual attention is required for per-head ALiBi slopes.
    """

    def __init__(self, d: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0, f"d_model={d} must be divisible by n_heads={n_heads}"
        self.n_heads  = n_heads
        self.head_dim = d // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm1  = RMSNorm(d)
        self.norm2  = RMSNorm(d)
        self.qkv    = nn.Linear(d, 3 * d, bias=False)
        self.o_proj = nn.Linear(d, d,     bias=False)
        self.ff_up  = nn.Linear(d, ff_mult * d, bias=False)
        self.ff_dn  = nn.Linear(ff_mult * d, d, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # bias: (n_heads, T, T)  ALiBi + causal mask
        B, T, d = x.shape
        H, D    = self.n_heads, self.head_dim

        # Attention (pre-norm)
        xn  = self.norm1(x)
        qkv = self.qkv(xn).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                    # each (B,H,T,D)

        scores = (q @ k.transpose(-2, -1)) * self.scale            # (B,H,T,T)
        scores = scores + bias                                      # broadcast over batch
        attn   = self.drop(torch.softmax(scores, dim=-1))
        out    = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        x = x + self.drop(self.o_proj(out))

        # Feed-forward (pre-norm, GELU)
        xn = self.norm2(x)
        x  = x + self.drop(self.ff_dn(self.drop(F.gelu(self.ff_up(xn)))))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class WaveGPT(nn.Module):
    """
    Phase-aware causal GPT.

    Args:
        vocab_size:  number of tokens
        d_model:     embedding / hidden dimension (default 48)
        n_heads:     attention heads (default 4, head_dim = d_model / n_heads)
        n_layers:    number of independent transformer blocks (default 2)
        max_seq_len: maximum sequence length
        ff_mult:     feedforward hidden = ff_mult × d_model (default 3)
        dropout:     dropout rate
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 48,
        n_heads:     int   = 4,
        n_layers:    int   = 2,
        max_seq_len: int   = 256,
        ff_mult:     int   = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.n_loops     = n_layers       # trainer compatibility

        self.tok_emb    = nn.Embedding(vocab_size, d_model)
        self.phase_proj = nn.Linear(4, d_model, bias=False)
        self.drop       = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _WaveBlock(d_model, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # tied weights

        # Precomputed ALiBi + causal mask — moves to device with model.to()
        self.register_buffer('attn_bias', _make_alibi_bias(n_heads, max_seq_len))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _phase_enc(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Causal running-phase encoding derived from INDENT / DEDENT tokens.

        Each INDENT shifts phase +π/4; each DEDENT shifts phase -π/4.
        Position t's encoding uses only tokens at positions 0..t-1, so
        the signal is strictly causal.

        The four-component encoding [sin(φ), cos(φ), sin(2φ), cos(2φ)]
        lets the model read both nesting depth (φ magnitude) and whether
        two positions share the same scope (phase difference = 0).
        """
        delta    = ((tokens == _INDENT_ID).float() * (math.pi / 4)
                    - (tokens == _DEDENT_ID).float() * (math.pi / 4))
        phi_incl = torch.cumsum(delta, dim=1)                       # (B, T) inclusive
        # Shift right by one to make it causal: position t sees 0..t-1
        phi = torch.cat([
            torch.zeros(tokens.size(0), 1, device=tokens.device, dtype=delta.dtype),
            phi_incl[:, :-1],
        ], dim=1)                                                    # (B, T)
        features = torch.stack([
            torch.sin(phi), torch.cos(phi),
            torch.sin(2 * phi), torch.cos(2 * phi),
        ], dim=-1)                                                   # (B, T, 4)
        return self.phase_proj(features)                             # (B, T, d_model)

    def forward(
        self,
        tokens:    torch.Tensor,          # (B, T) long
        positions: torch.Tensor = None,   # unused; ALiBi handles position
        mode:      str          = 'lm',
    ) -> dict:
        B, T = tokens.shape
        x    = self.drop(self.tok_emb(tokens) + self._phase_enc(tokens))
        bias = self.attn_bias[:, :T, :T]                            # (H, T, T)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x)
        return {'lm_logits': self.lm_head(x), 'hidden': x}

    def param_count(self) -> dict:
        total       = sum(p.numel() for p in self.parameters())
        embedding   = self.tok_emb.weight.numel()
        transformer = (sum(p.numel() for p in self.layers.parameters())
                       + sum(p.numel() for p in self.norm.parameters()))
        phase       = sum(p.numel() for p in self.phase_proj.parameters())
        return {
            'total': total, 'embedding': embedding,
            'transformer': transformer, 'memory': 0, 'heads': phase,
        }
