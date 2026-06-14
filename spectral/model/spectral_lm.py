"""
SpectralLM: Unified spectral language model.

One model. No separate Quinn + Transformer. The waveform IS the representation.

Architecture:
    Input tokens
        ↓
    SpectralEmbedding         character-composed init + MoPE basis
        ↓
    SpectralTransformer       multi-head attention in [real|imag] frequency space
        causal self-attention: reads the combined waveform
        each head can specialize to a frequency band
        ↓
    WaveformHead              predicts next-token waveform (completion task)
    LanguageModelHead         predicts next-token id (LM task)

Training modes:
    'waveform': waveform completion pre-training (learns frequency structure)
    'lm':       language modeling (learns token prediction)
    'joint':    both losses simultaneously
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import SpectralEmbedding


class SpectralTransformerLayer(nn.Module):
    """
    One Transformer layer operating on [real|imag] concatenated waveform.

    dim_in  = 2 * embed_dim  (real + imag concatenated)
    n_heads must divide dim_in evenly.

    Attention heads can specialize to frequency bands because the real and imag
    parts of each frequency dimension are laid out contiguously: the first
    embed_dim values are all real parts, the next embed_dim are all imag parts.
    """

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0

        self.attn  = nn.MultiheadAttention(dim, n_heads,
                                            dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with causal mask
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=causal_mask,
                          is_causal=(causal_mask is not None))
        x = x + residual

        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class SpectralLM(nn.Module):
    """
    Unified spectral language model.

    Args:
        vocab:         list of token strings (for character init)
        embed_dim:     per-modality embedding dim (output is 2*embed_dim)
        n_layers:      number of Transformer layers
        n_heads:       attention heads (must divide 2*embed_dim)
        max_seq_len:   maximum sequence length
        dropout:       dropout rate
    """

    def __init__(
        self,
        vocab:        list,
        embed_dim:    int = 64,
        n_layers:     int = 4,
        n_heads:      int = 8,
        max_seq_len:  int = 512,
        dropout:      float = 0.1,
        band_init:    bool = True,
        acoustic_init: bool = True,
    ):
        super().__init__()

        self.vocab_size  = len(vocab)
        self.embed_dim   = embed_dim
        self.dim         = 2 * embed_dim   # real + imag concatenated

        assert self.dim % n_heads == 0, \
            f"2*embed_dim ({self.dim}) must be divisible by n_heads ({n_heads})"

        # ── Embedding ────────────────────────────────────────────────────────
        self.embedding = SpectralEmbedding(
            vocab_size    = self.vocab_size,
            embed_dim     = embed_dim,
            max_seq_len   = max_seq_len,
            vocab         = vocab,
            band_init     = band_init,
            acoustic_init = acoustic_init,
        )

        # ── Transformer ──────────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            SpectralTransformerLayer(self.dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(self.dim)

        # ── Heads ────────────────────────────────────────────────────────────
        # Waveform head: predict next token's amplitude vector (embed_dim, not 2*embed_dim).
        # Target is amp[t+1], not the full waveform — cosine loss on amplitude is
        # meaningful (different tokens have different spectral fingerprints).
        # Predicting the full waveform after F.normalize degenerates to phase prediction.
        self.waveform_head = nn.Linear(self.dim, embed_dim)

        # LM head: predict next-token id
        self.lm_head = nn.Linear(self.dim, self.vocab_size)

        self._init_output_heads()

    def _init_output_heads(self):
        nn.init.normal_(self.waveform_head.weight, std=0.02)
        nn.init.zeros_(self.waveform_head.bias)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for causal attention."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(
        self,
        tokens:    torch.Tensor,   # (B, T) long
        positions: torch.Tensor,   # (B, T) long
        mode:      str = 'lm',     # 'lm', 'waveform', or 'joint'
    ) -> dict:
        """
        Forward pass.

        Returns dict with:
            'hidden':    (B, T, dim)      — final hidden states
            'lm_logits': (B, T, vocab)    — next-token logits (if mode includes lm)
            'waveform':  (B, T, dim)      — predicted next waveform (if waveform mode)
            'embedding': (B, T, dim)      — input waveform (for completion target)
        """
        B, T = tokens.shape

        # Spectral embedding: (B, T, 2*embed_dim)
        x = self.embedding(tokens, positions)
        input_waveform = x.detach()   # save for waveform completion target

        # Causal Transformer
        causal_mask = self._causal_mask(T, tokens.device)
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.norm(x)

        out = {'hidden': x, 'embedding': input_waveform}

        if mode in ('lm', 'joint'):
            out['lm_logits'] = self.lm_head(x)         # (B, T, vocab)

        if mode in ('waveform', 'joint'):
            out['waveform'] = self.waveform_head(x)    # (B, T, dim)

        return out

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        embed = sum(p.numel() for p in self.embedding.parameters())
        transformer = sum(p.numel() for p in self.layers.parameters())
        heads = (sum(p.numel() for p in self.waveform_head.parameters()) +
                 sum(p.numel() for p in self.lm_head.parameters()))
        return {
            'total': total,
            'embedding': embed,
            'transformer': transformer,
            'heads': heads,
        }
