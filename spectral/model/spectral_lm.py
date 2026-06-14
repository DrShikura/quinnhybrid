"""
SpectralLM: Unified spectral language model.

Architecture:
    Input tokens
        ↓
    SpectralEmbedding       Laplacian eigenvector init + MoPE sinusoidal PE
        ↓
    SpectralTransformerLayer × n_layers
        causal self-attention with per-head locality distance bias
        (structural heads → near-global attention; semantic heads → local)
        ↓
    WaveformHead            predicts next-token amplitude (waveform completion)
    LanguageModelHead       predicts next-token id (LM)

Training modes:
    'waveform': waveform completion pre-training
    'lm':       language modeling
    'joint':    both losses simultaneously
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import SpectralEmbedding


class SpectralTransformerLayer(nn.Module):
    """
    Transformer layer with learned per-head locality distance bias.

    Standard causal self-attention scores are augmented with a penalty term:
        score[b, h, i, j] -= |pos_i - pos_j| / sigma_h

    sigma_h is a per-head learnable scale initialized from the embedding's
    log_sigma (which encodes the band structure: structural dims get large sigma
    → near-zero penalty → global attention; semantic dims get small sigma →
    large penalty → highly local attention).

    Each head h attends to embedding dimensions [h*half : (h+1)*half] in both
    the real and imaginary halves (half = embed_dim // n_heads).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        log_sigma_init: torch.Tensor,   # (embed_dim,) from SpectralEmbedding.log_sigma
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = math.sqrt(self.head_dim)

        # Combined QKV + output projections
        self.in_proj   = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj  = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

        # Per-head locality scale — initialized from embedding band structure,
        # then trained independently from SpectralEmbedding.log_sigma.
        # half = embed_dim // n_heads: head h covers real dims [h*half:(h+1)*half].
        half = self.head_dim // 2
        with torch.no_grad():
            sigma_init  = torch.exp(log_sigma_init.detach())   # (embed_dim,)
            per_head_sigma = torch.stack([
                sigma_init[h * half : (h + 1) * half].mean()
                for h in range(n_heads)
            ])
        self.log_sigma_head = nn.Parameter(torch.log(per_head_sigma))  # (n_heads,)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor,
                causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        H, HD   = self.n_heads, self.head_dim

        # Pre-norm
        residual = x
        x = self.norm1(x)

        # QKV projections → split heads
        qkv = self.in_proj(x)                               # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, HD).transpose(1, 2)            # (B, H, T, HD)
        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Locality bias: -(|pos_i − pos_j| / σ_h) — penalizes distant attention
        pos   = torch.arange(T, device=x.device).float()
        dist  = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()          # (T, T)
        sigma = torch.exp(self.log_sigma_head)                        # (H,)
        bias  = -(dist.unsqueeze(0) / (sigma.view(H, 1, 1) + 1e-6)) # (H, T, T)
        scores = scores + bias.unsqueeze(0)                           # (B, H, T, T)

        # Causal mask (upper-triangular -inf)
        if causal_mask is not None:
            scores = scores + causal_mask.view(1, 1, T, T)

        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum → merge heads
        out = torch.matmul(attn, v)                                   # (B, H, T, HD)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x   = residual + self.out_proj(out)

        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class SpectralLM(nn.Module):
    """
    Unified spectral language model.

    Args:
        vocab:             list of token strings (for acoustic fallback init)
        embed_dim:         per-modality embedding dim (output is 2*embed_dim)
        n_layers:          number of Transformer layers
        n_heads:           attention heads (must divide 2*embed_dim)
        max_seq_len:       maximum sequence length
        dropout:           dropout rate
        band_init:         if True, initialize freq[] in three bands
        acoustic_init:     if True (and no Laplacian eigvecs), use acoustic amplitude init
        laplacian_eigvecs: (vocab_size, vocab_size) from SpectralDataset; if provided,
                           uses Laplacian eigenvectors for amplitude init
    """

    def __init__(
        self,
        vocab:             list,
        embed_dim:         int = 64,
        n_layers:          int = 4,
        n_heads:           int = 8,
        max_seq_len:       int = 512,
        dropout:           float = 0.1,
        band_init:         bool = True,
        acoustic_init:     bool = True,
        laplacian_eigvecs: 'torch.Tensor | None' = None,
    ):
        super().__init__()

        self.vocab_size = len(vocab)
        self.embed_dim  = embed_dim
        self.dim        = 2 * embed_dim   # real + imag concatenated

        assert self.dim % n_heads == 0, \
            f"2*embed_dim ({self.dim}) must be divisible by n_heads ({n_heads})"

        # ── Embedding ────────────────────────────────────────────────────────
        self.embedding = SpectralEmbedding(
            vocab_size        = self.vocab_size,
            embed_dim         = embed_dim,
            max_seq_len       = max_seq_len,
            vocab             = vocab,
            band_init         = band_init,
            acoustic_init     = acoustic_init,
            laplacian_eigvecs = laplacian_eigvecs,
        )

        # ── Transformer — locality bias init from embedding's log_sigma ──────
        log_sigma_init = self.embedding.log_sigma.data.clone()
        self.layers = nn.ModuleList([
            SpectralTransformerLayer(
                dim            = self.dim,
                n_heads        = n_heads,
                log_sigma_init = log_sigma_init,
                dropout        = dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(self.dim)

        # ── Heads ────────────────────────────────────────────────────────────
        # Waveform head: predict next token's amplitude vector (D, not 2D).
        # Amplitude target (not full waveform) avoids phase-prediction degeneracy.
        self.waveform_head = nn.Linear(self.dim, embed_dim)
        self.lm_head       = nn.Linear(self.dim, self.vocab_size)

        self._init_output_heads()

    def _init_output_heads(self):
        nn.init.normal_(self.waveform_head.weight, std=0.02)
        nn.init.zeros_(self.waveform_head.bias)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular -inf mask for causal attention. Shape: (T, T)."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(
        self,
        tokens:    torch.Tensor,   # (B, T) long
        positions: torch.Tensor,   # (B, T) long
        mode:      str = 'lm',
    ) -> dict:
        """
        Returns dict with:
            'hidden':    (B, T, dim)      final hidden states
            'lm_logits': (B, T, vocab)    next-token logits  (if mode includes lm)
            'waveform':  (B, T, D)        predicted next amplitude (if waveform mode)
            'embedding': (B, T, 2D)       input waveform (kept in graph for grad loss)
        """
        B, T = tokens.shape

        # Spectral embedding — kept in computation graph for gradient consistency loss
        x = self.embedding(tokens, positions)
        embedding_out = x

        # Causal Transformer with locality bias
        causal_mask = self._causal_mask(T, tokens.device)
        for layer in self.layers:
            x = layer(x, causal_mask)
        x = self.norm(x)

        out = {'hidden': x, 'embedding': embedding_out}

        if mode in ('lm', 'joint'):
            out['lm_logits'] = self.lm_head(x)         # (B, T, vocab)

        if mode in ('waveform', 'joint'):
            out['waveform'] = self.waveform_head(x)    # (B, T, D)

        return out

    def param_count(self) -> dict:
        total       = sum(p.numel() for p in self.parameters())
        embed       = sum(p.numel() for p in self.embedding.parameters())
        transformer = sum(p.numel() for p in self.layers.parameters())
        heads       = (sum(p.numel() for p in self.waveform_head.parameters()) +
                       sum(p.numel() for p in self.lm_head.parameters()))
        return {'total': total, 'embedding': embed,
                'transformer': transformer, 'heads': heads}
