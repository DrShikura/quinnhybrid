"""
Phase 2B: Structural Transformer Decoder.

A transformer decoder that uses QuINN's waveform as structural guidance.
The decoder:
  1. Embeds tokens to a vector space
  2. Cross-attends to QuINN's waveform (structural guidance)
  3. Self-attends for local context
  4. Predicts next token

Architecture:
  - Embedding layer: tokens → vectors
  - Cross-attention blocks: attend to QuINN waveform
  - Self-attention blocks: intra-token dependencies
  - Output head: → logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StructuralTransformerDecoder(nn.Module):
    """
    Transformer decoder with cross-attention to QuINN waveform guidance.

    The model predicts tokens autoregressively while respecting structural
    constraints encoded in QuINN's waveform.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: number of tokens
            embed_dim: embedding dimension (must match QuINN's embed_dim)
            hidden_dim: feed-forward hidden dimension
            n_heads: number of attention heads
            n_layers: number of transformer layers
            max_seq_len: maximum sequence length
            dropout: dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Transformer layers with cross-attention to waveform
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output head: → logits
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, p in self.named_parameters():
            if "weight" in name and len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0)

    def forward(
        self,
        token_ids: torch.Tensor,  # (B, T) long
        positions: torch.Tensor,  # (B, T) long
        quinn_waveform: torch.Tensor,  # (B, L, D) complex – structural guidance
        mask: Optional[torch.Tensor] = None,  # (B, T) bool – valid positions
    ) -> torch.Tensor:
        """
        Forward pass: predict logits for each position.

        Args:
            token_ids: (B, T) token indices
            positions: (B, T) position indices
            quinn_waveform: (B, L, D) complex waveform from QuINN
            mask: (B, T) bool, True = valid, False = padding

        Returns:
            (B, T, vocab_size) logits
        """
        B, T = token_ids.shape

        # Token embeddings
        x = self.token_embed(token_ids)  # (B, T, D)
        x = x + self.pos_embed(positions)  # add positional embeddings
        x = self.dropout(x)

        # Convert Quinn waveform from complex to real
        # Simple approach: concatenate real and imag parts
        quinn_real = torch.cat([quinn_waveform.real, quinn_waveform.imag], dim=-1)  # (B, L, 2D)

        # Apply transformer layers with cross-attention to waveform
        for layer in self.layers:
            x = layer(
                x=x,
                kv=quinn_real,  # use waveform as key/value for cross-attention
                self_attn_mask=self._get_self_attn_mask(T, device=x.device),
                padding_mask=mask,
            )

        # Predict logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def _get_self_attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for self-attention (autoregressive).

        (i, j) = -inf if j > i (can't attend to future tokens)
        """
        mask = torch.ones(seq_len, seq_len, device=device)
        mask = torch.triu(mask, diagonal=1).bool()
        return mask.unsqueeze(0)  # (1, T, T) for broadcasting


class TransformerDecoderLayer(nn.Module):
    """Single transformer layer with cross-attention to guidance."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention on tokens
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        # Cross-attention to Quinn waveform
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D) query
        kv: torch.Tensor,  # (B, L, D') key/value for cross-attention (waveform)
        self_attn_mask: Optional[torch.Tensor] = None,  # (1, T, T) causal mask
        padding_mask: Optional[torch.Tensor] = None,  # (B, T) padding locations
    ) -> torch.Tensor:
        """Apply one transformer layer."""
        # Self-attention
        x_norm = self.norm1(x)
        x_attn, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=self_attn_mask.squeeze(0) if self_attn_mask is not None else None,
            key_padding_mask=~padding_mask if padding_mask is not None else None,
        )
        x = x + self.dropout(x_attn)

        # Cross-attention to waveform guidance
        x_norm = self.norm2(x)
        x_cross, _ = self.cross_attn(
            x_norm, kv, kv,
            key_padding_mask=None,  # waveform is always valid (no padding)
        )
        x = x + self.dropout(x_cross)

        # Feed-forward
        x_norm = self.norm3(x)
        x_ff = self.ff(x_norm)
        x = x + self.dropout(x_ff)

        return x
