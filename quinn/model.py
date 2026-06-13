"""
QuINN: Quantum Interference Neural Network – Phase 1 implementation.

Predicts the complete structural waveform of a sequence given only a prefix.
The "waveform" is a complex-valued per-position representation; the model
must infer the global structural skeleton from local partial evidence.

Architecture:
    1. TokenFrequencyEncoder:  tokens → complex sinusoidal waveform (per position)
    2. InterferenceManifold:   refines the waveform, amplifying structural signal
    3. PrefixAggregator:       mask-aware mean-pool → fixed-size summary vector
    4. WaveformDecoder:        (summary, target positions) → predicted waveform
    5. LengthHead:             summary magnitudes → predicted sequence length
"""

import math
import torch
import torch.nn as nn

from .encoding import ComplexTokenEmbedding, MultiScalePositionEncoding
from .manifold import ComplexLinear, InterferenceManifold, ComplexLayerNorm


class WaveformDecoder(nn.Module):
    """
    Decodes a fixed-size summary into a predicted waveform at arbitrary positions.

    At each target position p, the summary is combined with a multi-scale
    sinusoidal positional encoding and processed through a small manifold.
    This gives the model the ability to predict different structural states
    at each position conditioned on what the prefix implies about the whole.
    """

    def __init__(self, manifold_dim: int, embed_dim: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.max_seq_len = max_seq_len

        # Project real position features to complex manifold space
        pos_feat_dim = 64  # number of sinusoidal position features (real-valued)
        self.pos_proj_real = nn.Linear(pos_feat_dim, manifold_dim)
        self.pos_proj_imag = nn.Linear(pos_feat_dim, manifold_dim)

        # Project summary (complex) to decoder starting point
        self.summary_proj = ComplexLinear(manifold_dim, manifold_dim)

        # Decoder manifold
        from .manifold import InterferenceManifold
        self.manifold = InterferenceManifold(manifold_dim, n_layers)

        # Output projection: manifold_dim → embed_dim (the target waveform space)
        self.out_proj = ComplexLinear(manifold_dim, embed_dim)

    def forward(
        self,
        summary: torch.Tensor,          # (B, manifold_dim) complex
        target_positions: torch.Tensor, # (B, L) long
    ) -> torch.Tensor:
        """Returns: (B, L, embed_dim) complex"""
        B, L = target_positions.shape

        # Multi-scale sinusoidal position features (real)
        pos_feats = MultiScalePositionEncoding.encode(
            target_positions, dim=64
        )  # (B, L, 64)

        # Project to complex
        pos_complex = torch.complex(
            self.pos_proj_real(pos_feats),
            self.pos_proj_imag(pos_feats),
        )  # (B, L, manifold_dim)

        # Broadcast summary to every target position and combine
        summary_exp = self.summary_proj(
            summary.unsqueeze(1).expand(B, L, -1)
        )  # (B, L, manifold_dim)

        x = summary_exp + pos_complex  # (B, L, manifold_dim)

        x = self.manifold(x)           # (B, L, manifold_dim)
        return self.out_proj(x)        # (B, L, embed_dim)


class QuINN(nn.Module):
    """
    Full QuINN model for Phase 1: waveform completion.

    Training usage:
        pred_waveform, true_waveform, pred_log_len = model(
            prefix_tokens, prefix_positions, prefix_mask,
            target_tokens, target_positions,
        )

    Inference (length estimation only):
        _, _, pred_log_len = model(
            prefix_tokens, prefix_positions, prefix_mask,
        )
        pred_len = exp(pred_log_len)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        manifold_dim: int = 128,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.max_seq_len = max_seq_len

        # ── Encoder ──────────────────────────────────────────────────────
        self.token_encoder = ComplexTokenEmbedding(vocab_size, embed_dim, max_seq_len)
        self.embed_proj = ComplexLinear(embed_dim, manifold_dim)
        self.encoder_manifold = InterferenceManifold(manifold_dim, n_encoder_layers)

        # ── Aggregator ───────────────────────────────────────────────────
        # Mask-aware complex mean-pool; no learned parameters needed here.
        # The manifold already did all the heavy lifting.
        self.summary_norm = ComplexLayerNorm(manifold_dim)

        # ── Decoder ──────────────────────────────────────────────────────
        self.decoder = WaveformDecoder(
            manifold_dim, embed_dim, n_decoder_layers, max_seq_len
        )

        # ── Length head ──────────────────────────────────────────────────
        # Takes |summary| (real magnitudes) → log(seq_len)
        # Using magnitudes discards phase information so the head focuses on
        # "how much structural content" rather than "what kind".
        self.length_head = nn.Sequential(
            nn.Linear(manifold_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _encode_and_aggregate(
        self,
        tokens: torch.Tensor,     # (B, L) long
        positions: torch.Tensor,  # (B, L) long
        mask: torch.Tensor,       # (B, L) bool – True = valid (not padding)
    ) -> torch.Tensor:            # (B, manifold_dim) complex
        """Encode prefix through manifold then mask-mean-pool to summary."""
        waveform = self.token_encoder(tokens, positions)    # (B, L, embed_dim) complex
        projected = self.embed_proj(waveform)              # (B, L, manifold_dim)
        processed = self.encoder_manifold(projected)       # (B, L, manifold_dim)
        processed = self.summary_norm(processed)

        # Masked mean pooling in complex space
        mask_f = mask.float().unsqueeze(-1)                # (B, L, 1)
        counts = mask_f.sum(dim=1).clamp(min=1)           # (B, 1)
        summary_real = (processed.real * mask_f).sum(dim=1) / counts
        summary_imag = (processed.imag * mask_f).sum(dim=1) / counts
        return torch.complex(summary_real, summary_imag)   # (B, manifold_dim)

    def _compute_true_waveform(
        self,
        tokens: torch.Tensor,     # (B, L) long
        positions: torch.Tensor,  # (B, L) long
    ) -> torch.Tensor:            # (B, L, embed_dim) complex
        """
        True waveform: raw complex token embeddings at each target position.
        Stop-gradient is applied so that the encoder only learns through the
        prefix prediction path, not by making the target easier to match —
        preventing a degenerate shortcut and reducing backward-pass cost by ~2.6x.
        """
        with torch.no_grad():
            return self.token_encoder(tokens, positions)

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        prefix_tokens: torch.Tensor,           # (B, P) long
        prefix_positions: torch.Tensor,        # (B, P) long
        prefix_mask: torch.Tensor,             # (B, P) bool
        target_tokens: torch.Tensor = None,    # (B, T) long  – full sequence tokens
        target_positions: torch.Tensor = None, # (B, T) long  – full sequence positions
    ):
        """
        Returns:
            predicted_waveform:  (B, T, embed_dim) complex  or None
            true_waveform:       (B, T, embed_dim) complex  or None
            pred_log_len:        (B,) float – log of predicted sequence length
        """
        summary = self._encode_and_aggregate(prefix_tokens, prefix_positions, prefix_mask)

        # Predicted waveform at target positions
        predicted_waveform = None
        true_waveform = None
        if target_positions is not None:
            predicted_waveform = self.decoder(summary, target_positions)
            if target_tokens is not None:
                true_waveform = self._compute_true_waveform(target_tokens, target_positions)

        # Length prediction from summary magnitudes
        pred_log_len = self.length_head(summary.abs()).squeeze(-1)  # (B,)

        return predicted_waveform, true_waveform, pred_log_len

    def estimate_length(
        self,
        prefix_tokens: torch.Tensor,
        prefix_positions: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: returns estimated sequence length (not log)."""
        _, _, pred_log_len = self.forward(prefix_tokens, prefix_positions, prefix_mask)
        return torch.exp(pred_log_len)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
