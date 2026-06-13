"""
Token-to-waveform encoding for QuINN.

Each token type is assigned learned sinusoidal frequencies. The complex-valued
encoding means structurally systematic co-occurrences interfere constructively
(phases align) while coincidental co-occurrences cancel out (phases random).
"""

import math
import torch
import torch.nn as nn


class ComplexTokenEmbedding(nn.Module):
    """
    Embeds tokens as complex sinusoidal waves with learned frequencies.

    For token t at position p:
        enc(t, p) = (A_t_real + i*A_t_imag) * exp(i * f_t * p / scale)

    where f_t (per-token frequency) and A_t (per-token complex amplitude)
    are both learned. Tokens with structural roles learn frequencies that
    produce coherent interference; coincidental co-occurrences learn
    incoherent frequencies that cancel on aggregation.
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Learned complex base amplitude for each token (real and imag parts)
        self.amplitude_real = nn.Embedding(vocab_size, embed_dim)
        self.amplitude_imag = nn.Embedding(vocab_size, embed_dim)

        # Learned log-frequency for each token × embed_dim frequency channels
        # Initialised with multiple scales to encourage multi-resolution encoding
        log_freqs_init = torch.randn(vocab_size, embed_dim) * 0.3
        self.log_frequencies = nn.Parameter(log_freqs_init)

        nn.init.normal_(self.amplitude_real.weight, std=0.02)
        nn.init.normal_(self.amplitude_imag.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens:    (B, L) long  – token indices
            positions: (B, L) long  – absolute position indices

        Returns:
            complex tensor of shape (B, L, embed_dim)
        """
        # (B, L, E)
        a_real = self.amplitude_real(tokens)
        a_imag = self.amplitude_imag(tokens)

        # Per-token frequencies (always positive via exp)
        freqs = torch.exp(self.log_frequencies[tokens])  # (B, L, E)

        # Phase angle: frequency × normalised position
        pos_norm = positions.float() / self.max_seq_len * 2 * math.pi  # (B, L)
        angles = freqs * pos_norm.unsqueeze(-1)  # (B, L, E)

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        # Complex multiply: (a_r + i*a_i) * (cos + i*sin)
        out_real = a_real * cos_a - a_imag * sin_a
        out_imag = a_real * sin_a + a_imag * cos_a

        return torch.complex(out_real, out_imag)  # (B, L, E)


class MultiScalePositionEncoding:
    """
    Static helper: sinusoidal position encodings at multiple frequency scales.
    Low-frequency components capture document-level structure;
    high-frequency components capture token-level micro-structure.
    These superimpose without interference — exactly what complex-valued
    representation is designed to hold simultaneously.
    """

    @staticmethod
    def encode(positions: torch.Tensor, dim: int) -> torch.Tensor:
        """
        positions: (B, L) long
        Returns: (B, L, dim) float (real), pairs of sin/cos at log-spaced freqs
        """
        device = positions.device
        half = dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
        p = positions.float().unsqueeze(-1)          # (B, L, 1)
        angles = p * freqs.unsqueeze(0).unsqueeze(0)  # (B, L, half)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, L, dim)
