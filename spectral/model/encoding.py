"""
SpectralEmbedding: character-composed token initialization + learned sinusoidal PE.

Design principles:
  1. Character-level composition (optional, --no-acoustic-init to disable):
       Each token's amplitude profile is derived from the characters that spell it.
       Characters map to frequency bands by acoustic class (vowel/stop/fricative/etc).
       This gives every token a unique spectral fingerprint before any training.

  2. Learned per-dimension frequency (MoPE-inspired):
       Each embedding dimension d has ONE learned parameter:
         f[d]:  frequency — how fast phase advances with position
       basis(pos, d) = amp[d] * exp(i * f[d] * pos / max_len * 2π)

       The Gaussian locality envelope (Morlet) has been removed. It was centered at
       position 0, making the semantic band (small init σ) essentially dead past
       position ~50 — a bug, not a feature. Multi-scale locality should live in the
       attention mechanism (ALiBi-style), not the embedding.

  3. Frequency band partitioning (optional, --no-band-init to disable):
       dims  0 .. STRUCTURAL_END:   structural band  (low freq, global patterns)
       dims  STRUCTURAL_END .. EXPR_END:  expression band  (mid freq)
       dims  EXPR_END .. embed_dim:  semantic band  (high freq, local patterns)

       Without band_init, all frequencies initialized uniformly across [0.02, 0.50]
       — use this to test whether band structure self-organizes under LM pressure.
"""

import math
import torch
import torch.nn as nn


# ── Character acoustic classes ─────────────────────────────────────────────────

_VOWELS      = set("aeiouAEIOU")
_STOPS       = set("bdgkptBDGKPT")
_FRICATIVES  = set("fsvzFSVZ")
_SONORANTS   = set("lmnrLMNR")
_APPROXIMANT = set("wyhWYH")

_CHAR_BAND_WEIGHTS = {
    "vowel":       (0.6, 0.3, 0.1),
    "sonorant":    (0.3, 0.5, 0.2),
    "approximant": (0.5, 0.3, 0.2),
    "stop":        (0.1, 0.2, 0.7),
    "fricative":   (0.1, 0.2, 0.7),
    "digit":       (0.2, 0.4, 0.4),
    "other":       (0.3, 0.4, 0.3),
}


def _char_class(ch: str) -> str:
    if ch in _VOWELS:      return "vowel"
    if ch in _SONORANTS:   return "sonorant"
    if ch in _APPROXIMANT: return "approximant"
    if ch in _STOPS:       return "stop"
    if ch in _FRICATIVES:  return "fricative"
    if ch.isdigit():       return "digit"
    return "other"


def char_spectral_profile(token_str: str, embed_dim: int,
                           structural_end: int, expr_end: int) -> torch.Tensor:
    """
    Compute a spectral amplitude profile for a token from its characters.
    Returns (embed_dim,) real tensor. Note: this provides useful signal for
    alphabetic keywords but is essentially flat for operators (`:=`, `->`, `**`).
    """
    if not token_str or token_str.startswith("<"):
        return torch.ones(embed_dim) * 0.02

    structural_w = expr_w = semantic_w = 0.0
    for ch in token_str:
        sw, ew, semw = _CHAR_BAND_WEIGHTS[_char_class(ch)]
        structural_w += sw
        expr_w       += ew
        semantic_w   += semw

    n = len(token_str)
    structural_w /= n
    expr_w       /= n
    semantic_w   /= n

    profile = torch.zeros(embed_dim)
    n_s = structural_end
    n_e = expr_end - structural_end
    n_m = embed_dim - expr_end

    if n_s > 0: profile[:structural_end]         = structural_w / math.sqrt(n_s)
    if n_e > 0: profile[structural_end:expr_end] = expr_w       / math.sqrt(n_e)
    if n_m > 0: profile[expr_end:]               = semantic_w   / math.sqrt(n_m)

    return profile * 0.02


class SpectralEmbedding(nn.Module):
    """
    Token embedding with learned amplitude per token and learned sinusoidal PE.

    For each token t at position p and embedding dimension d:
        real[d] = amp[t,d] * cos(f[d] * p / max_len * 2π)
        imag[d] = amp[t,d] * sin(f[d] * p / max_len * 2π)

    Output: (B, T, 2*embed_dim) concatenated [real | imag].

    Args:
        band_init:     if True, initialize frequencies in three bands
                       (structural=low, expression=mid, semantic=high).
                       if False, initialize uniformly in [0.02, 0.50] — use this
                       to test whether frequency structure self-organizes.
        acoustic_init: if True, initialize amplitude from character acoustic profiles.
                       if False, use default PyTorch random init.
    """

    STRUCTURAL_FRAC = 0.25
    EXPR_FRAC       = 0.50

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int,
                 vocab: list, band_init: bool = True, acoustic_init: bool = True):
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len

        self.structural_end = max(1, int(embed_dim * self.STRUCTURAL_FRAC))
        self.expr_end       = max(self.structural_end + 1,
                                  int(embed_dim * self.EXPR_FRAC))

        self.amplitude = nn.Embedding(vocab_size, embed_dim)
        if acoustic_init:
            self._init_amplitudes(vocab)

        self.freq = nn.Parameter(
            self._init_frequencies_banded() if band_init
            else torch.rand(embed_dim) * 0.48 + 0.02
        )

    def _init_amplitudes(self, vocab: list):
        with torch.no_grad():
            for idx, token_str in enumerate(vocab):
                profile = char_spectral_profile(
                    token_str, self.embed_dim,
                    self.structural_end, self.expr_end
                )
                self.amplitude.weight[idx] = profile

    def _init_frequencies_banded(self) -> torch.Tensor:
        """Partition frequency range into three bands by dim index."""
        freqs = torch.zeros(self.embed_dim)
        n_s = self.structural_end
        freqs[:n_s] = torch.linspace(0.02, 0.10, n_s)
        n_e = self.expr_end - self.structural_end
        freqs[self.structural_end:self.expr_end] = torch.linspace(0.10, 0.25, n_e)
        n_sem = self.embed_dim - self.expr_end
        freqs[self.expr_end:] = torch.linspace(0.25, 0.50, n_sem)
        return freqs

    def forward(self, tokens: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, T, 2*embed_dim): [amp*cos(phase) | amp*sin(phase)].
        """
        B, T = tokens.shape
        amp   = self.amplitude(tokens)                                  # (B, T, D)
        pos_f = positions.float().unsqueeze(-1) / self.max_seq_len      # (B, T, 1)
        phase = self.freq.unsqueeze(0).unsqueeze(0) * pos_f * 2 * math.pi  # (B, T, D)
        real_part = amp * torch.cos(phase)                              # (B, T, D)
        imag_part = amp * torch.sin(phase)                              # (B, T, D)
        return torch.cat([real_part, imag_part], dim=-1)               # (B, T, 2D)
