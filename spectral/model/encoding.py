"""
SpectralEmbedding: character-composed token initialization + MoPE positional encoding.

Design principles:
  1. Character-level composition:
       Each token's amplitude profile is derived from the characters that spell it.
       Characters map to frequency bands by acoustic class (vowel/stop/fricative/etc).
       This gives every token a unique spectral fingerprint before any training.

  2. MoPE (Morlet Positional Encoding):
       Each embedding dimension d has TWO learned parameters:
         f[d]:  frequency   — how fast phase advances with position
         s[d]:  log-sigma   — locality bandwidth (large = global, small = localized)
       basis(pos, d) = exp(i * f[d] * pos / max_len * 2π)
                     * exp(-pos² / (2 * sigma[d]²))   [Gaussian locality]

       Low-freq dims (0..L):   large sigma → global structural waves
       High-freq dims (H..D):  small sigma → localized semantic perturbations

  3. Frequency band partitioning:
       dims  0 .. STRUCTURAL_END:   structural band  (nesting, control flow)
       dims  STRUCTURAL_END .. EXPR_END:  expression band  (statements, clauses)
       dims  EXPR_END .. embed_dim:  semantic band  (individual tokens)
"""

import math
import torch
import torch.nn as nn


# ── Character acoustic classes ─────────────────────────────────────────────────
# Maps characters to their dominant frequency class.
# Based on approximate acoustic phonetic properties.

_VOWELS      = set("aeiouAEIOU")
_STOPS       = set("bdgkptBDGKPT")
_FRICATIVES  = set("fsvzFSVZ")
_SONORANTS   = set("lmnrLMNR")
_APPROXIMANT = set("wyhWYH")
_OTHER       = set("_-0123456789")   # underscores, digits, etc.

# Relative amplitude weight per frequency band for each acoustic class
#   (structural_weight, expression_weight, semantic_weight)
_CHAR_BAND_WEIGHTS = {
    "vowel":       (0.6, 0.3, 0.1),   # vowels: low-freq resonance dominant
    "sonorant":    (0.3, 0.5, 0.2),   # l/m/n/r: mid-freq
    "approximant": (0.5, 0.3, 0.2),   # w/y/h: low-mid
    "stop":        (0.1, 0.2, 0.7),   # d/t/p/k: high-freq burst
    "fricative":   (0.1, 0.2, 0.7),   # f/s/v/z: high-freq noise
    "digit":       (0.2, 0.4, 0.4),   # digits: mid-distributed
    "other":       (0.3, 0.4, 0.3),   # default: flat-ish
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

    Returns a real-valued tensor of shape (embed_dim,) representing the
    initial amplitude distribution across frequency bands.
    """
    if not token_str or token_str.startswith("<"):
        # Special/abstract tokens: flat initialization
        return torch.ones(embed_dim) * 0.02

    # Accumulate band weights across characters
    structural_w = 0.0
    expr_w       = 0.0
    semantic_w   = 0.0

    for ch in token_str:
        cls = _char_class(ch)
        sw, ew, semw = _CHAR_BAND_WEIGHTS[cls]
        structural_w += sw
        expr_w       += ew
        semantic_w   += semw

    n = len(token_str)
    structural_w /= n
    expr_w       /= n
    semantic_w   /= n

    # Build amplitude profile: each band gets its weight distributed
    # uniformly across its dimensions, with small Gaussian noise for diversity
    profile = torch.zeros(embed_dim)

    n_structural = structural_end
    n_expr       = expr_end - structural_end
    n_semantic   = embed_dim - expr_end

    if n_structural > 0:
        profile[:structural_end]      = structural_w / math.sqrt(n_structural)
    if n_expr > 0:
        profile[structural_end:expr_end] = expr_w    / math.sqrt(n_expr)
    if n_semantic > 0:
        profile[expr_end:]             = semantic_w  / math.sqrt(n_semantic)

    # Scale to small magnitude (training will adjust)
    profile = profile * 0.02

    return profile


class SpectralEmbedding(nn.Module):
    """
    Token embedding with character-composed initialization and MoPE positional encoding.

    Forward input:
        tokens:    (B, T) long  — token indices
        positions: (B, T) long  — position indices

    Forward output:
        (B, T, embed_dim) real  — concatenated [real; imag] of complex waveform,
                                  suitable as input to standard attention layers
    """

    STRUCTURAL_FRAC = 0.25   # fraction of dims in structural band
    EXPR_FRAC       = 0.50   # fraction of dims in structural + expression

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int,
                 vocab: list):
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len

        # Band boundaries
        self.structural_end = max(1, int(embed_dim * self.STRUCTURAL_FRAC))
        self.expr_end       = max(self.structural_end + 1,
                                  int(embed_dim * self.EXPR_FRAC))

        # ── Amplitude (token identity) ──────────────────────────────────────
        # Initialized from character spectral profiles, then learned
        self.amplitude = nn.Embedding(vocab_size, embed_dim)
        self._init_amplitudes(vocab)

        # ── MoPE parameters (positional encoding) ──────────────────────────
        # f[d]: frequency per dimension — initialized to cover full range
        # log_sigma[d]: log-bandwidth — initialized by band
        self.freq      = nn.Parameter(self._init_frequencies())
        self.log_sigma = nn.Parameter(self._init_log_sigmas())

    def _init_amplitudes(self, vocab: list):
        """Initialize amplitude embeddings from character spectral profiles."""
        with torch.no_grad():
            for idx, token_str in enumerate(vocab):
                profile = char_spectral_profile(
                    token_str, self.embed_dim,
                    self.structural_end, self.expr_end
                )
                self.amplitude.weight[idx] = profile

    def _init_frequencies(self) -> torch.Tensor:
        """
        Initialize frequencies hierarchically by band.

        Structural band:  low frequencies  (slow global oscillation)
        Expression band:  mid frequencies
        Semantic band:    high frequencies (fast local oscillation)
        """
        freqs = torch.zeros(self.embed_dim)

        # Structural: f in [0.02, 0.10]
        n_s = self.structural_end
        freqs[:n_s] = torch.linspace(0.02, 0.10, n_s)

        # Expression: f in [0.10, 0.25]
        n_e = self.expr_end - self.structural_end
        freqs[self.structural_end:self.expr_end] = torch.linspace(0.10, 0.25, n_e)

        # Semantic: f in [0.25, 0.50]
        n_sem = self.embed_dim - self.expr_end
        freqs[self.expr_end:] = torch.linspace(0.25, 0.50, n_sem)

        return freqs

    def _init_log_sigmas(self) -> torch.Tensor:
        """
        Initialize locality bandwidths by band.

        Structural band:  large sigma → global (active everywhere)
        Expression band:  medium sigma
        Semantic band:    small sigma → localized perturbations
        """
        log_sigmas = torch.zeros(self.embed_dim)

        # Structural: sigma ≈ max_seq_len (global)
        log_sigmas[:self.structural_end] = math.log(self.max_seq_len)

        # Expression: sigma ≈ max_seq_len / 4
        log_sigmas[self.structural_end:self.expr_end] = math.log(
            self.max_seq_len / 4
        )

        # Semantic: sigma ≈ max_seq_len / 16 (localized)
        log_sigmas[self.expr_end:] = math.log(
            max(1, self.max_seq_len / 16)
        )

        return log_sigmas

    def forward(self, tokens: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral embeddings.

        Returns (B, T, 2*embed_dim) real tensor — [real_part | imag_part]
        concatenated along last dim so standard linear layers can process it.
        """
        B, T = tokens.shape

        # Token amplitude: (B, T, D)
        amp = self.amplitude(tokens)  # real, (B, T, D)

        # MoPE positional basis
        # pos_f: (B, T, 1) normalized positions
        pos_f = positions.float().unsqueeze(-1) / self.max_seq_len  # (B, T, 1)

        # Phase: f[d] * pos * 2π  →  (B, T, D)
        phase = self.freq.unsqueeze(0).unsqueeze(0) * pos_f * 2 * math.pi

        # Gaussian locality envelope: exp(-pos² / (2σ²))
        # Center Gaussian at pos=0 (start of sequence) for global dims,
        # and let sigma determine reach. For large sigma (structural band),
        # envelope ≈ 1.0 everywhere. For small sigma (semantic), falls off fast.
        sigma  = torch.exp(self.log_sigma)                    # (D,)
        pos_sq = (positions.float() ** 2).unsqueeze(-1)       # (B, T, 1)
        envelope = torch.exp(
            -pos_sq / (2 * sigma.unsqueeze(0).unsqueeze(0) ** 2 + 1e-6)
        )                                                      # (B, T, D)

        # Complex waveform: amp * envelope * exp(i*phase)
        # Real part: amp * envelope * cos(phase)
        # Imag part: amp * envelope * sin(phase)
        modulated = amp * envelope                            # (B, T, D)
        real_part = modulated * torch.cos(phase)             # (B, T, D)
        imag_part = modulated * torch.sin(phase)             # (B, T, D)

        # Concatenate to (B, T, 2D) for use with real-valued attention
        return torch.cat([real_part, imag_part], dim=-1)     # (B, T, 2D)
