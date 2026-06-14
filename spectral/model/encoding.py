"""
SpectralEmbedding: graph-Laplacian amplitude initialization + learned sinusoidal PE.

Design principles:
  1. Laplacian eigenvector amplitude initialization (default):
       The normalized graph Laplacian of the symmetrized bigram co-occurrence
       matrix has eigenvectors naturally ordered from smooth (global) to
       oscillatory (local). Eigenvector 0 encodes token frequency; eigenvectors
       1..D capture progressively finer co-occurrence structure. This seeds every
       token with a spectral fingerprint derived from the actual corpus structure
       rather than character-level acoustic heuristics.

  2. Learned per-dimension frequency (MoPE-inspired):
       Each embedding dimension d has ONE learned parameter:
         f[d]: frequency — how fast phase advances with position
       basis(pos, d) = amp[d] * exp(i * f[d] * pos / max_len * 2π)

       Output is the [real | imag] concatenation: (B, T, 2*embed_dim).

  3. Frequency band partitioning (optional, --no-band-init to disable):
       dims  0 .. STRUCTURAL_END:       structural (low freq, global patterns)
       dims  STRUCTURAL_END .. EXPR_END: expression (mid freq)
       dims  EXPR_END .. embed_dim:     semantic (high freq, local patterns)

  4. log_sigma: learned locality scale per dimension (NOT used in SpectralEmbedding
       forward). It initializes the per-head sigma in SpectralTransformerLayer, where
       it controls the distance penalty in attention. Kept here for inspectability.

Note: the Gaussian envelope (Morlet) has been permanently removed. It was centered
at position 0, making the semantic band dead past ~50 tokens. Multi-scale locality
now lives in the attention mechanism via the sigma-based distance bias.
"""

import math
import torch
import torch.nn as nn


# ── Laplacian amplitude initialization ────────────────────────────────────────

def laplacian_amplitude_init(
    eigvecs: torch.Tensor,
    embed_dim: int,
) -> torch.Tensor:
    """
    Take the first embed_dim columns of the Laplacian eigenvector matrix and
    scale to 0.02 magnitude.

    Eigenvectors are ordered ascending by eigenvalue: column 0 is the smoothest
    graph signal (proportional to sqrt(degree) — encodes token frequency),
    column 1 is the Fiedler vector (principal cluster boundary), etc.

    Returns (vocab_size, embed_dim) float tensor.
    """
    V = eigvecs.shape[0]
    k = min(embed_dim, eigvecs.shape[1])
    init = eigvecs[:, :k].float()
    if k < embed_dim:
        init = torch.cat([init, torch.zeros(V, embed_dim - k)], dim=1)
    return init * 0.02


class SpectralEmbedding(nn.Module):
    """
    Token embedding with learned amplitude per token and learned sinusoidal PE.

    For each token t at position p and embedding dimension d:
        real[d] = amp[t,d] * cos(f[d] * p / max_len * 2π)
        imag[d] = amp[t,d] * sin(f[d] * p / max_len * 2π)

    Output: (B, T, 2*embed_dim) concatenated [real | imag].

    Args:
        laplacian_eigvecs: (vocab_size, vocab_size) tensor from compute_spectral_stats.
                           If None, falls back to acoustic_init (or random if both False).
        band_init:     if True, initialize frequencies in three bands.
        acoustic_init: fallback when laplacian_eigvecs is None.
    """

    STRUCTURAL_FRAC = 0.25
    EXPR_FRAC       = 0.50

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        vocab: list,
        band_init: bool = True,
        acoustic_init: bool = True,
        laplacian_eigvecs: 'torch.Tensor | None' = None,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len

        self.structural_end = max(1, int(embed_dim * self.STRUCTURAL_FRAC))
        self.expr_end       = max(self.structural_end + 1,
                                  int(embed_dim * self.EXPR_FRAC))

        # ── Amplitude initialization ──────────────────────────────────────────
        self.amplitude = nn.Embedding(vocab_size, embed_dim)
        if laplacian_eigvecs is not None:
            with torch.no_grad():
                self.amplitude.weight.copy_(
                    laplacian_amplitude_init(laplacian_eigvecs, embed_dim)
                )
        elif acoustic_init:
            self._init_amplitudes_acoustic(vocab)

        # ── Frequency initialization ──────────────────────────────────────────
        self.freq = nn.Parameter(
            self._init_frequencies_banded() if band_init
            else torch.rand(embed_dim) * 0.48 + 0.02
        )

        # ── Locality scale (inspectable; used to initialize attention layers) ─
        # NOT used in this module's forward — it initializes SpectralTransformerLayer's
        # log_sigma_head, then both are trained independently.
        # Large sigma = global reach (structural), small sigma = local (semantic).
        self.log_sigma = nn.Parameter(
            self._init_log_sigma_banded() if band_init
            else torch.zeros(embed_dim)
        )

    # ── Acoustic fallback (character-based, used when no Laplacian data) ──────

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

    def _char_class(self, ch: str) -> str:
        if ch in self._VOWELS:      return "vowel"
        if ch in self._SONORANTS:   return "sonorant"
        if ch in self._APPROXIMANT: return "approximant"
        if ch in self._STOPS:       return "stop"
        if ch in self._FRICATIVES:  return "fricative"
        if ch.isdigit():            return "digit"
        return "other"

    def _char_spectral_profile(self, token_str: str) -> torch.Tensor:
        if not token_str or token_str.startswith("<"):
            return torch.ones(self.embed_dim) * 0.02
        sw = ew = mw = 0.0
        for ch in token_str:
            s, e, m = self._CHAR_BAND_WEIGHTS[self._char_class(ch)]
            sw += s;  ew += e;  mw += m
        n = len(token_str)
        sw /= n;  ew /= n;  mw /= n
        profile = torch.zeros(self.embed_dim)
        n_s = self.structural_end
        n_e = self.expr_end - self.structural_end
        n_m = self.embed_dim - self.expr_end
        if n_s > 0: profile[:self.structural_end]         = sw / math.sqrt(n_s)
        if n_e > 0: profile[self.structural_end:self.expr_end] = ew / math.sqrt(n_e)
        if n_m > 0: profile[self.expr_end:]               = mw / math.sqrt(n_m)
        return profile * 0.02

    def _init_amplitudes_acoustic(self, vocab: list):
        with torch.no_grad():
            for idx, token_str in enumerate(vocab):
                self.amplitude.weight[idx] = self._char_spectral_profile(token_str)

    # ── Frequency and sigma initialization ───────────────────────────────────

    def _init_frequencies_banded(self) -> torch.Tensor:
        """Partition frequency range [0.02, 0.50] into three bands by dim index."""
        freqs = torch.zeros(self.embed_dim)
        freqs[:self.structural_end] = torch.linspace(
            0.02, 0.10, self.structural_end)
        freqs[self.structural_end:self.expr_end] = torch.linspace(
            0.10, 0.25, self.expr_end - self.structural_end)
        freqs[self.expr_end:] = torch.linspace(
            0.25, 0.50, self.embed_dim - self.expr_end)
        return freqs

    def _init_log_sigma_banded(self) -> torch.Tensor:
        """Locality scale init: structural=global (large σ), semantic=local (small σ)."""
        log_sigma = torch.zeros(self.embed_dim)
        # structural band: σ ≈ max_seq_len  (near-global attention reach)
        log_sigma[:self.structural_end] = math.log(max(self.max_seq_len, 1.0))
        # expression band: σ ≈ max_seq_len/4
        log_sigma[self.structural_end:self.expr_end] = math.log(
            max(self.max_seq_len / 4, 1.0))
        # semantic band:   σ ≈ max_seq_len/16  (highly local attention)
        log_sigma[self.expr_end:] = math.log(max(self.max_seq_len / 16, 1.0))
        return log_sigma

    def forward(self, tokens: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, T, 2*embed_dim): [amp*cos(phase) | amp*sin(phase)].

        During training, each token's amplitude vector is randomly sign-flipped
        (per-token scalar ±1, broadcast across all D dims). Laplacian eigenvectors
        are only defined up to sign, so this augmentation prevents the model from
        relying on absolute sign and regularizes the amplitude basis.
        """
        amp = self.amplitude(tokens)                                    # (B, T, D)
        if self.training:
            signs = (torch.randint(0, 2, (amp.size(0), amp.size(1), 1),
                                   device=amp.device) * 2 - 1).float()
            amp = amp * signs
        pos_f = positions.float().unsqueeze(-1) / self.max_seq_len      # (B, T, 1)
        phase = self.freq.unsqueeze(0).unsqueeze(0) * pos_f * 2 * math.pi  # (B, T, D)
        return torch.cat([amp * torch.cos(phase),
                          amp * torch.sin(phase)], dim=-1)              # (B, T, 2D)
