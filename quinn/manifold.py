"""
Complex-valued interference manifold for QuINN.

All layers operate in complex (C) space. The manifold transforms the
per-position complex waveform representations through phase-preserving
nonlinearities that enforce the interference properties: constructive
where structurally consistent, destructive where accidental.
"""

import math
import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer.

    Uses a single cfloat weight parameter and PyTorch's native complex
    matmul (x @ W.T), which is 2-3x faster on CPU than separating real
    and imaginary parts into 4 individual real matmuls.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = math.sqrt(1.0 / in_features)
        w_real = torch.empty(out_features, in_features).normal_(0, scale)
        w_imag = torch.empty(out_features, in_features).normal_(0, scale)
        self.weight = nn.Parameter(torch.complex(w_real, w_imag))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) complex  →  (..., out_features) complex"""
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    # Expose real/imag parts of weight for compatibility with test_manifold
    @property
    def weight_real(self):
        return self.weight.real

    @property
    def weight_imag(self):
        return self.weight.imag

    @property
    def bias_real(self):
        return self.bias.real if self.bias is not None else None

    @property
    def bias_imag(self):
        return self.bias.imag if self.bias is not None else None


class ModReLU(nn.Module):
    """
    Phase-preserving complex activation (Arjovsky et al., 2016):
        modReLU(z) = ReLU(|z| - b) * z / (|z| + eps)

    Acts only on the modulus. The phase angle is passed through unchanged,
    so structural direction (phase) is preserved while small-amplitude
    (accidental co-occurrence) components are gated away.
    """

    def __init__(self, n_features: int):
        super().__init__()
        # Positive bias threshold; initialised at 0.5 so gate is non-trivially open
        self.bias = nn.Parameter(torch.ones(n_features) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modulus = x.abs().clamp(min=1e-8)        # (... , F)
        threshold = torch.abs(self.bias)          # keep positive
        scale = torch.relu(modulus - threshold) / modulus
        return torch.complex(x.real * scale, x.imag * scale)


class ComplexLayerNorm(nn.Module):
    """
    Layer normalisation for complex tensors.

    Concatenates real and imaginary parts into a 2D real vector and applies
    standard LayerNorm, then splits back. This uses PyTorch's highly optimised
    C++ LayerNorm kernel rather than a manual Python loop.
    Scale and shift are applied jointly (single scale per complex feature).
    """

    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        # LayerNorm over 2*n_features (real concatenated with imag)
        self.norm = nn.LayerNorm(n_features * 2, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_features) complex
        # Stack real/imag along last dim → (..., 2*n_features) float
        stacked = torch.cat([x.real, x.imag], dim=-1)
        normed = self.norm(stacked)
        return torch.complex(
            normed[..., :self.n_features],
            normed[..., self.n_features:],
        )


class InterferenceManifoldLayer(nn.Module):
    """
    One layer of the interference manifold:
        ComplexLinear (expand) → ModReLU → ComplexLinear (contract) → ComplexLayerNorm
    with a complex residual connection.

    The expand-then-contract pattern (like a real-valued FFN) gives the manifold
    capacity to rotate phase relationships before gating small components.
    """

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        hidden = dim * expansion
        self.linear1 = ComplexLinear(dim, hidden)
        self.act = ModReLU(hidden)
        self.linear2 = ComplexLinear(hidden, dim)
        self.norm = ComplexLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return self.norm(x + residual)


class InterferenceManifold(nn.Module):
    """
    Stack of interference manifold layers.
    Each layer refines phase relationships: after N layers the representation
    should exhibit constructive interference for structurally coherent patterns
    and near-zero amplitude for coincidental co-occurrences.
    """

    def __init__(self, dim: int, n_layers: int, expansion: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [InterferenceManifoldLayer(dim, expansion) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
