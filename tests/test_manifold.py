"""Tests for quinn.manifold module."""

import pytest
import torch
import torch.nn as nn

from quinn.manifold import (
    ComplexLinear,
    ModReLU,
    ComplexLayerNorm,
    InterferenceManifoldLayer,
    InterferenceManifold,
)


class TestComplexLinear:
    def test_output_shape(self):
        layer = ComplexLinear(32, 64)
        x = torch.randn(4, 10, 32, dtype=torch.cfloat)
        out = layer(x)
        assert out.shape == (4, 10, 64)
        assert out.dtype == torch.complex64

    def test_linearity(self):
        """f(a*x + b*y) ≈ a*f(x) + b*f(y) for complex linear layer."""
        layer = ComplexLinear(16, 16, bias=False)
        x = torch.randn(2, 5, 16, dtype=torch.cfloat)
        y = torch.randn(2, 5, 16, dtype=torch.cfloat)
        a, b = 1.5 + 0.5j, -0.3 + 1.2j

        lhs = layer(a * x + b * y)
        rhs = a * layer(x) + b * layer(y)
        assert torch.allclose(lhs, rhs, atol=1e-5)

    def test_no_bias_option(self):
        layer = ComplexLinear(8, 8, bias=False)
        assert layer.bias_real is None
        assert layer.bias_imag is None

    def test_gradients_flow(self):
        layer = ComplexLinear(16, 16)
        x = torch.randn(2, 5, 16, dtype=torch.cfloat, requires_grad=True)
        out = layer(x)
        loss = out.abs().sum()
        loss.backward()
        assert x.grad is not None
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_complex_output_from_real_input(self):
        """Layer with nonzero imaginary weights should produce complex output from real input."""
        layer = ComplexLinear(4, 4)
        with torch.no_grad():
            layer.weight_imag.fill_(1.0)
        x = torch.randn(1, 3, 4, dtype=torch.cfloat)
        x.imag.zero_()
        out = layer(x)
        assert out.dtype == torch.complex64


class TestModReLU:
    def test_phase_preservation(self):
        """ModReLU should not change the phase of passed-through components."""
        act = ModReLU(8)
        with torch.no_grad():
            act.bias.fill_(0.0)  # no gating → all components pass through
        x = torch.randn(2, 5, 8, dtype=torch.cfloat) * 2.0  # large modulus
        out = act(x)
        phase_in  = torch.angle(x)
        phase_out = torch.angle(out)
        assert torch.allclose(phase_in, phase_out, atol=1e-4)

    def test_gating_small_amplitudes(self):
        """Components with modulus < bias should be zeroed out."""
        act = ModReLU(4)
        with torch.no_grad():
            act.bias.fill_(10.0)  # very high threshold
        x = torch.randn(2, 3, 4, dtype=torch.cfloat) * 0.1  # tiny modulus
        out = act(x)
        assert out.abs().max() < 1e-3

    def test_output_dtype(self):
        act = ModReLU(16)
        x = torch.randn(2, 5, 16, dtype=torch.cfloat)
        out = act(x)
        assert out.dtype == torch.complex64

    def test_gradients(self):
        act = ModReLU(8)
        x = torch.randn(2, 5, 8, dtype=torch.cfloat, requires_grad=True)
        out = act(x)
        out.abs().sum().backward()
        assert x.grad is not None


class TestComplexLayerNorm:
    def test_output_shape(self):
        norm = ComplexLayerNorm(32)
        x = torch.randn(3, 10, 32, dtype=torch.cfloat)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == torch.complex64

    def test_approximately_unit_variance(self):
        """After normalisation, variance should be close to 1."""
        norm = ComplexLayerNorm(64)
        # Reset LayerNorm weights to identity so we isolate normalisation
        with torch.no_grad():
            norm.norm.weight.fill_(1.0)
            norm.norm.bias.fill_(0.0)
        x = torch.randn(8, 20, 64, dtype=torch.cfloat) * 5.0  # large spread
        out = norm(x)
        # Each sample's feature-wise variance should be near 1
        var_r = out.real.var(dim=-1)
        var_i = out.imag.var(dim=-1)
        assert var_r.mean().item() < 2.0
        assert var_i.mean().item() < 2.0


class TestInterferenceManifoldLayer:
    def test_residual_preserves_shape(self):
        layer = InterferenceManifoldLayer(dim=32)
        x = torch.randn(2, 8, 32, dtype=torch.cfloat)
        out = layer(x)
        assert out.shape == x.shape

    def test_gradients(self):
        layer = InterferenceManifoldLayer(dim=16)
        x = torch.randn(2, 4, 16, dtype=torch.cfloat, requires_grad=True)
        out = layer(x)
        out.abs().mean().backward()
        assert x.grad is not None


class TestInterferenceManifold:
    def test_stacked_layers(self):
        manifold = InterferenceManifold(dim=32, n_layers=3)
        x = torch.randn(2, 10, 32, dtype=torch.cfloat)
        out = manifold(x)
        assert out.shape == x.shape

    def test_transforms_input(self):
        """Manifold output should differ from input (it actually transforms)."""
        manifold = InterferenceManifold(dim=16, n_layers=2)
        x = torch.randn(2, 5, 16, dtype=torch.cfloat)
        out = manifold(x)
        assert not torch.allclose(out.abs(), x.abs(), atol=0.01)
