"""Tests for quinn.encoding module."""

import math
import pytest
import torch

from quinn.encoding import ComplexTokenEmbedding, MultiScalePositionEncoding


class TestComplexTokenEmbedding:
    def setup_method(self):
        self.vocab_size = 50
        self.embed_dim = 16
        self.model = ComplexTokenEmbedding(self.vocab_size, self.embed_dim, max_seq_len=256)

    def test_output_shape(self):
        B, L = 3, 10
        tokens = torch.randint(0, self.vocab_size, (B, L))
        positions = torch.arange(L).unsqueeze(0).expand(B, -1)
        out = self.model(tokens, positions)
        assert out.shape == (B, L, self.embed_dim)
        assert out.dtype == torch.complex64

    def test_different_tokens_different_encodings(self):
        """Token 0 and token 1 should produce different encodings at same position."""
        pos = torch.zeros(1, 1, dtype=torch.long)
        enc0 = self.model(torch.tensor([[0]]), pos)
        enc1 = self.model(torch.tensor([[1]]), pos)
        assert not torch.allclose(enc0, enc1)

    def test_phase_changes_with_position(self):
        """Same token at different positions should have different phases."""
        t = torch.tensor([[0, 0]])
        pos = torch.tensor([[0, 10]])
        enc = self.model(t, pos)
        # Phase at position 0 vs position 10 should differ
        phase0 = torch.angle(enc[0, 0])
        phase10 = torch.angle(enc[0, 1])
        assert not torch.allclose(phase0, phase10)

    def test_gradients_flow(self):
        B, L = 2, 5
        tokens = torch.randint(0, self.vocab_size, (B, L))
        positions = torch.arange(L).unsqueeze(0).expand(B, -1)
        out = self.model(tokens, positions)
        loss = out.abs().mean()
        loss.backward()
        # All parameters should have gradients
        for name, p in self.model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_batch_consistency(self):
        """Encoding of a single sample should match its row in a batch."""
        tokens = torch.randint(0, self.vocab_size, (4, 8))
        positions = torch.arange(8).unsqueeze(0).expand(4, -1)
        batch_enc = self.model(tokens, positions)

        for i in range(4):
            single_enc = self.model(tokens[i:i+1], positions[i:i+1])
            assert torch.allclose(batch_enc[i:i+1], single_enc, atol=1e-5)


class TestMultiScalePositionEncoding:
    def test_output_shape(self):
        B, L, D = 2, 12, 64
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        enc = MultiScalePositionEncoding.encode(pos, D)
        assert enc.shape == (B, L, D)
        assert enc.dtype == torch.float32

    def test_different_positions_different_encodings(self):
        pos = torch.tensor([[0, 1, 2]])
        enc = MultiScalePositionEncoding.encode(pos, 64)
        assert not torch.allclose(enc[0, 0], enc[0, 1])
        assert not torch.allclose(enc[0, 1], enc[0, 2])

    def test_values_bounded(self):
        """sin/cos encodings must be in [-1, 1]."""
        pos = torch.arange(100).unsqueeze(0)
        enc = MultiScalePositionEncoding.encode(pos, 64)
        assert enc.min() >= -1.0 - 1e-6
        assert enc.max() <=  1.0 + 1e-6
