"""
Contrastive loss for Phase 2: distinguish correct waveforms from wrong ones.

NT-Xent (Normalized Temperature-scaled Cross Entropy) loss compares
the correct waveform summary against an incorrect one, pulling them
apart in embedding space.

In the context of QuINN:
  - Anchor: QuINN summary from prefix (same for positive and negative)
  - Positive: summary encoded from correct target sequence
  - Negative: summary encoded from wrong target sequence

The loss encourages the embedding space to respect structural correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveformContrastiveLoss(nn.Module):
    """
    NT-Xent (InfoNCE) loss on waveform summaries.

    For a batch of prefixes:
      - All samples share the same QuINN summary (from prefix)
      - Positive: summary of correct full sequence
      - Negative: summary of wrong full sequence

    Loss pulls the positive summary close to the prefix summary,
    pushes the negative far away.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        """
        Args:
            temperature: scaling factor (lower = sharper contrast)
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        prefix_summary: torch.Tensor,  # (B, D) complex
        positive_summary: torch.Tensor,  # (B, D) complex
        negative_summary: torch.Tensor,  # (B, D) complex
    ) -> torch.Tensor:
        """
        Args:
            prefix_summary: (B, D) complex summary from prefix
            positive_summary: (B, D) complex summary from correct target
            negative_summary: (B, D) complex summary from wrong target

        Returns:
            scalar loss
        """
        # Convert complex to real: take magnitude (discard phase for similarity metric)
        # Alternatively: use real and imaginary as separate dimensions
        prefix_real = torch.cat([prefix_summary.real, prefix_summary.imag], dim=-1)
        positive_real = torch.cat([positive_summary.real, positive_summary.imag], dim=-1)
        negative_real = torch.cat([negative_summary.real, negative_summary.imag], dim=-1)

        # Normalize
        prefix_real = F.normalize(prefix_real, dim=-1)
        positive_real = F.normalize(positive_real, dim=-1)
        negative_real = F.normalize(negative_real, dim=-1)

        # Similarity: cosine in normalized space
        B = prefix_real.shape[0]

        # pos_sim: (B,) — similarity between prefix and positive
        pos_sim = (prefix_real * positive_real).sum(dim=-1) / self.temperature

        # neg_sim: (B,) — similarity between prefix and negative
        neg_sim = (prefix_real * negative_real).sum(dim=-1) / self.temperature

        # NT-Xent loss: log-softmax over positive and negative
        # Loss = -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
        logits = torch.stack([pos_sim, neg_sim], dim=-1)  # (B, 2)
        labels = torch.zeros(B, dtype=torch.long, device=prefix_real.device)  # positive is class 0

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        return loss


class TripletWaveformLoss(nn.Module):
    """
    Triplet loss alternative: margin-based contrastive loss.

    L = max(0, neg_dist - pos_dist + margin)

    This directly optimizes: negative should be farther than positive + margin.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        """
        Args:
            margin: minimum separation
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        prefix_summary: torch.Tensor,  # (B, D) complex
        positive_summary: torch.Tensor,  # (B, D) complex
        negative_summary: torch.Tensor,  # (B, D) complex
    ) -> torch.Tensor:
        """
        Args:
            prefix_summary: anchor from prefix
            positive_summary: correct target
            negative_summary: wrong target

        Returns:
            scalar loss
        """
        # Convert complex to real
        prefix_real = torch.cat([prefix_summary.real, prefix_summary.imag], dim=-1)
        positive_real = torch.cat([positive_summary.real, positive_summary.imag], dim=-1)
        negative_real = torch.cat([negative_summary.real, negative_summary.imag], dim=-1)

        # Euclidean distance
        pos_dist = torch.norm(prefix_real - positive_real, dim=-1)
        neg_dist = torch.norm(prefix_real - negative_real, dim=-1)

        # Triplet loss
        loss = torch.relu(self.margin + pos_dist - neg_dist)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
