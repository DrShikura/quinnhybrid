"""
Loss functions for SpectralLM.

Waveform completion loss:
    Given prefix waveform → predict full waveform.
    Loss operates separately on structural / expression / semantic bands
    so the model can't just match one band and ignore the others.

Language model loss:
    Standard cross-entropy on next-token prediction.

Joint loss:
    Weighted combination. Waveform loss acts as regularizer/pre-trainer
    that keeps the model's frequency structure meaningful.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BandedWaveformLoss(nn.Module):
    """
    Waveform completion loss computed per frequency band.

    The [real|imag] waveform has shape (B, T, 2*embed_dim).
    We split the embed_dim dimensions into bands and compute
    cosine similarity loss within each band independently.

    This forces the model to maintain meaningful signal in ALL bands,
    not just the easiest one to learn.

    Args:
        embed_dim:      per-modality embedding dim
        structural_end: last structural-band dimension index
        expr_end:       last expression-band dimension index
        band_weights:   (structural_w, expr_w, semantic_w) — loss weights per band
    """

    def __init__(
        self,
        embed_dim:      int,
        structural_end: int,
        expr_end:       int,
        band_weights:   tuple = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.embed_dim      = embed_dim
        self.dim            = 2 * embed_dim
        self.structural_end = structural_end
        self.expr_end       = expr_end
        self.band_weights   = band_weights

        # Band slices in [real|imag] concatenated space
        # real part occupies dims [0:embed_dim], imag [embed_dim:2*embed_dim]
        # structural band: real[0:structural_end] + imag[0:structural_end]
        self.band_slices = {
            'structural': (
                list(range(0, structural_end)) +
                list(range(embed_dim, embed_dim + structural_end))
            ),
            'expression': (
                list(range(structural_end, expr_end)) +
                list(range(embed_dim + structural_end, embed_dim + expr_end))
            ),
            'semantic': (
                list(range(expr_end, embed_dim)) +
                list(range(embed_dim + expr_end, 2 * embed_dim))
            ),
        }

    def _cosine_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Scale-invariant waveform loss: 1 - cosine_similarity.
        Range [0, 2]. 0 = perfect match.
        """
        pred_norm   = F.normalize(pred,   dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cos_sim     = (pred_norm * target_norm).sum(dim=-1)  # (B, T) or (B,)

        loss = 1.0 - cos_sim   # (B, T) or (B,)

        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def forward(
        self,
        pred:   torch.Tensor,   # (B, T, 2*embed_dim) predicted waveform
        target: torch.Tensor,   # (B, T, 2*embed_dim) target waveform
        mask:   torch.Tensor | None = None,  # (B, T) float, 1 = included
    ) -> dict:
        """
        Compute banded waveform loss.

        Returns dict:
            'total':      weighted sum of band losses
            'structural': loss for structural band
            'expression': loss for expression band
            'semantic':   loss for semantic band
        """
        losses = {}
        total  = 0.0

        for band_name, (weight, slices) in zip(
            ['structural', 'expression', 'semantic'],
            zip(self.band_weights, [
                self.band_slices['structural'],
                self.band_slices['expression'],
                self.band_slices['semantic'],
            ])
        ):
            pred_band   = pred[..., slices]     # (B, T, band_size)
            target_band = target[..., slices]

            band_loss = self._cosine_loss(pred_band, target_band, mask)
            losses[band_name] = band_loss
            total = total + weight * band_loss

        losses['total'] = total
        return losses


class SpectralLoss(nn.Module):
    """
    Combined loss for SpectralLM.

    Args:
        embed_dim:        per-modality embedding dim
        structural_end:   structural band end index
        expr_end:         expression band end index
        lm_weight:        weight on language model loss
        waveform_weight:  weight on waveform completion loss
        band_weights:     per-band waveform loss weights
    """

    def __init__(
        self,
        embed_dim:        int,
        structural_end:   int,
        expr_end:         int,
        lm_weight:        float = 1.0,
        waveform_weight:  float = 0.5,
        band_weights:     tuple = (1.5, 1.0, 0.5),
    ):
        super().__init__()
        self.lm_weight       = lm_weight
        self.waveform_weight = waveform_weight

        self.waveform_loss = BandedWaveformLoss(
            embed_dim, structural_end, expr_end, band_weights
        )

    def forward(
        self,
        model_out:    dict,
        target_ids:   torch.Tensor,          # (B, T) long  — next-token ids
        target_wave:  torch.Tensor | None,   # (B, T, 2D)   — next-token waveforms
        pad_id:       int = 0,
        mode:         str = 'joint',
    ) -> dict:
        """
        Compute total loss.

        Returns dict with 'total' and per-component losses.
        """
        losses = {}
        total  = torch.tensor(0.0, device=target_ids.device)

        # Language model loss (next-token cross-entropy)
        if mode in ('lm', 'joint') and 'lm_logits' in model_out:
            logits = model_out['lm_logits']   # (B, T, vocab)
            # Shift: predict token t+1 from state t
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                target_ids[:, 1:].reshape(-1),
                ignore_index=pad_id,
            )
            losses['lm'] = lm_loss
            total = total + self.lm_weight * lm_loss

        # Waveform completion loss
        if mode in ('waveform', 'joint') and 'waveform' in model_out and target_wave is not None:
            pred_wave = model_out['waveform']   # (B, T, 2D)

            # Mask out padding
            pad_mask = (target_ids != pad_id).float()   # (B, T)

            wave_losses = self.waveform_loss(pred_wave[:, :-1], target_wave[:, 1:],
                                              mask=pad_mask[:, 1:])
            losses.update({f'wave_{k}': v for k, v in wave_losses.items()})
            total = total + self.waveform_weight * wave_losses['total']

        losses['total'] = total
        return losses
