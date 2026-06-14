"""
Loss functions for SpectralLM.

BandedWaveformLoss:
    Amplitude-band cosine loss for waveform completion. Predicts amp[t+1]
    from hidden[t] — operates on D-dim amplitude vectors, not the 2D waveform
    (F.normalize on 2D reduces to trivial phase prediction).

WaveformGradientConsistencyLoss:
    Penalizes large waveform jumps at syntactically predictable positions.
    Target is frozen per-token bigram entropy from the corpus — not model
    logits, so no second-order gradients needed. Weight is kept constant
    (not annealed) because the target never changes.

SpectralLoss:
    Combines all three losses. Waveform loss is cosine-annealed to zero;
    gradient consistency loss runs at constant weight throughout training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandedWaveformLoss(nn.Module):
    """
    Amplitude-band cosine loss for the waveform completion task.

    Operates on amplitude vectors (B, T, embed_dim) — NOT the full [real|imag]
    waveform. The full waveform was abandoned because F.normalize removes amplitude
    information, reducing the loss to a trivial phase-prediction task.

    Args:
        embed_dim:      per-modality embedding dim (target shape: B, T, embed_dim)
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
        self.structural_end = structural_end
        self.expr_end       = expr_end
        self.band_weights   = band_weights

        self.band_slices = {
            'structural': list(range(0, structural_end)),
            'expression': list(range(structural_end, expr_end)),
            'semantic':   list(range(expr_end, embed_dim)),
        }

    def _cosine_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
        pred_n  = F.normalize(pred,   dim=-1)
        target_n = F.normalize(target, dim=-1)
        loss = 1.0 - (pred_n * target_n).sum(dim=-1)  # (B, T)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss

    def forward(
        self,
        pred:   torch.Tensor,              # (B, T, embed_dim)
        target: torch.Tensor,              # (B, T, embed_dim)
        mask:   torch.Tensor | None = None,
    ) -> dict:
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
            band_loss      = self._cosine_loss(pred[..., slices], target[..., slices], mask)
            losses[band_name] = band_loss
            total = total + weight * band_loss
        losses['total'] = total
        return losses


class WaveformGradientConsistencyLoss(nn.Module):
    """
    Penalizes large waveform jumps at syntactically predictable positions.

    Intuition: tokens with low bigram entropy (e.g. DOT → always NAME, as → always
    NAME, import → NAME) are syntactically binding. At these positions the waveform
    should change smoothly; large spectral discontinuities should occur only at
    high-entropy positions (syntactically open, many valid successors).

    This teaches the model to express syntactic surprisal as waveform discontinuity.

    Args:
        entropy_threshold: H_norm value below which a position is "predictable".
                           With our corpus, ~57% of tokens fall below 0.4.
    """

    def __init__(self, entropy_threshold: float = 0.4):
        super().__init__()
        self.threshold = entropy_threshold

    def forward(
        self,
        waveform:       torch.Tensor,   # (B, T, 2D) input waveform
        token_ids:      torch.Tensor,   # (B, T) long
        frozen_entropy: torch.Tensor,   # (vocab_size,) H_norm — never backpropped
    ) -> torch.Tensor:
        # Waveform difference between adjacent positions
        delta    = waveform[:, 1:] - waveform[:, :-1]   # (B, T-1, 2D)
        grad_mag = delta.norm(dim=-1)                     # (B, T-1)

        # Per-position entropy from the frozen corpus statistics (no gradient)
        with torch.no_grad():
            ent = frozen_entropy[token_ids[:, :-1]]       # (B, T-1)

        # Penalize large jumps at predictable (low-entropy) positions
        low_ent_mask = (ent < self.threshold).float()
        return (low_ent_mask * grad_mag).mean()


class SpectralLoss(nn.Module):
    """
    Combined loss for SpectralLM.

    Args:
        embed_dim:          per-modality embedding dim
        structural_end:     structural band end index
        expr_end:           expression band end index
        lm_weight:          weight on language model loss
        waveform_weight:    weight on waveform completion loss (cosine-annealed)
        band_weights:       per-band waveform loss weights
        gradient_weight:    weight on gradient consistency loss (CONSTANT — not annealed)
        entropy_threshold:  H_norm threshold for "predictable" positions
    """

    def __init__(
        self,
        embed_dim:         int,
        structural_end:    int,
        expr_end:          int,
        lm_weight:         float = 1.0,
        waveform_weight:   float = 0.5,
        band_weights:      tuple = (1.5, 1.0, 0.5),
        gradient_weight:   float = 0.1,
        entropy_threshold: float = 0.4,
    ):
        super().__init__()
        self.lm_weight       = lm_weight
        self.waveform_weight = waveform_weight
        self.gradient_weight = gradient_weight   # never modified during training

        self.waveform_loss  = BandedWaveformLoss(
            embed_dim, structural_end, expr_end, band_weights
        )
        self.gradient_loss  = WaveformGradientConsistencyLoss(entropy_threshold)

    def forward(
        self,
        model_out:      dict,
        target_ids:     torch.Tensor,           # (B, T) long
        target_wave:    torch.Tensor | None,    # (B, T, D) amplitude targets
        frozen_entropy: torch.Tensor | None,    # (vocab_size,) H_norm
        pad_id:         int = 0,
        mode:           str = 'joint',
    ) -> dict:
        losses = {}
        total  = torch.tensor(0.0, device=target_ids.device)

        # Language model loss (next-token cross-entropy)
        if mode in ('lm', 'joint') and 'lm_logits' in model_out:
            logits = model_out['lm_logits']
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                target_ids[:, 1:].reshape(-1),
                ignore_index=pad_id,
            )
            losses['lm'] = lm_loss
            total = total + self.lm_weight * lm_loss

        # Amplitude completion loss
        if mode in ('waveform', 'joint') and 'waveform' in model_out and target_wave is not None:
            pred_amp = model_out['waveform']
            pad_mask = (target_ids != pad_id).float()
            wave_losses = self.waveform_loss(
                pred_amp[:, :-1], target_wave[:, 1:], mask=pad_mask[:, 1:]
            )
            losses.update({f'wave_{k}': v for k, v in wave_losses.items()})
            total = total + self.waveform_weight * wave_losses['total']

        # Waveform gradient consistency loss (runs in both train and val)
        if (self.gradient_weight > 0
                and 'embedding' in model_out
                and frozen_entropy is not None):
            grad_loss = self.gradient_loss(
                model_out['embedding'], target_ids, frozen_entropy
            )
            losses['grad'] = grad_loss
            total = total + self.gradient_weight * grad_loss

        losses['total'] = total
        return losses
