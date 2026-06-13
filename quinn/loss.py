"""
Loss functions for QuINN waveform completion training.

WaveformCompletionLoss: L2 distance in complex space between predicted
    and true waveforms. Decomposes into amplitude error and phase error
    so training can be monitored separately.

LengthPredictionLoss: smooth L1 on log-scale length predictions.
    Log-scale keeps the loss scale-invariant (off by 50 tokens on a
    100-token file is equally bad as off by 500 on a 1000-token file).

QuINNLoss: weighted combination of the two, with optional annealing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveformCompletionLoss(nn.Module):
    """
    Phase-alignment waveform completion loss (scale-invariant).

    Uses complex cosine similarity so the loss is invariant to amplitude
    scale. This avoids the collapse problem (where both encoder and decoder
    drift toward zero amplitude) and the explosion problem (where initial
    amplitude mismatch between encoder [~0.02] and decoder [~1.0] produces
    astronomically large gradients from an amplitude-matching term).

        phase_loss  = 1 - Re(pred * conj(true)) / (|pred| * |true|)   ∈ [0, 2]
        waveform_loss = mean(phase_loss)                               ∈ [0, 2]

    For monitoring, amplitude and actual L2 are also reported (detached).
    """

    def forward(
        self,
        predicted: torch.Tensor,   # (B, T, D) complex
        target: torch.Tensor,      # (B, T, D) complex
        mask: torch.Tensor = None, # (B, T) bool – True = valid position
    ) -> dict:
        # Complex cosine similarity per element
        dot       = predicted.real * target.real + predicted.imag * target.imag
        pred_norm = predicted.abs().clamp(min=1e-8)
        true_norm = target.abs().clamp(min=1e-8)
        cos_sim   = dot / (pred_norm * true_norm)  # ∈ [-1, 1]
        phase_loss = 1.0 - cos_sim                 # ∈ [0, 2]

        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)    # (B, T, 1)
            phase_loss = phase_loss * mask_f
            n_valid = mask_f.sum().clamp(min=1) * phase_loss.shape[-1]
            total_loss = phase_loss.sum() / n_valid
        else:
            total_loss = phase_loss.mean()

        # Monitoring metrics (no gradient needed)
        with torch.no_grad():
            amp_loss = (pred_norm - true_norm).pow(2).mean()
            diff = predicted - target
            l2_loss = (diff.real.pow(2) + diff.imag.pow(2)).mean()

        return {
            "waveform_loss": total_loss,
            "amplitude_loss": amp_loss,
            "phase_loss": total_loss.detach(),   # same as waveform_loss here
            "l2_loss": l2_loss,
        }


class LengthPredictionLoss(nn.Module):
    """
    Smooth L1 on log-scale length predictions.

        L = SmoothL1(log_pred_len, log_true_len)

    Predicting and comparing in log space means a factor-of-2 error
    has the same cost regardless of absolute length.
    """

    def forward(
        self,
        pred_log_len: torch.Tensor,  # (B,)
        true_len: torch.Tensor,      # (B,) integer counts
    ) -> torch.Tensor:
        true_log_len = torch.log(true_len.float().clamp(min=1))
        return F.smooth_l1_loss(pred_log_len, true_log_len)


class QuINNLoss(nn.Module):
    """
    Combined loss for QuINN Phase 1.

        total = waveform_loss + alpha * length_loss

    alpha starts high (length prediction is the validation metric) then
    can be annealed down as waveform quality improves.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.waveform_loss_fn = WaveformCompletionLoss()
        self.length_loss_fn = LengthPredictionLoss()

    def forward(
        self,
        predicted_waveform: torch.Tensor,  # (B, T, D) complex
        true_waveform: torch.Tensor,       # (B, T, D) complex
        pred_log_len: torch.Tensor,        # (B,)
        true_len: torch.Tensor,            # (B,)
        target_mask: torch.Tensor = None,  # (B, T) bool
    ) -> dict:
        wf = self.waveform_loss_fn(predicted_waveform, true_waveform, target_mask)
        len_loss = self.length_loss_fn(pred_log_len, true_len)

        total = wf["waveform_loss"] + self.alpha * len_loss

        return {
            "loss": total,
            "waveform_loss": wf["waveform_loss"],
            "amplitude_loss": wf["amplitude_loss"],
            "phase_loss": wf["phase_loss"],
            "length_loss": len_loss,
        }
