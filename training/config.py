"""
Training configuration for QuINN Phase 1.

Sized for a GTX 1060 (6 GB VRAM) — the most constrained target hardware.
All dimensions can be scaled up for RTX 3090 or cloud GPUs.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QuINNConfig:
    # ── Model dimensions ─────────────────────────────────────────────────
    embed_dim: int = 32          # complex embedding size (each token)
    manifold_dim: int = 64       # internal manifold representation dim
    n_encoder_layers: int = 2    # interference manifold depth (encoder)
    n_decoder_layers: int = 1    # waveform decoder depth
    max_seq_len: int = 650       # maximum sequence length (slightly above max_seq_len_data + BOS/EOS)

    # ── Data ─────────────────────────────────────────────────────────────
    corpus_file: str = "data/corpus.txt"
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05
    min_seq_len: int = 50
    max_seq_len_data: int = 600
    min_prefix_frac: float = 0.10
    max_prefix_frac: float = 0.90

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    n_epochs: int = 30
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 42

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_alpha: float = 1.0      # weight on length prediction loss

    # ── Checkpointing ────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 20

    # ── Device ───────────────────────────────────────────────────────────
    device: str = "auto"         # "auto", "cpu", "cuda", "mps"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
