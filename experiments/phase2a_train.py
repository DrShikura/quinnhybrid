"""
Phase 2A training script: Contrastive waveform learning.

Usage:
    python experiments/phase2a_train.py --epochs 30 --checkpoint-dir checkpoints/run_phase2a_1

This trains QuINN with multi-task loss:
  1. Waveform completion (positive target)
  2. Length prediction
  3. Contrastive: distinguish positive from negative waveforms
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from training.trainer_phase2a import QuINNTrainerPhase2A
from training.config import QuINNConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/phase2a")
    parser.add_argument("--contrastive-weight", type=float, default=1.0,
                        help="weight on contrastive loss relative to waveform+length")
    args = parser.parse_args()

    cfg = QuINNConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = QuINNTrainerPhase2A(cfg, contrastive_weight=args.contrastive_weight)
    trainer.train()


if __name__ == "__main__":
    main()
