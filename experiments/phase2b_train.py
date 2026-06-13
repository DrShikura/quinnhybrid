"""
Phase 2B training script: Hybrid Quinn/Transformer for token prediction.

The Transformer uses Quinn's waveform as structural guidance to predict tokens.

Usage:
    python experiments/phase2b_train.py --epochs 30 --checkpoint-dir checkpoints/phase2b_run1

Optional:
    --quinn-checkpoint checkpoints/phase2a_run1/quinn_epoch050.pt
        Load a trained Quinn from Phase 2A (default: random)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from training.trainer_phase2b import QuINNTransformerTrainer
from training.config import QuINNConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/phase2b")
    parser.add_argument("--quinn-checkpoint", default=None,
                        help="Path to Quinn checkpoint (Phase 2A or Phase 1)")
    args = parser.parse_args()

    cfg = QuINNConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = QuINNTransformerTrainer(cfg, quinn_checkpoint=args.quinn_checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
