"""
Phase 1: Waveform Completion Proof of Concept — Training Script

Usage:
    python experiments/phase1_train.py
    python experiments/phase1_train.py --fast    # quick smoke test (3 epochs)
    python experiments/phase1_train.py --epochs 50 --batch-size 32
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.collect import collect_corpus
from training.config import QuINNConfig
from training.trainer import QuINNTrainer


def main():
    parser = argparse.ArgumentParser(description="QuINN Phase 1 Training")
    parser.add_argument("--fast",       action="store_true", help="3-epoch smoke test")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--embed-dim",  type=int, default=None)
    parser.add_argument("--manifold-dim", type=int, default=None)
    parser.add_argument("--corpus",     type=str, default="data/corpus.txt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # ── Collect corpus if not already done ───────────────────────────────
    import os
    if not os.path.exists(args.corpus):
        print(f"Collecting Python stdlib files → {args.corpus}")
        n = collect_corpus(args.corpus)
        print(f"Collected {n} files")

    # ── Configure ────────────────────────────────────────────────────────
    cfg = QuINNConfig()
    cfg.corpus_file    = args.corpus
    cfg.checkpoint_dir = args.checkpoint_dir

    if args.fast:
        cfg.n_epochs          = 3
        cfg.batch_size        = 8
        cfg.log_every_n_steps = 5
        cfg.save_every_n_epochs = 3
        print("Fast mode: 3 epochs, batch_size=8")
    else:
        if args.epochs:     cfg.n_epochs     = args.epochs
        if args.batch_size: cfg.batch_size   = args.batch_size
        if args.lr:         cfg.learning_rate = args.lr
        if args.embed_dim:  cfg.embed_dim    = args.embed_dim
        if args.manifold_dim: cfg.manifold_dim = args.manifold_dim

    # ── Run ──────────────────────────────────────────────────────────────
    trainer = QuINNTrainer(cfg)
    history = trainer.train()

    print("\n✓ Training complete.")
    print(f"  Best val length acc: "
          f"{max(r['val']['length_acc_15pct'] for r in history if 'length_acc_15pct' in r['val']):.3f}")


if __name__ == "__main__":
    main()
