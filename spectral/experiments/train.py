"""
SpectralLM training entry point.

Usage:
    python spectral/experiments/train.py [--mode joint|lm|waveform] [--epochs N]

Defaults to joint training (waveform completion + language modeling).
"""

import argparse
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from spectral.model.spectral_lm import SpectralLM
from spectral.training.dataset import SpectralDataset, collate_fn
from spectral.training.trainer import SpectralTrainer, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',        default='joint',
                        choices=['joint', 'lm', 'waveform'])
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--embed-dim',   type=int, default=64)
    parser.add_argument('--n-layers',    type=int, default=4)
    parser.add_argument('--n-heads',     type=int, default=8)
    parser.add_argument('--batch-size',  type=int, default=16)
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--lm-weight',   type=float, default=1.0)
    parser.add_argument('--wave-weight', type=float, default=0.5)
    parser.add_argument('--corpus',      default='data/corpus.txt')
    parser.add_argument('--checkpoint-dir', default='checkpoints/spectral_run1')
    args = parser.parse_args()

    tokenizer = PythonStructuralTokenizer()

    config = {
        'mode':           args.mode,
        'embed_dim':      args.embed_dim,
        'n_layers':       args.n_layers,
        'n_heads':        args.n_heads,
        'max_seq_len':    args.max_seq_len,
        'n_epochs':       args.epochs,
        'lr':             args.lr,
        'lm_weight':      args.lm_weight,
        'waveform_weight': args.wave_weight,
        'band_weights':   [1.5, 1.0, 0.5],
        'weight_decay':   0.01,
    }

    print("Loading corpus...")
    train_ds = SpectralDataset(args.corpus, tokenizer,
                                max_seq_len=args.max_seq_len, split='train')
    val_ds   = SpectralDataset(args.corpus, tokenizer,
                                max_seq_len=args.max_seq_len, split='val')

    pad_id  = tokenizer.pad_id
    collate = partial(collate_fn, pad_id=pad_id)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True,  collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate, num_workers=0)

    print("\nBuilding SpectralLM...")
    model = build_model(config, tokenizer.vocab)

    trainer = SpectralTrainer(
        model          = model,
        train_loader   = train_dl,
        val_loader     = val_dl,
        config         = config,
        checkpoint_dir = args.checkpoint_dir,
        tokenizer      = tokenizer,
    )

    trainer.train()


if __name__ == '__main__':
    main()
