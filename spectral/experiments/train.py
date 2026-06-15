"""
SpectralLM training entry point.

Usage:
    python spectral/experiments/train.py [--mode joint|lm|waveform] [--epochs N]
    python spectral/experiments/train.py --resume checkpoints/spectral_small/spectral_best.pt
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
    parser.add_argument('--epochs',      type=int, default=50)
    parser.add_argument('--embed-dim',   type=int, default=32)
    parser.add_argument('--n-layers',    type=int, default=4)
    parser.add_argument('--n-heads',     type=int, default=4)
    parser.add_argument('--batch-size',  type=int, default=32)
    parser.add_argument('--max-seq-len', type=int, default=256)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--lm-weight',      type=float, default=1.0)
    parser.add_argument('--wave-weight',    type=float, default=0.5)
    parser.add_argument('--wave-schedule',  default='cosine',
                        choices=['cosine', 'linear', 'none'],
                        help='How to anneal waveform loss weight over epochs')
    parser.add_argument('--n-loops',         type=int,   default=3,
                        help='How many times to loop the single shared transformer layer (R)')
    parser.add_argument('--mem-dim',         type=int,   default=32,
                        help='Entity memory fast-weight dimension (0 to disable)')
    parser.add_argument('--memory-weight',   type=float, default=0.1,
                        help='Entity memory associative loss weight (constant; 0 to disable)')
    parser.add_argument('--gradient-weight', type=float, default=0.1,
                        help='Gradient consistency loss weight (constant; 0 to disable)')
    parser.add_argument('--entropy-threshold', type=float, default=0.4,
                        help='H_norm below which a position is "predictable"')
    parser.add_argument('--no-band-init', action='store_true',
                        help='Initialize all frequencies uniformly in [0.02,0.50] '
                             'instead of banded — tests whether structure self-organizes')
    parser.add_argument('--no-acoustic-init', action='store_true',
                        help='Fallback amplitude init: random instead of character '
                             'acoustic profiles (only used when --no-laplacian-init)')
    parser.add_argument('--no-laplacian-init', action='store_true',
                        help='Skip Laplacian eigenvector amplitude init '
                             '(falls back to acoustic or random)')
    parser.add_argument('--corpus',      default='data/corpus.txt',
                        help='.txt file with one Python file path per line')
    parser.add_argument('--corpus-dir',  default=None,
                        help='Directory to recursively search for .py files '
                             '(overrides --corpus if provided)')
    parser.add_argument('--checkpoint-dir', default='checkpoints/spectral_small')
    parser.add_argument('--resume',      default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device',      default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to train on (default: auto-detect)')
    parser.add_argument('--no-compile',  action='store_true',
                        help='Disable torch.compile (needed for older GPUs < SM 7.0)')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = PythonStructuralTokenizer()

    if args.resume:
        ckpt   = torch.load(args.resume, map_location='cpu')
        config = ckpt['config']
        print(f"Resuming from: {args.resume}  (epoch {ckpt['epoch']})")
    else:
        config = {
            'mode':              args.mode,
            'embed_dim':         args.embed_dim,
            'n_loops':           args.n_loops,
            'n_heads':           args.n_heads,
            'max_seq_len':       args.max_seq_len,
            'n_epochs':          args.epochs,
            'lr':                args.lr,
            'lm_weight':         args.lm_weight,
            'waveform_weight':   args.wave_weight,
            'wave_schedule':     args.wave_schedule,
            'band_init':         not args.no_band_init,
            'acoustic_init':     not args.no_acoustic_init,
            'laplacian_init':    not args.no_laplacian_init,
            'gradient_weight':   args.gradient_weight,
            'entropy_threshold': args.entropy_threshold,
            'mem_dim':           args.mem_dim,
            'memory_weight':     args.memory_weight,
            'band_weights':      [1.5, 1.0, 0.5],
            'weight_decay':      0.01,
        }

    config['n_epochs'] = args.epochs

    corpus_source = args.corpus_dir if args.corpus_dir else args.corpus
    print(f"Loading corpus from: {corpus_source}")
    train_ds = SpectralDataset(corpus_source, tokenizer,
                               max_seq_len=config['max_seq_len'], split='train')
    val_ds   = SpectralDataset(corpus_source, tokenizer,
                               max_seq_len=config['max_seq_len'], split='val')

    pad_id  = tokenizer.pad_id
    collate = partial(collate_fn, pad_id=pad_id)

    pin = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  collate_fn=collate, num_workers=0,
                          pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate, num_workers=0,
                          pin_memory=pin)

    # Laplacian eigenvectors for amplitude initialization (can be disabled)
    use_laplacian = config.get('laplacian_init', True)
    laplacian_eigvecs = train_ds.laplacian_eigvecs if use_laplacian else None

    print("\nBuilding SpectralLM...")
    model = build_model(config, tokenizer.vocab,
                        laplacian_eigvecs=laplacian_eigvecs)

    if device.type == 'cuda' and hasattr(torch, 'compile') and not args.no_compile:
        print("Compiling model with torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode='reduce-overhead')

    trainer = SpectralTrainer(
        model          = model,
        train_loader   = train_dl,
        val_loader     = val_dl,
        config         = config,
        checkpoint_dir = args.checkpoint_dir,
        tokenizer      = tokenizer,
        frozen_entropy = train_ds.frozen_entropy,
        device         = device,
    )

    if args.resume:
        start_epoch = trainer.resume(args.resume)
        trainer._start_epoch = start_epoch

    trainer.train()


if __name__ == '__main__':
    main()
