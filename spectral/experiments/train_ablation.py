"""
SpectralLM architecture ablation — new-arch vs old-arch baselines.

Four conditions isolate each new component against an LM-only reference:

  A  full-new:       joint + Laplacian init + gradient-consistency loss  ← full spec
  B  lm-only-new:    lm    + Laplacian init + no spectral losses         ← LM baseline
  C  no-laplacian:   joint + acoustic init  + gradient-consistency loss  ← Laplacian value
  D  no-grad-loss:   joint + Laplacian init + gradient_weight=0          ← grad-consistency value

Old-arch reference (previous experiments):
  exp_a_joint: 0.6968  (SVD init, nn.MHA, joint)
  exp_b_lm:    0.6927  (SVD init, nn.MHA, lm-only)  ← previous best

Usage:
    python spectral/experiments/train_ablation.py --epochs 30
    python spectral/experiments/train_ablation.py --corpus-dir /path/to/pyfiles --epochs 30
"""

import argparse
import json
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from spectral.training.dataset import SpectralDataset, collate_fn
from spectral.training.trainer import SpectralTrainer, build_model


# ── Condition definitions ────────────────────────────────────────────────────

CONDITIONS = {
    'A_full_new': {
        'label': 'A  full-new (joint + Laplacian + grad-consistency)',
        'mode': 'joint',
        'laplacian_init': True,
        'gradient_weight': 0.1,
        'wave_schedule': 'cosine',
        'waveform_weight': 0.5,
    },
    'B_lm_only_new': {
        'label': 'B  lm-only-new (LM only, Laplacian init)',
        'mode': 'lm',
        'laplacian_init': True,
        'gradient_weight': 0.0,
        'wave_schedule': 'none',
        'waveform_weight': 0.0,
    },
    'C_no_laplacian': {
        'label': 'C  no-laplacian (joint + acoustic init + grad-consistency)',
        'mode': 'joint',
        'laplacian_init': False,
        'gradient_weight': 0.1,
        'wave_schedule': 'cosine',
        'waveform_weight': 0.5,
    },
    'D_no_grad_loss': {
        'label': 'D  no-grad-loss (joint + Laplacian init, gradient_weight=0)',
        'mode': 'joint',
        'laplacian_init': True,
        'gradient_weight': 0.0,
        'wave_schedule': 'cosine',
        'waveform_weight': 0.5,
    },
}


def run_condition(name: str, cond: dict, args, tokenizer,
                  train_ds, val_ds, laplacian_eigvecs) -> dict:
    print(f"\n{'='*70}")
    print(f"  Running condition: {cond['label']}")
    print(f"{'='*70}")

    ckpt_dir = Path(args.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate, num_workers=0)

    config = {
        'mode':              cond['mode'],
        'embed_dim':         args.embed_dim,
        'n_layers':          args.n_layers,
        'n_heads':           args.n_heads,
        'max_seq_len':       args.max_seq_len,
        'n_epochs':          args.epochs,
        'lr':                args.lr,
        'lm_weight':         1.0,
        'waveform_weight':   cond['waveform_weight'],
        'wave_schedule':     cond['wave_schedule'],
        'band_init':         True,
        'acoustic_init':     True,     # fallback when laplacian_init=False
        'laplacian_init':    cond['laplacian_init'],
        'gradient_weight':   cond['gradient_weight'],
        'entropy_threshold': args.entropy_threshold,
        'band_weights':      [1.5, 1.0, 0.5],
        'weight_decay':      0.01,
    }

    eigvecs = laplacian_eigvecs if cond['laplacian_init'] else None
    model = build_model(config, tokenizer.vocab, laplacian_eigvecs=eigvecs)

    trainer = SpectralTrainer(
        model          = model,
        train_loader   = train_dl,
        val_loader     = val_dl,
        config         = config,
        checkpoint_dir = ckpt_dir,
        tokenizer      = tokenizer,
        frozen_entropy = train_ds.frozen_entropy,
    )
    trainer.train()

    best_val = min(r['val']['loss'] for r in trainer.history)
    best_lm  = min(r['val']['lm_loss'] for r in trainer.history)
    return {'best_val': best_val, 'best_lm': best_lm, 'history': trainer.history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',           type=int,   default=30)
    parser.add_argument('--embed-dim',        type=int,   default=32)
    parser.add_argument('--n-layers',         type=int,   default=4)
    parser.add_argument('--n-heads',          type=int,   default=4)
    parser.add_argument('--batch-size',       type=int,   default=32)
    parser.add_argument('--max-seq-len',      type=int,   default=256)
    parser.add_argument('--lr',               type=float, default=3e-4)
    parser.add_argument('--entropy-threshold',type=float, default=0.4)
    parser.add_argument('--corpus',           default='data/corpus.txt')
    parser.add_argument('--corpus-dir',       default=None)
    parser.add_argument('--checkpoint-dir',   default='checkpoints/ablation_new_arch',
                        help='Parent directory; each condition saves to a subdirectory')
    parser.add_argument('--conditions',       nargs='+',
                        choices=list(CONDITIONS.keys()),
                        default=list(CONDITIONS.keys()),
                        help='Which conditions to run (default: all four)')
    args = parser.parse_args()

    tokenizer = PythonStructuralTokenizer()
    corpus_source = args.corpus_dir if args.corpus_dir else args.corpus
    print(f"Loading corpus from: {corpus_source}")

    train_ds = SpectralDataset(corpus_source, tokenizer,
                               max_seq_len=args.max_seq_len, split='train')
    val_ds   = SpectralDataset(corpus_source, tokenizer,
                               max_seq_len=args.max_seq_len, split='val')

    # Laplacian eigvecs computed once, shared by all conditions that use them
    laplacian_eigvecs = train_ds.laplacian_eigvecs

    results = {}
    for name in args.conditions:
        results[name] = run_condition(
            name, CONDITIONS[name], args, tokenizer,
            train_ds, val_ds, laplacian_eigvecs,
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ABLATION RESULTS — new architecture")
    print(f"  epochs={args.epochs}  embed_dim={args.embed_dim}  "
          f"n_layers={args.n_layers}  seq_len={args.max_seq_len}")
    print(f"{'='*70}")
    print(f"  {'Condition':<52}  best_lm   best_val")
    print(f"  {'-'*52}  --------  --------")
    for name in args.conditions:
        r = results[name]
        label = CONDITIONS[name]['label']
        print(f"  {label:<52}  {r['best_lm']:.4f}    {r['best_val']:.4f}")

    print(f"\n  Old-arch reference (30 epochs, same corpus):")
    print(f"  {'exp_a_joint (SVD init, MHA, joint)':<52}  0.6968    0.6968")
    print(f"  {'exp_b_lm   (SVD init, MHA, lm-only)':<52}  0.6927    0.6927")

    # Save summary
    summary_path = Path(args.checkpoint_dir) / 'ablation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({name: {'best_lm': r['best_lm'], 'best_val': r['best_val']}
                   for name, r in results.items()}, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")


if __name__ == '__main__':
    main()
