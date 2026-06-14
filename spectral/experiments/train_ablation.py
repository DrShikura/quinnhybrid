"""
LM-only ablation for SpectralLM — identical architecture, no waveform objective.

Saves to a separate checkpoint directory so it never touches the joint-trained model.

Usage (Windows):
    python spectral/experiments/train_ablation.py \
        --corpus-dir "C:\\Users\\Danny\\AppData\\Local\\Python\\pythoncore-3.14-64\\Lib" \
        --epochs 50

After training, compare band specialization:
    python spectral/experiments/band_analysis.py --checkpoint checkpoints/spectral_small/spectral_best.pt
    python spectral/experiments/band_analysis.py --checkpoint checkpoints/spectral_lm_ablation/spectral_best.pt
"""

import argparse
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from spectral.training.dataset import SpectralDataset, collate_fn
from spectral.training.trainer import SpectralTrainer, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--embed-dim',   type=int,   default=32)
    parser.add_argument('--n-layers',    type=int,   default=4)
    parser.add_argument('--n-heads',     type=int,   default=4)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--max-seq-len', type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--corpus',      default='data/corpus.txt')
    parser.add_argument('--corpus-dir',  default=None)
    parser.add_argument('--checkpoint-dir', default='checkpoints/spectral_lm_ablation',
                        help='MUST differ from the joint checkpoint dir')
    args = parser.parse_args()

    # Safety check — refuse to write over the joint model
    ckpt_dir = Path(args.checkpoint_dir)
    if ckpt_dir.resolve() == Path('checkpoints/spectral_small').resolve():
        print("ERROR: --checkpoint-dir must not point at checkpoints/spectral_small")
        sys.exit(1)

    tokenizer = PythonStructuralTokenizer()

    # Identical architecture to joint model; mode = 'lm' only
    config = {
        'mode':            'lm',
        'embed_dim':       args.embed_dim,
        'n_layers':        args.n_layers,
        'n_heads':         args.n_heads,
        'max_seq_len':     args.max_seq_len,
        'n_epochs':        args.epochs,
        'lr':              args.lr,
        'lm_weight':       1.0,
        'waveform_weight': 0.0,
        'wave_schedule':   'none',
        'band_init':       True,    # same init as joint model for fair comparison
        'acoustic_init':   True,
        'band_weights':    [1.5, 1.0, 0.5],
        'weight_decay':    0.01,
    }

    corpus_source = args.corpus_dir if args.corpus_dir else args.corpus
    print(f"[LM-only ablation]  corpus: {corpus_source}")
    print(f"                    checkpoint → {args.checkpoint_dir}")

    train_ds = SpectralDataset(corpus_source, tokenizer,
                               max_seq_len=config['max_seq_len'], split='train')
    val_ds   = SpectralDataset(corpus_source, tokenizer,
                               max_seq_len=config['max_seq_len'], split='val')

    pad_id  = tokenizer.pad_id
    collate = partial(collate_fn, pad_id=pad_id)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate, num_workers=0)

    model   = build_model(config, tokenizer.vocab)
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
