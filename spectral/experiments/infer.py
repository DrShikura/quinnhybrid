"""
SpectralLM inference — run from a trained checkpoint.

Usage:
    python spectral/experiments/infer.py --checkpoint checkpoints/spectral_run1/spectral_best.pt
    python spectral/experiments/infer.py --checkpoint ... --prompt "def greet"
    python spectral/experiments/infer.py --checkpoint ... --beam
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from spectral.model.spectral_lm import SpectralLM
from spectral.model.generate import generate, generate_beam


def load_from_checkpoint(checkpoint_path: str):
    """Load model and config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = ckpt['config']

    tokenizer = PythonStructuralTokenizer()

    model = SpectralLM(
        vocab       = tokenizer.vocab,
        embed_dim   = config.get('embed_dim',   64),
        n_layers    = config.get('n_layers',     4),
        n_heads     = config.get('n_heads',      8),
        max_seq_len = config.get('max_seq_len', 512),
        dropout     = 0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    counts = model.param_count()
    print(f"Loaded: {checkpoint_path}")
    print(f"Parameters: {counts['total']:,}  |  val_loss: {ckpt.get('val_loss', '?'):.4f}")

    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt',     default='def ')
    parser.add_argument('--max-new',    type=int,   default=100)
    parser.add_argument('--temperature',type=float, default=0.8)
    parser.add_argument('--top-k',      type=int,   default=50)
    parser.add_argument('--top-p',      type=float, default=0.95)
    parser.add_argument('--beam',       action='store_true')
    parser.add_argument('--n-beams',    type=int,   default=4)
    parser.add_argument('--verbose',    action='store_true',
                        help='Print top-5 predictions at each generation step')
    args = parser.parse_args()

    model, tokenizer, config = load_from_checkpoint(args.checkpoint)

    print(f"\nPrompt: {repr(args.prompt)}")
    print("-" * 50)

    if args.beam:
        result = generate_beam(
            model, tokenizer,
            prompt   = args.prompt,
            max_new  = args.max_new,
            n_beams  = args.n_beams,
        )
    else:
        result = generate(
            model, tokenizer,
            prompt      = args.prompt,
            max_new     = args.max_new,
            temperature = args.temperature,
            top_k       = args.top_k,
            top_p       = args.top_p,
            verbose     = args.verbose,
        )

    print(result)


if __name__ == '__main__':
    main()
