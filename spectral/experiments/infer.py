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
from spectral.model.baseline_gpt import BaselineGPT
from spectral.model.wave_gpt import WaveGPT
from spectral.training.trainer import build_model
from spectral.model.generate import generate, generate_beam


def load_from_checkpoint(checkpoint_path: str, model_override: str = 'auto'):
    ckpt      = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config    = ckpt['config']
    tokenizer = PythonStructuralTokenizer()

    model_type = (model_override if model_override != 'auto'
                  else config.get('model_type', 'spectral'))

    if model_type == 'spectral':
        model = build_model(config, tokenizer.vocab)
    elif model_type == 'wave-gpt':
        d = config.get('d_model', config.get('embed_dim', 48))
        model = WaveGPT(
            len(tokenizer.vocab),
            d_model     = d,
            n_heads     = config.get('n_heads', 4),
            n_layers    = config.get('n_layers', config.get('n_loops', 2)),
            max_seq_len = config.get('max_seq_len', 256),
            ff_mult     = config.get('ff_mult', 3),
        )
    else:
        d, n = (64, 1) if model_type == 'baseline-1l' else (48, 2)
        model = BaselineGPT(len(tokenizer.vocab), d_model=d, n_heads=4,
                            n_layers=n, max_seq_len=config.get('max_seq_len', 256))

    model.load_state_dict(ckpt['model_state'])
    model.eval()
    counts = model.param_count()
    print(f"Loaded: {checkpoint_path}  [{model_type}]")
    print(f"Parameters: {counts['total']:,}  |  val_loss: {ckpt.get('val_loss', '?'):.4f}")
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model',      default='auto',
                        choices=['auto', 'spectral', 'baseline-1l', 'baseline-2l', 'wave-gpt'],
                        help='Model type (default: auto-detect from checkpoint)')
    parser.add_argument('--prompt',     default='def ')
    parser.add_argument('--max-new',    type=int,   default=100)
    parser.add_argument('--temperature',type=float, default=0.8)
    parser.add_argument('--top-k',      type=int,   default=50)
    parser.add_argument('--top-p',      type=float, default=0.95)
    parser.add_argument('--beam',       action='store_true')
    parser.add_argument('--n-beams',    type=int,   default=4)
    parser.add_argument('--verbose',    action='store_true',
                        help='Print top-5 predictions at each generation step')
    parser.add_argument('--device',     default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    model, tokenizer, config = load_from_checkpoint(args.checkpoint, args.model)

    # Interpret common escape sequences so --prompt "import os\ndef " works
    # on Windows where the shell passes \n as a literal backslash-n.
    prompt = args.prompt.replace('\\n', '\n').replace('\\t', '\t')

    print(f"\nPrompt: {repr(prompt)}")
    print("-" * 50)

    if args.beam:
        result = generate_beam(
            model, tokenizer,
            prompt   = prompt,
            max_new  = args.max_new,
            n_beams  = args.n_beams,
            device   = device,
        )
    else:
        result = generate(
            model, tokenizer,
            prompt      = prompt,
            max_new     = args.max_new,
            temperature = args.temperature,
            top_k       = args.top_k,
            top_p       = args.top_p,
            verbose     = args.verbose,
            device      = device,
        )

    print(result)


if __name__ == '__main__':
    main()
