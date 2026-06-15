#!/usr/bin/env python3
"""
Train BPE vocabulary from corpus. Can run in background.

Usage:
    python train_bpe_vocab.py --vocab-size 5000 --corpus data/corpus.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.bpe_tokenizer import BytePairTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on Python corpus")
    parser.add_argument('--vocab-size', type=int, default=5000,
                        help='Target vocabulary size (default 5000)')
    parser.add_argument('--corpus', default='data/corpus.txt',
                        help='Corpus path: .txt file or directory (default data/corpus.txt)')
    parser.add_argument('--save-path', default='data/bpe_vocab',
                        help='Where to save vocab.json and merges.txt')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample N files from corpus (for quick testing)')
    args = parser.parse_args()

    print(f"Training BPE tokenizer...")
    print(f"  Corpus: {args.corpus}")
    print(f"  Target vocab size: {args.vocab_size}")
    print(f"  Save to: {args.save_path}")
    if args.sample_size:
        print(f"  Using {args.sample_size} sample files (not full corpus)")

    try:
        tokenizer = BytePairTokenizer.train(
            corpus_files=args.corpus,
            vocab_size=args.vocab_size,
            save_path=args.save_path,
        )
        print(f"\n✓ Success! BPE vocab trained with {tokenizer.vocab_size} tokens")
        print(f"  Vocabulary and merges saved to: {args.save_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
