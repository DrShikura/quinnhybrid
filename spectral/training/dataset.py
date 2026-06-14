"""
Dataset for SpectralLM — simple next-token prediction setup.

Each sample: full tokenized file, returned as (token_ids, position_ids).
DataLoader handles batching and padding.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.tokenizer import PythonStructuralTokenizer


class SpectralDataset(Dataset):
    """
    Loads Python source files and returns full token sequences.

    Args:
        file_list:   path to corpus.txt (one file path per line), or list of paths
        tokenizer:   PythonStructuralTokenizer instance
        max_seq_len: truncate sequences longer than this
        min_seq_len: skip files shorter than this
        split:       'train', 'val', or 'test'
        val_frac:    fraction of files for validation
        test_frac:   fraction of files for test
        seed:        random seed for split
    """

    def __init__(
        self,
        file_list,
        tokenizer:   Optional[PythonStructuralTokenizer] = None,
        max_seq_len: int = 512,
        min_seq_len: int = 20,
        split:       str = 'train',
        val_frac:    float = 0.10,
        test_frac:   float = 0.05,
        seed:        int = 42,
    ):
        self.tokenizer   = tokenizer or PythonStructuralTokenizer()
        self.max_seq_len = max_seq_len

        # Load file paths — accepts:
        #   - a .txt file with one path per line
        #   - a directory (recursively finds all .py files)
        #   - a list of paths
        if isinstance(file_list, (str, Path)):
            p = Path(file_list)
            if p.is_dir():
                paths = sorted(p.rglob('*.py'))
            else:
                lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
                paths = [Path(ln.strip()) for ln in lines if ln.strip()]
        else:
            paths = [Path(p) for p in file_list]

        paths = [p for p in paths if p.exists()]
        if not paths:
            raise FileNotFoundError(
                f"No Python files found from corpus source: {file_list}\n"
                f"  Use --corpus-dir to point at a directory of .py files, "
                f"or --corpus for a file-list .txt."
            )

        # Tokenize all files
        all_samples: List[List[int]] = []
        for path in paths:
            try:
                source = path.read_text(encoding='utf-8', errors='replace')
                ids    = self.tokenizer.encode(source, add_special=True)
                if min_seq_len <= len(ids):
                    ids = ids[:max_seq_len]
                    all_samples.append(ids)
            except Exception:
                pass

        # Reproducible split
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n     = len(all_samples)
        n_val  = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))

        if split == 'test':
            self.samples = all_samples[:n_test]
        elif split == 'val':
            self.samples = all_samples[n_test:n_test + n_val]
        else:
            self.samples = all_samples[n_test + n_val:]

        print(f"  {split}: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.samples[idx]
        token_ids    = torch.tensor(ids, dtype=torch.long)
        position_ids = torch.arange(len(ids), dtype=torch.long)
        return token_ids, position_ids


def collate_fn(batch, pad_id: int = 0):
    """Pad batch to same length."""
    token_seqs, pos_seqs = zip(*batch)
    max_len = max(t.size(0) for t in token_seqs)

    padded_tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_pos    = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (tok, pos) in enumerate(zip(token_seqs, pos_seqs)):
        L = tok.size(0)
        padded_tokens[i, :L] = tok
        padded_pos[i, :L]    = pos

    return padded_tokens, padded_pos
