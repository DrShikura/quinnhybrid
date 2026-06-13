"""
PyTorch Dataset for QuINN Phase 1 waveform completion training.

Each sample is drawn from a Python source file:
  1. File is tokenised to structural token IDs (BOS...EOS)
  2. A random prefix fraction is sampled (10–90% of total length)
  3. Returns:
       prefix_ids       - token IDs for the prefix
       prefix_positions - position indices (0-based)
       all_ids          - token IDs for the full sequence (prediction target)
       all_positions    - position indices for the full sequence
       seq_len          - true total length (for length prediction target)

The DataLoader collate_fn pads batches and builds boolean masks.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import PythonStructuralTokenizer


class PythonCodeDataset(Dataset):
    """
    Dataset loading Python source files and producing random-prefix training samples.

    Each __getitem__ call samples a fresh random prefix fraction, so the same
    file produces different prefix lengths across epochs — essential for QuINN
    to learn from variable-length partial evidence.

    Args:
        file_list:       list of file paths (or a .txt file with one path per line)
        min_seq_len:     minimum total token count to include a file
        max_seq_len:     truncate sequences longer than this
        min_prefix_frac: minimum prefix fraction (default 0.10)
        max_prefix_frac: maximum prefix fraction (default 0.90)
        tokenizer:       shared tokenizer instance (creates new if None)
        seed:            optional RNG seed for reproducibility in validation set
    """

    def __init__(
        self,
        file_list,
        min_seq_len: int = 30,
        max_seq_len: int = 1024,
        min_prefix_frac: float = 0.10,
        max_prefix_frac: float = 0.90,
        tokenizer: Optional[PythonStructuralTokenizer] = None,
        seed: Optional[int] = None,
    ):
        self.max_seq_len = max_seq_len
        self.min_prefix_frac = min_prefix_frac
        self.max_prefix_frac = max_prefix_frac
        self.tokenizer = tokenizer or PythonStructuralTokenizer()
        self.rng = random.Random(seed)

        # Accept either a list of paths or a file containing paths
        if isinstance(file_list, (str, Path)):
            paths = Path(file_list).read_text().splitlines()
            paths = [Path(p.strip()) for p in paths if p.strip()]
        else:
            paths = [Path(p) for p in file_list]

        # Tokenise all files upfront and filter by length
        self.samples: List[List[int]] = []
        for path in paths:
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                ids = self.tokenizer.encode(source, add_special=True)
                # Filter only: include files strictly within length bounds.
                # Truncating to max_seq_len would cluster many files at exactly
                # that length, making the mean-prediction baseline artificially
                # accurate (since max_seq_len ≈ mean gives ~14% relative error
                # which falls within the ±15% success criterion).
                if min_seq_len <= len(ids) <= max_seq_len:
                    self.samples.append(ids)
            except Exception:
                pass  # silently skip unreadable files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ids = self.samples[idx]
        n = len(ids)

        # Sample a random prefix fraction
        frac = self.rng.uniform(self.min_prefix_frac, self.max_prefix_frac)
        prefix_len = max(2, int(round(n * frac)))
        prefix_len = min(prefix_len, n - 1)  # always at least 1 token in tail

        prefix_ids = ids[:prefix_len]
        all_ids = ids  # complete sequence

        return {
            "prefix_ids":       torch.tensor(prefix_ids, dtype=torch.long),
            "prefix_positions": torch.arange(prefix_len, dtype=torch.long),
            "all_ids":          torch.tensor(all_ids, dtype=torch.long),
            "all_positions":    torch.arange(n, dtype=torch.long),
            "seq_len":          torch.tensor(n, dtype=torch.long),
            "prefix_frac":      torch.tensor(frac, dtype=torch.float),
        }


def collate_fn(batch: list) -> dict:
    """
    Pads a batch of variable-length samples.
    Returns boolean masks (True = valid, not padding).
    """
    from .tokenizer import PAD_ID

    max_prefix = max(b["prefix_ids"].shape[0] for b in batch)
    max_all    = max(b["all_ids"].shape[0] for b in batch)

    prefix_ids_padded  = torch.full((len(batch), max_prefix), PAD_ID, dtype=torch.long)
    prefix_pos_padded  = torch.zeros(len(batch), max_prefix, dtype=torch.long)
    prefix_mask        = torch.zeros(len(batch), max_prefix, dtype=torch.bool)

    all_ids_padded     = torch.full((len(batch), max_all), PAD_ID, dtype=torch.long)
    all_pos_padded     = torch.zeros(len(batch), max_all, dtype=torch.long)
    all_mask           = torch.zeros(len(batch), max_all, dtype=torch.bool)

    seq_lens = torch.stack([b["seq_len"] for b in batch])
    prefix_fracs = torch.stack([b["prefix_frac"] for b in batch])

    for i, b in enumerate(batch):
        p = b["prefix_ids"].shape[0]
        t = b["all_ids"].shape[0]

        prefix_ids_padded[i, :p] = b["prefix_ids"]
        prefix_pos_padded[i, :p] = b["prefix_positions"]
        prefix_mask[i, :p] = True

        all_ids_padded[i, :t] = b["all_ids"]
        all_pos_padded[i, :t] = b["all_positions"]
        all_mask[i, :t] = True

    return {
        "prefix_tokens":    prefix_ids_padded,
        "prefix_positions": prefix_pos_padded,
        "prefix_mask":      prefix_mask,
        "target_tokens":    all_ids_padded,
        "target_positions": all_pos_padded,
        "target_mask":      all_mask,
        "seq_len":          seq_lens,
        "prefix_frac":      prefix_fracs,
    }
