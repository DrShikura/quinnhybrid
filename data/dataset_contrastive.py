"""
Phase 2A dataset: yields (positive, negative) pairs for contrastive training.

Each batch contains:
  - Prefix (shared positive and negative)
  - Target positive (correct full sequence)
  - Target negative (structurally-plausible wrong sequence)

Both share the same prefix fraction to ensure fair comparison.
"""

import random
import torch
from torch.utils.data import Dataset
from typing import Optional, List

from .tokenizer import PythonStructuralTokenizer
from .hard_negatives import HardNegativeGenerator
from .dataset import PythonCodeDataset, collate_fn as collate_fn_base


class PythonCodeDatasetContrastive(PythonCodeDataset):
    """
    Dataset that yields (positive, negative) pairs for contrastive training.

    For each file:
      - Positive: original sequence
      - Negative: hard negative (tokens swapped within categories)
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
        neg_seed: Optional[int] = 42,  # separate seed for negative generation
        neg_num_swaps: int = 2,  # how many tokens to swap
    ):
        super().__init__(
            file_list=file_list,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            min_prefix_frac=min_prefix_frac,
            max_prefix_frac=max_prefix_frac,
            tokenizer=tokenizer,
            seed=seed,
        )
        self.neg_generator = HardNegativeGenerator(seed=neg_seed, swap_count=neg_num_swaps)

    def __getitem__(self, idx: int) -> dict:
        """Return positive and negative samples."""
        ids = self.samples[idx]
        n = len(ids)

        # Sample a random prefix fraction (same for positive and negative)
        frac = self.rng.uniform(self.min_prefix_frac, self.max_prefix_frac)
        prefix_len = max(2, int(round(n * frac)))
        prefix_len = min(prefix_len, n - 1)

        # Positive: original sequence
        prefix_ids = ids[:prefix_len]
        all_ids_positive = ids

        # Negative: hard negative of the full sequence
        all_ids_negative = self.neg_generator.generate(ids)

        return {
            "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
            "prefix_positions": torch.arange(prefix_len, dtype=torch.long),
            "all_ids_positive": torch.tensor(all_ids_positive, dtype=torch.long),
            "all_positions_positive": torch.arange(n, dtype=torch.long),
            "all_ids_negative": torch.tensor(all_ids_negative, dtype=torch.long),
            "all_positions_negative": torch.arange(n, dtype=torch.long),
            "seq_len": torch.tensor(n, dtype=torch.long),
            "prefix_frac": torch.tensor(frac, dtype=torch.float),
        }


def collate_fn_contrastive(batch: list) -> dict:
    """
    Collate contrastive pairs: pads prefix, positive, and negative separately.

    Returns both positive and negative targets so the model can compute
    contrastive loss comparing their waveforms.
    """
    from .tokenizer import PAD_ID

    max_prefix = max(b["prefix_ids"].shape[0] for b in batch)
    max_all_pos = max(b["all_ids_positive"].shape[0] for b in batch)
    max_all_neg = max(b["all_ids_negative"].shape[0] for b in batch)

    prefix_ids_padded = torch.full((len(batch), max_prefix), PAD_ID, dtype=torch.long)
    prefix_pos_padded = torch.zeros(len(batch), max_prefix, dtype=torch.long)
    prefix_mask = torch.zeros(len(batch), max_prefix, dtype=torch.bool)

    all_ids_pos_padded = torch.full((len(batch), max_all_pos), PAD_ID, dtype=torch.long)
    all_pos_pos_padded = torch.zeros(len(batch), max_all_pos, dtype=torch.long)
    all_mask_pos = torch.zeros(len(batch), max_all_pos, dtype=torch.bool)

    all_ids_neg_padded = torch.full((len(batch), max_all_neg), PAD_ID, dtype=torch.long)
    all_pos_neg_padded = torch.zeros(len(batch), max_all_neg, dtype=torch.long)
    all_mask_neg = torch.zeros(len(batch), max_all_neg, dtype=torch.bool)

    seq_lens = torch.stack([b["seq_len"] for b in batch])
    prefix_fracs = torch.stack([b["prefix_frac"] for b in batch])

    for i, b in enumerate(batch):
        p = b["prefix_ids"].shape[0]
        t_pos = b["all_ids_positive"].shape[0]
        t_neg = b["all_ids_negative"].shape[0]

        # Prefix
        prefix_ids_padded[i, :p] = b["prefix_ids"]
        prefix_pos_padded[i, :p] = b["prefix_positions"]
        prefix_mask[i, :p] = True

        # Positive target
        all_ids_pos_padded[i, :t_pos] = b["all_ids_positive"]
        all_pos_pos_padded[i, :t_pos] = b["all_positions_positive"]
        all_mask_pos[i, :t_pos] = True

        # Negative target
        all_ids_neg_padded[i, :t_neg] = b["all_ids_negative"]
        all_pos_neg_padded[i, :t_neg] = b["all_positions_negative"]
        all_mask_neg[i, :t_neg] = True

    return {
        "prefix_tokens": prefix_ids_padded,
        "prefix_positions": prefix_pos_padded,
        "prefix_mask": prefix_mask,
        "target_tokens_positive": all_ids_pos_padded,
        "target_positions_positive": all_pos_pos_padded,
        "target_mask_positive": all_mask_pos,
        "target_tokens_negative": all_ids_neg_padded,
        "target_positions_negative": all_pos_neg_padded,
        "target_mask_negative": all_mask_neg,
        "seq_len": seq_lens,
        "prefix_frac": prefix_fracs,
    }
