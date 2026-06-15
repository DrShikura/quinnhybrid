"""
Dataset for SpectralLM — simple next-token prediction setup.

Each sample: full tokenized file, returned as (token_ids, position_ids).

Corpus-level statistics computed once from all tokenized files (before split):
  laplacian_eigvecs  — (vocab_size, vocab_size) eigenvectors of the normalized
                       graph Laplacian of the symmetrized bigram matrix; used to
                       initialize token amplitude profiles in SpectralEmbedding.
  frozen_entropy     — (vocab_size,) Shannon entropy per token's outgoing bigram
                       distribution, normalized to [0,1]; used by the waveform
                       gradient consistency loss (never updated during training).
"""

import hashlib
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.tokenizer import PythonStructuralTokenizer


def _corpus_cache_key(paths: List[Path], max_seq_len: int, min_seq_len: int) -> str:
    h = hashlib.md5()
    for p in sorted(paths):
        h.update(str(p).encode())
    h.update(f"{max_seq_len}:{min_seq_len}".encode())
    return h.hexdigest()[:16]


def compute_spectral_stats(
    samples: List[List[int]],
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    From tokenized corpus sequences, compute:
      1. Laplacian eigenvectors — for Laplacian amplitude initialization.
      2. Frozen entropy          — for waveform gradient consistency loss.

    Bigram matrix pipeline:
      raw counts C[i,j] → smooth (add 1/V) → symmetrize → normalize → Laplacian

    Returns:
      eigvecs         (vocab_size, vocab_size)  columns = eigenvectors (ascending λ)
      frozen_entropy  (vocab_size,)  H[i]/log(V) in [0,1]; higher = more variable
    """
    V = vocab_size

    # Raw bigram counts
    C = torch.zeros(V, V)
    for seq in samples:
        for i in range(len(seq) - 1):
            C[seq[i], seq[i + 1]] += 1.0

    # Laplacoscopic smoothing + symmetrize
    C = C + 1.0 / V                 # add-1/V smoothing (no isolated nodes)
    C_sym = (C + C.T) * 0.5         # make undirected

    # ── Normalized graph Laplacian: L = I − D^{-½} C_sym D^{-½} ──────────
    D = C_sym.sum(dim=1)            # (V,) degree
    D_inv_sqrt = 1.0 / (D.sqrt() + 1e-8)
    # L[i,j] = δ[i,j] − C_sym[i,j] / sqrt(D[i] * D[j])
    L = torch.eye(V) - D_inv_sqrt.unsqueeze(1) * C_sym * D_inv_sqrt.unsqueeze(0)

    # All eigenvectors, ascending by eigenvalue (smallest = smoothest graph signal)
    _evals, eigvecs = torch.linalg.eigh(L)   # (V,) and (V, V)

    # ── Per-token bigram entropy from C_sym row-normalized ─────────────────
    P = C_sym / C_sym.sum(dim=1, keepdim=True)          # (V, V) stochastic
    H = -(P * torch.log(P + 1e-9)).sum(dim=1)           # (V,) Shannon entropy
    frozen_entropy = (H / math.log(V)).clamp(0.0, 1.0)  # (V,) in [0, 1]

    return eigvecs, frozen_entropy


class SpectralDataset(Dataset):
    """
    Loads Python source files and returns full token sequences.

    Attributes set after __init__:
      laplacian_eigvecs  (vocab_size, vocab_size) — pass to SpectralEmbedding
      frozen_entropy     (vocab_size,)             — pass to SpectralTrainer

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

        # Disk cache for tokenized samples + spectral stats
        cache_dir  = Path("checkpoints") / ".dataset_cache"
        cache_key  = _corpus_cache_key(paths, max_seq_len, min_seq_len)
        cache_file = cache_dir / f"{cache_key}.pt"

        if cache_file.exists():
            print(f"  Loading dataset cache: {cache_key}")
            cached = torch.load(cache_file, weights_only=False)
            all_samples             = cached["samples"]
            self.laplacian_eigvecs  = cached["laplacian_eigvecs"]
            self.frozen_entropy     = cached["frozen_entropy"]
        else:
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

            # Corpus-level spectral statistics — computed from ALL samples before split
            vocab_size = len(self.tokenizer.vocab)
            self.laplacian_eigvecs, self.frozen_entropy = compute_spectral_stats(
                all_samples, vocab_size
            )

            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "samples":           all_samples,
                "laplacian_eigvecs": self.laplacian_eigvecs,
                "frozen_entropy":    self.frozen_entropy,
            }, cache_file)
            print(f"  Saved dataset cache: {cache_key}")

        # Reproducible split
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n      = len(all_samples)
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
