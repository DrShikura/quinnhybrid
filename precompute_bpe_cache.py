#!/usr/bin/env python3
"""
Pre-compute the SpectralDataset disk cache for the BPE tokenizer, in parallel.

The BPE encoder is pure-Python and slow (~1 file/s single-threaded), so a normal
`train.py --tokenizer bpe` run spends ~14 min tokenizing before training starts.
This script tokenizes the corpus across a small process pool and writes the *exact
same* cache file (same key + contents) that spectral/training/dataset.py expects,
so the subsequent training run loads it instantly.

Correctness: it replicates SpectralDataset's path resolution, min/max length
filtering, cache key, and spectral-stats computation exactly, so the produced
cache is a guaranteed hit for `train.py --tokenizer bpe --max-seq-len 256`.

RAM/CPU-bounded: caps workers (default 2) and pins BLAS/OMP threads; only the
parent builds the vocab x vocab spectral-stats matrices.
"""
import argparse
import sys
import time
from pathlib import Path
import multiprocessing as mp

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from data.bpe_tokenizer import BytePairTokenizer
from spectral.training.dataset import _corpus_cache_key, compute_spectral_stats
import torch

# Per-worker tokenizer (built once via Pool initializer; never pickled)
_TOK = None


def _init_worker(bpe_vocab: str):
    global _TOK
    _TOK = BytePairTokenizer(bpe_vocab)


def _encode_path(path_str: str):
    """Mirror SpectralDataset's per-file step. Return (ok, ids); ok=False -> skip."""
    try:
        source = Path(path_str).read_text(encoding="utf-8", errors="replace")
        ids = _TOK.encode(source, add_special=True)
        return True, ids
    except Exception:
        return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--bpe-vocab", default="data/bpe_vocab")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--min-seq-len", type=int, default=20)   # SpectralDataset default
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunksize", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.workers)  # keep the eigh burst polite

    # Resolve paths EXACTLY like SpectralDataset (corpus order, existing only)
    corpus = Path(args.corpus)
    lines = corpus.read_text(encoding="utf-8", errors="replace").splitlines()
    paths = [Path(ln.strip()) for ln in lines if ln.strip()]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No existing files in corpus.")
        return 1

    tok = BytePairTokenizer(args.bpe_vocab)
    vocab_size = len(tok.vocab)

    cache_dir = Path("checkpoints") / ".dataset_cache"
    cache_key = _corpus_cache_key(paths, args.max_seq_len, args.min_seq_len, vocab_size)
    cache_file = cache_dir / f"{cache_key}.pt"
    print(f"corpus files (existing): {len(paths)}")
    print(f"vocab_size={vocab_size}  max_seq_len={args.max_seq_len}  "
          f"min_seq_len={args.min_seq_len}  workers={args.workers}")
    print(f"cache key : {cache_key}")
    print(f"cache file: {cache_file}", flush=True)
    if cache_file.exists():
        print("Cache already exists - nothing to do.")
        return 0

    # Parallel tokenize, preserving corpus order (imap yields in input order)
    t0 = time.time()
    all_samples = []
    kept = skipped_short = skipped_err = 0
    path_strs = [str(p) for p in paths]
    n = len(path_strs)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.bpe_vocab,)) as pool:
        for i, (ok, ids) in enumerate(
                pool.imap(_encode_path, path_strs, args.chunksize), 1):
            if not ok:
                skipped_err += 1
            elif len(ids) >= args.min_seq_len:
                all_samples.append(ids[:args.max_seq_len])
                kept += 1
            else:
                skipped_short += 1
            if i % 100 == 0 or i == n:
                el = time.time() - t0
                print(f"  [{i}/{n}] kept={kept} short={skipped_short} "
                      f"err={skipped_err}  {i / el:.2f} files/s  "
                      f"elapsed={el:.0f}s", flush=True)

    print(f"Tokenized: {kept} samples kept in {time.time() - t0:.0f}s", flush=True)

    print(f"Computing spectral stats (eigh on {vocab_size}x{vocab_size})...",
          flush=True)
    ts = time.time()
    eigvecs, frozen_entropy = compute_spectral_stats(all_samples, vocab_size)
    print(f"  stats done in {time.time() - ts:.0f}s  eigvecs={tuple(eigvecs.shape)}",
          flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": all_samples,
                "laplacian_eigvecs": eigvecs,
                "frozen_entropy": frozen_entropy}, cache_file)
    sz = cache_file.stat().st_size / 1e6
    print(f"Saved cache: {cache_file}  ({sz:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
