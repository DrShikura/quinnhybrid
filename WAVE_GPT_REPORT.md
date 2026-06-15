# WaveGPT Training Report — Structural vs BPE Tokenizer

**Date:** 2026-06-15
**Hardware:** NVIDIA RTX 3090 (GPU 1) — isolated via `CUDA_VISIBLE_DEVICES=1`
**Model:** `WaveGPT` (`spectral/model/wave_gpt.py`)
**Status:** All runs completed successfully (exit 0)

---

## 1. Executive Summary

We trained WaveGPT on a Python source corpus under two tokenizers (81-token **structural** and 5000-token **BPE**) at two run lengths (50 and 1000 epochs), to see how the architecture behaves and how tokenization affects results.

Headline findings:

1. **50 epochs is far too short for both tokenizers.** Extending to 1000 epochs improved structural perplexity 2.81 → **1.88** and BPE perplexity 53.3 → **11.5**.
2. **Structural training saturates cleanly with no overfitting.** The 81-token abstraction is a strong implicit regularizer; train and val stay locked together even at 1000 epochs. Sweet spot ≈ 200–300 epochs.
3. **BPE training overfits this small corpus.** Validation loss bottoms at ~epoch 260, then rises while training loss keeps falling (final train/val gap of −1.13 nats). Best checkpoint ≈ epoch 260.
4. **BPE extracts more real information despite a worse raw perplexity.** Normalized by entropy removed from the uniform baseline, BPE removes **6.08 nats/token** vs structural's **3.76** — it models actual identifier subwords, a harder and richer target. Raw perplexity is **not** comparable across different vocab sizes.

---

## 2. Environment

| Item | Value |
|---|---|
| GPU used | RTX 3090 (GPU index 1), compute capability 8.6, 24 GB |
| Other GPU (untouched) | RTX 5090 (GPU index 0) — running a separate `iqn_btc_clean` RL job the entire time |
| GPU pinning | `CUDA_VISIBLE_DEVICES=1` (the training script's `--device` flag only accepts `auto/cpu/cuda/mps`, not a device index) |
| Python env | project virtual environment (Python 3.14) |
| PyTorch | 2.12.0+cu130 |
| Compilation | `--no-compile` (avoids torch.compile/cudagraph issues with the data-dependent phase encoding; no effect on results) |
| Thread budget | `OMP/MKL/OPENBLAS_NUM_THREADS=2` for CPU-side work, to stay polite to the RL job and bound RAM |

GPU isolation held for the entire session — the RTX 5090 / RL job was never touched.

---

## 3. Model Architecture

`WaveGPT` is a small phase-aware causal transformer:

- **Positional encoding:** ALiBi (Attention with Linear Biases), per-head slopes — no learned position embeddings.
- **Phase encoding:** each `INDENT` token shifts a running phase by +π/4, each `DEDENT` by −π/4, injecting nesting depth as a continuous signal without a stack.
- **Normalization:** RMSNorm.
- **Embeddings:** tied (input embedding shared with the output LM head).
- **No biases** in projection layers.
- **Config:** `d_model=48`, `n_layers=4`, `n_heads=4`, `ff_mult=3` (FFN hidden = 144), `max_seq_len=256`.

| Tokenizer | Vocab | Embedding params | Transformer params | Total params |
|---|---:|---:|---:|---:|
| Structural | 81 | 3,888 | 92,592 | **96,672** |
| BPE | 5,000 | 240,000 | 92,592 | **332,784** |

The only architectural difference between the two is the vocabulary/embedding size; the transformer body is identical.

---

## 4. Data

- **Corpus source:** `data/corpus.txt` — 3,247 file paths (libreoffice + Python stdlib), of which **967 exist** on this machine.
- **Filtering:** files tokenized, kept if ≥ 20 tokens (`min_seq_len`), truncated to 256 tokens (`max_seq_len`).
- **Split:** reproducible shuffle (seed 42), `val_frac=0.10`, `test_frac=0.05` → **821 train / 96 val** samples.
- **Tokenizers:**
  - *Structural* — 81 fixed tokens mapping Python syntax to roles (`NAME_FUNC`, `OP_ARITH`, `INDENT`, `STRING`, …). No training required.
  - *BPE* — 5,000-token subword vocab (4,735 merges) trained on the corpus; preserves identifier information as subword sequences.

---

## 5. BPE Cache Pre-Computation (performance work)

The pure-Python BPE encoder (`data/bpe_tokenizer.py::_encode_subword`) loops over all ~4,735 merges for **every** identifier/number/comment token, rescanning the token list each time — O(merges × length) per token. Benchmarked at ~1.1 files/s single-threaded → ~14 min just to tokenize the corpus before training could start.

**Fix:** a parallel, RAM/thread-bounded pre-compute script ([`precompute_bpe_cache.py`](precompute_bpe_cache.py)) that:
- replicates `SpectralDataset`'s exact path resolution, length filtering, cache key, and spectral-stats computation, so the produced cache is a guaranteed hit for `train.py --tokenizer bpe --max-seq-len 256`;
- tokenizes across a 2-process pool (preserving corpus order via `imap`);
- computes the vocab×vocab spectral stats (`eigh` on a 5000×5000 matrix) once in the parent.

**Result:** 965 samples tokenized in **257 s** with 2 workers (0 errors), `eigh` in 6 s, **100.6 MB** cache written (key `8568a2837b17afdd`). Peak RSS ~1.1 GB — well within the ~20 GB budget. Both BPE training runs then loaded this cache instantly.

---

## 6. Methodology

Common training configuration: AdamW (weight decay 0.01), OneCycleLR (peak LR 3e-4, cosine anneal), batch size 32, gradient clip 1.0, language-modeling mode. Each run saves `spectral_best.pt` (best val), `spectral_final.pt`, an every-10-epoch snapshot, and `training_history.json`.

Commands (all prefixed with `CUDA_VISIBLE_DEVICES=1` and run with the venv Python):

```bash
# BPE vocab (one-time)
python train_bpe_vocab.py --vocab-size 5000 --corpus data/corpus.txt --save-path data/bpe_vocab

# BPE cache pre-compute (one-time, 2 workers)
OMP_NUM_THREADS=2 python precompute_bpe_cache.py --workers 2

# Structural
python spectral/experiments/train.py --model wave-gpt --tokenizer structural \
  --epochs   50 --no-compile --checkpoint-dir checkpoints/wave_gpt_structural
python spectral/experiments/train.py --model wave-gpt --tokenizer structural \
  --epochs 1000 --no-compile --checkpoint-dir checkpoints/wave_gpt_structural_1k

# BPE
python spectral/experiments/train.py --model wave-gpt --tokenizer bpe --bpe-vocab data/bpe_vocab \
  --epochs   50 --no-compile --checkpoint-dir checkpoints/wave_gpt_bpe
python spectral/experiments/train.py --model wave-gpt --tokenizer bpe --bpe-vocab data/bpe_vocab \
  --epochs 1000 --no-compile --checkpoint-dir checkpoints/wave_gpt_bpe_1k
```

---

## 7. Results

### 7.1 Summary

| Run | Params | Epochs | Wall time | Best val loss | Best val ppl | Best @ epoch | Final train | Final val | Train−Val gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structural 50ep | 96.7K | 50 | 17 s | 1.0320 | 2.81 | 50 | 1.0881 | 1.0320 | +0.056 |
| **structural 1000ep** | 96.7K | 1000 | 275 s | **0.6338** | **1.88** | 757 | 0.5778 | 0.6345 | −0.057 |
| bpe 50ep | 332.8K | 50 | 35 s | 3.9768 | 53.34 | 50 | 3.9038 | 3.9768 | −0.073 |
| **bpe 1000ep** | 332.8K | 1000 | 377 s | **2.4405** | **11.48** | 260 | 1.4053 | 2.5301 | **−1.125** |

### 7.2 Trajectories

**Structural, 1000 epochs** (saturates, no overfitting):

| Epoch | Train | Val | Val ppl |
|---:|---:|---:|---:|
| 1 | 4.367 | 4.334 | 76.3 |
| 10 | 3.281 | 3.136 | 23.0 |
| 50 | 0.942 | 0.889 | 2.43 |
| 100 | 0.785 | 0.754 | 2.13 |
| 200 | 0.684 | 0.682 | 1.98 |
| 300 | 0.637 | 0.656 | 1.93 |
| 500 | 0.599 | 0.639 | 1.89 |
| 757 (best) | — | 0.634 | 1.88 |
| 1000 | 0.578 | 0.635 | 1.89 |

**BPE, 1000 epochs** (overfits after ~epoch 260):

| Epoch | Train | Val | Val ppl |
|---:|---:|---:|---:|
| 1 | 8.494 | 8.455 | 4697 |
| 10 | 7.399 | 7.283 | 1456 |
| 50 | 3.298 | 3.391 | 29.7 |
| 100 | 2.482 | 2.776 | 16.1 |
| 200 | 1.903 | 2.469 | 11.8 |
| **260 (best)** | — | **2.441** | **11.48** |
| 400 | 1.554 | 2.468 | 11.8 |
| 600 | 1.450 | 2.513 | 12.4 |
| 800 | 1.417 | 2.527 | 12.5 |
| 1000 | 1.405 | 2.530 | 12.55 |

After epoch 260, BPE validation rises monotonically while training loss keeps dropping — textbook overfitting in a small-data, higher-capacity regime.

### 7.3 Cross-Tokenizer Normalization

Raw loss/perplexity is not comparable across tokenizers (81-token vs 5000-token vocabularies have very different per-token baselines: uniform loss = ln(81)=4.39 vs ln(5000)=8.52). A fairer view is **nats of entropy removed from the uniform baseline**:

| Run | Best val loss | Uniform baseline | Nats removed | % of uniform |
|---|---:|---:|---:|---:|
| structural 50ep | 1.032 | 4.39 | 3.36 | 76.5% |
| structural 1000ep | 0.634 | 4.39 | 3.76 | 85.6% |
| bpe 50ep | 3.977 | 8.52 | 4.54 | 53.3% |
| bpe 1000ep | 2.441 | 8.52 | **6.08** | 71.3% |

By this measure BPE captures substantially more absolute information per token (6.08 vs 3.76 nats) — it is modeling real subword identifiers rather than abstract structural roles — even though its raw perplexity looks far worse.

---

## 8. Analysis & Recommendations

- **Both tokenizers were badly undertrained at 50 epochs.** Use longer schedules.
- **Structural:** best operating point ≈ **200–300 epochs**. Beyond ~500 the gains are negligible (val flat at ~1.89 ppl) and there is no overfitting — the 81-token abstraction regularizes effectively. Use `checkpoints/wave_gpt_structural_1k/spectral_best.pt` (epoch 757, ppl 1.88) or simply train ~250 epochs.
- **BPE:** best operating point ≈ **epoch 260** (ppl 11.48). Training longer hurts validation. The best checkpoint is preserved as `checkpoints/wave_gpt_bpe_1k/spectral_best.pt` regardless of the overfit tail.
- **To improve BPE further:** the limiting factor is data/regularization, not epochs. Options: (a) collect more training files (the corpus only resolves to ~965 files); (b) increase dropout / weight decay; (c) reduce embedding capacity or tie more aggressively. The larger 5000-token embedding (240K params) over 821 files is the main overfitting driver.

---

## 9. Artifacts

| Path | Contents |
|---|---|
| `checkpoints/wave_gpt_structural{,_1k}/` | structural 50ep / 1000ep |
| `checkpoints/wave_gpt_bpe{,_1k}/` | BPE 50ep / 1000ep |
| `checkpoints/.dataset_cache/*.pt` | tokenized dataset caches (structural + BPE) |
| `data/bpe_vocab/` | BPE vocab + merges (5000 tokens) |
| `logs/*.log` | per-run training logs |
| `precompute_bpe_cache.py` | parallel BPE cache builder |
| `eval_structure.py` | structural-syntax evaluation harness |
| `webui.py` | mobile inference UI (Flask) |

**Note:** each run directory was pruned to just `spectral_best.pt` + `training_history.json`; the every-10-epoch snapshots (~730 MB across the two 1000-epoch runs) were removed. All `*.pt`, `*.log`, `checkpoints/`, and `data/bpe_vocab/` are git-ignored (generated artifacts).

---

## 10. Reproduction

1. Ensure `data/corpus.txt` exists and the venv has torch.
2. Build the BPE vocab and cache (Section 6).
3. Run any of the four training commands (Section 6) with `CUDA_VISIBLE_DEVICES=1`.
4. Inspect `checkpoints/<run>/training_history.json` for per-epoch train/val losses.
