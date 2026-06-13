# QuINN Phase 1 — Experimental Findings

*Ongoing log of results, observations, and architectural decisions.*

---

## Architecture Summary (Phase 1)

**QuINN** is a structural waveform completion model (not a language model). Given a
prefix of structural tokens, it predicts: (a) the complete structural waveform for
the full sequence, and (b) the total sequence length. Implemented as five stages:

1. **ComplexTokenEmbedding** (`quinn/encoding.py`)
   - Each structural token type has learned complex amplitude (real + imaginary embedding)
   - Each token has learned per-channel frequencies modulating its phase with position
   - Encoding: `(A_real + i·A_imag) × exp(i·f·p/max_len·2π)`

2. **InterferenceManifold** (`quinn/manifold.py`)
   - Stack of `ComplexLinear → ModReLU → ComplexLayerNorm` layers with residual connections
   - `ModReLU`: gates based on amplitude only, preserves phase — essential for interference
   - `ComplexLinear`: native `torch.cfloat` matmul (2.9× faster than 4 real matmuls)

3. **PrefixAggregator**
   - Mask-aware complex mean-pooling → fixed-size summary vector `(B, manifold_dim) complex`

4. **WaveformDecoder** (`quinn/model.py`)
   - `(summary, sinusoidal position features) → predicted waveform` at each target position
   - Stop-gradient on true waveform (prevents encoder shortcut, 2.6× backward speedup)

5. **LengthHead** (`quinn/model.py`)
   - `[|summary|, log(prefix_len)] → log(seq_len)` via real MLP
   - `log(prefix_len)` is critical for monotonicity (see Run 4 findings)

**Vocabulary**: 74 structural Python tokens (keywords, brackets, operators, NAME/NUMBER/STRING)

---

## Phase 1 Training Configuration (Runs 4–5)

| Parameter | Value |
|-----------|-------|
| embed_dim | 32 |
| manifold_dim | 64 |
| n_encoder_layers | 2 |
| n_decoder_layers | 1 |
| max_seq_len | 650 |
| batch_size | 32 |
| learning_rate | 3e-4 |
| Data | Python stdlib, 1077 files, 50–600 tokens |
| Prefix fraction range | 10%–90% |

**Corpus strategy**: Filter (not truncate) files at `max_seq_len=600`. Truncation clusters
files at exactly 600 tokens, making the mean-prediction baseline artificially high
(~81% ±15% accuracy). Filtering gives a baseline of ~13%.

---

## Phase 1 Success Criterion

> QuINN's token count estimates at 50% prefix converge to within ±15% of true length
> significantly more reliably than the baseline (mean training length prediction),
> and improve monotonically as prefix length increases.

**Baseline**: Predict the mean training sequence length for every file (~305 tokens).
Test set baseline accuracy: **12.7%** (at ±15% threshold).

---

## Results

### Run 4 — 30 epochs, scale-invariant cosine loss, stop-gradient

- Dataset: 1077 files (50–600 tokens), 55 test files, baseline = 12.7%
- Loss: pure cosine phase-alignment + log-scale SmoothL1 on length
- Model: embed_dim=32, manifold_dim=64, 2 encoder / 1 decoder layers

**Diagnostic evaluation at fixed prefix fractions** (epoch 30 checkpoint):

| Prefix | QuINN ±15% | Baseline ±15% | Beats | Median rel err |
|--------|-----------|--------------|-------|---------------|
| 10% | 0.200 | 0.127 | ✓ | 0.427 |
| 20% | 0.291 | 0.127 | ✓ | 0.284 |
| 30% | 0.364 | 0.127 | ✓ | 0.226 |
| 40% | 0.400 | 0.127 | ✓ | 0.221 |
| 50% | **0.509** | 0.127 | ✓ | 0.146 |
| 60% | 0.564 | 0.127 | ✓ | 0.123 |
| 70% | 0.327 | 0.127 | ✓ | 0.172 |
| 80% | 0.236 | 0.127 | ✓ | 0.300 |
| 90% | 0.182 | 0.127 | ✓ | 0.408 |

**Phase 1 verdict**: At 50% prefix: PASS ✓ (0.509 vs 0.127, 4× baseline)  
Monotonically improving: **FAIL ✗** (peaks at 60%, drops sharply at 70–90%)

**Root cause of non-monotonicity**: The length head received only `|summary|` (content
features), with no information about how many tokens were seen. At 70–90% prefix, the
mean-pooled summary has similar amplitude to 50–60%, so the model cannot distinguish
"I've seen 100 tokens" from "I've seen 400 tokens." Accuracy peaks where prefix content
is most discriminative for mean-pooling, not where evidence is strongest.

---

### Run 5 — 50 epochs, prefix_len-aware length head (in progress)

**Key change**: Added `log(prefix_len)` as a feature to the length head:
```python
# Before (Run 4):
pred_log_len = self.length_head(summary.abs())  # content only

# After (Run 5):
log_prefix_len = torch.log(prefix_mask.sum(dim=1).float().clamp(min=1))
pred_log_len = self.length_head(
    torch.cat([summary.abs(), log_prefix_len.unsqueeze(-1)], dim=-1)
)
```

With prefix length as a feature, the model can learn:
`total_len ≈ f(content_features, prefix_len)`, where knowing prefix_len directly
allows calibrating uncertainty — longer prefix → tighter estimate → monotone improvement.

**Status**: Training in progress.

---

## Key Architectural Observations

### Stop-gradient on true waveform
Computing the target waveform under `torch.no_grad()` (analogous to a BYOL target
network) prevents a degenerate shortcut where the encoder learns to directly output
targets rather than building generalizable structure representations. It also reduces
backward pass cost by ~2.6× since gradients don't flow through target computation.

### Why cosine phase-alignment loss (not L2)
Early experiments used L2 waveform loss. This collapsed because:
1. Encoder embeddings initialise with amplitude ~0.02; decoder outputs ~1.0
2. L2 loss is dominated by amplitude mismatch: `(0.02 - 1.0)^2 ≈ 0.96` per element
3. Both branches collapse toward zero to minimise the squared difference

Cosine similarity is scale-invariant: it measures structural alignment regardless of
amplitude. This eliminates the collapse mode and keeps the loss bounded in `[0, 2]`.

### Filter-not-truncate for dataset
Truncating sequences to max_seq_len clusters many files at exactly that boundary.
If max_seq_len ≈ mean length, the constant-baseline predictor gets ~14% relative
error on these clustered files — inflating baseline accuracy to ~81% and making
Phase 1 success trivial to fake. Filtering (skipping overlength files) preserves
the true distribution and gives a baseline of ~13%.

### Why log-scale length prediction
`SmoothL1(log(pred), log(true))` is scale-invariant: predicting 200 tokens for a
100-token file has the same loss as predicting 2000 for a 1000-token file. This
matches the ±15% success criterion which is also relative, not absolute.

### ComplexLinear performance
Native `torch.cfloat` matmul (`x @ W.T`, `W ∈ ℂ^{out×in}`) is 2.9× faster than
the naive 4-real-matmul decomposition. PyTorch's cuBLAS/LAPACK handles complex
GEMM efficiently when both operands are contiguous `cfloat` tensors.

---

## Open Questions (from Roadmap)

1. **Autoregressivity of QuINN state**: Can the interference state be accumulated
   step-by-step without recomputing from scratch? Currently we recompute on every call.
   Acceptable for Phase 1 but will need to change for transformer integration.

2. **Single-manifold vs multi-scale manifolds**: The current implementation uses one
   flat manifold for all structural scales. Whether one manifold can stably learn
   both document-level and token-level structure is an empirical question Phase 1
   helps answer.

3. **Hard negative generation (Phase 2)**: QuINN currently trains only on real sequences.
   Phase 2 requires a pretrained LM to generate structurally-wrong-but-statistically-
   plausible substitutions. Deferred until Phase 1 completes.

---

*Last updated: Run 5 launched with prefix_len-aware length head (2026-06-13)*
