# QuINN Phase 1 — Experimental Findings

*Ongoing log of results, observations, and architectural decisions.*

---

## Architecture Summary (Phase 1)

**QuINN** is implemented as a three-stage pipeline:

1. **ComplexTokenEmbedding** (`quinn/encoding.py`)
   - Each structural token type has a learned complex amplitude (real + imaginary embedding)
   - Each token also has learned per-channel frequencies that modulate its phase with position
   - Encoding: `(A_real + i·A_imag) × exp(i·f·p/max_len·2π)`
   - This gives each token a characteristic sinusoidal "signature" across positions

2. **InterferenceManifold** (`quinn/manifold.py`)
   - Stack of `ComplexLinear → ModReLU → ComplexLayerNorm` layers with residual connections
   - `ModReLU`: acts only on modulus, preserves phase — gates small-amplitude (coincidental) components
   - Complex LayerNorm: normalises jointly over real and imaginary parts

3. **WaveformDecoder + LengthHead** (`quinn/model.py`)
   - Prefix encodings are mask-pooled to a summary vector
   - Decoder: (summary, sinusoidal position features) → predicted waveform at each target position
   - LengthHead: `|summary|` magnitudes → `log(seq_len)` via real MLP

**Vocabulary**: 80 structural Python tokens (keywords, brackets, operators, NAME/NUMBER/STRING categories)

---

## Phase 1 Training Configuration

| Parameter | Value |
|-----------|-------|
| embed_dim | 64 |
| manifold_dim | 128 |
| n_encoder_layers | 4 |
| n_decoder_layers | 2 |
| batch_size | 16 |
| learning_rate | 3e-4 |
| n_epochs | 30 |
| Data | Python stdlib (~400 files) |
| Max seq len | 1024 tokens |
| Prefix fraction range | 10%–90% |

---

## Phase 1 Success Criterion

> QuINN's token count estimates at 50% prefix converge to within ±15% of true length
> significantly more reliably than the baseline (mean training length prediction),
> and improve monotonically as prefix length increases.

**Baseline**: Predict the mean training sequence length for every file.

---

## Results

*(To be filled in after training runs.)*

### Run 1 — Initial Configuration

**Status**: Not yet run

| Prefix | QuINN ±15% | Baseline ±15% | Beats baseline |
|--------|-----------|--------------|---------------|
| 10%    | —         | —            | —             |
| 30%    | —         | —            | —             |
| 50%    | —         | —            | —             |
| 70%    | —         | —            | —             |
| 90%    | —         | —            | —             |

**Phase 1 verdict**: Pending

---

## Key Architectural Observations

### What ModReLU contributes
The phase-preserving gate is essential for the interference property. A standard
ReLU applied to real/imaginary separately would corrupt phase relationships between
tokens — exactly the structural information QuINN is supposed to preserve. `ModReLU`
gates based on amplitude (how "confidently" a component is expressed) while leaving
the directional (structural) information intact.

### Why log-scale length prediction
The loss is computed as `SmoothL1(log(pred), log(true))`. This is scale-invariant:
predicting 200 tokens when the true length is 100 has the same loss as predicting
2000 when the true is 1000. This is appropriate because the "difficulty" of length
estimation scales with relative error, not absolute error.

### The training signal
The waveform completion loss requires QuINN to predict the complex embedding of
every token in the complete sequence from just the prefix. Since the complex embeddings
encode structural role (via token type → frequency characteristics), this forces QuINN
to learn which structural positions are *obligated* by what the prefix has already done.
The loss has no way to reward predicting the specific identifier name — only the
structural category.

---

## Open Questions (from Roadmap)

1. **Autoregressivity of QuINN state**: Can the interference state be accumulated
   step-by-step without recomputing from scratch? Currently we recompute on every call.
   This is acceptable for Phase 1 but will need to change for transformer integration.

2. **Single-manifold vs multi-scale manifolds**: The current implementation uses one
   flat manifold for all structural scales. Low-frequency components encode document
   structure; high-frequency encode token-scale micro-structure. Whether one manifold
   can stably learn both is an open empirical question Phase 1 will help answer.

3. **Hard negative generation (Phase 2)**: QuINN currently trains only on real sequences.
   Phase 2 requires a pretrained standard LM to generate structurally-wrong-but-
   statistically-plausible substitutions. This is deferred but the contrastive loss
   infrastructure should be added after Phase 1 validates waveform completion.

---

*Last updated: Phase 1 implementation — pre-training*
