# QuINN Phase 2 — Hybrid Architecture & Contrastive Training

## Phase 2 Goals

1. **Validate waveform quality** via contrastive learning: distinguish correct structure from plausible alternatives
2. **Build hybrid QuINN/Transformer**: use QuINN waveform as structural guidance for a language model
3. **End-to-end integration**: train both components jointly

---

## Architecture Overview

### QuINN → Structural Waveform
```
Input: prefix tokens → QuINN → [waveform, length_prediction]
  - waveform: (B, L, embed_dim) complex — predicted structural shape
  - length: scalar — predicted sequence length
```

### Transformer Decoder (new)
```
Input: [QuINN waveform, language model embeddings] → cross-attention, self-attention
  - Uses QuINN waveform as structural guidance
  - Predicts actual tokens (semantics, names, etc.)
  - Cross-attention: "attend to the structural patterns QuINN identified"
```

### Contrastive Loss (new)
```
For each sequence:
  - Positive: (waveform_correct, tokens_correct, pred_len_correct)
  - Negative: (waveform_correct, tokens_wrong, pred_len_wrong) — same structure, wrong tokens
  - Loss: NT-Xent or triplet — waveform should be close to correct, far from wrong
```

---

## Phase 2A: Contrastive Training Infrastructure

### 1. Hard Negative Generation
Strategy: Generate structurally-plausible but semantically-wrong sequences
- **Method 1 (heuristic)**: Swap tokens within the same category (NAME↔NAME, OP_ARITH↔OP_ARITH)
- **Method 2 (LM-based)**: Use a pretrained code LM to sample plausible alternatives

### 2. Dataset Extension
```python
class PythonCodeDatasetWithNegatives(PythonCodeDataset):
    def __getitem__(self, idx):
        original_ids = self.samples[idx]
        negative_ids = hard_negative_generator(original_ids)
        
        return {
            "positive_ids": original_ids,
            "negative_ids": negative_ids,
            "prefix_frac": ...,
            ...
        }
```

### 3. Contrastive Loss
```python
class WaveformContrastiveLoss(nn.Module):
    """NT-Xent on waveform embeddings"""
    def forward(self, wf_correct, wf_wrong, tau=0.07):
        # Pull together, push apart in embedding space
        # Use final summary vector as anchor
```

---

## Phase 2B: Hybrid QuINN/Transformer

### 1. Transformer Decoder
```python
class StructuralTransformerDecoder(nn.Module):
    """
    Takes QuINN waveform as cross-attention key/value.
    Predicts tokens autoregressively while respecting structural constraints.
    """
    def __init__(self, vocab_size, embed_dim, n_layers, ...):
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # Cross-attention: attend to QuINN waveform
        self.cross_attn_layers = nn.ModuleList([...])
        # Self-attention: token-to-token dependencies
        self.self_attn_layers = nn.ModuleList([...])
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, token_ids, positions, quinn_waveform):
        # token_embed + positional → query for cross-attention
        # Cross-attend to quinn_waveform (guidance)
        # Self-attend for local context
        # → logits
```

### 2. Hybrid Training Loop
```
For each batch:
  1. Encode prefix through QuINN
  2. Get waveform summary + length prediction
  3. Decode target sequence through StructuralTransformerDecoder
  4. Loss = waveform_loss + language_modeling_loss + contrastive_loss
```

### 3. Evaluation Metrics
- **Waveform quality**: structural accuracy via contrastive margin
- **Token prediction**: exact match, perplexity on test set
- **Hybrid performance**: can transformer respect structural constraints better with guidance?

---

## Implementation Roadmap

### Phase 2A (Contrastive) — First:
1. Implement `HardNegativeGenerator` with heuristic swaps
2. Extend dataset to yield (positive, negative) pairs
3. Implement `WaveformContrastiveLoss` (NT-Xent on summaries)
4. Train QuINN with both waveform + contrastive loss (keep length as auxiliary)
5. Validate: does waveform embedding space respect structural similarity?

### Phase 2B (Hybrid) — After 2A validates:
1. Implement `StructuralTransformerDecoder`
2. Add cross-attention from tokens to QuINN waveform
3. Train end-to-end: QuINN + Transformer on same prefix→target task
4. Evaluate: does transformer improve with structural guidance?
5. Ablate: remove QuINN guidance to quantify benefit

---

## Architectural Decisions

### Why cross-attention (not just embeddings)?
The transformer must actively decide *which structural patterns* are relevant to the current prediction. Cross-attention lets it ignore irrelevant parts of the waveform and focus on local structure.

### Why keep waveform + length during Phase 2A?
Length is a validated signal. Contrastive loss alone might not constrain the embedding space enough. Multi-task learning (waveform + length + contrastive) provides richer training signal.

### Why heuristic hard negatives first (not LM-based)?
- Simpler to implement and debug
- Tests whether QuINN can distinguish swapped tokens
- LM-based negatives require a pretrained model (another dependency)
- Can upgrade to LM-based later once heuristic approach is validated

---

*Last updated: Phase 2 planning (2026-06-13)*
