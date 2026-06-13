# Phase 2A Results: Contrastive Learning Analysis

**Date**: 2026-06-13  
**Duration**: 50 epochs on 1077 files, batch size 32  
**Outcome**: Contrastive margin developed, but modest

---

## Results Summary

| Metric | Epoch 1 | Epoch 50 | Change |
|--------|---------|----------|--------|
| Contrastive Loss | 0.6914 | 0.5954 | -0.0960 (13.8% drop) |
| Waveform Loss | 0.9461 | 0.6894 | -0.2567 (27.1% improvement) |
| Length Loss | 4.9018 | 0.0050 | -4.897 (perfect) |
| Total Val Loss | 6.5393 | 1.2898 | -5.249 |

---

## Contrastive Loss Trajectory

```
Epoch 1-10:  0.6914 → 0.6922  (essentially flat)
Epoch 10-20: 0.6922 → 0.6785  (slow decline)
Epoch 20-30: 0.6785 → 0.6582  (modest decline)
Epoch 30-40: 0.6582 → 0.6205  (steeper, peaks at 0.6762)
Epoch 40-50: 0.6205 → 0.5954  (slight improvement with noise)
```

**Baseline**: Random 2-class NT-Xent = ln(2) ≈ 0.6931
**Range**: [0, 1.386], where 0 = perfect separation

**Interpretation**: Loss decreased from random (0.693) → 0.595, but still far from perfect (0). This indicates:
- Positive and negative waveforms separated somewhat
- Contrastive signal is weak relative to baseline
- Structural swaps (within categories) don't create dramatically different waveforms

---

## Analysis: Why Modest Improvement?

### Hypothesis 1: Hard Negatives Aren't Hard Enough
Current strategy: swap tokens **within** structural categories
- Example: keyword → different keyword (def ↔ return)
- Example: NAME ↔ NAME, LPAREN ↔ RPAREN

These preserve a lot of structure. A true hard negative would swap **across** categories:
- NAME ↔ KEYWORD (identifier becomes keyword)
- LPAREN ↔ LBRACKET (parens become brackets)

**Cross-category swaps** are structurally wrong and should create more distinct waveforms.

### Hypothesis 2: Multi-Task Learning Conflicts
The three loss terms have different gradients:
- **Waveform loss**: wants accurate reconstruction → sharp embeddings
- **Length loss**: wants accurate prediction → particular scaling
- **Contrastive loss**: wants separation → different objective

These may not align optimally. Pure contrastive training (without waveform/length) might show larger margins.

### Hypothesis 3: Phase 1 Learned Robustness to Swaps
QuINN was trained on random prefixes that naturally included all token categories. The model learned to predict despite seeing arbitrary variations. This robustness might make it harder to find structurally-meaningfully-different negatives through simple swaps.

---

## What the Results Mean

### ✓ Positive
- Contrastive learning worked (loss decreased from random)
- Length prediction fully retained from Phase 1 (loss → 0.005)
- Waveform reconstruction improved (0.946 → 0.689)
- No collapse or training instability

### ⚠ Concerning
- Modest contrastive margin (0.693 → 0.595 is ~14% drop vs possible 69% drop)
- Plateau after epoch 30 suggests saturation
- Gradient conflicts may be limiting margin growth

---

## Recommendations for Phase 2B

### Option A: Continue with Current Setup
- Hard negatives are working (loss decreased)
- Proceed to Phase 2B and test if Transformer benefits from waveform guidance
- The contrastive margin might be sufficient for structural differentiation

**Pros**: Fast path to integration testing  
**Cons**: Don't know if modest margin is enough

### Option B: Improve Hard Negative Generation
- Generate cross-category swaps (NAME ↔ KEYWORD, bracket ↔ delimiter)
- Higher structural "wrongness" → larger margin expected
- Retrain Phase 2A with better negatives

**Pros**: Validation that waveform truly captures structure  
**Cons**: Additional ~4 hour training

### Option C: Isolated Contrastive Training
- Train *only* on contrastive loss (remove waveform/length)
- Measure maximum achievable margin
- Inform whether current setup is optimal

**Pros**: Diagnostic (what's the ceiling?)  
**Cons**: Doesn't solve the practical problem

---

## Next Steps

**Recommended**: Option A → Phase 2B
- The contrastive margin exists (0.695 → 0.595)
- Phase 1 knowledge is fully retained
- Phase 2B will test if this guidance helps Transformer
- If Transformer shows no improvement, debug via Option B

**If Phase 2B shows no benefit**:
- Consider Option B (cross-category negatives)
- Or revisit: is structural waveform the right representation?

---

## Technical Notes

**Hard Negatives Used**:
- Strategy: swap tokens within same structural category
- Count: 1-2 swaps per sequence
- Categories: KEYWORD, OPERATOR, DELIMITER, NAME, NUMBER, STRING, INDENT, etc.

**Contrastive Loss**:
- Type: NT-Xent (InfoNCE)
- Temperature: 0.07
- Anchor: prefix summary
- Positive: correct target summary
- Negative: wrong target summary

**Multi-Task Weights**:
- Waveform loss: 1.0
- Length loss: 1.0 (alpha=1.0 in config)
- Contrastive loss: 1.0 (configurable)

---

*Phase 2A validates that contrastive training works, but leaves open the question of whether the learned distinction is semantically meaningful (testable in Phase 2B).*
