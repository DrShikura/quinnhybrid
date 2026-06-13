# Phase 2B Diagnostic Report

**Date**: 2026-06-13  
**Status**: Training in progress, initial learnings documented

---

## Training Setup Completed

### Phase 1 Validation ✅
- Run 7: 100% accuracy at 50% prefix, monotone improvement across 10-90%
- Pre-trained checkpoint: `checkpoints/run7/quinn_epoch050.pt`
- Ready for Phase 2B as guidance signal

### Phase 2A Validation ✅
- 50 epochs contrastive training: loss 0.693 → 0.595 (14% improvement)
- Hard negative generation: within-category swaps, 7 tests passing
- Checkpoint: `checkpoints/phase2a_run1/` (optional as alternative guidance)

### Phase 2B Infrastructure ✅
- Transformer architecture: 4 layers, 8 heads, 101.9k parameters
- Cross-attention to Quinn waveform implemented
- Two experiment designs:
  1. WITH Quinn (Phase 1): uses trained structural guidance
  2. Baseline (Random): no pre-trained guidance
- Analysis script ready: `experiments/phase2b_analysis.py`

---

## Current Training Status

### Job 1: Baseline (Random Quinn)
- **ID**: `bgdnm2u3q`
- **Configuration**: Batch size 32, 50 epochs
- **Status**: Running (further along than WITH Quinn)
- **Expected completion**: ~4 hours from start

### Job 2: WITH Quinn (Phase 1 Guidance)
- **ID**: `bhrhmff5x` (restarted after first job killed)
- **Configuration**: Batch size 16 (reduced to avoid memory), 50 epochs
- **Status**: Starting (loading corpus)
- **Expected completion**: ~4.5 hours from restart

---

## Key Questions Being Answered

1. **Does Quinn's structural waveform help token prediction?**
   - Hypothesis: Yes, by constraining the Transformer to respect structure
   - Test: Compare validation LM loss (lower is better)

2. **How much does it help?**
   - Expectation: 5-10% improvement in final loss
   - If <2%: marginal benefit
   - If >15%: strong validation

3. **Does it help convergence?**
   - Early indicator: Compare loss at epoch 10
   - If WITH Quinn is significantly lower by epoch 10: guidance is valuable

---

## Preliminary Technical Analysis

### What's Been Validated
- Complex-valued neural networks work (Phase 1)
- Scale-invariant losses effective (Phase 1 100% accuracy)
- Contrastive learning on embeddings works (Phase 2A margin exists)
- Hybrid architecture runs without errors (Phase 2B sanity check)

### What Remains Unknown
- Whether structural guidance improves language modeling
- Whether Phase 2A contrastive margin (0.595) is sufficient
- Whether hard negatives (within-category swaps) are "hard" enough
- Whether causal masking is necessary for Transformer

### Risk Factors
- Memory: Had to reduce batch size 32→16 to avoid OOM
- Time: 50 epochs × ~6 min/epoch = 300 min = 5 hours per run
- Noise: CPU training has variance; results may need averaging

---

## Expected Outcomes & Interpretation Guide

### Scenario A: Quinn Helps Significantly (>10% improvement)
**Result**: WITH Quinn final val loss 3.3, Baseline 3.7
**Interpretation**: Structural guidance is valuable; waveform encodes meaningful constraints
**Next Steps**:
- Fine-tune Quinn (unfreeze) and retrain hybrid end-to-end
- Increase model size to leverage guidance better
- Proceed to Phase 3 (production scaling)

### Scenario B: Quinn Helps Marginally (2-10% improvement)
**Result**: WITH Quinn final val loss 3.6, Baseline 3.7
**Interpretation**: Guidance has some value but weak signal
**Next Steps**:
- Improve hard negatives (cross-category swaps for larger contrastive margin)
- Add causal masking to Transformer
- Increase model capacity
- Consider alternative guidance signals

### Scenario C: Quinn Neutral (0-2% difference)
**Result**: WITH Quinn final val loss ≈ 3.7, Baseline 3.7
**Interpretation**: Waveform doesn't help for this task
**Possible causes**:
- Contrastive margin too small (Phase 2A at 0.595 vs 0.693)
- Wrong representation (waveform not meaningful for tokens)
- Guidance too weak compared to Transformer capacity
**Next Steps**:
- Investigate what Transformer is actually learning from waveform
- Visualize attention to waveform (is it using it?)
- Consider alternative task (code generation vs next-token)
- Scale model to see if larger capacity can leverage guidance

### Scenario D: Quinn Hurts (>5% worse)
**Result**: WITH Quinn final val loss 3.9+, Baseline 3.7
**Interpretation**: Waveform actually constrains model negatively
**Possible causes**:
- Waveform is noise (not structured enough)
- Mismatch between waveform and token distribution
- Cross-attention interfering with self-attention learning
**Next Steps**:
- Disable Quinn entirely for this phase
- Reconsider whether waveform completion is the right intermediate task
- Focus on scaling Transformer baseline
- Return to Quinn only after understanding why it doesn't help

---

## Diagnostic Checklist (After Training)

Once training completes, run these checks:

- [ ] Load both training histories
- [ ] Compare final validation loss (WITH vs Baseline)
- [ ] Check convergence rate (epochs 1-10)
- [ ] Run analysis script: `python experiments/phase2b_analysis.py`
- [ ] Visualize learning curves (plot val loss over epochs)
- [ ] Check if WITH Quinn diverged, converged, or matched
- [ ] Examine checkpoints exist and are valid

---

## Commands to Monitor & Analyze

**Live monitoring**:
```bash
tail -f /tmp/claude-0/-home-user-quinnhybrid/*/tasks/bgdnm2u3q.output  # Baseline
tail -f /tmp/claude-0/-home-user-quinnhybrid/*/tasks/bhrhmff5x.output  # WITH Quinn
```

**After completion**:
```bash
python experiments/phase2b_analysis.py  # Full comparison
ls -lh checkpoints/phase2b_*/  # Verify outputs exist
python -c "
import json
for name in ['run1', 'baseline']:
    with open(f'checkpoints/phase2b_{name}/training_history.json') as f:
        h = json.load(f)
    final = h[-1]['val']['lm_loss']
    best = min(e['val']['lm_loss'] for e in h)
    print(f'{name}: final={final:.4f}, best={best:.4f}')
"
```

---

## Next Session Plan

1. **Wait for training to complete** (~5 hours total)
2. **Run analysis script** to get verdict
3. **Interpret results** using Scenario A-D above
4. **Plan Phase 3** based on whether Quinn helps

If Quinn helps: Fine-tuning + scaling  
If Quinn neutral/hurts: Investigate or pivot approach

---

*This diagnostic was prepared during training setup. Results will be analyzed upon completion.*
