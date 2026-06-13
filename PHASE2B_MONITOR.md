# Phase 2B Training Monitor

**Started**: 2026-06-13  
**Configuration**: 50 epochs, batch size 32, 1077 files (915 train, 107 val, 55 test)

---

## Active Training Jobs

| Job ID | Configuration | Status | Expected Duration |
|--------|--------------|--------|-------------------|
| `bf1bpujv2` | WITH Quinn (Phase 1 checkpoint) | Running | ~4 hours |
| `bgdnm2u3q` | WITHOUT Quinn (random init) | Running | ~4 hours |

**Log files**:
- `bf1bpujv2`: `/tmp/claude-0/-home-user-quinnhybrid/701c2acb-fae8-57e0-a51e-65ea7c134750/tasks/bf1bpujv2.output`
- `bgdnm2u3q`: `/tmp/claude-0/-home-user-quinnhybrid/701c2acb-fae8-57e0-a51e-65ea7c134750/tasks/bgdnm2u3q.output`

---

## Real-Time Metrics (Updated as training progresses)

### With Quinn (bf1bpujv2)
```
Progress: [====>....................................] 8/50 (16%)
Epoch:    In progress
Val LM Loss (last): TBD
Trend:    TBD
```

### Baseline (bgdnm2u3q)
```
Progress: [====>....................................] 8/50 (16%)
Epoch:    In progress
Val LM Loss (last): TBD
Trend:    TBD
```

---

## Key Milestones to Watch

**Epoch 10**: 
- Both runs should show clear trend (converging or diverging?)
- Quinn should show advantage if it exists

**Epoch 25**:
- Halfway point; trend should be clear
- Any unexpected behavior would indicate architecture issues

**Epoch 50**:
- Final results ready for analysis
- Run analysis script to compare

---

## Analysis Commands (After Training)

**Check final results**:
```bash
python experiments/phase2b_analysis.py
```

**View WITH Quinn training history**:
```bash
python -c "
import json
with open('checkpoints/phase2b_run1/training_history.json') as f:
    hist = json.load(f)
print('Epoch  Train LM Loss  Val LM Loss')
for e in hist[-10:]:  # last 10 epochs
    print(f\"{e['epoch']:3d}    {e['train']['lm_loss']:.4f}        {e['val']['lm_loss']:.4f}\")
"
```

**View Baseline training history**:
```bash
python -c "
import json
with open('checkpoints/phase2b_baseline/training_history.json') as f:
    hist = json.load(f)
print('Epoch  Train LM Loss  Val LM Loss')
for e in hist[-10:]:  # last 10 epochs
    print(f\"{e['epoch']:3d}    {e['train']['lm_loss']:.4f}        {e['val']['lm_loss']:.4f}\")
"
```

---

## Expected Outcomes

**Optimistic** (Quinn helps):
- WITH Quinn converges faster (lower loss in first 10 epochs)
- Final val loss 5-10% lower than baseline
- Interpretation: Structural guidance improves token prediction

**Neutral** (Quinn doesn't help):
- Both runs converge similarly
- Final val loss ~same
- Interpretation: Contrastive margin too small or wrong feature

**Pessimistic** (Quinn hurts):
- Baseline lower than WITH Quinn
- Interpretation: Waveform doesn't encode useful structure for this task

---

## Diagnostic Steps (If Needed)

### If Quinn doesn't help:
1. Check if waveform is being used correctly
   - Verify: `waveform_proj` is learning meaningful transformations
   - Check: cross-attention weights (is it looking at waveform?)

2. Improve hard negatives
   - Currently: within-category swaps (NAME↔NAME)
   - Try: cross-category swaps (NAME↔KEYWORD)
   - Retrain Phase 2A for larger contrastive margin

3. Add causal masking
   - Currently: Transformer can attend to future tokens
   - Impact: might hurt by overfitting to target

4. Scale up model
   - Currently: small (embed_dim=32, 108k params)
   - Try: embed_dim=64, more layers

### If Quinn helps but marginal:
1. Fine-tune Quinn (unfreeze parameters)
2. Retrain hybrid end-to-end
3. Larger model + more epochs

---

## Fallback Options

If Phase 2B shows no benefit:
1. **Alternative 1**: Keep Quinn frozen, train Transformer for longer
2. **Alternative 2**: Improve Phase 2A negatives, retrain
3. **Alternative 3**: Test on a different task (code completion, not next-token)
4. **Alternative 4**: Skip Phase 2B, move directly to Phase 3 (scaling)

---

*Status: Training in progress. Will update with results when complete.*
