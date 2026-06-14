# Phase 1 Waveform Analysis: Quinn as 'Brain Waves'

**Analysis Date**: 1781388454.7949512
**Checkpoint**: checkpoints/run7/quinn_epoch050.pt

## Learned Frequency Structure

Estimated frequencies across 32 embedding dimensions:
```
  Dimension  0: f =   0.139
  Dimension  1: f =   0.207
  Dimension  2: f =   0.232
  Dimension  3: f =   0.139
  Dimension  4: f =   0.131
  Dimension  5: f =   0.180
  Dimension  6: f =   0.109
  Dimension  7: f =   0.177
  Dimension  8: f =   0.274
  Dimension  9: f =   0.177
  Dimension 10: f =   0.182
  Dimension 11: f =   0.166
  Dimension 12: f =   0.366
  Dimension 13: f =   0.130
  Dimension 14: f =   0.155
  Dimension 15: f =   0.137
  Dimension 16: f =   0.114
  Dimension 17: f =   0.161
  Dimension 18: f =   0.068
  Dimension 19: f =   0.148
  Dimension 20: f =   0.135
  Dimension 21: f =   0.090
  Dimension 22: f =   0.133
  Dimension 23: f =   0.138
  Dimension 24: f =   0.142
  Dimension 25: f =   0.231
  Dimension 26: f =   0.127
  Dimension 27: f =   0.177
  Dimension 28: f =   0.125
  Dimension 29: f =   0.235
  Dimension 30: f =   0.199
  Dimension 31: f =   0.201
```

## Token Type Frequency Signatures

| Token Type | Count | Spectral Centroid | Amplitude Mean | Amplitude Std | Examples |
|---|---|---|---|---|---|
| BRACKET/DELIMITER | 9 | 16.28 | 0.0230 | 0.0101 | LPAREN, RPAREN, LBRACKET |
| KEYWORD | 47 | 15.42 | 0.0246 | 0.0125 | def, class, return |
| NAME | 1 | 16.71 | 0.0249 | 0.0000 | NAME |
| NUMBER | 1 | 15.34 | 0.0248 | 0.0000 | NUMBER |
| OPERATOR | 6 | 15.99 | 0.0238 | 0.0098 | OP_ASSIGN, OP_AUGASSIGN, OP_ARITH |
| STRING | 1 | 16.30 | 0.0280 | 0.0000 | STRING |
| STRUCTURAL | 5 | 15.45 | 0.0230 | 0.0113 | NEWLINE, NL, INDENT |
| [SPECIAL] | 4 | 15.48 | 0.0275 | 0.0131 | <PAD>, <BOS>, <EOS> |

## Interpretation

### Hypothesis 1: Nesting Levels Correlate with Frequency
- KEYWORD tokens (def, if, for, etc.) should show distinctive frequencies
- Result: [See spectral centroid column above]

### Hypothesis 2: Token Roles Have Distinctive Patterns
- NAME tokens (variables) vs KEYWORD vs OPERATOR should separate
- Result: [See amplitude profiles]

### Hypothesis 3: Phase Coherence Measures Structural Agreement
- Tokens that co-occur structurally should have aligned phases
- Example: 'if' and ':' should have high coherence

### Next Steps
1. Compare frequency signatures with code structure metrics
2. Validate that high-frequency dimensions track fine structure (tokens)
3. Validate that low-frequency dimensions track broad structure (nesting)
4. Use phase as interpretability signal for model attention
