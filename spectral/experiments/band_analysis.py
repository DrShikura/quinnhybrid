"""
Band specialization analysis for SpectralLM.

Tests whether the three frequency bands actually specialized during training:
  - Structural band  (dims 0..structural_end):    global patterns (def, class, INDENT)
  - Expression band  (dims structural_end..expr_end): statement-level patterns
  - Semantic band    (dims expr_end..embed_dim):   local token patterns (ops, literals)

Checks:
  1. Frequency drift  — did learned freq[] stay in their initialized ranges?
  2. Sigma drift      — did structural dims keep large sigma (global locality)?
  3. Amplitude clustering — do structural tokens activate the structural band
                            more than semantic tokens, and vice versa?
  4. Token rankings   — which tokens most strongly activate each band?

Usage:
    python spectral/experiments/band_analysis.py --checkpoint checkpoints/spectral_small/spectral_best.pt
"""

import argparse
import sys
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from spectral.training.trainer import build_model


# ── Token category labels ────────────────────────────────────────────────────

_STRUCTURAL = {
    'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'with',
    'try', 'except', 'finally', 'return', 'yield', 'raise', 'pass',
    'break', 'continue', 'import', 'from', 'as', 'and', 'or', 'not',
    'in', 'is', 'lambda', 'global', 'nonlocal', 'del', 'assert',
    'INDENT', 'DEDENT', 'NEWLINE', 'NL', 'COLON',
}
_EXPRESSION = {
    'NAME', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'LBRACE', 'RBRACE',
    'COMMA', 'DOT', 'SEMICOLON', 'ARROW', 'STAR', 'DOUBLESTAR',
    'OP_ASSIGN', 'OP_AUGASSIGN',
}
_SEMANTIC = {
    'NUMBER', 'STRING', 'ELLIPSIS',
    'OP_ARITH', 'OP_COMPARE', 'OP_BITWISE', 'OP_SHIFT',
}


def token_category(tok: str) -> str:
    if tok in _STRUCTURAL:  return 'structural'
    if tok in _EXPRESSION:  return 'expression'
    if tok in _SEMANTIC:    return 'semantic'
    return 'other'


def band_mean_amplitude(amp_row: torch.Tensor,
                        structural_end: int, expr_end: int) -> tuple:
    """Return (structural_mean, expression_mean, semantic_mean) for one token row."""
    s = amp_row[:structural_end].abs().mean().item()
    e = amp_row[structural_end:expr_end].abs().mean().item()
    m = amp_row[expr_end:].abs().mean().item()
    return s, e, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    ckpt      = torch.load(args.checkpoint, map_location='cpu')
    config    = ckpt['config']
    tokenizer = PythonStructuralTokenizer()
    model     = build_model(config, tokenizer.vocab)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    emb   = model.embedding
    D     = emb.embed_dim
    s_end = emb.structural_end
    e_end = emb.expr_end

    print(f"\nCheckpoint : {args.checkpoint}")
    print(f"Epoch      : {ckpt.get('epoch')}   val_loss: {ckpt.get('val_loss', '?'):.4f}")
    print(f"embed_dim  : {D}   bands: 0-{s_end} / {s_end}-{e_end} / {e_end}-{D}")
    print(f"Vocab size : {len(tokenizer.vocab)}")

    amp       = emb.amplitude.weight.detach()   # (V, D)
    freq      = emb.freq.detach()               # (D,)
    log_sigma = emb.log_sigma.detach()          # (D,)
    sigma     = log_sigma.exp()

    # ── 1. Frequency drift ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. FREQUENCY DRIFT (init vs learned)")
    print("=" * 60)

    init_ranges = {
        'structural': (0.02, 0.10),
        'expression': (0.10, 0.25),
        'semantic':   (0.25, 0.50),
    }
    bands = {
        'structural': freq[:s_end],
        'expression': freq[s_end:e_end],
        'semantic':   freq[e_end:],
    }
    for name, f_slice in bands.items():
        lo, hi = init_ranges[name]
        mn, mx = f_slice.min().item(), f_slice.max().item()
        mean   = f_slice.mean().item()
        in_range = ((f_slice >= lo) & (f_slice <= hi)).float().mean().item()
        print(f"  {name:<12}  init=[{lo:.2f},{hi:.2f}]  "
              f"learned=[{mn:.3f},{mx:.3f}]  mean={mean:.3f}  "
              f"in_range={in_range*100:.0f}%")

    # ── 2. Sigma (locality) drift ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. LOCALITY BANDWIDTH (sigma) DRIFT")
    print("=" * 60)

    init_sigmas = {
        'structural': emb.max_seq_len,
        'expression': emb.max_seq_len / 4,
        'semantic':   max(1, emb.max_seq_len / 16),
    }
    sigma_bands = {
        'structural': sigma[:s_end],
        'expression': sigma[s_end:e_end],
        'semantic':   sigma[e_end:],
    }
    for name, s_slice in sigma_bands.items():
        init_s = init_sigmas[name]
        mn, mx = s_slice.min().item(), s_slice.max().item()
        mean   = s_slice.mean().item()
        print(f"  {name:<12}  init_σ={init_s:.0f}  "
              f"learned_σ=[{mn:.1f},{mx:.1f}]  mean={mean:.1f}")

    # ── 3. Amplitude band specialization ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. AMPLITUDE BAND SPECIALIZATION BY TOKEN CATEGORY")
    print("   (higher = more activation in that band)")
    print("=" * 60)

    cat_totals = {c: [0.0, 0.0, 0.0, 0] for c in ('structural', 'expression', 'semantic', 'other')}
    for idx, tok in enumerate(tokenizer.vocab):
        cat = token_category(tok)
        s, e, m = band_mean_amplitude(amp[idx], s_end, e_end)
        cat_totals[cat][0] += s
        cat_totals[cat][1] += e
        cat_totals[cat][2] += m
        cat_totals[cat][3] += 1

    print(f"\n  {'Category':<14} {'Count':>5}  {'Structural':>11}  {'Expression':>11}  {'Semantic':>9}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*11}  {'-'*11}  {'-'*9}")
    for cat, (s, e, m, n) in cat_totals.items():
        if n == 0: continue
        print(f"  {cat:<14} {n:>5}  {s/n:>11.5f}  {e/n:>11.5f}  {m/n:>9.5f}")

    # ── 4. Top tokens per band ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. TOP 10 TOKENS BY BAND ACTIVATION")
    print("=" * 60)

    s_act = amp[:, :s_end].abs().mean(dim=1)       # (V,)
    e_act = amp[:, s_end:e_end].abs().mean(dim=1)
    m_act = amp[:, e_end:].abs().mean(dim=1)

    for label, act in [('Structural band', s_act),
                       ('Expression band', e_act),
                       ('Semantic band',   m_act)]:
        top = act.topk(10).indices.tolist()
        toks = [(tokenizer.vocab[i], act[i].item()) for i in top]
        print(f"\n  {label} top tokens:")
        for tok, val in toks:
            cat = token_category(tok)
            print(f"    {tok:<20}  {val:.5f}  [{cat}]")

    # ── 5. Specialization index ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. SPECIALIZATION INDEX")
    print("   (How much does each token category prefer its 'home' band?)")
    print("   SI = home_band_activation / sum_of_all_band_activations")
    print("=" * 60)

    for cat, home_band in [('structural', 0), ('expression', 1), ('semantic', 2)]:
        toks_in_cat = [i for i, tok in enumerate(tokenizer.vocab)
                       if token_category(tok) == cat]
        if not toks_in_cat:
            continue
        rows = amp[toks_in_cat]   # (N, D)
        s_m = rows[:, :s_end].abs().mean().item()
        e_m = rows[:, s_end:e_end].abs().mean().item()
        m_m = rows[:, e_end:].abs().mean().item()
        vals = [s_m, e_m, m_m]
        total = sum(vals) + 1e-12
        si = vals[home_band] / total
        print(f"  {cat:<14}  band_acts=[{s_m:.5f}, {e_m:.5f}, {m_m:.5f}]  "
              f"SI(home)={si:.3f}")

    # ── 6. Structural vs Semantic contrast ────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. STRUCTURAL vs SEMANTIC BAND CONTRAST")
    print("   For each token: structural_act - semantic_act")
    print("   Positive = lives in structural band; Negative = lives in semantic band")
    print("=" * 60)

    contrasts = []
    for idx, tok in enumerate(tokenizer.vocab):
        if tok.startswith('<'): continue
        s_a = amp[idx, :s_end].abs().mean().item()
        m_a = amp[idx, e_end:].abs().mean().item()
        contrasts.append((tok, s_a - m_a, token_category(tok)))

    contrasts.sort(key=lambda x: x[1], reverse=True)

    print("\n  Most STRUCTURAL (positive = structural-dominant):")
    for tok, c, cat in contrasts[:10]:
        print(f"    {tok:<22}  {c:+.5f}  [{cat}]")

    print("\n  Most SEMANTIC (negative = semantic-dominant):")
    for tok, c, cat in contrasts[-10:]:
        print(f"    {tok:<22}  {c:+.5f}  [{cat}]")


if __name__ == '__main__':
    main()
