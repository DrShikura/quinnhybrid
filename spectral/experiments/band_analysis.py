"""
Band specialization analysis for SpectralLM.

Checks:
  1. Frequency drift       — did learned freq[] stay in their initialized ranges?
                             (only meaningful when band_init=True)
  2. Frequency clustering  — do learned frequencies cluster into low/mid/high groups
                             even when initialized uniformly? (self-organization test)
  3. Amplitude clustering  — do structural tokens activate the structural band more
                             than semantic tokens?
  4. Token rankings        — which tokens most strongly activate each band?
  5. Specialization index  — quantifies home-band preference per category
  6. Structural/semantic contrast per token

Note: the Gaussian locality envelope (log_sigma) has been removed from SpectralEmbedding
because it was centered at position 0, making the semantic band essentially dead past
position ~50. The frequency hierarchy alone now carries multi-scale structure.

Usage:
    python spectral/experiments/band_analysis.py \\
        --checkpoint checkpoints/spectral_small/spectral_best.pt

    # Compare self-organizing vs banded init:
    python spectral/experiments/band_analysis.py \\
        --checkpoint checkpoints/spectral_self_organized/spectral_best.pt
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
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()

    emb   = model.embedding
    D     = emb.embed_dim
    s_end = emb.structural_end
    e_end = emb.expr_end

    band_init     = config.get('band_init', True)
    acoustic_init = config.get('acoustic_init', True)

    print(f"\nCheckpoint    : {args.checkpoint}")
    print(f"Epoch         : {ckpt.get('epoch')}   val_loss: {ckpt.get('val_loss', '?'):.4f}")
    print(f"embed_dim     : {D}   bands: 0-{s_end} / {s_end}-{e_end} / {e_end}-{D}")
    print(f"band_init     : {band_init}    acoustic_init: {acoustic_init}")
    print(f"Vocab size    : {len(tokenizer.vocab)}")

    amp  = emb.amplitude.weight.detach()   # (V, D)
    freq = emb.freq.detach()               # (D,)

    # ── 1. Frequency drift ────────────────────────────────────────────────────
    if band_init:
        print("\n" + "=" * 60)
        print("1. FREQUENCY DRIFT (init ranges vs learned)")
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
            lo, hi   = init_ranges[name]
            mn, mx   = f_slice.min().item(), f_slice.max().item()
            mean     = f_slice.mean().item()
            in_range = ((f_slice >= lo) & (f_slice <= hi)).float().mean().item()
            print(f"  {name:<12}  init=[{lo:.2f},{hi:.2f}]  "
                  f"learned=[{mn:.3f},{mx:.3f}]  mean={mean:.3f}  "
                  f"in_range={in_range*100:.0f}%")

    # ── 2. Frequency clustering (self-organization test) ─────────────────────
    print("\n" + "=" * 60)
    print("2. FREQUENCY SELF-ORGANIZATION")
    print("   Sort all dims by learned frequency, check if they cluster")
    print("   in the ranges matching structural/expression/semantic.")
    print("   Key test for --no-band-init runs.")
    print("=" * 60)

    sorted_freqs, sorted_idx = freq.sort()
    n_low  = (freq < 0.10).sum().item()
    n_mid  = ((freq >= 0.10) & (freq < 0.25)).sum().item()
    n_high = (freq >= 0.25).sum().item()
    expected_low  = s_end
    expected_mid  = e_end - s_end
    expected_high = D - e_end

    print(f"\n  freq < 0.10  (structural range): {n_low:3d} dims  "
          f"(expected {expected_low} if banded init)")
    print(f"  0.10 ≤ freq < 0.25 (expr range): {n_mid:3d} dims  "
          f"(expected {expected_mid})")
    print(f"  freq ≥ 0.25 (semantic range):    {n_high:3d} dims  "
          f"(expected {expected_high})")
    print(f"\n  Frequency range: [{freq.min():.3f}, {freq.max():.3f}]  "
          f"mean={freq.mean():.3f}  std={freq.std():.3f}")
    print(f"\n  Lowest-freq 8 dims (most global): {sorted_freqs[:8].tolist()}")
    print(f"  Highest-freq 8 dims (most local): {sorted_freqs[-8:].tolist()}")

    # ── 3. Amplitude band specialization ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. AMPLITUDE BAND SPECIALIZATION BY TOKEN CATEGORY")
    print("=" * 60)

    cat_totals = {c: [0.0, 0.0, 0.0, 0] for c in ('structural', 'expression', 'semantic', 'other')}
    for idx, tok in enumerate(tokenizer.vocab):
        cat = token_category(tok)
        s, e, m = band_mean_amplitude(amp[idx], s_end, e_end)
        cat_totals[cat][0] += s
        cat_totals[cat][1] += e
        cat_totals[cat][2] += m
        cat_totals[cat][3] += 1

    print(f"\n  {'Category':<14} {'Count':>5}  {'Struct':>8}  {'Expr':>8}  {'Semantic':>8}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
    for cat, (s, e, m, n) in cat_totals.items():
        if n == 0: continue
        dominant = ['struct','expr','semantic'][[s/n, e/n, m/n].index(max(s/n, e/n, m/n))]
        print(f"  {cat:<14} {n:>5}  {s/n:>8.5f}  {e/n:>8.5f}  {m/n:>8.5f}  ← {dominant}")

    # ── 4. Top tokens per band ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. TOP 10 TOKENS BY BAND ACTIVATION")
    print("=" * 60)

    s_act = amp[:, :s_end].abs().mean(dim=1)
    e_act = amp[:, s_end:e_end].abs().mean(dim=1)
    m_act = amp[:, e_end:].abs().mean(dim=1)

    for label, act in [('Structural band', s_act),
                       ('Expression band', e_act),
                       ('Semantic band',   m_act)]:
        top  = act.topk(10).indices.tolist()
        toks = [(tokenizer.vocab[i], act[i].item()) for i in top]
        print(f"\n  {label}:")
        for tok, val in toks:
            print(f"    {tok:<22}  {val:.5f}  [{token_category(tok)}]")

    # ── 5. Specialization index ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. SPECIALIZATION INDEX  (0.333 = random)")
    print("=" * 60)
    for cat, home_band in [('structural', 0), ('expression', 1), ('semantic', 2)]:
        toks_in_cat = [i for i, tok in enumerate(tokenizer.vocab)
                       if token_category(tok) == cat]
        if not toks_in_cat: continue
        rows = amp[toks_in_cat]
        s_m = rows[:, :s_end].abs().mean().item()
        e_m = rows[:, s_end:e_end].abs().mean().item()
        m_m = rows[:, e_end:].abs().mean().item()
        vals  = [s_m, e_m, m_m]
        total = sum(vals) + 1e-12
        si    = vals[home_band] / total
        print(f"  {cat:<14}  [{s_m:.5f}, {e_m:.5f}, {m_m:.5f}]  "
              f"SI(home)={si:.3f}  {'✓' if si > 0.333 else '✗'}")

    # ── 6. Per-token structural vs semantic contrast ──────────────────────────
    print("\n" + "=" * 60)
    print("6. STRUCTURAL vs SEMANTIC CONTRAST PER TOKEN")
    print("   positive = lives in structural band; negative = semantic band")
    print("=" * 60)

    contrasts = []
    for idx, tok in enumerate(tokenizer.vocab):
        if tok.startswith('<'): continue
        s_a = amp[idx, :s_end].abs().mean().item()
        m_a = amp[idx, e_end:].abs().mean().item()
        contrasts.append((tok, s_a - m_a, token_category(tok)))
    contrasts.sort(key=lambda x: x[1], reverse=True)

    print("\n  Most STRUCTURAL:")
    for tok, c, cat in contrasts[:10]:
        print(f"    {tok:<22}  {c:+.5f}  [{cat}]")
    print("\n  Most SEMANTIC:")
    for tok, c, cat in contrasts[-10:]:
        print(f"    {tok:<22}  {c:+.5f}  [{cat}]")


if __name__ == '__main__':
    main()
