"""
Attention head specialization probe for SpectralLM.

Tests whether different attention heads attend preferentially to different
frequency bands — which would confirm that the transformer is actually using
the spectral structure in the embeddings, not just treating them as flat vectors.

Method: for each head h in each layer l, compute the L2 norm of each input
dimension's contribution to that head's Q and K projections. High norm = that
dimension strongly influences where this head attends.

If the structural band (dims 0..structural_end in real, embed_dim..embed_dim+structural_end
in imag) has systematically higher norm in some heads, and the semantic band has higher
norm in other heads, that's mechanistic confirmation.

Usage:
    python spectral/experiments/attention_probe.py \\
        --checkpoint checkpoints/spectral_small/spectral_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from spectral.training.trainer import build_model


def band_sensitivity(weight_slice: torch.Tensor,
                     embed_dim: int, structural_end: int, expr_end: int) -> dict:
    """
    Given a (head_dim, model_dim) weight matrix for one head,
    compute how sensitive this head is to each frequency band.

    The model_dim = 2*embed_dim is laid out as [real_0..real_D | imag_0..imag_D].
    Structural band: real[0:s_end] + imag[embed_dim:embed_dim+s_end]
    Expression band: real[s_end:e_end] + imag[embed_dim+s_end:embed_dim+e_end]
    Semantic band:   real[e_end:D]   + imag[embed_dim+e_end:2D]
    """
    D = embed_dim
    # Column norms: how much does each input dim affect this head's output
    col_norms = weight_slice.norm(dim=0)   # (model_dim,)

    s_idx = list(range(0, structural_end)) + list(range(D, D + structural_end))
    e_idx = list(range(structural_end, expr_end)) + list(range(D + structural_end, D + expr_end))
    m_idx = list(range(expr_end, D)) + list(range(D + expr_end, 2 * D))

    return {
        'structural': col_norms[s_idx].mean().item(),
        'expression': col_norms[e_idx].mean().item(),
        'semantic':   col_norms[m_idx].mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    ckpt      = torch.load(args.checkpoint, map_location='cpu')
    config    = ckpt['config']
    tokenizer = PythonStructuralTokenizer()
    model     = build_model(config, tokenizer.vocab)
    # Load only weights whose shapes match (handles old checkpoints with different
    # waveform_head shape before the amplitude-target fix).
    state = ckpt['model_state']
    current = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in current and current[k].shape == v.shape}
    skipped = [k for k in state if k not in compatible]
    if skipped:
        print(f"  (skipping {len(skipped)} mismatched keys: {skipped})")
    model.load_state_dict(compatible, strict=False)
    model.eval()

    emb     = model.embedding
    D       = emb.embed_dim
    s_end   = emb.structural_end
    e_end   = emb.expr_end
    n_heads = config.get('n_heads', 4)
    dim     = 2 * D
    head_dim = dim // n_heads

    print(f"\nCheckpoint : {args.checkpoint}")
    print(f"Epoch      : {ckpt.get('epoch')}   val_loss: {ckpt.get('val_loss', '?'):.4f}")
    print(f"embed_dim  : {D}   dim={dim}   n_heads={n_heads}   head_dim={head_dim}")
    print(f"Bands      : struct=0-{s_end}  expr={s_end}-{e_end}  sem={e_end}-{D}")

    print("\n" + "=" * 70)
    print("ATTENTION HEAD BAND SENSITIVITY")
    print("  Q+K column norm averaged over each frequency band.")
    print("  Dominant band per head shown at right.")
    print("  If heads specialize → different dominant bands across heads/layers.")
    print("=" * 70)

    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attn
        # in_proj_weight: (3*dim, dim) = [Q_weight; K_weight; V_weight]
        W = attn.in_proj_weight.detach()   # (3*dim, dim)
        Q_weight = W[:dim, :]              # (dim, dim)
        K_weight = W[dim:2*dim, :]         # (dim, dim)

        print(f"\n  Layer {layer_idx}:")
        print(f"  {'Head':<6} {'Q_struct':>9} {'Q_expr':>9} {'Q_sem':>9}  "
              f"{'K_struct':>9} {'K_expr':>9} {'K_sem':>9}  {'dominant'}")
        print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")

        head_summaries = []
        for h in range(n_heads):
            Q_h = Q_weight[h * head_dim:(h+1) * head_dim, :]   # (head_dim, dim)
            K_h = K_weight[h * head_dim:(h+1) * head_dim, :]

            q_sens = band_sensitivity(Q_h, D, s_end, e_end)
            k_sens = band_sensitivity(K_h, D, s_end, e_end)

            # Dominant band = highest avg of Q+K sensitivity
            combined = {
                b: q_sens[b] + k_sens[b]
                for b in ('structural', 'expression', 'semantic')
            }
            dominant = max(combined, key=combined.get)
            head_summaries.append(dominant)

            print(f"  H{h:<5} {q_sens['structural']:>9.4f} {q_sens['expression']:>9.4f} "
                  f"{q_sens['semantic']:>9.4f}  "
                  f"{k_sens['structural']:>9.4f} {k_sens['expression']:>9.4f} "
                  f"{k_sens['semantic']:>9.4f}  {dominant}")

        # Diversity summary for this layer
        unique = set(head_summaries)
        if len(unique) > 1:
            print(f"  → Layer {layer_idx} heads specialize across bands: {head_summaries}")
        else:
            print(f"  → Layer {layer_idx} all heads prefer: {head_summaries[0]} (no specialization)")

    # ── Overall specialization summary ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL: do any heads consistently specialize?")
    print("=" * 70)

    band_counts = {'structural': 0, 'expression': 0, 'semantic': 0}
    total_heads = 0

    for layer in model.layers:
        W = layer.attn.in_proj_weight.detach()
        Q_weight = W[:dim, :]
        K_weight = W[dim:2*dim, :]
        for h in range(n_heads):
            Q_h = Q_weight[h * head_dim:(h+1) * head_dim, :]
            K_h = K_weight[h * head_dim:(h+1) * head_dim, :]
            q_s = band_sensitivity(Q_h, D, s_end, e_end)
            k_s = band_sensitivity(K_h, D, s_end, e_end)
            combined = {b: q_s[b] + k_s[b] for b in q_s}
            dominant = max(combined, key=combined.get)
            band_counts[dominant] += 1
            total_heads += 1

    print(f"\n  Total heads: {total_heads} across {len(model.layers)} layers")
    for band, count in band_counts.items():
        print(f"  {band:<12}: {count:2d} heads dominant  "
              f"({count/total_heads*100:.0f}%)")

    random_pct = 100 / 3
    print(f"\n  Random baseline: each band ~{random_pct:.0f}%")
    dominated = max(band_counts, key=band_counts.get)
    if band_counts[dominated] / total_heads > 0.5:
        print(f"  → {dominated} band dominates ({band_counts[dominated]/total_heads*100:.0f}%)"
              f" — no specialization, heads converge to one band")
    else:
        print(f"  → Heads distributed across bands — specialization present")


if __name__ == '__main__':
    main()
