"""
Phase 1 Evaluation Script

Evaluates a trained QuINN checkpoint against the Phase 1 success criterion:

    "QuINN's token count estimates at 50% prefix converge to within ±15% of
     true length significantly more reliably than the baseline, and improve
     monotonically as prefix length increases."

Outputs:
  - Per-prefix-fraction accuracy table (QuINN vs baseline)
  - Monotonicity check across prefix fractions
  - Scatter plot of predicted vs true lengths (saved as PNG)
  - Summary verdict: PASS / FAIL for Phase 1 success condition

Usage:
    python experiments/phase1_eval.py --checkpoint checkpoints/quinn_epoch030.pt
"""

import argparse
import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np


def evaluate_checkpoint(checkpoint_path: str, corpus_file: str):
    from training.config import QuINNConfig
    from training.trainer import QuINNTrainer
    from data.tokenizer import PythonStructuralTokenizer
    from quinn import QuINN
    from data import PythonCodeDataset, collate_fn
    from torch.utils.data import DataLoader, random_split

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = QuINNConfig(**{k: cfg_dict[k] for k in cfg_dict if hasattr(QuINNConfig, k)})
    cfg.corpus_file = corpus_file

    tok = PythonStructuralTokenizer()
    model = QuINN(
        vocab_size=tok.vocab_size,
        embed_dim=cfg.embed_dim,
        manifold_dim=cfg.manifold_dim,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        max_seq_len=cfg.max_seq_len,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean_train_len = ckpt.get("mean_train_len", 200.0)

    # Load test set
    full_dataset = PythonCodeDataset(
        cfg.corpus_file,
        min_seq_len=cfg.min_seq_len,
        max_seq_len=cfg.max_seq_len_data,
        min_prefix_frac=0.05,
        max_prefix_frac=0.95,
        seed=42,
    )
    n = len(full_dataset)
    n_train = int(n * cfg.train_split)
    n_val   = int(n * cfg.val_split)
    n_test  = n - n_train - n_val
    _, _, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn)

    # ── Collect predictions across all prefix fractions ───────────────────
    prefix_fracs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    results = []

    print(f"\n{'Prefix':>8}  {'QuINN±15%':>10}  {'Baseline±15%':>13}  {'Beats baseline':>15}")
    print("-" * 55)

    prev_quinn_acc = None
    monotone = True

    for frac in prefix_fracs:
        quinn_correct = 0
        baseline_correct = 0
        total = 0
        all_pred = []
        all_true = []

        with torch.no_grad():
            for batch in test_loader:
                bf = batch["prefix_frac"]
                keep = (bf >= frac - 0.05) & (bf <= frac + 0.05)
                if keep.sum() == 0:
                    continue

                b = {k: v[keep] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                _, _, pred_log_len = model(
                    b["prefix_tokens"], b["prefix_positions"], b["prefix_mask"]
                )
                pred_len  = torch.exp(pred_log_len)
                true_len  = b["seq_len"].float()
                base_len  = torch.full_like(true_len, mean_train_len)

                quinn_correct   += ((pred_len - true_len).abs() / true_len.clamp(min=1) < 0.15).sum().item()
                baseline_correct += ((base_len - true_len).abs() / true_len.clamp(min=1) < 0.15).sum().item()
                total += true_len.shape[0]
                all_pred.extend(pred_len.tolist())
                all_true.extend(true_len.tolist())

        if total == 0:
            continue

        quinn_acc    = quinn_correct    / total
        baseline_acc = baseline_correct / total
        beats        = quinn_acc > baseline_acc

        if prev_quinn_acc is not None and quinn_acc < prev_quinn_acc - 0.01:
            monotone = False
        prev_quinn_acc = quinn_acc

        results.append({
            "prefix_frac":    frac,
            "n_samples":      total,
            "quinn_acc_15pct":    quinn_acc,
            "baseline_acc_15pct": baseline_acc,
            "beats_baseline": beats,
            "mean_pred_len":  float(np.mean(all_pred)),
            "mean_true_len":  float(np.mean(all_true)),
        })

        print(f"  {frac:.0%}      {quinn_acc:.3f}        {baseline_acc:.3f}           {'✓' if beats else '✗'}")

    # ── Phase 1 success criterion ─────────────────────────────────────────
    result_50 = next((r for r in results if r["prefix_frac"] == 0.50), None)

    print("\n── Phase 1 Success Criterion ────────────────────────────────────────")
    if result_50:
        passes_50 = result_50["quinn_acc_15pct"] > result_50["baseline_acc_15pct"]
        print(f"  At 50% prefix: QuINN={result_50['quinn_acc_15pct']:.3f}  Baseline={result_50['baseline_acc_15pct']:.3f}")
        print(f"  Beats baseline at 50%: {'PASS ✓' if passes_50 else 'FAIL ✗'}")
    else:
        passes_50 = False
        print("  No samples at 50% prefix — cannot evaluate criterion")

    print(f"  Monotonically improving: {'PASS ✓' if monotone else 'FAIL ✗'}")

    overall_pass = passes_50 and monotone
    print(f"\n  ══ Phase 1 overall: {'PASS ✓' if overall_pass else 'FAIL — needs more training or architecture revision'} ══")

    # ── Save results ──────────────────────────────────────────────────────
    out = {
        "checkpoint": checkpoint_path,
        "mean_train_len": mean_train_len,
        "results_by_prefix": results,
        "monotone": monotone,
        "phase1_pass": overall_pass,
    }
    out_path = os.path.join(os.path.dirname(checkpoint_path), "phase1_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Optional plot ─────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fracs = [r["prefix_frac"] for r in results]
        quinn_accs   = [r["quinn_acc_15pct"]    for r in results]
        baseline_accs = [r["baseline_acc_15pct"] for r in results]

        plt.figure(figsize=(8, 5))
        plt.plot(fracs, quinn_accs,    "b-o", label="QuINN ±15%", linewidth=2)
        plt.plot(fracs, baseline_accs, "r--s", label="Baseline (mean length) ±15%", linewidth=2)
        plt.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        plt.xlabel("Prefix fraction")
        plt.ylabel("Fraction within ±15% of true length")
        plt.title("Phase 1: Length Prediction Accuracy vs Prefix Length")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(fracs, [f"{f:.0%}" for f in fracs])
        plt.tight_layout()

        plot_path = os.path.join(os.path.dirname(checkpoint_path), "phase1_length_accuracy.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved → {plot_path}")
    except ImportError:
        pass  # matplotlib optional

    return overall_pass


def main():
    parser = argparse.ArgumentParser(description="QuINN Phase 1 Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--corpus", default="data/corpus.txt", help="Corpus file list")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.corpus)


if __name__ == "__main__":
    main()
