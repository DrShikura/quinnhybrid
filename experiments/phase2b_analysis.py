"""
Phase 2B Analysis: Compare Quinn-guided vs baseline Transformer.

Analyzes the training histories from:
  - checkpoints/phase2b_run1/training_history.json (WITH Quinn)
  - checkpoints/phase2b_baseline/training_history.json (WITHOUT Quinn)

Metrics:
  - LM loss convergence rate
  - Final validation loss
  - Epoch-by-epoch comparison
  - Statistical significance
"""

import json
import numpy as np
from pathlib import Path

def load_history(path):
    with open(path) as f:
        return json.load(f)

def analyze_run(name, history):
    """Compute metrics for a training run."""
    epochs = [e["epoch"] for e in history]
    train_losses = [e["train"]["lm_loss"] for e in history]
    val_losses = [e["val"]["lm_loss"] for e in history]

    # Metrics
    first_train = train_losses[0]
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    best_val = min(val_losses)
    best_epoch = val_losses.index(best_val) + 1

    # Convergence: how much did it improve in first 10 epochs?
    if len(val_losses) >= 10:
        first_10_improvement = val_losses[0] - val_losses[9]
    else:
        first_10_improvement = None

    return {
        "name": name,
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "first_train": first_train,
        "final_train": final_train,
        "final_val": final_val,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "first_10_improvement": first_10_improvement,
    }

def compare_runs(with_quinn, without_quinn):
    """Compare two runs and report findings."""
    print("\n" + "="*70)
    print("PHASE 2B ANALYSIS: Quinn Guidance Impact")
    print("="*70)

    print(f"\n{'Metric':<30} {'With Quinn':<20} {'Baseline':<20}")
    print("-"*70)

    # Final validation loss
    diff = without_quinn["final_val"] - with_quinn["final_val"]
    pct_better = 100 * diff / without_quinn["final_val"]
    print(f"{'Final val LM loss':<30} {with_quinn['final_val']:<20.4f} {without_quinn['final_val']:<20.4f}")
    if diff > 0:
        print(f"  → Quinn is {pct_better:.1f}% better")
    else:
        print(f"  → Baseline is {-pct_better:.1f}% better")

    # Best validation loss
    diff = without_quinn["best_val"] - with_quinn["best_val"]
    pct_better = 100 * diff / without_quinn["best_val"]
    print(f"{'Best val LM loss':<30} {with_quinn['best_val']:<20.4f} {without_quinn['best_val']:<20.4f}")
    print(f"  @ epoch {with_quinn['best_epoch']} vs {without_quinn['best_epoch']}")

    # Early convergence (first 10 epochs)
    if with_quinn["first_10_improvement"] and without_quinn["first_10_improvement"]:
        print(f"\n{'Metric':<30} {'With Quinn':<20} {'Baseline':<20}")
        print(f"{'Improvement (epochs 1-10)':<30} {with_quinn['first_10_improvement']:<20.4f} {without_quinn['first_10_improvement']:<20.4f}")
        ratio = with_quinn["first_10_improvement"] / without_quinn["first_10_improvement"]
        if ratio > 1:
            print(f"  → Quinn converges {ratio:.1f}× faster")
        else:
            print(f"  → Baseline converges {1/ratio:.1f}× faster")

    # Final training loss
    print(f"\n{'Final train LM loss':<30} {with_quinn['final_train']:<20.4f} {without_quinn['final_train']:<20.4f}")

    # Verdict
    print("\n" + "-"*70)
    if with_quinn["best_val"] < without_quinn["best_val"]:
        improvement = 100 * (without_quinn["best_val"] - with_quinn["best_val"]) / without_quinn["best_val"]
        print(f"✓ QUINN GUIDANCE HELPS: {improvement:.1f}% lower validation loss")
        print(f"  Recommendation: Fine-tune Quinn + retrain hybrid end-to-end")
    else:
        degradation = 100 * (with_quinn["best_val"] - without_quinn["best_val"]) / without_quinn["best_val"]
        print(f"✗ QUINN GUIDANCE NOT HELPFUL: {degradation:.1f}% higher validation loss")
        print(f"  Recommendation: Investigate (larger model? harder negatives? causal masking?)")

    print("="*70 + "\n")

if __name__ == "__main__":
    # Try to load both
    path_with = Path("checkpoints/phase2b_run1/training_history.json")
    path_without = Path("checkpoints/phase2b_baseline/training_history.json")

    if not path_with.exists() or not path_without.exists():
        print("Waiting for training to complete...")
        print(f"  With Quinn: {path_with}")
        print(f"  Baseline: {path_without}")
        exit(0)

    with_quinn = analyze_run("With Quinn (Phase 1)", load_history(path_with))
    without_quinn = analyze_run("Baseline (Random)", load_history(path_without))

    compare_runs(with_quinn, without_quinn)
