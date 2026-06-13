"""
Phase 1 Diagnostic: evaluate a checkpoint at FIXED prefix fractions.

Unlike the noisy @50% metric during training (which randomly samples prefix
fractions and then filters), this script evaluates the same files repeatedly
at each target fraction by overriding the random sampling.

Usage:
    python experiments/phase1_diagnostic.py --checkpoint checkpoints/run4/quinn_epoch030.pt
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from quinn import QuINN
from quinn.loss import QuINNLoss
from data.tokenizer import PythonStructuralTokenizer
from data.dataset import PythonCodeDataset, collate_fn
from training.config import QuINNConfig


def evaluate_at_fraction(model, samples, fraction: float, tokenizer, device, batch_size=64):
    """
    Evaluate all samples at exactly `fraction` prefix length.
    Returns dict with accuracy metrics.
    """
    model.eval()
    pad_id = tokenizer.pad_id

    all_pred_lens = []
    all_true_lens = []

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        batch_size_actual = len(batch_samples)

        max_prefix = max(max(1, int(round(len(s) * fraction))) for s in batch_samples)
        max_all = max(len(s) for s in batch_samples)

        prefix_tokens  = torch.full((batch_size_actual, max_prefix), pad_id, dtype=torch.long)
        prefix_pos     = torch.zeros(batch_size_actual, max_prefix, dtype=torch.long)
        prefix_mask    = torch.zeros(batch_size_actual, max_prefix, dtype=torch.bool)

        for i, s in enumerate(batch_samples):
            n = len(s)
            prefix_len = max(1, int(round(n * fraction)))
            prefix_len = min(prefix_len, n - 1)
            prefix_tokens[i, :prefix_len] = torch.tensor(s[:prefix_len], dtype=torch.long)
            prefix_pos[i, :prefix_len] = torch.arange(prefix_len)
            prefix_mask[i, :prefix_len] = True

        with torch.no_grad():
            _, _, pred_log_len = model(
                prefix_tokens.to(device),
                prefix_pos.to(device),
                prefix_mask.to(device),
            )

        pred_lens = torch.exp(pred_log_len).cpu()
        true_lens = torch.tensor([len(s) for s in batch_samples], dtype=torch.float)

        all_pred_lens.extend(pred_lens.tolist())
        all_true_lens.extend(true_lens.tolist())

    pred_arr = np.array(all_pred_lens)
    true_arr = np.array(all_true_lens)
    rel_err  = np.abs(pred_arr - true_arr) / true_arr

    return {
        "fraction": fraction,
        "n_samples": len(true_arr),
        "acc_15pct": float((rel_err < 0.15).mean()),
        "median_rel_err": float(np.median(rel_err)),
        "mean_pred_len": float(pred_arr.mean()),
        "mean_true_len": float(true_arr.mean()),
    }


def run_diagnostic(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = QuINNConfig(**{k: cfg_dict[k] for k in cfg_dict if hasattr(QuINNConfig(), k)})

    tok = PythonStructuralTokenizer()
    device = torch.device("cpu")

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

    mean_train_len = ckpt.get("mean_train_len", 303.0)

    # Load full corpus and get test split
    full_dataset = PythonCodeDataset(
        cfg.corpus_file,
        min_seq_len=cfg.min_seq_len,
        max_seq_len=cfg.max_seq_len_data,
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

    # Raw samples from test set
    test_samples = [full_dataset.samples[i] for i in test_set.indices]
    print(f"\nTest set: {len(test_samples)} files  (mean len={np.mean([len(s) for s in test_samples]):.0f})")
    print(f"Baseline mean length: {mean_train_len:.0f}")

    # Compute baseline accuracy at each fraction
    true_lens = np.array([len(s) for s in test_samples], dtype=float)
    baseline_acc = float((np.abs(mean_train_len - true_lens) / true_lens < 0.15).mean())

    print(f"\n{'Fraction':>10}  {'QuINN ±15%':>12}  {'Baseline ±15%':>14}  {'Beats':>6}  {'Median rel err':>16}")
    print("-" * 70)

    results = []
    prev_acc = None
    monotone = True

    for frac in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        r = evaluate_at_fraction(model, test_samples, frac, tok, device)
        r["baseline_acc_15pct"] = baseline_acc
        r["beats_baseline"] = r["acc_15pct"] > baseline_acc
        results.append(r)

        if prev_acc is not None and r["acc_15pct"] < prev_acc - 0.02:
            monotone = False
        prev_acc = r["acc_15pct"]

        print(
            f"  {frac:.0%}        "
            f"{r['acc_15pct']:.3f}         "
            f"{baseline_acc:.3f}           "
            f"{'✓' if r['beats_baseline'] else '✗'}    "
            f"{r['median_rel_err']:.3f}"
        )

    # Phase 1 verdict
    result_50 = next(r for r in results if r["fraction"] == 0.50)
    passes_50 = result_50["acc_15pct"] > baseline_acc
    overall = passes_50 and monotone

    print(f"\n── Phase 1 Success Criterion ────────────────────────────────────")
    print(f"  At 50% prefix: QuINN={result_50['acc_15pct']:.3f}  Baseline={baseline_acc:.3f}  → {'PASS ✓' if passes_50 else 'FAIL ✗'}")
    print(f"  Monotonically improving: {'PASS ✓' if monotone else 'FAIL ✗'}")
    print(f"\n  ══ Phase 1 overall: {'PASS ✓' if overall else 'FAIL'} ══\n")

    # Save results
    out = {
        "checkpoint": checkpoint_path,
        "baseline_acc_15pct": baseline_acc,
        "mean_train_len": mean_train_len,
        "results": results,
        "monotone": monotone,
        "phase1_pass": overall,
    }
    out_path = os.path.join(os.path.dirname(checkpoint_path), "phase1_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results → {out_path}")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    run_diagnostic(args.checkpoint)
