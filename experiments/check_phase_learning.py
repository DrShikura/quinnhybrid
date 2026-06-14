"""
Quick check: Are token-specific phase offsets actually being learned?

If A_imag values are mostly noise or uniform, then phase offsets aren't meaningful.
If A_imag varies by token type, then phase IS doing load-bearing work.
"""

import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from quinn.model import QuINN


def analyze_phase_learning(checkpoint_path):
    """
    Extract learned A_real and A_imag, compute phase offsets by token type.
    """
    print("\n" + "="*70)
    print("CHECKING: Are Token-Specific Phase Offsets Learned?")
    print("="*70)

    # Load checkpoint
    device = torch.device("cpu")
    tokenizer = PythonStructuralTokenizer()
    vocab_size = tokenizer.vocab_size

    model = QuINN(
        vocab_size=vocab_size,
        embed_dim=32,
        manifold_dim=64,
        n_encoder_layers=2,
        n_decoder_layers=1,
        max_seq_len=650
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    # Extract amplitude weights
    A_real = model.token_encoder.amplitude_real.weight.detach().numpy()  # (vocab_size, embed_dim)
    A_imag = model.token_encoder.amplitude_imag.weight.detach().numpy()  # (vocab_size, embed_dim)

    print(f"\nAmplitude shapes: A_real {A_real.shape}, A_imag {A_imag.shape}")

    # Compute phase offsets: atan2(A_imag, A_real)
    phases = np.arctan2(A_imag, A_real)  # (vocab_size, embed_dim) in [-π, π]

    print("\n" + "-"*70)
    print("PHASE OFFSET STATISTICS")
    print("-"*70)

    print(f"\nA_imag (imaginary amplitude component):")
    print(f"  Mean: {np.mean(A_imag):.6f}")
    print(f"  Std:  {np.std(A_imag):.6f}")
    print(f"  Min:  {np.min(A_imag):.6f}")
    print(f"  Max:  {np.max(A_imag):.6f}")

    print(f"\nPhase offsets (atan2(A_imag, A_real)):")
    print(f"  Mean: {np.mean(phases):.6f}")
    print(f"  Std:  {np.std(phases):.6f}")
    print(f"  Min:  {np.min(phases):.6f}")
    print(f"  Max:  {np.max(phases):.6f}")

    # Compare A_imag vs A_real magnitude
    A_mag = np.sqrt(A_real**2 + A_imag**2)
    A_real_contribution = A_real / (A_mag + 1e-8)
    A_imag_contribution = A_imag / (A_mag + 1e-8)

    print(f"\nA_real contribution to magnitude: mean = {np.mean(np.abs(A_real_contribution)):.4f}")
    print(f"A_imag contribution to magnitude: mean = {np.mean(np.abs(A_imag_contribution)):.4f}")

    # By token type
    print("\n" + "-"*70)
    print("PHASE OFFSETS BY TOKEN TYPE")
    print("-"*70)

    token_types = {
        "CONTROL_FLOW": ["def", "class", "return", "if", "for", "while", "try", "except"],
        "OPERATOR": [t for t in tokenizer.vocab if t.startswith("OP_")],
        "BRACKET": ["LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE"],
        "NAME": ["NAME"],
        "NUMBER": ["NUMBER"],
        "STRING": ["STRING"],
        "STRUCTURAL": ["INDENT", "DEDENT", "NEWLINE", "NL"],
    }

    type_stats = {}

    for type_name, type_tokens in token_types.items():
        # Find indices for this type
        indices = []
        for token_str in type_tokens:
            try:
                idx = tokenizer.vocab.index(token_str)
                indices.append(idx)
            except ValueError:
                pass

        if not indices:
            continue

        # Extract phases for these tokens
        type_phases = phases[indices]  # (n_tokens, embed_dim)
        type_A_imag = A_imag[indices]
        type_A_real = A_real[indices]

        # Statistics
        mean_phase = np.mean(type_phases)
        std_phase = np.std(type_phases)
        mean_A_imag = np.mean(type_A_imag)
        mean_A_real = np.mean(type_A_real)
        mean_mag = np.mean(A_mag[indices])

        type_stats[type_name] = {
            "count": len(indices),
            "mean_phase": mean_phase,
            "std_phase": std_phase,
            "mean_A_imag": mean_A_imag,
            "mean_A_real": mean_A_real,
            "mean_mag": mean_mag,
        }

        print(f"\n{type_name} ({len(indices)} tokens):")
        print(f"  Mean phase offset:     {mean_phase:7.4f} rad")
        print(f"  Std phase offset:      {std_phase:7.4f} rad")
        print(f"  Mean A_imag:           {mean_A_imag:7.5f}")
        print(f"  Mean A_real:           {mean_A_real:7.5f}")
        print(f"  Mean magnitude:        {mean_mag:7.5f}")
        print(f"  A_imag/Magnitude:      {mean_A_imag/mean_mag:7.4f}")

    # Key question: Does token type predict phase offset?
    print("\n" + "-"*70)
    print("KEY QUESTION: Does Token Type Predict Phase Offset?")
    print("-"*70)

    # Compute between-type variance vs within-type variance
    all_type_means = [s["mean_phase"] for s in type_stats.values()]
    all_type_counts = [s["count"] for s in type_stats.values()]

    global_mean = np.mean(phases)
    between_type_var = np.sum([count * (mean - global_mean)**2
                               for count, mean in zip(all_type_counts, all_type_means)]) / sum(all_type_counts)

    within_type_var = np.var(phases)  # Total variance

    ratio = between_type_var / (within_type_var + 1e-10)

    print(f"\nGlobal mean phase: {global_mean:.4f}")
    print(f"Between-type variance: {between_type_var:.6f}")
    print(f"Total variance:        {within_type_var:.6f}")
    print(f"Ratio (between/total): {ratio:.4f}")

    if ratio > 0.1:
        print("\n✓ SIGNIFICANT: Token type explains >10% of phase variance")
        print("  → Phase offsets ARE being meaningfully learned by token type")
        return "PHASE_LEARNED"
    elif ratio > 0.05:
        print("\n~ WEAK: Token type explains 5-10% of phase variance")
        print("  → Some structure, but mostly noise")
        return "WEAK_SIGNAL"
    else:
        print("\n✗ NOT SIGNIFICANT: Token type explains <5% of phase variance")
        print("  → Phase offsets are essentially random/noise")
        print("  → A_imag is NOT encoding token-specific phase information")
        return "NO_PHASE_LEARNING"


def compare_amplitude_vs_phase(checkpoint_path):
    """
    Compare: amplitude variation vs phase variation by token type.
    This directly answers: amplitude does heavy lifting, phase doesn't?
    """
    print("\n" + "="*70)
    print("COMPARISON: Amplitude vs Phase Load-Bearing")
    print("="*70)

    device = torch.device("cpu")
    tokenizer = PythonStructuralTokenizer()
    vocab_size = tokenizer.vocab_size

    model = QuINN(
        vocab_size=vocab_size,
        embed_dim=32,
        manifold_dim=64,
        n_encoder_layers=2,
        n_decoder_layers=1,
        max_seq_len=650
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    A_real = model.token_encoder.amplitude_real.weight.detach().numpy()
    A_imag = model.token_encoder.amplitude_imag.weight.detach().numpy()

    # Magnitude (what we measured in H2)
    mag = np.sqrt(A_real**2 + A_imag**2)
    phases = np.arctan2(A_imag, A_real)

    # Token type grouping
    token_groups = {
        "CONTROL_FLOW": ["def", "class", "return", "if", "for", "while", "try", "except"],
        "OPERATOR": [t for t in tokenizer.vocab if t.startswith("OP_")],
    }

    print(f"\nMagnitude (amplitude) variance by token type:")
    cf_idx = [tokenizer.vocab.index(t) for t in token_groups["CONTROL_FLOW"] if t in tokenizer.vocab]
    op_idx = [tokenizer.vocab.index(t) for t in token_groups["OPERATOR"] if t in tokenizer.vocab]

    cf_mag = np.mean(mag[cf_idx])
    op_mag = np.mean(mag[op_idx])
    mag_diff = abs(cf_mag - op_mag) / max(cf_mag, op_mag)

    print(f"  CONTROL_FLOW mean mag: {cf_mag:.5f}")
    print(f"  OPERATOR mean mag:     {op_mag:.5f}")
    print(f"  Difference: {mag_diff:.1%}")

    print(f"\nPhase offset variance by token type:")
    cf_phase = np.mean(phases[cf_idx])
    op_phase = np.mean(phases[op_idx])
    phase_diff = abs(cf_phase - op_phase) / np.pi

    print(f"  CONTROL_FLOW mean phase: {cf_phase:.5f} rad")
    print(f"  OPERATOR mean phase:     {op_phase:.5f} rad")
    print(f"  Difference: {phase_diff:.1%} of π")

    print(f"\nConclusion:")
    print(f"  Amplitude distinguishes token types: {mag_diff:.1%}")
    print(f"  Phase distinguishes token types:    {phase_diff:.1%}")

    if mag_diff > phase_diff * 5:
        print(f"\n  → AMPLITUDE does {mag_diff/phase_diff:.1f}× more work than PHASE")
        print("  → Phase is NOT the load-bearing mechanism")
        return "AMPLITUDE_DOMINANT"
    else:
        print(f"\n  → Phase and amplitude comparable")
        return "BALANCED"


if __name__ == "__main__":
    checkpoint_path = "checkpoints/run7/quinn_epoch050.pt"

    if not Path(checkpoint_path).exists():
        print(f"ERROR: {checkpoint_path} not found")
        exit(1)

    result1 = analyze_phase_learning(checkpoint_path)
    result2 = compare_amplitude_vs_phase(checkpoint_path)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Phase learning: {result1}")
    print(f"Amplitude vs Phase: {result2}")

    if result1 == "NO_PHASE_LEARNING":
        print("\n→ CONCLUSION: Phase is NOT being learned by token type")
        print("  A_imag is essentially noise, not encoding token structure")
        print("  The sinusoidal representation is overkill for what Quinn learned")
    elif result1 == "WEAK_SIGNAL":
        print("\n→ CONCLUSION: Phase has weak signal")
        print("  Some learning but amplitude dominates")
    else:
        print("\n→ CONCLUSION: Phase IS being learned meaningfully")
        print("  Token type predicts phase offset")
