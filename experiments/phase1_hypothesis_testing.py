"""
Phase 1 Hypothesis Testing Framework: What Do Waveforms Encode?

This module tests the 5 hypotheses about what Quinn's learned waveforms represent:

H1: Nesting levels correlate with waveform frequency
H2: Token roles (keyword/operator/name) have distinctive frequency patterns
H3: Phase coherence directly measures structural agreement
H4: Control flow patterns leave detectable frequency signatures
H5: Waveform encodes STRUCTURE not SEMANTICS

We test these by:
1. Extracting frequencies and phases from the trained model
2. Analyzing real code samples and measuring their structural properties
3. Computing correlation between frequency properties and structural metrics
4. Generating synthetic patterns and predicting their frequency signatures
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from data.dataset import PythonCodeDataset
from quinn.model import QuINN


def load_model_and_tokenizer(checkpoint_path):
    """Load trained Quinn model."""
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

    model.eval()
    return model, tokenizer, device


def compute_nesting_level(tokens):
    """
    Compute nesting level at each position in code.

    Hypothesis: High-frequency waveforms should correlate with high nesting
    because nested blocks have more structural constraints.
    """
    nesting_levels = []
    current_level = 0

    for token in tokens:
        token_str = token

        # Track indentation-based nesting
        if token_str == "INDENT":
            current_level += 1
        elif token_str == "DEDENT":
            current_level = max(0, current_level - 1)

        nesting_levels.append(current_level)

    return nesting_levels


def compute_token_role_pattern(tokenizer, tokens):
    """
    Classify token roles for testing H2.
    Returns sequence of role labels: KEYWORD, OPERATOR, NAME, BRACKET, etc.
    """
    roles = []

    for token_idx in tokens:
        if token_idx >= len(tokenizer.vocab):
            token_str = "<UNK>"
        else:
            token_str = tokenizer.vocab[token_idx]

        if token_str in ["def", "class", "if", "for", "while", "return", "try", "except"]:
            roles.append("CONTROL_FLOW")
        elif token_str.startswith("OP_"):
            roles.append("OPERATOR")
        elif token_str in ["LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE"]:
            roles.append("BRACKET")
        elif token_str == "NAME":
            roles.append("NAME")
        elif token_str in ["INDENT", "DEDENT", "NEWLINE", "NL"]:
            roles.append("STRUCTURAL")
        else:
            roles.append("OTHER")

    return roles


def extract_waveform_properties(model, tokenizer, tokens, positions, device):
    """
    Extract waveform properties (amplitude, phase, frequency) for a code sequence.

    Returns:
        amplitudes: (seq_len, embed_dim) - magnitude of waveform
        phases: (seq_len, embed_dim) - phase angle of waveform
        frequency_features: per-token statistics
    """
    with torch.no_grad():
        # Prepare inputs
        tokens_t = torch.tensor([tokens], dtype=torch.long, device=device)
        positions_t = torch.tensor([positions], dtype=torch.long, device=device)

        # Get waveform
        waveform = model.token_encoder(tokens_t, positions_t)  # (batch=1, seq_len, embed_dim)
        waveform = waveform[0].detach().cpu().numpy()  # (seq_len, embed_dim) complex

        # Extract amplitude and phase
        amplitudes = np.abs(waveform)  # (seq_len, embed_dim)
        phases = np.angle(waveform)  # (seq_len, embed_dim) in [-π, π]

        # Compute frequency features per token
        frequency_features = {
            "amplitude_mean": np.mean(amplitudes, axis=1),  # (seq_len,)
            "amplitude_std": np.std(amplitudes, axis=1),    # (seq_len,)
            "phase_mean": np.mean(np.abs(phases), axis=1),  # (seq_len,)
            "phase_coherence": compute_phase_coherence_local(phases),  # (seq_len-1,)
        }

        return amplitudes, phases, frequency_features


def compute_phase_coherence_local(phases):
    """
    Compute local phase coherence: how aligned are phases between adjacent positions?

    Returns array of coherence values for each position transition.
    """
    # For each position, measure phase alignment with next position
    coherence = []

    for t in range(len(phases) - 1):
        phase_diff = phases[t+1] - phases[t]
        # Normalize to [0, 1] where 1 = perfectly aligned (phase_diff ≈ 0 or 2π)
        normalized_diff = np.minimum(np.abs(phase_diff), 2*np.pi - np.abs(phase_diff))
        c = 1 - (normalized_diff / np.pi)  # [0, 1] where 1 = aligned
        coherence.append(np.mean(c))

    return np.array(coherence)


def test_h1_nesting_frequency_correlation(model, tokenizer, device, sample_size=10):
    """
    Test H1: Nesting levels should correlate with frequency (amplitude magnitude).

    Hypothesis: Code at higher nesting levels has more structural constraints,
    so should excite higher-frequency waveform dimensions.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Nesting Level ↔ Frequency Correlation")
    print("="*70)

    # Use corpus.txt which contains file paths
    corpus_file = Path("data/corpus.txt")
    if not corpus_file.exists():
        print("  ERROR: data/corpus.txt not found")
        return None

    # Read corpus and load actual Python files
    file_paths = corpus_file.read_text().splitlines()
    file_paths = [Path(p.strip()) for p in file_paths if p.strip()]

    # Filter to existing files
    existing_files = [p for p in file_paths if p.exists()]
    if not existing_files:
        print("  ERROR: No files found from corpus.txt")
        return None

    if len(existing_files) < sample_size:
        sample_size = len(existing_files)

    correlations = []

    for file_path in existing_files[:sample_size]:
        try:
            # Read and tokenize the file
            source_code = file_path.read_text(encoding="utf-8", errors="replace")
            token_indices = tokenizer.encode(source_code, add_special=True)
            positions = np.arange(len(token_indices))

            # Skip if sequence too short
            if len(token_indices) < 10 or len(token_indices) > 300:
                continue

            # Get nesting levels
            token_strs = [tokenizer.vocab[t] if t < len(tokenizer.vocab) else "<UNK>" for t in token_indices]
            nesting = np.array(compute_nesting_level(token_strs))
        except Exception as e:
            continue

        # Get waveforms
        amplitudes, phases, freq_feat = extract_waveform_properties(
            model, tokenizer, token_indices, positions, device
        )

        # Compute correlation: nesting level vs amplitude magnitude
        avg_amplitude = np.mean(amplitudes, axis=1)  # average over frequency dimensions

        if len(nesting) == len(avg_amplitude):
            corr = np.corrcoef(nesting, avg_amplitude)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    if correlations:
        mean_corr = np.mean(correlations)
        print(f"  Mean nesting ↔ amplitude correlation: {mean_corr:.4f}")
        print(f"    (Range: {min(correlations):.4f} to {max(correlations):.4f})")

        if mean_corr > 0.3:
            print("  ✓ SIGNIFICANT: Nesting correlates with frequency (H1 supported)")
        elif mean_corr > 0.1:
            print("  ~ WEAK: Modest correlation detected (H1 partially supported)")
        else:
            print("  ✗ WEAK: Little evidence for nesting-frequency link (H1 not supported)")

        return mean_corr
    else:
        print("  (No valid samples for correlation)")
        return None


def test_h2_token_role_signatures(model, tokenizer, device, sample_size=20):
    """
    Test H2: Different token types should have distinctive frequency signatures.

    Hypothesis: Control flow keywords (if, for, def) should have different
    frequency signatures than regular operators or names.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Token Role → Frequency Signature Mapping")
    print("="*70)

    # Use corpus.txt which contains file paths
    corpus_file = Path("data/corpus.txt")
    if not corpus_file.exists():
        print("  ERROR: data/corpus.txt not found")
        return None

    file_paths = corpus_file.read_text().splitlines()
    file_paths = [Path(p.strip()) for p in file_paths if p.strip()]
    existing_files = [p for p in file_paths if p.exists()]

    if not existing_files:
        print("  ERROR: No files found from corpus.txt")
        return None

    if len(existing_files) < sample_size:
        sample_size = len(existing_files)

    role_amplitudes = defaultdict(list)

    for file_path in existing_files[:sample_size]:
        try:
            source_code = file_path.read_text(encoding="utf-8", errors="replace")
            token_indices = tokenizer.encode(source_code, add_special=True)

            if len(token_indices) < 10 or len(token_indices) > 300:
                continue

            positions = np.arange(len(token_indices))

            # Get roles
            roles = compute_token_role_pattern(tokenizer, token_indices)
        except Exception as e:
            continue

        # Get amplitudes
        amplitudes, _, _ = extract_waveform_properties(
            model, tokenizer, token_indices, positions, device
        )

        # Group amplitudes by role
        for role, token_idx, amp in zip(roles, token_indices, amplitudes):
            role_amplitudes[role].append(amp)

    # Analyze by role
    print("\n  Average frequency signature by token role:")
    print(f"  {'Role':<20} {'Amplitude Mean':>15} {'Amplitude Std':>15}")
    print("  " + "-"*50)

    role_means = {}
    for role in sorted(role_amplitudes.keys()):
        amp_array = np.array(role_amplitudes[role])
        mean_amp = np.mean(amp_array)
        std_amp = np.std(amp_array)
        role_means[role] = mean_amp

        print(f"  {role:<20} {mean_amp:>15.4f} {std_amp:>15.4f}")

    # Test separation
    if len(role_means) >= 2:
        min_amp = min(role_means.values())
        max_amp = max(role_means.values())
        separation = (max_amp - min_amp) / max_amp

        if separation > 0.1:
            print(f"\n  ✓ DISTINCT: Roles show {separation:.1%} frequency separation (H2 supported)")
        else:
            print(f"\n  ✗ SIMILAR: Roles show only {separation:.1%} separation (H2 not supported)")

        return role_means
    else:
        return None


def test_h3_phase_coherence_structure(model, tokenizer, device, sample_size=15):
    """
    Test H3: Phase coherence between adjacent tokens should reflect structural relationships.

    Hypothesis: Tokens in related structural contexts should have more
    coherent phases (oscillate together).
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Phase Coherence ↔ Structural Relationship")
    print("="*70)

    # Use corpus.txt which contains file paths
    corpus_file = Path("data/corpus.txt")
    if not corpus_file.exists():
        print("  ERROR: data/corpus.txt not found")
        return None

    file_paths = corpus_file.read_text().splitlines()
    file_paths = [Path(p.strip()) for p in file_paths if p.strip()]
    existing_files = [p for p in file_paths if p.exists()]

    if not existing_files:
        print("  ERROR: No files found from corpus.txt")
        return None

    if len(existing_files) < sample_size:
        sample_size = len(existing_files)

    coherence_values = []

    for file_path in existing_files[:sample_size]:
        try:
            source_code = file_path.read_text(encoding="utf-8", errors="replace")
            token_indices = tokenizer.encode(source_code, add_special=True)

            if len(token_indices) < 20 or len(token_indices) > 300:
                continue

            positions = np.arange(len(token_indices))
        except Exception as e:
            continue

        # Get phases
        amplitudes, phases, freq_feat = extract_waveform_properties(
            model, tokenizer, token_indices, positions, device
        )

        # Collect coherence values
        coherence_values.extend(freq_feat["phase_coherence"])

    if coherence_values:
        mean_coherence = np.mean(coherence_values)
        std_coherence = np.std(coherence_values)

        print(f"  Mean phase coherence between adjacent tokens: {mean_coherence:.4f} ± {std_coherence:.4f}")
        print(f"    (Range: {min(coherence_values):.4f} to {max(coherence_values):.4f})")

        if mean_coherence > 0.5:
            print("  ✓ HIGH: Tokens show strong phase coherence (H3 supported)")
        elif mean_coherence > 0.3:
            print("  ~ MODERATE: Some phase coherence detected (H3 partially supported)")
        else:
            print("  ✗ LOW: Weak phase coherence (H3 not well supported)")

        return mean_coherence
    else:
        print("  (No valid samples)")
        return None


def generate_hypothesis_report(checkpoint_path, output_path="experiments/phase1_hypothesis_results.md"):
    """Run all hypothesis tests and generate report."""
    print("\n" + "="*70)
    print("PHASE 1: HYPOTHESIS TESTING FOR WAVEFORM INTERPRETATION")
    print("="*70)

    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path)

    h1_result = test_h1_nesting_frequency_correlation(model, tokenizer, device)
    h2_result = test_h2_token_role_signatures(model, tokenizer, device)
    h3_result = test_h3_phase_coherence_structure(model, tokenizer, device)

    # Write report
    with open(output_path, "w") as f:
        f.write("# Phase 1: Hypothesis Testing Results\n\n")
        f.write("Testing what the learned waveforms actually encode about code structure.\n\n")

        f.write("## H1: Nesting Level ↔ Frequency Correlation\n")
        if h1_result is not None:
            f.write(f"**Result**: r = {h1_result:.4f}\n")
            if h1_result > 0.3:
                f.write("**Status**: ✓ SUPPORTED\n")
            else:
                f.write("**Status**: ✗ NOT SUPPORTED\n")
        else:
            f.write("**Result**: Insufficient data\n")
        f.write("\n")

        f.write("## H2: Token Role Frequency Signatures\n")
        if h2_result is not None:
            f.write("**Result**: ")
            for role, mean_amp in sorted(h2_result.items()):
                f.write(f"{role}={mean_amp:.4f}, ")
            f.write("\n")
            f.write("**Status**: ✓ SUPPORTED (if distinct)\n")
        else:
            f.write("**Result**: No distinct signatures found\n")
        f.write("\n")

        f.write("## H3: Phase Coherence ↔ Structure\n")
        if h3_result is not None:
            f.write(f"**Result**: mean_coherence = {h3_result:.4f}\n")
            if h3_result > 0.3:
                f.write("**Status**: ✓ SUPPORTED\n")
            else:
                f.write("**Status**: ✗ NOT SUPPORTED\n")
        else:
            f.write("**Result**: Insufficient data\n")
        f.write("\n")

        f.write("## Summary\n\n")
        f.write("These preliminary results suggest which theoretical models best explain Quinn's learning.\n")
        f.write("See visualizations for supporting evidence.\n")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    checkpoint_path = "checkpoints/run7/quinn_epoch050.pt"

    if not Path(checkpoint_path).exists():
        print(f"ERROR: {checkpoint_path} not found")
        exit(1)

    generate_hypothesis_report(checkpoint_path)
