"""
Phase 1 Waveform Analysis: Interpret learned representations as "brain waves"

Extracts and visualizes waveforms from Phase 1 checkpoint to understand:
1. What frequencies does Quinn learn? Are they related to code structure?
2. Do different token types have distinctive frequency signatures?
3. Does phase coherence correlate with structural relationships?
4. Can we visualize model thinking as frequency patterns?

This implements the "brain waves" theory of model interpretability.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer import PythonStructuralTokenizer
from quinn.model import QuINN


def load_quinn_checkpoint(checkpoint_path, tokenizer):
    """Load trained Quinn model from checkpoint."""
    device = torch.device("cpu")

    # Create model with correct vocab size
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
    return model, device


def extract_waveforms(model, tokenizer_obj, max_tokens=100):
    """
    Extract waveforms for all tokens to understand frequency signatures.

    Returns:
        waveforms: dict[token_idx] -> (L, 32) complex array for sequence length L=100
        amplitudes: dict[token_idx] -> (embed_dim,) learned amplitudes
        frequencies: (embed_dim,) learned frequency parameters
    """
    with torch.no_grad():
        # Get embedding layer
        embed = model.token_encoder

        # Create a full sequence of each token to see its learned waveform
        vocab_size = tokenizer_obj.vocab_size

        waveforms = {}
        amplitudes = {}

        print(f"Extracting waveforms for {vocab_size} tokens...")

        for token_idx in range(vocab_size):
            # Create a dummy sequence filled with this token
            tokens = torch.full((1, max_tokens), token_idx, dtype=torch.long)
            positions = torch.arange(max_tokens, dtype=torch.long).unsqueeze(0)  # (1, max_tokens)

            # Get waveform (will have learned amplitudes and frequencies)
            waveform = embed(tokens, positions)  # (batch=1, seq_len, embed_dim=32)

            # Store complex waveform and extract amplitude
            waveforms[token_idx] = waveform[0].detach().numpy()  # (max_tokens, 32) complex

            # Extract learned amplitude (magnitude of first position)
            # Amplitude is stored as separate real/imaginary in the model
            # We can recover it from the waveform itself
            first_pos = waveform[0, 0]  # (embed_dim,) at position 0
            amplitude_mag = torch.abs(first_pos).detach().numpy()
            amplitudes[token_idx] = amplitude_mag

        # Extract learned frequencies from embedding layer
        # These are in the position-modulation term: exp(i * f * p / max_len * 2π)
        # For now we'll estimate from phase progression across positions
        frequencies = extract_frequencies(waveforms, max_tokens)

    return waveforms, amplitudes, frequencies


def extract_frequencies(waveforms, max_tokens):
    """
    Estimate learned frequencies by analyzing phase progression across sequence.

    For each frequency dimension, the phase should progress as:
    phase(p) = 2π * f * p / max_len

    We can estimate f from the phase difference between positions.
    """
    frequencies = []
    embed_dim = waveforms[0].shape[1]

    # Use token 0 as reference to extract frequency progression
    reference_waveform = waveforms[0]  # (max_tokens, embed_dim) complex

    for d in range(embed_dim):
        column = reference_waveform[:, d]  # (max_tokens,) complex values

        # Extract phase: angle tells us position in oscillation
        phases = np.angle(column)

        # Unwrap phase to account for 2π wrapping
        phases_unwrapped = np.unwrap(phases)

        # Linear fit: phase = a*position + b
        positions = np.arange(len(phases))
        coeffs = np.polyfit(positions, phases_unwrapped, 1)
        slope = coeffs[0]  # rate of phase change

        # Convert slope back to frequency
        # phase = 2π * f * p / max_len => slope = 2π * f / max_len
        # => f = slope * max_len / (2π)
        f = slope * max_tokens / (2 * np.pi)
        frequencies.append(f)

    return np.array(frequencies)


def analyze_token_types(waveforms, amplitudes, tokenizer):
    """
    Analyze whether different token types have distinctive frequency signatures.

    Token types: NAME, KEYWORD, NUMBER, STRING, OPERATOR, PUNCTUATION, etc.
    """
    # Group tokens by type
    token_types = defaultdict(list)

    for token_idx, token_str in enumerate(tokenizer.vocab):
        if not token_str.strip() or token_str == "<PAD>":
            token_types["[SPECIAL]"].append(token_idx)
        elif token_str.startswith("<") or token_str in ["[CLS]", "[SEP]", "[MASK]", "[UNK]"]:
            token_types["[SPECIAL]"].append(token_idx)
        elif token_str.startswith("OP_"):
            token_types["OPERATOR"].append(token_idx)
        elif token_str in ["LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "LBRACE", "RBRACE", "COLON", "COMMA", "DOT"]:
            token_types["BRACKET/DELIMITER"].append(token_idx)
        elif token_str == "NAME":
            token_types["NAME"].append(token_idx)
        elif token_str == "NUMBER":
            token_types["NUMBER"].append(token_idx)
        elif token_str == "STRING":
            token_types["STRING"].append(token_idx)
        elif token_str in ["INDENT", "DEDENT", "NEWLINE", "NL", "ENDMARKER"]:
            token_types["STRUCTURAL"].append(token_idx)
        else:
            # Other tokens are keywords
            token_types["KEYWORD"].append(token_idx)

    # Analyze frequency signatures per type
    type_signatures = {}

    for token_type, token_indices in token_types.items():
        if not token_indices:
            continue

        # Collect amplitude profiles for this type
        amp_profiles = [amplitudes[idx] for idx in token_indices]
        amp_mean = np.mean(amp_profiles, axis=0)
        amp_std = np.std(amp_profiles, axis=0)

        # Spectral centroid: weighted average of frequencies
        spectral_centroid = np.sum(amp_mean * np.arange(len(amp_mean))) / np.sum(amp_mean) if np.sum(amp_mean) > 0 else 0

        type_signatures[token_type] = {
            "count": len(token_indices),
            "amp_mean": amp_mean,
            "amp_std": amp_std,
            "spectral_centroid": spectral_centroid,
            "examples": [tokenizer.vocab[idx] for idx in token_indices[:3]],
        }

    return token_types, type_signatures


def compute_phase_coherence(waveforms, tokenizer):
    """
    Compute phase coherence between tokens as measure of structural agreement.

    Hypothesis: Tokens that often appear together in structure should have
    high phase coherence (their waveforms oscillate in sync).
    """
    # Get phases for each token at first few positions
    phases = {}

    for token_idx, waveform in waveforms.items():
        if token_idx >= len(tokenizer.idx_to_token):
            continue

        # Use first position's phase as token signature
        first_pos = waveform[0]  # (embed_dim,) complex
        token_phase = np.angle(first_pos)
        phases[token_idx] = token_phase

    # Compute pairwise phase differences
    phase_coherence = {}
    token_indices = list(phases.keys())

    for i, idx1 in enumerate(token_indices[:20]):  # Sample for efficiency
        for idx2 in token_indices[i+1:20]:
            phase_diff = np.abs(phases[idx1] - phases[idx2])
            # Normalize to [0, 1] where 0 = perfectly aligned, 1 = opposite
            coherence = 1 - (np.minimum(phase_diff, 2*np.pi - phase_diff) / np.pi)

            token1 = tokenizer.vocab[idx1] if idx1 < len(tokenizer.vocab) else "?"
            token2 = tokenizer.vocab[idx2] if idx2 < len(tokenizer.vocab) else "?"
            phase_coherence[(token1, token2)] = coherence.mean()

    return phase_coherence


def visualize_amplitude_spectrum(amplitudes, tokenizer, output_path="phase1_amplitude_spectrum.png"):
    """Visualize learned amplitude spectrum."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Top 20 tokens by average amplitude
    amp_averages = {idx: np.mean(amp) for idx, amp in amplitudes.items()}
    top_tokens = sorted(amp_averages.items(), key=lambda x: x[1], reverse=True)[:20]

    ax = axes[0, 0]
    tokens = [tokenizer.vocab[idx] if idx < len(tokenizer.vocab) else f"T{idx}" for idx, _ in top_tokens]
    values = [v for _, v in top_tokens]
    ax.barh(tokens, values, color="steelblue")
    ax.set_xlabel("Average Amplitude")
    ax.set_title("Top 20 Tokens by Amplitude")
    ax.invert_yaxis()

    # Plot 2: Amplitude histogram
    ax = axes[0, 1]
    all_amplitudes = []
    for amp_array in amplitudes.values():
        all_amplitudes.extend(amp_array)
    ax.hist(all_amplitudes, bins=50, color="coral", edgecolor="black")
    ax.set_xlabel("Amplitude Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Learned Amplitudes")

    # Plot 3: Embedding dimension variance
    ax = axes[1, 0]
    dim_variance = np.var([amp for amp in amplitudes.values()], axis=0)
    ax.plot(dim_variance, marker="o", linestyle="-", color="green", linewidth=2)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Amplitude Variance")
    ax.set_title("Per-Dimension Variance Across Tokens")
    ax.grid(True, alpha=0.3)

    # Plot 4: Token type signatures
    ax = axes[1, 1]
    token_types, type_sigs = analyze_token_types(amplitudes, amplitudes, tokenizer)

    type_names = []
    spectral_centroids = []
    for ttype, sig in sorted(type_sigs.items()):
        type_names.append(f"{ttype}\n(n={sig['count']})")
        spectral_centroids.append(sig["spectral_centroid"])

    ax.bar(range(len(type_names)), spectral_centroids, color="purple", alpha=0.7)
    ax.set_xticks(range(len(type_names)))
    ax.set_xticklabels(type_names, rotation=45, ha="right")
    ax.set_ylabel("Spectral Centroid")
    ax.set_title("Token Type Frequency Signatures")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def visualize_phase_evolution(waveforms, tokenizer, sample_tokens=None, output_path="phase1_phase_evolution.png"):
    """Visualize how phase evolves across sequence positions."""
    if sample_tokens is None:
        # Get indices of key tokens from vocab
        sample_tokens = []
        for token in ["def", "return", "if", "NAME"]:
            try:
                idx = tokenizer.vocab.index(token)
                sample_tokens.append(idx)
            except ValueError:
                sample_tokens.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for plot_idx, token_idx in enumerate(sample_tokens):
        if token_idx >= len(waveforms):
            continue

        ax = axes[plot_idx]
        waveform = waveforms[token_idx]  # (seq_len, embed_dim) complex

        # Extract phase across positions for first frequency dimension
        phases = np.angle(waveform[:, 0])
        phases_unwrapped = np.unwrap(phases)

        ax.plot(phases_unwrapped, marker=".", linestyle="-", linewidth=2, markersize=4)
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Phase (unwrapped)")
        token_name = tokenizer.vocab[token_idx] if token_idx < len(tokenizer.vocab) else f"Token {token_idx}"
        ax.set_title(f"Phase Evolution: {token_name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def visualize_waveform_heatmap(waveforms, tokenizer, sample_tokens=None, output_path="phase1_waveform_heatmap.png"):
    """Visualize waveform magnitude as heatmap (position × frequency)."""
    if sample_tokens is None:
        # Get indices of key tokens from vocab
        sample_tokens = []
        for token in ["def", "return", "if", "NAME"]:
            try:
                idx = tokenizer.vocab.index(token)
                sample_tokens.append(idx)
            except ValueError:
                sample_tokens.append(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for plot_idx, token_idx in enumerate(sample_tokens):
        if token_idx >= len(waveforms):
            continue

        ax = axes[plot_idx]
        waveform = waveforms[token_idx]  # (seq_len=100, embed_dim=32) complex

        # Convert to magnitude
        magnitude = np.abs(waveform)  # (100, 32)

        im = ax.imshow(magnitude.T, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Frequency Dimension")
        token_name = tokenizer.vocab[token_idx] if token_idx < len(tokenizer.vocab) else f"Token {token_idx}"
        ax.set_title(f"Waveform Magnitude: {token_name}")
        plt.colorbar(im, ax=ax, label="Magnitude")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def generate_interpretability_report(checkpoint_path, output_dir="experiments/phase1_analysis"):
    """Generate comprehensive waveform analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PHASE 1 WAVEFORM ANALYSIS: 'Brain Waves' Interpretability")
    print("="*70)

    # Load model and tokenizer
    print("\n[1/5] Loading Quinn checkpoint...")
    tokenizer = PythonStructuralTokenizer()
    model, device = load_quinn_checkpoint(checkpoint_path, tokenizer)

    # Extract waveforms
    print("[2/5] Extracting waveforms from token embeddings...")
    waveforms, amplitudes, frequencies = extract_waveforms(model, tokenizer)

    # Analyze token types
    print("[3/5] Analyzing token type signatures...")
    token_types, type_signatures = analyze_token_types(waveforms, amplitudes, tokenizer)

    # Generate visualizations
    print("[4/5] Generating visualizations...")
    visualize_amplitude_spectrum(amplitudes, tokenizer, output_path=str(output_dir / "amplitude_spectrum.png"))
    visualize_phase_evolution(waveforms, tokenizer, output_path=str(output_dir / "phase_evolution.png"))
    visualize_waveform_heatmap(waveforms, tokenizer, output_path=str(output_dir / "waveform_heatmap.png"))

    # Generate report
    print("[5/5] Writing interpretability report...")
    report_path = output_dir / "INTERPRETABILITY_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# Phase 1 Waveform Analysis: Quinn as 'Brain Waves'\n\n")
        f.write(f"**Analysis Date**: {Path(checkpoint_path).stat().st_mtime}\n")
        f.write(f"**Checkpoint**: {checkpoint_path}\n\n")

        f.write("## Learned Frequency Structure\n\n")
        f.write(f"Estimated frequencies across 32 embedding dimensions:\n")
        f.write("```\n")
        for d, f_val in enumerate(frequencies):
            f.write(f"  Dimension {d:2d}: f = {f_val:7.3f}\n")
        f.write("```\n\n")

        f.write("## Token Type Frequency Signatures\n\n")
        f.write(f"| Token Type | Count | Spectral Centroid | Amplitude Mean | Amplitude Std | Examples |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for ttype, sig in sorted(type_signatures.items()):
            amp_mean = np.mean(sig["amp_mean"])
            amp_std = np.mean(sig["amp_std"])
            examples = ", ".join(sig["examples"])
            f.write(f"| {ttype} | {sig['count']} | {sig['spectral_centroid']:.2f} | {amp_mean:.4f} | {amp_std:.4f} | {examples} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("### Hypothesis 1: Nesting Levels Correlate with Frequency\n")
        f.write("- KEYWORD tokens (def, if, for, etc.) should show distinctive frequencies\n")
        f.write("- Result: [See spectral centroid column above]\n\n")

        f.write("### Hypothesis 2: Token Roles Have Distinctive Patterns\n")
        f.write("- NAME tokens (variables) vs KEYWORD vs OPERATOR should separate\n")
        f.write("- Result: [See amplitude profiles]\n\n")

        f.write("### Hypothesis 3: Phase Coherence Measures Structural Agreement\n")
        f.write("- Tokens that co-occur structurally should have aligned phases\n")
        f.write("- Example: 'if' and ':' should have high coherence\n\n")

        f.write("### Next Steps\n")
        f.write("1. Compare frequency signatures with code structure metrics\n")
        f.write("2. Validate that high-frequency dimensions track fine structure (tokens)\n")
        f.write("3. Validate that low-frequency dimensions track broad structure (nesting)\n")
        f.write("4. Use phase as interpretability signal for model attention\n")

    print(f"\nAnalysis complete! Report saved to: {report_path}\n")
    print("Generated files:")
    for f in sorted(output_dir.glob("*.png")) + [report_path]:
        print(f"  - {f}")

    return output_dir


if __name__ == "__main__":
    checkpoint_path = "checkpoints/run7/quinn_epoch050.pt"

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure Phase 1 training completed successfully.")
        exit(1)

    output_dir = generate_interpretability_report(checkpoint_path)
