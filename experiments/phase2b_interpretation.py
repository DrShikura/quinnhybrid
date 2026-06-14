"""
Phase 2B Interpretation: Using Option C insights to understand training results.

Analyzes Phase 2B training histories using the theoretical framework from Option C:
- Phase coherence analysis (H3 validation)
- Token role frequency signatures (H2 validation)
- Learning dynamics (does Quinn guidance improve early convergence?)
- Effectiveness of cross-attention mechanism

Produces interpretability-focused report explaining WHY Quinn helps (or doesn't).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_history(path):
    """Load training history from JSON checkpoint."""
    with open(path) as f:
        return json.load(f)


def analyze_convergence_dynamics(with_quinn_hist, baseline_hist):
    """
    Analyze convergence rates using Option C insights.

    Quinn should help by:
    1. Lower initial loss (better structured representations)
    2. Faster improvement in first 10 epochs (good guidance signal)
    3. Better final loss (constraints from structure help)
    """
    print("\n" + "="*70)
    print("CONVERGENCE DYNAMICS: Do Quinn's structured representations help?")
    print("="*70)

    # Extract validation losses
    quinn_val = [e["val"]["lm_loss"] for e in with_quinn_hist]
    baseline_val = [e["val"]["lm_loss"] for e in baseline_hist]

    # Epoch 1 comparison
    epoch1_quinn = quinn_val[0]
    epoch1_baseline = baseline_val[0]
    epoch1_diff = ((epoch1_baseline - epoch1_quinn) / epoch1_baseline) * 100

    print(f"\nEpoch 1 Validation Loss:")
    print(f"  WITH Quinn:   {epoch1_quinn:.4f}")
    print(f"  Baseline:     {epoch1_baseline:.4f}")
    print(f"  Advantage:    {epoch1_diff:.1f}%")

    if epoch1_diff > 5:
        print("  → STRONG START: Quinn's pre-trained structure helps immediately")
    elif epoch1_diff > 0:
        print("  → MODEST START: Slight advantage from Quinn")
    else:
        print("  → NO ADVANTAGE: Both start similarly (Quinn not being used effectively?)")

    # First 10 epochs improvement
    if len(quinn_val) >= 10 and len(baseline_val) >= 10:
        quinn_improvement = quinn_val[0] - quinn_val[9]
        baseline_improvement = baseline_val[0] - baseline_val[9]
        improvement_ratio = quinn_improvement / baseline_improvement if baseline_improvement > 0 else 1

        print(f"\nConvergence Rate (Epochs 1-10):")
        print(f"  WITH Quinn improvement:   {quinn_improvement:.4f}")
        print(f"  Baseline improvement:     {baseline_improvement:.4f}")
        print(f"  Ratio (Quinn/Baseline):   {improvement_ratio:.2f}x")

        if improvement_ratio > 1.2:
            print("  → FASTER CONVERGENCE: Quinn's guidance accelerates learning")
        elif improvement_ratio > 0.9:
            print("  → SIMILAR RATE: Both converge at same pace")
        else:
            print("  → SLOWER START: Quinn might hurt early (wrong guidance?)")

    # Final loss
    quinn_final = quinn_val[-1]
    baseline_final = baseline_val[-1]
    final_diff = ((baseline_final - quinn_final) / baseline_final) * 100

    print(f"\nFinal Validation Loss (Epoch {len(quinn_val)}):")
    print(f"  WITH Quinn:   {quinn_final:.4f}")
    print(f"  Baseline:     {baseline_final:.4f}")
    print(f"  Advantage:    {final_diff:.1f}%")

    if final_diff > 10:
        print("  → STRONG: Quinn guidance provides major improvement")
        return "STRONG_HELP"
    elif final_diff > 5:
        print("  → MODERATE: Quinn guidance provides meaningful improvement")
        return "MODERATE_HELP"
    elif final_diff > 2:
        print("  → MARGINAL: Quinn guidance provides small improvement")
        return "MARGINAL_HELP"
    elif final_diff > -2:
        print("  → NEUTRAL: Quinn doesn't help or hurt")
        return "NEUTRAL"
    else:
        print("  → HURTS: Quinn guidance makes things worse (unexpected!)")
        return "HURTS"


def interpret_result(result_type, with_quinn_hist, baseline_hist):
    """
    Interpret Phase 2B results using Option C framework.

    Return strategic recommendations based on result type.
    """
    print("\n" + "="*70)
    print("INTERPRETATION: What does this result mean?")
    print("="*70)

    if result_type == "STRONG_HELP":
        print("\n✓ QUINN'S STRUCTURAL GUIDANCE IS EFFECTIVE")
        print("\nWhat this means:")
        print("  - Cross-attention to waveform helps Transformer learn token sequences")
        print("  - Phase coherence (H3) signal transfers to language modeling task")
        print("  - Token role frequency signatures (H2) provide useful constraints")
        print("\nTheoretical validation:")
        print("  - Phase coherence from Option C analysis predicts this outcome")
        print("  - Quinn learned meaningful structure (not noise)")
        print("  - Waveform encodes syntax as hypothesized")
        print("\nNext steps (Option B: Scaling):")
        print("  1. Fine-tune Quinn parameters (currently frozen)")
        print("  2. Increase model capacity (embed_dim 32→64, more layers)")
        print("  3. Larger training data to leverage improved architecture")
        print("  4. Implement phase-aware loss (Option C improvement #1)")
        print("  5. Test hierarchical frequencies (Option C improvement #3)")
        print("\nArchitecture improvements to try:")
        print("  - Phase-weighted loss by token role")
        print("  - Longer training (more epochs) to fully utilize guidance")
        print("  - Causal masking to prevent Transformer seeing future tokens")
        return "scaling"

    elif result_type == "MODERATE_HELP":
        print("\n~ QUINN'S GUIDANCE HELPS MODESTLY")
        print("\nWhat this means:")
        print("  - Structural signal exists but weak")
        print("  - Cross-attention is working but not fully leveraged")
        print("  - Phase coherence signal may need amplification")
        print("\nDiagnostic questions:")
        print("  - Is Transformer attending to waveform? (Check attention weights)")
        print("  - Do only certain frequencies help? (Ablation: zero out dims)")
        print("  - Does batch size matter? (Rerun with batch_size=32)")
        print("\nImprovements to try (Option C #2 & #4):")
        print("  1. Phase coherence regularization (encourage aligned phases)")
        print("  2. Role-weighted cross-attention (weight by token importance)")
        print("  3. Increase waveform resolution (more embedding dimensions)")
        print("  4. Retrain Phase 2A with harder negatives (larger contrastive margin)")
        print("  5. Causal masking (prevent future token leakage)")
        return "investigate"

    elif result_type == "MARGINAL_HELP":
        print("\n← QUINN'S GUIDANCE PROVIDES MINIMAL BENEFIT")
        print("\nWhat this means:")
        print("  - Waveform signal too weak for language modeling task")
        print("  - Cross-attention may not be the right mechanism")
        print("  - Contrastive margin from Phase 2A may be insufficient")
        print("\nDiagnostic checks:")
        print("  - CRITICAL: Examine attention patterns (is Transformer using Quinn?)")
        print("  - Test with causal masking (prevent cheating from future tokens)")
        print("  - Ablation: What happens if we use random waveform instead?")
        print("\nDecision point:")
        print("  Option 1: Improve Phase 2A negatives, retrain Phase 2B")
        print("    - Hypothesis: Phase 2A contrastive margin too small")
        print("    - Action: Cross-category swaps (KEYWORD↔OPERATOR)")
        print("  Option 2: Skip Quinn guidance, scale baseline Transformer")
        print("    - Hypothesis: Waveform task doesn't align with next-token prediction")
        print("    - Action: Larger baseline model may work better")
        print("\nOption C investigation reveals:")
        print("  - Phase coherence signal exists (H3 valid)")
        print("  - Token role frequencies distinctive (H2 valid)")
        print("  - But might not transfer to LM task (domain mismatch?)")
        return "investigate"

    elif result_type == "NEUTRAL":
        print("\n= NO SIGNIFICANT DIFFERENCE")
        print("\nWhat this means:")
        print("  - Waveform constraints neither help nor hurt")
        print("  - Transformer ignores waveform input (attention weights ≈ 0?)")
        print("  - Task mismatch: structure useful for completion, not next-token")
        print("\nDiagnostic: MUST INVESTIGATE")
        print("  1. Plot attention weights to waveform (is it used?)")
        print("  2. Zero-out waveform entirely (does loss change?)")
        print("  3. Random waveform (does random constraint hurt?)")
        print("  4. Causal masking test (does future visibility help baseline?)")
        print("\nHypotheses:")
        print("  - Transformer learned to ignore structure for this task")
        print("  - Waveform completion and next-token prediction misaligned")
        print("  - Cross-attention mechanism ineffective (try other architectures)")
        print("\nNext steps:")
        print("  - Return to Phase 2A analysis (margin too small?)")
        print("  - Consider alternative guidance mechanism (not cross-attention)")
        print("  - Test if baseline needs larger capacity to match Quinn+Transformer")
        return "rethink"

    elif result_type == "HURTS":
        print("\n✗ QUINN'S GUIDANCE HURTS PERFORMANCE (UNEXPECTED!)")
        print("\nWhat this means:")
        print("  - Waveform constraints conflict with next-token prediction")
        print("  - Cross-attention interferes with self-attention learning")
        print("  - Contrastive signal contradicts language modeling objective")
        print("\nTheory issue (Option C):")
        print("  - Phase coherence might enforce TOO MUCH structure")
        print("  - Language modeling needs flexibility to match training variation")
        print("  - Structure useful for CODE SHAPE, not TOKEN PREDICTION")
        print("\nImmediate action:")
        print("  1. Disable Quinn entirely for Phase 2B")
        print("  2. Scale baseline Transformer to match hybrid capacity")
        print("  3. Test if Quinn helps on a DIFFERENT task (code generation, not prediction)")
        print("\nArchitecture reconsideration:")
        print("  - Maybe waveform completion ≠ right intermediate task")
        print("  - Try alternative: token type prediction (not position)")
        print("  - Or: control flow structure prediction (not completion)")
        return "rethink"

    return "unknown"


def generate_phase2b_interpretation_report(output_path="experiments/phase2b_interpretation.md"):
    """Generate full Phase 2B interpretation report."""
    # Paths to training histories
    with_quinn_path = Path("checkpoints/phase2b_run1/training_history.json")
    baseline_path = Path("checkpoints/phase2b_baseline/training_history.json")

    if not with_quinn_path.exists() or not baseline_path.exists():
        print("Training not yet complete. Check back when checkpoints are ready.")
        return None

    # Load histories
    with_quinn_hist = load_training_history(with_quinn_path)
    baseline_hist = load_training_history(baseline_path)

    # Analyze
    result_type = analyze_convergence_dynamics(with_quinn_hist, baseline_hist)
    next_steps = interpret_result(result_type, with_quinn_hist, baseline_hist)

    # Write report
    with open(output_path, "w") as f:
        f.write("# Phase 2B Interpretation Report\n\n")
        f.write("## Result Type\n\n")
        f.write(f"**Finding**: {result_type}\n\n")
        f.write(f"**Next Steps**: {next_steps}\n\n")

        f.write("## Raw Metrics\n\n")
        f.write(f"- WITH Quinn final val loss: {with_quinn_hist[-1]['val']['lm_loss']:.4f}\n")
        f.write(f"- Baseline final val loss: {baseline_hist[-1]['val']['lm_loss']:.4f}\n")
        f.write(f"- Improvement: {((baseline_hist[-1]['val']['lm_loss'] - with_quinn_hist[-1]['val']['lm_loss']) / baseline_hist[-1]['val']['lm_loss']) * 100:.1f}%\n")

        f.write("\n## Theoretical Context (Option C)\n\n")
        f.write("Our Phase 1 waveform analysis found:\n")
        f.write("- **H2**: Token role frequency signatures are distinctive (21% separation) ✓\n")
        f.write("- **H3**: Phase coherence between adjacent tokens is significant (0.52 mean) ✓\n")
        f.write("- **Interpretation**: Quinn learned meaningful structure\n\n")

        f.write("Phase 2B tests whether this structure helps language modeling.\n")
        f.write("Result determines feasibility of Option B (scaling) and Option A (publication).\n")

    print(f"\nReport saved to: {output_path}")
    return next_steps


if __name__ == "__main__":
    generate_phase2b_interpretation_report()
