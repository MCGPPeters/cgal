#!/usr/bin/env python
"""
Example usage of the Consensus-Gated Plasticity module (Issue #2).

This script demonstrates how to use the ConsensusGatingModule to modulate
learning rates based on agreement with voting consensus.
"""

from cgal.config import ConsensusGatingConfig
from cgal.learning_modules import ConsensusGatingModule


def main():
    print("=" * 70)
    print("CGAL Consensus-Gated Plasticity - Example Usage")
    print("=" * 70)
    print()

    # Create module with consensus gating enabled
    config = ConsensusGatingConfig(
        consensus_gated_plasticity=True,
        alpha=0.7,
        baseline_rate=1.0,
        enable_logging=True
    )
    module = ConsensusGatingModule(config)

    # Example 1: Perfect Agreement (reinforce strongly)
    print("Example 1: Perfect Agreement")
    print("-" * 70)
    lm_hypothesis = {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}
    consensus = {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}
    print(f"LM hypothesis: {lm_hypothesis}")
    print(f"Consensus: {consensus}")

    gating_factor = module.compute_gating_factor(lm_hypothesis, consensus)
    print(f"Gating factor (g_m): {gating_factor:.4f}")
    print(f"Interpretation: Strong reinforcement (perfect agreement)")

    # Apply to learning rate
    learning_rate = 0.1
    gated_lr = module.apply_gating(learning_rate, lm_hypothesis, consensus)
    print(f"Original learning rate: {learning_rate}")
    print(f"Gated learning rate: {gated_lr:.4f}")
    print()

    # Example 2: Complete Disagreement (reduce reinforcement)
    print("Example 2: Complete Disagreement")
    print("-" * 70)
    lm_hypothesis = {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}
    consensus = {'object_id': 'plate', 'pose': [5.0, 6.0, 7.0]}
    print(f"LM hypothesis: {lm_hypothesis}")
    print(f"Consensus: {consensus}")

    gating_factor = module.compute_gating_factor(lm_hypothesis, consensus)
    print(f"Gating factor (g_m): {gating_factor:.4f}")
    print(f"Expected: {(1 - config.alpha):.4f} (= 1 - alpha)")
    print(f"Interpretation: Reduced reinforcement (no agreement)")

    learning_rate = 0.1
    gated_lr = module.apply_gating(learning_rate, lm_hypothesis, consensus)
    print(f"Original learning rate: {learning_rate}")
    print(f"Gated learning rate: {gated_lr:.4f}")
    print()

    # Example 3: Partial Agreement (same object, different pose)
    print("Example 3: Partial Agreement")
    print("-" * 70)
    lm_hypothesis = {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}
    consensus = {'object_id': 'cup', 'pose': [5.0, 6.0, 7.0]}
    print(f"LM hypothesis: {lm_hypothesis}")
    print(f"Consensus: {consensus}")

    gating_factor = module.compute_gating_factor(lm_hypothesis, consensus)
    print(f"Gating factor (g_m): {gating_factor:.4f}")
    print(f"Interpretation: Moderate reinforcement (same object, different pose)")

    learning_rate = 0.1
    gated_lr = module.apply_gating(learning_rate, lm_hypothesis, consensus)
    print(f"Original learning rate: {learning_rate}")
    print(f"Gated learning rate: {gated_lr:.4f}")
    print()

    # Example 4: No Consensus Yet (early in episode)
    print("Example 4: No Consensus Yet (Early in Episode)")
    print("-" * 70)
    lm_hypothesis = {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}
    consensus = None
    print(f"LM hypothesis: {lm_hypothesis}")
    print(f"Consensus: {consensus}")

    gating_factor = module.compute_gating_factor(lm_hypothesis, consensus)
    print(f"Gating factor (g_m): {gating_factor:.4f}")
    print(f"Interpretation: Full reinforcement (no consensus to gate against yet)")
    print()

    # Example 5: Learning Trajectory
    print("Example 5: Learning Trajectory (Improving Agreement)")
    print("-" * 70)
    print("As LM learns, its hypotheses align better with consensus...")
    print()

    trajectories = [
        ({'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}, {'object_id': 'plate', 'pose': [5.0, 6.0, 7.0]}),
        ({'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}, {'object_id': 'cup', 'pose': [5.0, 6.0, 7.0]}),
        ({'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}, {'object_id': 'cup', 'pose': [1.5, 2.5, 3.5]}),
        ({'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}, {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}),
    ]

    for i, (lm_hyp, cons_hyp) in enumerate(trajectories, 1):
        gating = module.compute_gating_factor(lm_hyp, cons_hyp)
        agreement = module.compute_agreement(lm_hyp, cons_hyp)
        print(f"  Step {i}: agreement={agreement:.2f}, g_m={gating:.4f}")

    print()

    # Show statistics
    print("Statistics Summary")
    print("-" * 70)
    stats = module.get_statistics()
    print(f"Total gating computations: {stats['count']}")
    print(f"Mean gating factor: {stats['mean']:.4f}")
    print(f"Std deviation: {stats['std']:.4f}")
    print(f"Min gating factor: {stats['min']:.4f}")
    print(f"Max gating factor: {stats['max']:.4f}")
    print()

    # Example 6: Baseline Mode (Feature Disabled)
    print("Example 6: Baseline Mode (Feature Disabled)")
    print("-" * 70)
    baseline_config = ConsensusGatingConfig(
        consensus_gated_plasticity=False  # Disabled
    )
    baseline_module = ConsensusGatingModule(baseline_config)

    lm_hypothesis = {'object_id': 'cup'}
    consensus = {'object_id': 'plate'}
    gating_factor = baseline_module.compute_gating_factor(lm_hypothesis, consensus)

    print(f"LM hypothesis: {lm_hypothesis}")
    print(f"Consensus: {consensus}")
    print(f"Gating factor (g_m): {gating_factor:.4f}")
    print("Note: When disabled, always returns 1.0 - baseline behavior preserved")
    print()

    # Example 7: Custom Alpha (Pure Consensus-Based)
    print("Example 7: Custom Alpha (Pure Consensus-Based)")
    print("-" * 70)
    pure_config = ConsensusGatingConfig(
        consensus_gated_plasticity=True,
        alpha=1.0,  # Pure consensus influence
        baseline_rate=0.0,
        enable_logging=False
    )
    pure_module = ConsensusGatingModule(pure_config)

    print("With alpha=1.0 (pure consensus-based gating):")

    # Perfect agreement
    lm_hyp = {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}
    cons_hyp = {'object_id': 'cup', 'pose': [1.0, 2.0, 3.0]}
    gating = pure_module.compute_gating_factor(lm_hyp, cons_hyp)
    print(f"  Perfect agreement → g_m = {gating:.4f}")

    # No agreement
    lm_hyp = {'object_id': 'cup'}
    cons_hyp = {'object_id': 'plate'}
    gating = pure_module.compute_gating_factor(lm_hyp, cons_hyp)
    print(f"  No agreement → g_m = {gating:.4f}")
    print()

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
