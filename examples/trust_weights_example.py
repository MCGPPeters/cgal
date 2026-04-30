"""Example demonstrating learned trust weights in CGAL.

This example shows how trust weights between learning modules evolve based on
voting accuracy. Reliable voters gain trust; unreliable ones lose it.
"""

import numpy as np
from cgal.config.trust_weights_config import TrustWeightsConfig
from cgal.learning_modules.trust_weights import TrustWeightsModule


def simulate_voting_rounds():
    """Simulate multiple voting rounds with one unreliable module."""
    print("=" * 60)
    print("Learned Trust Weights Example")
    print("=" * 60)

    # Configure trust weights with relatively fast learning
    config = TrustWeightsConfig(
        learned_trust_weights=True,
        trust_learning_rate=0.1,  # Faster learning for demo
        trust_min=0.1,
        trust_max=1.0,
        enable_logging=False
    )

    # Initialize trust weights module
    trust_module = TrustWeightsModule(config)
    module_ids = [1, 2, 3]
    trust_module.initialize_trust(module_ids)

    print("\nInitial trust matrix (all uniform):")
    print_trust_matrix(trust_module, module_ids)

    # Simulate 20 voting rounds
    # Modules 1 and 2 are reliable (agree with consensus)
    # Module 3 is unreliable (disagrees with consensus)
    print("\nSimulating 20 voting rounds...")
    print("- Modules 1 and 2: Reliable (always vote 'cup')")
    print("- Module 3: Unreliable (always votes 'plate')")
    print("- Consensus: 'cup' (majority vote)")

    for round_num in range(1, 21):
        module_votes = {
            1: {'object_id': 'cup'},
            2: {'object_id': 'cup'},
            3: {'object_id': 'plate'}  # Always wrong
        }
        consensus = {'object_id': 'cup'}

        trust_module.update_all_trust(module_votes, consensus)

        if round_num in [1, 5, 10, 15, 20]:
            print(f"\nAfter round {round_num}:")
            print_trust_matrix(trust_module, module_ids)

    # Show final statistics
    print("\nFinal statistics:")
    stats = trust_module.get_statistics()
    print(f"  Mean trust: {stats['mean']:.4f}")
    print(f"  Std dev:    {stats['std']:.4f}")
    print(f"  Min trust:  {stats['min']:.4f}")
    print(f"  Max trust:  {stats['max']:.4f}")

    # Demonstrate vote weighting
    print("\nVote weighting example:")
    print("Module 1 receives votes from modules 2 and 3:")
    neighbor_votes = {
        2: {'object_id': 'cup'},
        3: {'object_id': 'plate'}
    }
    weights = trust_module.weight_votes(1, neighbor_votes)
    print(f"  Weight for module 2: {weights[2]:.4f} (reliable)")
    print(f"  Weight for module 3: {weights[3]:.4f} (unreliable)")


def print_trust_matrix(trust_module, module_ids):
    """Print trust matrix in readable format."""
    matrix = trust_module.get_trust_matrix_array(module_ids)

    print("  Trust matrix (rows trust columns):")
    print("      " + "  ".join([f"M{mid}" for mid in module_ids]))
    for i, from_id in enumerate(module_ids):
        row_values = [f"{matrix[i, j]:.2f}" for j in range(len(module_ids))]
        print(f"  M{from_id}  " + "  ".join(row_values))


def demonstrate_baseline_mode():
    """Show that trust weights are disabled in baseline mode."""
    print("\n" + "=" * 60)
    print("Baseline Mode (Feature Disabled)")
    print("=" * 60)

    config = TrustWeightsConfig(
        learned_trust_weights=False,  # Feature disabled
    )

    trust_module = TrustWeightsModule(config)
    trust_module.initialize_trust([1, 2, 3])

    # Manually set a different trust value
    trust_module.trust_matrix[(1, 2)] = 0.3

    # get_trust should still return 1.0 when feature is disabled
    trust = trust_module.get_trust(1, 2)

    print("\nWith learned_trust_weights=False:")
    print(f"  Internal trust_matrix[(1,2)] = 0.3")
    print(f"  get_trust(1, 2) returns: {trust:.1f}")
    print("  → Feature disabled, trust always 1.0 (baseline behavior)")


def demonstrate_recovery():
    """Show that trust can recover if unreliable module improves."""
    print("\n" + "=" * 60)
    print("Trust Recovery Example")
    print("=" * 60)

    config = TrustWeightsConfig(
        learned_trust_weights=True,
        trust_learning_rate=0.1,
        trust_min=0.1,
        trust_max=1.0
    )

    trust_module = TrustWeightsModule(config)
    trust_module.initialize_trust([1, 2])

    # Module 2 performs poorly for 10 rounds
    print("\nPhase 1: Module 2 performs poorly (10 rounds)")
    for _ in range(10):
        module_votes = {
            1: {'object_id': 'cup'},
            2: {'object_id': 'plate'}  # Wrong
        }
        consensus = {'object_id': 'cup'}
        trust_module.update_all_trust(module_votes, consensus)

    trust_after_bad = trust_module.get_trust(1, 2)
    print(f"  Trust in module 2: {trust_after_bad:.4f}")

    # Module 2 improves and performs well for 20 rounds
    print("\nPhase 2: Module 2 improves (20 rounds)")
    for _ in range(20):
        module_votes = {
            1: {'object_id': 'cup'},
            2: {'object_id': 'cup'}  # Now correct
        }
        consensus = {'object_id': 'cup'}
        trust_module.update_all_trust(module_votes, consensus)

    trust_after_good = trust_module.get_trust(1, 2)
    print(f"  Trust in module 2: {trust_after_good:.4f}")
    print("  → Trust recovers as module improves!")


if __name__ == "__main__":
    simulate_voting_rounds()
    demonstrate_baseline_mode()
    demonstrate_recovery()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
