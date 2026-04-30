"""Example demonstrating salience-tagged replay in CGAL.

This example shows how patterns accumulate salience based on consensus agreement
and novelty, and how high-salience patterns are preferentially replayed during
offline consolidation phases.
"""

import numpy as np
from cgal.config.salience_replay_config import SalienceReplayConfig
from cgal.learning_modules.salience_replay import SalienceReplayModule


def simulate_learning_with_replay():
    """Simulate learning with salience-tagged replay."""
    print("=" * 60)
    print("Salience-Tagged Replay Example")
    print("=" * 60)

    # Configure salience replay
    config = SalienceReplayConfig(
        salience_tagged_replay=True,
        alpha_consensus=0.6,
        alpha_novelty=0.4,
        decay_rate=0.98,
        replay_interval=5,
        num_replay_samples=10,
        homeostatic_downscaling=True,
        homeostatic_factor=0.99,
        enable_logging=False
    )

    # Initialize module
    replay_module = SalienceReplayModule(config)

    # Create patterns representing different learning experiences
    # Patterns 0-4: Common, high-consensus patterns
    # Patterns 5-7: Novel patterns
    # Patterns 8-9: Low-consensus patterns
    for i in range(10):
        replay_module.add_pattern(i, {'pattern_id': i, 'data': f'pattern_{i}'})

    print("\nSimulating 20 episodes of learning...")
    print("- Patterns 0-4: High consensus, low novelty")
    print("- Patterns 5-7: High novelty, medium consensus")
    print("- Patterns 8-9: Low consensus, low novelty")

    # Track replay counts
    replay_counts = {i: 0 for i in range(10)}

    def learning_function(pattern_data):
        """Simulated learning function that tracks replays."""
        pattern_id = pattern_data['pattern_id']
        replay_counts[pattern_id] += 1

    # Simulate 20 episodes
    for episode in range(1, 21):
        # Update saliences based on simulated observations

        # Patterns 0-4: High consensus (reliable)
        for i in range(5):
            agreement = np.random.uniform(0.8, 1.0)
            novelty = np.random.uniform(0.0, 0.2)
            replay_module.update_salience(i, agreement, novelty)

        # Patterns 5-7: High novelty (novel experiences)
        for i in range(5, 8):
            agreement = np.random.uniform(0.4, 0.7)
            novelty = np.random.uniform(0.7, 1.0)
            replay_module.update_salience(i, agreement, novelty)

        # Patterns 8-9: Low consensus and novelty (unreliable/common)
        for i in range(8, 10):
            agreement = np.random.uniform(0.0, 0.3)
            novelty = np.random.uniform(0.0, 0.2)
            replay_module.update_salience(i, agreement, novelty)

        # Advance time
        replay_module.on_step()

        # Episode end (replay happens every 5 episodes)
        replay_module.on_episode_end(learning_function)

        # Print salience at intervals
        if episode in [5, 10, 15, 20]:
            print(f"\nAfter episode {episode}:")
            print_salience_summary(replay_module)

    # Print final replay statistics
    print("\n" + "=" * 60)
    print("Replay Statistics")
    print("=" * 60)
    print("\nNumber of times each pattern was replayed:")
    for i in range(10):
        category = get_pattern_category(i)
        print(f"  Pattern {i} ({category:20s}): {replay_counts[i]:3d} times")

    # Show top salient patterns
    print("\nTop 5 most salient patterns:")
    top_patterns = replay_module.get_top_salient_patterns(k=5)
    for rank, (pattern_id, salience) in enumerate(top_patterns, 1):
        category = get_pattern_category(pattern_id)
        print(f"  {rank}. Pattern {pattern_id} ({category:20s}): salience={salience:.4f}")

    # Overall statistics
    stats = replay_module.get_statistics()
    print(f"\nOverall statistics:")
    print(f"  Total patterns: {stats['num_patterns']}")
    print(f"  Mean salience: {stats['mean_salience']:.4f}")
    print(f"  Max salience: {stats['max_salience']:.4f}")
    print(f"  Min salience: {stats['min_salience']:.4f}")
    print(f"  Replay phases: {stats['num_replay_phases']}")


def get_pattern_category(pattern_id: int) -> str:
    """Get human-readable category for a pattern."""
    if pattern_id < 5:
        return "High consensus"
    elif pattern_id < 8:
        return "High novelty"
    else:
        return "Low value"


def print_salience_summary(replay_module):
    """Print salience summary for all patterns."""
    print("  Pattern saliences:")
    for i in range(10):
        salience = replay_module.get_salience(i)
        category = get_pattern_category(i)
        print(f"    Pattern {i} ({category:20s}): {salience:.4f}")


def demonstrate_salience_components():
    """Demonstrate how consensus and novelty contribute to salience."""
    print("\n" + "=" * 60)
    print("Salience Components Demo")
    print("=" * 60)

    config = SalienceReplayConfig(
        salience_tagged_replay=True,
        alpha_consensus=0.6,
        alpha_novelty=0.4
    )

    replay_module = SalienceReplayModule(config)

    # Create three patterns with different profiles
    patterns = {
        'high_consensus': 0,
        'high_novelty': 1,
        'both': 2
    }

    for name, pid in patterns.items():
        replay_module.add_pattern(pid, {'name': name})

    # Update with different scores
    print("\nUpdating patterns with different profiles:")

    # High consensus, low novelty
    replay_module.update_salience(0, agreement_score=1.0, novelty_score=0.0)
    print(f"  High consensus: agreement=1.0, novelty=0.0")
    print(f"    → salience = 0.6*1.0 + 0.4*0.0 = {replay_module.get_salience(0):.2f}")

    # Low consensus, high novelty
    replay_module.update_salience(1, agreement_score=0.0, novelty_score=1.0)
    print(f"  High novelty: agreement=0.0, novelty=1.0")
    print(f"    → salience = 0.6*0.0 + 0.4*1.0 = {replay_module.get_salience(1):.2f}")

    # Both high
    replay_module.update_salience(2, agreement_score=1.0, novelty_score=1.0)
    print(f"  Both high: agreement=1.0, novelty=1.0")
    print(f"    → salience = 0.6*1.0 + 0.4*1.0 = {replay_module.get_salience(2):.2f}")


def demonstrate_decay():
    """Demonstrate salience decay over time."""
    print("\n" + "=" * 60)
    print("Salience Decay Demo")
    print("=" * 60)

    config = SalienceReplayConfig(
        salience_tagged_replay=True,
        alpha_consensus=1.0,
        alpha_novelty=0.0,
        decay_rate=0.9
    )

    replay_module = SalienceReplayModule(config)
    replay_module.add_pattern(0, {'data': 'test'})

    # Give it high salience
    replay_module.update_salience(0, agreement_score=1.0, novelty_score=0.0)

    print(f"\nInitial salience: {replay_module.get_salience(0):.4f}")
    print(f"Decay rate: {config.decay_rate}")

    print("\nSalience over time (no new updates):")
    for step in [0, 5, 10, 20, 50]:
        replay_module.current_step = step
        salience = replay_module.get_salience(0)
        print(f"  Step {step:3d}: salience = {salience:.4f}")


def demonstrate_baseline_mode():
    """Show that baseline behavior is preserved when feature is disabled."""
    print("\n" + "=" * 60)
    print("Baseline Mode (Feature Disabled)")
    print("=" * 60)

    config = SalienceReplayConfig(
        salience_tagged_replay=False  # Disabled
    )

    replay_module = SalienceReplayModule(config)
    replay_module.add_pattern(0, {'data': 'test'})

    # Try to update salience
    replay_module.update_salience(0, agreement_score=1.0, novelty_score=1.0)

    replay_count = []

    def learning_fn(data):
        replay_count.append(1)

    # Try to run replay
    replay_module.run_replay_phase(learning_fn)

    print("\nWith salience_tagged_replay=False:")
    print(f"  Pattern salience: {replay_module.get_salience(0):.1f}")
    print(f"  Replay count: {len(replay_count)}")
    print("  → Feature disabled, no salience tracking or replay")


if __name__ == "__main__":
    simulate_learning_with_replay()
    demonstrate_salience_components()
    demonstrate_decay()
    demonstrate_baseline_mode()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
