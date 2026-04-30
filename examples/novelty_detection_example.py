#!/usr/bin/env python
"""
Example usage of the Novelty Detection module (Issue #3).

This script demonstrates how to use the NoveltyDetector to analyze
hypothesis distributions and detect novel patterns.
"""

from cgal.config import NoveltyDetectionConfig
from cgal.learning_modules import NoveltyDetector


def main():
    print("=" * 70)
    print("CGAL Novelty Detection - Example Usage")
    print("=" * 70)
    print()

    # Create detector with novelty detection enabled
    config = NoveltyDetectionConfig(
        hypothesis_novelty_detection=True,
        novelty_threshold=0.7,
        enable_logging=True
    )
    detector = NoveltyDetector(config)

    # Example 1: Familiar pattern (sharp distribution, high confidence)
    print("Example 1: Familiar Pattern")
    print("-" * 70)
    familiar_scores = {
        "object_A": 10.0,
        "object_B": 2.0,
        "object_C": 1.0,
    }
    print(f"Hypothesis scores: {familiar_scores}")
    is_novel, score = detector.is_novel(familiar_scores)
    print(f"Is novel: {is_novel}")
    print(f"Novelty score: {score:.4f}")
    print(f"Interpretation: {'Novel! Route to new pattern' if is_novel else 'Familiar! Reinforce existing pattern'}")
    print()

    # Example 2: Novel pattern (broad distribution, low confidence)
    print("Example 2: Novel Pattern")
    print("-" * 70)
    novel_scores = {
        "object_A": 1.0,
        "object_B": 1.0,
        "object_C": 1.0,
        "object_D": 1.0,
    }
    print(f"Hypothesis scores: {novel_scores}")
    is_novel, score = detector.is_novel(novel_scores)
    print(f"Is novel: {is_novel}")
    print(f"Novelty score: {score:.4f}")
    print(f"Interpretation: {'Novel! Route to new pattern' if is_novel else 'Familiar! Reinforce existing pattern'}")
    print()

    # Example 3: Ambiguous pattern (sharp but not confident)
    print("Example 3: Ambiguous Pattern")
    print("-" * 70)
    ambiguous_scores = {
        "object_A": 3.0,
        "object_B": 2.5,
        "object_C": 0.5,
    }
    print(f"Hypothesis scores: {ambiguous_scores}")
    is_novel, score = detector.is_novel(ambiguous_scores)
    print(f"Is novel: {is_novel}")
    print(f"Novelty score: {score:.4f}")
    print(f"Interpretation: {'Novel! Route to new pattern' if is_novel else 'Familiar! Reinforce existing pattern'}")
    print()

    # Example 4: Gradually increasing confidence
    print("Example 4: Learning Trajectory (Gradually Increasing Confidence)")
    print("-" * 70)
    print("Observing object_A multiple times, confidence increases...")
    print()

    trajectories = [
        {"object_A": 1.0, "object_B": 1.0, "object_C": 1.0},
        {"object_A": 2.0, "object_B": 1.0, "object_C": 1.0},
        {"object_A": 5.0, "object_B": 1.0, "object_C": 1.0},
        {"object_A": 10.0, "object_B": 1.0, "object_C": 1.0},
    ]

    for i, scores in enumerate(trajectories, 1):
        is_novel, score = detector.is_novel(scores)
        print(f"  Observation {i}: novelty={score:.4f}, is_novel={is_novel}")

    print()

    # Show statistics
    print("Statistics Summary")
    print("-" * 70)
    stats = detector.get_statistics()
    print(f"Total observations: {stats['count']}")
    print(f"Mean novelty: {stats['mean']:.4f}")
    print(f"Std novelty: {stats['std']:.4f}")
    print(f"Min novelty: {stats['min']:.4f}")
    print(f"Max novelty: {stats['max']:.4f}")
    print()

    # Example 5: Baseline mode (feature disabled)
    print("Example 5: Baseline Mode (Feature Disabled)")
    print("-" * 70)
    baseline_config = NoveltyDetectionConfig(
        hypothesis_novelty_detection=False  # Disabled
    )
    baseline_detector = NoveltyDetector(baseline_config)

    test_scores = {"object_A": 1.0, "object_B": 1.0, "object_C": 1.0}
    is_novel, score = baseline_detector.is_novel(test_scores)
    print(f"Hypothesis scores: {test_scores}")
    print(f"Is novel: {is_novel}")
    print(f"Novelty score: {score:.4f}")
    print("Note: When disabled, always returns (False, 0.0) - baseline behavior preserved")
    print()

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
