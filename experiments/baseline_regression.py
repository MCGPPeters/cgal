"""Baseline regression experiment.

Tests that CGAL mechanisms work correctly when enabled vs disabled.
When all CGAL flags are disabled, performance should match pure baseline.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cgal.config import (
    ConsensusGatingConfig,
    NoveltyDetectionConfig,
    TrustWeightsConfig,
    SalienceReplayConfig
)
from cgal.learning_modules import (
    ConsensusGatingModule,
    NoveltyDetector,
    TrustWeightsModule,
    SalienceReplayModule
)
from experiments.synthetic_data import (
    SyntheticObjectDataset,
    create_learning_network,
    voting_consensus
)


def run_baseline_experiment(
    enable_cgal: bool = False,
    num_objects: int = 10,
    num_modules: int = 5,
    num_training_obs_per_object: int = 20,
    num_test_obs_per_object: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """Run baseline experiment with or without CGAL.

    Args:
        enable_cgal: Whether to enable CGAL mechanisms.
        num_objects: Number of objects in dataset.
        num_modules: Number of learning modules.
        num_training_obs_per_object: Training observations per object.
        num_test_obs_per_object: Test observations per object.
        seed: Random seed.

    Returns:
        Results dictionary with accuracy and other metrics.
    """
    np.random.seed(seed)

    # Create dataset
    dataset = SyntheticObjectDataset(
        num_objects=num_objects,
        feature_dim=50,
        seed=seed
    )

    # Create learning modules
    modules = create_learning_network(num_modules=num_modules)

    # Initialize CGAL components
    if enable_cgal:
        consensus_config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7
        )
        novelty_config = NoveltyDetectionConfig(
            novelty_detection_enabled=True
        )
        trust_config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.05
        )
        replay_config = SalienceReplayConfig(
            salience_tagged_replay=True,
            replay_interval=5
        )

        consensus_module = ConsensusGatingModule(consensus_config)
        novelty_detector = NoveltyDetector(novelty_config)
        trust_module = TrustWeightsModule(trust_config)
        replay_module = SalienceReplayModule(replay_config)

        # Initialize trust matrix
        module_ids = [m.module_id for m in modules]
        trust_module.initialize_trust(module_ids)
    else:
        consensus_module = None
        novelty_detector = None
        trust_module = None
        replay_module = None

    # Training phase
    training_accuracy = []

    for object_id in dataset.get_all_objects():
        for obs_idx in range(num_training_obs_per_object):
            # Get observation
            features = dataset.get_observation(object_id, noise=0.05)

            # Each module observes and generates hypothesis
            hypotheses = []
            for module in modules:
                hyp = module.observe(object_id, features)
                hypotheses.append(hyp)

            # Compute consensus
            consensus = voting_consensus(hypotheses)

            # Determine learning rates
            learning_rates = [1.0] * len(modules)

            if enable_cgal:
                # Apply consensus gating
                for i, (module, hyp) in enumerate(zip(modules, hypotheses)):
                    agreement = consensus_module.compute_agreement(hyp, consensus)
                    gating_factor = consensus_module.compute_gating_factor(hyp, consensus)
                    learning_rates[i] = gating_factor

                # Update trust weights
                module_votes = {m.module_id: hyp for m, hyp in zip(modules, hypotheses)}
                trust_module.update_all_trust(module_votes, consensus)

                # Update novelty
                novelty_detector.observe(consensus)

            # Learn with computed rates
            for module, lr in zip(modules, learning_rates):
                module.learn(object_id, features, learning_rate=lr * 0.1)

            # Track training accuracy
            correct = consensus['object_id'] == object_id
            training_accuracy.append(1.0 if correct else 0.0)

    # Test phase
    test_accuracy = []

    for object_id in dataset.get_all_objects():
        for obs_idx in range(num_test_obs_per_object):
            features = dataset.get_observation(object_id, noise=0.05)

            # Get hypotheses
            hypotheses = []
            for module in modules:
                hyp = module.observe(object_id, features)
                hypotheses.append(hyp)

            # Consensus
            consensus = voting_consensus(hypotheses)

            # Evaluate
            correct = consensus['object_id'] == object_id
            test_accuracy.append(1.0 if correct else 0.0)

    return {
        'enable_cgal': enable_cgal,
        'seed': seed,
        'num_objects': num_objects,
        'num_modules': num_modules,
        'training_accuracy': float(np.mean(training_accuracy)),
        'test_accuracy': float(np.mean(test_accuracy)),
        'num_training_samples': len(training_accuracy),
        'num_test_samples': len(test_accuracy)
    }


def run_multiple_seeds(enable_cgal: bool, num_seeds: int = 5) -> List[Dict[str, Any]]:
    """Run experiment with multiple seeds.

    Args:
        enable_cgal: Whether to enable CGAL.
        num_seeds: Number of random seeds to try.

    Returns:
        List of results, one per seed.
    """
    results = []
    for seed in range(num_seeds):
        print(f"  Running seed {seed}...")
        result = run_baseline_experiment(enable_cgal=enable_cgal, seed=seed)
        results.append(result)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Regression Experiment")
    print("=" * 60)

    # Run with CGAL disabled (baseline)
    print("\nRunning BASELINE (CGAL disabled)...")
    baseline_results = run_multiple_seeds(enable_cgal=False, num_seeds=5)

    # Run with CGAL enabled
    print("\nRunning CGAL ENABLED...")
    cgal_results = run_multiple_seeds(enable_cgal=True, num_seeds=5)

    # Aggregate results
    baseline_test_acc = [r['test_accuracy'] for r in baseline_results]
    cgal_test_acc = [r['test_accuracy'] for r in cgal_results]

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nBaseline test accuracy: {np.mean(baseline_test_acc):.3f} ± {np.std(baseline_test_acc):.3f}")
    print(f"CGAL test accuracy:     {np.mean(cgal_test_acc):.3f} ± {np.std(cgal_test_acc):.3f}")
    print(f"Difference:             {np.mean(cgal_test_acc) - np.mean(baseline_test_acc):.3f}")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results_data = {
        'experiment': 'baseline_regression',
        'baseline_results': baseline_results,
        'cgal_results': cgal_results,
        'summary': {
            'baseline_mean': float(np.mean(baseline_test_acc)),
            'baseline_std': float(np.std(baseline_test_acc)),
            'cgal_mean': float(np.mean(cgal_test_acc)),
            'cgal_std': float(np.std(cgal_test_acc)),
            'difference': float(np.mean(cgal_test_acc) - np.mean(baseline_test_acc))
        }
    }

    output_file = output_dir / 'baseline_regression.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_file}")
