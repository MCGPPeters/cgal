"""Noise robustness experiment.

Tests whether learned trust weights help the network tolerate noisy modules.
Some modules receive corrupted inputs while others see clean data.
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


def run_noise_robustness_experiment(
    enable_cgal: bool = False,
    num_objects: int = 10,
    num_modules: int = 5,
    fraction_noisy_modules: float = 0.3,
    noise_level: float = 0.5,
    num_training_obs_per_object: int = 20,
    num_test_obs_per_object: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """Run noise robustness experiment.

    Args:
        enable_cgal: Whether to enable CGAL mechanisms (especially trust weights).
        num_objects: Number of objects.
        num_modules: Number of learning modules.
        fraction_noisy_modules: Fraction of modules that receive noisy input.
        noise_level: Standard deviation of Gaussian noise added to features.
        num_training_obs_per_object: Training observations per object.
        num_test_obs_per_object: Test observations per object.
        seed: Random seed.

    Returns:
        Results dictionary.
    """
    np.random.seed(seed)

    # Create dataset
    dataset = SyntheticObjectDataset(
        num_objects=num_objects,
        feature_dim=50,
        seed=seed
    )

    # Determine which modules are noisy
    num_noisy = int(num_modules * fraction_noisy_modules)
    noisy_module_indices = list(range(num_noisy))
    noise_levels = [noise_level if i in noisy_module_indices else 0.0 for i in range(num_modules)]

    print(f"    Noisy modules: {noisy_module_indices} (noise level: {noise_level})")

    # Create learning modules with noise
    modules = create_learning_network(num_modules=num_modules, noise_levels=noise_levels)

    # Initialize CGAL components
    if enable_cgal:
        consensus_config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7
        )
        novelty_config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True
        )
        trust_config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.1,  # Faster learning for trust
            trust_min=0.1
        )
        replay_config = SalienceReplayConfig(
            salience_tagged_replay=False  # Not critical for this experiment
        )

        consensus_module = ConsensusGatingModule(consensus_config)
        novelty_detector = NoveltyDetector(novelty_config)
        trust_module = TrustWeightsModule(trust_config)
        replay_module = SalienceReplayModule(replay_config)

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
            features = dataset.get_observation(object_id, noise=0.02)  # Base noise

            # Get hypotheses (modules apply their own noise internally)
            hypotheses = []
            for module in modules:
                hyp = module.observe(object_id, features)
                hypotheses.append(hyp)

            # Compute consensus
            consensus = voting_consensus(hypotheses)

            # Learning rates
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

            # Learn
            for module, lr in zip(modules, learning_rates):
                module.learn(object_id, features, learning_rate=lr * 0.1)

            # Track accuracy
            correct = consensus['object_id'] == object_id
            training_accuracy.append(1.0 if correct else 0.0)

    # Test phase
    test_accuracy = []

    for object_id in dataset.get_all_objects():
        for obs_idx in range(num_test_obs_per_object):
            features = dataset.get_observation(object_id, noise=0.02)

            hypotheses = []
            for module in modules:
                hyp = module.observe(object_id, features)
                hypotheses.append(hyp)

            consensus = voting_consensus(hypotheses)

            correct = consensus['object_id'] == object_id
            test_accuracy.append(1.0 if correct else 0.0)

    # Analyze trust weights if CGAL enabled
    trust_analysis = None
    if enable_cgal:
        # Get average trust in noisy vs clean modules
        trust_in_noisy = []
        trust_in_clean = []

        for from_id in module_ids:
            for to_id in module_ids:
                if from_id != to_id:
                    trust = trust_module.get_trust(from_id, to_id)
                    if to_id in noisy_module_indices:
                        trust_in_noisy.append(trust)
                    else:
                        trust_in_clean.append(trust)

        trust_analysis = {
            'mean_trust_in_noisy': float(np.mean(trust_in_noisy)) if trust_in_noisy else 0.0,
            'mean_trust_in_clean': float(np.mean(trust_in_clean)) if trust_in_clean else 0.0,
            'trust_difference': float(np.mean(trust_in_clean) - np.mean(trust_in_noisy)) if (trust_in_clean and trust_in_noisy) else 0.0
        }

    return {
        'enable_cgal': enable_cgal,
        'seed': seed,
        'num_noisy_modules': num_noisy,
        'noise_level': noise_level,
        'training_accuracy': float(np.mean(training_accuracy)),
        'test_accuracy': float(np.mean(test_accuracy)),
        'trust_analysis': trust_analysis
    }


def run_multiple_seeds(enable_cgal: bool, num_seeds: int = 5) -> List[Dict[str, Any]]:
    """Run experiment with multiple seeds."""
    results = []
    for seed in range(num_seeds):
        print(f"  Seed {seed}:")
        result = run_noise_robustness_experiment(enable_cgal=enable_cgal, seed=seed)
        results.append(result)
        if result['trust_analysis']:
            print(f"    Trust in clean: {result['trust_analysis']['mean_trust_in_clean']:.3f}, "
                  f"Trust in noisy: {result['trust_analysis']['mean_trust_in_noisy']:.3f}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Noise Robustness Experiment")
    print("=" * 60)
    print("\n30% of modules receive noisy input.")
    print("Testing whether trust weights help maintain accuracy.\n")

    # Run baseline
    print("Running BASELINE (CGAL disabled)...")
    baseline_results = run_multiple_seeds(enable_cgal=False, num_seeds=5)

    # Run CGAL
    print("\nRunning CGAL ENABLED (with trust weights)...")
    cgal_results = run_multiple_seeds(enable_cgal=True, num_seeds=5)

    # Aggregate
    baseline_test_acc = [r['test_accuracy'] for r in baseline_results]
    cgal_test_acc = [r['test_accuracy'] for r in cgal_results]

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nBaseline test accuracy: {np.mean(baseline_test_acc):.3f} ± {np.std(baseline_test_acc):.3f}")
    print(f"CGAL test accuracy:     {np.mean(cgal_test_acc):.3f} ± {np.std(cgal_test_acc):.3f}")
    print(f"Improvement:            {np.mean(cgal_test_acc) - np.mean(baseline_test_acc):.3f}")

    # Trust analysis
    if cgal_results[0]['trust_analysis']:
        trust_in_clean = [r['trust_analysis']['mean_trust_in_clean'] for r in cgal_results]
        trust_in_noisy = [r['trust_analysis']['mean_trust_in_noisy'] for r in cgal_results]

        print(f"\nTrust in clean modules: {np.mean(trust_in_clean):.3f} ± {np.std(trust_in_clean):.3f}")
        print(f"Trust in noisy modules: {np.mean(trust_in_noisy):.3f} ± {np.std(trust_in_noisy):.3f}")
        print(f"Trust difference:       {np.mean(trust_in_clean) - np.mean(trust_in_noisy):.3f}")
        print("\n(Trust weights learned to down-weight noisy modules)")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results_data = {
        'experiment': 'noise_robustness',
        'baseline_results': baseline_results,
        'cgal_results': cgal_results,
        'summary': {
            'baseline_mean': float(np.mean(baseline_test_acc)),
            'baseline_std': float(np.std(baseline_test_acc)),
            'cgal_mean': float(np.mean(cgal_test_acc)),
            'cgal_std': float(np.std(cgal_test_acc)),
            'improvement': float(np.mean(cgal_test_acc) - np.mean(baseline_test_acc))
        }
    }

    output_file = output_dir / 'noise_robustness.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_file}")
