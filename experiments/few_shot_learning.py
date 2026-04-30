"""Few-shot learning experiment.

Tests whether consensus-gated plasticity improves sample efficiency.
How many observations are needed to reach target accuracy?
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
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


def run_few_shot_experiment(
    enable_cgal: bool = False,
    num_objects: int = 10,
    num_modules: int = 5,
    max_obs_per_object: int = 30,
    test_interval: int = 2,
    seed: int = 42
) -> Dict[str, Any]:
    """Run few-shot learning experiment.

    Train with varying numbers of observations and measure accuracy curves.

    Args:
        enable_cgal: Whether to enable CGAL mechanisms.
        num_objects: Number of objects.
        num_modules: Number of learning modules.
        max_obs_per_object: Maximum observations per object during training.
        test_interval: Test accuracy every N observations.
        seed: Random seed.

    Returns:
        Results dictionary with learning curves.
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
            salience_tagged_replay=False
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

    # Track accuracy over training
    learning_curve = []
    observation_count = 0

    for obs_round in range(max_obs_per_object):
        # Train on all objects
        for object_id in dataset.get_all_objects():
            observation_count += 1
            features = dataset.get_observation(object_id, noise=0.05)

            # Get hypotheses
            hypotheses = []
            for module in modules:
                hyp = module.observe(object_id, features)
                hypotheses.append(hyp)

            # Consensus
            consensus = voting_consensus(hypotheses)

            # Learning rates
            learning_rates = [1.0] * len(modules)

            if enable_cgal:
                # Apply CGAL mechanisms
                for i, (module, hyp) in enumerate(zip(modules, hypotheses)):
                    agreement = consensus_module.compute_agreement(hyp, consensus)
                    gating_factor = consensus_module.compute_gating_factor(hyp, consensus)
                    learning_rates[i] = gating_factor

                # Update trust
                module_votes = {m.module_id: hyp for m, hyp in zip(modules, hypotheses)}
                trust_module.update_all_trust(module_votes, consensus)

            # Learn
            for module, lr in zip(modules, learning_rates):
                module.learn(object_id, features, learning_rate=lr * 0.1)

        # Test periodically
        if (obs_round + 1) % test_interval == 0:
            test_accuracy = []

            for test_object_id in dataset.get_all_objects():
                for test_idx in range(10):  # 10 test observations per object
                    features = dataset.get_observation(test_object_id, noise=0.05)

                    hypotheses = []
                    for module in modules:
                        hyp = module.observe(test_object_id, features)
                        hypotheses.append(hyp)

                    consensus = voting_consensus(hypotheses)
                    correct = consensus['object_id'] == test_object_id
                    test_accuracy.append(1.0 if correct else 0.0)

            learning_curve.append({
                'num_observations': (obs_round + 1) * num_objects,
                'test_accuracy': float(np.mean(test_accuracy))
            })

    return {
        'enable_cgal': enable_cgal,
        'seed': seed,
        'learning_curve': learning_curve
    }


def run_multiple_seeds(enable_cgal: bool, num_seeds: int = 5) -> List[Dict[str, Any]]:
    """Run experiment with multiple seeds."""
    results = []
    for seed in range(num_seeds):
        print(f"  Seed {seed}...")
        result = run_few_shot_experiment(enable_cgal=enable_cgal, seed=seed)
        results.append(result)
    return results


def find_observations_to_threshold(learning_curve: List[Dict], threshold: float = 0.8) -> int:
    """Find number of observations needed to reach accuracy threshold."""
    for point in learning_curve:
        if point['test_accuracy'] >= threshold:
            return point['num_observations']
    return learning_curve[-1]['num_observations']  # Didn't reach threshold


if __name__ == "__main__":
    print("=" * 60)
    print("Few-Shot Learning Experiment")
    print("=" * 60)

    # Run baseline
    print("\nRunning BASELINE (CGAL disabled)...")
    baseline_results = run_multiple_seeds(enable_cgal=False, num_seeds=5)

    # Run CGAL
    print("\nRunning CGAL ENABLED...")
    cgal_results = run_multiple_seeds(enable_cgal=True, num_seeds=5)

    # Analyze sample efficiency
    threshold = 0.8
    print(f"\n" + "=" * 60)
    print(f"Sample Efficiency Analysis (threshold: {threshold:.1%})")
    print("=" * 60)

    baseline_obs_to_threshold = [
        find_observations_to_threshold(r['learning_curve'], threshold)
        for r in baseline_results
    ]
    cgal_obs_to_threshold = [
        find_observations_to_threshold(r['learning_curve'], threshold)
        for r in cgal_results
    ]

    baseline_mean = np.mean(baseline_obs_to_threshold)
    cgal_mean = np.mean(cgal_obs_to_threshold)

    print(f"\nBaseline observations to {threshold:.0%}: {baseline_mean:.1f} ± {np.std(baseline_obs_to_threshold):.1f}")
    print(f"CGAL observations to {threshold:.0%}:     {cgal_mean:.1f} ± {np.std(cgal_obs_to_threshold):.1f}")
    print(f"Speedup factor:                    {baseline_mean / cgal_mean:.2f}x")

    # Final accuracy comparison
    baseline_final_acc = [r['learning_curve'][-1]['test_accuracy'] for r in baseline_results]
    cgal_final_acc = [r['learning_curve'][-1]['test_accuracy'] for r in cgal_results]

    print(f"\nFinal accuracy (after full training):")
    print(f"  Baseline: {np.mean(baseline_final_acc):.3f} ± {np.std(baseline_final_acc):.3f}")
    print(f"  CGAL:     {np.mean(cgal_final_acc):.3f} ± {np.std(cgal_final_acc):.3f}")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results_data = {
        'experiment': 'few_shot_learning',
        'baseline_results': baseline_results,
        'cgal_results': cgal_results,
        'summary': {
            'threshold': threshold,
            'baseline_obs_mean': float(baseline_mean),
            'baseline_obs_std': float(np.std(baseline_obs_to_threshold)),
            'cgal_obs_mean': float(cgal_mean),
            'cgal_obs_std': float(np.std(cgal_obs_to_threshold)),
            'speedup_factor': float(baseline_mean / cgal_mean) if cgal_mean > 0 else 0.0,
            'baseline_final_acc': float(np.mean(baseline_final_acc)),
            'cgal_final_acc': float(np.mean(cgal_final_acc))
        }
    }

    output_file = output_dir / 'few_shot_learning.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_file}")
