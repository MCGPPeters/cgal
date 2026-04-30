"""Continual learning experiment.

Tests catastrophic forgetting: does CGAL help retain earlier objects
when new objects are added incrementally?
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


def run_continual_learning_experiment(
    enable_cgal: bool = False,
    num_objects: int = 15,
    objects_per_phase: int = 5,
    num_modules: int = 5,
    num_obs_per_object: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """Run continual learning experiment.

    Objects are introduced in phases. After each phase, we test accuracy
    on ALL previously-seen objects to detect catastrophic forgetting.

    Args:
        enable_cgal: Whether to enable CGAL mechanisms.
        num_objects: Total number of objects.
        objects_per_phase: Objects introduced per phase.
        num_modules: Number of learning modules.
        num_obs_per_object: Observations per object during training.
        seed: Random seed.

    Returns:
        Results dictionary with per-phase accuracies.
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
            hypothesis_novelty_detection=True
        )
        trust_config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.05
        )
        replay_config = SalienceReplayConfig(
            salience_tagged_replay=True,
            replay_interval=5,
            num_replay_samples=20
        )

        consensus_module = ConsensusGatingModule(consensus_config)
        novelty_detector = NoveltyDetector(novelty_config)
        trust_module = TrustWeightsModule(trust_config)
        replay_module = SalienceReplayModule(replay_config)

        module_ids = [m.module_id for m in modules]
        trust_module.initialize_trust(module_ids)

        # Pattern counter for replay
        pattern_id_counter = [0]
    else:
        consensus_module = None
        novelty_detector = None
        trust_module = None
        replay_module = None

    # Divide objects into phases
    all_objects = dataset.get_all_objects()
    num_phases = len(all_objects) // objects_per_phase
    phases = [
        all_objects[i*objects_per_phase:(i+1)*objects_per_phase]
        for i in range(num_phases)
    ]

    # Track accuracy after each phase
    phase_results = []
    learned_objects = []

    for phase_idx, phase_objects in enumerate(phases):
        print(f"\n  Phase {phase_idx + 1}: Learning objects {len(learned_objects)} to {len(learned_objects) + len(phase_objects)}")

        # Learn new objects
        for object_id in phase_objects:
            for obs_idx in range(num_obs_per_object):
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

                    # Track salience
                    novelty_score = novelty_detector.score(consensus)
                    novelty_detector.observe(consensus)
                    agreement = consensus_module.compute_agreement(hypotheses[0], consensus)

                    pattern = replay_module.add_pattern(pattern_id_counter[0], {
                        'object_id': object_id,
                        'features': features
                    })
                    replay_module.update_salience(pattern_id_counter[0], agreement, novelty_score)
                    pattern_id_counter[0] += 1
                    replay_module.on_step()

                # Learn
                for module, lr in zip(modules, learning_rates):
                    module.learn(object_id, features, learning_rate=lr * 0.1)

        learned_objects.extend(phase_objects)

        # Test on ALL learned objects
        test_accuracy_by_object = {}

        for test_object_id in learned_objects:
            correct_count = 0
            total_count = 10

            for test_idx in range(total_count):
                features = dataset.get_observation(test_object_id, noise=0.05)

                hypotheses = []
                for module in modules:
                    hyp = module.observe(test_object_id, features)
                    hypotheses.append(hyp)

                consensus = voting_consensus(hypotheses)
                if consensus['object_id'] == test_object_id:
                    correct_count += 1

            test_accuracy_by_object[test_object_id] = correct_count / total_count

        # Compute accuracies for old vs new objects
        old_objects = learned_objects[:-objects_per_phase] if phase_idx > 0 else []
        new_objects = phase_objects

        old_acc = np.mean([test_accuracy_by_object[obj] for obj in old_objects]) if old_objects else 0.0
        new_acc = np.mean([test_accuracy_by_object[obj] for obj in new_objects])
        overall_acc = np.mean(list(test_accuracy_by_object.values()))

        phase_results.append({
            'phase': phase_idx + 1,
            'num_learned_objects': len(learned_objects),
            'old_object_accuracy': float(old_acc),
            'new_object_accuracy': float(new_acc),
            'overall_accuracy': float(overall_acc)
        })

        print(f"    Old objects: {old_acc:.3f}, New objects: {new_acc:.3f}, Overall: {overall_acc:.3f}")

    return {
        'enable_cgal': enable_cgal,
        'seed': seed,
        'num_phases': num_phases,
        'phase_results': phase_results
    }


def run_multiple_seeds(enable_cgal: bool, num_seeds: int = 5) -> List[Dict[str, Any]]:
    """Run experiment with multiple seeds."""
    results = []
    for seed in range(num_seeds):
        print(f"\n  Seed {seed}:")
        result = run_continual_learning_experiment(enable_cgal=enable_cgal, seed=seed)
        results.append(result)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Continual Learning Experiment")
    print("=" * 60)

    # Run baseline
    print("\nRunning BASELINE (CGAL disabled)...")
    baseline_results = run_multiple_seeds(enable_cgal=False, num_seeds=5)

    # Run CGAL
    print("\nRunning CGAL ENABLED...")
    cgal_results = run_multiple_seeds(enable_cgal=True, num_seeds=5)

    # Analyze catastrophic forgetting
    print("\n" + "=" * 60)
    print("Catastrophic Forgetting Analysis")
    print("=" * 60)

    # Compare accuracy drop on old objects
    def get_forgetting_metric(results: List[Dict]) -> Tuple[float, float]:
        """Compute how much old object accuracy drops over phases."""
        forgetting_scores = []

        for result in results:
            phases = result['phase_results']
            if len(phases) >= 2:
                # Compare first phase accuracy to last phase old-object accuracy
                initial_acc = phases[0]['overall_accuracy']
                final_old_acc = phases[-1]['old_object_accuracy']
                forgetting = initial_acc - final_old_acc
                forgetting_scores.append(forgetting)

        return np.mean(forgetting_scores), np.std(forgetting_scores)

    baseline_forgetting, baseline_forgetting_std = get_forgetting_metric(baseline_results)
    cgal_forgetting, cgal_forgetting_std = get_forgetting_metric(cgal_results)

    print(f"\nBaseline forgetting: {baseline_forgetting:.3f} ± {baseline_forgetting_std:.3f}")
    print(f"CGAL forgetting:     {cgal_forgetting:.3f} ± {cgal_forgetting_std:.3f}")
    print(f"Improvement:         {baseline_forgetting - cgal_forgetting:.3f}")
    print("\n(Lower forgetting = better retention of early objects)")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    results_data = {
        'experiment': 'continual_learning',
        'baseline_results': baseline_results,
        'cgal_results': cgal_results,
        'summary': {
            'baseline_forgetting_mean': float(baseline_forgetting),
            'baseline_forgetting_std': float(baseline_forgetting_std),
            'cgal_forgetting_mean': float(cgal_forgetting),
            'cgal_forgetting_std': float(cgal_forgetting_std),
            'improvement': float(baseline_forgetting - cgal_forgetting)
        }
    }

    output_file = output_dir / 'continual_learning.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_file}")
