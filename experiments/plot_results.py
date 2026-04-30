"""Visualization utilities for CGAL experiment results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys


def load_results(results_dir: Path, experiment_name: str) -> Dict:
    """Load experiment results from JSON file."""
    results_file = results_dir / f"{experiment_name}.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r') as f:
        return json.load(f)


def plot_baseline_regression(results_dir: Path, output_dir: Path):
    """Plot baseline regression results."""
    data = load_results(results_dir, "baseline_regression")
    summary = data['summary']

    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = ['Baseline', 'CGAL']
    means = [summary['baseline_mean'], summary['cgal_mean']]
    stds = [summary['baseline_std'], summary['cgal_std']]

    x = np.arange(len(conditions))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Baseline Regression: CGAL vs Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'baseline_regression.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_continual_learning(results_dir: Path, output_dir: Path):
    """Plot continual learning results."""
    data = load_results(results_dir, "continual_learning")

    # Aggregate phase results across seeds
    baseline_results = data['baseline_results']
    cgal_results = data['cgal_results']

    num_phases = len(baseline_results[0]['phase_results'])

    baseline_old_acc = []
    cgal_old_acc = []
    baseline_overall_acc = []
    cgal_overall_acc = []

    for phase_idx in range(num_phases):
        # Collect across seeds
        baseline_old = [r['phase_results'][phase_idx]['old_object_accuracy']
                       for r in baseline_results if phase_idx > 0]
        cgal_old = [r['phase_results'][phase_idx]['old_object_accuracy']
                   for r in cgal_results if phase_idx > 0]

        baseline_overall = [r['phase_results'][phase_idx]['overall_accuracy']
                           for r in baseline_results]
        cgal_overall = [r['phase_results'][phase_idx]['overall_accuracy']
                       for r in cgal_results]

        baseline_old_acc.append(np.mean(baseline_old) if baseline_old else 0.0)
        cgal_old_acc.append(np.mean(cgal_old) if cgal_old else 0.0)
        baseline_overall_acc.append(np.mean(baseline_overall))
        cgal_overall_acc.append(np.mean(cgal_overall))

    phases = np.arange(1, num_phases + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Overall accuracy
    ax1.plot(phases, baseline_overall_acc, 'o-', label='Baseline', color='#3498db', linewidth=2)
    ax1.plot(phases, cgal_overall_acc, 's-', label='CGAL', color='#e74c3c', linewidth=2)
    ax1.set_xlabel('Learning Phase', fontsize=12)
    ax1.set_ylabel('Overall Accuracy', fontsize=12)
    ax1.set_title('Continual Learning: Overall Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.0])

    # Plot 2: Old object accuracy (catastrophic forgetting)
    ax2.plot(phases[1:], baseline_old_acc[1:], 'o-', label='Baseline', color='#3498db', linewidth=2)
    ax2.plot(phases[1:], cgal_old_acc[1:], 's-', label='CGAL', color='#e74c3c', linewidth=2)
    ax2.set_xlabel('Learning Phase', fontsize=12)
    ax2.set_ylabel('Old Object Accuracy', fontsize=12)
    ax2.set_title('Catastrophic Forgetting Test', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()
    output_file = output_dir / 'continual_learning.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_noise_robustness(results_dir: Path, output_dir: Path):
    """Plot noise robustness results."""
    data = load_results(results_dir, "noise_robustness")
    summary = data['summary']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Test accuracy comparison
    conditions = ['Baseline', 'CGAL']
    means = [summary['baseline_mean'], summary['cgal_mean']]
    stds = [summary['baseline_std'], summary['cgal_std']]

    x = np.arange(len(conditions))
    ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#3498db', '#e74c3c'])
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Noise Robustness: Accuracy with 30% Noisy Modules', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10)

    # Plot 2: Trust analysis (if available)
    cgal_results = data['cgal_results']
    if cgal_results[0]['trust_analysis']:
        trust_in_clean = [r['trust_analysis']['mean_trust_in_clean'] for r in cgal_results]
        trust_in_noisy = [r['trust_analysis']['mean_trust_in_noisy'] for r in cgal_results]

        trust_types = ['Clean Modules', 'Noisy Modules']
        trust_means = [np.mean(trust_in_clean), np.mean(trust_in_noisy)]
        trust_stds = [np.std(trust_in_clean), np.std(trust_in_noisy)]

        x2 = np.arange(len(trust_types))
        ax2.bar(x2, trust_means, yerr=trust_stds, capsize=5, alpha=0.7,
               color=['#2ecc71', '#e67e22'])
        ax2.set_ylabel('Mean Trust Weight', fontsize=12)
        ax2.set_title('Learned Trust Weights', fontsize=14, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(trust_types)
        ax2.set_ylim([0, 1.2])
        ax2.grid(axis='y', alpha=0.3)

        for i, (mean, std) in enumerate(zip(trust_means, trust_stds)):
            ax2.text(i, mean + std + 0.05, f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'noise_robustness.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_few_shot_learning(results_dir: Path, output_dir: Path):
    """Plot few-shot learning results."""
    data = load_results(results_dir, "few_shot_learning")

    # Extract learning curves
    baseline_results = data['baseline_results']
    cgal_results = data['cgal_results']

    # Average across seeds
    baseline_curves = [r['learning_curve'] for r in baseline_results]
    cgal_curves = [r['learning_curve'] for r in cgal_results]

    # Align curves (they should all have same observation points)
    num_points = len(baseline_curves[0])
    observations = [point['num_observations'] for point in baseline_curves[0]]

    baseline_accs = np.array([[curve[i]['test_accuracy'] for curve in baseline_curves]
                             for i in range(num_points)])
    cgal_accs = np.array([[curve[i]['test_accuracy'] for curve in cgal_curves]
                         for i in range(num_points)])

    baseline_mean = baseline_accs.mean(axis=1)
    baseline_std = baseline_accs.std(axis=1)
    cgal_mean = cgal_accs.mean(axis=1)
    cgal_std = cgal_accs.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(observations, baseline_mean, 'o-', label='Baseline', color='#3498db', linewidth=2)
    ax.fill_between(observations, baseline_mean - baseline_std, baseline_mean + baseline_std,
                    alpha=0.2, color='#3498db')

    ax.plot(observations, cgal_mean, 's-', label='CGAL', color='#e74c3c', linewidth=2)
    ax.fill_between(observations, cgal_mean - cgal_std, cgal_mean + cgal_std,
                    alpha=0.2, color='#e74c3c')

    # Mark threshold
    threshold = data['summary']['threshold']
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label=f'{threshold:.0%} threshold')

    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Few-Shot Learning: Sample Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    output_file = output_dir / 'few_shot_learning.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_all_results(results_dir: Path = None, output_dir: Path = None):
    """Generate all plots."""
    if results_dir is None:
        results_dir = Path(__file__).parent / 'results'
    if output_dir is None:
        output_dir = results_dir / 'plots'

    output_dir.mkdir(exist_ok=True)

    print("Generating plots...")
    print("=" * 60)

    try:
        plot_baseline_regression(results_dir, output_dir)
    except Exception as e:
        print(f"  Error plotting baseline_regression: {e}")

    try:
        plot_continual_learning(results_dir, output_dir)
    except Exception as e:
        print(f"  Error plotting continual_learning: {e}")

    try:
        plot_noise_robustness(results_dir, output_dir)
    except Exception as e:
        print(f"  Error plotting noise_robustness: {e}")

    try:
        plot_few_shot_learning(results_dir, output_dir)
    except Exception as e:
        print(f"  Error plotting few_shot_learning: {e}")

    print("=" * 60)
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    plot_all_results()
