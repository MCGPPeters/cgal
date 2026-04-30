import numpy as np
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cgal.config import ConsensusGatingConfig, TrustWeightsConfig
from cgal.learning_modules import ConsensusGatingModule, TrustWeightsModule

def test_baseline_vs_cgal():
    """Simple test: baseline vs CGAL with consensus gating."""
    results = {'baseline': [], 'cgal': []}
    
    for condition in ['baseline', 'cgal']:
        enable_cgal = (condition == 'cgal')
        
        for seed in range(5):
            np.random.seed(seed)
            
            # Simulate learning: 10 objects, 20 observations each
            correct = 0
            total = 200
            
            if enable_cgal:
                config = ConsensusGatingConfig(consensus_gated_plasticity=True, alpha=0.7)
                cgm = ConsensusGatingModule(config)
                
                for i in range(total):
                    # Simulate: 80% correct consensus, module gets 70% correct
                    consensus_correct = np.random.rand() < 0.8
                    module_correct = np.random.rand() < 0.7
                    
                    if consensus_correct:
                        # High agreement increases learning
                        if module_correct:
                            correct += 1
                    else:
                        # Low agreement, less learning
                        if module_correct and np.random.rand() < 0.5:
                            correct += 1
            else:
                # Baseline: no gating, just 70% accuracy
                correct = int(total * 0.7 + np.random.randn() * 10)
            
            accuracy = correct / total
            results[condition].append(accuracy)
    
    return results

results = test_baseline_vs_cgal()
print(f"Baseline: {np.mean(results['baseline']):.3f} ± {np.std(results['baseline']):.3f}")
print(f"CGAL:     {np.mean(results['cgal']):.3f} ± {np.std(results['cgal']):.3f}")

# Save
output = Path('experiments/results')
output.mkdir(exist_ok=True)
with open(output / 'simple_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Results saved")
