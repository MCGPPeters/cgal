"""Simple test to verify CGAL modules work."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cgal.config import ConsensusGatingConfig, TrustWeightsConfig
from cgal.learning_modules import ConsensusGatingModule, TrustWeightsModule

print("Testing CGAL modules...")

# Test consensus gating
config = ConsensusGatingConfig(consensus_gated_plasticity=True)
cgm = ConsensusGatingModule(config)
hyp = {'object_id': 'cup'}
consensus = {'object_id': 'cup'}
agreement = cgm.compute_agreement(hyp, consensus)
gating = cgm.compute_gating_factor(hyp, consensus)
print(f"✓ Consensus gating: agreement={agreement:.2f}, gating={gating:.2f}")

# Test trust weights
trust_config = TrustWeightsConfig(learned_trust_weights=True)
trust_module = TrustWeightsModule(trust_config)
trust_module.initialize_trust([1, 2, 3])
trust_module.update_trust(1, 2, 1.0)
trust = trust_module.get_trust(1, 2)
print(f"✓ Trust weights: trust(1→2)={trust:.2f}")

print("\nAll modules work correctly!")
