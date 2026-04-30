# CGAL Monty Integration - Quick Start

This is a condensed version of the full [MONTY_INTEGRATION_GUIDE.md](MONTY_INTEGRATION_GUIDE.md). Use this for a quick overview, then refer to the full guide for details.

## 🚀 Quick Start (30 minutes)

### 1. Fork and Setup
```bash
# Fork https://github.com/thousandbrainsproject/tbp.monty to your org
git clone https://github.com/<your-org>/tbp.monty.cgal
cd tbp.monty.cgal
git checkout -b cgal/main

conda create -n tbp.monty python=3.10
conda activate tbp.monty
pip install -e .
```

### 2. Copy CGAL Modules
```bash
# From this tracking repo, copy to your Monty fork:
mkdir -p src/tbp/monty/frameworks/models/cgal/
cp <path-to-cgal-tracking>/src/cgal/config/*.py src/tbp/monty/frameworks/models/cgal/
cp <path-to-cgal-tracking>/src/cgal/learning_modules/*.py src/tbp/monty/frameworks/models/cgal/
```

### 3. Add Config (5 lines)
**File:** `src/tbp/monty/conf/learning_module/default.yaml`

```yaml
# Add these lines:
consensus_gated_plasticity: false
hypothesis_novelty_detection: false
learned_trust_weights: false
salience_tagged_replay: false
```

### 4. Integrate (Key Changes)

**Consensus Gating** (in learning_module.py):
```python
# Init: self.consensus_gating = ConsensusGatingModule(config) if enabled

# In update():
gating_factor = self.consensus_gating.compute_gating_factor(hypothesis, consensus)
effective_lr = self.learning_rate * gating_factor
```

**Novelty Detection** (in learning_module.py):
```python
# Init: self.novelty_detector = NoveltyDetector(config) if enabled

# Before allocating pattern:
is_novel, score = self.novelty_detector.is_novel(hypothesis_scores)
if is_novel: allocate_new_pattern()
```

**Trust Weights** (in evidence_matching.py):
```python
# Init: self.trust_weights = TrustWeightsModule(config) if enabled

# In compute_consensus():
weights = self.trust_weights.weight_votes(from_id, votes)
consensus = weighted_voting(votes, weights)
self.trust_weights.update_all_trust(votes, consensus)
```

**Salience Replay** (in learning_module.py):
```python
# Init: self.salience_replay = SalienceReplayModule(config) if enabled

# Tag patterns:
self.salience_replay.update_salience(pattern_id, agreement, novelty)

# On episode end:
self.salience_replay.on_episode_end(replay_learning_function)
```

## 📋 Integration Checklist

- [ ] Fork tbp.monty
- [ ] Copy CGAL modules
- [ ] Add config parameters
- [ ] Integrate consensus gating
- [ ] Integrate novelty detection
- [ ] Integrate trust weights
- [ ] Integrate salience replay
- [ ] Run baseline (verify no change)
- [ ] Run CGAL experiments
- [ ] Analyze results

## 🎯 Expected Timeline

- **Days 1-2:** Setup and config
- **Days 3-4:** Consensus gating
- **Days 5-6:** Novelty detection
- **Days 7-9:** Trust weights
- **Days 10-12:** Salience replay
- **Days 13-15:** Experiments
- **Total:** 2-3 weeks

## 📖 Full Documentation

- [Complete Integration Guide](MONTY_INTEGRATION_GUIDE.md) - 30+ pages
- [CGAL Mechanism Docs](ISSUE_2_IMPLEMENTATION.md) - Individual mechanism details
- [Results Writeup](CGAL_RESULTS.md) - Current status and findings

## ⚠️ Key Principles

1. **Feature Flags:** All CGAL code must be behind config flags
2. **Baseline Preservation:** With flags=False, behavior must be identical
3. **Test Each Step:** Unit test → Integration test → Experiment
4. **Honest Reporting:** Document both positive and negative results

## 🆘 Need Help?

- See [Common Issues](MONTY_INTEGRATION_GUIDE.md#common-issues-and-solutions) in full guide
- Check tracking repo: https://github.com/MCGPPeters/cgal
- Reference implementations: `src/cgal/` in tracking repo
- Unit tests: `tests/test_*.py` for examples

## 🎉 Success Criteria

✅ Baseline experiments run identically with CGAL off
✅ CGAL experiments complete successfully
✅ Results show directional improvement (or document why not)
✅ Statistical analysis supports conclusions
✅ Findings honestly reported in CGAL_RESULTS.md
