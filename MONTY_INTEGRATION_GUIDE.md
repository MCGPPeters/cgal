# CGAL-Monty Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the CGAL (Consensus-Gated Associative Learning) mechanisms into the tbp.monty codebase. The standalone CGAL implementations in this repository are complete and tested; this guide shows how to integrate them into Monty's learning loop.

**Status:** This tracking repository contains validated CGAL prototypes. Integration with Monty (Issue #1) requires:
1. Forking tbp.monty
2. Adapting CGAL modules to Monty's APIs
3. Integrating into Monty's learning loop
4. Running validation experiments

**Estimated Effort:** 2-3 weeks for a developer familiar with Monty's architecture.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Monty Architecture Overview](#monty-architecture-overview)
3. [Integration Strategy](#integration-strategy)
4. [Step-by-Step Integration](#step-by-step-integration)
5. [Testing Strategy](#testing-strategy)
6. [Validation Checklist](#validation-checklist)
7. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Prerequisites

### Required Knowledge
- Python 3.10+ development
- Understanding of Monty's learning module architecture
- Familiarity with consensus voting mechanisms
- Experience with Hydra configuration framework

### Required Setup
1. **Fork tbp.monty**: Create fork at `<your-org>/tbp.monty.cgal`
2. **Clone and setup**:
   ```bash
   git clone https://github.com/<your-org>/tbp.monty.cgal
   cd tbp.monty.cgal
   git checkout -b cgal/main
   ```
3. **Install dependencies**:
   ```bash
   conda create -n tbp.monty python=3.10
   conda activate tbp.monty
   pip install -e .
   ```
4. **Verify baseline**: Run one experiment to confirm setup works

### CGAL Modules to Integrate
All modules are in this tracking repository under `src/cgal/`:
- `learning_modules/consensus_gating.py` - Issue #2
- `learning_modules/novelty_detection.py` - Issue #3
- `learning_modules/trust_weights.py` - Issue #4
- `learning_modules/salience_replay.py` - Issue #5
- `config/*.py` - Configuration classes

---

## Monty Architecture Overview

### Key Components for CGAL Integration

#### 1. Learning Modules (`src/tbp/monty/frameworks/models/evidence_matching/learning_module.py`)

**Purpose**: Core learning unit that maintains a graph of learned patterns.

**Key Methods**:
- `update()` - Updates patterns based on observations
- `get_hypothesis()` - Returns current hypothesis about object identity
- `store_new_pattern()` - Allocates new pattern node
- `_update_graph()` - Internal graph update logic

**CGAL Integration Points**:
- Consensus gating: Modify `update()` to apply gating factor
- Novelty detection: Add novelty check before `store_new_pattern()`
- Trust weights: Use in `get_hypothesis()` weighting
- Salience replay: Hook into episode end callbacks

#### 2. Evidence Matching Agent (`src/tbp/monty/frameworks/models/evidence_matching/evidence_matching.py`)

**Purpose**: Coordinates multiple learning modules and manages voting.

**Key Methods**:
- `matching_step()` - Performs one step of matching/learning
- `get_votes()` - Collects votes from all modules
- `compute_consensus()` - Aggregates votes into consensus
- `update_learning_modules()` - Triggers learning in modules

**CGAL Integration Points**:
- Trust weights: Weight votes in `compute_consensus()`
- Consensus gating: Pass consensus to modules in `update_learning_modules()`

#### 3. Configuration System (`src/tbp/monty/conf/`)

**Purpose**: Hydra-based configuration for experiments.

**Key Files**:
- `experiment/*.yaml` - Experiment definitions
- `learning_module/*.yaml` - Learning module configs
- `model/*.yaml` - Model architecture configs

**CGAL Integration Points**:
- Add CGAL config parameters to learning module configs
- Create new experiment configs for CGAL experiments

#### 4. Sensor Modules (`src/tbp/monty/frameworks/models/sensor_modules.py`)

**Purpose**: Process sensory observations and extract features.

**CGAL Integration Points**:
- Minimal - mostly transparent to CGAL mechanisms
- May need to pass through additional metadata

---

## Integration Strategy

### Phase 1: Add Configuration (1-2 days)

**Goal**: Add CGAL config parameters without changing behavior.

**Tasks**:
1. Copy CGAL config classes to Monty
2. Add config parameters to learning module configs
3. Ensure all flags default to `False` (baseline behavior)

**Validation**: Baseline experiments run identically with CGAL flags off.

### Phase 2: Integrate Consensus Gating (2-3 days)

**Goal**: Modulate learning rates based on consensus agreement.

**Tasks**:
1. Copy `ConsensusGatingModule` to Monty
2. Modify learning module's `update()` method
3. Pass consensus from agent to learning modules
4. Apply gating factor to learning updates

**Validation**:
- Unit tests verify gating factor computation
- Integration test shows different learning rates
- Baseline preserved when flag is `False`

### Phase 3: Integrate Novelty Detection (2-3 days)

**Goal**: Route novel observations to new pattern allocation.

**Tasks**:
1. Copy `NoveltyDetector` to Monty
2. Add novelty computation to learning module
3. Gate `store_new_pattern()` on novelty threshold
4. Track novelty scores per episode

**Validation**:
- Novel patterns trigger new allocation
- Familiar patterns don't allocate unnecessarily
- Baseline preserved when flag is `False`

### Phase 4: Integrate Trust Weights (3-4 days)

**Goal**: Learn per-module trust based on voting accuracy.

**Tasks**:
1. Copy `TrustWeightsModule` to Monty
2. Add trust matrix to evidence matching agent
3. Weight votes by trust in consensus computation
4. Update trust after each voting round

**Validation**:
- Trust matrix learns correct weights
- Unreliable modules down-weighted
- Baseline preserved when flag is `False`

### Phase 5: Integrate Salience Replay (3-4 days)

**Goal**: Replay high-salience patterns during offline phases.

**Tasks**:
1. Copy `SalienceReplayModule` to Monty
2. Tag patterns with salience scores
3. Add replay phase between episodes
4. Sample and replay high-salience patterns

**Validation**:
- High-salience patterns replayed more often
- Replay phase doesn't break learning
- Baseline preserved when flag is `False`

### Phase 6: Run Experiments (3-5 days)

**Goal**: Validate CGAL improves learning on target tasks.

**Tasks**:
1. Create CGAL experiment configs
2. Run 4 experiment types (baseline, continual, noise, few-shot)
3. Collect and analyze results
4. Update CGAL_RESULTS.md with findings

**Validation**:
- Results show directional improvement (or not)
- Statistical significance on key metrics
- Honest reporting of positive and negative results

---

## Step-by-Step Integration

### Step 1: Fork and Setup (Day 1)

```bash
# 1. Fork tbp.monty on GitHub to <your-org>/tbp.monty.cgal

# 2. Clone your fork
git clone https://github.com/<your-org>/tbp.monty.cgal
cd tbp.monty.cgal

# 3. Create CGAL branch
git checkout -b cgal/main

# 4. Setup environment
conda create -n tbp.monty python=3.10
conda activate tbp.monty
pip install -e .

# 5. Run baseline experiment
python run.py experiment=randrot_noise_10distinctobj_surf_agent

# 6. Document setup in CGAL_NOTES.md
cat > CGAL_NOTES.md << 'EOF'
# CGAL Experimental Fork

This is an experimental fork of tbp.monty for testing Consensus-Gated
Associative Learning (CGAL) mechanisms.

**Parent Repository:** https://github.com/thousandbrainsproject/tbp.monty
**Tracking Repository:** https://github.com/MCGPPeters/cgal
**Branch:** cgal/main

## CGAL Mechanisms

1. Consensus-Gated Plasticity (Issue #2)
2. Novelty Detection (Issue #3)
3. Learned Trust Weights (Issue #4)
4. Salience-Tagged Replay (Issue #5)

All mechanisms are behind feature flags and default to disabled.
EOF

git add CGAL_NOTES.md
git commit -m "Add CGAL_NOTES.md"
git push origin cgal/main
```

### Step 2: Copy CGAL Modules (Day 1-2)

```bash
# 1. Create CGAL directory in Monty
mkdir -p src/tbp/monty/frameworks/models/cgal

# 2. Copy CGAL modules from tracking repo
# (Assuming you have the tracking repo cloned at ~/cgal)
cp ~/cgal/src/cgal/config/*.py src/tbp/monty/frameworks/models/cgal/
cp ~/cgal/src/cgal/learning_modules/*.py src/tbp/monty/frameworks/models/cgal/

# 3. Create __init__.py
cat > src/tbp/monty/frameworks/models/cgal/__init__.py << 'EOF'
"""CGAL (Consensus-Gated Associative Learning) mechanisms."""

from .consensus_gating import ConsensusGatingModule, ConsensusGatingConfig
from .novelty_detection import NoveltyDetector, NoveltyDetectionConfig
from .trust_weights import TrustWeightsModule, TrustWeightsConfig
from .salience_replay import SalienceReplayModule, SalienceReplayConfig

__all__ = [
    "ConsensusGatingModule",
    "ConsensusGatingConfig",
    "NoveltyDetector",
    "NoveltyDetectionConfig",
    "TrustWeightsModule",
    "TrustWeightsConfig",
    "SalienceReplayModule",
    "SalienceReplayConfig",
]
EOF

git add src/tbp/monty/frameworks/models/cgal/
git commit -m "Add CGAL modules from tracking repository"
```

### Step 3: Add Configuration Parameters (Day 2)

**File:** `src/tbp/monty/conf/learning_module/default.yaml`

Add CGAL parameters:

```yaml
# CGAL Mechanism Flags (all default to False for baseline)
consensus_gated_plasticity: false
hypothesis_novelty_detection: false
learned_trust_weights: false
salience_tagged_replay: false

# CGAL Hyperparameters
consensus_alpha: 0.7              # Weight of consensus in gating
consensus_baseline_rate: 0.5      # Baseline learning rate
novelty_threshold: 0.7            # Threshold for novelty routing
trust_learning_rate: 0.05         # Trust weight update rate
trust_min: 0.1                    # Minimum trust value
trust_max: 1.0                    # Maximum trust value
salience_alpha_consensus: 0.5     # Consensus weight in salience
salience_alpha_novelty: 0.5       # Novelty weight in salience
salience_decay_rate: 0.99         # Salience decay per step
replay_interval: 10               # Episodes between replay
num_replay_samples: 50            # Patterns to replay per phase
```

Commit:
```bash
git add src/tbp/monty/conf/learning_module/default.yaml
git commit -m "Add CGAL configuration parameters"
```

### Step 4: Integrate Consensus Gating (Days 3-4)

**File:** `src/tbp/monty/frameworks/models/evidence_matching/learning_module.py`

**Changes needed:**

1. **Import CGAL modules:**
```python
from tbp.monty.frameworks.models.cgal import (
    ConsensusGatingModule,
    ConsensusGatingConfig
)
```

2. **Initialize in `__init__`:**
```python
def __init__(self, config):
    # ... existing initialization ...

    # Initialize CGAL consensus gating
    if config.get('consensus_gated_plasticity', False):
        cgal_config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=config.get('consensus_alpha', 0.7),
            baseline_rate=config.get('consensus_baseline_rate', 0.5)
        )
        self.consensus_gating = ConsensusGatingModule(cgal_config)
    else:
        self.consensus_gating = None
```

3. **Modify `update()` method:**
```python
def update(self, observation, consensus=None):
    """Update learning with optional consensus gating.

    Args:
        observation: Current observation data
        consensus: Optional consensus from voting (for CGAL)
    """
    # Compute gating factor if CGAL enabled
    gating_factor = 1.0  # Default: no gating

    if self.consensus_gating is not None and consensus is not None:
        hypothesis = self.get_hypothesis()
        gating_factor = self.consensus_gating.compute_gating_factor(
            hypothesis, consensus
        )

    # Apply gating to learning rate
    effective_lr = self.learning_rate * gating_factor

    # Perform update with gated learning rate
    self._update_graph(observation, learning_rate=effective_lr)
```

**Test:**
```python
# tests/unit/test_consensus_gating_integration.py
def test_consensus_gating_enabled():
    """Test that consensus gating modulates learning rate."""
    config = {'consensus_gated_plasticity': True, 'consensus_alpha': 0.7}
    lm = LearningModule(config)

    # High agreement should increase learning
    hypothesis = {'object_id': 'cup'}
    consensus = {'object_id': 'cup'}
    lm.update(observation, consensus)
    # Assert learning rate was increased

def test_consensus_gating_baseline():
    """Test that disabling flag preserves baseline."""
    config = {'consensus_gated_plasticity': False}
    lm = LearningModule(config)

    lm.update(observation, consensus)
    # Assert learning rate unchanged
```

Commit:
```bash
git add src/tbp/monty/frameworks/models/evidence_matching/learning_module.py
git add tests/unit/test_consensus_gating_integration.py
git commit -m "Integrate consensus gating into learning module"
```

### Step 5: Integrate Novelty Detection (Days 5-6)

**File:** Same `learning_module.py`

**Changes:**

1. **Initialize novelty detector:**
```python
from tbp.monty.frameworks.models.cgal import (
    NoveltyDetector,
    NoveltyDetectionConfig
)

def __init__(self, config):
    # ... existing ...

    if config.get('hypothesis_novelty_detection', False):
        novelty_config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            novelty_threshold=config.get('novelty_threshold', 0.7)
        )
        self.novelty_detector = NoveltyDetector(novelty_config)
    else:
        self.novelty_detector = None
```

2. **Add novelty check:**
```python
def should_allocate_new_pattern(self, observation):
    """Determine if new pattern should be allocated.

    Returns:
        bool: True if pattern should be allocated
    """
    if self.novelty_detector is None:
        # Baseline: use existing logic
        return self._baseline_allocation_logic(observation)

    # CGAL: check novelty
    hypothesis_scores = self.get_hypothesis_distribution()
    is_novel, novelty_score = self.novelty_detector.is_novel(hypothesis_scores)

    return is_novel
```

### Step 6: Integrate Trust Weights (Days 7-9)

**File:** `src/tbp/monty/frameworks/models/evidence_matching/evidence_matching.py`

**Changes:**

1. **Initialize trust module in agent:**
```python
from tbp.monty.frameworks.models.cgal import (
    TrustWeightsModule,
    TrustWeightsConfig
)

def __init__(self, config):
    # ... existing ...

    if config.get('learned_trust_weights', False):
        trust_config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=config.get('trust_learning_rate', 0.05),
            trust_min=config.get('trust_min', 0.1),
            trust_max=config.get('trust_max', 1.0)
        )
        self.trust_weights = TrustWeightsModule(trust_config)

        # Initialize trust matrix with module IDs
        module_ids = [lm.id for lm in self.learning_modules]
        self.trust_weights.initialize_trust(module_ids)
    else:
        self.trust_weights = None
```

2. **Weight votes by trust:**
```python
def compute_consensus(self, votes):
    """Compute consensus from module votes.

    Args:
        votes: Dict mapping module_id to vote

    Returns:
        Consensus vote
    """
    if self.trust_weights is None:
        # Baseline: uniform weighting
        return self._uniform_consensus(votes)

    # CGAL: weight by trust
    weighted_votes = {}
    for from_id, vote in votes.items():
        # Weight this vote by trust from other modules
        neighbor_votes = {k: v for k, v in votes.items() if k != from_id}
        weights = self.trust_weights.weight_votes(from_id, neighbor_votes)

        # Aggregate with weights
        weighted_votes[from_id] = (vote, weights)

    consensus = self._weighted_consensus(weighted_votes)

    # Update trust based on agreement with consensus
    self.trust_weights.update_all_trust(votes, consensus)

    return consensus
```

### Step 7: Integrate Salience Replay (Days 10-12)

**File:** Multiple files for replay integration

**Changes:**

1. **Initialize in learning module:**
```python
from tbp.monty.frameworks.models.cgal import (
    SalienceReplayModule,
    SalienceReplayConfig
)

def __init__(self, config):
    # ... existing ...

    if config.get('salience_tagged_replay', False):
        replay_config = SalienceReplayConfig(
            salience_tagged_replay=True,
            replay_interval=config.get('replay_interval', 10),
            num_replay_samples=config.get('num_replay_samples', 50)
        )
        self.salience_replay = SalienceReplayModule(replay_config)
        self.pattern_id_counter = 0
    else:
        self.salience_replay = None
```

2. **Tag patterns with salience:**
```python
def update(self, observation, consensus=None, novelty_score=0.0):
    """Update with salience tagging."""
    # ... existing update logic ...

    if self.salience_replay is not None:
        # Compute agreement with consensus
        hypothesis = self.get_hypothesis()
        if self.consensus_gating:
            agreement = self.consensus_gating.compute_agreement(
                hypothesis, consensus
            )
        else:
            agreement = 1.0 if hypothesis == consensus else 0.0

        # Add pattern and update salience
        pattern = self.salience_replay.add_pattern(
            self.pattern_id_counter,
            {'observation': observation, 'hypothesis': hypothesis}
        )
        self.salience_replay.update_salience(
            self.pattern_id_counter,
            agreement,
            novelty_score
        )
        self.pattern_id_counter += 1
        self.salience_replay.on_step()
```

3. **Add replay phase:**
```python
def on_episode_end(self):
    """Called at episode end - trigger replay if enabled."""
    if self.salience_replay is None:
        return

    def replay_learning_function(pattern_data):
        """Re-run learning on replayed pattern."""
        observation = pattern_data['observation']
        self._update_graph(observation, learning_rate=0.1)

    self.salience_replay.on_episode_end(replay_learning_function)
```

### Step 8: Create CGAL Experiments (Days 13-15)

**Create:** `src/tbp/monty/conf/experiment/cgal/`

**Files:**
1. `baseline_regression.yaml` - Standard YCB with CGAL off
2. `continual_learning.yaml` - Incremental object introduction
3. `noise_robustness.yaml` - 30% modules with noise
4. `few_shot.yaml` - Limited observations per object

**Example config:**
```yaml
# continual_learning.yaml
defaults:
  - /experiment/default
  - override /learning_module: cgal_enabled

experiment_class: ContinualLearningExperiment
num_objects: 15
objects_per_phase: 5
num_episodes: 100

learning_module:
  consensus_gated_plasticity: true
  hypothesis_novelty_detection: true
  learned_trust_weights: true
  salience_tagged_replay: true

  # Tuned hyperparameters
  consensus_alpha: 0.7
  novelty_threshold: 0.7
  trust_learning_rate: 0.05
  replay_interval: 5
```

---

## Testing Strategy

### Unit Tests
Location: `tests/unit/test_cgal_*.py`

**Test each mechanism independently:**
1. `test_cgal_consensus_gating.py` - Gating factor computation
2. `test_cgal_novelty.py` - Novelty detection logic
3. `test_cgal_trust.py` - Trust matrix updates
4. `test_cgal_replay.py` - Replay sampling and execution

**Test baseline preservation:**
```python
def test_all_cgal_flags_disabled():
    """Verify baseline behavior when all CGAL flags are False."""
    config = {
        'consensus_gated_plasticity': False,
        'hypothesis_novelty_detection': False,
        'learned_trust_weights': False,
        'salience_tagged_replay': False
    }

    # Run experiment with CGAL off
    results_cgal_off = run_experiment(config)

    # Run experiment with default config (no CGAL)
    results_baseline = run_experiment({})

    # Assert byte-identical behavior
    assert results_cgal_off == results_baseline
```

### Integration Tests
Location: `tests/integration/test_cgal_integration.py`

**Test mechanisms work together:**
```python
def test_cgal_full_stack():
    """Test all CGAL mechanisms enabled together."""
    config = {
        'consensus_gated_plasticity': True,
        'hypothesis_novelty_detection': True,
        'learned_trust_weights': True,
        'salience_tagged_replay': True
    }

    agent = create_agent(config)

    # Run episode
    for step in range(100):
        observation = get_observation()
        agent.step(observation)

    agent.on_episode_end()

    # Verify all mechanisms activated
    assert agent.consensus_gating.num_updates > 0
    assert agent.novelty_detector.novelty_count > 0
    assert agent.trust_weights.get_statistics()['num_updates'] > 0
    assert len(agent.salience_replay.replay_history) > 0
```

### Experiment Tests
Location: `tests/experiments/`

**Run mini versions of experiments:**
```python
def test_continual_learning_experiment():
    """Run small-scale continual learning experiment."""
    config = load_config('experiment/cgal/continual_learning.yaml')
    config.num_objects = 6  # Reduced for testing
    config.objects_per_phase = 2

    results = run_experiment(config)

    # Verify experiment completed
    assert results['num_phases'] == 3
    assert 'forgetting_metric' in results
```

---

## Validation Checklist

### Phase 1: Setup ✅
- [ ] Fork created at `<your-org>/tbp.monty.cgal`
- [ ] Branch `cgal/main` created
- [ ] Environment setup complete
- [ ] Baseline experiment runs successfully
- [ ] `CGAL_NOTES.md` added

### Phase 2: Code Integration ✅
- [ ] CGAL modules copied to Monty
- [ ] Configuration parameters added
- [ ] Consensus gating integrated
- [ ] Novelty detection integrated
- [ ] Trust weights integrated
- [ ] Salience replay integrated

### Phase 3: Testing ✅
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Baseline preservation verified
- [ ] Each mechanism works individually
- [ ] All mechanisms work together

### Phase 4: Experiments ✅
- [ ] Experiment configs created
- [ ] Baseline regression runs
- [ ] Continual learning experiment runs
- [ ] Noise robustness experiment runs
- [ ] Few-shot learning experiment runs

### Phase 5: Analysis ✅
- [ ] Results collected for all experiments
- [ ] Hypothesis H1 tested (catastrophic forgetting)
- [ ] Hypothesis H2 tested (noise robustness)
- [ ] Hypothesis H3 tested (sample efficiency)
- [ ] Hypothesis H4 tested (no regression)
- [ ] Statistical analysis complete
- [ ] Plots generated

### Phase 6: Documentation ✅
- [ ] Results documented in `CGAL_RESULTS.md`
- [ ] Honest interpretation of findings
- [ ] Limitations clearly stated
- [ ] Next steps recommended
- [ ] Code commented and documented

---

## Common Issues and Solutions

### Issue 1: Import Errors
**Problem:** CGAL modules not found after copying.

**Solution:**
```bash
# Ensure __init__.py exists
touch src/tbp/monty/frameworks/models/cgal/__init__.py

# Reinstall package
pip install -e .
```

### Issue 2: Config Not Recognized
**Problem:** Hydra doesn't recognize new CGAL parameters.

**Solution:**
```yaml
# Add to config YAML with proper nesting
learning_module:
  consensus_gated_plasticity: false  # Correct

# Not at top level:
consensus_gated_plasticity: false  # Wrong
```

### Issue 3: Baseline Behavior Changed
**Problem:** Experiments behave differently even with flags off.

**Solution:**
```python
# Ensure all CGAL code is properly gated
if self.consensus_gating is not None:  # Good
    # CGAL code here

# Don't check flag again:
if self.config.consensus_gated_plasticity:  # Redundant
```

### Issue 4: Trust Matrix Dimension Mismatch
**Problem:** Module IDs don't match trust matrix dimensions.

**Solution:**
```python
# Always initialize trust with actual module IDs
module_ids = [lm.id for lm in self.learning_modules]
self.trust_weights.initialize_trust(module_ids)

# Don't use range(num_modules)
```

### Issue 5: Replay Phase Too Slow
**Problem:** Replay takes too long between episodes.

**Solution:**
```python
# Reduce replay samples
replay_config = SalienceReplayConfig(
    num_replay_samples=20,  # Reduce from 50
    replay_interval=10      # Replay less frequently
)
```

### Issue 6: Novelty Always High/Low
**Problem:** Novelty detector doesn't distinguish patterns.

**Solution:**
```python
# Check hypothesis distribution format
hypothesis_scores = {
    'object_1': 10.0,  # Should be dict of scores
    'object_2': 2.0,
    'object_3': 1.0
}

# Not single value:
hypothesis_score = 0.8  # Wrong format
```

---

## Next Steps After Integration

1. **Run Baseline Validation**
   - Verify all experiments with CGAL off match original Monty
   - Document any unavoidable differences

2. **Run CGAL Experiments**
   - Execute all 4 experiment types with 5 seeds each
   - Collect comprehensive metrics

3. **Analyze Results**
   - Compare baseline vs CGAL quantitatively
   - Perform statistical tests
   - Create visualization plots

4. **Document Findings**
   - Update `CGAL_RESULTS.md` in tracking repo
   - Include positive and negative results
   - Provide honest interpretation

5. **Iterate**
   - Based on results, tune hyperparameters
   - Try ablations (one mechanism at a time)
   - Explore mechanism interactions

---

## Additional Resources

### Reference Implementations
- Standalone CGAL: `https://github.com/MCGPPeters/cgal`
- Implementation guides: `ISSUE_{2,3,4,5}_IMPLEMENTATION.md`
- Test suites: `tests/test_*.py` in tracking repo

### Monty Documentation
- Monty repository: `https://github.com/thousandbrainsproject/tbp.monty`
- Getting started: `https://thousandbrainsproject.readme.io/docs/getting-started`
- Paper: Leadholm et al. 2025, arXiv:2507.04494

### Contact
- For questions about CGAL mechanisms: See tracking repo issues
- For questions about Monty: See Monty repository discussions
- For integration issues: Document in `CGAL_NOTES.md` in your fork

---

## Conclusion

This integration guide provides a comprehensive roadmap for adding CGAL mechanisms to Monty. The standalone implementations are complete and tested; integration primarily involves:

1. Adapting data structures to Monty's format
2. Hooking into Monty's learning loop at appropriate points
3. Ensuring baseline behavior is preserved when flags are off
4. Running validation experiments to test effectiveness

The estimated 2-3 week timeline assumes a developer familiar with Monty's architecture. The work is straightforward but requires careful attention to:
- Proper feature flagging
- Correct data flow between components
- Thorough testing at each integration step

Following this guide systematically will ensure a successful integration and provide the empirical evidence needed to validate (or refute) CGAL's effectiveness in practice.
