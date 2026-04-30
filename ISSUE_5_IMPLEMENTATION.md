# Issue #5 Implementation: Salience-Tagged Replay

## Overview

This document describes the implementation of **salience-tagged replay** in the CGAL framework (Issue #5). This mechanism tags patterns with importance scores (salience) based on consensus agreement and novelty, then preferentially replays high-salience patterns during offline consolidation phases.

## Core Concept

In biological systems, important experiences are consolidated during sleep through replay. Similarly, CGAL's salience-tagged replay implements:

1. **Salience Tagging**: Patterns accumulate salience based on:
   - **Consensus agreement**: How well the pattern aligned with network consensus
   - **Novelty**: How unexpected or informative the pattern was

2. **Preferential Replay**: During offline phases (analogous to sleep), high-salience patterns are replayed more often to deepen consolidation

3. **Homeostatic Downscaling**: Un-replayed patterns gradually weaken, preventing runaway growth

## Formula

### Salience Update
```
salience[t+1] = salience[t] * decay_rate + α_consensus * agreement + α_novelty * novelty
```

Where:
- **salience[t]**: Current salience value (importance score)
- **decay_rate**: Salience decay per step (default: 0.99)
- **α_consensus**: Weight for consensus agreement (default: 0.5)
- **α_novelty**: Weight for novelty score (default: 0.5)
- **agreement**: Agreement with consensus, in [0, 1] (from Issue #2)
- **novelty**: Novelty score, in [0, 1] (from Issue #3)

### Replay Sampling
Patterns are sampled with probability proportional to their salience:
```
P(pattern_i) = salience[i] / Σ(salience[j])
```

Higher salience → more likely to be replayed.

## Implementation

### 1. Configuration (`SalienceReplayConfig`)

**Location**: `src/cgal/config/salience_replay_config.py`

```python
@dataclass
class SalienceReplayConfig:
    salience_tagged_replay: bool = False     # Feature flag
    alpha_consensus: float = 0.5             # Weight for consensus
    alpha_novelty: float = 0.5               # Weight for novelty
    decay_rate: float = 0.99                 # Salience decay per step
    replay_interval: int = 10                # Episodes between replay
    num_replay_samples: int = 50             # Patterns per replay phase
    homeostatic_downscaling: bool = True     # Enable downscaling
    homeostatic_factor: float = 0.99         # Downscaling factor
    enable_logging: bool = True              # Enable logging
    log_interval: int = 10                   # Episodes between logging
```

**Parameters**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `salience_tagged_replay` | False | bool | Feature flag for enabling salience replay |
| `alpha_consensus` | 0.5 | [0, 1] | Weight for consensus agreement in salience |
| `alpha_novelty` | 0.5 | [0, 1] | Weight for novelty score in salience |
| `decay_rate` | 0.99 | (0, 1] | Salience decay rate per step |
| `replay_interval` | 10 | ≥0 | Episodes between offline replay phases |
| `num_replay_samples` | 50 | >0 | Number of patterns to replay per phase |
| `homeostatic_downscaling` | True | bool | Whether to apply homeostatic downscaling |
| `homeostatic_factor` | 0.99 | (0, 1] | Downscaling factor for un-replayed patterns |
| `enable_logging` | True | bool | Whether to log replay statistics |
| `log_interval` | 10 | ≥0 | Episodes between logging |

### 2. Pattern Class

**Location**: `src/cgal/learning_modules/salience_replay.py`

```python
class Pattern:
    """Represents a stored pattern with salience tracking."""

    def __init__(self, pattern_id: int, data: Any):
        self.pattern_id = pattern_id
        self.data = data              # Pattern data (features, activations, etc.)
        self.salience: float = 0.0    # Current salience value
        self.last_update_step: int = 0  # For lazy decay
```

### 3. Salience Replay Module (`SalienceReplayModule`)

**Location**: `src/cgal/learning_modules/salience_replay.py`

**Key Methods**:

#### `add_pattern(pattern_id: int, data: Any) -> Pattern`
Add a new pattern to storage with initial salience = 0.0.

```python
pattern = replay_module.add_pattern(
    pattern_id=1,
    data={'features': [0.1, 0.2, 0.3]}
)
```

#### `update_salience(pattern_id: int, agreement_score: float, novelty_score: float)`
Update salience based on consensus agreement and novelty.

```python
replay_module.update_salience(
    pattern_id=1,
    agreement_score=0.9,  # From Issue #2 (consensus gating)
    novelty_score=0.3     # From Issue #3 (novelty detection)
)
# salience += 0.5 * 0.9 + 0.5 * 0.3 = 0.6
```

#### `get_salience(pattern_id: int) -> float`
Get current salience with lazy decay applied.

```python
salience = replay_module.get_salience(pattern_id=1)
# Automatically applies decay based on time since last update
```

#### `sample_patterns_by_salience(num_samples: int) -> List[Pattern]`
Sample patterns weighted by salience.

```python
samples = replay_module.sample_patterns_by_salience(num_samples=10)
# Returns 10 patterns, with probability proportional to salience
```

#### `run_replay_phase(learning_function: Callable)`
Run an offline replay phase, sampling and replaying high-salience patterns.

```python
def learning_fn(pattern_data):
    # Re-run learning update on pattern
    model.update(pattern_data)

replay_module.run_replay_phase(learning_fn)
```

#### `on_episode_end(learning_function: Optional[Callable])`
Called at episode end. Runs replay if interval is reached.

```python
replay_module.on_episode_end(learning_function)
# Automatically runs replay every `replay_interval` episodes
```

#### Other utilities:
- `on_step()`: Increment time counter
- `get_statistics()`: Get statistics about salience and replay
- `get_top_salient_patterns(k)`: Get top-k most salient patterns
- `reset()`: Reset all patterns and statistics

## Integration with CGAL

### Typical Usage Flow

1. **Initialization**:
```python
from cgal.config import SalienceReplayConfig
from cgal.learning_modules import SalienceReplayModule

config = SalienceReplayConfig(salience_tagged_replay=True)
replay_module = SalienceReplayModule(config)
```

2. **During Learning** (each observation):
```python
# Add pattern to storage
pattern = replay_module.add_pattern(pattern_id, pattern_data)

# Update salience based on consensus and novelty
agreement = consensus_module.compute_agreement(hypothesis, consensus)
novelty = novelty_detector.score(pattern_data)

replay_module.update_salience(pattern_id, agreement, novelty)
replay_module.on_step()
```

3. **Episode End** (triggers replay periodically):
```python
def learning_function(pattern_data):
    # Re-run learning on pattern
    learning_module.update(pattern_data)

replay_module.on_episode_end(learning_function)
# Automatically runs replay every `replay_interval` episodes
```

### Integration with Other CGAL Components

Salience-tagged replay integrates with:

1. **Consensus-Gated Plasticity (Issue #2)**:
   - Agreement scores from consensus gating are used to compute salience
   - High-agreement patterns get higher salience
   - Formula: `salience += α_consensus * agreement_score`

2. **Novelty Detection (Issue #3)**:
   - Novelty scores are used to compute salience
   - Novel patterns get higher salience
   - Formula: `salience += α_novelty * novelty_score`

3. **Learned Trust Weights (Issue #4)**:
   - Trust evolution patterns could be tagged with salience
   - Salient voting rounds could be replayed to refine trust estimates

**Combined Example**:
```python
# During observation
agreement = consensus_module.compute_agreement(hypothesis, consensus)
novelty = novelty_detector.score_hypothesis(hypothesis)
gating_factor = consensus_module.compute_gating_factor(hypothesis, consensus)

# Update pattern with gated learning
learning_magnitude = base_lr * gating_factor
pattern.update(learning_magnitude)

# Tag pattern with salience
replay_module.update_salience(pattern.id, agreement, novelty)
```

## Behavioral Properties

### 1. Lazy Salience Decay

Salience decays exponentially over time without requiring eager updates:
- Decay is applied when `get_salience()` is called
- Formula: `salience_decayed = salience * (decay_rate ^ steps_since_update)`
- Benefits: Efficient, no need to update all patterns every step

### 2. Weighted Sampling

High-salience patterns are replayed more often:
- Sampling probability: `P(pattern) ∝ salience`
- Allows highly salient patterns to be sampled multiple times per phase
- Low-salience patterns are rarely replayed

### 3. Homeostatic Downscaling

Prevents runaway growth of pattern strengths:
- During replay, all patterns are downscaled by `homeostatic_factor`
- Replayed patterns get re-strengthened by learning
- Un-replayed patterns gradually weaken
- Net effect: Value-aligned consolidation

### 4. Dual-Signal Salience

Salience combines two sources:
- **Consensus agreement**: Patterns that matched network consensus
- **Novelty**: Patterns that were unexpected/informative
- Weight balance controlled by `α_consensus` and `α_novelty`

### 5. Baseline Preservation

When `salience_tagged_replay=False`:
- Salience is not updated
- Replay phases are skipped
- Preserves baseline CGAL behavior (no replay)

## Examples

### Example 1: Basic Usage

```python
from cgal.config import SalienceReplayConfig
from cgal.learning_modules import SalienceReplayModule

# Configure
config = SalienceReplayConfig(
    salience_tagged_replay=True,
    alpha_consensus=0.6,
    alpha_novelty=0.4,
    replay_interval=5
)

# Initialize
replay_module = SalienceReplayModule(config)

# Add patterns
for i in range(10):
    replay_module.add_pattern(i, {'pattern_id': i})

# Simulate learning over 20 episodes
for episode in range(20):
    # Update saliences
    for i in range(10):
        agreement = compute_agreement(i)  # From Issue #2
        novelty = compute_novelty(i)      # From Issue #3
        replay_module.update_salience(i, agreement, novelty)

    # Step
    replay_module.on_step()

    # Episode end (replay every 5 episodes)
    replay_module.on_episode_end(learning_function)
```

### Example 2: Salience Components

```python
config = SalienceReplayConfig(
    salience_tagged_replay=True,
    alpha_consensus=0.7,  # Emphasize consensus
    alpha_novelty=0.3     # De-emphasize novelty
)

replay_module = SalienceReplayModule(config)
replay_module.add_pattern(1, {'data': 'test'})

# High consensus, low novelty
replay_module.update_salience(1, agreement_score=1.0, novelty_score=0.0)
print(f"Salience: {replay_module.get_salience(1)}")  # 0.7 * 1.0 + 0.3 * 0.0 = 0.7

# Low consensus, high novelty
replay_module.update_salience(2, agreement_score=0.0, novelty_score=1.0)
print(f"Salience: {replay_module.get_salience(2)}")  # 0.7 * 0.0 + 0.3 * 1.0 = 0.3
```

### Example 3: Monitoring Top Salient Patterns

```python
# After learning
top_patterns = replay_module.get_top_salient_patterns(k=5)

for rank, (pattern_id, salience) in enumerate(top_patterns, 1):
    print(f"{rank}. Pattern {pattern_id}: salience={salience:.4f}")

# Output:
# 1. Pattern 42: salience=15.2341
# 2. Pattern 17: salience=12.8932
# 3. Pattern 8: salience=11.3421
# ...
```

### Example 4: Custom Replay Function

```python
# Define custom learning function
def custom_learning_fn(pattern_data):
    # Extract features
    features = pattern_data['features']

    # Re-run learning update
    model.hebbian_update(features, strength=0.1)

    # Log replay
    logger.info(f"Replayed pattern {pattern_data['pattern_id']}")

# Run replay with custom function
replay_module.run_replay_phase(custom_learning_fn)
```

## Testing

**Test Suite**: `tests/test_salience_replay.py`

**Coverage**: 32 tests across 3 test classes

### Test Classes

1. **TestSalienceReplayConfig** (8 tests)
   - Default configuration
   - Valid custom configuration
   - Invalid parameters (alpha_consensus, alpha_novelty, decay_rate, replay_interval, num_replay_samples, homeostatic_factor)

2. **TestPattern** (1 test)
   - Pattern initialization

3. **TestSalienceReplayModule** (20 tests)
   - Initialization
   - Adding patterns
   - Salience updates (enabled/disabled)
   - Salience accumulation
   - Salience decay (lazy)
   - Getting salience for non-existent patterns
   - Sampling patterns (empty, uniform, weighted)
   - Replay phases (enabled/disabled)
   - Episode end (with/without replay trigger)
   - Statistics computation
   - Top salient patterns
   - Reset functionality
   - High-salience pattern consolidation
   - Baseline behavior preservation

4. **TestSalienceReplayEdgeCases** (3 tests)
   - Single pattern
   - Many patterns (scalability)
   - Zero decay rate boundary

### Test Results

All 32 tests pass:

```
============================= test session starts ==============================
tests/test_salience_replay.py::TestSalienceReplayConfig::test_default_config PASSED
tests/test_salience_replay.py::TestSalienceReplayConfig::test_valid_config PASSED
tests/test_salience_replay.py::TestSalienceReplayConfig::test_invalid_alpha_consensus PASSED
tests/test_salience_replay.py::TestSalienceReplayConfig::test_invalid_alpha_novelty PASSED
tests/test_salience_replay.py::TestSalienceReplayConfig::test_invalid_decay_rate PASSED
tests/test_salience_replay.py::TestSalienceReplayConfig::test_invalid_replay_interval PASSED
tests/test_salience_replay.py::TestSalienceReplayConfig::test_invalid_num_replay_samples PASSED
tests/test_salience_replay.py::TestSalienceReplayConfig::test_invalid_homeostatic_factor PASSED
tests/test_salience_replay.py::TestPattern::test_pattern_initialization PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_initialization PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_add_pattern PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_update_salience_disabled PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_update_salience_enabled PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_salience_accumulation PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_salience_decay PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_get_salience_nonexistent_pattern PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_sample_patterns_empty PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_sample_patterns_uniform PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_sample_patterns_weighted PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_run_replay_phase_disabled PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_run_replay_phase_enabled PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_on_episode_end_no_replay PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_on_episode_end_with_replay PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_get_statistics_empty PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_get_statistics_with_patterns PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_get_top_salient_patterns PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_reset PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_high_salience_patterns_consolidated PASSED
tests/test_salience_replay.py::TestSalienceReplayModule::test_baseline_behavior_preserved PASSED
tests/test_salience_replay.py::TestSalienceReplayEdgeCases::test_single_pattern PASSED
tests/test_salience_replay.py::TestSalienceReplayEdgeCases::test_many_patterns PASSED
tests/test_salience_replay.py::TestSalienceReplayEdgeCases::test_zero_decay_rate_boundary PASSED

============================== 32 passed in 0.14s
```

## Example Outputs

Running `examples/salience_replay_example.py` demonstrates:

### Scenario 1: Salience Evolution
- Patterns 0-4: High consensus (reliable)
- Patterns 5-7: High novelty (novel experiences)
- Patterns 8-9: Low value (unreliable and common)

After 20 episodes, saliences reflect value:
```
Pattern 7 (High novelty): salience=11.46
Pattern 6 (High novelty): salience=11.22
Pattern 5 (High novelty): salience=10.97
Pattern 3 (High consensus): salience=9.66
Pattern 9 (Low value): salience=2.19
```

### Scenario 2: Replay Statistics
High-value patterns are replayed more often:
```
Pattern 3 (High consensus): 10 times
Pattern 1 (High consensus): 7 times
Pattern 2 (High consensus): 5 times
...
Pattern 9 (Low value): 0 times
```

### Scenario 3: Salience Decay
Salience decays exponentially without updates:
```
Step   0: salience = 1.0000
Step   5: salience = 0.5905
Step  10: salience = 0.3487
Step  20: salience = 0.1216
Step  50: salience = 0.0052
```

## Design Decisions

### 1. Lazy Salience Decay
**Choice**: Compute decay when reading salience, not every step

**Rationale**:
- More efficient: O(1) per read vs O(N) per step
- Mathematically equivalent
- Scales to large pattern sets

### 2. Dual-Signal Salience
**Choice**: Combine consensus agreement and novelty

**Rationale**:
- Captures two forms of value:
  - Consensus agreement: Patterns that worked (reliable)
  - Novelty: Patterns that were informative (unexpected)
- Weight balance allows tuning for different scenarios
- Reflects biological principles (reward + surprise)

### 3. Weighted Sampling with Replacement
**Choice**: Sample patterns proportional to salience, with replacement

**Rationale**:
- Highly salient patterns can be replayed multiple times
- Natural interpretation: importance determines replay frequency
- Simpler than sampling without replacement

### 4. Separate Pattern Storage
**Choice**: Store patterns in separate Pattern objects, not in learning module

**Rationale**:
- Cleaner separation of concerns
- Allows replay module to be used with any learning module
- Pattern storage can be easily extended or replaced

### 5. Homeostatic Downscaling
**Choice**: Apply global downscaling to all patterns during replay

**Rationale**:
- Prevents runaway growth of pattern strengths
- Un-replayed patterns gradually fade
- Replayed patterns are re-strengthened by learning
- Biological analog: synaptic downscaling during sleep

### 6. Feature Flag
**Choice**: `salience_tagged_replay` bool flag

**Rationale**:
- Allows A/B comparison with baseline CGAL
- Easy to disable for ablation studies
- Preserves backward compatibility
- When disabled, no salience tracking or replay

## Performance Considerations

### Memory
- Pattern storage: O(N) where N = number of patterns
- Each pattern stores: ID, data, salience, timestamp
- For 1000 patterns: ~1000 Pattern objects

### Computation
- `update_salience()`: O(1) update
- `get_salience()`: O(1) with lazy decay
- `sample_patterns_by_salience()`: O(N) to compute probabilities + O(K log K) sampling
- `run_replay_phase()`: O(K * C) where K = num_replay_samples, C = learning cost

### Scalability
- Tested with up to 100 patterns in unit tests
- Designed to scale to ~10,000 patterns
- For larger pattern sets, consider:
  - Top-K sampling (only consider highest salience patterns)
  - Stratified sampling (sample from bins)
  - Distributed storage

## Future Extensions

### Potential Enhancements

1. **Adaptive Replay Interval**:
   - Current: Fixed interval
   - Extension: Adapt interval based on learning progress or novelty

2. **Multi-Level Salience**:
   - Current: Single salience value
   - Extension: Separate salience for different learning objectives

3. **Contextual Replay**:
   - Current: Sample independently
   - Extension: Replay patterns in context (sequences, episodes)

4. **Gradient-Based Salience**:
   - Current: Heuristic (consensus + novelty)
   - Extension: Learn salience from gradients or loss

5. **Sparse Pattern Storage**:
   - Current: Store all patterns
   - Extension: Prune low-salience patterns periodically

6. **Distributed Replay**:
   - Current: Single replay module
   - Extension: Distribute replay across multiple workers

7. **Salience Visualization**:
   - Track salience evolution over time
   - Visualize salience heatmaps
   - Plot replay statistics

## References

- CGAL Framework: Consensus-gated associative learning
- Issue #2: Consensus-Gated Plasticity (provides agreement scores)
- Issue #3: Novelty Detection (provides novelty scores)
- Issue #4: Learned Trust Weights (could use replay for trust refinement)
- Issue #5: Salience-Tagged Replay ← This document

## Summary

Salience-tagged replay (Issue #5) enables CGAL to:
- **Tag patterns with salience**: Combine consensus agreement and novelty
- **Preferentially replay high-value patterns**: Consolidate important experiences
- **Apply homeostatic downscaling**: Prevent runaway growth
- **Integrate with CGAL components**: Uses outputs from Issues #2 and #3
- **Preserve baseline**: Feature flag allows disabling for comparison

This implements CGAL's intrinsic-motivation coupling and replay-driven consolidation, allowing the system to selectively strengthen patterns that are both reliable (high consensus) and informative (high novelty).
