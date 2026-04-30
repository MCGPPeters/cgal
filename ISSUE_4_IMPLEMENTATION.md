# Issue #4 Implementation: Learned Trust Weights

## Overview

This document describes the implementation of **learned trust weights** between learning modules in the CGAL framework (Issue #4). Trust weights allow modules to learn which neighbors are reliable voters and weight their votes accordingly.

## Core Concept

In CGAL, learning modules vote on object identity. Some modules may be more reliable than others due to:
- Better learned representations
- More informative sensory input
- Less noise in their observations

Rather than weighting all neighbors equally, learned trust weights allow each module to learn which neighbors consistently agree with consensus, and trust them more.

## Formula

The trust weight from module m to module n is updated after each voting round:

```
W[m,n] += γ * (agreement(n, consensus) - W[m,n])
```

Where:
- **W[m,n]**: Trust weight from module m to module n, in range [trust_min, trust_max]
- **γ** (gamma): Trust learning rate (default: 0.05)
- **agreement(n, consensus)**: Binary agreement score:
  - 1.0 if module n's vote matches consensus object ID
  - 0.0 otherwise

Trust weights are:
- **Directional**: W[m,n] ≠ W[n,m] in general
- **Clipped**: Constrained to [trust_min, trust_max] to prevent complete silencing
- **Exponential moving average**: Recent voting history matters more than distant past

## Implementation

### 1. Configuration (`TrustWeightsConfig`)

**Location**: `src/cgal/config/trust_weights_config.py`

```python
@dataclass
class TrustWeightsConfig:
    learned_trust_weights: bool = False  # Feature flag
    trust_learning_rate: float = 0.05    # γ in formula
    trust_min: float = 0.1               # Minimum trust value
    trust_max: float = 1.0               # Maximum trust value
    log_interval: int = 10               # Episodes between logging
    enable_logging: bool = True          # Whether to log trust evolution
```

**Parameters**:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learned_trust_weights` | False | bool | Feature flag for enabling trust weights |
| `trust_learning_rate` | 0.05 | (0, 1] | Learning rate γ for trust updates |
| `trust_min` | 0.1 | (0, 1) | Minimum trust value (prevents complete silencing) |
| `trust_max` | 1.0 | (0, 1] | Maximum trust value |
| `log_interval` | 10 | ≥0 | Episodes between logging (0 = no logging) |
| `enable_logging` | True | bool | Whether to log trust evolution |

**Validation**:
- `trust_learning_rate` must be in (0, 1]
- `trust_min` must be in (0, 1)
- `trust_max` must be in (0, 1]
- `trust_min` must be ≤ `trust_max`
- `log_interval` must be non-negative

### 2. Trust Weights Module (`TrustWeightsModule`)

**Location**: `src/cgal/learning_modules/trust_weights.py`

**Key Methods**:

#### `initialize_trust(module_ids: List[int])`
Initialize trust matrix for a set of modules. All pairs start with trust = 1.0 (uniform trust).

```python
trust_module.initialize_trust([1, 2, 3])
# Creates trust weights for all pairs: (1,2), (1,3), (2,1), (2,3), (3,1), (3,2)
# All initialized to 1.0
```

#### `get_trust(from_id: int, to_id: int) -> float`
Get trust weight from one module to another.

- If `learned_trust_weights=False`: Always returns 1.0 (baseline behavior)
- If `learned_trust_weights=True`: Returns learned trust weight

```python
trust = trust_module.get_trust(from_id=1, to_id=2)
# Returns trust that module 1 has in module 2
```

#### `update_trust(from_id: int, to_id: int, agreement: float)`
Update single trust weight based on agreement score.

```python
trust_module.update_trust(
    from_id=1,
    to_id=2,
    agreement=1.0  # Module 2 agreed with consensus
)
```

#### `update_all_trust(module_votes: Dict, consensus: Dict)`
Update all trust weights after a voting round.

```python
module_votes = {
    1: {'object_id': 'cup'},
    2: {'object_id': 'cup'},
    3: {'object_id': 'plate'}
}
consensus = {'object_id': 'cup'}

trust_module.update_all_trust(module_votes, consensus)
# Modules 1 and 2's trust stays high (agreed with consensus)
# Module 3's trust decreases (disagreed with consensus)
```

#### `weight_votes(from_id: int, neighbor_votes: Dict) -> Dict[int, float]`
Get trust weights for weighting neighbor votes.

```python
neighbor_votes = {
    2: {'object_id': 'cup'},
    3: {'object_id': 'plate'}
}
weights = trust_module.weight_votes(from_id=1, neighbor_votes=neighbor_votes)
# Returns: {2: 0.95, 3: 0.25}  (example values after learning)
```

#### Other utilities:
- `get_trust_matrix_dict()`: Get full trust matrix as dictionary
- `get_trust_matrix_array(module_ids)`: Get trust matrix as 2D numpy array
- `get_statistics()`: Get statistics (mean, std, min, max, count)
- `reset()`: Reset all trust weights to 1.0
- `on_episode_end()`: Called at episode end for periodic logging

## Integration with CGAL

### Typical Usage Flow

1. **Initialization**:
```python
from cgal.config import TrustWeightsConfig
from cgal.learning_modules import TrustWeightsModule

config = TrustWeightsConfig(learned_trust_weights=True)
trust_module = TrustWeightsModule(config)
trust_module.initialize_trust(module_ids)
```

2. **During Voting**:
```python
# Each module weights its neighbors' votes
weights = trust_module.weight_votes(module_id, neighbor_votes)

# Use weights in voting aggregation
weighted_votes = aggregate_votes(neighbor_votes, weights)
```

3. **After Consensus**:
```python
# Update trust based on voting outcomes
trust_module.update_all_trust(module_votes, consensus)
```

4. **Episode End**:
```python
# Log trust evolution periodically
trust_module.on_episode_end()
```

### Integration with Other CGAL Components

Trust weights work alongside:

1. **Consensus-Gated Plasticity (Issue #2)**:
   - Trust weights determine influence of neighbors
   - Consensus gating determines own learning rate
   - Together: learn who to listen to AND when to learn

2. **Novelty Detection (Issue #3)**:
   - Novelty can trigger trust re-evaluation
   - Novel situations may require different trust patterns

3. **Future Salience-Tagged Replay (Issue #5)**:
   - High-agreement or high-disagreement situations can be tagged as salient
   - Replay salient voting rounds to refine trust estimates

## Behavioral Properties

### 1. Exponential Moving Average
Trust weights use exponential moving average (EMA) updates:
- Recent voting rounds have more influence than distant past
- Controlled by `trust_learning_rate`
- Higher learning rate → faster adaptation
- Lower learning rate → more stable, resistant to noise

### 2. Trust Clipping
Trust is clipped to [trust_min, trust_max]:
- **trust_min** (default 0.1): Prevents complete silencing
  - Even unreliable modules retain some influence
  - Allows recovery if module improves
- **trust_max** (default 1.0): Upper bound on trust
  - Typically left at 1.0 (maximum trust)

### 3. Trust Decay and Recovery
- **Decay**: Consistent disagreement → trust decreases exponentially toward trust_min
- **Recovery**: If module improves, trust recovers exponentially toward trust_max
- Speed determined by `trust_learning_rate`

### 4. Baseline Preservation
When `learned_trust_weights=False`:
- `get_trust()` always returns 1.0
- Trust matrix is not updated
- Preserves baseline CGAL behavior (uniform trust)

## Examples

### Example 1: Basic Usage

```python
from cgal.config import TrustWeightsConfig
from cgal.learning_modules import TrustWeightsModule

# Configure trust weights
config = TrustWeightsConfig(
    learned_trust_weights=True,
    trust_learning_rate=0.1,
    trust_min=0.1
)

# Initialize module
trust_module = TrustWeightsModule(config)
trust_module.initialize_trust([1, 2, 3])

# Simulate voting rounds
for episode in range(20):
    module_votes = {
        1: {'object_id': 'cup'},
        2: {'object_id': 'cup'},
        3: {'object_id': 'plate'}  # Unreliable
    }
    consensus = {'object_id': 'cup'}

    trust_module.update_all_trust(module_votes, consensus)
    trust_module.on_episode_end()

# Check trust in module 3 (should be low)
trust_1_3 = trust_module.get_trust(1, 3)
print(f"Trust in module 3: {trust_1_3:.4f}")  # ~0.12 after 20 rounds
```

### Example 2: Vote Weighting

```python
# Module 1 receives votes from neighbors
neighbor_votes = {
    2: {'object_id': 'cup'},
    3: {'object_id': 'plate'}
}

# Get trust weights
weights = trust_module.weight_votes(from_id=1, neighbor_votes=neighbor_votes)

# Weight the votes
def weighted_vote(neighbor_votes, weights):
    vote_counts = {}
    for neighbor_id, vote in neighbor_votes.items():
        obj_id = vote['object_id']
        vote_counts[obj_id] = vote_counts.get(obj_id, 0) + weights[neighbor_id]
    return max(vote_counts, key=vote_counts.get)

result = weighted_vote(neighbor_votes, weights)
# 'cup' wins because module 2 has higher trust than module 3
```

### Example 3: Baseline Mode

```python
# Disable learned trust weights
config = TrustWeightsConfig(learned_trust_weights=False)
trust_module = TrustWeightsModule(config)
trust_module.initialize_trust([1, 2])

# Even if we manually set different values
trust_module.trust_matrix[(1, 2)] = 0.3

# get_trust always returns 1.0 when disabled
trust = trust_module.get_trust(1, 2)
assert trust == 1.0  # Baseline behavior preserved
```

## Testing

**Test Suite**: `tests/test_trust_weights.py`

**Coverage**: 32 tests across 3 test classes

### Test Classes

1. **TestTrustWeightsConfig** (7 tests)
   - Default configuration
   - Valid custom configuration
   - Invalid parameters (learning rate, trust_min, trust_max, min/max relationship, log_interval)

2. **TestTrustWeightsModule** (22 tests)
   - Initialization
   - Trust initialization
   - Get trust (enabled/disabled)
   - Update trust (enabled/disabled, increases, decreases)
   - Trust clipping (min and max)
   - Update all trust after voting
   - Agreement computation (same object, different object, empty)
   - Vote weighting
   - Trust matrix retrieval (dict and array)
   - Statistics computation
   - Reset functionality
   - Broken module scenario (loses trust over time)
   - Episode end logging
   - Baseline behavior preservation

3. **TestTrustWeightsEdgeCases** (3 tests)
   - Single module (no neighbors)
   - Many modules (scalability)
   - Zero learning rate (no updates)

### Test Results

All 32 tests pass:

```
============================= test session starts ==============================
tests/test_trust_weights.py::TestTrustWeightsConfig::test_default_config PASSED
tests/test_trust_weights.py::TestTrustWeightsConfig::test_valid_config PASSED
tests/test_trust_weights.py::TestTrustWeightsConfig::test_invalid_learning_rate PASSED
tests/test_trust_weights.py::TestTrustWeightsConfig::test_invalid_trust_min PASSED
tests/test_trust_weights.py::TestTrustWeightsConfig::test_invalid_trust_max PASSED
tests/test_trust_weights.py::TestTrustWeightsConfig::test_invalid_min_max_relationship PASSED
tests/test_trust_weights.py::TestTrustWeightsConfig::test_invalid_log_interval PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_initialization PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_initialize_trust PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_get_trust_disabled PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_get_trust_enabled PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_update_trust_disabled PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_update_trust_increases PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_update_trust_decreases PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_trust_clipped_to_min PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_trust_clipped_to_max PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_update_all_trust PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_compute_agreement_same_object PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_compute_agreement_different_object PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_compute_agreement_empty PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_weight_votes PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_get_trust_matrix_dict PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_get_trust_matrix_array PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_get_statistics_empty PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_get_statistics_with_weights PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_reset PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_broken_module_loses_trust PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_on_episode_end_logging PASSED
tests/test_trust_weights.py::TestTrustWeightsModule::test_baseline_behavior_preserved PASSED
tests/test_trust_weights.py::TestTrustWeightsEdgeCases::test_single_module PASSED
tests/test_trust_weights.py::TestTrustWeightsEdgeCases::test_many_modules PASSED
tests/test_trust_weights.py::TestTrustWeightsEdgeCases::test_trust_learning_rate_zero PASSED

============================== 32 passed in 0.14s
```

## Example Outputs

Running `examples/trust_weights_example.py` demonstrates:

### Scenario 1: Broken Module Loses Trust
- Modules 1 and 2 consistently vote correctly
- Module 3 consistently votes incorrectly
- After 20 rounds:
  - Trust in modules 1 and 2: 1.00 (maximum)
  - Trust in module 3: 0.12 (near minimum)

```
After round 20:
  Trust matrix (rows trust columns):
      M1  M2  M3
  M1  0.00  1.00  0.12
  M2  1.00  0.00  0.12
  M3  1.00  1.00  0.00
```

### Scenario 2: Baseline Mode
- With `learned_trust_weights=False`
- Trust always returns 1.0 regardless of internal values
- Preserves original CGAL behavior

### Scenario 3: Trust Recovery
- Module starts with low trust due to poor performance
- Module improves and starts voting correctly
- Trust gradually recovers over ~20 rounds
- Demonstrates that trust can be regained

## Design Decisions

### 1. Binary Agreement Score
**Choice**: 1.0 if object IDs match, 0.0 otherwise

**Rationale**:
- Simple and interpretable
- Matches typical voting scenario (discrete object identity)
- Can be extended to continuous agreement in future (e.g., pose similarity)

### 2. Exponential Moving Average
**Choice**: W[m,n] += γ * (agreement - W[m,n])

**Rationale**:
- Standard EMA update rule
- Recent history weighted more than distant past
- Single parameter (γ) controls adaptation speed
- Proven effective in online learning scenarios

### 3. Trust Clipping
**Choice**: Clip to [trust_min=0.1, trust_max=1.0]

**Rationale**:
- **trust_min > 0**: Prevents complete silencing
  - Allows recovery if module improves
  - Maintains diversity in voting
  - Prevents catastrophic forgetting
- **trust_max = 1.0**: Natural upper bound (complete trust)

### 4. Directional Trust
**Choice**: W[m,n] ≠ W[n,m] (separate weights)

**Rationale**:
- Each module learns its own trust relationships
- Allows asymmetric trust (A trusts B more than B trusts A)
- Supports hierarchical or specialized module roles

### 5. Feature Flag
**Choice**: `learned_trust_weights` bool flag

**Rationale**:
- Allows A/B comparison with baseline CGAL
- Easy to disable for ablation studies
- Preserves backward compatibility
- When disabled, returns 1.0 (uniform trust)

## Performance Considerations

### Memory
- Trust matrix: O(N²) where N = number of modules
- For 10 modules: 90 trust weights (excludes self-trust)
- For 100 modules: 9,900 trust weights
- Storage: Dict[(int, int), float] for sparse representation

### Computation
- `update_all_trust()`: O(N²) updates per voting round
- `get_trust()`: O(1) lookup
- `weight_votes()`: O(K) where K = number of neighbors

### Scalability
- Tested with up to 10 modules in unit tests
- Designed to scale to ~100 modules
- For larger networks, consider sparse trust graphs or hierarchical grouping

## Future Extensions

### Potential Enhancements

1. **Continuous Agreement Score**:
   - Current: Binary (1.0 or 0.0)
   - Extension: Continuous agreement based on pose similarity, feature overlap, etc.

2. **Adaptive Learning Rate**:
   - Current: Fixed γ
   - Extension: Adapt γ based on uncertainty or novelty

3. **Trust Uncertainty**:
   - Track confidence intervals on trust weights
   - Use Bayesian updates instead of EMA

4. **Asymmetric Clipping**:
   - Different trust_min for different module pairs
   - Context-dependent trust bounds

5. **Trust Decay**:
   - Gradual decay toward uniform trust if no recent voting
   - Prevents overreliance on stale trust estimates

6. **Sparse Trust Graphs**:
   - Only maintain trust for K nearest neighbors
   - Reduces memory from O(N²) to O(NK)

7. **Meta-Learning Trust**:
   - Learn optimal trust_learning_rate per module
   - Adapt trust dynamics based on task statistics

## References

- CGAL Framework: Consensus-gated associative learning
- Issue #2: Consensus-Gated Plasticity (learning **when** to learn)
- Issue #3: Novelty Detection (detecting new patterns)
- Issue #4: Learned Trust Weights (learning **who** to trust) ← This document
- Issue #5: Salience-Tagged Replay (selective memory consolidation)

## Summary

Learned trust weights (Issue #4) enable CGAL learning modules to:
- **Learn who to trust**: Discover which neighbors consistently vote reliably
- **Weight votes accordingly**: Give more influence to trusted neighbors
- **Adapt over time**: Trust evolves as voting accuracy changes
- **Recover from errors**: Low trust can increase if module improves
- **Preserve baseline**: Feature flag allows disabling for comparison

This complements consensus-gated plasticity (Issue #2, learning **when** to update) by addressing learning **who to listen to**.
