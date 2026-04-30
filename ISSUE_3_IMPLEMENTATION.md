# Issue #3: Novelty Detection - Implementation Complete

This document describes the implementation of novelty detection from hypothesis distribution shape (CGAL Issue #3).

## Overview

The novelty detection mechanism analyzes the entropy and peak confidence of hypothesis distributions to determine whether an observation represents a novel pattern or a familiar one. This implements CGAL Section 3.8's claim that hypothesis-generation machinery inherently performs novelty detection as a side effect.

## Key Formula

```
novelty_score = entropy(hypothesis_dist) * (1 - peak_confidence)
```

Where:
- **entropy** = `-sum(p * log(p))` over normalized distribution
- **peak_confidence** = `max(p)` after normalization
- Result is normalized to [0, 1]

## Interpretation

- **Sharp distribution + high peak confidence** → **low novelty** (familiar pattern)
- **Broad distribution + low peak confidence** → **high novelty** (novel pattern)
- **Sharp distribution + low peak confidence** → **ambiguous** (wait for more evidence)

## Implementation

### Module Structure

```
src/cgal/
├── config/
│   └── novelty_config.py      # Configuration parameters
└── learning_modules/
    └── novelty_detection.py    # Core novelty detector

tests/
└── test_novelty_detection.py   # Unit tests (32 tests, all passing)
```

### Configuration

The `NoveltyDetectionConfig` class provides all configuration parameters:

```python
from src.cgal.config import NoveltyDetectionConfig

config = NoveltyDetectionConfig(
    hypothesis_novelty_detection=True,   # Enable/disable feature
    novelty_threshold=0.7,                # Threshold for new pattern allocation
    min_entropy=0.0,                      # Min entropy for normalization
    max_entropy=None,                     # Max entropy (auto-computed if None)
    enable_logging=True                   # Log novelty scores
)
```

### Usage Example

```python
from src.cgal.config import NoveltyDetectionConfig
from src.cgal.learning_modules import NoveltyDetector

# Create detector with config
config = NoveltyDetectionConfig(hypothesis_novelty_detection=True)
detector = NoveltyDetector(config)

# Analyze hypothesis scores
hypothesis_scores = {
    "object_A": 5.0,
    "object_B": 3.0,
    "object_C": 2.0
}

# Check if novel
is_novel, novelty_score = detector.is_novel(hypothesis_scores)

if is_novel:
    print(f"Novel pattern detected! Score: {novelty_score:.3f}")
    # Route to new pattern allocation
else:
    print(f"Familiar pattern. Score: {novelty_score:.3f}")
    # Reinforce existing pattern

# Get statistics
stats = detector.get_statistics()
print(f"Mean novelty: {stats['mean']:.3f}")
print(f"Std novelty: {stats['std']:.3f}")
```

## Core Methods

### NoveltyDetector

#### `compute_novelty_score(hypothesis_scores, normalize=True)`
Computes the novelty score from hypothesis distribution.

**Parameters:**
- `hypothesis_scores`: Dict mapping hypothesis IDs to confidence values
- `normalize`: Whether to normalize to [0, 1] (default: True)

**Returns:** Novelty score (0 = familiar, 1 = novel)

#### `is_novel(hypothesis_scores)`
Determines if observation is novel based on threshold.

**Parameters:**
- `hypothesis_scores`: Dict mapping hypothesis IDs to confidence values

**Returns:** Tuple of (is_novel: bool, novelty_score: float)

#### `get_statistics()`
Returns statistics about detected novelty scores (mean, std, min, max, count).

#### `reset_history()`
Clears the history of novelty scores.

## Acceptance Criteria Status

✅ **All acceptance criteria met:**

- [x] Config flag `hypothesis_novelty_detection: bool` added, defaulting to `False`
- [x] When flag is `True`, LM computes `novelty_score ∈ [0, 1]` per observation
- [x] Formula: `novelty_score = entropy(hypothesis_distribution) * (1 - peak_confidence)`
- [x] Normalized to [0, 1] with clear normalization strategy
- [x] When `novelty_score > novelty_threshold`, observations can be routed to new pattern allocation
- [x] When flag is `False`, behavior is baseline (returns False, 0.0)
- [x] Logs the novelty score per step for inspection in experiments
- [x] Unit tests verify:
  - High novelty for fresh inputs (uniform distributions)
  - Low novelty for familiar patterns (sharp, confident distributions)
  - Baseline behavior preserved when disabled

## Test Results

All 32 unit tests pass:
- 5 tests for configuration validation
- 23 tests for novelty detection functionality
- 4 tests for edge cases (single hypothesis, negative scores, large scores, many hypotheses)

```bash
$ python -m pytest tests/test_novelty_detection.py -v
================================ 32 passed in 0.13s ================================
```

## Integration Points

### With Issue #5 (Salience-Tagged Replay)
The novelty score is designed to be accessible for downstream use as a salience tag:

```python
# In salience computation (Issue #5):
salience = alpha_consensus * agreement_score + alpha_novelty * novelty_score
```

The `NoveltyDetector` exposes novelty scores through:
1. Return value from `is_novel()` method
2. `novelty_scores_history` attribute for batch processing
3. `get_statistics()` method for aggregate analysis

## Key Design Decisions

### 1. Normalization Strategy
Novelty scores are normalized based on theoretical maximum:
- Maximum entropy occurs with uniform distribution: `log(n_hypotheses)`
- Maximum novelty score: `max_entropy * (1 - 1/n)`
- Actual scores are divided by this theoretical maximum

### 2. Baseline Preservation
When `hypothesis_novelty_detection=False`:
- `is_novel()` always returns `(False, 0.0)`
- No computation overhead
- Byte-identical to baseline behavior (no novelty detection)

### 3. Logging Control
Separate `enable_logging` flag allows:
- Running with novelty detection enabled but logging disabled (for performance)
- Collecting statistics only when needed
- No memory overhead when logging is off

### 4. Edge Case Handling
- **Empty hypotheses**: Returns novelty = 1.0 (maximum novelty)
- **Single hypothesis**: Low novelty if confident, high if uncertain
- **Zero scores**: Treats as uniform distribution
- **Negative scores**: Clipped to zero before normalization

## Performance Characteristics

- **Time complexity**: O(n) where n = number of hypotheses
- **Space complexity**: O(n) for probability normalization
- **Memory overhead (logging enabled)**: O(t) where t = number of observations
- **Memory overhead (logging disabled)**: O(1)

## Future Enhancements

Potential improvements for future iterations:

1. **Pattern Routing**: Currently just returns `is_novel` flag. Could be extended to automatically trigger new pattern allocation in a learning module.

2. **Adaptive Thresholds**: Could adjust `novelty_threshold` based on running statistics.

3. **Temporal Smoothing**: Could smooth novelty scores over a window to reduce noise.

4. **Alternative Metrics**: Could experiment with other entropy measures (Rényi, Tsallis) or different combinations of entropy and confidence.

## References

- CGAL Framework Section 3.8: Novelty detection from hypothesis distribution shape
- Shannon Entropy: [Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- Issue Template: `.github/issues/03-novelty-detection.md`

## Dependencies

- `numpy>=1.21.0`: For array operations and entropy computation
- `pytest>=7.0.0`: For unit tests
- `pytest-cov>=3.0.0`: For test coverage analysis

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
# Run all novelty detection tests
python -m pytest tests/test_novelty_detection.py -v

# Run with coverage
python -m pytest tests/test_novelty_detection.py --cov=src/cgal/learning_modules --cov-report=term-missing
```

## Contact

For questions or issues related to this implementation, please refer to Issue #3 in the GitHub repository.
