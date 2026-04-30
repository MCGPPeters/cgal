"""Unit tests for learned trust weights module.

Tests the TrustWeightsModule implementation from Issue #4.
"""

import numpy as np
import pytest

from cgal.config.trust_weights_config import TrustWeightsConfig
from cgal.learning_modules.trust_weights import TrustWeightsModule


class TestTrustWeightsConfig:
    """Test TrustWeightsConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrustWeightsConfig()
        assert config.learned_trust_weights is False
        assert config.trust_learning_rate == 0.05
        assert config.trust_min == 0.1
        assert config.trust_max == 1.0
        assert config.log_interval == 10
        assert config.enable_logging is True

    def test_valid_config(self):
        """Test valid custom configuration."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.1,
            trust_min=0.2,
            trust_max=0.9,
            log_interval=5,
            enable_logging=False
        )
        assert config.learned_trust_weights is True
        assert config.trust_learning_rate == 0.1
        assert config.trust_min == 0.2

    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises error."""
        with pytest.raises(ValueError, match="trust_learning_rate must be in"):
            TrustWeightsConfig(trust_learning_rate=1.5)

        with pytest.raises(ValueError, match="trust_learning_rate must be in"):
            TrustWeightsConfig(trust_learning_rate=-0.1)

    def test_invalid_trust_min(self):
        """Test that invalid trust_min raises error."""
        with pytest.raises(ValueError, match="trust_min must be in"):
            TrustWeightsConfig(trust_min=0.0)

        with pytest.raises(ValueError, match="trust_min must be in"):
            TrustWeightsConfig(trust_min=1.5)

    def test_invalid_trust_max(self):
        """Test that invalid trust_max raises error."""
        with pytest.raises(ValueError, match="trust_max must be in"):
            TrustWeightsConfig(trust_max=1.5)

    def test_invalid_min_max_relationship(self):
        """Test that trust_min > trust_max raises error."""
        with pytest.raises(ValueError, match="trust_min must be <= trust_max"):
            TrustWeightsConfig(trust_min=0.8, trust_max=0.5)

    def test_invalid_log_interval(self):
        """Test that negative log_interval raises error."""
        with pytest.raises(ValueError, match="log_interval must be non-negative"):
            TrustWeightsConfig(log_interval=-1)


class TestTrustWeightsModule:
    """Test TrustWeightsModule functionality."""

    def test_initialization(self):
        """Test module initialization."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)
        assert module.config == config
        assert len(module.trust_matrix) == 0
        assert module.episode_count == 0

    def test_initialize_trust(self):
        """Test initializing trust matrix for modules."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        module_ids = [1, 2, 3]
        module.initialize_trust(module_ids)

        # Should have n*(n-1) entries (no self-trust)
        assert len(module.trust_matrix) == 6  # 3*2

        # All should be initialized to 1.0
        for (from_id, to_id), weight in module.trust_matrix.items():
            assert weight == 1.0
            assert from_id != to_id

    def test_get_trust_disabled(self):
        """Test that trust is always 1.0 when feature is disabled."""
        config = TrustWeightsConfig(learned_trust_weights=False)
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        # Manually set a different value
        module.trust_matrix[(1, 2)] = 0.5

        # Should still return 1.0 when disabled
        trust = module.get_trust(1, 2)
        assert trust == 1.0

    def test_get_trust_enabled(self):
        """Test getting trust values when feature is enabled."""
        config = TrustWeightsConfig(learned_trust_weights=True)
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        module.trust_matrix[(1, 2)] = 0.7

        trust = module.get_trust(1, 2)
        assert trust == 0.7

    def test_update_trust_disabled(self):
        """Test that trust doesn't update when feature is disabled."""
        config = TrustWeightsConfig(learned_trust_weights=False)
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        initial_trust = module.trust_matrix[(1, 2)]

        module.update_trust(1, 2, agreement=0.0)

        # Should not change
        assert module.trust_matrix[(1, 2)] == initial_trust

    def test_update_trust_increases(self):
        """Test that trust increases with positive agreement."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.1
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        module.trust_matrix[(1, 2)] = 0.5

        # High agreement should increase trust
        module.update_trust(1, 2, agreement=1.0)

        new_trust = module.trust_matrix[(1, 2)]
        # W = 0.5 + 0.1 * (1.0 - 0.5) = 0.5 + 0.05 = 0.55
        assert np.isclose(new_trust, 0.55)

    def test_update_trust_decreases(self):
        """Test that trust decreases with negative agreement."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.1
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        module.trust_matrix[(1, 2)] = 0.8

        # Low agreement should decrease trust
        module.update_trust(1, 2, agreement=0.0)

        new_trust = module.trust_matrix[(1, 2)]
        # W = 0.8 + 0.1 * (0.0 - 0.8) = 0.8 - 0.08 = 0.72
        assert np.isclose(new_trust, 0.72)

    def test_trust_clipped_to_min(self):
        """Test that trust is clipped to trust_min."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=1.0,  # Fast update
            trust_min=0.1
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])

        # Apply multiple bad updates
        for _ in range(10):
            module.update_trust(1, 2, agreement=0.0)

        # Should not go below trust_min
        trust = module.trust_matrix[(1, 2)]
        assert trust >= 0.1
        assert np.isclose(trust, 0.1)

    def test_trust_clipped_to_max(self):
        """Test that trust is clipped to trust_max."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=1.0,  # Fast update
            trust_max=0.9
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        module.trust_matrix[(1, 2)] = 0.85

        # Apply multiple perfect updates
        for _ in range(10):
            module.update_trust(1, 2, agreement=1.0)

        # Should not go above trust_max
        trust = module.trust_matrix[(1, 2)]
        assert trust <= 0.9
        assert np.isclose(trust, 0.9)

    def test_update_all_trust(self):
        """Test updating trust for all modules after voting."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.1
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2, 3])

        module_votes = {
            1: {'object_id': 'cup'},
            2: {'object_id': 'cup'},
            3: {'object_id': 'plate'}
        }
        consensus = {'object_id': 'cup'}

        module.update_all_trust(module_votes, consensus)

        # Modules 1 and 2 agreed with consensus, so trust in them should stay high
        # Module 3 disagreed, so trust in it should decrease
        # Each module updates trust in the others

        # Module 1's trust in 2 should remain at 1.0 (2 agreed, already at max)
        trust_1_2 = module.get_trust(1, 2)
        assert trust_1_2 == 1.0  # Started at 1.0, agreement=1.0, stays at max

        # Module 1's trust in 3 should decrease (3 disagreed)
        trust_1_3 = module.get_trust(1, 3)
        assert trust_1_3 < 1.0  # agreement=0.0, should decrease

    def test_compute_agreement_same_object(self):
        """Test agreement computation for same object."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        vote = {'object_id': 'cup'}
        consensus = {'object_id': 'cup'}

        agreement = module._compute_agreement(vote, consensus)
        assert agreement == 1.0

    def test_compute_agreement_different_object(self):
        """Test agreement computation for different objects."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        vote = {'object_id': 'cup'}
        consensus = {'object_id': 'plate'}

        agreement = module._compute_agreement(vote, consensus)
        assert agreement == 0.0

    def test_compute_agreement_empty(self):
        """Test agreement computation with empty votes."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        agreement = module._compute_agreement({}, {'object_id': 'cup'})
        assert agreement == 0.0

        agreement = module._compute_agreement({'object_id': 'cup'}, {})
        assert agreement == 0.0

    def test_weight_votes(self):
        """Test getting weights for neighbor votes."""
        config = TrustWeightsConfig(learned_trust_weights=True)
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2, 3])
        module.trust_matrix[(1, 2)] = 0.8
        module.trust_matrix[(1, 3)] = 0.6

        neighbor_votes = {
            2: {'object_id': 'cup'},
            3: {'object_id': 'plate'}
        }

        weights = module.weight_votes(1, neighbor_votes)

        assert weights[2] == 0.8
        assert weights[3] == 0.6

    def test_get_trust_matrix_dict(self):
        """Test getting trust matrix as dict."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        matrix_dict = module.get_trust_matrix_dict()

        assert len(matrix_dict) == 2  # 2*1
        assert (1, 2) in matrix_dict
        assert (2, 1) in matrix_dict

    def test_get_trust_matrix_array(self):
        """Test getting trust matrix as numpy array."""
        config = TrustWeightsConfig(learned_trust_weights=True)
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2, 3])
        module.trust_matrix[(1, 2)] = 0.8
        module.trust_matrix[(1, 3)] = 0.6

        matrix = module.get_trust_matrix_array([1, 2, 3])

        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 0.8  # Trust from 1 to 2
        assert matrix[0, 2] == 0.6  # Trust from 1 to 3
        assert matrix[0, 0] == 0.0  # No self-trust

    def test_get_statistics_empty(self):
        """Test statistics with no trust weights."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        stats = module.get_statistics()
        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_get_statistics_with_weights(self):
        """Test statistics computation."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2, 3])
        module.trust_matrix[(1, 2)] = 0.8
        module.trust_matrix[(2, 1)] = 0.6
        module.trust_matrix[(1, 3)] = 0.9

        stats = module.get_statistics()
        assert stats["count"] == 6
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_reset(self):
        """Test resetting trust weights."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        module.trust_matrix[(1, 2)] = 0.5
        module.episode_count = 10

        module.reset()

        # All should be reset to 1.0
        assert module.trust_matrix[(1, 2)] == 1.0
        assert module.episode_count == 0
        assert len(module.trust_history) == 0

    def test_broken_module_loses_trust(self):
        """Test that a broken module (random votes) loses trust over time."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.1
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2, 3])

        # Simulate voting rounds where module 3 always disagrees
        for _ in range(20):
            module_votes = {
                1: {'object_id': 'cup'},
                2: {'object_id': 'cup'},
                3: {'object_id': 'plate'}  # Always wrong
            }
            consensus = {'object_id': 'cup'}
            module.update_all_trust(module_votes, consensus)

        # Trust in module 3 should have decayed
        trust_1_3 = module.get_trust(1, 3)
        trust_2_3 = module.get_trust(2, 3)

        assert trust_1_3 < 0.5  # Significantly decreased
        assert trust_2_3 < 0.5

        # Trust in reliable modules should remain high
        trust_1_2 = module.get_trust(1, 2)
        trust_2_1 = module.get_trust(2, 1)

        assert trust_1_2 > 0.9
        assert trust_2_1 > 0.9

    def test_on_episode_end_logging(self):
        """Test episode end logging."""
        config = TrustWeightsConfig(
            enable_logging=True,
            log_interval=5
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])

        # Call on_episode_end multiple times
        for i in range(10):
            module.on_episode_end()

        # Should have logged at episodes 5 and 10
        assert len(module.trust_history) == 2
        assert module.episode_count == 10

    def test_baseline_behavior_preserved(self):
        """Test that baseline behavior is preserved when feature is disabled."""
        config_disabled = TrustWeightsConfig(learned_trust_weights=False)
        module_disabled = TrustWeightsModule(config_disabled)

        module_disabled.initialize_trust([1, 2])

        # Apply updates
        module_disabled.update_trust(1, 2, agreement=0.0)

        # Trust should remain 1.0 when getting (feature disabled)
        trust = module_disabled.get_trust(1, 2)
        assert trust == 1.0


class TestTrustWeightsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_module(self):
        """Test with a single module (no neighbors)."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        module.initialize_trust([1])

        # Should have no trust weights (no neighbors)
        assert len(module.trust_matrix) == 0

    def test_many_modules(self):
        """Test with many modules."""
        config = TrustWeightsConfig()
        module = TrustWeightsModule(config)

        module_ids = list(range(10))
        module.initialize_trust(module_ids)

        # Should have n*(n-1) = 10*9 = 90 entries
        assert len(module.trust_matrix) == 90

    def test_trust_learning_rate_zero(self):
        """Test with learning rate = 0 (no updates)."""
        config = TrustWeightsConfig(
            learned_trust_weights=True,
            trust_learning_rate=0.0
        )
        module = TrustWeightsModule(config)

        module.initialize_trust([1, 2])
        initial_trust = module.trust_matrix[(1, 2)]

        module.update_trust(1, 2, agreement=0.0)

        # Should not change with lr=0
        assert module.trust_matrix[(1, 2)] == initial_trust
