"""Unit tests for consensus-gated plasticity module.

Tests the ConsensusGatingModule implementation from Issue #2.
"""

import numpy as np
import pytest

from cgal.config.consensus_gating_config import ConsensusGatingConfig
from cgal.learning_modules.consensus_gating import ConsensusGatingModule


class TestConsensusGatingConfig:
    """Test ConsensusGatingConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConsensusGatingConfig()
        assert config.consensus_gated_plasticity is False
        assert config.alpha == 0.7
        assert config.baseline_rate == 1.0
        assert config.agreement_tolerance == 0.1
        assert config.enable_logging is True

    def test_valid_config(self):
        """Test valid custom configuration."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.8,
            baseline_rate=0.5,
            agreement_tolerance=0.2,
            enable_logging=False
        )
        assert config.consensus_gated_plasticity is True
        assert config.alpha == 0.8
        assert config.baseline_rate == 0.5
        assert config.agreement_tolerance == 0.2
        assert config.enable_logging is False

    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ConsensusGatingConfig(alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            ConsensusGatingConfig(alpha=-0.1)

    def test_invalid_baseline_rate(self):
        """Test that invalid baseline_rate raises error."""
        with pytest.raises(ValueError, match="baseline_rate must be in"):
            ConsensusGatingConfig(baseline_rate=1.5)

        with pytest.raises(ValueError, match="baseline_rate must be in"):
            ConsensusGatingConfig(baseline_rate=-0.1)

    def test_invalid_agreement_tolerance(self):
        """Test that negative agreement_tolerance raises error."""
        with pytest.raises(ValueError, match="agreement_tolerance must be non-negative"):
            ConsensusGatingConfig(agreement_tolerance=-0.1)


class TestConsensusGatingModule:
    """Test ConsensusGatingModule functionality."""

    def test_initialization(self):
        """Test module initialization."""
        config = ConsensusGatingConfig()
        module = ConsensusGatingModule(config)
        assert module.config == config
        assert len(module.gating_factors_history) == 0
        assert module._last_gating_factor == 1.0

    def test_compute_agreement_same_object_same_pose(self):
        """Test perfect agreement (same object, same pose)."""
        config = ConsensusGatingConfig()
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}

        agreement = module.compute_agreement(lm_hyp, consensus_hyp)
        assert agreement == 1.0

    def test_compute_agreement_same_object_close_pose(self):
        """Test agreement with same object and pose within tolerance."""
        config = ConsensusGatingConfig(agreement_tolerance=0.2)
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_A', 'pose': [1.05, 2.05, 3.05]}

        agreement = module.compute_agreement(lm_hyp, consensus_hyp)
        # Distance = sqrt(0.05^2 + 0.05^2 + 0.05^2) ≈ 0.087 < 0.2
        assert agreement == 1.0

    def test_compute_agreement_same_object_different_pose(self):
        """Test partial agreement (same object, different pose)."""
        config = ConsensusGatingConfig(agreement_tolerance=0.1)
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_A', 'pose': [5.0, 6.0, 7.0]}

        agreement = module.compute_agreement(lm_hyp, consensus_hyp)
        assert agreement == 0.5

    def test_compute_agreement_same_object_no_pose(self):
        """Test partial agreement when pose is not available."""
        config = ConsensusGatingConfig()
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_A'}

        agreement = module.compute_agreement(lm_hyp, consensus_hyp)
        assert agreement == 0.5

    def test_compute_agreement_different_objects(self):
        """Test no agreement (different objects)."""
        config = ConsensusGatingConfig()
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_B', 'pose': [1.0, 2.0, 3.0]}

        agreement = module.compute_agreement(lm_hyp, consensus_hyp)
        assert agreement == 0.0

    def test_compute_agreement_empty_hypotheses(self):
        """Test agreement with empty hypotheses."""
        config = ConsensusGatingConfig()
        module = ConsensusGatingModule(config)

        agreement = module.compute_agreement({}, {})
        assert agreement == 0.0

        agreement = module.compute_agreement(None, {'object_id': 'obj_A'})
        assert agreement == 0.0

    def test_compute_gating_factor_disabled(self):
        """Test that gating factor is 1.0 when feature is disabled."""
        config = ConsensusGatingConfig(consensus_gated_plasticity=False)
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_B', 'pose': [5.0, 6.0, 7.0]}

        gating_factor = module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert gating_factor == 1.0

    def test_compute_gating_factor_perfect_agreement(self):
        """Test gating factor with perfect agreement."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=1.0,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}

        # agreement = 1.0
        # g_m = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        gating_factor = module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert np.isclose(gating_factor, 1.0)

    def test_compute_gating_factor_no_agreement(self):
        """Test gating factor with no agreement."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=1.0,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_B', 'pose': [5.0, 6.0, 7.0]}

        # agreement = 0.0
        # g_m = 0.7 * 0.0 + 0.3 * 1.0 = 0.3 = (1 - alpha)
        gating_factor = module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert np.isclose(gating_factor, 0.3)

    def test_compute_gating_factor_partial_agreement(self):
        """Test gating factor with partial agreement."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=1.0,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_A', 'pose': [5.0, 6.0, 7.0]}

        # agreement = 0.5 (same object, different pose)
        # g_m = 0.7 * 0.5 + 0.3 * 1.0 = 0.35 + 0.3 = 0.65
        gating_factor = module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert np.isclose(gating_factor, 0.65)

    def test_compute_gating_factor_no_consensus(self):
        """Test gating factor when no consensus exists yet."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=1.0,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}

        # No consensus → use baseline_rate
        gating_factor = module.compute_gating_factor(lm_hyp, None)
        assert gating_factor == 1.0

    def test_compute_gating_factor_with_custom_baseline(self):
        """Test gating factor with custom baseline rate."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=0.5,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_B'}

        # agreement = 0.0
        # g_m = 0.7 * 0.0 + 0.3 * 0.5 = 0.15
        gating_factor = module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert np.isclose(gating_factor, 0.15)

    def test_apply_gating(self):
        """Test applying gating to update magnitude."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=1.0,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}

        update_magnitude = 0.5
        gated_update = module.apply_gating(update_magnitude, lm_hyp, consensus_hyp)

        # Perfect agreement → g_m = 1.0 → gated = 0.5 * 1.0 = 0.5
        assert np.isclose(gated_update, 0.5)

    def test_apply_gating_reduces_update(self):
        """Test that gating reduces update when agreement is low."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=1.0,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_B'}

        update_magnitude = 1.0
        gated_update = module.apply_gating(update_magnitude, lm_hyp, consensus_hyp)

        # No agreement → g_m = 0.3 → gated = 1.0 * 0.3 = 0.3
        assert np.isclose(gated_update, 0.3)

    def test_logging_enabled(self):
        """Test that gating factors are logged when logging is enabled."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            enable_logging=True
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp1 = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0]}
        consensus_hyp2 = {'object_id': 'obj_B', 'pose': [5.0, 6.0, 7.0]}

        module.compute_gating_factor(lm_hyp, consensus_hyp1)
        module.compute_gating_factor(lm_hyp, consensus_hyp2)

        assert len(module.gating_factors_history) == 2

    def test_logging_disabled(self):
        """Test that gating factors are not logged when logging is disabled."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_A'}

        module.compute_gating_factor(lm_hyp, consensus_hyp)

        assert len(module.gating_factors_history) == 0

    def test_get_last_gating_factor(self):
        """Test retrieving the last computed gating factor."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.7,
            baseline_rate=1.0,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_B'}

        module.compute_gating_factor(lm_hyp, consensus_hyp)

        last_factor = module.get_last_gating_factor()
        assert np.isclose(last_factor, 0.3)

    def test_get_statistics_empty(self):
        """Test statistics with no history."""
        config = ConsensusGatingConfig()
        module = ConsensusGatingModule(config)

        stats = module.get_statistics()
        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_get_statistics_with_history(self):
        """Test statistics computation with history."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            enable_logging=True
        )
        module = ConsensusGatingModule(config)

        # Add some gating factors
        lm_hyp = {'object_id': 'obj_A'}
        module.compute_gating_factor(lm_hyp, {'object_id': 'obj_A'})
        module.compute_gating_factor(lm_hyp, {'object_id': 'obj_B'})
        module.compute_gating_factor(lm_hyp, None)

        stats = module.get_statistics()
        assert stats["count"] == 3
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_reset_history(self):
        """Test resetting gating factor history."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            enable_logging=True
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_A'}

        module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert len(module.gating_factors_history) == 1

        module.reset_history()
        assert len(module.gating_factors_history) == 0
        assert module._last_gating_factor == 1.0

    def test_baseline_behavior_preserved(self):
        """Test that baseline behavior is preserved when feature is disabled."""
        config_disabled = ConsensusGatingConfig(consensus_gated_plasticity=False)
        config_enabled = ConsensusGatingConfig(consensus_gated_plasticity=True)

        module_disabled = ConsensusGatingModule(config_disabled)
        module_enabled = ConsensusGatingModule(config_enabled)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_B'}

        # When disabled, should always return 1.0
        gating_disabled = module_disabled.compute_gating_factor(lm_hyp, consensus_hyp)
        assert gating_disabled == 1.0

        # When enabled, should compute actual gating factor
        gating_enabled = module_enabled.compute_gating_factor(lm_hyp, consensus_hyp)
        assert gating_enabled < 1.0  # Should be reduced due to disagreement


class TestConsensusGatingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_alpha_zero(self):
        """Test with alpha=0 (no consensus influence)."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=0.0,
            baseline_rate=0.5,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_B'}

        # alpha=0 → g_m always equals baseline_rate regardless of agreement
        gating_factor = module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert np.isclose(gating_factor, 0.5)

    def test_alpha_one(self):
        """Test with alpha=1 (pure consensus-based gating)."""
        config = ConsensusGatingConfig(
            consensus_gated_plasticity=True,
            alpha=1.0,
            baseline_rate=0.5,
            enable_logging=False
        )
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A'}
        consensus_hyp = {'object_id': 'obj_B'}

        # alpha=1, agreement=0 → g_m = 1.0 * 0.0 + 0.0 * 0.5 = 0.0
        gating_factor = module.compute_gating_factor(lm_hyp, consensus_hyp)
        assert np.isclose(gating_factor, 0.0)

    def test_numpy_array_poses(self):
        """Test with numpy array poses."""
        config = ConsensusGatingConfig()
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': np.array([1.0, 2.0, 3.0])}
        consensus_hyp = {'object_id': 'obj_A', 'pose': np.array([1.0, 2.0, 3.0])}

        agreement = module.compute_agreement(lm_hyp, consensus_hyp)
        assert agreement == 1.0

    def test_high_dimensional_poses(self):
        """Test with high-dimensional poses."""
        config = ConsensusGatingConfig(agreement_tolerance=0.5)
        module = ConsensusGatingModule(config)

        lm_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        consensus_hyp = {'object_id': 'obj_A', 'pose': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}

        agreement = module.compute_agreement(lm_hyp, consensus_hyp)
        assert agreement == 1.0
