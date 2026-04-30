"""Unit tests for salience-tagged replay module.

Tests the SalienceReplayModule implementation from Issue #5.
"""

import numpy as np
import pytest

from cgal.config.salience_replay_config import SalienceReplayConfig
from cgal.learning_modules.salience_replay import SalienceReplayModule, Pattern


class TestSalienceReplayConfig:
    """Test SalienceReplayConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SalienceReplayConfig()
        assert config.salience_tagged_replay is False
        assert config.alpha_consensus == 0.5
        assert config.alpha_novelty == 0.5
        assert config.decay_rate == 0.99
        assert config.replay_interval == 10
        assert config.num_replay_samples == 50
        assert config.homeostatic_downscaling is True
        assert config.homeostatic_factor == 0.99
        assert config.enable_logging is True
        assert config.log_interval == 10

    def test_valid_config(self):
        """Test valid custom configuration."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            alpha_consensus=0.7,
            alpha_novelty=0.3,
            decay_rate=0.95,
            replay_interval=5,
            num_replay_samples=100
        )
        assert config.salience_tagged_replay is True
        assert config.alpha_consensus == 0.7
        assert config.alpha_novelty == 0.3

    def test_invalid_alpha_consensus(self):
        """Test that invalid alpha_consensus raises error."""
        with pytest.raises(ValueError, match="alpha_consensus must be in"):
            SalienceReplayConfig(alpha_consensus=1.5)

        with pytest.raises(ValueError, match="alpha_consensus must be in"):
            SalienceReplayConfig(alpha_consensus=-0.1)

    def test_invalid_alpha_novelty(self):
        """Test that invalid alpha_novelty raises error."""
        with pytest.raises(ValueError, match="alpha_novelty must be in"):
            SalienceReplayConfig(alpha_novelty=1.5)

        with pytest.raises(ValueError, match="alpha_novelty must be in"):
            SalienceReplayConfig(alpha_novelty=-0.1)

    def test_invalid_decay_rate(self):
        """Test that invalid decay_rate raises error."""
        with pytest.raises(ValueError, match="decay_rate must be in"):
            SalienceReplayConfig(decay_rate=1.5)

        with pytest.raises(ValueError, match="decay_rate must be in"):
            SalienceReplayConfig(decay_rate=0.0)

    def test_invalid_replay_interval(self):
        """Test that negative replay_interval raises error."""
        with pytest.raises(ValueError, match="replay_interval must be non-negative"):
            SalienceReplayConfig(replay_interval=-1)

    def test_invalid_num_replay_samples(self):
        """Test that invalid num_replay_samples raises error."""
        with pytest.raises(ValueError, match="num_replay_samples must be positive"):
            SalienceReplayConfig(num_replay_samples=0)

        with pytest.raises(ValueError, match="num_replay_samples must be positive"):
            SalienceReplayConfig(num_replay_samples=-5)

    def test_invalid_homeostatic_factor(self):
        """Test that invalid homeostatic_factor raises error."""
        with pytest.raises(ValueError, match="homeostatic_factor must be in"):
            SalienceReplayConfig(homeostatic_factor=1.5)

        with pytest.raises(ValueError, match="homeostatic_factor must be in"):
            SalienceReplayConfig(homeostatic_factor=0.0)


class TestPattern:
    """Test Pattern class."""

    def test_pattern_initialization(self):
        """Test pattern initialization."""
        pattern = Pattern(pattern_id=1, data={'feature': [1, 2, 3]})
        assert pattern.pattern_id == 1
        assert pattern.data == {'feature': [1, 2, 3]}
        assert pattern.salience == 0.0
        assert pattern.last_update_step == 0


class TestSalienceReplayModule:
    """Test SalienceReplayModule functionality."""

    def test_initialization(self):
        """Test module initialization."""
        config = SalienceReplayConfig()
        module = SalienceReplayModule(config)
        assert module.config == config
        assert len(module.patterns) == 0
        assert module.current_step == 0
        assert module.episode_count == 0

    def test_add_pattern(self):
        """Test adding patterns."""
        config = SalienceReplayConfig()
        module = SalienceReplayModule(config)

        pattern = module.add_pattern(1, {'data': 'test'})
        assert pattern.pattern_id == 1
        assert pattern.data == {'data': 'test'}
        assert 1 in module.patterns

    def test_update_salience_disabled(self):
        """Test that salience doesn't update when feature is disabled."""
        config = SalienceReplayConfig(salience_tagged_replay=False)
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})
        module.update_salience(1, agreement_score=1.0, novelty_score=0.5)

        # Salience should remain 0.0 when disabled
        assert module.get_salience(1) == 0.0

    def test_update_salience_enabled(self):
        """Test salience updates when feature is enabled."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            alpha_consensus=0.6,
            alpha_novelty=0.4
        )
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})
        module.update_salience(1, agreement_score=1.0, novelty_score=0.5)

        # salience = 0.6 * 1.0 + 0.4 * 0.5 = 0.6 + 0.2 = 0.8
        assert np.isclose(module.get_salience(1), 0.8)

    def test_salience_accumulation(self):
        """Test that salience accumulates over multiple updates."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            alpha_consensus=0.5,
            alpha_novelty=0.5
        )
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})

        # First update: salience = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        module.update_salience(1, agreement_score=1.0, novelty_score=0.0)
        assert np.isclose(module.get_salience(1), 0.5)

        # Second update: salience = 0.5 + (0.5 * 0.0 + 0.5 * 1.0) = 1.0
        module.update_salience(1, agreement_score=0.0, novelty_score=1.0)
        assert np.isclose(module.get_salience(1), 1.0)

    def test_salience_decay(self):
        """Test that salience decays over time."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            alpha_consensus=0.5,
            alpha_novelty=0.5,
            decay_rate=0.9
        )
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})
        module.update_salience(1, agreement_score=1.0, novelty_score=1.0)

        initial_salience = module.get_salience(1)
        assert np.isclose(initial_salience, 1.0)

        # Advance time and update salience (triggers lazy decay)
        module.current_step = 5
        decayed_salience = module.get_salience(1)

        # Should decay: 1.0 * (0.9 ** 5) ≈ 0.59
        expected_salience = 1.0 * (0.9 ** 5)
        assert np.isclose(decayed_salience, expected_salience)

    def test_get_salience_nonexistent_pattern(self):
        """Test getting salience for non-existent pattern."""
        config = SalienceReplayConfig()
        module = SalienceReplayModule(config)

        salience = module.get_salience(999)
        assert salience == 0.0

    def test_sample_patterns_empty(self):
        """Test sampling from empty pattern set."""
        config = SalienceReplayConfig(salience_tagged_replay=True)
        module = SalienceReplayModule(config)

        samples = module.sample_patterns_by_salience(10)
        assert len(samples) == 0

    def test_sample_patterns_uniform(self):
        """Test sampling when all saliences are zero (uniform)."""
        config = SalienceReplayConfig(salience_tagged_replay=True)
        module = SalienceReplayModule(config)

        # Add patterns with zero salience
        for i in range(5):
            module.add_pattern(i, {'data': i})

        samples = module.sample_patterns_by_salience(3)
        assert len(samples) == 3

    def test_sample_patterns_weighted(self):
        """Test that high-salience patterns are sampled more often."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            alpha_consensus=1.0,
            alpha_novelty=0.0
        )
        module = SalienceReplayModule(config)

        # Create patterns with different saliences
        for i in range(5):
            module.add_pattern(i, {'data': i})

        # Pattern 4 gets very high salience
        for _ in range(10):
            module.update_salience(4, agreement_score=1.0, novelty_score=0.0)

        # Others get low salience
        for i in range(4):
            module.update_salience(i, agreement_score=0.1, novelty_score=0.0)

        # Sample many times and check pattern 4 appears most often
        pattern_counts = {i: 0 for i in range(5)}
        for _ in range(100):
            samples = module.sample_patterns_by_salience(1)
            if samples:
                pattern_counts[samples[0].pattern_id] += 1

        # Pattern 4 should be sampled most often
        assert pattern_counts[4] > pattern_counts[0]

    def test_run_replay_phase_disabled(self):
        """Test that replay doesn't run when feature is disabled."""
        config = SalienceReplayConfig(salience_tagged_replay=False)
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})

        replay_count = []

        def learning_fn(data):
            replay_count.append(1)

        module.run_replay_phase(learning_fn)

        # No replay should happen when disabled
        assert len(replay_count) == 0

    def test_run_replay_phase_enabled(self):
        """Test that replay runs when feature is enabled."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            num_replay_samples=5
        )
        module = SalienceReplayModule(config)

        # Add patterns
        for i in range(10):
            module.add_pattern(i, {'data': i})
            module.update_salience(i, agreement_score=0.5, novelty_score=0.5)

        replayed_data = []

        def learning_fn(data):
            replayed_data.append(data)

        module.run_replay_phase(learning_fn)

        # Should have replayed 5 patterns
        assert len(replayed_data) == 5

    def test_on_episode_end_no_replay(self):
        """Test episode end without replay (wrong interval)."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            replay_interval=10
        )
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})

        replay_count = []

        def learning_fn(data):
            replay_count.append(1)

        # Episodes 1-9 should not trigger replay
        for _ in range(9):
            module.on_episode_end(learning_fn)

        assert len(replay_count) == 0
        assert module.episode_count == 9

    def test_on_episode_end_with_replay(self):
        """Test episode end triggering replay."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            replay_interval=5,
            num_replay_samples=2
        )
        module = SalienceReplayModule(config)

        for i in range(5):
            module.add_pattern(i, {'data': i})
            module.update_salience(i, agreement_score=0.5, novelty_score=0.5)

        replay_count = []

        def learning_fn(data):
            replay_count.append(1)

        # Episode 5 should trigger replay
        for _ in range(5):
            module.on_episode_end(learning_fn)

        # Should have replayed 2 patterns at episode 5
        assert len(replay_count) == 2
        assert module.episode_count == 5

    def test_get_statistics_empty(self):
        """Test statistics with no patterns."""
        config = SalienceReplayConfig()
        module = SalienceReplayModule(config)

        stats = module.get_statistics()
        assert stats["num_patterns"] == 0
        assert stats["mean_salience"] == 0.0

    def test_get_statistics_with_patterns(self):
        """Test statistics computation."""
        config = SalienceReplayConfig(salience_tagged_replay=True)
        module = SalienceReplayModule(config)

        for i in range(5):
            module.add_pattern(i, {'data': i})
            module.update_salience(i, agreement_score=float(i) / 5, novelty_score=0.0)

        stats = module.get_statistics()
        assert stats["num_patterns"] == 5
        assert 0.0 <= stats["mean_salience"] <= 1.0
        assert stats["min_salience"] <= stats["mean_salience"] <= stats["max_salience"]

    def test_get_top_salient_patterns(self):
        """Test getting top-k salient patterns."""
        config = SalienceReplayConfig(salience_tagged_replay=True)
        module = SalienceReplayModule(config)

        # Create patterns with different saliences
        for i in range(10):
            module.add_pattern(i, {'data': i})
            # Pattern i gets salience proportional to i
            for _ in range(i):
                module.update_salience(i, agreement_score=1.0, novelty_score=0.0)

        top_patterns = module.get_top_salient_patterns(k=3)

        # Top 3 should be patterns 9, 8, 7 (highest salience)
        top_ids = [pid for pid, _ in top_patterns]
        assert top_ids[0] == 9
        assert top_ids[1] == 8
        assert top_ids[2] == 7

        # Saliences should be sorted descending
        saliences = [s for _, s in top_patterns]
        assert saliences[0] >= saliences[1] >= saliences[2]

    def test_reset(self):
        """Test resetting module."""
        config = SalienceReplayConfig(salience_tagged_replay=True)
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})
        module.update_salience(1, agreement_score=1.0, novelty_score=1.0)
        module.current_step = 10
        module.episode_count = 5

        module.reset()

        assert len(module.patterns) == 0
        assert module.current_step == 0
        assert module.episode_count == 0
        assert len(module.replay_history) == 0

    def test_high_salience_patterns_consolidated(self):
        """Test that high-salience patterns get replayed more in consolidation."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            alpha_consensus=0.5,
            alpha_novelty=0.5,
            num_replay_samples=50
        )
        module = SalienceReplayModule(config)

        # Create 10 patterns
        # Patterns 0-4: low salience (rarely agree with consensus)
        # Patterns 5-9: high salience (often agree with consensus)
        for i in range(10):
            module.add_pattern(i, {'data': i})

        # Simulate several episodes
        for episode in range(20):
            # Update saliences
            for i in range(5):
                module.update_salience(i, agreement_score=0.1, novelty_score=0.0)
            for i in range(5, 10):
                module.update_salience(i, agreement_score=1.0, novelty_score=0.5)

        # Track which patterns get replayed
        replay_counts = {i: 0 for i in range(10)}

        def learning_fn(data):
            replay_counts[data['data']] += 1

        # Run replay phase
        module.run_replay_phase(learning_fn)

        # High-salience patterns (5-9) should be replayed more than low-salience (0-4)
        high_salience_replays = sum(replay_counts[i] for i in range(5, 10))
        low_salience_replays = sum(replay_counts[i] for i in range(5))

        assert high_salience_replays > low_salience_replays

    def test_baseline_behavior_preserved(self):
        """Test that baseline behavior is preserved when feature is disabled."""
        config_disabled = SalienceReplayConfig(salience_tagged_replay=False)
        module_disabled = SalienceReplayModule(config_disabled)

        module_disabled.add_pattern(1, {'data': 'test'})
        module_disabled.update_salience(1, agreement_score=1.0, novelty_score=1.0)

        # Salience should remain 0.0 when getting (feature disabled)
        salience = module_disabled.get_salience(1)
        assert salience == 0.0


class TestSalienceReplayEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_pattern(self):
        """Test with a single pattern."""
        config = SalienceReplayConfig(salience_tagged_replay=True)
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})
        module.update_salience(1, agreement_score=1.0, novelty_score=0.0)

        samples = module.sample_patterns_by_salience(5)
        # Should sample the single pattern (possibly multiple times)
        assert len(samples) > 0

    def test_many_patterns(self):
        """Test with many patterns."""
        config = SalienceReplayConfig(salience_tagged_replay=True)
        module = SalienceReplayModule(config)

        # Add 100 patterns
        for i in range(100):
            module.add_pattern(i, {'data': i})
            module.update_salience(i, agreement_score=0.5, novelty_score=0.5)

        samples = module.sample_patterns_by_salience(20)
        assert len(samples) == 20

    def test_zero_decay_rate_boundary(self):
        """Test with decay rate very close to 1.0 (no decay)."""
        config = SalienceReplayConfig(
            salience_tagged_replay=True,
            decay_rate=1.0
        )
        module = SalienceReplayModule(config)

        module.add_pattern(1, {'data': 'test'})
        module.update_salience(1, agreement_score=1.0, novelty_score=0.0)

        initial_salience = module.get_salience(1)

        # Advance time
        module.current_step = 100

        # With decay_rate=1.0, salience should not decay
        decayed_salience = module.get_salience(1)
        assert np.isclose(decayed_salience, initial_salience)
