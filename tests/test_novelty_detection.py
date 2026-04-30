"""Unit tests for novelty detection module.

Tests the NoveltyDetector implementation from Issue #3.
"""

import math
import numpy as np
import pytest

from cgal.config.novelty_config import NoveltyDetectionConfig
from cgal.learning_modules.novelty_detection import NoveltyDetector


class TestNoveltyDetectionConfig:
    """Test NoveltyDetectionConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NoveltyDetectionConfig()
        assert config.hypothesis_novelty_detection is False
        assert config.novelty_threshold == 0.7
        assert config.min_entropy == 0.0
        assert config.max_entropy is None
        assert config.enable_logging is True

    def test_valid_config(self):
        """Test valid custom configuration."""
        config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            novelty_threshold=0.8,
            min_entropy=0.0,
            max_entropy=5.0,
            enable_logging=False
        )
        assert config.hypothesis_novelty_detection is True
        assert config.novelty_threshold == 0.8
        assert config.max_entropy == 5.0

    def test_invalid_novelty_threshold(self):
        """Test that invalid novelty threshold raises error."""
        with pytest.raises(ValueError, match="novelty_threshold must be in"):
            NoveltyDetectionConfig(novelty_threshold=1.5)

        with pytest.raises(ValueError, match="novelty_threshold must be in"):
            NoveltyDetectionConfig(novelty_threshold=-0.1)

    def test_invalid_min_entropy(self):
        """Test that negative min_entropy raises error."""
        with pytest.raises(ValueError, match="min_entropy must be non-negative"):
            NoveltyDetectionConfig(min_entropy=-1.0)

    def test_invalid_max_entropy(self):
        """Test that max_entropy <= min_entropy raises error."""
        with pytest.raises(ValueError, match="max_entropy must be greater than"):
            NoveltyDetectionConfig(min_entropy=5.0, max_entropy=3.0)


class TestNoveltyDetector:
    """Test NoveltyDetector functionality."""

    def test_initialization(self):
        """Test detector initialization."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)
        assert detector.config == config
        assert len(detector.novelty_scores_history) == 0

    def test_normalize_probabilities(self):
        """Test probability normalization."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        # Test normal case
        scores = {"h1": 2.0, "h2": 1.0, "h3": 1.0}
        probs, hypotheses = detector.normalize_probabilities(scores)
        assert len(probs) == 3
        assert len(hypotheses) == 3
        assert np.isclose(np.sum(probs), 1.0)
        assert np.isclose(probs[hypotheses.index("h1")], 0.5)

    def test_normalize_probabilities_empty(self):
        """Test normalization with empty scores."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        scores = {}
        probs, hypotheses = detector.normalize_probabilities(scores)
        assert len(probs) == 0
        assert len(hypotheses) == 0

    def test_normalize_probabilities_zero_scores(self):
        """Test normalization when all scores are zero."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        scores = {"h1": 0.0, "h2": 0.0}
        probs, hypotheses = detector.normalize_probabilities(scores)
        assert len(probs) == 2
        # Should give uniform distribution
        assert np.allclose(probs, [0.5, 0.5])

    def test_compute_entropy_deterministic(self):
        """Test entropy of deterministic distribution (entropy = 0)."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        probs = np.array([1.0, 0.0, 0.0])
        entropy = detector.compute_entropy(probs)
        assert np.isclose(entropy, 0.0)

    def test_compute_entropy_uniform(self):
        """Test entropy of uniform distribution (maximum entropy)."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = detector.compute_entropy(probs)
        expected = math.log(4)  # log(4) for uniform over 4 items
        assert np.isclose(entropy, expected)

    def test_compute_entropy_mixed(self):
        """Test entropy of mixed distribution."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        probs = np.array([0.5, 0.3, 0.2])
        entropy = detector.compute_entropy(probs)
        expected = -(0.5 * math.log(0.5) + 0.3 * math.log(0.3) + 0.2 * math.log(0.2))
        assert np.isclose(entropy, expected)

    def test_compute_peak_confidence(self):
        """Test peak confidence computation."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        probs = np.array([0.6, 0.3, 0.1])
        peak = detector.compute_peak_confidence(probs)
        assert np.isclose(peak, 0.6)

    def test_compute_peak_confidence_empty(self):
        """Test peak confidence with empty array."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        probs = np.array([])
        peak = detector.compute_peak_confidence(probs)
        assert peak == 0.0

    def test_compute_max_entropy(self):
        """Test maximum entropy calculation."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        assert np.isclose(detector.compute_max_entropy(1), 0.0)
        assert np.isclose(detector.compute_max_entropy(2), math.log(2))
        assert np.isclose(detector.compute_max_entropy(4), math.log(4))

    def test_novelty_score_familiar_pattern(self):
        """Test low novelty score for familiar pattern (sharp + high confidence)."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        # Sharp distribution with high peak confidence -> familiar
        scores = {"h1": 10.0, "h2": 1.0, "h3": 1.0}
        novelty = detector.compute_novelty_score(scores, normalize=True)

        # Should be low novelty (familiar pattern)
        assert 0.0 <= novelty < 0.3

    def test_novelty_score_novel_pattern(self):
        """Test high novelty score for novel pattern (broad + low confidence)."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        # Broad (uniform) distribution -> novel
        scores = {"h1": 1.0, "h2": 1.0, "h3": 1.0, "h4": 1.0}
        novelty = detector.compute_novelty_score(scores, normalize=True)

        # Should be high novelty (novel pattern)
        assert 0.5 < novelty <= 1.0

    def test_novelty_score_empty_hypotheses(self):
        """Test novelty score with no hypotheses."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        scores = {}
        novelty = detector.compute_novelty_score(scores, normalize=True)

        # Empty should mean maximum novelty
        assert novelty == 1.0

    def test_novelty_score_normalization(self):
        """Test that normalized novelty scores are in [0, 1]."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        test_cases = [
            {"h1": 10.0, "h2": 1.0},
            {"h1": 1.0, "h2": 1.0, "h3": 1.0},
            {"h1": 5.0},
            {"h1": 2.0, "h2": 2.0, "h3": 2.0, "h4": 2.0, "h5": 2.0},
        ]

        for scores in test_cases:
            novelty = detector.compute_novelty_score(scores, normalize=True)
            assert 0.0 <= novelty <= 1.0, f"Score {novelty} out of range for {scores}"

    def test_is_novel_disabled(self):
        """Test that is_novel returns False when feature is disabled."""
        config = NoveltyDetectionConfig(hypothesis_novelty_detection=False)
        detector = NoveltyDetector(config)

        scores = {"h1": 1.0, "h2": 1.0, "h3": 1.0}  # Would be novel if enabled
        is_novel, score = detector.is_novel(scores)

        assert is_novel is False
        assert score == 0.0

    def test_is_novel_enabled_above_threshold(self):
        """Test is_novel when score exceeds threshold."""
        config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            novelty_threshold=0.5,
            enable_logging=False
        )
        detector = NoveltyDetector(config)

        # Uniform distribution should have high novelty
        scores = {"h1": 1.0, "h2": 1.0, "h3": 1.0}
        is_novel, score = detector.is_novel(scores)

        assert is_novel is True
        assert score > 0.5

    def test_is_novel_enabled_below_threshold(self):
        """Test is_novel when score is below threshold."""
        config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            novelty_threshold=0.9,
            enable_logging=False
        )
        detector = NoveltyDetector(config)

        # Sharp distribution should have low novelty
        scores = {"h1": 100.0, "h2": 1.0, "h3": 1.0}
        is_novel, score = detector.is_novel(scores)

        assert is_novel is False
        assert score < 0.9

    def test_logging_enabled(self):
        """Test that novelty scores are logged when logging is enabled."""
        config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            enable_logging=True
        )
        detector = NoveltyDetector(config)

        scores1 = {"h1": 10.0, "h2": 1.0}
        scores2 = {"h1": 1.0, "h2": 1.0}

        detector.is_novel(scores1)
        detector.is_novel(scores2)

        assert len(detector.novelty_scores_history) == 2

    def test_logging_disabled(self):
        """Test that novelty scores are not logged when logging is disabled."""
        config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            enable_logging=False
        )
        detector = NoveltyDetector(config)

        scores = {"h1": 10.0, "h2": 1.0}
        detector.is_novel(scores)

        assert len(detector.novelty_scores_history) == 0

    def test_get_statistics_empty(self):
        """Test statistics with no history."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        stats = detector.get_statistics()
        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_get_statistics_with_history(self):
        """Test statistics computation with history."""
        config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            enable_logging=True
        )
        detector = NoveltyDetector(config)

        # Add some scores
        detector.is_novel({"h1": 10.0, "h2": 1.0})
        detector.is_novel({"h1": 1.0, "h2": 1.0})
        detector.is_novel({"h1": 5.0, "h2": 5.0})

        stats = detector.get_statistics()
        assert stats["count"] == 3
        assert 0.0 <= stats["mean"] <= 1.0
        assert 0.0 <= stats["std"] <= 1.0
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_reset_history(self):
        """Test resetting novelty score history."""
        config = NoveltyDetectionConfig(
            hypothesis_novelty_detection=True,
            enable_logging=True
        )
        detector = NoveltyDetector(config)

        detector.is_novel({"h1": 10.0, "h2": 1.0})
        detector.is_novel({"h1": 1.0, "h2": 1.0})
        assert len(detector.novelty_scores_history) == 2

        detector.reset_history()
        assert len(detector.novelty_scores_history) == 0

    def test_baseline_behavior_preserved(self):
        """Test that baseline behavior is preserved when feature is disabled."""
        config_disabled = NoveltyDetectionConfig(hypothesis_novelty_detection=False)
        config_enabled = NoveltyDetectionConfig(hypothesis_novelty_detection=True)

        detector_disabled = NoveltyDetector(config_disabled)
        detector_enabled = NoveltyDetector(config_enabled)

        scores = {"h1": 5.0, "h2": 3.0, "h3": 2.0}

        # When disabled, should always return False and 0.0
        is_novel_disabled, score_disabled = detector_disabled.is_novel(scores)
        assert is_novel_disabled is False
        assert score_disabled == 0.0

        # When enabled, should compute actual score
        is_novel_enabled, score_enabled = detector_enabled.is_novel(scores)
        assert score_enabled > 0.0  # Should compute real score


class TestNoveltyDetectorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_hypothesis(self):
        """Test with single hypothesis."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        scores = {"h1": 1.0}
        novelty = detector.compute_novelty_score(scores, normalize=True)

        # Single hypothesis with confidence should be familiar (low novelty)
        assert 0.0 <= novelty <= 0.5

    def test_negative_scores(self):
        """Test handling of negative scores."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        scores = {"h1": -1.0, "h2": 5.0, "h3": 3.0}
        novelty = detector.compute_novelty_score(scores, normalize=True)

        # Should handle gracefully and return valid score
        assert 0.0 <= novelty <= 1.0

    def test_very_large_scores(self):
        """Test handling of very large score values."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        scores = {"h1": 1e10, "h2": 1e5, "h3": 1e3}
        novelty = detector.compute_novelty_score(scores, normalize=True)

        # Should handle gracefully
        assert 0.0 <= novelty <= 1.0

    def test_many_hypotheses(self):
        """Test with many hypotheses."""
        config = NoveltyDetectionConfig()
        detector = NoveltyDetector(config)

        # Create uniform distribution over many hypotheses
        scores = {f"h{i}": 1.0 for i in range(100)}
        novelty = detector.compute_novelty_score(scores, normalize=True)

        # High entropy, low peak confidence -> high novelty
        assert novelty > 0.7
