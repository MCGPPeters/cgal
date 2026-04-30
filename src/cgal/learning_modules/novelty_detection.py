"""Novelty detection from hypothesis distribution shape.

This module implements CGAL's novelty detection mechanism (Issue #3), which
computes a novelty signal from the entropy and peak confidence of the hypothesis
distribution.

Key formula:
    novelty_score = entropy(hypothesis_dist) * (1 - peak_confidence)

Where:
- entropy = -sum(p * log(p)) over normalized distribution
- peak_confidence = max(p) after normalization
- Result is normalized to [0, 1]

References:
    CGAL Section 3.8: Novelty detection from hypothesis-generation machinery
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


logger = logging.getLogger(__name__)


class NoveltyDetector:
    """Detects novelty from hypothesis distribution shape.

    Analyzes the entropy and peak confidence of a hypothesis distribution to
    determine whether an observation represents a novel pattern or familiar one.

    - Sharp distribution + high peak confidence = familiar pattern
    - Broad distribution + low peak confidence = novel pattern
    - Sharp distribution + low peak confidence = ambiguous (wait for more evidence)
    """

    def __init__(self, config: Any):
        """Initialize novelty detector.

        Args:
            config: NoveltyDetectionConfig instance with detection parameters.
        """
        self.config = config
        self.novelty_scores_history: List[float] = []
        self._max_entropy_cache: Optional[float] = None

    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """Compute Shannon entropy of a probability distribution.

        Args:
            probabilities: Normalized probability distribution (sums to 1).

        Returns:
            Shannon entropy: -sum(p * log(p)) where log is natural logarithm.
            Returns 0 if distribution is deterministic.
        """
        # Filter out zero probabilities to avoid log(0)
        nonzero_probs = probabilities[probabilities > 0]
        if len(nonzero_probs) == 0:
            return 0.0

        # Compute Shannon entropy
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
        return float(entropy)

    def compute_peak_confidence(self, probabilities: np.ndarray) -> float:
        """Compute peak confidence (maximum probability).

        Args:
            probabilities: Normalized probability distribution.

        Returns:
            Maximum probability value in [0, 1].
        """
        if len(probabilities) == 0:
            return 0.0
        return float(np.max(probabilities))

    def normalize_probabilities(
        self,
        hypothesis_scores: Dict[Any, float]
    ) -> Tuple[np.ndarray, List[Any]]:
        """Normalize hypothesis scores to probabilities.

        Args:
            hypothesis_scores: Dict mapping hypothesis IDs to confidence scores.

        Returns:
            Tuple of (normalized probabilities array, list of hypothesis IDs in same order).
        """
        if not hypothesis_scores:
            return np.array([]), []

        hypotheses = list(hypothesis_scores.keys())
        scores = np.array([hypothesis_scores[h] for h in hypotheses])

        # Handle case where all scores are zero or negative
        if np.all(scores <= 0):
            # Uniform distribution
            probabilities = np.ones(len(scores)) / len(scores)
        else:
            # Normalize positive scores
            scores = np.maximum(scores, 0)  # Clip negatives to zero
            total = np.sum(scores)
            if total > 0:
                probabilities = scores / total
            else:
                probabilities = np.ones(len(scores)) / len(scores)

        return probabilities, hypotheses

    def compute_max_entropy(self, n_hypotheses: int) -> float:
        """Compute maximum possible entropy for n hypotheses.

        Maximum entropy occurs with uniform distribution.

        Args:
            n_hypotheses: Number of hypotheses in distribution.

        Returns:
            Maximum entropy: log(n_hypotheses).
        """
        if n_hypotheses <= 1:
            return 0.0
        return math.log(n_hypotheses)

    def compute_novelty_score(
        self,
        hypothesis_scores: Dict[Any, float],
        normalize: bool = True
    ) -> float:
        """Compute novelty score from hypothesis distribution.

        Formula: novelty_score = entropy(dist) * (1 - peak_confidence)

        Interpretation:
        - High entropy + low peak confidence → high novelty (novel pattern)
        - Low entropy + high peak confidence → low novelty (familiar pattern)
        - Low entropy + low peak confidence → ambiguous (wait for evidence)

        Args:
            hypothesis_scores: Dict mapping hypothesis IDs to confidence values.
            normalize: Whether to normalize the score to [0, 1]. Default: True.

        Returns:
            Novelty score in [0, 1] if normalized, otherwise unnormalized value.
        """
        if not hypothesis_scores:
            # No hypotheses means maximum novelty
            return 1.0 if normalize else float('inf')

        # Convert scores to probabilities
        probabilities, _ = self.normalize_probabilities(hypothesis_scores)

        # Compute entropy and peak confidence
        entropy = self.compute_entropy(probabilities)
        peak_conf = self.compute_peak_confidence(probabilities)

        # Compute raw novelty score
        raw_score = entropy * (1 - peak_conf)

        if not normalize:
            return raw_score

        # Normalize to [0, 1]
        n_hypotheses = len(hypothesis_scores)
        max_entropy = self.compute_max_entropy(n_hypotheses)

        if max_entropy == 0:
            # Single hypothesis case
            normalized_score = 0.0 if peak_conf > 0.5 else 1.0
        else:
            # Theoretical maximum raw score occurs at:
            # entropy = max_entropy, peak_confidence = 0 (impossible, but use 1/n)
            # So max_raw_score ≈ max_entropy * (1 - 1/n)
            max_raw_score = max_entropy * (1 - 1/n_hypotheses) if n_hypotheses > 1 else 1.0
            normalized_score = raw_score / max_raw_score if max_raw_score > 0 else 0.0
            # Clip to [0, 1] in case of numerical issues
            normalized_score = np.clip(normalized_score, 0.0, 1.0)

        return float(normalized_score)

    def is_novel(
        self,
        hypothesis_scores: Dict[Any, float]
    ) -> Tuple[bool, float]:
        """Determine if observation represents a novel pattern.

        Args:
            hypothesis_scores: Dict mapping hypothesis IDs to confidence values.

        Returns:
            Tuple of (is_novel boolean, novelty_score float).
            is_novel is True if novelty_score > novelty_threshold.
        """
        if not self.config.hypothesis_novelty_detection:
            # Feature disabled, always return non-novel
            return False, 0.0

        novelty_score = self.compute_novelty_score(hypothesis_scores, normalize=True)

        # Log the score if logging is enabled
        if self.config.enable_logging:
            self.novelty_scores_history.append(novelty_score)
            logger.debug(
                f"Novelty score: {novelty_score:.4f} "
                f"(threshold: {self.config.novelty_threshold:.4f})"
            )

        is_novel = novelty_score > self.config.novelty_threshold

        return is_novel, novelty_score

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected novelty scores.

        Returns:
            Dictionary with statistics (mean, std, min, max, count) of novelty scores.
        """
        if not self.novelty_scores_history:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        scores = np.array(self.novelty_scores_history)
        return {
            "count": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }

    def reset_history(self):
        """Clear the history of novelty scores."""
        self.novelty_scores_history = []
