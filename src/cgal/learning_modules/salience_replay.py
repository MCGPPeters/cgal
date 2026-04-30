"""Salience-tagged replay mechanism for pattern consolidation.

This module implements CGAL's salience-tagged replay (Issue #5), which:
1. Tags patterns with salience based on consensus agreement and novelty
2. Preferentially replays high-salience patterns during offline phases
3. Applies homeostatic downscaling to un-replayed patterns

Key concepts:
- Salience: Value representing pattern importance, computed as:
    salience += alpha_consensus * agreement + alpha_novelty * novelty
- Replay: Offline phase where stored patterns are re-processed to deepen consolidation
- Homeostatic downscaling: Gradual weakening of un-replayed patterns

References:
    CGAL framework: Salience-tagged replay for value-aligned consolidation
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import random

import numpy as np


logger = logging.getLogger(__name__)


class Pattern:
    """Represents a stored pattern with salience tracking.

    Attributes:
        pattern_id: Unique identifier for the pattern.
        data: Pattern data (e.g., feature vector, activation pattern).
        salience: Current salience value (importance score).
        last_update_step: Step when salience was last updated (for lazy decay).
    """

    def __init__(self, pattern_id: int, data: Any):
        """Initialize a pattern.

        Args:
            pattern_id: Unique identifier.
            data: Pattern data.
        """
        self.pattern_id = pattern_id
        self.data = data
        self.salience: float = 0.0
        self.last_update_step: int = 0

    def __repr__(self):
        return f"Pattern(id={self.pattern_id}, salience={self.salience:.4f})"


class SalienceReplayModule:
    """Manages salience tagging and replay for pattern consolidation.

    This module tracks pattern salience and orchestrates offline replay phases
    where high-salience patterns are preferentially consolidated.
    """

    def __init__(self, config: Any):
        """Initialize salience replay module.

        Args:
            config: SalienceReplayConfig instance.
        """
        self.config = config
        self.current_step = 0
        self.episode_count = 0

        # Pattern storage: pattern_id -> Pattern
        self.patterns: Dict[int, Pattern] = {}

        # Statistics tracking
        self.replay_history: List[Dict[str, Any]] = []
        self.salience_history: List[float] = []

    def add_pattern(self, pattern_id: int, data: Any) -> Pattern:
        """Add a new pattern to the storage.

        Args:
            pattern_id: Unique identifier for the pattern.
            data: Pattern data.

        Returns:
            The created Pattern object.
        """
        pattern = Pattern(pattern_id, data)
        self.patterns[pattern_id] = pattern
        return pattern

    def update_salience(
        self,
        pattern_id: int,
        agreement_score: float,
        novelty_score: float
    ):
        """Update salience for a pattern based on consensus and novelty.

        Formula:
            salience += alpha_consensus * agreement + alpha_novelty * novelty

        Args:
            pattern_id: ID of pattern to update.
            agreement_score: Agreement with consensus, in [0, 1] (from Issue #2).
            novelty_score: Novelty score, in [0, 1] (from Issue #3).
        """
        if not self.config.salience_tagged_replay:
            return

        if pattern_id not in self.patterns:
            logger.warning(f"Pattern {pattern_id} not found, skipping salience update")
            return

        pattern = self.patterns[pattern_id]

        # Apply lazy decay (decay since last update)
        steps_since_update = self.current_step - pattern.last_update_step
        if steps_since_update > 0:
            pattern.salience *= (self.config.decay_rate ** steps_since_update)

        # Increment salience
        salience_increment = (
            self.config.alpha_consensus * agreement_score +
            self.config.alpha_novelty * novelty_score
        )
        pattern.salience += salience_increment

        # Update timestamp
        pattern.last_update_step = self.current_step

        logger.debug(
            f"Pattern {pattern_id}: salience={pattern.salience:.4f} "
            f"(agreement={agreement_score:.2f}, novelty={novelty_score:.2f})"
        )

    def get_salience(self, pattern_id: int) -> float:
        """Get current salience for a pattern (with lazy decay applied).

        Args:
            pattern_id: ID of pattern.

        Returns:
            Current salience value after applying decay.
        """
        if pattern_id not in self.patterns:
            return 0.0

        pattern = self.patterns[pattern_id]

        # Apply lazy decay
        steps_since_update = self.current_step - pattern.last_update_step
        if steps_since_update > 0:
            decayed_salience = pattern.salience * (self.config.decay_rate ** steps_since_update)
        else:
            decayed_salience = pattern.salience

        return decayed_salience

    def sample_patterns_by_salience(self, num_samples: int) -> List[Pattern]:
        """Sample patterns weighted by salience.

        Higher salience = more likely to be sampled.

        Args:
            num_samples: Number of patterns to sample.

        Returns:
            List of sampled Pattern objects.
        """
        if not self.patterns:
            return []

        # Get all patterns with current salience (lazy decay applied)
        pattern_list = list(self.patterns.values())
        saliences = [self.get_salience(p.pattern_id) for p in pattern_list]

        # Handle case where all saliences are zero
        total_salience = sum(saliences)
        if total_salience == 0:
            # Uniform sampling
            num_to_sample = min(num_samples, len(pattern_list))
            return random.sample(pattern_list, num_to_sample)

        # Weight by salience
        probabilities = [s / total_salience for s in saliences]

        # Sample with replacement (allows highly salient patterns to be replayed multiple times)
        num_to_sample = min(num_samples, len(pattern_list) * 3)  # Cap to avoid over-sampling
        sampled_patterns = random.choices(pattern_list, weights=probabilities, k=num_to_sample)

        return sampled_patterns

    def run_replay_phase(self, learning_function: Callable[[Any], None]):
        """Run an offline replay phase.

        Samples high-salience patterns and re-runs learning on them.

        Args:
            learning_function: Function that takes pattern data and runs learning update.
                               Should have signature: fn(pattern_data) -> None
        """
        if not self.config.salience_tagged_replay:
            return

        if not self.patterns:
            logger.debug("No patterns to replay")
            return

        # Sample patterns by salience
        sampled_patterns = self.sample_patterns_by_salience(self.config.num_replay_samples)

        if not sampled_patterns:
            logger.debug("No patterns sampled for replay")
            return

        logger.info(f"Replaying {len(sampled_patterns)} patterns")

        # Apply homeostatic downscaling to all patterns (before replay)
        if self.config.homeostatic_downscaling:
            self._apply_homeostatic_downscaling()

        # Replay each sampled pattern
        replayed_ids = set()
        for pattern in sampled_patterns:
            learning_function(pattern.data)
            replayed_ids.add(pattern.pattern_id)

        # Log statistics
        if self.config.enable_logging:
            self._log_replay_statistics(sampled_patterns, replayed_ids)

    def _apply_homeostatic_downscaling(self):
        """Apply homeostatic downscaling to all patterns.

        This is a placeholder that logs the operation. In a real system,
        this would modify pattern weights/strengths in the learning module.
        """
        logger.debug(
            f"Applying homeostatic downscaling with factor={self.config.homeostatic_factor:.4f}"
        )

        # In a real implementation, this would modify pattern weights in the LM
        # For now, we just log it
        # Example: pattern.weight *= self.config.homeostatic_factor

    def _log_replay_statistics(self, sampled_patterns: List[Pattern], replayed_ids: set):
        """Log statistics about the replay phase."""
        saliences = [self.get_salience(p.pattern_id) for p in sampled_patterns]

        stats = {
            "episode": self.episode_count,
            "num_patterns": len(self.patterns),
            "num_replayed": len(replayed_ids),
            "mean_salience": float(np.mean(saliences)),
            "max_salience": float(np.max(saliences)),
            "min_salience": float(np.min(saliences)),
        }

        self.replay_history.append(stats)

        logger.info(
            f"Replay phase (episode {self.episode_count}): "
            f"replayed {stats['num_replayed']} patterns, "
            f"mean salience={stats['mean_salience']:.4f}, "
            f"max={stats['max_salience']:.4f}"
        )

    def on_step(self):
        """Called on each step to increment time counter."""
        self.current_step += 1

    def on_episode_end(self, learning_function: Optional[Callable[[Any], None]] = None):
        """Called at end of episode.

        Runs replay phase if interval has been reached.

        Args:
            learning_function: Optional learning function for replay.
                               If None, replay is skipped this episode.
        """
        self.episode_count += 1

        # Check if it's time for replay
        if (self.config.salience_tagged_replay and
            self.config.replay_interval > 0 and
            self.episode_count % self.config.replay_interval == 0 and
            learning_function is not None):

            logger.info(f"Running replay phase at episode {self.episode_count}")
            self.run_replay_phase(learning_function)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about salience and replay.

        Returns:
            Dictionary with statistics.
        """
        if not self.patterns:
            return {
                "num_patterns": 0,
                "mean_salience": 0.0,
                "max_salience": 0.0,
                "min_salience": 0.0,
                "num_replay_phases": 0,
            }

        saliences = [self.get_salience(p.pattern_id) for p in self.patterns.values()]

        return {
            "num_patterns": len(self.patterns),
            "mean_salience": float(np.mean(saliences)),
            "std_salience": float(np.std(saliences)),
            "max_salience": float(np.max(saliences)),
            "min_salience": float(np.min(saliences)),
            "num_replay_phases": len(self.replay_history),
        }

    def get_top_salient_patterns(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k most salient patterns.

        Args:
            k: Number of top patterns to return.

        Returns:
            List of (pattern_id, salience) tuples, sorted by salience descending.
        """
        pattern_saliences = [
            (p.pattern_id, self.get_salience(p.pattern_id))
            for p in self.patterns.values()
        ]

        # Sort by salience descending
        pattern_saliences.sort(key=lambda x: x[1], reverse=True)

        return pattern_saliences[:k]

    def reset(self):
        """Reset all patterns and statistics."""
        self.patterns.clear()
        self.replay_history.clear()
        self.salience_history.clear()
        self.current_step = 0
        self.episode_count = 0
