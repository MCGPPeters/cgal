"""Learned trust weights between learning modules.

This module implements CGAL's learned trust weights (Issue #4), which allows
learning modules to learn which neighbors are reliable voters.

Key formula:
    W[m,n] += gamma * (agreement(n, consensus) - W[m,n])

Where:
- W[m,n]: Trust weight from module m to module n in [trust_min, trust_max]
- gamma: Trust learning rate (default 0.05)
- agreement(n, consensus): How well n's vote aligned with consensus in [0, 1]

References:
    CGAL framework: Learning who to listen to via trust weights
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np


logger = logging.getLogger(__name__)


class TrustWeightsModule:
    """Manages learned trust weights between learning modules.

    Trust weights are directional: W[m,n] represents module m's trust in module n.
    Modules whose votes consistently track consensus get higher trust; those that
    consistently dissent get lower trust.
    """

    def __init__(self, config: Any, num_modules: Optional[int] = None):
        """Initialize trust weights module.

        Args:
            config: TrustWeightsConfig instance with trust parameters.
            num_modules: Number of learning modules (optional, for pre-allocation).
        """
        self.config = config
        self.num_modules = num_modules

        # Trust matrix: (from_id, to_id) -> weight
        self.trust_matrix: Dict[Tuple[int, int], float] = {}

        # History for logging and analysis
        self.trust_history: List[Dict[Tuple[int, int], float]] = []
        self.episode_count = 0

    def initialize_trust(self, module_ids: List[int]):
        """Initialize trust weights for a set of modules.

        All pairs start with trust = 1.0 (uniform trust).

        Args:
            module_ids: List of module IDs.
        """
        for m_id in module_ids:
            for n_id in module_ids:
                if m_id != n_id:  # No self-trust
                    self.trust_matrix[(m_id, n_id)] = 1.0

    def get_trust(self, from_id: int, to_id: int) -> float:
        """Get trust weight from one module to another.

        Args:
            from_id: ID of module whose trust we're querying.
            to_id: ID of module being trusted.

        Returns:
            Trust weight in [trust_min, trust_max], or 1.0 if not initialized.
        """
        if not self.config.learned_trust_weights:
            # Feature disabled, always return 1.0
            return 1.0

        return self.trust_matrix.get((from_id, to_id), 1.0)

    def update_trust(
        self,
        from_id: int,
        to_id: int,
        agreement: float
    ):
        """Update trust weight based on agreement with consensus.

        Formula: W[m,n] += gamma * (agreement - W[m,n])

        Args:
            from_id: ID of module updating its trust.
            to_id: ID of module being evaluated.
            agreement: How well to_id's vote agreed with consensus, in [0, 1].
        """
        if not self.config.learned_trust_weights:
            # Feature disabled, don't update
            return

        # Get current trust
        current_trust = self.trust_matrix.get((from_id, to_id), 1.0)

        # Apply exponential moving average update
        new_trust = current_trust + self.config.trust_learning_rate * (agreement - current_trust)

        # Clip to [trust_min, trust_max]
        new_trust = np.clip(new_trust, self.config.trust_min, self.config.trust_max)

        # Store updated trust
        self.trust_matrix[(from_id, to_id)] = float(new_trust)

        logger.debug(
            f"Trust update: W[{from_id},{to_id}] = {current_trust:.4f} -> {new_trust:.4f} "
            f"(agreement={agreement:.4f})"
        )

    def update_all_trust(
        self,
        module_votes: Dict[int, Dict[str, Any]],
        consensus: Dict[str, Any]
    ):
        """Update trust weights for all modules based on voting round.

        Each module updates its trust in each neighbor based on how well
        the neighbor's vote aligned with the winning consensus.

        Args:
            module_votes: Dict mapping module_id to vote (hypothesis dict).
            consensus: The consensus hypothesis.
        """
        if not self.config.learned_trust_weights:
            return

        module_ids = list(module_votes.keys())

        # Compute agreement of each module with consensus
        agreements = {}
        for module_id, vote in module_votes.items():
            agreements[module_id] = self._compute_agreement(vote, consensus)

        # Each module updates trust in each neighbor
        for from_id in module_ids:
            for to_id in module_ids:
                if from_id != to_id:
                    self.update_trust(from_id, to_id, agreements[to_id])

    def _compute_agreement(
        self,
        vote: Dict[str, Any],
        consensus: Dict[str, Any]
    ) -> float:
        """Compute agreement between a vote and consensus.

        Simple version: 1.0 if object IDs match, 0.0 otherwise.

        Args:
            vote: Module's vote (hypothesis dict with 'object_id').
            consensus: Consensus hypothesis (dict with 'object_id').

        Returns:
            Agreement score in [0, 1].
        """
        if not vote or not consensus:
            return 0.0

        vote_obj_id = vote.get('object_id')
        consensus_obj_id = consensus.get('object_id')

        return 1.0 if vote_obj_id == consensus_obj_id else 0.0

    def weight_votes(
        self,
        from_id: int,
        neighbor_votes: Dict[int, Any]
    ) -> Dict[int, float]:
        """Get trust weights for weighting neighbor votes.

        Args:
            from_id: ID of module whose perspective we're using.
            neighbor_votes: Dict mapping neighbor IDs to their votes.

        Returns:
            Dict mapping neighbor IDs to trust weights.
        """
        weights = {}
        for neighbor_id in neighbor_votes.keys():
            weights[neighbor_id] = self.get_trust(from_id, neighbor_id)
        return weights

    def log_trust_weights(self):
        """Log current trust weights (called periodically)."""
        if not self.config.enable_logging:
            return

        # Store snapshot
        self.trust_history.append(dict(self.trust_matrix))

        # Log summary statistics
        if self.trust_matrix:
            weights = np.array(list(self.trust_matrix.values()))
            logger.info(
                f"Episode {self.episode_count}: Trust weights - "
                f"mean={np.mean(weights):.4f}, std={np.std(weights):.4f}, "
                f"min={np.min(weights):.4f}, max={np.max(weights):.4f}"
            )

    def on_episode_end(self):
        """Called at the end of each episode for logging."""
        self.episode_count += 1

        if (self.config.log_interval > 0 and
            self.episode_count % self.config.log_interval == 0):
            self.log_trust_weights()

    def get_trust_matrix_dict(self) -> Dict[Tuple[int, int], float]:
        """Get the full trust matrix.

        Returns:
            Dict mapping (from_id, to_id) pairs to trust weights.
        """
        return dict(self.trust_matrix)

    def get_trust_matrix_array(self, module_ids: List[int]) -> np.ndarray:
        """Get trust matrix as a 2D numpy array.

        Args:
            module_ids: List of module IDs (determines array ordering).

        Returns:
            2D array where entry [i,j] is trust from module_ids[i] to module_ids[j].
        """
        n = len(module_ids)
        matrix = np.ones((n, n))

        for i, from_id in enumerate(module_ids):
            for j, to_id in enumerate(module_ids):
                if from_id != to_id:
                    matrix[i, j] = self.get_trust(from_id, to_id)
                else:
                    matrix[i, j] = 0.0  # No self-trust

        return matrix

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about trust weights.

        Returns:
            Dictionary with statistics (mean, std, min, max, count).
        """
        if not self.trust_matrix:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        weights = np.array(list(self.trust_matrix.values()))
        return {
            "count": len(weights),
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
        }

    def reset(self):
        """Reset all trust weights to 1.0."""
        for key in self.trust_matrix:
            self.trust_matrix[key] = 1.0
        self.trust_history = []
        self.episode_count = 0
