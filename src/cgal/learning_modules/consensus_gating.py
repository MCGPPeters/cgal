"""Consensus-gated plasticity mechanism.

This module implements CGAL's consensus-gated plasticity (Issue #2), which
modulates learning rate based on agreement with network-wide voting consensus.

Key formula:
    g_m = alpha * agreement + (1-alpha) * baseline_rate

Where:
- g_m: gating factor in [0, 1] applied to pattern update magnitude
- alpha: weight of consensus agreement (default 0.7)
- agreement: measure of how well LM's hypothesis matches consensus in [0, 1]
- baseline_rate: fallback learning rate (default 1.0)

References:
    CGAL framework: Consensus-gated plasticity as credit-assignment signal
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


logger = logging.getLogger(__name__)


class ConsensusGatingModule:
    """Implements consensus-gated plasticity for learning modules.

    Uses voting consensus as a credit-assignment signal: when a module's
    hypothesis agrees with consensus, reinforce strongly (g_m → 1);
    when it disagrees, reduce reinforcement (g_m → baseline_rate).
    """

    def __init__(self, config: Any):
        """Initialize consensus gating module.

        Args:
            config: ConsensusGatingConfig instance with gating parameters.
        """
        self.config = config
        self.gating_factors_history: List[float] = []
        self._last_gating_factor: float = 1.0

    def compute_agreement(
        self,
        lm_hypothesis: Dict[str, Any],
        consensus_hypothesis: Dict[str, Any]
    ) -> float:
        """Compute agreement between LM hypothesis and consensus.

        Agreement scoring:
        - 1.0: Same object-id AND pose within tolerance
        - 0.5: Same object-id but different pose
        - 0.0: Different object-id

        Args:
            lm_hypothesis: Dict with keys 'object_id' and optionally 'pose'
            consensus_hypothesis: Dict with keys 'object_id' and optionally 'pose'

        Returns:
            Agreement score in [0, 1].
        """
        if not lm_hypothesis or not consensus_hypothesis:
            # No hypothesis means no agreement information
            return 0.0

        lm_obj_id = lm_hypothesis.get('object_id')
        consensus_obj_id = consensus_hypothesis.get('object_id')

        # Different object IDs → no agreement
        if lm_obj_id != consensus_obj_id:
            return 0.0

        # Same object ID → check pose if available
        lm_pose = lm_hypothesis.get('pose')
        consensus_pose = consensus_hypothesis.get('pose')

        if lm_pose is None or consensus_pose is None:
            # Same object but no pose information → partial agreement
            return 0.5

        # Compute pose distance
        lm_pose_array = np.array(lm_pose) if not isinstance(lm_pose, np.ndarray) else lm_pose
        consensus_pose_array = np.array(consensus_pose) if not isinstance(consensus_pose, np.ndarray) else consensus_pose

        pose_distance = np.linalg.norm(lm_pose_array - consensus_pose_array)

        # Within tolerance → full agreement
        if pose_distance <= self.config.agreement_tolerance:
            return 1.0

        # Same object, pose differs → partial agreement
        return 0.5

    def compute_gating_factor(
        self,
        lm_hypothesis: Optional[Dict[str, Any]],
        consensus_hypothesis: Optional[Dict[str, Any]]
    ) -> float:
        """Compute gating factor for pattern updates.

        Formula: g_m = alpha * agreement + (1-alpha) * baseline_rate

        Args:
            lm_hypothesis: Learning module's top hypothesis
            consensus_hypothesis: Network-wide consensus hypothesis

        Returns:
            Gating factor in [0, 1] to apply to pattern updates.
        """
        if not self.config.consensus_gated_plasticity:
            # Feature disabled, always return 1.0 (no gating)
            return 1.0

        # If no consensus yet, use baseline rate (early in episode)
        if consensus_hypothesis is None:
            gating_factor = self.config.baseline_rate
        else:
            # Compute agreement and apply formula
            agreement = self.compute_agreement(lm_hypothesis, consensus_hypothesis)
            gating_factor = (
                self.config.alpha * agreement +
                (1 - self.config.alpha) * self.config.baseline_rate
            )

        # Log if enabled
        if self.config.enable_logging:
            self.gating_factors_history.append(gating_factor)
            logger.debug(
                f"Gating factor: {gating_factor:.4f} "
                f"(alpha={self.config.alpha:.2f})"
            )

        self._last_gating_factor = gating_factor
        return gating_factor

    def apply_gating(
        self,
        update_magnitude: float,
        lm_hypothesis: Optional[Dict[str, Any]],
        consensus_hypothesis: Optional[Dict[str, Any]]
    ) -> float:
        """Apply consensus gating to an update magnitude.

        Args:
            update_magnitude: Original update magnitude (e.g., learning rate)
            lm_hypothesis: Learning module's hypothesis
            consensus_hypothesis: Network consensus hypothesis

        Returns:
            Gated update magnitude (update_magnitude * g_m).
        """
        gating_factor = self.compute_gating_factor(lm_hypothesis, consensus_hypothesis)
        return update_magnitude * gating_factor

    def get_last_gating_factor(self) -> float:
        """Get the most recently computed gating factor.

        Returns:
            Last gating factor, or 1.0 if none computed yet.
        """
        return self._last_gating_factor

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about gating factors.

        Returns:
            Dictionary with statistics (mean, std, min, max, count).
        """
        if not self.gating_factors_history:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        factors = np.array(self.gating_factors_history)
        return {
            "count": len(factors),
            "mean": float(np.mean(factors)),
            "std": float(np.std(factors)),
            "min": float(np.min(factors)),
            "max": float(np.max(factors)),
        }

    def reset_history(self):
        """Clear the history of gating factors."""
        self.gating_factors_history = []
        self._last_gating_factor = 1.0
