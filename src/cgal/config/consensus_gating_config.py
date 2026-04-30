"""Configuration for consensus-gated plasticity mechanism.

This module defines the configuration parameters for CGAL's consensus-gated
plasticity (Issue #2), which uses voting consensus as a credit-assignment signal.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConsensusGatingConfig:
    """Configuration for consensus-gated plasticity.

    Attributes:
        consensus_gated_plasticity: Whether to enable consensus-gated plasticity.
            Defaults to False for baseline compatibility.
        alpha: Weight of consensus agreement in gating factor computation.
            Value in [0, 1]. Default: 0.7.
            Formula: g_m = alpha * agreement + (1-alpha) * baseline_rate
        baseline_rate: Baseline learning rate when there's no consensus information.
            Value in [0, 1]. Default: 1.0 (no gating).
        agreement_tolerance: Tolerance for pose agreement in distance units.
            Default: 0.1 (10% tolerance).
        enable_logging: Whether to log gating factors per step. Default: True.
    """

    consensus_gated_plasticity: bool = False
    alpha: float = 0.7
    baseline_rate: float = 1.0
    agreement_tolerance: float = 0.1
    enable_logging: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.alpha <= 1:
            raise ValueError(
                f"alpha must be in [0, 1], got {self.alpha}"
            )
        if not 0 <= self.baseline_rate <= 1:
            raise ValueError(
                f"baseline_rate must be in [0, 1], got {self.baseline_rate}"
            )
        if self.agreement_tolerance < 0:
            raise ValueError(
                f"agreement_tolerance must be non-negative, got {self.agreement_tolerance}"
            )
