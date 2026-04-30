"""Configuration for learned trust weights mechanism.

This module defines the configuration parameters for CGAL's learned trust weights
(Issue #4), which allows learning modules to learn who to trust based on voting history.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrustWeightsConfig:
    """Configuration for learned trust weights between LMs.

    Attributes:
        learned_trust_weights: Whether to enable learned trust weights.
            Defaults to False for baseline compatibility.
        trust_learning_rate: Learning rate (gamma) for trust weight updates.
            Value in [0, 1]. Default: 0.05.
            Formula: W[m,n] += gamma * (agreement(n, consensus) - W[m,n])
        trust_min: Minimum trust weight to prevent complete silencing.
            Value in (0, 1]. Default: 0.1.
        trust_max: Maximum trust weight. Default: 1.0.
        log_interval: Log trust weights every N episodes. Default: 10.
            Set to 0 to disable periodic logging.
        enable_logging: Whether to track trust weight history. Default: True.
    """

    learned_trust_weights: bool = False
    trust_learning_rate: float = 0.05
    trust_min: float = 0.1
    trust_max: float = 1.0
    log_interval: int = 10
    enable_logging: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.trust_learning_rate <= 1:
            raise ValueError(
                f"trust_learning_rate must be in [0, 1], got {self.trust_learning_rate}"
            )
        if not 0 < self.trust_min <= 1:
            raise ValueError(
                f"trust_min must be in (0, 1], got {self.trust_min}"
            )
        if not 0 < self.trust_max <= 1:
            raise ValueError(
                f"trust_max must be in (0, 1], got {self.trust_max}"
            )
        if self.trust_min > self.trust_max:
            raise ValueError(
                f"trust_min must be <= trust_max, got min={self.trust_min}, max={self.trust_max}"
            )
        if self.log_interval < 0:
            raise ValueError(
                f"log_interval must be non-negative, got {self.log_interval}"
            )
