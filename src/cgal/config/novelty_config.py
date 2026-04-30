"""Configuration for novelty detection mechanism.

This module defines the configuration parameters for CGAL's novelty detection
from hypothesis distribution shape (Issue #3).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NoveltyDetectionConfig:
    """Configuration for hypothesis-based novelty detection.

    Attributes:
        hypothesis_novelty_detection: Whether to enable novelty detection from
            hypothesis distribution shape. Defaults to False for baseline compatibility.
        novelty_threshold: Threshold above which observations are routed to new
            pattern allocation. Value in [0, 1]. Default: 0.7.
        min_entropy: Minimum possible entropy for normalization (default: 0.0).
        max_entropy: Maximum possible entropy for normalization. If None, will be
            computed based on the number of hypotheses. Default: None.
        enable_logging: Whether to log novelty scores per step. Default: True.
    """

    hypothesis_novelty_detection: bool = False
    novelty_threshold: float = 0.7
    min_entropy: float = 0.0
    max_entropy: Optional[float] = None
    enable_logging: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.novelty_threshold <= 1:
            raise ValueError(
                f"novelty_threshold must be in [0, 1], got {self.novelty_threshold}"
            )
        if self.min_entropy < 0:
            raise ValueError(
                f"min_entropy must be non-negative, got {self.min_entropy}"
            )
        if self.max_entropy is not None and self.max_entropy <= self.min_entropy:
            raise ValueError(
                f"max_entropy must be greater than min_entropy, "
                f"got max={self.max_entropy}, min={self.min_entropy}"
            )
