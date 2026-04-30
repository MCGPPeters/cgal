"""Configuration for salience-tagged replay mechanism.

This configuration controls CGAL's salience-tagged replay system (Issue #5),
which prioritizes consolidation of high-value patterns during offline replay.
"""

from dataclasses import dataclass


@dataclass
class SalienceReplayConfig:
    """Configuration for salience-tagged replay.

    Salience tagging boosts plasticity for:
    1. High-confidence consensus events (from Issue #2)
    2. High-novelty events (from Issue #3)

    During offline replay phases, high-salience patterns are preferentially
    replayed to deepen consolidation.

    Attributes:
        salience_tagged_replay: Whether to enable salience-tagged replay.
        alpha_consensus: Weight for consensus agreement in salience (default: 0.5).
        alpha_novelty: Weight for novelty score in salience (default: 0.5).
        decay_rate: Salience decay rate per step (default: 0.99).
        replay_interval: Episodes between replay phases (default: 10).
        num_replay_samples: Number of patterns to replay per phase (default: 50).
        homeostatic_downscaling: Whether to apply homeostatic downscaling (default: True).
        homeostatic_factor: Downscaling factor for un-replayed patterns (default: 0.99).
        enable_logging: Whether to log replay statistics (default: True).
        log_interval: Episodes between logging (default: 10).
    """

    salience_tagged_replay: bool = False
    alpha_consensus: float = 0.5
    alpha_novelty: float = 0.5
    decay_rate: float = 0.99
    replay_interval: int = 10
    num_replay_samples: int = 50
    homeostatic_downscaling: bool = True
    homeostatic_factor: float = 0.99
    enable_logging: bool = True
    log_interval: int = 10

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.alpha_consensus <= 1.0:
            raise ValueError(f"alpha_consensus must be in [0, 1], got {self.alpha_consensus}")

        if not 0.0 <= self.alpha_novelty <= 1.0:
            raise ValueError(f"alpha_novelty must be in [0, 1], got {self.alpha_novelty}")

        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0, 1], got {self.decay_rate}")

        if self.replay_interval < 0:
            raise ValueError(f"replay_interval must be non-negative, got {self.replay_interval}")

        if self.num_replay_samples <= 0:
            raise ValueError(f"num_replay_samples must be positive, got {self.num_replay_samples}")

        if not 0.0 < self.homeostatic_factor <= 1.0:
            raise ValueError(f"homeostatic_factor must be in (0, 1], got {self.homeostatic_factor}")

        if self.log_interval < 0:
            raise ValueError(f"log_interval must be non-negative, got {self.log_interval}")
