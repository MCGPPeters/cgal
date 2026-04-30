"""Configuration management for CGAL."""

from .novelty_config import NoveltyDetectionConfig
from .consensus_gating_config import ConsensusGatingConfig
from .trust_weights_config import TrustWeightsConfig
from .salience_replay_config import SalienceReplayConfig

__all__ = [
    "NoveltyDetectionConfig",
    "ConsensusGatingConfig",
    "TrustWeightsConfig",
    "SalienceReplayConfig",
]
