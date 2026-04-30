"""Learning modules for CGAL framework."""

from .novelty_detection import NoveltyDetector
from .consensus_gating import ConsensusGatingModule
from .trust_weights import TrustWeightsModule
from .salience_replay import SalienceReplayModule, Pattern

__all__ = [
    "NoveltyDetector",
    "ConsensusGatingModule",
    "TrustWeightsModule",
    "SalienceReplayModule",
    "Pattern",
]
