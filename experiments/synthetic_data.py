"""Synthetic data generation for CGAL experiments.

This module creates synthetic learning scenarios to test CGAL mechanisms
without requiring full Monty integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import random


class SyntheticLearningModule:
    """Simulates a learning module with pattern recognition capabilities."""

    def __init__(self, module_id: int, reliability: float = 1.0, noise_level: float = 0.0):
        """Initialize a synthetic learning module.

        Args:
            module_id: Unique identifier for this module.
            reliability: How reliably this module recognizes patterns (0-1).
            noise_level: Amount of noise in observations (0-1).
        """
        self.module_id = module_id
        self.reliability = reliability
        self.noise_level = noise_level
        self.learned_patterns: Dict[str, np.ndarray] = {}
        self.observation_count = 0

    def observe(self, object_id: str, features: np.ndarray) -> Dict[str, Any]:
        """Observe an object and generate a hypothesis.

        Args:
            object_id: Ground truth object ID (not used by module, only for eval).
            features: Feature vector of the object.

        Returns:
            Hypothesis dictionary with object_id, confidence, and features.
        """
        self.observation_count += 1

        # Add noise to features
        if self.noise_level > 0:
            features = features + np.random.normal(0, self.noise_level, features.shape)

        # If we haven't learned any patterns yet, return low confidence
        if not self.learned_patterns:
            return {
                'object_id': object_id if random.random() < self.reliability else 'unknown',
                'confidence': 0.3,
                'features': features
            }

        # Find best matching pattern
        best_match = None
        best_distance = float('inf')

        for pattern_id, pattern_features in self.learned_patterns.items():
            distance = np.linalg.norm(features - pattern_features)
            if distance < best_distance:
                best_distance = distance
                best_match = pattern_id

        # Determine if match is correct based on reliability
        correct = random.random() < self.reliability
        predicted_id = best_match if correct else random.choice(list(self.learned_patterns.keys()))

        # Confidence based on distance (closer = higher confidence)
        confidence = np.exp(-best_distance)

        return {
            'object_id': predicted_id,
            'confidence': confidence,
            'features': features
        }

    def learn(self, object_id: str, features: np.ndarray, learning_rate: float = 1.0):
        """Learn or update a pattern.

        Args:
            object_id: Object ID to learn.
            features: Feature vector to associate with this object.
            learning_rate: How much to update (0-1).
        """
        if object_id in self.learned_patterns:
            # Update existing pattern
            self.learned_patterns[object_id] = (
                (1 - learning_rate) * self.learned_patterns[object_id] +
                learning_rate * features
            )
        else:
            # Learn new pattern
            self.learned_patterns[object_id] = features.copy()


class SyntheticObjectDataset:
    """Generates synthetic objects with feature vectors."""

    def __init__(self, num_objects: int = 10, feature_dim: int = 50, seed: int = 42):
        """Initialize dataset.

        Args:
            num_objects: Number of distinct objects.
            feature_dim: Dimensionality of feature vectors.
            seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.num_objects = num_objects
        self.feature_dim = feature_dim

        # Generate unique feature vectors for each object
        self.object_features = {}
        for i in range(num_objects):
            object_id = f"object_{i:02d}"
            # Create a distinctive feature vector
            features = np.random.randn(feature_dim)
            features /= np.linalg.norm(features)  # Normalize
            self.object_features[object_id] = features

        self.object_ids = list(self.object_features.keys())

    def get_observation(self, object_id: str, noise: float = 0.0) -> np.ndarray:
        """Get an observation of an object.

        Args:
            object_id: Which object to observe.
            noise: Amount of Gaussian noise to add.

        Returns:
            Feature vector (possibly with noise).
        """
        if object_id not in self.object_features:
            raise ValueError(f"Unknown object: {object_id}")

        features = self.object_features[object_id].copy()
        if noise > 0:
            features += np.random.normal(0, noise, features.shape)
            features /= np.linalg.norm(features)  # Re-normalize

        return features

    def get_random_object(self) -> str:
        """Get a random object ID."""
        return random.choice(self.object_ids)

    def get_all_objects(self) -> List[str]:
        """Get all object IDs."""
        return self.object_ids.copy()


def create_learning_network(
    num_modules: int = 5,
    reliability_range: Tuple[float, float] = (0.8, 1.0),
    noise_levels: List[float] = None
) -> List[SyntheticLearningModule]:
    """Create a network of learning modules.

    Args:
        num_modules: Number of modules in the network.
        reliability_range: (min, max) reliability for modules.
        noise_levels: Optional list of noise levels for each module.
                     If None, all modules have zero noise.

    Returns:
        List of SyntheticLearningModule instances.
    """
    modules = []

    if noise_levels is None:
        noise_levels = [0.0] * num_modules
    elif len(noise_levels) != num_modules:
        raise ValueError(f"noise_levels must have length {num_modules}")

    for i in range(num_modules):
        reliability = random.uniform(*reliability_range)
        module = SyntheticLearningModule(
            module_id=i,
            reliability=reliability,
            noise_level=noise_levels[i]
        )
        modules.append(module)

    return modules


def voting_consensus(hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute consensus from module hypotheses via voting.

    Args:
        hypotheses: List of hypothesis dicts from modules.

    Returns:
        Consensus hypothesis (most common object_id).
    """
    if not hypotheses:
        return {'object_id': 'unknown', 'confidence': 0.0}

    # Count votes
    votes = {}
    for hyp in hypotheses:
        obj_id = hyp['object_id']
        votes[obj_id] = votes.get(obj_id, 0) + 1

    # Find winner
    consensus_id = max(votes, key=votes.get)
    consensus_confidence = votes[consensus_id] / len(hypotheses)

    return {
        'object_id': consensus_id,
        'confidence': consensus_confidence
    }
