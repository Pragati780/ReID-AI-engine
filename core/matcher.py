"""
core/matcher.py

Responsibility: compare a reference embedding against candidate embeddings
using cosine similarity, and decide which candidates count as a "match".

Why cosine similarity specifically: ArcFace is trained with an angular
margin loss, meaning the network directly optimizes the ANGLE between
same-identity embeddings to be small and between different-identity
embeddings to be large. Cosine similarity measures exactly that angle
(via its cosine), ignoring vector magnitude -- which is what you want,
since magnitude can vary with image quality/lighting in ways that aren't
identity-relevant. This is why cosine similarity, rather than raw
Euclidean distance or dot product, is the standard metric for ArcFace-style
embeddings.

This module has no dependency on insightface at all -- it only operates on
numpy arrays -- which is exactly what lets you swap it for a FAISS/pgvector
lookup later without touching detector/aligner/embedder code.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors, in [-1, 1].
    If inputs are already L2-normalized (as our embedder guarantees),
    this is equivalent to a simple dot product -- we still normalize
    defensively here in case a caller passes in a raw vector.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class MatchResult:
    similarity: float
    is_match: bool
    is_borderline: bool  # near the threshold -- worth a human glance


class FaceMatcher:
    def __init__(self, threshold: float = 0.38, low_confidence_margin: float = 0.05):
        """
        Args:
            threshold: cosine similarity cutoff above which we call it a match.
                       This MUST be calibrated on your own data (see
                       eval/eval_metrics.py) rather than trusted blindly --
                       published benchmark thresholds are a starting point,
                       not a guarantee, especially with an imperfect
                       reference image.
            low_confidence_margin: scores within `threshold +/- margin` are
                       flagged as "borderline" so they can be surfaced for
                       manual review instead of silently accepted/rejected.
        """
        self.threshold = threshold
        self.low_confidence_margin = low_confidence_margin

    def compare(self, reference_embedding: np.ndarray, candidate_embedding: np.ndarray) -> MatchResult:
        sim = cosine_similarity(reference_embedding, candidate_embedding)
        is_match = sim >= self.threshold
        is_borderline = abs(sim - self.threshold) <= self.low_confidence_margin
        return MatchResult(similarity=sim, is_match=is_match, is_borderline=is_borderline)

    def compare_many(self, reference_embedding: np.ndarray, candidate_embeddings: List[np.ndarray]) -> List[MatchResult]:
        return [self.compare(reference_embedding, c) for c in candidate_embeddings]
