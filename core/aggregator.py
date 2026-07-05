"""
core/aggregator.py

Responsibility: turn a list of independent, per-frame match decisions into
clean, human-readable timestamp RANGES -- and remove noise along the way.

Two real-world problems this solves:

1. FRAGMENTATION: a person doesn't vanish and reappear just because one
   frame was motion-blurred or had an unlucky detection failure. Without a
   "gap tolerance", a single missed frame in the middle of a 10-second
   appearance would incorrectly produce three separate short appearances
   instead of one continuous one.

2. FALSE-POSITIVE BLIPS: a single isolated matched frame (e.g. a
   look-alike stranger for one frame) is much less trustworthy than a
   sustained multi-second match. We filter out appearances shorter than a
   minimum duration.

This module is pure Python/dataclasses -- no model dependencies -- so it's
independently unit-testable (see tests/test_aggregator.py) without needing
insightface installed at all.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FrameMatch:
    """A single frame's match result for one detected face."""
    timestamp_sec: float
    similarity: float
    frame_index: int
    bbox: tuple


@dataclass
class Appearance:
    """A merged, continuous appearance of the target person."""
    start_sec: float
    end_sec: float
    max_similarity: float
    avg_similarity: float
    num_frames: int
    frame_matches: List[FrameMatch] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "max_similarity": round(self.max_similarity, 4),
            "avg_similarity": round(self.avg_similarity, 4),
            "num_frames": self.num_frames,
        }


def aggregate_matches_to_appearances(
    matches: List[FrameMatch],
    gap_tolerance_sec: float = 1.0,
    min_duration_sec: float = 0.5,
) -> List[Appearance]:
    """
    Args:
        matches: list of FrameMatch for frames where similarity >= threshold.
                 Does NOT need to be pre-sorted.
        gap_tolerance_sec: merge two matched frames into the same appearance
                            if the time gap between them is <= this value.
        min_duration_sec: drop any resulting appearance shorter than this
                           (likely a false positive / one-off blip).

    Returns:
        List of Appearance, sorted chronologically, each representing one
        continuous stretch of time the person appears to be present.
    """
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda m: m.timestamp_sec)

    appearances: List[Appearance] = []
    current_group: List[FrameMatch] = [sorted_matches[0]]

    for m in sorted_matches[1:]:
        gap = m.timestamp_sec - current_group[-1].timestamp_sec
        if gap <= gap_tolerance_sec:
            current_group.append(m)
        else:
            appearances.append(_group_to_appearance(current_group))
            current_group = [m]
    appearances.append(_group_to_appearance(current_group))

    # Filter out too-short / likely-spurious appearances.
    filtered = [a for a in appearances if a.duration_sec >= min_duration_sec]

    return filtered


def _group_to_appearance(group: List[FrameMatch]) -> Appearance:
    sims = [m.similarity for m in group]
    return Appearance(
        start_sec=group[0].timestamp_sec,
        end_sec=group[-1].timestamp_sec,
        max_similarity=max(sims),
        avg_similarity=sum(sims) / len(sims),
        num_frames=len(group),
        frame_matches=group,
    )
