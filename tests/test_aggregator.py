import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.aggregator import FrameMatch, aggregate_matches_to_appearances


def _fm(t, sim=0.8, idx=None):
    return FrameMatch(timestamp_sec=t, similarity=sim, frame_index=idx or int(t * 10), bbox=(0, 0, 10, 10))


def test_empty_input_returns_empty_list():
    assert aggregate_matches_to_appearances([]) == []


def test_single_match_becomes_single_appearance():
    matches = [_fm(1.0)]
    appearances = aggregate_matches_to_appearances(matches, gap_tolerance_sec=1.0, min_duration_sec=0.0)
    assert len(appearances) == 1
    assert appearances[0].start_sec == 1.0
    assert appearances[0].end_sec == 1.0


def test_close_matches_merge_into_one_appearance():
    # Frames at 1.0, 1.5, 2.0 -- gaps of 0.5s, well within 1.0s tolerance.
    matches = [_fm(1.0), _fm(1.5), _fm(2.0)]
    appearances = aggregate_matches_to_appearances(matches, gap_tolerance_sec=1.0, min_duration_sec=0.0)
    assert len(appearances) == 1
    assert appearances[0].start_sec == 1.0
    assert appearances[0].end_sec == 2.0
    assert appearances[0].num_frames == 3


def test_bridges_a_single_missed_frame_gap():
    # Simulates: matched, matched, [missed frame due to motion blur], matched
    # Gap between 2.0 and 3.5 is 1.5s; with a 2.0s tolerance it should merge.
    matches = [_fm(1.0), _fm(2.0), _fm(3.5)]
    appearances = aggregate_matches_to_appearances(matches, gap_tolerance_sec=2.0, min_duration_sec=0.0)
    assert len(appearances) == 1


def test_large_gap_splits_into_separate_appearances():
    matches = [_fm(1.0), _fm(1.5), _fm(20.0), _fm(20.5)]
    appearances = aggregate_matches_to_appearances(matches, gap_tolerance_sec=1.0, min_duration_sec=0.0)
    assert len(appearances) == 2
    assert appearances[0].end_sec == 1.5
    assert appearances[1].start_sec == 20.0


def test_short_appearance_filtered_by_min_duration():
    # A single isolated matched frame -> duration 0 -> filtered out
    # when min_duration_sec > 0 (protects against one-frame false positives).
    matches = [_fm(5.0)]
    appearances = aggregate_matches_to_appearances(matches, gap_tolerance_sec=1.0, min_duration_sec=0.5)
    assert len(appearances) == 0


def test_confidence_scores_computed_correctly():
    matches = [_fm(1.0, sim=0.6), _fm(1.5, sim=0.9), _fm(2.0, sim=0.7)]
    appearances = aggregate_matches_to_appearances(matches, gap_tolerance_sec=1.0, min_duration_sec=0.0)
    assert len(appearances) == 1
    assert appearances[0].max_similarity == 0.9
    assert abs(appearances[0].avg_similarity - (0.6 + 0.9 + 0.7) / 3) < 1e-9


def test_unsorted_input_is_handled_correctly():
    # Function must sort internally -- caller shouldn't need to pre-sort.
    # Gaps here (1.0, 1.5) are both within a 2.0s tolerance, so this should
    # merge into one appearance once correctly sorted.
    matches = [_fm(3.5), _fm(1.0), _fm(2.0)]
    appearances = aggregate_matches_to_appearances(matches, gap_tolerance_sec=2.0, min_duration_sec=0.0)
    assert len(appearances) == 1
    assert appearances[0].start_sec == 1.0
    assert appearances[0].end_sec == 3.5


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {fn.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
