import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.matcher import cosine_similarity, FaceMatcher


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b) - 0.0) < 1e-6


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_cosine_similarity_ignores_magnitude():
    a = np.array([1.0, 1.0])
    b = np.array([5.0, 5.0])  # same direction, different magnitude
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6


def test_cosine_similarity_zero_vector_handled_safely():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert cosine_similarity(a, b) == 0.0


def test_matcher_flags_match_above_threshold():
    matcher = FaceMatcher(threshold=0.5, low_confidence_margin=0.05)
    ref = np.array([1.0, 0.0])
    same_person = np.array([0.9, 0.1])  # close direction -> high similarity
    result = matcher.compare(ref, same_person)
    assert result.is_match is True


def test_matcher_flags_non_match_below_threshold():
    matcher = FaceMatcher(threshold=0.5, low_confidence_margin=0.05)
    ref = np.array([1.0, 0.0])
    different_person = np.array([0.0, 1.0])  # orthogonal -> similarity ~0
    result = matcher.compare(ref, different_person)
    assert result.is_match is False


def test_matcher_borderline_detection():
    matcher = FaceMatcher(threshold=0.5, low_confidence_margin=0.05)
    ref = np.array([1.0, 0.0])
    # Construct a vector whose cosine similarity to ref is exactly ~0.5
    borderline = np.array([0.5, np.sqrt(1 - 0.5 ** 2)])
    result = matcher.compare(ref, borderline)
    assert result.is_borderline is True


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
