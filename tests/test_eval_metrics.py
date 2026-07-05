import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_metrics import temporal_iou, evaluate


def test_iou_identical_ranges_is_one():
    assert abs(temporal_iou((1.0, 5.0), (1.0, 5.0)) - 1.0) < 1e-9


def test_iou_disjoint_ranges_is_zero():
    assert temporal_iou((1.0, 2.0), (5.0, 6.0)) == 0.0


def test_iou_partial_overlap():
    # [1,5] and [3,7]: intersection = [3,5] = 2, union = [1,7] = 6
    iou = temporal_iou((1.0, 5.0), (3.0, 7.0))
    assert abs(iou - (2.0 / 6.0)) < 1e-9


def test_evaluate_perfect_match():
    predicted = [(1.0, 5.0), (10.0, 12.0)]
    ground_truth = [(1.0, 5.0), (10.0, 12.0)]
    metrics = evaluate(predicted, ground_truth, iou_threshold=0.3)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_evaluate_missed_detection_lowers_recall():
    predicted = [(1.0, 5.0)]
    ground_truth = [(1.0, 5.0), (10.0, 12.0)]  # second appearance never detected
    metrics = evaluate(predicted, ground_truth, iou_threshold=0.3)
    assert metrics["recall"] == 0.5
    assert metrics["false_negatives"] == 1


def test_evaluate_false_positive_lowers_precision():
    predicted = [(1.0, 5.0), (50.0, 52.0)]  # second prediction has no matching ground truth
    ground_truth = [(1.0, 5.0)]
    metrics = evaluate(predicted, ground_truth, iou_threshold=0.3)
    assert metrics["precision"] == 0.5
    assert metrics["false_positives"] == 1


def test_evaluate_no_predictions_no_ground_truth():
    metrics = evaluate([], [], iou_threshold=0.3)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


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
