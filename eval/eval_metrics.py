"""
eval/eval_metrics.py

Responsibility: measure whether the pipeline's predicted appearances actually
match ground truth, using metrics borrowed from temporal action localization:

    - Precision: of predicted appearances, how many correspond to a real one?
    - Recall: of real appearances, how many did we catch?
    - Temporal IoU: how well does a predicted range overlap the true range?

You build ground truth by watching a short test video once and manually
writing down the timestamp ranges the target person appears
(see eval/ground_truth/example_ground_truth.json for the expected format).

This is intentionally simple (greedy matching, not the Hungarian algorithm)
-- appropriate for an MVP with a handful of appearances per test video.
"""

import argparse
import json
from typing import List, Tuple


def temporal_iou(range_a: Tuple[float, float], range_b: Tuple[float, float]) -> float:
    """Intersection-over-union of two [start, end] time ranges."""
    start_a, end_a = range_a
    start_b, end_b = range_b

    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(start_a, start_b)
    union_end = max(end_a, end_b)
    union = union_end - union_start

    if union == 0:
        return 0.0
    return intersection / union


def evaluate(
    predicted_ranges: List[Tuple[float, float]],
    ground_truth_ranges: List[Tuple[float, float]],
    iou_threshold: float = 0.3,
) -> dict:
    """
    Greedily matches each predicted range to the best-overlapping,
    not-yet-used ground truth range. A predicted range counts as a
    true positive if its best IoU with any unused ground truth range
    exceeds iou_threshold.

    Args:
        predicted_ranges: list of (start_sec, end_sec) from result.json.
        ground_truth_ranges: list of (start_sec, end_sec), manually labeled.
        iou_threshold: minimum overlap to count as a correct detection.
                       0.3 is a reasonable starting point for an MVP --
                       tighten it if you need more precise localization.

    Returns:
        dict with precision, recall, f1, mean_iou_of_matches, and raw counts.
    """
    matched_gt_indices = set()
    true_positives = 0
    ious_of_matches = []

    for pred in predicted_ranges:
        best_iou = 0.0
        best_gt_idx = None
        for i, gt in enumerate(ground_truth_ranges):
            if i in matched_gt_indices:
                continue
            iou = temporal_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_gt_idx is not None and best_iou >= iou_threshold:
            true_positives += 1
            matched_gt_indices.add(best_gt_idx)
            ious_of_matches.append(best_iou)

    false_positives = len(predicted_ranges) - true_positives
    false_negatives = len(ground_truth_ranges) - len(matched_gt_indices)

    precision = true_positives / len(predicted_ranges) if predicted_ranges else 0.0
    recall = true_positives / len(ground_truth_ranges) if ground_truth_ranges else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = sum(ious_of_matches) / len(ious_of_matches) if ious_of_matches else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou_of_matches": round(mean_iou, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "num_predicted": len(predicted_ranges),
        "num_ground_truth": len(ground_truth_ranges),
    }


def _load_ranges_from_result_json(path: str) -> List[Tuple[float, float]]:
    with open(path) as f:
        data = json.load(f)
    return [(a["start_sec"], a["end_sec"]) for a in data["appearances"]]


def _load_ranges_from_ground_truth_json(path: str) -> List[Tuple[float, float]]:
    with open(path) as f:
        data = json.load(f)
    return [(r["start_sec"], r["end_sec"]) for r in data["appearances"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted appearances against ground truth")
    parser.add_argument("--predicted", required=True, help="Path to result.json from a pipeline run")
    parser.add_argument("--ground-truth", required=True, help="Path to a ground truth JSON file")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()

    predicted = _load_ranges_from_result_json(args.predicted)
    ground_truth = _load_ranges_from_ground_truth_json(args.ground_truth)

    metrics = evaluate(predicted, ground_truth, iou_threshold=args.iou_threshold)
    print(json.dumps(metrics, indent=2))
