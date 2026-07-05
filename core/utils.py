"""
core/utils.py

Responsibility: I/O and visualization helpers that don't belong to any single
ML concept -- video frame sampling, saving face crops/frames, drawing
debugging annotations, and writing structured results.

IMPORTANT: this module intentionally contains NO matching/decision logic.
Every function here only acts on decisions that were already made upstream
(FaceMatcher.compare() decided is_match / is_borderline; the pipeline decided
which faces belong to which frame). Keeping that boundary is what lets
core/detector.py, aligner.py, embedder.py, matcher.py, aggregator.py stay
pure and independently testable, and keeps this file a dumb I/O layer that's
safe to swap out later (e.g. writing to S3 instead of local disk).
"""

import csv
import json
import os
from typing import Dict, Generator, List, Tuple

import cv2
import numpy as np


def sample_frames(
    video_path: str,
    target_fps: float = 2.0,
    resize_max_width: int = 1280,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Yields (frame_index, timestamp_sec, frame_bgr) at approximately `target_fps`,
    regardless of the video's native frame rate.

    We deliberately do NOT process every single frame:
      - Faces don't appear/disappear inside a fraction of a second, so dense
        sampling mostly adds redundant compute, not accuracy.
      - Lower fps keeps the MVP fast enough to iterate on.

    Args:
        video_path: path to a video file readable by OpenCV.
        target_fps: how many frames per second to actually analyze.
        resize_max_width: if a frame is wider than this, downscale it
                           (keeping aspect ratio) before returning, since
                           very large frames slow down detection for no
                           accuracy benefit at typical face sizes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(native_fps / target_fps))

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / native_fps

                h, w = frame.shape[:2]
                if w > resize_max_width:
                    scale = resize_max_width / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                yield frame_index, timestamp_sec, frame

            frame_index += 1
    finally:
        cap.release()


# -----------------------------------------------------------------------
# Shared filename convention
# -----------------------------------------------------------------------
# All three visual output folders (matched_faces/, matched_frames/,
# annotated_frames/) use the SAME naming pattern for a given event, so a
# reviewer can find the crop, the raw frame, and the annotated frame for the
# same match just by matching filenames across folders.
def _build_output_filename(frame_index: int, timestamp_sec: float, similarity: float) -> str:
    return f"frame_{frame_index:06d}_t{timestamp_sec:.2f}s_sim{similarity:.3f}.jpg"


def save_matched_face_crop(
    crop_bgr: np.ndarray,
    output_dir: str,
    frame_index: int,
    timestamp_sec: float,
    similarity: float,
) -> str:
    """
    Saves a single matched face crop to matched_faces/.

    NOTE: this saves whatever crop the caller passes in. In the current
    pipeline that is the pre-alignment bounding-box crop (not the aligned
    112x112 image used for embedding) -- preserved exactly as-is per the
    "keep unchanged" requirement. See run_pipeline.py for the call site if
    you want the aligned crop saved here instead.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = _build_output_filename(frame_index, timestamp_sec, similarity)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, crop_bgr)
    return path


def save_matched_frame(
    frame_bgr: np.ndarray,
    output_dir: str,
    frame_index: int,
    timestamp_sec: float,
    similarity: float,
) -> str:
    """
    Saves the ORIGINAL, unmodified video frame whenever a match was found
    in it. No drawing happens here -- this is the raw frame exactly as
    sampled from the video, for cases where you want to see full context
    around a match without any overlay.

    Callers are responsible for only calling this once per frame (see
    run_pipeline.py's de-duplication) -- this function does not de-duplicate
    on its own, since it has no concept of "frame identity" beyond the
    filename it's given.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = _build_output_filename(frame_index, timestamp_sec, similarity)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, frame_bgr)
    return path


def draw_match_annotations(frame_bgr: np.ndarray, matches: List[Dict]) -> np.ndarray:
    """
    Draws a bounding box + similarity/timestamp label for every match found
    in a single frame, and returns a NEW annotated copy. The original frame
    is never modified in place, since matched_frames/ needs the untouched
    version saved separately from this annotated one.

    This function performs NO matching or thresholding logic -- every
    decision it visualizes (is this a match? is it borderline?) was already
    made by FaceMatcher upstream. It only reads the `is_borderline` flag to
    pick a color; it never recomputes it.

    Args:
        frame_bgr: the original video frame to annotate.
        matches: list of dicts, one per matched face in this frame, each with:
            "bbox": (x1, y1, x2, y2)
            "similarity": float
            "timestamp_sec": float
            "is_borderline": bool -- decided by FaceMatcher.compare()

    Returns:
        Annotated copy of the frame (BGR), safe to save independently of
        the original.
    """
    annotated = frame_bgr.copy()

    GREEN = (0, 255, 0)      # BGR -- normal (confident) match
    YELLOW = (0, 255, 255)   # BGR -- borderline match, near the threshold

    for m in matches:
        x1, y1, x2, y2 = m["bbox"]
        color = YELLOW if m["is_borderline"] else GREEN

        # Bounding box around the matched face.
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Label: similarity score + timestamp, drawn just above the box.
        label = f"sim={m['similarity']:.3f}  t={m['timestamp_sec']:.2f}s"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(text_h + 4, y1 - 8)

        # Filled background behind the text so it stays legible regardless
        # of what's behind it in the frame.
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - 4),
            (x1 + text_w + 4, label_y + 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # black text reads clearly on both green and yellow
            1,
            cv2.LINE_AA,
        )

    return annotated


def save_annotated_frame(
    annotated_frame_bgr: np.ndarray,
    output_dir: str,
    frame_index: int,
    timestamp_sec: float,
    similarity: float,
) -> str:
    """Saves an already-annotated frame (output of draw_match_annotations) to
    annotated_frames/. Kept as a separate function from save_matched_frame
    so the two folders can be toggled independently via config."""
    os.makedirs(output_dir, exist_ok=True)
    filename = _build_output_filename(frame_index, timestamp_sec, similarity)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, annotated_frame_bgr)
    return path


def save_results_json(appearances: List, reference_quality: dict, output_path: str) -> None:
    payload = {
        "reference_quality": reference_quality,
        "person_found": len(appearances) > 0,
        "num_appearances": len(appearances),
        "appearances": [a.to_dict() for a in appearances],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_results_csv(appearances: List, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["start_sec", "end_sec", "duration_sec", "max_similarity", "avg_similarity", "num_frames"]
        )
        writer.writeheader()
        for a in appearances:
            writer.writerow(a.to_dict())


def save_embeddings_npy(embeddings: List[np.ndarray], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if len(embeddings) == 0:
        np.save(output_path, np.array([]))
        return
    np.save(output_path, np.vstack(embeddings))
