"""
pipeline/run_pipeline.py

Orchestrates the full flow:

    reference image ---> detect -> align -> embed ---> reference_embedding
    video frames     ---> detect -> align -> embed ---> per-frame candidate embeddings
                                                                |
                                                                v
                                              cosine similarity vs reference
                                                                |
                                                                v
                                          threshold -> gap-tolerant merge -> Appearances
                                                                |
                                                                v
                                          JSON + CSV + matched_faces/ + matched_frames/
                                          + annotated_frames/ + embeddings + logs on disk

This script is intentionally a thin orchestrator: all real logic lives in
core/*.py, which is what keeps each piece independently testable and lets
this script eventually be replaced by a FastAPI endpoint without touching
the underlying logic.

Usage:
    python pipeline/run_pipeline.py \\
        --reference data/reference_images/person.jpg \\
        --video data/videos/clip.mp4 \\
        --config config/config.yaml \\
        --run-id demo_run_01
"""

import argparse
import logging
import os
import sys
import time

import cv2
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import FaceDetector
from core.aligner import FaceAligner
from core.embedder import FaceEmbedder
from core.quality import assess_face_quality
from core.matcher import FaceMatcher
from core.aggregator import FrameMatch, aggregate_matches_to_appearances
from core.utils import (
    sample_frames,
    save_matched_face_crop,
    save_matched_frame,
    draw_match_annotations,
    save_annotated_frame,
    save_results_json,
    save_results_csv,
    save_embeddings_npy,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("face_reid_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def process_reference_image(reference_path, detector, aligner, embedder, cfg, logger):
    """Detects, aligns, embeds the reference face and reports its quality.
    Uses the LARGEST detected face, on the assumption the reference image
    is primarily of the target person (a reasonable MVP assumption; flagged
    in logs if multiple faces are found so the user can verify)."""
    image = cv2.imread(reference_path)
    if image is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_path}")

    faces = detector.detect(image)
    if not faces:
        raise ValueError(
            "No face detected in the reference image. Try a clearer image, "
            "or one where the face is larger / less occluded."
        )

    if len(faces) > 1:
        logger.warning(f"{len(faces)} faces found in reference image; using the largest one.")

    target_face = max(faces, key=lambda f: f.width * f.height)

    quality = assess_face_quality(
        image[int(target_face.bbox[1]):int(target_face.bbox[3]), int(target_face.bbox[0]):int(target_face.bbox[2])],
        min_face_size_px=cfg["quality"]["min_face_size_px"],
        blur_variance_threshold=cfg["quality"]["blur_variance_threshold"],
    )

    if quality.is_low_quality and cfg["quality"]["warn_on_low_quality_reference"]:
        logger.warning(
            f"Reference image quality is LOW (sharpness={quality.sharpness_score:.1f}, "
            f"size={quality.width}x{quality.height}). Similarity scores may be less reliable; "
            "treat borderline matches with extra caution."
        )

    aligned = aligner.align(image, target_face.kps)
    embedding = embedder.embed_aligned(aligned)
    if embedding is None:
        raise ValueError("Failed to generate an embedding from the reference image.")

    return embedding, quality


def process_video(video_path, reference_embedding, detector, aligner, embedder, matcher, cfg, run_output_dir, logger):
    """
    Walks through sampled video frames, detects every face in each frame,
    embeds it, and compares it against the reference embedding.

    Produces three kinds of visual debugging output, all gated behind
    "only save when matched" and written at most once per frame:

        matched_faces/     one crop per matched FACE (same crop behavior as
                            before this feature was added -- unchanged).
        matched_frames/     the original, unmodified frame, once per frame
                            that contained at least one match.
        annotated_frames/  the same original frame, once per frame, with
                            every matched face's box + score + timestamp
                            drawn on it (green = normal match, yellow =
                            borderline match).

    All matching/thresholding decisions still happen in FaceMatcher
    (core/matcher.py) -- this function only reads the MatchResult it
    returns and decides what to save and when. Drawing itself happens in
    core/utils.py, which never re-derives is_match/is_borderline; it only
    visualizes the flags this function passes it.

    Returns:
        frame_matches: list[FrameMatch] for every face whose similarity
                        cleared the matcher's threshold (these feed into
                        aggregator.py to become clean timestamp ranges).
        all_embeddings: every embedding generated during the run (useful
                        for debugging, and for the future multi-video /
                        embedding-database extension).
    """
    frame_matches = []
    all_embeddings = []
    faces_seen = 0

    matched_faces_dir = os.path.join(run_output_dir, "matched_faces")
    matched_frames_dir = os.path.join(run_output_dir, "matched_frames")
    annotated_frames_dir = os.path.join(run_output_dir, "annotated_frames")

    # Defensive de-duplication for the two FRAME-level outputs. Our sampling
    # loop naturally visits each frame_index once already, but we track this
    # explicitly so a frame can never be written twice to matched_frames/ or
    # annotated_frames/, regardless of how many faces matched inside it.
    saved_frame_output_ids = set()

    start_time = time.time()
    for frame_index, timestamp_sec, frame in sample_frames(
        video_path,
        target_fps=cfg["video"]["frame_sample_fps"],
        resize_max_width=cfg["video"]["resize_max_width"],
    ):
        faces = detector.detect(frame)

        # Collect every match found in THIS frame before writing any
        # frame-level output, so matched_frames/ and annotated_frames/ are
        # each written exactly once per frame -- with every matched face's
        # box drawn together on a single annotated image, not one image per
        # face.
        matches_in_this_frame = []

        for face in faces:
            faces_seen += 1
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            quality = assess_face_quality(
                crop,
                min_face_size_px=cfg["quality"]["min_face_size_px"],
                blur_variance_threshold=cfg["quality"]["blur_variance_threshold"],
            )
            if quality.is_too_small:
                # Too small to trust an embedding from -- skip rather than
                # inject noise into the results. (We do NOT skip on blur
                # alone here: a slightly blurry frame of the correct person
                # is still useful signal; we only hard-skip on size, and
                # let similarity scores speak for themselves otherwise.)
                continue

            aligned = aligner.align(frame, face.kps)
            embedding = embedder.embed_aligned(aligned)
            if embedding is None:
                continue

            if cfg["output"]["save_all_embeddings"]:
                all_embeddings.append(embedding)

            # --- The only matching decision in this whole function ---
            match_result = matcher.compare(reference_embedding, embedding)

            if not match_result.is_match:
                continue  # requirement: only ever save frames that clear the threshold

            frame_matches.append(
                FrameMatch(
                    timestamp_sec=timestamp_sec,
                    similarity=match_result.similarity,
                    frame_index=frame_index,
                    bbox=(x1, y1, x2, y2),
                )
            )

            if cfg["output"]["save_matched_faces"]:
                save_matched_face_crop(crop, matched_faces_dir, frame_index, timestamp_sec, match_result.similarity)

            # Record this face's box/score/borderline-flag for the
            # frame-level outputs below, once every face in the frame has
            # been checked. We pass the flag through as-is -- no re-deciding
            # it here or in utils.py.
            matches_in_this_frame.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "similarity": match_result.similarity,
                    "timestamp_sec": timestamp_sec,
                    "is_borderline": match_result.is_borderline,
                }
            )

            if match_result.is_borderline:
                logger.info(
                    f"Borderline match at t={timestamp_sec:.2f}s "
                    f"(similarity={match_result.similarity:.4f}, near threshold) -- worth a manual look."
                )

        # --- Frame-level outputs: written once per frame, only if it had >=1 match ---
        if matches_in_this_frame and frame_index not in saved_frame_output_ids:
            saved_frame_output_ids.add(frame_index)

            # Use the strongest similarity found in the frame for the
            # filename, so files can be skimmed/sorted by confidence.
            best_similarity = max(m["similarity"] for m in matches_in_this_frame)

            if cfg["output"]["save_matched_frames"]:
                save_matched_frame(frame, matched_frames_dir, frame_index, timestamp_sec, best_similarity)

            if cfg["output"]["save_annotated_frames"]:
                annotated = draw_match_annotations(frame, matches_in_this_frame)
                save_annotated_frame(annotated, annotated_frames_dir, frame_index, timestamp_sec, best_similarity)

    elapsed = time.time() - start_time
    logger.info(
        f"Video processed in {elapsed:.1f}s | raw face detections: {faces_seen} | "
        f"frames matched above threshold: {len(frame_matches)}"
    )
    return frame_matches, all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Face Re-Identification MVP pipeline")
    parser.add_argument("--reference", required=True, help="Path to the reference face image")
    parser.add_argument("--video", required=True, help="Path to the video to search")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-id", default=None, help="Name for this run's output folder (default: timestamp)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(cfg["output"]["base_output_dir"], run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    logger = setup_logging(os.path.join(run_output_dir, "logs", "pipeline.log"))
    logger.info(f"=== Starting run '{run_id}' ===")
    logger.info(f"Reference image: {args.reference}")
    logger.info(f"Video: {args.video}")
    logger.info(f"Config: {args.config} -> {cfg}")

    # --- Load models once, reuse across reference + every video frame ---
    detector = FaceDetector(
        pack_name=cfg["model"]["pack_name"],
        ctx_id=cfg["model"]["ctx_id"],
        det_size=cfg["model"]["det_size"],
        det_thresh=cfg["model"]["det_thresh"],
    )
    aligner = FaceAligner(image_size=cfg["model"]["embedding_image_size"])
    embedder = FaceEmbedder(pack_name=cfg["model"]["pack_name"], ctx_id=cfg["model"]["ctx_id"])
    matcher = FaceMatcher(
        threshold=cfg["matching"]["similarity_threshold"],
        low_confidence_margin=cfg["matching"]["low_confidence_margin"],
    )

    # --- Step 1: reference embedding ---
    reference_embedding, ref_quality = process_reference_image(
        args.reference, detector, aligner, embedder, cfg, logger
    )
    logger.info(
        f"Reference embedding generated (sharpness={ref_quality.sharpness_score:.1f}, "
        f"size={ref_quality.width}x{ref_quality.height}, low_quality={ref_quality.is_low_quality})"
    )

    # --- Step 2: scan the video ---
    frame_matches, all_embeddings = process_video(
        args.video, reference_embedding, detector, aligner, embedder, matcher, cfg, run_output_dir, logger
    )

    # --- Step 3: aggregate frame matches into clean timestamp ranges ---
    appearances = aggregate_matches_to_appearances(
        frame_matches,
        gap_tolerance_sec=cfg["aggregation"]["gap_tolerance_seconds"],
        min_duration_sec=cfg["aggregation"]["min_appearance_duration_seconds"],
    )
    logger.info(f"Final appearances after aggregation/filtering: {len(appearances)}")
    for a in appearances:
        logger.info(f"  -> {a.start_sec:.2f}s - {a.end_sec:.2f}s (max_sim={a.max_similarity:.4f}, frames={a.num_frames})")

    # --- Step 4: save outputs ---
    save_results_json(
        appearances,
        reference_quality={
            "sharpness_score": ref_quality.sharpness_score,
            "width": ref_quality.width,
            "height": ref_quality.height,
            "is_low_quality": ref_quality.is_low_quality,
        },
        output_path=os.path.join(run_output_dir, "result.json"),
    )
    save_results_csv(appearances, output_path=os.path.join(run_output_dir, "result.csv"))

    if cfg["output"]["save_all_embeddings"]:
        save_embeddings_npy(all_embeddings, os.path.join(run_output_dir, "all_embeddings.npy"))
        save_embeddings_npy([reference_embedding], os.path.join(run_output_dir, "reference_embedding.npy"))

    logger.info(f"=== Run '{run_id}' complete. Outputs in: {run_output_dir} ===")


if __name__ == "__main__":
    main()
