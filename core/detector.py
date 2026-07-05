"""
core/detector.py

Responsibility: given a raw BGR image (OpenCV format), find every face in it
and return its bounding box, 5-point landmarks, and detector confidence.

We deliberately do NOT use a general object detector (e.g. YOLO) here.
Face embedding models (ArcFace) are trained on faces that were aligned using
precise 5-point landmarks (eyes, nose, mouth corners). A general detector
either lacks these landmarks or produces noisier ones, which silently
degrades every downstream embedding -- this is one of the most common,
hardest-to-spot bugs in face recognition pipelines.

We use insightface's bundled detector (SCRFD, the direct successor to
RetinaFace and part of the same architectural family: single-stage,
anchor-based, multi-scale, trained jointly with landmark regression).
It ships as part of the "buffalo_l" model pack alongside the ArcFace
recognition model, so detector and embedder are guaranteed to be
compatible with each other.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class DetectedFace:
    """A single detected face, in the coordinate space of the input image."""
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    kps: np.ndarray                           # shape (5, 2) -- 5 landmark points
    det_score: float                          # detector confidence, 0..1

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class FaceDetector:
    """Thin, purpose-specific wrapper around insightface's detection module."""

    def __init__(
        self,
        pack_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
    ):
        """
        Args:
            pack_name: insightface model pack to pull the detector from.
            ctx_id: -1 for CPU, 0/1/... for GPU device index.
            det_size: input resolution the detector resizes to internally.
            det_thresh: minimum confidence to keep a detected face.
        """
        self.det_thresh = det_thresh
        # allowed_modules restricts loading to just the detection model,
        # so we don't pay the (larger) cost of loading recognition weights
        # twice -- FaceEmbedder loads recognition separately.
        self.app = FaceAnalysis(name=pack_name, allowed_modules=["detection"])
        self.app.prepare(ctx_id=ctx_id, det_size=tuple(det_size), det_thresh=det_thresh)

    def detect(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Args:
            image_bgr: image as loaded by cv2.imread / cv2.VideoCapture (BGR, HxWx3).

        Returns:
            List of DetectedFace, one per face found. Empty list if none found.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        raw_faces = self.app.get(image_bgr)
        results = []
        for f in raw_faces:
            if f.det_score < self.det_thresh:
                continue
            results.append(
                DetectedFace(
                    bbox=tuple(f.bbox.tolist()),
                    kps=np.array(f.kps, dtype=np.float32),
                    det_score=float(f.det_score),
                )
            )
        return results
