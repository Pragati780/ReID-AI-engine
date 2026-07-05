"""
core/aligner.py

Responsibility: warp a detected face into the canonical pose ArcFace expects
(112x112, eyes roughly horizontal and at fixed coordinates) using the 5-point
landmarks from the detector.

Why this matters: ArcFace was trained exclusively on images preprocessed this
exact way. If you skip alignment and just crop the bounding box, the network
sees faces at inconsistent rotation/scale/framing, and embedding quality
drops noticeably -- even though the crop "looks fine" to a human eye.
This is the single most under-appreciated step in a face-recognition pipeline,
and it's the reason we insist on a landmark-producing detector (detector.py).
"""

import numpy as np
from insightface.utils import face_align


class FaceAligner:
    def __init__(self, image_size: int = 112):
        """
        Args:
            image_size: output crop size. 112 is the standard ArcFace input size.
        """
        self.image_size = image_size

    def align(self, image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: the full original image the face was detected in.
            kps: (5, 2) landmark array from DetectedFace.kps.

        Returns:
            aligned face crop, shape (image_size, image_size, 3), BGR.
        """
        kps = np.asarray(kps, dtype=np.float32)
        aligned = face_align.norm_crop(image_bgr, landmark=kps, image_size=self.image_size)
        return aligned
