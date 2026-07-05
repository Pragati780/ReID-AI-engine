"""
core/quality.py

Responsibility: flag low-quality face crops (too small, too blurry) BEFORE
they get embedded and silently produce an unreliable similarity score.

This module doesn't try to FIX bad images (e.g. no super-resolution --
see project README for why that's deliberately out of scope for the MVP).
It only measures and reports quality, so the rest of the pipeline can make
an informed decision: warn the user, lower confidence, or skip the face.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityReport:
    is_too_small: bool
    is_blurry: bool
    sharpness_score: float   # variance of Laplacian; higher = sharper
    width: int
    height: int

    @property
    def is_low_quality(self) -> bool:
        return self.is_too_small or self.is_blurry


def assess_face_quality(
    face_crop_bgr: np.ndarray,
    min_face_size_px: int = 40,
    blur_variance_threshold: float = 30.0,
) -> QualityReport:
    """
    Args:
        face_crop_bgr: a cropped face region (before or after alignment --
                        works either way, but be consistent within a run).
        min_face_size_px: minimum acceptable width/height in pixels.
        blur_variance_threshold: minimum acceptable variance of the Laplacian.
                                  Lower variance = blurrier image. This threshold
                                  is a reasonable starting point; recalibrate it
                                  against a few of your own sharp vs. blurry
                                  examples if results seem off.

    Returns:
        QualityReport describing whether this face is usable.
    """
    h, w = face_crop_bgr.shape[:2]
    is_too_small = (w < min_face_size_px) or (h < min_face_size_px)

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = sharpness_score < blur_variance_threshold

    return QualityReport(
        is_too_small=is_too_small,
        is_blurry=is_blurry,
        sharpness_score=sharpness_score,
        width=w,
        height=h,
    )
