"""
core/embedder.py

Responsibility: turn an ALREADY ALIGNED 112x112 face crop into a fixed-length
identity vector (embedding) using ArcFace.

Why ArcFace: it's trained with an additive angular margin loss, which
explicitly pushes embeddings of the same identity close together in angle
and embeddings of different identities far apart -- regardless of pose,
lighting, or minor image quality issues. That robustness to intra-class
variation is exactly what we need given the reference image may be blurry,
tilted, or low resolution, and the video may show different lighting/angles.

We L2-normalize every embedding. Two reasons:
    1. It matches how ArcFace was trained/evaluated -- comparisons are meant
       to happen on the unit hypersphere.
    2. Once vectors are unit-length, cosine similarity and Euclidean distance
       become monotonically related, so we can use whichever is convenient
       (we use cosine similarity in matcher.py) without losing information.

--------------------------------------------------------------------------
NOTE on the loading strategy (insightface >= 0.7 / current releases):

Earlier insightface versions let you construct `FaceAnalysis(allowed_modules=
["recognition"])` to load ONLY the recognition model. In current releases,
`FaceAnalysis.__init__` unconditionally runs `assert 'detection' in
self.models` and then `self.det_model = self.models['detection']`, even when
`allowed_modules` excludes detection. Passing `allowed_modules=["recognition"]`
now filters the detection model out of `self.models` and then immediately
fails that assertion -- it's a genuine contradiction in the current API, not
a configuration mistake.

`FaceAnalysis` is a convenience wrapper built on top of a lower-level,
model-agnostic loader: `insightface.model_zoo.get_model()`. That function
loads exactly one ONNX file and has no dependency on any other model being
present. This is the same function `FaceAnalysis` calls internally for every
model it loads -- so using it directly here is the intended lower-level API,
not a hack around it.

This also has a nice side benefit for our architecture: the embedder no
longer instantiates a `FaceAnalysis` app at all, so it never loads (or pays
the cost of loading) a detection model. `detector.py` already owns the one
and only detection model in this pipeline -- so this fix also directly
satisfies "avoid loading unnecessary models twice."
--------------------------------------------------------------------------
"""

import glob
import os
from typing import Optional

import numpy as np
from insightface.model_zoo import model_zoo
from insightface.utils import ensure_available


class FaceEmbedder:
    def __init__(self, pack_name: str = "buffalo_l", ctx_id: int = -1, root: str = "~/.insightface"):
        """
        Args:
            pack_name: insightface model pack to pull the recognition model from.
                       Must match the pack used by FaceDetector so that both
                       models were trained/calibrated together.
            ctx_id: -1 for CPU, 0/1/... for GPU device index.
            root: insightface's model cache root (same default FaceAnalysis uses).
                  Kept as a parameter rather than hardcoded so it can be pointed
                  at a custom cache location if needed (e.g. in a container).
        """
        # ensure_available() is the same helper FaceAnalysis uses internally to
        # locate (and, if missing, download) a named model pack's files under
        # `root`. Calling it directly here -- instead of instantiating
        # FaceAnalysis -- gets us the model directory without triggering
        # FaceAnalysis's detection requirement.
        model_dir = ensure_available("models", pack_name, root=root)

        onnx_files = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx files found for model pack '{pack_name}' in {model_dir}. "
                "Check that the pack downloaded/extracted correctly."
            )

        rec_model = None
        for onnx_file in onnx_files:
            # model_zoo.get_model() inspects a single ONNX file's input/output
            # shapes to decide what kind of model it is (detection, recognition,
            # landmark, attribute, etc.) and returns the matching wrapper class.
            # This is the exact same routing FaceAnalysis relies on -- we're
            # just doing it for one file instead of a whole pack.
            candidate = model_zoo.get_model(onnx_file)
            if candidate is None:
                continue
            if getattr(candidate, "taskname", None) == "recognition":
                rec_model = candidate
                break  # found it -- no need to load/inspect the remaining files
            del candidate  # not recognition (e.g. detection/landmark/attribute) -- discard

        if rec_model is None:
            raise RuntimeError(
                f"No recognition (ArcFace) model found among the ONNX files in {model_dir}. "
                f"Files checked: {[os.path.basename(f) for f in onnx_files]}"
            )

        rec_model.prepare(ctx_id)
        self._rec_model = rec_model

    def embed_aligned(self, aligned_face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            aligned_face_bgr: output of FaceAligner.align() -- a 112x112x3 BGR crop.

        Returns:
            L2-normalized embedding vector (typically 512-d), or None if the
            input is invalid.
        """
        if aligned_face_bgr is None or aligned_face_bgr.size == 0:
            return None

        embedding = self._rec_model.get_feat(aligned_face_bgr)
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm
