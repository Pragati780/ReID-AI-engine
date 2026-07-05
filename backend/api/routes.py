"""
backend/api/routes.py

All HTTP endpoints for the face-search web app. This file is pure web
plumbing: validating uploads, generating run IDs, calling
services/pipeline_service.py (which treats the AI pipeline as a black box),
and shaping/returning JSON or files. No face-recognition logic lives here.
"""

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from models.schemas import SearchResponse
from services import pipeline_service
from services.pipeline_service import PipelineExecutionError, PipelineTimeoutError

logger = logging.getLogger("face_reid_api")

router = APIRouter()

# --------------------------------------------------------------------------
# Path configuration (backend-local storage -- separate from the pipeline's
# own outputs/ directory, which belongs to the AI project and is only ever
# READ by this backend, never written to directly).
# --------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = BACKEND_ROOT / "uploads"
RESPONSES_DIR = BACKEND_ROOT / "responses"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _tail(text: str, n_lines: int = 15) -> str:
    """Returns the last few lines of a (potentially long) traceback, for a
    client-facing error message that doesn't dump an entire stack trace."""
    lines = text.strip().splitlines()
    return "\n".join(lines[-n_lines:]) if lines else "(no output captured)"


def _save_upload(upload: UploadFile, destination: Path) -> None:
    """Streams an UploadFile to disk without loading it fully into memory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    upload.file.close()


def _build_search_response(run_id: str, processing_time_sec: Optional[float]) -> SearchResponse:
    """
    Assembles the API's response shape purely by READING what the pipeline
    already produced (result.json + the matched_frames/annotated_frames
    folders it already writes) -- no re-computation of anything the AI
    pipeline is responsible for.
    """
    result_json: Dict = pipeline_service.load_result_json(run_id)

    matched_frames = pipeline_service.list_output_images(run_id, "matched_frames")
    annotated_frames = pipeline_service.list_output_images(run_id, "annotated_frames")

    return SearchResponse(
        run_id=run_id,
        processing_time_sec=processing_time_sec,
        person_found=result_json["person_found"],
        num_appearances=result_json["num_appearances"],
        overall_best_similarity=pipeline_service.overall_best_similarity(result_json),
        reference_quality=result_json["reference_quality"],
        appearances=result_json["appearances"],
        matched_frames=matched_frames,
        annotated_frames=annotated_frames,
        reference_image_url=f"/api/reference/{run_id}",
        csv_download_url=f"/api/download/csv/{run_id}",
        json_download_url=f"/api/download/json/{run_id}",
    )


def _cache_response(run_id: str, response: SearchResponse) -> None:
    """Caches the assembled response so GET /api/result/{run_id} can be
    served instantly later without recomputing processing_time (which only
    exists at the moment the pipeline finishes running)."""
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RESPONSES_DIR / f"{run_id}.json"
    with open(cache_path, "w") as f:
        f.write(response.model_dump_json(indent=2))


@router.get("/health")
def health_check():
    """Simple liveness check -- does not touch the AI pipeline at all."""
    return {"status": "ok"}


@router.post("/search", response_model=SearchResponse)
def search(reference: UploadFile = File(...), video: UploadFile = File(...)):
    """
    Accepts a reference image + a video, runs the existing face-recognition
    pipeline against them, and returns the structured result.

    This endpoint is intentionally a normal (synchronous) function rather
    than `async def`: FastAPI runs sync route functions in a worker thread
    pool automatically, so the long-running, CPU-bound pipeline subprocess
    (which can take anywhere from seconds to minutes) never blocks the
    server's event loop or other concurrent requests.
    """
    ref_ext = _ext(reference.filename or "")
    video_ext = _ext(video.filename or "")

    if ref_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported reference image type '{ref_ext}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )
    if video_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{video_ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    run_id = f"web_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    reference_path = UPLOADS_DIR / run_id / f"reference{ref_ext}"
    video_path = UPLOADS_DIR / run_id / f"video{video_ext}"

    try:
        _save_upload(reference, reference_path)
        _save_upload(video, video_path)
    except OSError as exc:
        logger.exception("Failed to save uploaded files for run '%s'", run_id)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {exc}") from exc

    logger.info("Starting face search run '%s' (reference=%s, video=%s)", run_id, reference_path.name, video_path.name)

    try:
        processing_time_sec = pipeline_service.run_face_search(reference_path, video_path, run_id)
    except PipelineTimeoutError as exc:
        logger.error("Run '%s' timed out: %s", run_id, exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except PipelineExecutionError as exc:
        logger.error("Run '%s' failed.\nSTDOUT:\n%s\nSTDERR:\n%s", run_id, exc.stdout, exc.stderr)
        raise HTTPException(
            status_code=422,
            detail=f"Face search failed while processing this reference/video pair: {_tail(exc.stderr)}",
        ) from exc

    try:
        response = _build_search_response(run_id, processing_time_sec)
    except FileNotFoundError as exc:
        logger.exception("Pipeline reported success but outputs are missing for run '%s'", run_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _cache_response(run_id, response)
    logger.info("Run '%s' complete in %.2fs (person_found=%s)", run_id, processing_time_sec, response.person_found)
    return response


@router.get("/result/{run_id}", response_model=SearchResponse)
def get_result(run_id: str):
    """
    Returns the complete result for a previous run. Serves from the cached
    response written at search-time when available (fast path, preserves the
    original processing_time); otherwise rebuilds the response directly from
    result.json (fallback path -- e.g. after a server restart wiped the
    in-memory-adjacent cache file, though the cache itself is on disk so this
    mainly guards against a missing/corrupted cache file).
    """
    cache_path = RESPONSES_DIR / f"{run_id}.json"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="application/json")

    try:
        return _build_search_response(run_id, processing_time_sec=None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/frame/{run_id}/{filename}")
def get_matched_frame(run_id: str, filename: str):
    """Serves one image from outputs/{run_id}/matched_frames/{filename}."""
    try:
        path = pipeline_service.resolve_safe_output_file(run_id, "matched_frames", filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@router.get("/annotated/{run_id}/{filename}")
def get_annotated_frame(run_id: str, filename: str):
    """Serves one image from outputs/{run_id}/annotated_frames/{filename}."""
    try:
        path = pipeline_service.resolve_safe_output_file(run_id, "annotated_frames", filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@router.get("/reference/{run_id}")
def get_reference_image(run_id: str):
    """
    Serves back the reference image the user originally uploaded for this
    run, so the Results page can display it alongside the findings.

    (This isn't AI pipeline output -- it's simply the file the backend
    itself saved under backend/uploads/{run_id}/ at search time.)
    """
    try:
        path = pipeline_service.find_uploaded_reference_image(run_id, UPLOADS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)


@router.get("/download/json/{run_id}")
def download_json(run_id: str):
    """Downloads the pipeline's raw result.json as a file attachment."""
    try:
        path = pipeline_service.resolve_result_file(run_id, "result.json")
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/json", filename=f"{run_id}_result.json")


@router.get("/download/csv/{run_id}")
def download_csv(run_id: str):
    """Downloads the pipeline's raw result.csv as a file attachment."""
    try:
        path = pipeline_service.resolve_result_file(run_id, "result.csv")
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="text/csv", filename=f"{run_id}_result.csv")
