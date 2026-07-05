"""
backend/tests/test_api.py

Exercises every backend endpoint end-to-end using FastAPI's TestClient.

The ONE thing we mock is services.pipeline_service.run_face_search itself --
i.e. the actual subprocess call into the AI pipeline. That's deliberate:
running the real pipeline requires downloaded ArcFace/SCRFD model weights,
which is an environment concern, not something this test suite should
depend on. Everything else (file saving, path-traversal protection, response
shaping, error handling, file serving/downloads) is exercised for real
against real files on disk.

Run with:
    cd backend && python3 tests/test_api.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import services.pipeline_service as pipeline_service  # noqa: E402
from app import app  # noqa: E402

client = TestClient(app)

PROJECT_ROOT = pipeline_service.PROJECT_ROOT
TEST_RUN_ID = "test_run_manual_setup"


def _make_fake_pipeline_output(run_id: str):
    """Manually creates the files the REAL pipeline would produce, so we can
    test the backend's reading/serving logic without running any AI model."""
    run_dir = PROJECT_ROOT / "outputs" / run_id
    (run_dir / "matched_frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "annotated_frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "matched_faces").mkdir(parents=True, exist_ok=True)

    # A minimal, valid 1x1 JPEG (enough for FileResponse / content-type checks).
    tiny_jpeg_bytes = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300030202020202"
        "03020202030303030406040404040408060605060909080a0a090809090a0c"
        "0f0c0a0b0e0b09090d110d0e0f101011100a0c12131210130f101010ffc900"
        "0b080001000101011100ffcc00060010100005ffda0008010100003f00d2cf20ffd9"
    )
    (run_dir / "matched_frames" / "frame_000010_t1.00s_sim0.900.jpg").write_bytes(tiny_jpeg_bytes)
    (run_dir / "annotated_frames" / "frame_000010_t1.00s_sim0.900.jpg").write_bytes(tiny_jpeg_bytes)

    result_json = {
        "reference_quality": {"sharpness_score": 120.5, "width": 200, "height": 200, "is_low_quality": False},
        "person_found": True,
        "num_appearances": 1,
        "appearances": [
            {
                "start_sec": 1.0,
                "end_sec": 2.0,
                "duration_sec": 1.0,
                "max_similarity": 0.9,
                "avg_similarity": 0.85,
                "num_frames": 2,
            }
        ],
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result_json, f)

    with open(run_dir / "result.csv", "w") as f:
        f.write("start_sec,end_sec,duration_sec,max_similarity,avg_similarity,num_frames\n1.0,2.0,1.0,0.9,0.85,2\n")

    return run_dir


def _cleanup(run_id: str):
    run_dir = PROJECT_ROOT / "outputs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    upload_dir = BACKEND_ROOT / "uploads" / run_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    cache_file = BACKEND_ROOT / "responses" / f"{run_id}.json"
    if cache_file.exists():
        cache_file.unlink()


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_result_for_unknown_run_returns_404():
    resp = client.get("/api/result/does_not_exist")
    assert resp.status_code == 404


def test_get_result_rebuilds_from_result_json_when_no_cache():
    _make_fake_pipeline_output(TEST_RUN_ID)
    try:
        resp = client.get(f"/api/result/{TEST_RUN_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == TEST_RUN_ID
        assert data["person_found"] is True
        assert data["num_appearances"] == 1
        assert data["overall_best_similarity"] == 0.9
        assert data["matched_frames"] == ["frame_000010_t1.00s_sim0.900.jpg"]
        assert data["annotated_frames"] == ["frame_000010_t1.00s_sim0.900.jpg"]
        assert data["csv_download_url"] == f"/api/download/csv/{TEST_RUN_ID}"
        assert data["json_download_url"] == f"/api/download/json/{TEST_RUN_ID}"
    finally:
        _cleanup(TEST_RUN_ID)


def test_get_matched_frame_serves_image():
    _make_fake_pipeline_output(TEST_RUN_ID)
    try:
        resp = client.get(f"/api/frame/{TEST_RUN_ID}/frame_000010_t1.00s_sim0.900.jpg")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0
    finally:
        _cleanup(TEST_RUN_ID)


def test_get_annotated_frame_serves_image():
    _make_fake_pipeline_output(TEST_RUN_ID)
    try:
        resp = client.get(f"/api/annotated/{TEST_RUN_ID}/frame_000010_t1.00s_sim0.900.jpg")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
    finally:
        _cleanup(TEST_RUN_ID)


def test_get_frame_missing_file_returns_404():
    _make_fake_pipeline_output(TEST_RUN_ID)
    try:
        resp = client.get(f"/api/frame/{TEST_RUN_ID}/does_not_exist.jpg")
        assert resp.status_code == 404
    finally:
        _cleanup(TEST_RUN_ID)


def test_path_traversal_is_rejected_at_service_layer():
    """Directly exercises the path-traversal guard in pipeline_service,
    independent of how any particular ASGI server normalizes slashes in URLs."""
    _make_fake_pipeline_output(TEST_RUN_ID)
    try:
        threw = False
        try:
            pipeline_service.resolve_safe_output_file(TEST_RUN_ID, "matched_frames", "../../../etc/passwd")
        except ValueError:
            threw = True
        assert threw, "Expected a ValueError for a path-traversal attempt"
    finally:
        _cleanup(TEST_RUN_ID)


def test_download_json_and_csv():
    _make_fake_pipeline_output(TEST_RUN_ID)
    try:
        resp_json = client.get(f"/api/download/json/{TEST_RUN_ID}")
        assert resp_json.status_code == 200
        assert resp_json.headers["content-type"].startswith("application/json")
        assert resp_json.json()["person_found"] is True

        resp_csv = client.get(f"/api/download/csv/{TEST_RUN_ID}")
        assert resp_csv.status_code == 200
        assert "text/csv" in resp_csv.headers["content-type"]
        assert "max_similarity" in resp_csv.text
    finally:
        _cleanup(TEST_RUN_ID)


def test_search_endpoint_end_to_end_with_mocked_pipeline():
    """
    Exercises the full POST /api/search flow -- upload validation, file
    saving, calling into pipeline_service, reading its outputs back, caching
    the response -- with only the actual AI subprocess call mocked out.
    """
    original_run_face_search = pipeline_service.run_face_search

    captured_run_id = {}

    def fake_run_face_search(reference_path, video_path, run_id, config_path=None):
        # Assert the backend actually saved the uploaded files before calling us.
        assert reference_path.exists()
        assert video_path.exists()
        captured_run_id["run_id"] = run_id
        _make_fake_pipeline_output(run_id)
        return 3.14  # fake processing time

    pipeline_service.run_face_search = fake_run_face_search
    try:
        files = {
            "reference": ("ref.jpg", b"\xff\xd8\xff\xe0fakejpegbytes", "image/jpeg"),
            "video": ("clip.mp4", b"fakevideobytes", "video/mp4"),
        }
        resp = client.post("/api/search", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["processing_time_sec"] == 3.14
        assert data["person_found"] is True
        assert data["reference_image_url"] == f"/api/reference/{captured_run_id['run_id']}"

        # GET /api/result/{run_id} should now hit the cached response.
        resp2 = client.get(f"/api/result/{captured_run_id['run_id']}")
        assert resp2.status_code == 200
        assert resp2.json()["processing_time_sec"] == 3.14

        # Reference image should be retrievable too.
        resp3 = client.get(f"/api/reference/{captured_run_id['run_id']}")
        assert resp3.status_code == 200
    finally:
        pipeline_service.run_face_search = original_run_face_search
        if "run_id" in captured_run_id:
            _cleanup(captured_run_id["run_id"])


def test_search_rejects_unsupported_file_types():
    files = {
        "reference": ("ref.txt", b"not an image", "text/plain"),
        "video": ("clip.mp4", b"fakevideobytes", "video/mp4"),
    }
    resp = client.post("/api/search", files=files)
    assert resp.status_code == 400


def test_search_reports_pipeline_failure_as_422():
    original_run_face_search = pipeline_service.run_face_search

    def failing_run_face_search(reference_path, video_path, run_id, config_path=None):
        raise pipeline_service.PipelineExecutionError(
            "boom", stdout="", stderr="Traceback...\nValueError: No face detected in the reference image."
        )

    pipeline_service.run_face_search = failing_run_face_search
    try:
        files = {
            "reference": ("ref.jpg", b"\xff\xd8\xff\xe0fakejpegbytes", "image/jpeg"),
            "video": ("clip.mp4", b"fakevideobytes", "video/mp4"),
        }
        resp = client.post("/api/search", files=files)
        assert resp.status_code == 422
        assert "No face detected" in resp.json()["detail"]
    finally:
        pipeline_service.run_face_search = original_run_face_search
        # clean up whatever run_id got created for the uploads
        for d in (BACKEND_ROOT / "uploads").glob("web_*"):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {fn.__name__} -> {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
