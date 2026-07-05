# Face Search — Backend (FastAPI)

A thin FastAPI web layer around the existing, unmodified face-recognition
pipeline (`core/*.py` + `pipeline/run_pipeline.py`). This backend does not
contain any face-recognition logic itself — it saves uploads, invokes the
pipeline as a black-box subprocess, and serves back whatever the pipeline
already writes to `outputs/{run_id}/`.

```
backend/
├── app.py                    # FastAPI app, CORS, startup hooks
├── api/
│   └── routes.py              # all HTTP endpoints
├── services/
│   └── pipeline_service.py    # the ONLY module that knows the AI pipeline exists
├── models/
│   └── schemas.py              # Pydantic request/response models
├── uploads/                    # saved reference images + videos, per run_id
├── responses/                  # cached JSON responses, per run_id
├── static/                     # reserved for future use
├── tests/
│   └── test_api.py              # endpoint tests (pipeline subprocess mocked)
└── requirements.txt
```

## Why the pipeline is called as a subprocess, not imported

`services/pipeline_service.py` calls `pipeline/run_pipeline.py` exactly the
way you'd run it from the command line:

```
python pipeline/run_pipeline.py --reference <path> --video <path> --config <path> --run-id <id>
```

This was a deliberate choice, not a shortcut:

- **Zero changes needed** to `detector.py`, `aligner.py`, `embedder.py`,
  `matcher.py`, `aggregator.py`, or `run_pipeline.py` — they already expose
  exactly the CLI contract the backend needs.
- **True isolation.** insightface/onnxruntime load real model weights into
  memory; running the pipeline in its own short-lived process means that
  memory is released the instant a request finishes, and a crash inside the
  AI code (e.g. a corrupt video) can never take the API server down.
- **Structurally enforced boundary.** "Treat the pipeline as a black box"
  stops being just a comment — the backend process literally cannot reach
  into pipeline internals, because it never imports them.

## Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The backend also needs the AI pipeline's own dependencies installed (it
invokes `python pipeline/run_pipeline.py` using the same Python
interpreter), so also run, from the project root:

```bash
pip install -r requirements.txt   # the root-level AI pipeline requirements
```

## Running

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- API root: http://localhost:8000/
- Interactive docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

## Running tests

```bash
cd backend
python tests/test_api.py
```

These tests exercise every endpoint against a real FastAPI `TestClient`,
including upload validation, file serving, path-traversal protection, and
error handling. The only thing mocked out is the actual subprocess call
into the AI pipeline (`pipeline_service.run_face_search`) — that's an
environment concern (it needs downloaded ArcFace/SCRFD model weights), not
something these tests should depend on.

---

## API Documentation

All endpoints are prefixed with `/api`.

### `POST /api/search`

Runs a full face search: uploads a reference image and a video, invokes the
AI pipeline, and returns the structured result once processing completes.

**Request:** `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `reference` | file | Image. Allowed: `.jpg .jpeg .png .bmp .webp` |
| `video` | file | Video. Allowed: `.mp4 .mov .avi .mkv .webm` |

**Response:** `200 OK`

```json
{
  "run_id": "web_20260704_180512_a1b2c3",
  "processing_time_sec": 42.7,
  "person_found": true,
  "num_appearances": 2,
  "overall_best_similarity": 0.91,
  "reference_quality": {
    "sharpness_score": 145.2,
    "width": 640,
    "height": 480,
    "is_low_quality": false
  },
  "appearances": [
    { "start_sec": 6.24, "end_sec": 8.16, "duration_sec": 1.92, "max_similarity": 0.91, "avg_similarity": 0.87, "num_frames": 4 }
  ],
  "matched_frames": ["frame_000156_t6.24s_sim0.910.jpg", "..."],
  "annotated_frames": ["frame_000156_t6.24s_sim0.910.jpg", "..."],
  "reference_image_url": "/api/reference/web_20260704_180512_a1b2c3",
  "csv_download_url": "/api/download/csv/web_20260704_180512_a1b2c3",
  "json_download_url": "/api/download/json/web_20260704_180512_a1b2c3"
}
```

**Errors:**

| Status | When |
|---|---|
| `400` | Unsupported file type for reference or video |
| `422` | The pipeline itself failed (e.g. no face detected in the reference image) — `detail` contains the pipeline's own error message |
| `504` | Processing exceeded the configured timeout (default 20 minutes) |

---

### `GET /api/result/{run_id}`

Returns the complete result for a previously completed run — same shape as
`POST /api/search`'s response. Used by the frontend when a Results page is
loaded directly (e.g. a page refresh or shared link) instead of navigated
to right after a search.

`404` if `run_id` doesn't exist.

---

### `GET /api/frame/{run_id}/{filename}`

Serves one image from `outputs/{run_id}/matched_frames/{filename}` — the
original, unannotated video frame where a match was found.

`404` if the run or file doesn't exist. `400` if `filename` is invalid
(path traversal attempts are rejected before touching disk).

---

### `GET /api/annotated/{run_id}/{filename}`

Serves one image from `outputs/{run_id}/annotated_frames/{filename}` — the
same frame with bounding boxes, similarity scores, and timestamps drawn on
it (green = confident match, yellow = borderline match).

Same error behavior as `/api/frame`.

---

### `GET /api/reference/{run_id}`

Serves back the reference image the user originally uploaded for this run,
so the Results page can display it. `404` if not found.

---

### `GET /api/download/json/{run_id}`

Downloads the pipeline's raw `result.json` as a file attachment.

---

### `GET /api/download/csv/{run_id}`

Downloads the pipeline's raw `result.csv` as a file attachment.

---

### `GET /api/health`

Liveness check. Returns `{"status": "ok"}`. Does not touch the AI pipeline.

---

## Configuration

- **CORS origins** are set in `app.py` (`CORSMiddleware`). By default it
  allows `http://localhost:5173` and `http://localhost:3000` (Vite/CRA dev
  servers). Update this list for your deployed frontend's origin(s) in
  production.
- **Pipeline behavior** (similarity threshold, frame sample rate, etc.) is
  controlled entirely by `config/config.yaml` at the project root — the
  backend never overrides or duplicates these settings; it just passes
  `--config config/config.yaml` through to the pipeline unchanged.
- **Pipeline timeout** is set in `services/pipeline_service.py`
  (`PIPELINE_TIMEOUT_SECONDS`, default 20 minutes).

## Future scalability suggestions

See the root-level notes at the end of the top-level project README for
how this backend is intended to evolve (background job queue, database,
multi-video search, etc.) without needing to touch the AI pipeline.
