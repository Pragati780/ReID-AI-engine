<div align="center">

# Face Re-Identification

**Find a person in any video, from a single reference photo.**

Upload one reference image and one video — get back every timestamp that person appears,
with a confidence score for each appearance, matched face crops, and annotated frames.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3-38BDF8?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace%20%2B%20SCRFD-orange)](https://github.com/deepinsight/insightface)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Overview](#-overview) •
[Architecture](#-architecture) •
[Quick Start](#-quick-start) •
[Usage](#-usage) •
[API Reference](#-api-reference) •
[Testing](#-testing) •
[Roadmap](#-roadmap)

</div>

---

## Overview

**Face Re-Identification** is a full-stack re-identification application. Given a
single reference photo — even one that's blurry, off-angle, or low
resolution — it scans a video and reports:

- **Whether** the person appears at all
- **When** — every appearance as a clean timestamp range (e.g. `6.24s → 8.16s`)
- **How confidently** — a cosine-similarity score per appearance
- **Visual proof** — matched face crops, raw matched frames, and annotated
  frames with bounding boxes drawn on

The project is deliberately split into two independent layers:

| Layer | What it is | Can be used |
|---|---|---|
| **AI Pipeline** | Detection → alignment → ArcFace embedding → cosine matching → timestamp aggregation | Standalone, via CLI |
| **Web Application** | FastAPI backend + React frontend | On top of the pipeline, unmodified |

The web app never reaches into the AI pipeline's internals — it calls it as
a black-box subprocess, exactly the way you'd run it from a terminal. This
means the core recognition logic can be tested, understood, and trusted
independently of the web layer built around it.

<img width="1283" height="623" alt="image" src="https://github.com/user-attachments/assets/e823f900-a7ba-480b-b800-ad17f09d6a2f" />

---

## Architecture

<img width="1422" height="1442" alt="image" src="https://github.com/user-attachments/assets/bd1c32bf-97b6-4ba8-b45f-f04386fd3d79" />


**Why a subprocess, not an import?** Running `pipeline/run_pipeline.py` as
an isolated process — instead of importing `core/*.py` into the backend —
means the AI pipeline can never be modified from web code, model memory is
released the instant a request finishes, and a crash on a corrupt video
can never take the API server down with it. It's a structural guarantee,
not a convention.

### Pipeline stages, in detail
                                                              ▼
<img width="1865" height="478" alt="image" src="https://github.com/user-attachments/assets/4fc8981f-36f9-43d2-91d0-74b624774019" />


Every design decision behind this (why SCRFD over YOLO, why ArcFace, why
cosine similarity, how the similarity threshold is calibrated, how
timestamp fragmentation is prevented) is documented as inline commentary
in the corresponding module — start at `core/detector.py` and read through
`aligner.py → embedder.py → matcher.py → aggregator.py` in order.

---

## Key Features

**AI Pipeline**
- One-shot recognition from a single, imperfect reference image
- SCRFD face detection (RetinaFace family) with 5-point landmark alignment
- ArcFace 512-d embeddings via `insightface`
- Quality gating for blurry/undersized reference or candidate faces
- Gap-tolerant timestamp aggregation (no fragmented, noisy appearances)
- Fully unit-tested matching/aggregation logic — no model download required to verify the logic itself

**Backend (FastAPI)**
- Clean multipart upload → pipeline invocation → structured JSON response
- Serves matched frames, annotated frames, reference image, and CSV/JSON downloads
- Path-traversal-safe file serving
- Friendly error messages for unsupported files, pipeline failures, and timeouts
- Fully covered by endpoint tests (pipeline subprocess mocked, everything else real)

**Frontend (React + Vite + Tailwind)**
- Drag-and-drop upload with live image/video preview
- Full-screen animated processing state
- Results dashboard: stats, chronological timeline, image galleries with a built-in lightbox
- One-click JSON/CSV downloads
- Responsive layout, dark mode

---

## Tech Stack

| Category | Technology |
|---|---|
| Face detection | SCRFD (insightface `buffalo_l` pack, RetinaFace family) |
| Face recognition | ArcFace (insightface `buffalo_l` pack) |
| AI runtime | ONNX Runtime |
| Video/image processing | OpenCV, NumPy |
| Backend framework | FastAPI, Pydantic, Uvicorn |
| Frontend framework | React 18, Vite, React Router |
| Styling | Tailwind CSS |
| HTTP client | Axios |
| Icons | Lucide React |
| Testing | Python `unittest`-style scripts, FastAPI `TestClient` |

---

## Project Structure

```
face-reid-project/
├── core/                       # AI pipeline modules — one responsibility each
│   ├── detector.py               # face detection (SCRFD) + landmarks
│   ├── aligner.py                 # landmark-based face alignment
│   ├── embedder.py                # ArcFace embedding generation
│   ├── quality.py                  # blur / size quality gating
│   ├── matcher.py                   # cosine similarity matching
│   ├── aggregator.py                 # frame matches → timestamp ranges
│   └── utils.py                       # video sampling, saving, drawing
│
├── pipeline/
│   └── run_pipeline.py            # CLI orchestrator tying core/ together
│
├── config/
│   └── config.yaml                 # every tunable threshold, in one place
│
├── eval/
│   ├── eval_metrics.py              # precision / recall / temporal IoU
│   └── ground_truth/                 # manually-labeled test videos
│
├── tests/                         # AI pipeline unit tests (no model needed)
│
├── data/                          # your reference images & videos go here
├── outputs/                       # pipeline results, per run_id
│
├── backend/                       # FastAPI web layer
│   ├── app.py                       # app entrypoint, CORS
│   ├── api/routes.py                 # HTTP endpoints
│   ├── services/pipeline_service.py   # the ONLY file that knows the AI pipeline exists
│   ├── models/schemas.py               # Pydantic request/response models
│   ├── uploads/, responses/, static/
│   └── tests/test_api.py
│
├── frontend/                      # React web app
│   └── src/
│       ├── components/               # Navbar, Card, StatCard, galleries, etc.
│       ├── pages/                     # HomePage, ResultsPage
│       ├── services/api.js             # the ONLY file that knows backend URLs
│       └── context/ThemeContext.jsx     # dark mode
│
└── notebooks/                     # exploratory threshold calibration, debugging
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- ~2 GB free disk space (for the ArcFace/SCRFD model pack, downloaded automatically on first run)
- Internet access on first run (to download pretrained model weights)

### 1. Clone and install the AI pipeline

```bash
git clone <your-repo-url> ReID-AI-engine
cd ReID-AI-engine
pip install -r requirements.txt
```

### 2. Install and run the backend

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000** (interactive docs at `/docs`).

### 3. Install and run the frontend

```bash
cd frontend
npm install
cp .env.example .env      # defaults to http://localhost:8000 -- adjust if needed
npm run dev
```

Open **http://localhost:5173** and you're ready to search.

> On first search, `insightface` downloads the `buffalo_l` model pack
> (~300MB) automatically. This only happens once.

---

## Usage

### Option A — Web App (recommended)

1. Open the app, upload a reference image and a video.
2. Click **Run Search** and wait for processing to complete.
3. Review the results: headline stats, appearance timeline, matched and
   annotated frame galleries, and download the raw JSON/CSV.

### Option B — Command line (no web layer)

```bash
python pipeline/run_pipeline.py \
    --reference data/reference_images/person.jpg \
    --video data/videos/clip.mp4 \
    --config config/config.yaml \
    --run-id demo_run_01
```

Outputs land in `outputs/demo_run_01/`:

```
outputs/demo_run_01/
├── result.json              # structured appearances + confidence scores
├── result.csv                 # same data, flat table
├── matched_faces/               # cropped face per matched frame
├── matched_frames/                # original, unmodified frame per match
├── annotated_frames/                # frame + bounding box + score + timestamp
├── reference_embedding.npy
├── all_embeddings.npy
└── logs/pipeline.log
```

---

## API Reference

Full request/response schemas and error codes are documented in
[`backend/README.md`](backend/README.md). Summary:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/search` | Upload reference image + video, run the full pipeline, return results |
| `GET` | `/api/result/{run_id}` | Fetch a previously completed run's result |
| `GET` | `/api/frame/{run_id}/{filename}` | Serve a matched (unannotated) frame |
| `GET` | `/api/annotated/{run_id}/{filename}` | Serve an annotated frame (boxes + labels) |
| `GET` | `/api/reference/{run_id}` | Serve back the uploaded reference image |
| `GET` | `/api/download/json/{run_id}` | Download raw `result.json` |
| `GET` | `/api/download/csv/{run_id}` | Download raw `result.csv` |
| `GET` | `/api/health` | Liveness check |

Interactive Swagger docs are always available at `/docs` while the backend
is running.

---

## Configuration

All pipeline behavior is controlled from a single file:
[`config/config.yaml`](config/config.yaml). The backend never overrides or
duplicates these values — it passes the same config file straight through
to the pipeline.

| Key | Default | Meaning |
|---|---|---|
| `matching.similarity_threshold` | `0.38` | Cosine similarity cutoff for a "match" — calibrate on your own data via `eval/eval_metrics.py` |
| `video.frame_sample_fps` | `2` | How densely the video is scanned |
| `aggregation.gap_tolerance_seconds` | `1.0` | Bridges brief gaps (motion blur/occlusion) into one continuous appearance |
| `aggregation.min_appearance_duration_seconds` | `0.5` | Drops appearances shorter than this (likely false positives) |
| `quality.blur_variance_threshold` | `30.0` | Below this, a face is flagged as too blurry to fully trust |
| `model.pack_name` | `buffalo_l` | insightface model pack (SCRFD detector + ArcFace recognizer) |

The frontend's only configurable value is `VITE_API_BASE_URL` in
`frontend/.env`, pointing it at wherever the backend is deployed.

---

## Testing

```bash
# AI pipeline logic (matching, aggregation, evaluation metrics) -- no model download needed
python tests/test_matcher.py
python tests/test_aggregator.py
python tests/test_eval_metrics.py

# Backend API (every endpoint, error paths, path-traversal protection)
cd backend && python tests/test_api.py
```

The backend tests mock only the actual AI subprocess call (which needs
downloaded model weights) — file saving, validation, response shaping, and
every HTTP endpoint are exercised for real against a live `TestClient`.

### Measuring real-world accuracy

1. Watch a short test video once and record the true appearance timestamps
   in the format shown in `eval/ground_truth/example_ground_truth.json`.
2. Run the pipeline on that video.
3. Compare:
   ```bash
   python eval/eval_metrics.py \
       --predicted outputs/demo_run_01/result.json \
       --ground-truth eval/ground_truth/my_test_video.json
   ```
   This reports precision, recall, F1, and mean temporal IoU — use it to
   calibrate `matching.similarity_threshold`, not just trust a demo run.

---

## Design Philosophy & What's Deliberately Not Included

This project favors a small, well-understood, independently testable core
over premature infrastructure:

- **No super-resolution on blurry references** — models like GFPGAN can
  hallucinate facial detail that isn't real, risking a sharper but *wrong*
  identity signature.
- **No custom-trained or fine-tuned embedding model** — pretrained ArcFace
  is already extremely strong for one-shot matching.
- **No full-body re-identification (yet)** — a genuinely different problem
  (different embedding models, e.g. OSNet) — see [Roadmap](#-roadmap).
- **No vector database (yet)** — for one reference vs. one video, an
  in-memory comparison is enough; see [Roadmap](#-roadmap) for when this changes.
- **Synchronous request handling** — `POST /api/search` blocks until the
  pipeline finishes. Simple and predictable for an MVP; see Roadmap for
  the background-job upgrade path.

---

## Roadmap

| Direction | What changes | What doesn't |
|---|---|---|
| **Background jobs** | `POST /api/search` returns immediately; frontend polls for status | AI pipeline stays a subprocess call |
| **Persistent run metadata** | A real database (Postgres/SQLite) replaces filesystem checks | `result.json` stays the pipeline's own output format |
| **Multi-video search** | `core/matcher.py`'s cosine loop → FAISS / pgvector nearest-neighbor lookup | `core/detector.py`, `aligner.py`, `embedder.py` unchanged |
| **Person re-identification** | A second embedder (e.g. OSNet) added alongside `FaceEmbedder` for body appearance | Face pipeline remains the primary signal |
| **Live progress** | WebSockets/SSE streaming frame-by-frame progress | Requires background jobs first |
| **Auth & multi-tenancy** | Per-user run history, access control | `run_id`-based URLs already make results individually addressable |

---

## Contributing

Contributions are welcome. A few ground rules to keep the architecture's
guarantees intact:

1. **Never import `core/*.py` into `backend/`.** The AI pipeline is invoked
   as a subprocess on purpose — see [Architecture](#-architecture).
2. **New pipeline behavior belongs in `config/config.yaml`**, not hardcoded
   in `core/` or `pipeline/`.
3. **Every new `core/` function should have a corresponding test** in
   `tests/` that doesn't require downloading model weights, where feasible.
4. **The frontend only talks to the backend through `src/services/api.js`.**
   No component should construct a backend URL directly.

Please open an issue describing the change before submitting a large PR.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) — SCRFD
  detection and ArcFace recognition models
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [FastAPI](https://fastapi.tiangolo.com/) and [React](https://react.dev/)
  for making the web layer straightforward to build around an unmodified
  AI core
