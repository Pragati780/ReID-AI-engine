"""
backend/services/pipeline_service.py

This module is the ONLY place in the backend that knows the existing AI
pipeline exists. It never imports anything from core/ or pipeline/ --
it invokes pipeline/run_pipeline.py as a separate OS process, exactly the
same way you'd run it from the command line. This is a deliberate
architectural choice, not a limitation:

    1. The AI pipeline is explicitly off-limits to modify or duplicate.
       Importing its internals into a long-lived FastAPI process would
       tempt future changes to reach into core/*.py from web code. Running
       it as a subprocess makes "black box" structurally true, not just a
       comment.
    2. insightface/onnxruntime load real model weights into memory. Running
       the pipeline in its own short-lived process means that memory is
       fully released the moment a request finishes, and a crash in the AI
       code (e.g. a corrupt video file) can never take the API server down
       with it -- it just returns a non-zero exit code we handle cleanly.
    3. Zero changes to pipeline/run_pipeline.py were required: it already
       exposes exactly the CLI contract (--reference, --video, --config,
       --run-id) this module needs.

Everything below this docstring is plain process/file orchestration --
no face-recognition logic lives here.
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Path configuration
# --------------------------------------------------------------------------
# backend/services/pipeline_service.py -> backend/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = PROJECT_ROOT / "pipeline" / "run_pipeline.py"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# How long a single search is allowed to run before we give up and report a
# timeout to the client, rather than hanging a request forever on a huge video.
PIPELINE_TIMEOUT_SECONDS = 60 * 20  # 20 minutes


class PipelineExecutionError(RuntimeError):
    """Raised when the AI pipeline subprocess exits with a non-zero status."""

    def __init__(self, message: str, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class PipelineTimeoutError(RuntimeError):
    """Raised when the AI pipeline subprocess exceeds PIPELINE_TIMEOUT_SECONDS."""


def run_face_search(reference_path: Path, video_path: Path, run_id: str, config_path: Optional[Path] = None) -> float:
    """
    Invokes the existing pipeline exactly as documented in its own CLI usage:

        python pipeline/run_pipeline.py --reference <path> --video <path>
                                          --config <path> --run-id <run_id>

    Args:
        reference_path: path to the saved reference image on disk.
        video_path: path to the saved video on disk.
        run_id: unique identifier for this run -- becomes the output folder
                name under outputs/, exactly as run_pipeline.py already
                supports via --run-id.
        config_path: optional override; defaults to the project's own
                     config/config.yaml so pipeline behavior (thresholds,
                     sampling rate, etc.) is controlled in exactly one
                     place, not duplicated into backend code.

    Returns:
        Wall-clock processing time in seconds.

    Raises:
        PipelineExecutionError: if the pipeline exits non-zero (e.g. no face
                                 found in the reference image, unreadable video).
        PipelineTimeoutError: if the pipeline runs longer than the configured timeout.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    
    import sys

    print("=" * 60)
    print("Backend Python:", sys.executable)
    print("=" * 60)
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--reference", str(reference_path),
        "--video", str(video_path),
        "--config", str(config_path),
        "--run-id", run_id,
    ]
    print("Command:", command)
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),      # so config's relative "outputs" path resolves correctly
            capture_output=True,
            text=True,
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineTimeoutError(
            f"Face search for run '{run_id}' exceeded {PIPELINE_TIMEOUT_SECONDS}s and was aborted."
        ) from exc

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise PipelineExecutionError(
            f"Pipeline exited with code {result.returncode} for run '{run_id}'.",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return elapsed


# --------------------------------------------------------------------------
# Reading pipeline outputs back out
# --------------------------------------------------------------------------

def get_run_output_dir(run_id: str) -> Path:
    """Returns outputs/{run_id}, raising FileNotFoundError if it doesn't exist."""
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No results found for run_id '{run_id}'.")
    return run_dir


def load_result_json(run_id: str) -> Dict:
    """Reads the pipeline's own result.json for a run, untouched and unparsed
    beyond standard JSON decoding -- we don't recompute anything the pipeline
    already computed."""
    run_dir = get_run_output_dir(run_id)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json missing for run_id '{run_id}' -- the pipeline may not have finished.")
    with open(result_path, "r") as f:
        return json.load(f)


def list_output_images(run_id: str, subfolder: str) -> List[str]:
    """
    Lists filenames (not full paths) inside outputs/{run_id}/{subfolder},
    sorted so the frontend gallery order is stable and chronological
    (filenames already encode frame index first -- see core/utils.py's
    _build_output_filename -- so a plain sort is chronological for free).
    """
    folder = get_run_output_dir(run_id) / subfolder
    if not folder.is_dir():
        return []
    files = sorted(os.path.basename(p) for p in glob.glob(str(folder / "*.jpg")))
    return files


def overall_best_similarity(result_json: Dict) -> Optional[float]:
    """Highest max_similarity across all appearances, or None if no appearances."""
    appearances = result_json.get("appearances", [])
    if not appearances:
        return None
    return max(a["max_similarity"] for a in appearances)


def resolve_safe_output_file(run_id: str, subfolder: str, filename: str) -> Path:
    """
    Resolves outputs/{run_id}/{subfolder}/{filename} and guarantees the
    result is actually inside that folder -- rejecting any filename that
    tries to escape it (e.g. "../../etc/passwd") before ever touching disk.

    This is deliberately paranoid: `filename` comes directly from a URL
    path parameter, i.e. from the outside world.
    """
    folder = (get_run_output_dir(run_id) / subfolder).resolve()
    candidate = (folder / filename).resolve()

    if not str(candidate).startswith(str(folder) + os.sep) and candidate != folder:
        raise ValueError(f"Invalid filename '{filename}'.")
    if not candidate.is_file():
        raise FileNotFoundError(f"File '{filename}' not found in {subfolder} for run_id '{run_id}'.")
    return candidate


def resolve_result_file(run_id: str, filename: str) -> Path:
    """Resolves outputs/{run_id}/{filename} for the top-level result.json /
    result.csv files (no subfolder), with the same path-traversal guard."""
    run_dir = get_run_output_dir(run_id).resolve()
    candidate = (run_dir / filename).resolve()

    if not str(candidate).startswith(str(run_dir) + os.sep) and candidate != run_dir:
        raise ValueError(f"Invalid filename '{filename}'.")
    if not candidate.is_file():
        raise FileNotFoundError(f"'{filename}' not found for run_id '{run_id}'.")
    return candidate


def find_uploaded_reference_image(run_id: str, uploads_dir: Path) -> Path:
    """
    Locates the reference image saved for this run under backend/uploads/{run_id}/.
    Uploaded reference files are always saved as "reference.<original-extension>"
    (see api/routes.py) -- this just finds whichever extension was used.
    """
    run_upload_dir = uploads_dir / run_id
    matches = sorted(glob.glob(str(run_upload_dir / "reference.*")))
    if not matches:
        raise FileNotFoundError(f"No reference image found for run_id '{run_id}'.")
    return Path(matches[0])
