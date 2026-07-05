/**
 * src/services/api.js
 *
 * The ONLY module in the frontend that knows the backend's URLs. Every
 * component talks to the backend through the functions exported here --
 * nothing else in the frontend constructs a request URL by hand. This is
 * what makes "frontend communicates ONLY with FastAPI" an enforced
 * structural fact rather than just a convention: there is exactly one
 * place that would need to change if the API's base URL or shape ever did.
 */

import axios from "axios";

// Configurable via a .env file (VITE_API_BASE_URL=http://localhost:8000).
// Falls back to the FastAPI backend's default local dev address.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: `${API_BASE_URL}/api`,
});

/**
 * Uploads a reference image + video and runs the face search.
 * This call blocks until the backend finishes processing (the backend
 * itself runs the AI pipeline synchronously), so it can take anywhere from
 * a few seconds to a few minutes depending on video length.
 *
 * @param {File} referenceFile
 * @param {File} videoFile
 * @returns {Promise<object>} SearchResponse payload
 */
export async function searchFaces(referenceFile, videoFile) {
  const formData = new FormData();
  formData.append("reference", referenceFile);
  formData.append("video", videoFile);

  const response = await client.post("/search", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    // Video processing can legitimately take minutes -- don't let axios's
    // default timeout cut it off.
    timeout: 20 * 60 * 1000,
  });
  return response.data;
}

/**
 * Fetches a previously computed result by run_id (e.g. on a page refresh
 * or when sharing a results link).
 */
export async function getResult(runId) {
  const response = await client.get(`/result/${runId}`);
  return response.data;
}

/** Full URL for a matched (unannotated) frame image. */
export function matchedFrameUrl(runId, filename) {
  return `${API_BASE_URL}/api/frame/${runId}/${filename}`;
}

/** Full URL for an annotated frame image (boxes + score + timestamp drawn on it). */
export function annotatedFrameUrl(runId, filename) {
  return `${API_BASE_URL}/api/annotated/${runId}/${filename}`;
}

/** Full URL for the originally uploaded reference image. */
export function referenceImageUrl(runId) {
  return `${API_BASE_URL}/api/reference/${runId}`;
}

/** Full URL to download the raw result.json for a run. */
export function jsonDownloadUrl(runId) {
  return `${API_BASE_URL}/api/download/json/${runId}`;
}

/** Full URL to download the raw result.csv for a run. */
export function csvDownloadUrl(runId) {
  return `${API_BASE_URL}/api/download/csv/${runId}`;
}

export default {
  searchFaces,
  getResult,
  matchedFrameUrl,
  annotatedFrameUrl,
  referenceImageUrl,
  jsonDownloadUrl,
  csvDownloadUrl,
};
