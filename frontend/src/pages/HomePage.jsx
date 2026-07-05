import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, AlertTriangle } from "lucide-react";
import FileUploadCard from "../components/FileUploadCard";
import LoadingSpinner from "../components/LoadingSpinner";
import { searchFaces } from "../services/api";

/**
 * src/pages/HomePage.jsx
 *
 * Upload screen: pick a reference image + a video, then run the search.
 * While the request is in flight, this page swaps its content for
 * <LoadingSpinner /> rather than navigating away, since the backend call
 * is a single blocking request that resolves once the pipeline finishes.
 */
export default function HomePage() {
  const [referenceFile, setReferenceFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const navigate = useNavigate();

  const canSubmit = Boolean(referenceFile && videoFile) && !isLoading;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setError(null);
    setIsLoading(true);
    try {
      const result = await searchFaces(referenceFile, videoFile);
      navigate(`/results/${result.run_id}`, { state: { result } });
    } catch (err) {
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        "Something went wrong while processing your request.";
      setError(detail);
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <LoadingSpinner message="Processing video..." />;
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-12">
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-4xl">
          Find someone in your video
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-slate-500 dark:text-slate-400">
          Upload a reference photo and a video. We'll detect every appearance of
          that person, with timestamps and confidence scores for each match.
        </p>
      </div>

      {error && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-300">
          <AlertTriangle size={18} className="mt-0.5 shrink-0" />
          <p>{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <FileUploadCard
          kind="image"
          file={referenceFile}
          onSelect={setReferenceFile}
          onClear={() => setReferenceFile(null)}
        />
        <FileUploadCard
          kind="video"
          file={videoFile}
          onSelect={setVideoFile}
          onClear={() => setVideoFile(null)}
        />
      </div>

      <div className="mt-8 flex justify-center">
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="flex items-center gap-2 rounded-xl bg-brand-600 px-8 py-3 text-sm font-semibold text-white shadow-lg shadow-brand-600/25 transition enabled:hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Search size={18} />
          Run Search
        </button>
      </div>
    </div>
  );
}
