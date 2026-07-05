import { useEffect, useState } from "react";
import { useLocation, useParams, Link } from "react-router-dom";
import {
  UserCheck,
  UserX,
  Gauge,
  Timer,
  ListChecks,
  Download,
  FileJson,
  FileSpreadsheet,
  ArrowLeft,
} from "lucide-react";
import Card from "../components/Card";
import StatCard from "../components/StatCard";
import TimelineItem from "../components/TimelineItem";
import ImageGalleryCard from "../components/ImageGalleryCard";
import LoadingSpinner from "../components/LoadingSpinner";
import {
  getResult,
  matchedFrameUrl,
  annotatedFrameUrl,
  referenceImageUrl,
  jsonDownloadUrl,
  csvDownloadUrl,
} from "../services/api";

/**
 * src/pages/ResultsPage.jsx
 *
 * Shows everything the backend returned for one run: reference image,
 * headline stats, the appearance timeline, both image galleries, and
 * download links. Works two ways:
 *   1. Navigated to directly from HomePage with the result already in
 *      router state (no extra network call).
 *   2. Loaded directly via URL (e.g. a refresh or a shared link) --
 *      in that case it fetches GET /api/result/{runId} itself.
 */
export default function ResultsPage() {
  const { runId } = useParams();
  const location = useLocation();

  const [result, setResult] = useState(location.state?.result ?? null);
  const [isLoading, setIsLoading] = useState(!location.state?.result);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (result) return; // already have it from navigation state
    let cancelled = false;

    setIsLoading(true);
    getResult(runId)
      .then((data) => {
        if (!cancelled) setResult(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err?.response?.data?.detail || "Could not load this result.");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  if (isLoading) {
    return <LoadingSpinner message="Loading results..." />;
  }

  if (error || !result) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-16 text-center">
        <p className="text-lg font-semibold text-slate-800 dark:text-slate-100">
          {error || "No result found."}
        </p>
        <Link
          to="/"
          className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-brand-600 hover:underline dark:text-brand-400"
        >
          <ArrowLeft size={15} /> Back to search
        </Link>
      </div>
    );
  }

  const {
    person_found,
    num_appearances,
    overall_best_similarity,
    processing_time_sec,
    appearances,
    matched_frames,
    annotated_frames,
  } = result;

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <Link
        to="/"
        className="mb-6 inline-flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-brand-600 dark:text-slate-400 dark:hover:text-brand-400"
      >
        <ArrowLeft size={15} /> New search
      </Link>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Reference image */}
        <Card className="flex flex-col items-center gap-4 lg:col-span-1">
          <p className="self-start text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Reference Image
          </p>
          <img
            src={referenceImageUrl(runId)}
            alt="Reference"
            className="max-h-64 w-full rounded-xl object-contain"
          />
        </Card>

        {/* Headline stats */}
        <div className="grid grid-cols-2 gap-4 lg:col-span-2">
          <StatCard
            icon={person_found ? UserCheck : UserX}
            label="Person Found"
            value={person_found ? "Yes" : "No"}
            accent={person_found ? "green" : "rose"}
          />
          <StatCard
            icon={Gauge}
            label="Best Similarity"
            value={overall_best_similarity != null ? overall_best_similarity.toFixed(3) : "—"}
            accent="brand"
          />
          <StatCard
            icon={ListChecks}
            label="Appearances"
            value={num_appearances}
            accent="amber"
          />
          <StatCard
            icon={Timer}
            label="Processing Time"
            value={processing_time_sec != null ? `${processing_time_sec.toFixed(1)}s` : "—"}
            accent="brand"
          />
        </div>
      </div>

      {/* Timeline */}
      <section className="mt-8">
        <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-white">Timeline</h2>
        {appearances.length === 0 ? (
          <Card className="text-center text-sm text-slate-500 dark:text-slate-400">
            This person was not found in the video.
          </Card>
        ) : (
          <div className="space-y-2">
            {appearances.map((appearance, i) => (
              <TimelineItem key={`${appearance.start_sec}-${i}`} appearance={appearance} index={i} />
            ))}
          </div>
        )}
      </section>

      {/* Matched frames gallery */}
      <section className="mt-8">
        <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-white">Matched Frames</h2>
        <Card>
          <ImageGalleryCard
            filenames={matched_frames}
            buildUrl={(filename) => matchedFrameUrl(runId, filename)}
            emptyLabel="No matched frames were saved for this run."
          />
        </Card>
      </section>

      {/* Annotated frames gallery */}
      <section className="mt-8">
        <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-white">Annotated Frames</h2>
        <Card>
          <ImageGalleryCard
            filenames={annotated_frames}
            buildUrl={(filename) => annotatedFrameUrl(runId, filename)}
            emptyLabel="No annotated frames were saved for this run."
          />
        </Card>
      </section>

      {/* Downloads */}
      <section className="mt-8 mb-12">
        <h2 className="mb-3 text-lg font-semibold text-slate-900 dark:text-white">Downloads</h2>
        <div className="flex flex-wrap gap-3">
          <a
            href={jsonDownloadUrl(runId)}
            className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition hover:border-brand-300 hover:text-brand-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-brand-600"
          >
            <FileJson size={16} /> Download JSON
            <Download size={14} className="text-slate-400" />
          </a>
          <a
            href={csvDownloadUrl(runId)}
            className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm transition hover:border-brand-300 hover:text-brand-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-brand-600"
          >
            <FileSpreadsheet size={16} /> Download CSV
            <Download size={14} className="text-slate-400" />
          </a>
        </div>
      </section>
    </div>
  );
}
