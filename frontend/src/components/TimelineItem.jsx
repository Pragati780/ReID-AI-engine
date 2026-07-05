import { Clock } from "lucide-react";

/**
 * src/components/TimelineItem.jsx
 *
 * Renders one appearance range, e.g. "6.24s -> 8.16s", with its confidence
 * scores. Purely presentational -- the numbers come straight from the
 * backend's AppearanceSchema (which itself mirrors the pipeline's own
 * aggregator output).
 */
function formatSeconds(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = (totalSeconds % 60).toFixed(2);
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

export default function TimelineItem({ appearance, index }) {
  const confidencePercent = Math.round(appearance.max_similarity * 100);

  return (
    <div className="flex items-center gap-4 rounded-xl border border-slate-200 bg-white px-4 py-3 transition hover:border-brand-300 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-brand-600">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
        <Clock size={16} />
      </div>

      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-slate-800 dark:text-slate-100">
          Appearance {index + 1}:{" "}
          <span className="font-mono text-brand-600 dark:text-brand-400">
            {formatSeconds(appearance.start_sec)} → {formatSeconds(appearance.end_sec)}
          </span>
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Duration {appearance.duration_sec.toFixed(2)}s · {appearance.num_frames} matched frame
          {appearance.num_frames === 1 ? "" : "s"} · avg similarity {appearance.avg_similarity.toFixed(3)}
        </p>
      </div>

      <div className="shrink-0 text-right">
        <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{confidencePercent}%</p>
        <p className="text-[11px] text-slate-400">confidence</p>
      </div>
    </div>
  );
}
