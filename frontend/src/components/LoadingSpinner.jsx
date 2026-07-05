import { Loader2 } from "lucide-react";

/**
 * src/components/LoadingSpinner.jsx
 *
 * Full-viewport "processing" state shown while the backend is busy running
 * the AI pipeline. The backend call is synchronous and can take a while
 * for longer videos, so this screen deliberately communicates "this is
 * normal, please wait" rather than looking like the app has stalled.
 */
export default function LoadingSpinner({ message = "Processing video..." }) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 text-center">
      <div className="relative flex h-24 w-24 items-center justify-center">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-400/30" />
        <span className="absolute inline-flex h-16 w-16 animate-pulse-slow rounded-full bg-brand-500/20" />
        <Loader2 size={40} className="relative animate-spin text-brand-600 dark:text-brand-400" strokeWidth={2.2} />
      </div>

      <div className="space-y-2">
        <p className="text-lg font-semibold text-slate-800 dark:text-slate-100">{message}</p>
        <p className="max-w-sm text-sm text-slate-500 dark:text-slate-400">
          Detecting faces, generating embeddings, and matching them against your
          reference image. Larger videos take longer -- feel free to leave this
          tab open.
        </p>
      </div>

      <div className="h-1.5 w-64 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
        <div className="h-full w-1/3 animate-[loading-bar_1.4s_ease-in-out_infinite] rounded-full bg-brand-500" />
      </div>

      {/* Indeterminate progress bar keyframes, scoped locally via a style tag
          since Tailwind's default keyframe set doesn't include this shape. */}
      <style>{`
        @keyframes loading-bar {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(120%); }
          100% { transform: translateX(320%); }
        }
      `}</style>
    </div>
  );
}
