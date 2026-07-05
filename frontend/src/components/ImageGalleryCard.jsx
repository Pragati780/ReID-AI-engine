import { useState } from "react";
import { X, ImageOff } from "lucide-react";

/**
 * src/components/ImageGalleryCard.jsx
 *
 * A responsive grid of thumbnails with a lightweight built-in lightbox
 * (no extra dependency needed for an MVP). Used for both the "Matched
 * Frames" and "Annotated Frames" galleries -- they only differ in which
 * URL-building function and filenames are passed in.
 */
export default function ImageGalleryCard({ filenames, buildUrl, emptyLabel }) {
  const [activeImage, setActiveImage] = useState(null);

  if (!filenames || filenames.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-slate-300 py-12 text-slate-400 dark:border-slate-700">
        <ImageOff size={22} />
        <p className="text-sm">{emptyLabel}</p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
        {filenames.map((filename) => (
          <button
            key={filename}
            onClick={() => setActiveImage(filename)}
            className="group relative aspect-square overflow-hidden rounded-xl border border-slate-200 bg-slate-100 transition hover:ring-2 hover:ring-brand-400 dark:border-slate-800 dark:bg-slate-800"
          >
            <img
              src={buildUrl(filename)}
              alt={filename}
              loading="lazy"
              className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
            />
          </button>
        ))}
      </div>

      {activeImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6 backdrop-blur-sm"
          onClick={() => setActiveImage(null)}
        >
          <button
            className="absolute right-6 top-6 flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-white transition hover:bg-white/20"
            onClick={() => setActiveImage(null)}
            aria-label="Close preview"
          >
            <X size={20} />
          </button>
          <img
            src={buildUrl(activeImage)}
            alt={activeImage}
            className="max-h-[85vh] max-w-[90vw] rounded-xl object-contain shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
