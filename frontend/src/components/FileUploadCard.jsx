import { useRef } from "react";
import { UploadCloud, ImageIcon, FileVideo, X } from "lucide-react";
import Card from "./Card";

/**
 * src/components/FileUploadCard.jsx
 *
 * A single drag/click upload zone, reused for both the reference image and
 * the video. `kind` controls the accepted file type and the preview shown
 * once a file is selected.
 */
export default function FileUploadCard({ kind, file, onSelect, onClear }) {
  const inputRef = useRef(null);

  const isImage = kind === "image";
  const accept = isImage ? "image/*" : "video/*";
  const title = isImage ? "Reference Image" : "Video";
  const subtitle = isImage
    ? "The face we'll search for"
    : "The footage to search through";

  const handleFiles = (fileList) => {
    const selected = fileList?.[0];
    if (selected) onSelect(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  return (
    <Card className="flex h-full flex-col">
      <div className="mb-4 flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
          {isImage ? <ImageIcon size={16} /> : <FileVideo size={16} />}
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>
        </div>
      </div>

      {!file ? (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="flex flex-1 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-10 text-center transition hover:border-brand-400 hover:bg-brand-50/50 dark:border-slate-700 dark:bg-slate-800/40 dark:hover:border-brand-500 dark:hover:bg-brand-500/5"
        >
          <UploadCloud className="text-slate-400 dark:text-slate-500" size={28} />
          <p className="text-sm text-slate-600 dark:text-slate-300">
            <span className="font-medium text-brand-600 dark:text-brand-400">Click to upload</span> or drag and drop
          </p>
          <p className="text-xs text-slate-400 dark:text-slate-500">
            {isImage ? "JPG, PNG, or WEBP" : "MP4, MOV, AVI, MKV, or WEBM"}
          </p>
          <input
            ref={inputRef}
            type="file"
            accept={accept}
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      ) : (
        <div className="flex flex-1 flex-col gap-3">
          <div className="relative flex flex-1 items-center justify-center overflow-hidden rounded-xl bg-slate-100 dark:bg-slate-800">
            {isImage ? (
              <img
                src={URL.createObjectURL(file)}
                alt="Reference preview"
                className="max-h-56 w-full rounded-xl object-contain"
              />
            ) : (
              <video
                src={URL.createObjectURL(file)}
                controls
                className="max-h-56 w-full rounded-xl bg-black object-contain"
              />
            )}
            <button
              onClick={onClear}
              aria-label="Remove file"
              className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-slate-900/70 text-white transition hover:bg-slate-900"
            >
              <X size={14} />
            </button>
          </div>
          <div className="truncate rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
            {file.name} <span className="text-slate-400">({(file.size / (1024 * 1024)).toFixed(1)} MB)</span>
          </div>
        </div>
      )}
    </Card>
  );
}
