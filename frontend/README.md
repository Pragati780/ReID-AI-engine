# Face Search — Frontend (React + Vite + Tailwind)

A modern, responsive web UI for the Face Search backend. This app never
imports or contains any AI logic — it communicates with the FastAPI backend
exclusively through `src/services/api.js`.

```
frontend/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── .env.example
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── index.css
    ├── context/
    │   └── ThemeContext.jsx      # dark mode toggle
    ├── components/
    │   ├── Navbar.jsx
    │   ├── Card.jsx
    │   ├── StatCard.jsx
    │   ├── FileUploadCard.jsx
    │   ├── LoadingSpinner.jsx
    │   ├── TimelineItem.jsx
    │   └── ImageGalleryCard.jsx
    ├── pages/
    │   ├── HomePage.jsx
    │   └── ResultsPage.jsx
    └── services/
        └── api.js                # the ONLY module that knows backend URLs
```

## Installation

```bash
cd frontend
npm install
cp .env.example .env      # adjust VITE_API_BASE_URL if your backend isn't on localhost:8000
```

## Running (development)

```bash
npm run dev
```

Opens at http://localhost:5173 by default. The backend must already be
running (see `backend/README.md`) for searches to work — CORS is
pre-configured on the backend for this exact origin.

## Building for production

```bash
npm run build      # outputs static files to dist/
npm run preview    # serve the production build locally to sanity-check it
```

Deploy the contents of `dist/` to any static host (Nginx, Vercel, Netlify,
S3 + CloudFront, etc.), and set `VITE_API_BASE_URL` at build time to point
at your deployed backend's URL.

## Pages

- **Home (`/`)** — upload a reference image and a video, preview both,
  and run the search. While the backend is processing, this page shows a
  full-screen loading state (spinner + indeterminate progress bar +
  "Processing video...") instead of navigating away, since the search
  request blocks until the pipeline finishes.
- **Results (`/results/:runId`)** — reference image, headline stats
  (person found, best similarity, number of appearances, processing time),
  a chronological timeline of appearances, matched-frame and
  annotated-frame galleries (with a built-in lightbox), and JSON/CSV
  download links. Works both right after a search (using data passed via
  router state) and when loaded directly by URL (fetches
  `GET /api/result/:runId` itself).

## How the frontend talks to the backend

Every network call goes through `src/services/api.js` — no component ever
builds a backend URL by hand. That module exports:

- `searchFaces(referenceFile, videoFile)` → `POST /api/search`
- `getResult(runId)` → `GET /api/result/:runId`
- `matchedFrameUrl(runId, filename)`, `annotatedFrameUrl(runId, filename)`,
  `referenceImageUrl(runId)` → build `<img src="...">` URLs for the three
  file-serving endpoints
- `jsonDownloadUrl(runId)`, `csvDownloadUrl(runId)` → build `<a href="...">`
  URLs for the two download endpoints

The API base URL comes from `VITE_API_BASE_URL` (see `.env.example`), so
switching environments (local, staging, production) never requires a code
change — only a different `.env` value.

## Dark mode

Toggled via the moon/sun icon in the navbar. Implemented with Tailwind's
`darkMode: "class"` strategy and a small `ThemeContext` that persists the
choice in `localStorage` and respects the OS's `prefers-color-scheme` on
first load.
