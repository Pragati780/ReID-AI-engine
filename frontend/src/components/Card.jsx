/**
 * src/components/Card.jsx
 *
 * The base visual container used across the app: rounded corners, soft
 * shadow, consistent light/dark background. Every "card-like" surface in
 * the UI composes this instead of repeating the same Tailwind classes.
 */
export default function Card({ children, className = "" }) {
  return (
    <div
      className={`rounded-2xl border border-slate-200/80 bg-white p-6 shadow-sm shadow-slate-200/60 dark:border-slate-800 dark:bg-slate-900 dark:shadow-none ${className}`}
    >
      {children}
    </div>
  );
}
