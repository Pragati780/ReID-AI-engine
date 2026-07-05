import { Link } from "react-router-dom";
import { Search, Moon, Sun } from "lucide-react";
import { useTheme } from "../context/ThemeContext";

/**
 * src/components/Navbar.jsx
 *
 * Top navigation bar shown on every page. Purely presentational -- it
 * doesn't know anything about search state or results.
 */
export default function Navbar() {
  const { isDark, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/80 backdrop-blur-md dark:border-slate-800 dark:bg-slate-950/80">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-2 text-slate-900 dark:text-white">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-white shadow-md shadow-brand-600/30">
            <Search size={18} strokeWidth={2.5} />
          </span>
          <span className="text-lg font-semibold tracking-tight">Person Re-Identification</span>
        </Link>

        <button
          onClick={toggleTheme}
          aria-label="Toggle dark mode"
          className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
        >
          {isDark ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </header>
  );
}
