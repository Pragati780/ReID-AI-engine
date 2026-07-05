import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import ResultsPage from "./pages/ResultsPage";
import { ThemeProvider } from "./context/ThemeContext";

/**
 * src/App.jsx
 *
 * Top-level layout + routing. Only two routes exist for this MVP:
 *   "/"                  -> upload + run a search
 *   "/results/:runId"    -> view a completed search's results
 */
export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="min-h-screen bg-slate-50 text-slate-900 transition-colors duration-300 dark:bg-slate-950 dark:text-slate-100">
          <Navbar />
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/results/:runId" element={<ResultsPage />} />
          </Routes>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}
