import { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom";
import { useMemo, useCallback } from "react";

// --- Custom Hook for Data Fetching ---
// This hook centralizes the logic for fetching data, handling loading, and catching errors.
const useFetchData = (url, processor) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const jsonData = await response.json();
        // The optional 'processor' function can reshape the data before setting state.
        setData(processor ? processor(jsonData) : jsonData);
      } catch (e) {
        setError(e.message);
        console.error("Failed to fetch data:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [url, processor]); // Re-run effect if URL or processor changes

  return { data, loading, error };
};

// --- UI Components ---
const LoadingSpinner = () => (
  <div className="flex justify-center items-center p-12">
    <div className="w-8 h-8 border-4 border-gray-300 border-t-gray-700 rounded-full animate-spin"></div>
  </div>
);

const ErrorMessage = ({ message }) => (
  <div className="p-6 bg-red-50 text-red-800 border border-red-200 rounded-lg">
    <p><span className="font-semibold">Error:</span> {message}</p>
  </div>
);

// --- Reusable Card/Item Components ---
const ArticleCard = ({ article }) => (
  <article className="py-6 border-b border-gray-200 last:border-b-0">
    <h2 className="text-2xl font-semibold font-serif text-gray-900 mb-2 hover:text-red-800 transition-colors">
      <a href={article.url} target="_blank" rel="noopener noreferrer">
        {article.title}
      </a>
    </h2>
    <p className="text-gray-700 mb-4 leading-relaxed">{article.excerpt}</p>
    <div className="flex flex-wrap gap-2 mb-4">
      {article.topics?.map((t, i) => (
        <span key={i} className="px-2.5 py-1 bg-gray-100 text-gray-700 rounded-full text-xs font-medium">
          {t.name}
        </span>
      ))}
    </div>
    <a href={article.url} target="_blank" rel="noopener noreferrer" className="text-red-800 font-semibold hover:underline text-sm">
      Read the full story →
    </a>
  </article>
);

const HistoryEvent = ({ event }) => (
  <div className="py-5 border-b border-gray-200 last:border-b-0">
    <p className="text-sm font-semibold text-red-800 mb-1">{event.year}</p>
    <p className="text-gray-800 leading-relaxed mb-2">{event.description}</p>
    {event.wikipedia?.length > 0 && (
      <a href={event.wikipedia[0].wikipedia} target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:underline text-sm font-medium">
        Learn more on Wikipedia →
      </a>
    )}
  </div>
);

// --- Page Components ---
function Articles() {
  const { data: articles, loading, error } = useFetchData("/api/articles");

  return (
    <div className="max-w-3xl mx-auto p-6 md:p-8">
      <h1 className="text-4xl font-bold font-serif text-gray-900 mb-6 border-b pb-4">
        The World in Brief
      </h1>
      {loading && <LoadingSpinner />}
      {error && <ErrorMessage message={error} />}
      {articles && (
        <div>
          {articles.map((a) => (
            <ArticleCard key={a.id} article={a} />
          ))}
        </div>
      )}
    </div>
  );
}

function History() {
  // By using useMemo, the URL is calculated only once when the component first mounts.
  const url = useMemo(() => {
    const now = new Date();
    const month = now.getMonth() + 1;
    const day = now.getDate();
    return `https://byabbe.se/on-this-day/${month}/${day}/events.json`;
  }, []); // Empty dependency array [] ensures this runs only once.

  // By using useCallback, this function keeps the same reference between renders,
  // preventing the useEffect in useFetchData from re-running unnecessarily.
  const processHistoryData = useCallback((data) => {
    return (data.events || []).filter(
      (e) => parseInt(e.year, 10) >= 1850
    );
  }, []); // Empty dependency array [] ensures this function is created only once.

  const { data: events, loading, error } = useFetchData(
    url,
    processHistoryData
  );

  // This can also be memoized for a minor performance gain.
  const formattedDate = useMemo(() => {
    return new Date().toLocaleDateString("en-US", {
      month: "long",
      day: "numeric",
    });
  }, []);

  return (
    <div className="max-w-3xl mx-auto p-6 md:p-8">
      <h1 className="text-4xl font-bold font-serif text-gray-900 mb-2">
        On This Day
      </h1>
      <p className="text-gray-600 mb-6 border-b pb-4">
        Key historical events from{" "}
        <span className="font-semibold">{formattedDate}</span>.
      </p>
      {loading && <LoadingSpinner />}
      {error && <ErrorMessage message={error} />}
      {events && (
        <div>
          {events.map((e, i) => (
            <HistoryEvent key={i} event={e} />
          ))}
        </div>
      )}
    </div>
  );
}

// --- Navigation and Main App ---
function NavBar() {
  const location = useLocation();
  const linkStyle = (path) =>
    `px-3 py-2 rounded-md text-sm font-semibold transition-colors ${
      location.pathname === path
        ? "bg-red-800 text-white"
        : "text-gray-700 hover:bg-gray-200"
    }`;

  return (
    <header className="bg-[#F7F5F0] border-b border-gray-200 sticky top-0 z-10">
      <nav className="max-w-3xl mx-auto flex items-center justify-between p-4">
        {/* Logo */}
        <Link to="/" className="text-xl font-bold font-serif text-red-800">
          Daylight - News that covers the whole world
        </Link>
        {/* Nav Links */}
        <div className="flex gap-2">
          <Link to="/" className={linkStyle("/")}>
            Briefing
          </Link>
          <Link to="/history" className={linkStyle("/history")}>
            On This Day
          </Link>
        </div>
      </nav>
    </header>
  );
}

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#F7F5F0] text-gray-800 font-sans">
        <NavBar />
        <main>
          <Routes>
            <Route path="/" element={<Articles />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}