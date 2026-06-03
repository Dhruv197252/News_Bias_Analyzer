import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar  from './components/Navbar';
import Home    from './pages/Home';
import Result  from './pages/Result';
import History from './pages/History';
import Engines from './pages/Engines';
import About   from './pages/About';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Navbar />
        <main className="app-main">
          <Routes>
            <Route path="/"           element={<Home />}    />
            <Route path="/result/:id" element={<Result />}  />
            <Route path="/history"    element={<History />} />
            <Route path="/engines"    element={<Engines />} />
            <Route path="/about"      element={<About />}   />
            <Route path="*"           element={<NotFound />} />
          </Routes>
        </main>
        <footer className="app-footer">
          <span>SLANT v2.0 — Open Source News Bias Analyzer</span>
          <a
            href="https://github.com/Dhruv197252/News_Bias_Analyzer"
            target="_blank"
            rel="noopener noreferrer"
          >
            View on GitHub
          </a>
        </footer>
      </div>
    </BrowserRouter>
  );
}

function NotFound() {
  return (
    <div className="page-container">
      <div className="empty-state">
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Page Not Found
        </h2>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
          The page you are looking for does not exist.
        </p>
        <a href="/" className="btn btn-primary" style={{ marginTop: '0.5rem' }}>Return to Analyzer</a>
      </div>
    </div>
  );
}
