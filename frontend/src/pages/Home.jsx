import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import './Home.css';

const SAMPLE_TEXT = `The radical government's catastrophic new policy has triggered widespread outrage among experts who warn of devastating consequences for ordinary citizens. Critics say the reckless administration has shamefully ignored warnings from economists, pushing forward with what many call a shameful and destructive agenda that will hurt working families.`;

const ENGINES = [
  { label: 'ML Classifier',       desc: 'TF-IDF + Logistic Regression trained on 3,000+ labeled sentences' },
  { label: 'Sentiment Analysis',  desc: 'VADER emotional intensity scoring across full article' },
  { label: 'Subjectivity Engine', desc: 'TextBlob opinion-to-fact ratio detection' },
  { label: 'Passive Voice',       desc: 'Syntactic pattern detection for agency-obscuring grammar' },
  { label: 'Loaded Language',     desc: 'Curated lexicon of charged and manipulative vocabulary' },
  { label: 'Hedge Detection',     desc: 'Epistemic hedging and certainty inflation analysis' },
];

export default function Home() {
  const navigate = useNavigate();
  const [mode, setMode]         = useState('url');
  const [url, setUrl]           = useState('');
  const [text, setText]         = useState('');
  const [headline, setHeadline] = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  async function handleAnalyze() {
    setError('');
    setLoading(true);
    try {
      let result;
      if (mode === 'url') {
        if (!url.trim()) { setError('Please enter a URL.'); setLoading(false); return; }
        result = await api.analyzeUrl(url.trim());
      } else {
        if (text.trim().split(' ').length < 10) {
          setError('Please enter at least 10 words of text.');
          setLoading(false);
          return;
        }
        result = await api.analyzeText(text.trim(), headline.trim());
      }
      sessionStorage.setItem(`analysis_${result.id}`, JSON.stringify(result));
      navigate(`/result/${result.id}`);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleAnalyze();
  }

  return (
    <div className="home-page">

      {/* ── Hero ── */}
      <section className="hero fade-up">
        <div className="hero-kicker">Media Intelligence Platform</div>
        <h1 className="hero-title">
          Detect Bias in<br />
          <span className="hero-title-accent">Any News Article</span>
        </h1>
        <p className="hero-sub">
          SLANT runs every article through six independent NLP engines — ML classification,
          sentiment analysis, passive voice detection, loaded language scoring, and Gemini AI
          explanation — delivering a transparent, multi-dimensional bias assessment in seconds.
        </p>

        <div className="hero-stats">
          <div className="stat-item">
            <span className="stat-val">6</span>
            <span className="stat-label">NLP Engines</span>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <span className="stat-val">3,000+</span>
            <span className="stat-label">Training Samples</span>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <span className="stat-val">40+</span>
            <span className="stat-label">Tracked Domains</span>
          </div>
          <div className="stat-divider" />
          <div className="stat-item">
            <span className="stat-val">Open</span>
            <span className="stat-label">Source</span>
          </div>
        </div>
      </section>

      {/* ── Analyzer ── */}
      <section className="analyzer-section fade-up" style={{ animationDelay: '0.08s' }}>
        <div className="analyzer-card card">
          <div className="tabs" style={{ marginBottom: '1.5rem' }}>
            <button
              className={`tab ${mode === 'url' ? 'active' : ''}`}
              onClick={() => { setMode('url'); setError(''); }}
            >
              Analyze by URL
            </button>
            <button
              className={`tab ${mode === 'text' ? 'active' : ''}`}
              onClick={() => { setMode('text'); setError(''); }}
            >
              Analyze by Text
            </button>
          </div>

          {mode === 'url' ? (
            <div className="input-group">
              <label className="input-label">Article URL</label>
              <input
                id="url-input"
                type="url"
                value={url}
                onChange={e => setUrl(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="https://www.ndtv.com/article..."
                autoFocus
              />
              <span className="input-hint">
                Supports NDTV, The Hindu, Indian Express, BBC, Reuters, CNN, and 100+ other outlets.
              </span>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="input-group">
                <label className="input-label">Headline — Optional</label>
                <input
                  type="text"
                  value={headline}
                  onChange={e => setHeadline(e.target.value)}
                  placeholder="Article headline..."
                />
              </div>
              <div className="input-group">
                <label className="input-label">Article Text</label>
                <textarea
                  value={text}
                  onChange={e => setText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Paste article content here (minimum 10 words)..."
                  rows={8}
                  style={{ minHeight: '160px' }}
                />
              </div>
              <button
                className="btn btn-ghost btn-sm"
                style={{ alignSelf: 'flex-start' }}
                onClick={() => setText(SAMPLE_TEXT)}
              >
                Load sample biased text
              </button>
            </div>
          )}

          {error && (
            <div className="alert alert-error" style={{ marginTop: '1.25rem' }}>
              {error}
            </div>
          )}

          <div className="analyzer-actions">
            <button
              id="analyze-btn"
              className="btn btn-primary btn-lg"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? (
                <><span className="spinner" />Analyzing Article...</>
              ) : (
                'Run Bias Analysis'
              )}
            </button>
            {loading && (
              <span className="analyzing-hint">
                Scraping article and running 6 NLP engines — typically 10–20 seconds.
              </span>
            )}
          </div>

          <div className="analyzer-footer">
            <span>Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to analyze</span>
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section className="pipeline-section fade-up" style={{ animationDelay: '0.14s' }}>
        <div className="pipeline-header">
          <h2 className="pipeline-title">How the Analysis Works</h2>
          <p className="pipeline-sub">
            Every analysis runs through a sequential multi-engine pipeline designed for objectivity and transparency.
          </p>
        </div>

        <div className="pipeline-steps">
          {[
            {
              n: '01',
              title: 'Extract',
              desc: 'Multi-engine scraper (trafilatura, newspaper3k, BeautifulSoup) extracts full article text, headline, and metadata from any news URL.'
            },
            {
              n: '02',
              title: 'Classify',
              desc: 'A TF-IDF + Logistic Regression model trained on the BABE dataset applies binary bias classification with calibrated probability.'
            },
            {
              n: '03',
              title: 'Score',
              desc: 'Five auxiliary NLP engines independently score sentiment intensity, subjectivity, passive voice density, loaded vocabulary, and epistemic hedging.'
            },
            {
              n: '04',
              title: 'Synthesize',
              desc: 'A weighted composite score is calculated from all engines. Gemini 1.5 Flash provides a natural language explanation and neutral rewrite.'
            },
          ].map(step => (
            <div key={step.n} className="pipeline-step card-sm">
              <div className="step-num">{step.n}</div>
              <div className="step-body">
                <div className="step-title">{step.title}</div>
                <div className="step-desc">{step.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Engines ── */}
      <section className="engines-section fade-up" style={{ animationDelay: '0.2s' }}>
        <div className="engines-header">
          <h2 className="pipeline-title">Six Independent Engines</h2>
          <p className="pipeline-sub">
            Each engine evaluates a distinct dimension of bias. No single signal drives the verdict.
          </p>
        </div>
        <div className="engines-grid">
          {ENGINES.map((eng, i) => (
            <div key={i} className="engine-info-card card-sm">
              <div className="eic-index">{String(i + 1).padStart(2, '0')}</div>
              <div className="eic-body">
                <div className="eic-label">{eng.label}</div>
                <div className="eic-desc">{eng.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
