import './About.css';

const ENGINES = [
  { name: 'ML Classifier',       weight: '60%', desc: 'XGBoost + TF-IDF trained on BABE (3k+ expert-labelled news sentences) and AllSides headlines. Labels: Left/Right = Biased (1), Center = Neutral (0).' },
  { name: 'Emotional Intensity', weight: '10%', desc: 'VADER SentimentIntensityAnalyzer measures raw emotional charge. High negativity or positivity indicates sensationalist framing.' },
  { name: 'Subjectivity',        weight: '10%', desc: 'TextBlob opinion vs fact ratio. Editorial opinions score higher than factual reporting.' },
  { name: 'Passive Voice',       weight: '10%', desc: 'spaCy dependency parsing detects passive constructions like "was arrested" that obscure agency and responsibility.' },
  { name: 'Loaded Language',     weight:  '5%', desc: 'Curated lexicon of charged/manipulative vocabulary. Matches politically loaded terms and flags them.' },
  { name: 'Hedge Detection',     weight:  '5%', desc: 'Detects epistemic hedges ("allegedly", "reportedly") and certainty inflation ("absolutely", "devastatingly").' },
];

export default function About() {
  return (
    <div className="page-container about-page">
      {/* ── Hero ── */}
      <div className="about-hero card">
        <div className="about-hero-inner">
          <div className="about-logo">
            <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="2" y="3" width="12" height="1.5" rx="0.75" fill="white"/>
              <rect x="2" y="7" width="8" height="1.5" rx="0.75" fill="white"/>
              <rect x="2" y="11" width="10" height="1.5" rx="0.75" fill="white"/>
            </svg>
          </div>
          <div>
            <h1 className="about-title">About SLANT</h1>
            <p className="about-version">v2.0 &mdash; News Bias Analyzer</p>
          </div>
        </div>
        <p className="about-desc">
          SLANT is an open-source, AI-powered media bias analyzer that uses a 6-engine NLP pipeline
          and Google Gemini AI to detect political bias, emotional manipulation, loaded language,
          and spin in news articles. Built for readers who want to consume news critically.
        </p>
        <div className="about-links">
          <a
            href="https://github.com/Dhruv197252/News_Bias_Analyzer"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
          >
            View on GitHub
          </a>
        </div>
      </div>

      {/* ── Label System ── */}
      <div className="card">
        <h2 className="section-title-lg" style={{ marginBottom: '0.4rem' }}>Binary Label System</h2>
        <p className="about-section-desc">
          SLANT uses a binary classification model (Biased vs Neutral). Political leaning data from
          AllSides is mapped to this binary system during training:
        </p>
        <div className="label-table-wrap">
          <div className="label-table">
            <div className="label-row header">
              <span>Source Label</span><span>&rarr;</span><span>SLANT Label</span><span>Rationale</span>
            </div>
            {[
              { src: 'Left',         slant: 'Biased (1)',  note: 'Ideologically slanted content',     type: 'biased' },
              { src: 'Right',        slant: 'Biased (1)',  note: 'Ideologically slanted content',     type: 'biased' },
              { src: 'Center',       slant: 'Neutral (0)', note: 'Balanced, objective reporting',     type: 'neutral' },
              { src: 'BABE Neutral', slant: 'Neutral (0)', note: 'Expert-labelled neutral sentences', type: 'neutral' },
              { src: 'BABE Biased',  slant: 'Biased (1)',  note: 'Expert-labelled biased sentences',  type: 'biased' },
            ].map((r, i) => (
              <div key={i} className="label-row">
                <span className="label-src">{r.src}</span>
                <span className="label-arrow">&rarr;</span>
                <span className={`label-slant label-${r.type}`}>{r.slant}</span>
                <span className="label-note">{r.note}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Engines ── */}
      <div className="card">
        <h2 className="section-title-lg" style={{ marginBottom: '0.4rem' }}>The 6-Engine Pipeline</h2>
        <p className="about-section-desc">
          Every article runs through all six engines. Their outputs are weighted and combined into
          a single composite bias score (0.0 &ndash; 1.0):
        </p>
        <div className="engines-list">
          {ENGINES.map(e => (
            <div key={e.name} className="engine-row-about">
              <div className="er-left">
                <div className="er-index">{e.weight}</div>
                <div>
                  <div className="er-name">{e.name}</div>
                  <div className="er-desc">{e.desc}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="score-formula">
          <code>
            Composite = 0.60&times;ML + 0.10&times;Emotion + 0.10&times;Subjectivity + 0.10&times;Passive + 0.05&times;Lexicon + 0.05&times;Hedge
          </code>
        </div>
      </div>

      {/* ── Datasets & Tech ── */}
      <div className="grid-2">
        <div className="card">
          <h2 className="section-title-lg" style={{ marginBottom: '1.25rem' }}>Training Data</h2>
          <div className="dataset-list">
            {[
              { name: 'BABE', desc: '3,700+ sentence-level expert-labelled news sentences (biased/neutral)' },
              { name: 'AllSides', desc: 'Thousands of headlines with Left/Center/Right political leaning labels' },
              { name: 'MBFC', desc: '40+ domain-level bias and factual accuracy ratings from MediaBiasFactCheck' },
            ].map(d => (
              <div key={d.name} className="dataset-item">
                <div className="ds-name">{d.name}</div>
                <div className="ds-desc">{d.desc}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h2 className="section-title-lg" style={{ marginBottom: '1.25rem' }}>Tech Stack</h2>
          <div className="stack-list">
            {[
              { layer: 'Frontend',  tech: 'React 18 + Vite + Recharts' },
              { layer: 'Backend',   tech: 'FastAPI + Python 3.11' },
              { layer: 'ML',        tech: 'XGBoost + scikit-learn TF-IDF' },
              { layer: 'NLP',       tech: 'spaCy + NLTK VADER + TextBlob' },
              { layer: 'Scraping',  tech: 'trafilatura → newspaper3k → bs4' },
              { layer: 'AI',        tech: 'Gemini 1.5 Flash' },
            ].map(s => (
              <div key={s.layer} className="stack-item">
                <span className="stack-layer">{s.layer}</span>
                <span className="stack-tech">{s.tech}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Verdict Guide ── */}
      <div className="card">
        <h2 className="section-title-lg" style={{ marginBottom: '1.25rem' }}>Score Interpretation Guide</h2>
        <div className="verdict-guide">
          {[
            { range: '0% – 29%',  label: 'Appears Neutral',       color: 'var(--c-neutral)', desc: 'Factual, balanced reporting with little detectable bias' },
            { range: '30% – 46%', label: 'Slightly Opinionated',  color: 'var(--c-slight)',  desc: 'Mild framing effects. Generally reliable with awareness.' },
            { range: '47% – 62%', label: 'Moderate Bias',         color: 'var(--c-moderate)',desc: 'Noticeable spin, emotional language, or one-sided framing.' },
            { range: '63% – 77%', label: 'Highly Opinionated',    color: 'var(--c-high)',    desc: 'Significant bias. Strong emotional language and loaded words.' },
            { range: '78% – 100%',label: 'Extreme Bias Detected', color: 'var(--c-extreme)', desc: 'Heavy propaganda, misinformation risk, or pure opinion.' },
          ].map(v => (
            <div key={v.label} className="verdict-item">
              <div className="vi-bar" style={{ background: v.color }} />
              <div className="vi-content">
                <div className="vi-top">
                  <span className="vi-label" style={{ color: v.color }}>{v.label}</span>
                  <span className="vi-range">{v.range}</span>
                </div>
                <div className="vi-desc">{v.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Disclaimer ── */}
      <div className="alert alert-info" style={{ marginTop: '0.5rem', marginBottom: '2rem' }}>
        <strong>Disclaimer:</strong> SLANT is an AI-powered research tool. Bias scores are probabilistic
        estimates based on linguistic patterns, not definitive fact-checks. Always read multiple
        sources and apply critical thinking.
      </div>
    </div>
  );
}
