import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import BiasGauge from '../components/BiasGauge';
import EngineCards from '../components/EngineCards';
// import GenAIPanel from '../components/GenAIPanel';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip
} from 'recharts';
import './Result.css';

function SectionScore({ label, score }) {
  const pct   = Math.round((score || 0) * 100);
  const color = score < 0.3 ? '#2eb87e' : score < 0.5 ? '#4f7ac7' : score < 0.65 ? '#c9922a' : '#c94444';
  return (
    <div className="section-score">
      <span className="ss-label">{label}</span>
      <div className="score-bar-track" style={{ flex: 1 }}>
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="ss-pct" style={{ color }}>{pct}%</span>
    </div>
  );
}

const LEANING_COLOR = {
  'Left': '#4f7ac7', 'Right': '#c94444', 'Center': '#2eb87e',
  'Left-Center': '#7aa3e0', 'Right-Center': '#d96a6a',
  'Extreme Right': '#a52d2d', 'Extreme Left': '#1e3d7a',
};

export default function Result() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');

  useEffect(() => {
    const cached = sessionStorage.getItem(`analysis_${id}`);
    if (cached) { setData(JSON.parse(cached)); setLoading(false); return; }
    api.getAnalysis(id)
      .then(d  => { setData(d);          setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [id]);

  if (loading) return (
    <div className="page-container">
      <div className="loading-overlay">
        <div className="spinner spinner-lg" />
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Loading analysis...</p>
      </div>
    </div>
  );

  if (error) return (
    <div className="page-container">
      <div className="alert alert-error">{error}</div>
      <button className="btn btn-ghost" style={{ marginTop: '1rem' }} onClick={() => navigate(-1)}>
        Return
      </button>
    </div>
  );

  if (!data) return null;

  const scores = data.engine_scores || {};
  const loadedWords = scores.loaded_words || [];
  const domainInfo  = data.domain_info || {};

  const radarData = [
    { subject: 'ML',          value: Math.round((scores.ml_probability    || 0) * 100) },
    { subject: 'Emotion',     value: Math.round((scores.emotion_intensity || 0) * 100) },
    { subject: 'Subjectivity',value: Math.round((scores.subjectivity_score|| 0) * 100) },
    { subject: 'Passive',     value: Math.round((scores.passive_score     || 0) * 100) },
    { subject: 'Lexicon',     value: Math.round((scores.lexicon_score     || 0) * 100) },
    { subject: 'Hedge',       value: Math.round((scores.hedge_score       || 0) * 100) },
  ];

  return (
    <div className="result-page page-container">

      {/* ── Top bar ── */}
      <div className="result-topbar">
        <button className="btn btn-ghost btn-sm" onClick={() => navigate(-1)}>
          Back
        </button>
        <div className="result-chips">
          {/* {data.domain     && <span className="chip">{data.domain}</span>} */}
          {data.scrape_engine && <span className="chip chip-dim">via {data.scrape_engine}</span>}
          {data.word_count > 0 && <span className="chip chip-dim">{data.word_count.toLocaleString()} words</span>}
        </div>
      </div>

      {/* ── Headline ── */}
      {data.headline && (
        <div className="result-headline card">
          <div className="section-title" style={{ marginBottom: '0.6rem' }}>Headline</div>
          <h1 className="result-headline-text">{data.headline}</h1>
        </div>
      )}

      {/* ── Main grid ── */}
      <div className="result-main">

        {/* Left column */}
        <div className="result-left">
          {/* Gauge */}
          <div className="card gauge-card">
            <BiasGauge
              score={data.composite_score || 0}
              verdict={data.verdict}
              mlLabel={data.ml_label}
            />
          </div>

          {/* Domain
          {domainInfo.leaning && (
            <div className="card">
              <div className="section-title" style={{ marginBottom: '1rem' }}>Source Profile</div>
              <div className="domain-grid">
                <div className="domain-item">
                  <span className="di-label">Political Leaning</span>
                  <span className="di-val" style={{ color: LEANING_COLOR[domainInfo.leaning] || 'var(--text-primary)' }}>
                    {domainInfo.leaning}
                  </span>
                </div>
                <div className="domain-item">
                  <span className="di-label">Factual Reporting</span>
                  <span className="di-val">{domainInfo.factual}</span>
                </div>
                {domainInfo.country && (
                  <div className="domain-item">
                    <span className="di-label">Country</span>
                    <span className="di-val">{domainInfo.country}</span>
                  </div>
                )}
              </div>
            </div>
          )} */}

          {/* Article sections */}
          <div className="card">
            <div className="section-title" style={{ marginBottom: '1rem' }}>Article Sections</div>
            {[
              { label: 'Headline', key: 'headline_score' },
              { label: 'Opening',  key: 'beginning_score' },
              { label: 'Middle',   key: 'middle_score' },
              { label: 'Closing',  key: 'end_score' },
            ].map(s => (
              <SectionScore key={s.key} label={s.label} score={scores[s.key]} />
            ))}
          </div>
        </div>

        {/* Right column */}
        <div className="result-right">
          {/* Engine breakdown */}
          <div className="card">
            <div className="section-title" style={{ marginBottom: '1.25rem' }}>Engine Breakdown</div>
            <EngineCards scores={scores} />
          </div>

          {/* Radar chart */}
          <div className="card">
            <div className="section-title" style={{ marginBottom: '1rem' }}>Bias Radar</div>
            <ResponsiveContainer width="100%" height={210}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.05)" />
                <PolarAngleAxis
                  dataKey="subject"
                  tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'Inter' }}
                />
                <Radar
                  name="Score"
                  dataKey="value"
                  stroke="var(--accent)"
                  fill="var(--accent)"
                  fillOpacity={0.12}
                  strokeWidth={1.5}
                />
                <Tooltip
                  contentStyle={{
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '0.8rem',
                  }}
                  formatter={(v) => [`${v}%`, 'Score']}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Loaded words */}
          {loadedWords.length > 0 && (
            <div className="card">
              <div className="section-title" style={{ marginBottom: '0.85rem' }}>Loaded Language Detected</div>
              <div className="word-chips">
                {[...new Set(loadedWords.map(w => Array.isArray(w) ? w[0] : w))].slice(0, 20).map((word, i) => (
                  <span key={i} className="word-chip">{word}</span>
                ))}
              </div>
            </div>
          )}

          {/* Writing style */}
          {scores.opinion_label && (
            <div className="card opinion-card">
              <div className="opinion-row">
                <div>
                  <div className="section-title" style={{ marginBottom: '0.3rem' }}>Writing Style</div>
                  <div className="opinion-value">{scores.opinion_label}</div>
                </div>
                <div className="opinion-pct">
                  {Math.round((scores.opinion_ratio || 0) * 100)}%
                  <span>editorial</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── GenAI Panel ── */}
      {/* <GenAIPanel
        analysisId={data.id}
        text={data.body_preview || ''}
        headline={data.headline || ''}
      /> */}

      {/* ── Article Preview ── */}
      {data.body_preview && (
        <div className="card">
          <div className="section-title" style={{ marginBottom: '0.85rem' }}>Article Preview</div>
          <p className="body-preview">{data.body_preview}</p>
        </div>
      )}
    </div>
  );
}
