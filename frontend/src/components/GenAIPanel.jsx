import { useState } from 'react';
import { api } from '../api/client';
import './GenAIPanel.css';

const TABS = [
  { id: 'explain',   label: 'Explain Bias',    desc: 'A detailed explanation of why this article exhibits bias and which signals triggered the assessment.' },
  { id: 'rewrite',   label: 'Neutral Rewrite', desc: 'A balanced rewrite of the article with emotionally charged language and framing removed.' },
  { id: 'summarize', label: 'Key Facts',       desc: 'Three verifiable, factual bullet points extracted from the article without editorial framing.' },
];

export default function GenAIPanel({ analysisId, text, headline }) {
  const [activeTab, setActiveTab] = useState('explain');
  const [results, setResults]     = useState({});
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState('');

  async function handleGenerate() {
    setLoading(true);
    setError('');
    try {
      let res;
      if (activeTab === 'explain')   res = await api.explain(analysisId, text, headline);
      if (activeTab === 'rewrite')   res = await api.rewrite(analysisId, text, headline);
      if (activeTab === 'summarize') res = await api.summarize(analysisId, text, headline);
      setResults(prev => ({ ...prev, [activeTab]: res }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const currentResult = results[activeTab];
  const currentTab    = TABS.find(t => t.id === activeTab);

  function getResultText(res) {
    if (!res) return null;
    return res.explanation || res.rewrite || res.summary || '';
  }

  return (
    <div className="genai-panel card">
      {/* Header */}
      <div className="genai-header">
        <div className="genai-title">
          <span>AI Analysis</span>
          <span className="gemini-badge">Gemini 1.5 Flash</span>
        </div>
        <p className="genai-header-desc">
          Powered by Google Gemini — provides contextual explanation and neutral language alternatives.
        </p>
      </div>

      {/* Tabs */}
      <div className="genai-tabs">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`genai-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Description */}
      <p className="genai-desc">{currentTab?.desc}</p>

      {/* Error */}
      {error && (
        <div className="alert alert-error" style={{ margin: '0.75rem 0' }}>
          {error}
        </div>
      )}

      {/* Result or generate button */}
      {currentResult && getResultText(currentResult) ? (
        <div className="genai-result">
          <pre className="genai-text">{getResultText(currentResult)}</pre>
          <button className="btn btn-ghost btn-sm" onClick={handleGenerate} disabled={loading}>
            {loading ? <><span className="spinner" />Regenerating...</> : 'Regenerate'}
          </button>
        </div>
      ) : (
        <button
          className="btn btn-primary genai-generate-btn"
          onClick={handleGenerate}
          disabled={loading}
        >
          {loading ? (
            <><span className="spinner" />Generating with Gemini...</>
          ) : (
            `Generate ${currentTab?.label}`
          )}
        </button>
      )}
    </div>
  );
}
