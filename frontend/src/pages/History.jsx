import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import './History.css';

const VERDICT_COLOR = {
  'Appears Neutral':       'var(--c-neutral)',
  'Slightly Opinionated':  'var(--c-slight)',
  'Moderate Bias':         'var(--c-moderate)',
  'Highly Opinionated':    'var(--c-high)',
  'Extreme Bias Detected': 'var(--c-extreme)',
};

function ScorePill({ score }) {
  const pct = Math.round((score || 0) * 100);
  let color = 'var(--c-neutral)';
  if (score >= 0.3 && score < 0.5)  color = 'var(--c-slight)';
  if (score >= 0.5 && score < 0.65) color = 'var(--c-moderate)';
  if (score >= 0.65)                color = 'var(--c-high)';
  
  return (
    <span className="score-pill" style={{ color, borderColor: `color-mix(in srgb, ${color} 30%, transparent)`, background: `color-mix(in srgb, ${color} 10%, transparent)` }}>
      {pct}%
    </span>
  );
}

export default function History() {
  const [data, setData]       = useState(null);
  const [page, setPage]       = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');

  useEffect(() => {
    setLoading(true);
    api.getHistory(page)
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [page]);

  return (
    <div className="page-container">
      <div className="history-header">
        <div>
          <h1 className="section-title-lg">Analysis History</h1>
          <p className="history-sub">A comprehensive log of all articles processed by SLANT.</p>
        </div>
        <Link to="/" className="btn btn-primary">New Analysis</Link>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {loading ? (
        <div className="loading-overlay">
          <div className="spinner spinner-lg" />
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Loading records...</p>
        </div>
      ) : !data || data.items.length === 0 ? (
        <div className="empty-state">
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>No analyses yet</h3>
          <p style={{ fontSize: '0.875rem', marginBottom: '1.25rem' }}>Run your first article through the pipeline to see it here.</p>
          <Link to="/" className="btn btn-secondary">Analyze Now</Link>
        </div>
      ) : (
        <>
          <div className="history-table-wrap card">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Headline</th>
                  <th>Source Domain</th>
                  <th>Score</th>
                  <th>Verdict</th>
                  <th>ML Flag</th>
                  <th>Date</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {data.items.map(item => (
                  <tr key={item.id} className="history-row">
                    <td className="hl-cell">
                      <span className="hl-text">{item.headline || '(No Headline Available)'}</span>
                    </td>
                    <td>
                      {item.domain
                        ? <span className="domain-tag">{item.domain}</span>
                        : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                    </td>
                    <td><ScorePill score={item.composite_score} /></td>
                    <td>
                      <span className="verdict-text" style={{ color: VERDICT_COLOR[item.verdict] || 'var(--text-primary)' }}>
                        {item.verdict}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${item.ml_label === 1 ? 'badge-high' : 'badge-neutral'}`}>
                        {item.ml_label === 1 ? 'Biased' : 'Neutral'}
                      </span>
                    </td>
                    <td className="date-cell">
                      {item.created_at ? new Date(item.created_at).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '—'}
                    </td>
                    <td className="action-cell">
                      <Link to={`/result/${item.id}`} className="btn btn-ghost btn-sm">View Report</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {data.total > 20 && (
            <div className="pagination">
              <button
                className="btn btn-secondary btn-sm"
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
              >Previous</button>
              <span className="page-info">
                Page {page} of {Math.ceil(data.total / 20)} <span className="page-total">({data.total} records)</span>
              </span>
              <button
                className="btn btn-secondary btn-sm"
                disabled={page * 20 >= data.total}
                onClick={() => setPage(p => p + 1)}
              >Next</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
