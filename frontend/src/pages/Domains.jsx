import { useState, useEffect } from 'react';
import { api } from '../api/client';
import './Domains.css';

const LEANING_ORDER = [
  'Extreme Left', 'Left', 'Left-Center', 'Center',
  'Right-Center', 'Right', 'Extreme Right'
];

const LEANING_COLOR = {
  'Extreme Left':  '#1e3d7a',
  'Left':          '#4f7ac7',
  'Left-Center':   '#7aa3e0',
  'Center':        '#2eb87e',
  'Right-Center':  '#d96a6a',
  'Right':         '#c94444',
  'Extreme Right': '#a52d2d',
};

const FACTUAL_COLOR = {
  'Very High': '#2eb87e',
  'High':      '#3bc58b',
  'Mostly':    '#c9922a',
  'Mixed':     '#d97d25',
  'Low':       '#c94444',
  'Very Low':  '#a52d2d',
};

function DomainCard({ domain, info }) {
  const lc = LEANING_COLOR[info.leaning] || 'var(--text-muted)';
  const fc = FACTUAL_COLOR[info.factual] || 'var(--text-muted)';

  return (
    <div className="domain-card-item">
      <div className="dci-top">
        <div className="dci-name">{info.name || domain}</div>
        <div className="dci-url">{domain}</div>
      </div>
      <div className="dci-tags">
        <span className="dci-tag" style={{ color: lc, borderColor: `color-mix(in srgb, ${lc} 30%, transparent)`, background: `color-mix(in srgb, ${lc} 10%, transparent)` }}>
          {info.leaning || 'Unknown Leaning'}
        </span>
        {info.factual && (
          <span className="dci-tag" style={{ color: fc, borderColor: `color-mix(in srgb, ${fc} 30%, transparent)`, background: `color-mix(in srgb, ${fc} 10%, transparent)` }}>
            {info.factual} Factual
          </span>
        )}
        {info.country === 'IN' && (
          <span className="dci-tag country-tag">IN</span>
        )}
      </div>
      {info.region && (
        <div className="dci-region">{info.region} &middot; {info.type}</div>
      )}
    </div>
  );
}

export default function Domains() {
  const [domains, setDomains] = useState([]);
  const [filter, setFilter]   = useState('All');
  const [search, setSearch]   = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');

  useEffect(() => {
    api.getDomains()
      .then(d => { setDomains(d.domains || []); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const leanings = ['All', ...LEANING_ORDER.filter(l =>
    domains.some(d => d.leaning === l)
  )];

  const filtered = domains.filter(d => {
    const matchFilter = filter === 'All' || d.leaning === filter;
    const matchSearch = search === '' ||
      (d.domain || '').toLowerCase().includes(search.toLowerCase()) ||
      (d.name  || '').toLowerCase().includes(search.toLowerCase());
    return matchFilter && matchSearch;
  });

  const grouped = {};
  LEANING_ORDER.forEach(l => {
    const items = filtered.filter(d => d.leaning === l);
    if (items.length) grouped[l] = items;
  });

  return (
    <div className="page-container">
      <div className="domains-header">
        <div>
          <h1 className="section-title-lg">Domain Intelligence</h1>
          <p className="domains-sub">Database of {domains.length} tracked news sources with institutional leaning and factual accuracy ratings.</p>
        </div>
      </div>

      {/* Spectrum bar */}
      <div className="spectrum-bar card">
        <div className="spectrum-label left-label">Left</div>
        <div className="spectrum-track">
          {LEANING_ORDER.map(l => {
            const count = domains.filter(d => d.leaning === l).length;
            if (!count) return null;
            return (
              <div
                key={l}
                className={`spectrum-seg ${filter === l ? 'active' : ''}`}
                style={{ background: LEANING_COLOR[l], flex: count }}
                onClick={() => setFilter(filter === l ? 'All' : l)}
                title={`${l}: ${count} sources`}
              >
                <span className="seg-count">{count}</span>
              </div>
            );
          })}
        </div>
        <div className="spectrum-label right-label">Right</div>
      </div>

      {/* Filters */}
      <div className="domains-filters">
        <input
          type="text"
          placeholder="Search by name or domain..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="domain-search"
        />
        <div className="filter-chips">
          {leanings.map(l => (
            <button
              key={l}
              className={`filter-chip ${filter === l ? 'active' : ''}`}
              style={filter === l && l !== 'All' ? {
                color: LEANING_COLOR[l],
                borderColor: `color-mix(in srgb, ${LEANING_COLOR[l]} 40%, transparent)`,
                background: `color-mix(in srgb, ${LEANING_COLOR[l]} 15%, transparent)`,
              } : {}}
              onClick={() => setFilter(l)}
            >
              {l}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {loading ? (
        <div className="loading-overlay">
          <div className="spinner spinner-lg" />
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Loading domain database...</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>No domains found</h3>
          <p style={{ fontSize: '0.875rem' }}>Adjust your search or filter criteria.</p>
        </div>
      ) : filter === 'All' ? (
        <div className="domains-grouped">
          {Object.entries(grouped).map(([leaning, items]) => (
            <div key={leaning} className="leaning-group">
              <div className="leaning-group-header">
                <span
                  className="leaning-dot"
                  style={{ background: LEANING_COLOR[leaning] }}
                />
                <span className="leaning-name" style={{ color: LEANING_COLOR[leaning] }}>
                  {leaning}
                </span>
                <span className="leaning-count">{items.length} sources</span>
              </div>
              <div className="domains-grid">
                {items.map(d => (
                  <DomainCard key={d.domain} domain={d.domain} info={d} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="domains-grid">
          {filtered.map(d => (
            <DomainCard key={d.domain} domain={d.domain} info={d} />
          ))}
        </div>
      )}
    </div>
  );
}
