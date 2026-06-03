import './EngineCards.css';

const ENGINES = [
  { key: 'ml_probability',     label: 'ML Classifier',      desc: 'Trained on 3,000+ expert-labeled news sentences' },
  { key: 'emotion_intensity',  label: 'Emotional Intensity', desc: 'VADER sentiment emotional charge across full text' },
  { key: 'subjectivity_score', label: 'Subjectivity',        desc: 'TextBlob opinion vs. factual statement ratio' },
  { key: 'passive_score',      label: 'Passive Voice',       desc: 'Grammar patterns that obscure agency and accountability' },
  { key: 'lexicon_score',      label: 'Loaded Language',     desc: 'Frequency of charged and manipulative vocabulary' },
  { key: 'hedge_score',        label: 'Hedge Language',      desc: 'Epistemic hedges and unwarranted certainty signals' },
];

function getColor(val) {
  if (val < 0.3)  return '#2eb87e';
  if (val < 0.5)  return '#4f7ac7';
  if (val < 0.65) return '#c9922a';
  return '#c94444';
}

export default function EngineCards({ scores = {} }) {
  return (
    <div className="engine-grid">
      {ENGINES.map((eng, i) => {
        const val   = scores[eng.key] ?? 0;
        const pct   = Math.round(val * 100);
        const color = getColor(val);
        return (
          <div
            key={eng.key}
            className="engine-row"
            style={{ animationDelay: `${i * 0.06}s` }}
          >
            <div className="eng-header">
              <div className="eng-meta">
                <span className="eng-label">{eng.label}</span>
                <span className="eng-desc">{eng.desc}</span>
              </div>
              <span className="eng-pct" style={{ color }}>{pct}%</span>
            </div>
            <div className="eng-bar-track">
              <div
                className="eng-bar-fill"
                style={{ width: `${Math.min(val * 100, 100)}%`, background: color }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
