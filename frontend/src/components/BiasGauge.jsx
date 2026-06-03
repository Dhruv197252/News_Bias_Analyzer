import { useEffect, useRef } from 'react';
import './BiasGauge.css';

const SCORE_CONFIG = [
  { max: 0.30, label: 'Appears Neutral',      color: '#2eb87e', cls: 'neutral'  },
  { max: 0.47, label: 'Slightly Opinionated', color: '#4f7ac7', cls: 'slight'   },
  { max: 0.63, label: 'Moderate Bias',        color: '#c9922a', cls: 'moderate' },
  { max: 0.78, label: 'Highly Opinionated',   color: '#c94444', cls: 'high'     },
  { max: 1.00, label: 'Extreme Bias',         color: '#a52d2d', cls: 'extreme'  },
];

function getConfig(score) {
  return SCORE_CONFIG.find(c => score <= c.max) || SCORE_CONFIG[SCORE_CONFIG.length - 1];
}

export default function BiasGauge({ score = 0, verdict = '', mlLabel = 0 }) {
  const circleRef = useRef(null);
  const cfg = getConfig(score);
  const pct = Math.round(score * 100);

  const radius = 68;
  const circ   = 2 * Math.PI * radius;
  const arcLen = circ * 0.75;
  const offset = circ * 0.125;
  const fillLen = arcLen * (1 - score);

  useEffect(() => {
    const el = circleRef.current;
    if (!el) return;
    el.style.strokeDashoffset = circ;
    requestAnimationFrame(() => {
      el.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)';
      el.style.strokeDashoffset = fillLen + offset;
    });
  }, [score]);

  return (
    <div className={`gauge-wrap gauge-${cfg.cls}`}>
      <svg className="gauge-svg" viewBox="0 0 160 160" fill="none">
        {/* Track */}
        <circle
          cx="80" cy="80" r={radius}
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="10"
          strokeDasharray={`${arcLen} ${circ - arcLen}`}
          strokeDashoffset={-offset}
          strokeLinecap="round"
          transform="rotate(135 80 80)"
        />
        {/* Fill */}
        <circle
          ref={circleRef}
          cx="80" cy="80" r={radius}
          stroke={cfg.color}
          strokeWidth="10"
          strokeDasharray={`${arcLen} ${circ - arcLen}`}
          strokeDashoffset={arcLen + offset}
          strokeLinecap="round"
          transform="rotate(135 80 80)"
        />
      </svg>

      <div className="gauge-center">
        <span className="gauge-pct" style={{ color: cfg.color }}>{pct}%</span>
        <span className="gauge-sublabel">bias score</span>
      </div>

      <div className="gauge-footer">
        <div className="gauge-verdict-label" style={{ color: cfg.color }}>
          {verdict || cfg.label}
        </div>
        <div className={`gauge-ml-tag ${mlLabel === 1 ? 'biased' : 'neutral'}`}>
          ML: {mlLabel === 1 ? 'Biased' : 'Neutral'}
        </div>
      </div>
    </div>
  );
}
