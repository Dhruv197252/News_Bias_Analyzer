// API client — all requests go through here
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

async function request(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);

  const res = await fetch(`${BASE_URL}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Analyze
  analyzeText: (text, headline = '') =>
    request('POST', '/api/analyze/text', { text, headline }),

  analyzeUrl: (url) =>
    request('POST', '/api/analyze/url', { url }),

  // History
  getHistory: (page = 1, per_page = 20) =>
    request('GET', `/api/history?page=${page}&per_page=${per_page}`),

  getAnalysis: (id) =>
    request('GET', `/api/history/${id}`),

  // Domains
  getDomains: () =>
    request('GET', '/api/domains'),

  // GenAI
  explain: (analysis_id, text, headline) =>
    request('POST', '/api/genai/explain', { analysis_id, text, headline }),

  rewrite: (analysis_id, text, headline) =>
    request('POST', '/api/genai/rewrite', { analysis_id, text, headline }),

  summarize: (analysis_id, text, headline) =>
    request('POST', '/api/genai/summarize', { analysis_id, text, headline }),

  // Health
  health: () => request('GET', '/api/health'),
};
