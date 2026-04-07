"""
Final app.py — Complete Streamlit Dashboard
Streamlined for Single Article Analysis ONLY.
UI Update: Clean SaaS redesign — no emojis, soft palette, light + dark mode.
"""

import streamlit as st
import pandas as pd
from utils.scorer import load_engines, analyze_chunk, analyze_article
from utils.scraper import scrape_article

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "News Bias Analyzer",
    page_icon  = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>◎</text></svg>",
    layout     = "wide",
)

# ── Design System CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── CSS Variables: Light Mode ── */
    :root {
        --bg:           #f7f7f5;
        --surface:      #ffffff;
        --surface-2:    #f0f0ec;
        --border:       #e4e4de;
        --border-focus: #a0a096;
        --text-primary: #1a1a18;
        --text-secondary: #6b6b65;
        --text-muted:   #9b9b95;

        --accent-green:  #2d6a4f;
        --accent-blue:   #1e3a5f;
        --accent-orange: #b5540a;
        --accent-red:    #9b1c1c;
        --accent-grey:   #6b6b65;

        --fill-green:  #d8f3dc;
        --fill-blue:   #dbeafe;
        --fill-orange: #fde8d0;
        --fill-red:    #fde8e8;
        --fill-grey:   #ebebea;

        --track-bg:    #e4e4de;
        --radius-sm:   6px;
        --radius-md:   10px;
        --radius-lg:   16px;
        --font:        'DM Sans', sans-serif;
        --font-mono:   'DM Mono', monospace;
        --shadow:      0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md:   0 4px 12px rgba(0,0,0,0.08);
    }

    /* ── CSS Variables: Dark Mode ── */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg:             #141412;
            --surface:        #1e1e1b;
            --surface-2:      #28281f;
            --border:         #2e2e28;
            --border-focus:   #4a4a44;
            --text-primary:   #e8e8e2;
            --text-secondary: #8a8a82;
            --text-muted:     #5a5a55;

            --accent-green:  #74c69d;
            --accent-blue:   #7eb5f5;
            --accent-orange: #f4a261;
            --accent-red:    #e07070;
            --accent-grey:   #8a8a82;

            --fill-green:  #1a2e22;
            --fill-blue:   #1a2540;
            --fill-orange: #2e1e0e;
            --fill-red:    #2e1414;
            --fill-grey:   #222220;

            --track-bg:    #2e2e28;
            --shadow:      0 1px 3px rgba(0,0,0,0.3);
            --shadow-md:   0 4px 12px rgba(0,0,0,0.4);
        }
    }

    /* ── Base Reset ── */
    html, body, [class*="css"] {
        font-family: var(--font) !important;
        color: var(--text-primary) !important;
        background-color: var(--bg) !important;
    }

    /* ── App container ── */
    .main .block-container {
        padding: 2rem 3rem 4rem !important;
        max-width: 1200px !important;
    }

    /* ── Page title ── */
    h1 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.25rem !important;
    }

    h2, h3 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-primary) !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Stinfo / warning / success boxes ── */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        box-shadow: none !important;
        font-size: 0.875rem !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: var(--font) !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 0.8rem !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--border-focus) !important;
        box-shadow: none !important;
    }

    /* ── Radio ── */
    .stRadio > div {
        gap: 1.5rem !important;
    }
    .stRadio label {
        font-size: 0.875rem !important;
        color: var(--text-secondary) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--text-primary) !important;
        color: var(--bg) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-family: var(--font) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.4rem !important;
        letter-spacing: 0.01em !important;
        cursor: pointer !important;
        transition: opacity 0.15s ease !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        opacity: 0.82 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
        padding-bottom: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        color: var(--text-muted) !important;
        font-family: var(--font) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1rem !important;
        margin-bottom: -1px !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom-color: var(--text-primary) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem 0 0 0 !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }
    .streamlit-expanderContent {
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        background: var(--surface) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--text-primary) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ──────────────────────────────────────────────
       Custom Component Styles
    ────────────────────────────────────────────── */

    /* Hero score display */
    .hero-score {
        font-size: 4rem;
        font-weight: 600;
        text-align: center;
        letter-spacing: -0.04em;
        line-height: 1;
        font-family: 'DM Mono', monospace;
    }

    /* Verdict pill */
    .verdict-pill {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
    }

    /* Section card */
    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.1rem;
        margin-bottom: 0.5rem;
        box-shadow: var(--shadow);
    }
    .section-card-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .section-card-verdict {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.1rem;
    }
    .section-card-score {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-family: 'DM Mono', monospace;
        margin-bottom: 0.55rem;
    }

    /* Progress bar */
    .progress-wrap {
        background: var(--track-bg);
        border-radius: 99px;
        height: 5px;
        width: 100%;
        overflow: hidden;
    }
    .progress-fill {
        height: 5px;
        border-radius: 99px;
        transition: width 0.4s ease;
    }
    .progress-label {
        text-align: right;
        font-size: 0.72rem;
        color: var(--text-muted);
        font-family: 'DM Mono', monospace;
        margin-top: 3px;
    }

    /* Engine card */
    .engine-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 0.9rem 1rem;
        height: 100%;
        box-shadow: var(--shadow);
    }
    .engine-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    .engine-value {
        font-size: 1.6rem;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
        letter-spacing: -0.02em;
        line-height: 1;
        margin-bottom: 0.35rem;
    }
    .engine-sublabel {
        font-size: 0.75rem;
        color: var(--text-secondary);
    }

    /* Tag chips */
    .tag-chip {
        display: inline-block;
        border-radius: var(--radius-sm);
        padding: 3px 10px;
        margin: 3px 2px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid transparent;
    }
    .loaded-word-chip {
        display: inline-block;
        background: var(--fill-red);
        border: 1px solid var(--accent-red);
        border-radius: var(--radius-sm);
        padding: 3px 10px;
        margin: 3px 2px;
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--accent-red);
        font-family: 'DM Mono', monospace;
    }

    /* Highlight sentence blocks */
    .sentence-passive {
        background: var(--fill-orange);
        border-left: 3px solid var(--accent-orange);
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        font-size: 0.875rem;
        color: var(--text-primary);
        line-height: 1.5;
    }
    .sentence-gap {
        background: var(--fill-red);
        border-left: 3px solid var(--accent-red);
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        font-size: 0.875rem;
        color: var(--text-primary);
        line-height: 1.5;
    }

    /* Label pill (inline status) */
    .status-label {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        padding: 2px 8px;
        border-radius: 99px;
        margin-left: 6px;
        vertical-align: middle;
    }

    /* Section heading line */
    .section-heading {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin: 1.5rem 0 0.75rem 0;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.4rem;
    }

    /* Interpretation guide rows */
    .guide-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.875rem;
    }
    .guide-row:last-child { border-bottom: none; }
    .guide-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-top: 4px;
        flex-shrink: 0;
    }
    .guide-range {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-muted);
        min-width: 80px;
        flex-shrink: 0;
    }

</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

COLOR_MAP = {
    "green":  "#2d6a4f",
    "blue":   "#1e3a5f",
    "orange": "#b5540a",
    "red":    "#9b1c1c",
    "grey":   "#6b6b65",
}

FILL_MAP = {
    "green":  "var(--fill-green)",
    "blue":   "var(--fill-blue)",
    "orange": "var(--fill-orange)",
    "red":    "var(--fill-red)",
    "grey":   "var(--fill-grey)",
}

ENTITY_COLORS = {
    "PERSON": "var(--accent-blue)",
    "ORG":    "#6b3fa0",
    "GPE":    "var(--accent-green)",
    "NORP":   "var(--accent-orange)",
    "LAW":    "#0d7377",
    "EVENT":  "var(--accent-red)",
}

FRAMING_COLORS = {
    "Negative Framing": "var(--accent-red)",
    "Positive Framing": "var(--accent-green)",
    "Neutral Framing":  "var(--accent-grey)",
}

FRAMING_FILLS = {
    "Negative Framing": "var(--fill-red)",
    "Positive Framing": "var(--fill-green)",
    "Neutral Framing":  "var(--fill-grey)",
}

NEWS_DOMAINS = [
    "reuters.com", "cnn.com", "foxnews.com",
    "theguardian.com", "apnews.com", "nytimes.com",
    "washingtonpost.com", "politico.com", "thehill.com",
    "nbcnews.com", "npr.org", "bloomberg.com",
    "aljazeera.com", "usatoday.com", "breitbart.com",
    "huffpost.com", "nypost.com",
    "thehindu.com", "ndtv.com", "scroll.in",
    "thewire.in", "theprint.in", "firstpost.com",
    "timesofindia.com", "indianexpress.com",
    "hindustantimes.com", "tribuneindia.com",
    "deccanherald.com", "telegraphindia.com",
]

# ── Load Engines ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_engines():
    return load_engines()

# ── UI Helpers ────────────────────────────────────────────────────────────────

def _css_color(name: str) -> str:
    return COLOR_MAP.get(name, "#6b6b65")

def _css_fill(name: str) -> str:
    return FILL_MAP.get(name, "var(--fill-grey)")


def score_bar(score: float, color: str, thin: bool = False) -> None:
    hex_color = _css_color(color)
    pct = int(min(score, 1.0) * 100)
    h = "4px" if thin else "5px"
    st.markdown(f"""
    <div class="progress-wrap" style="height:{h}">
        <div class="progress-fill"
             style="width:{pct}%;background:{hex_color};height:{h}"></div>
    </div>
    <div class="progress-label">{pct}%</div>
    """, unsafe_allow_html=True)


def verdict_badge(verdict: str, color: str) -> None:
    hex_color = _css_color(color)
    fill      = _css_fill(color)
    st.markdown(f"""
    <div style="text-align:center;margin-top:0.5rem;">
        <span class="verdict-pill"
              style="background:{fill};color:{hex_color};
                     border:1px solid {hex_color}40;">
            {verdict}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_section_card(data: dict) -> None:
    label     = data.get("label", "")
    verdict   = data.get("verdict", "")
    score     = data.get("composite_score", 0.0)
    color     = data.get("color", "grey")
    hex_color = _css_color(color)
    fill      = _css_fill(color)
    pct       = int(score * 100)
    st.markdown(f"""
    <div class="section-card" style="border-top:3px solid {hex_color};">
        <div class="section-card-label">{label}</div>
        <div class="section-card-verdict">{verdict}</div>
        <div class="section-card-score">{pct}% bias score</div>
        <div class="progress-wrap">
            <div class="progress-fill"
                 style="width:{pct}%;background:{hex_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_engine_card(label: str, score: float, sublabel: str, color: str) -> None:
    hex_color = _css_color(color)
    pct = int(min(score, 1.0) * 100)
    st.markdown(f"""
    <div class="engine-card">
        <div class="engine-label">{label}</div>
        <div class="engine-value" style="color:{hex_color};">{pct}%</div>
        <div class="progress-wrap" style="margin:0.45rem 0 0.4rem 0;">
            <div class="progress-fill"
                 style="width:{pct}%;background:{hex_color};"></div>
        </div>
        <div class="engine-sublabel">{sublabel}</div>
    </div>
    """, unsafe_allow_html=True)


def render_ner_panel(ner_result: dict) -> None:
    if not ner_result or ner_result.get("total_unique", 0) == 0:
        st.info("No named entities detected.")
        return

    st.markdown(f"<div style='font-size:0.875rem;color:var(--text-secondary);margin-bottom:0.75rem'>{ner_result['summary']}</div>", unsafe_allow_html=True)

    if ner_result.get("most_mentioned"):
        for ent in ner_result["most_mentioned"]:
            frame      = ent["framing"].get("frame_label", "Neutral Framing")
            type_code  = ent.get("type_code", "")
            chip_color = FRAMING_COLORS.get(frame, "var(--accent-grey)")
            chip_fill  = FRAMING_FILLS.get(frame, "var(--fill-grey)")
            type_color = ENTITY_COLORS.get(type_code, "var(--accent-grey)")
            st.markdown(f"""
            <span class="tag-chip"
                  style="background:{chip_fill};color:{chip_color};
                         border-color:{chip_color}40;">
                {ent['text']}
            </span>
            <span class="tag-chip"
                  style="background:var(--surface-2);color:var(--text-muted);
                         border-color:var(--border);font-size:0.72rem;
                         font-family:'DM Mono',monospace;">
                {ent['type']} &middot; {ent['count']}x
            </span>
            """, unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-heading'>Negatively Framed</div>", unsafe_allow_html=True)
        neg = ner_result.get("negatively_framed", [])
        if neg:
            for e in neg:
                st.markdown(f"<div style='font-size:0.875rem;padding:2px 0;color:var(--accent-red)'>— {e}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:var(--text-muted);font-size:0.875rem'>None detected</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='section-heading'>Positively Framed</div>", unsafe_allow_html=True)
        pos = ner_result.get("positively_framed", [])
        if pos:
            for e in pos:
                st.markdown(f"<div style='font-size:0.875rem;padding:2px 0;color:var(--accent-green)'>— {e}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:var(--text-muted);font-size:0.875rem'>None detected</span>", unsafe_allow_html=True)


def render_passive_panel(passive_result: dict) -> None:
    if not passive_result:
        return

    score     = passive_result.get("score", 0.0)
    label     = passive_result.get("label", "")
    signal    = passive_result.get("bias_signal", "")
    passive_s = passive_result.get("passive_sentences", [])
    gaps      = passive_result.get("responsibility_gaps", [])
    total     = passive_result.get("total_sentences", 0)
    p_count   = passive_result.get("passive_count", 0)

    color = (
        "green"  if score < 0.15 else
        "blue"   if score < 0.30 else
        "orange" if score < 0.50 else
        "red"
    )
    score_bar(score, color)
    st.markdown(f"<div style='font-size:0.8rem;color:var(--text-secondary);margin-top:0.25rem;'><strong>{label}</strong> — {signal}</div>", unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sentences", total)
    with col2:
        st.metric("Passive Voice", p_count)
    with col3:
        st.metric("Agentless", len(gaps))

    if gaps:
        st.markdown("<div class='section-heading'>Agentless Passives — Responsibility Hidden</div>", unsafe_allow_html=True)
        for s in gaps[:5]:
            st.markdown(f'<div class="sentence-gap">{s}</div>', unsafe_allow_html=True)

    other_passives = [
        s["full_sent"] for s in passive_s
        if s["full_sent"] not in gaps
    ]
    if other_passives:
        st.markdown("<div class='section-heading'>Other Passive Constructions</div>", unsafe_allow_html=True)
        for s in other_passives[:5]:
            st.markdown(f'<div class="sentence-passive">{s}</div>', unsafe_allow_html=True)


def render_linguistic_panel(ling_dict: dict) -> None:
    if not ling_dict:
        return

    FEATURE_LABELS = {
        "passive_rate":           ("Passive Voice Rate",     "orange"),
        "lexicon_score":          ("Loaded Word Density",    "red"),
        "subjectivity":           ("Subjectivity",           "blue"),
        "emotion_intensity":      ("Emotional Intensity",    "orange"),
        "sent_length_variance":   ("Sentence Variance",      "blue"),
        "punctuation_density":    ("Exclamation / ? Density","red"),
        "first_person_rate":      ("First-Person Voice",     "orange"),
        "quote_density_inverted": ("Opinion vs Attribution", "blue"),
    }

    for key, (label, color) in FEATURE_LABELS.items():
        val = ling_dict.get(key, 0.0)
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(
                f"<div style='font-size:0.8rem;color:var(--text-secondary);"
                f"padding-top:5px;'>{label}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            score_bar(min(val, 1.0), color)


def render_loaded_words(loaded_words, category_counts):
    if not loaded_words:
        st.info("No loaded or emotionally charged words detected.")
        return
    unique_words = list(set(w for w, _ in loaded_words))
    chips = "".join(
        f'<span class="loaded-word-chip">{w}</span>'
        for w in sorted(unique_words)
    )
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
    if category_counts:
        st.markdown("<div class='section-heading'>By Category</div>", unsafe_allow_html=True)
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            st.markdown(
                f"<div style='font-size:0.875rem;padding:3px 0;"
                f"color:var(--text-secondary)'>"
                f"<strong style='color:var(--text-primary)'>"
                f"{cat.replace('_',' ').title()}</strong>"
                f" &nbsp;{count} hit{'s' if count != 1 else ''}</div>",
                unsafe_allow_html=True,
            )

# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:0.5rem;">
        <h1 style="margin-bottom:0.2rem;">News Bias Analyzer</h1>
        <p style="font-size:0.9rem;color:var(--text-secondary);margin:0;">
            Detect linguistic bias, emotional framing, and loaded language
            using a 5-engine NLP pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("What type of articles work best?"):
        st.markdown("""
        <div style="font-size:0.875rem;color:var(--text-secondary);line-height:1.7;">
        <p>This tool is mathematically optimized for <strong style="color:var(--text-primary)">
        English-language political, economic, and social journalism.</strong></p>

        <div class="guide-row">
            <div class="guide-dot" style="background:var(--accent-green);"></div>
            <div><strong style="color:var(--text-primary)">Best inputs:</strong>
            Articles about elections, government policy, supreme court rulings,
            and international relations (Reuters, AP News, CNN, Fox News).</div>
        </div>
        <div class="guide-row">
            <div class="guide-dot" style="background:var(--accent-red);"></div>
            <div><strong style="color:var(--text-primary)">Not supported:</strong>
            Sports reporting (aggressive verbs misread as partisan),
            product or movie reviews (designed to be 100% subjective),
            academic papers (scientific passive voice flagged as manipulation).</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    with st.spinner("Loading analysis engines..."):
        pipeline, sia, nlp = get_engines()

    # ── Input Mode ────────────────────────────────────────────────────────
    st.markdown("<div class='section-heading'>Input</div>", unsafe_allow_html=True)
    input_mode = st.radio(
        "Input method",
        ["Paste a URL", "Paste Raw Text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    headline   = ""
    body_text  = ""
    body_paras = []
    ready      = False

    # ── URL Mode ──────────────────────────────────────────────────────────
    if input_mode == "Paste a URL":
        url = st.text_input(
            "Article URL",
            placeholder="https://reuters.com/article/... — English political or social news only",
        )
        if st.button("Analyse Article", type="primary"):
            if not url.strip():
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching article..."):
                    scraped = scrape_article(url)

                if not scraped["success"]:
                    st.error(
                        f"Could not retrieve that URL.\n\n"
                        f"**Reason:** {scraped['error']}\n\n"
                        f"Try pasting the article text directly."
                    )
                elif scraped["word_count"] < 50:
                    st.warning(
                        "Very little text was extracted — "
                        "the site may require JavaScript rendering. "
                        "Try pasting the text directly."
                    )
                else:
                    headline   = scraped["headline"]
                    body_text  = scraped["body_text"]
                    body_paras = scraped["body_paras"]
                    ready      = True

                    is_news = any(domain in url.lower() for domain in NEWS_DOMAINS)
                    if not is_news:
                        st.warning(
                            "Domain not recognised as a news outlet. "
                            "This tool is optimised for political journalism — "
                            "results for other content types may not be meaningful."
                        )
                    else:
                        st.success("Recognised news domain — results are reliable.")

    # ── Raw Text Mode ─────────────────────────────────────────────────────
    else:
        headline = st.text_input(
            "Headline (optional)",
            placeholder="Paste the article headline here...",
        )
        body_text_input = st.text_area(
            "Article Body",
            placeholder="Paste the full article text here...",
            height=250,
        )
        if st.button("Analyse Text", type="primary"):
            if not body_text_input.strip():
                st.warning("Please paste some article text first.")
            else:
                body_text  = body_text_input
                body_paras = [
                    p.strip() for p in body_text.split("\n")
                    if len(p.strip().split()) >= 5
                ]
                if not body_paras:
                    body_paras = [body_text]
                ready = True

    # ── Results ───────────────────────────────────────────────────────────
    if ready:
        with st.spinner("Running analysis..."):
            results = analyze_article(
                headline   = headline,
                body_text  = body_text,
                body_paras = body_paras,
                pipeline   = pipeline,
                sia        = sia,
                nlp        = nlp,
            )

        overall = results["overall"]
        st.markdown("---")

        # ── Hero Score ────────────────────────────────────────────────────
        col_l, col_c, col_r = st.columns([1, 1, 1])
        with col_c:
            hex_color = _css_color(overall["color"])
            st.markdown(
                f'<div class="hero-score" style="color:{hex_color};">'
                f'{int(overall["composite_score"]*100)}%'
                f'</div>',
                unsafe_allow_html=True,
            )
            verdict_badge(overall["verdict"], overall["color"])

        st.markdown("")

        # ── Interpretation Guide ──────────────────────────────────────────
        with st.expander("How to interpret this score"):
            st.markdown("""
            <div style="font-size:0.875rem;color:var(--text-secondary);">
            <p style="margin-bottom:0.75rem;">The overall score measures
            <strong style="color:var(--text-primary)">linguistic bias intensity</strong>
            — how heavily the author guides your emotional response — not political
            direction (left vs. right).</p>
            </div>
            """, unsafe_allow_html=True)

            tiers = [
                ("#2d6a4f", "0 – 29%",  "Appears Neutral",
                 "Objective, dry, and purely informational. High use of active voice and direct quotes."),
                ("#1e3a5f", "30 – 46%", "Slightly Opinionated",
                 "Standard journalism with some editorial color. Adds narrative drama but remains factual."),
                ("#b5540a", "47 – 62%", "Moderate Bias",
                 "Leans into commentary. The journalist is quietly guiding the reader."),
                ("#9b1c1c", "63 – 77%", "Highly Opinionated",
                 "Heavily partisan writing designed to persuade or spark outrage."),
                ("#6b1c1c", "78 – 100%","Extreme Bias",
                 "Maximum emotional intensity, extreme subjectivity, high density of manipulative language."),
            ]
            for dot_color, rng, label, desc in tiers:
                st.markdown(f"""
                <div class="guide-row">
                    <div class="guide-dot" style="background:{dot_color};"></div>
                    <div class="guide-range">{rng}</div>
                    <div><strong style="color:var(--text-primary)">{label}</strong>
                    — {desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── 5 Engine Score Row ────────────────────────────────────────────
        st.markdown("<div class='section-heading'>Engine Breakdown</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)

        passive_score = overall.get("passive", {}).get("score", 0.0)
        passive_label = overall.get("passive", {}).get("label", "N/A")
        lex_score     = min(overall.get("lexicon_score", 0.0) * 10, 1.0)
        unique_hits   = overall.get("unique_loaded_words", 0)

        with c1:
            render_engine_card("ML Model", overall["ml_probability"],
                               overall["ml_label"], overall["color"])
        with c2:
            render_engine_card("Emotion", overall["emotion_intensity"],
                               overall["emotion_label"], "orange")
        with c3:
            render_engine_card("Subjectivity", overall["subjectivity_score"],
                               overall["subjectivity_label"], "blue")
        with c4:
            render_engine_card("Passive Voice", passive_score,
                               passive_label, "orange")
        with c5:
            render_engine_card("Loaded Words", lex_score,
                               f"{unique_hits} unique hits", "red")

        st.markdown("---")

        # ── Tabs ──────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Headline vs Body",
            "Narrative Arc",
            "Named Entities",
            "Passive Voice",
            "Loaded Language",
            "Linguistic Features",
            "Quote vs Opinion",
        ])

        # Tab 1 — Headline vs Body
        with tab1:
            if headline:
                hc1, hc2 = st.columns(2)
                with hc1:
                    render_section_card(results["headline"])
                    st.markdown(
                        f"<div style='font-size:0.8rem;color:var(--text-muted);"
                        f"margin-top:0.4rem;font-style:italic;'>"
                        f"{headline[:120]}{'...' if len(headline)>120 else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with hc2:
                    render_section_card(results["overall"])
            else:
                st.info("No headline provided. Enter a headline above to enable this comparison.")

        # Tab 2 — Narrative Arc
        with tab2:
            st.markdown(
                "<div style='font-size:0.875rem;color:var(--text-secondary);"
                "margin-bottom:1rem;'>How does tone shift across the article? "
                "Bias often escalates toward the end.</div>",
                unsafe_allow_html=True,
            )
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                render_section_card(results["beginning"])
            with ac2:
                render_section_card(results["middle"])
            with ac3:
                render_section_card(results["end"])

            st.markdown("")
            arc_data = pd.DataFrame({
                "Section": ["Beginning", "Middle", "End"],
                "Bias Score (%)": [
                    int(results["beginning"]["composite_score"] * 100),
                    int(results["middle"]["composite_score"]    * 100),
                    int(results["end"]["composite_score"]       * 100),
                ]
            })
            st.bar_chart(arc_data.set_index("Section"))

        # Tab 3 — Named Entities
        with tab3:
            st.markdown(
                "<div style='font-size:0.875rem;color:var(--text-secondary);"
                "margin-bottom:1rem;'>Which real-world entities appear and how "
                "are they framed? Negative framing around specific people or "
                "groups is a strong bias signal.</div>",
                unsafe_allow_html=True,
            )
            render_ner_panel(overall.get("ner", {}))

        # Tab 4 — Passive Voice
        with tab4:
            st.markdown(
                "<div style='font-size:0.875rem;color:var(--text-secondary);"
                "margin-bottom:1rem;'>Passive voice obscures who is responsible "
                "for actions. Agentless passives completely erase the actor — "
                "the strongest form of narrative manipulation.</div>",
                unsafe_allow_html=True,
            )
            render_passive_panel(overall.get("passive", {}))

        # Tab 5 — Loaded Language
        with tab5:
            st.markdown(
                "<div style='font-size:0.875rem;color:var(--text-secondary);"
                "margin-bottom:1rem;'>Words that carry emotional or political "
                "connotations beyond their literal meaning.</div>",
                unsafe_allow_html=True,
            )
            render_loaded_words(overall["loaded_words"], overall["category_counts"])

        # Tab 6 — Linguistic Features
        with tab6:
            st.markdown(
                "<div style='font-size:0.875rem;color:var(--text-secondary);"
                "margin-bottom:1rem;'>8 handcrafted signals that reveal "
                "HOW the text is structured — beyond just which words are used.</div>",
                unsafe_allow_html=True,
            )
            render_linguistic_panel(overall.get("linguistic", {}))

        # Tab 7 — Quote vs Opinion
        with tab7:
            st.markdown(
                "<div style='font-size:0.875rem;color:var(--text-secondary);"
                "margin-bottom:1rem;'>How much is the journalist's own voice "
                "versus attributed reporting? A high opinion ratio signals "
                "more editorial, less sourcing.</div>",
                unsafe_allow_html=True,
            )
            qo = overall.get("quote_opinion", {})
            if qo.get("total_sentences", 0) == 0:
                st.info("No text available for quote / opinion analysis.")
            else:
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Total Sentences",  qo["total_sentences"])
                with mc2:
                    st.metric("Quoted",           qo["quoted_sentences"])
                with mc3:
                    st.metric("Attributed",       qo["attributed_sentences"])
                with mc4:
                    st.metric("Opinion",          qo["opinion_sentences"])

                st.markdown("")
                score_bar(qo["opinion_ratio"], "orange")
                st.markdown(
                    f"<div style='font-size:0.8rem;color:var(--text-secondary);"
                    f"margin-top:0.25rem;'>"
                    f"Opinion ratio <strong style='color:var(--text-primary)'>"
                    f"{int(qo['opinion_ratio']*100)}%</strong>"
                    f" — {qo['opinion_label']}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<hr style='margin:1.25rem 0;'>", unsafe_allow_html=True)
                ex = qo.get("examples", {})
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div class='section-heading'>Quoted</div>", unsafe_allow_html=True)
                    if ex.get("quoted"):
                        for s in ex["quoted"]:
                            st.markdown(f'<div class="sentence-passive">{s}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='font-size:0.8rem;color:var(--text-muted)'>None detected.</span>", unsafe_allow_html=True)
                with c2:
                    st.markdown("<div class='section-heading'>Attributed</div>", unsafe_allow_html=True)
                    if ex.get("attributed"):
                        for s in ex["attributed"]:
                            st.markdown(f'<div class="sentence-passive">{s}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='font-size:0.8rem;color:var(--text-muted)'>None detected.</span>", unsafe_allow_html=True)
                with c3:
                    st.markdown("<div class='section-heading'>Opinion</div>", unsafe_allow_html=True)
                    if ex.get("opinion"):
                        for s in ex["opinion"]:
                            st.markdown(f'<div class="sentence-gap">{s}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='font-size:0.8rem;color:var(--text-muted)'>None detected.</span>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Full Comparison Table ─────────────────────────────────────────
        st.markdown("<div class='section-heading'>Full Section Comparison</div>", unsafe_allow_html=True)
        rows = []
        for key in ["headline", "beginning", "middle", "end", "overall"]:
            d = results[key]
            rows.append({
                "Section":      d["label"],
                "Verdict":      d["verdict"],
                "Composite":    f"{int(d['composite_score']*100)}%",
                "ML Model":     f"{int(d['ml_probability']*100)}%",
                "Emotion":      f"{int(d['emotion_intensity']*100)}%",
                "Subjectivity": f"{int(d['subjectivity_score']*100)}%",
                "Passive":      f"{int(d.get('passive',{}).get('score',0)*100)}%",
                "Entities":     d.get('ner',{}).get('total_unique', 0),
                "Opinion %":    f"{int(d.get('quote_opinion',{}).get('opinion_ratio',0)*100)}%",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── Footer ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:var(--text-muted);line-height:1.8;'>"
            "TF-IDF + Logistic Regression &nbsp;&middot;&nbsp; "
            "spaCy NER &nbsp;&middot;&nbsp; "
            "Passive Voice Detection &nbsp;&middot;&nbsp; "
            "VADER Sentiment &nbsp;&middot;&nbsp; "
            "TextBlob &nbsp;&middot;&nbsp; "
            "Custom Bias Lexicon &nbsp;&middot;&nbsp; "
            "10 Linguistic Features &nbsp;&middot;&nbsp; "
            "Quote vs Opinion Ratio &nbsp;&middot;&nbsp; "
            "Trained on BABE Dataset (3,121 expert-labelled sentences) &nbsp;&middot;&nbsp; "
            "Optimised for English-language political journalism"
            "</div>",
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()