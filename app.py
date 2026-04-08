"""
News Bias Analysis Engine — Premium Dashboard
Dark-themed analytical interface with glassmorphism, animated gauges,
and modern micro-interactions.
"""

import streamlit as st
import pandas as pd
from utils.scorer import load_engines, analyze_chunk, analyze_article
from utils.scraper import scrape_article

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="News Bias Analysis Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Dark Glassmorphism Design System ─────────────────────────────

st.markdown("""
<style>
    /* ── Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── CSS Variables ── */
    :root {
        --bg-primary: #0a0f1e;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.6);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.15);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --cyan: #06b6d4;
        --violet: #8b5cf6;
        --emerald: #10b981;
        --amber: #f59e0b;
        --rose: #f43f5e;
        --indigo: #6366f1;
        --blue: #3b82f6;
    }

    /* ── Global Resets ── */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    .stApp {
        background: linear-gradient(145deg, #0a0f1e 0%, #111827 50%, #0f172a 100%) !important;
    }

    /* ── Main Container ── */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }
    
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,15,30,0.95) 0%, rgba(17,24,39,0.95) 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }

    /* ── Headers ── */
    h1 {
        background: linear-gradient(135deg, #06b6d4, #8b5cf6, #f43f5e) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        font-size: 2.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* ── Markdown text ── */
    .stMarkdown p, .stMarkdown li {
        color: var(--text-secondary) !important;
    }
    .stMarkdown strong {
        color: var(--text-primary) !important;
    }

    /* ── Dividers ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--border-glass), var(--indigo), var(--border-glass), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Inputs (dark styled) ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--indigo) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15) !important;
    }
    .stTextInput label, .stTextArea label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"],
    .stButton > button {
        background: linear-gradient(135deg, var(--indigo), var(--violet)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }

    /* ── Radio Buttons ── */
    .stRadio > div {
        gap: 0.5rem !important;
    }
    .stRadio label {
        color: var(--text-secondary) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid var(--border-glass) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.85rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)) !important;
        color: var(--text-primary) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        box-shadow: 0 0 15px rgba(99,102,241,0.1) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99,102,241,0.2) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.08) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 0 0 10px 10px !important;
        color: var(--text-secondary) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid var(--border-glass) !important;
    }

    /* ── Info/Warning/Error boxes ── */
    .stAlert {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        color: var(--text-secondary) !important;
    }

    /* ── Caption ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    /* ══════════════════════════════════════════════════════════════════════
       CUSTOM COMPONENT STYLES
       ══════════════════════════════════════════════════════════════════════ */

    /* ── Radial Gauge ── */
    .gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.5rem 0;
    }
    .gauge-svg {
        filter: drop-shadow(0 0 20px rgba(99,102,241,0.2));
    }
    .gauge-score-text {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 2.8rem;
        fill: var(--text-primary);
    }
    .gauge-label-text {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        fill: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    @keyframes gauge-fill {
        from { stroke-dashoffset: 283; }
    }
    .gauge-arc {
        animation: gauge-fill 1.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }

    /* ── Verdict Badge ── */
    .verdict-box {
        padding: 0.6rem 1.8rem;
        border-radius: 50px;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0.75rem auto 1.5rem auto;
        display: inline-block;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .verdict-box:hover {
        transform: scale(1.03);
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    }
    .glass-card:hover {
        border-color: rgba(99,102,241,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2), 0 0 20px rgba(99,102,241,0.06);
        transform: translateY(-2px);
    }

    /* ── Section Card (Narrative) ── */
    .section-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .section-card:hover {
        border-color: rgba(99,102,241,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    /* ── Engine Card ── */
    .engine-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.2rem;
        backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .engine-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        border-radius: 14px 14px 0 0;
    }
    .engine-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }
    .engine-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .engine-weight {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .engine-score {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0.5rem 0;
        line-height: 1;
    }
    .engine-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-weight: 500;
        margin-top: 0.5rem;
    }

    /* ── Progress Bars (Dark) ── */
    .progress-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 6px;
        position: relative;
    }
    .progress-fill {
        height: 100%;
        border-radius: 8px;
        position: relative;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        animation: shimmer 2s infinite;
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* ── Loaded Word Chips ── */
    .loaded-word {
        display: inline-block;
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.25);
        border-radius: 50px;
        padding: 5px 14px;
        margin: 4px 4px 4px 0;
        font-size: 0.82rem;
        color: #fb7185;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: default;
    }
    .loaded-word:hover {
        background: rgba(244, 63, 94, 0.18);
        border-color: rgba(244, 63, 94, 0.4);
        box-shadow: 0 0 15px rgba(244, 63, 94, 0.12);
        transform: translateY(-1px);
    }

    /* ── Entity Chips ── */
    .entity-chip {
        display: inline-block;
        border-radius: 50px;
        padding: 5px 14px;
        margin: 4px 4px 4px 0;
        font-size: 0.82rem;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: default;
        backdrop-filter: blur(8px);
    }
    .entity-chip:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* ── Sentence Highlights (Dark) ── */
    .passive-sentence {
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid var(--amber);
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.9rem;
        color: var(--text-secondary) !important;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    .passive-sentence:hover {
        background: rgba(245, 158, 11, 0.12);
    }
    .gap-sentence {
        background: rgba(244, 63, 94, 0.08);
        border-left: 3px solid var(--rose);
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.9rem;
        color: var(--text-secondary) !important;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    .gap-sentence:hover {
        background: rgba(244, 63, 94, 0.12);
    }

    /* ── Hedge Sentence Highlights ── */
    .hedge-epistemic {
        background: rgba(6, 182, 212, 0.08);
        border-left: 3px solid var(--cyan);
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.9rem;
        color: var(--text-secondary) !important;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    .hedge-epistemic:hover {
        background: rgba(6, 182, 212, 0.12);
    }
    .hedge-inflation {
        background: rgba(139, 92, 246, 0.08);
        border-left: 3px solid var(--violet);
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.9rem;
        color: var(--text-secondary) !important;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    .hedge-inflation:hover {
        background: rgba(139, 92, 246, 0.12);
    }
    .hedge-both {
        background: rgba(251, 146, 60, 0.08);
        border-left: 3px solid #fb923c;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.9rem;
        color: var(--text-secondary) !important;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    .hedge-both:hover {
        background: rgba(251, 146, 60, 0.12);
    }
    .hedge-phrase-tag {
        display: inline-block;
        border-radius: 50px;
        padding: 2px 10px;
        margin: 2px 3px 2px 0;
        font-size: 0.72rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }

    /* ── Hero subtitle ── */
    .hero-subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.9rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Methodology Box ── */
    .method-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
    }

    /* ── Animate on load ── */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .animate-in {
        animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }

    /* ── Domain Notice Box ── */
    .domain-box {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(12px);
    }
    .domain-box h4 {
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem;
    }
    .domain-box p, .domain-box li {
        color: var(--text-secondary) !important;
        font-size: 0.9rem;
    }

    /* ── Footer ── */
    .footer-text {
        color: var(--text-muted);
        font-size: 0.75rem;
        line-height: 1.8;
        text-align: center;
        padding: 1rem 0;
    }
    .footer-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--indigo), transparent);
        margin: 2rem 0 1rem 0;
        opacity: 0.3;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Bar chart ── */
    .stBarChart {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

COLOR_MAP = {
    "green":  "#10b981",
    "blue":   "#3b82f6",
    "orange": "#f59e0b",
    "red":    "#ef4444",
    "grey":   "#64748b",
}

# Accent colors for dark theme (brighter versions)
ACCENT_MAP = {
    "green":  "#34d399",
    "blue":   "#60a5fa",
    "orange": "#fbbf24",
    "red":    "#fb7185",
    "grey":   "#94a3b8",
}

ENTITY_COLORS = {
    "PERSON": "#60a5fa",
    "ORG":    "#a78bfa",
    "GPE":    "#34d399",
    "NORP":   "#fbbf24",
    "LAW":    "#2dd4bf",
    "EVENT":  "#fb7185",
}

FRAMING_COLORS = {
    "Negative Framing": "#fb7185",
    "Positive Framing": "#34d399",
    "Neutral Framing":  "#94a3b8",
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

def get_gauge_color(score: float) -> str:
    """Returns a gradient-appropriate color based on score severity."""
    if score < 0.30:
        return "#10b981"   # emerald
    elif score < 0.47:
        return "#06b6d4"   # cyan
    elif score < 0.63:
        return "#f59e0b"   # amber
    elif score < 0.78:
        return "#f97316"   # orange
    else:
        return "#f43f5e"   # rose


def render_radial_gauge(score: float, label: str = "BIAS SCORE") -> None:
    """Renders an animated SVG radial gauge."""
    pct = int(score * 100)
    color = get_gauge_color(score)
    # SVG arc: radius=45, circumference ≈ 283
    circumference = 283
    offset = circumference - (circumference * score)

    st.markdown(f"""
    <div class="gauge-container animate-in">
        <svg class="gauge-svg" width="200" height="200" viewBox="0 0 100 100">
            <!-- Background arc -->
            <circle cx="50" cy="50" r="45"
                    fill="none"
                    stroke="rgba(255,255,255,0.05)"
                    stroke-width="6"
                    transform="rotate(-90 50 50)" />
            <!-- Glow filter -->
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <!-- Score arc -->
            <circle cx="50" cy="50" r="45"
                    fill="none"
                    stroke="{color}"
                    stroke-width="6"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"
                    transform="rotate(-90 50 50)"
                    filter="url(#glow)"
                    class="gauge-arc"
                    style="stroke-dashoffset: {offset};" />
            <!-- Score text -->
            <text x="50" y="47" text-anchor="middle" class="gauge-score-text">{pct}</text>
            <text x="50" y="60" text-anchor="middle" class="gauge-label-text">{label}</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)


def score_bar(score: float, color: str) -> None:
    """Renders a dark-themed progress bar with shimmer effect."""
    accent = ACCENT_MAP.get(color, "#94a3b8")
    pct = int(score * 100)
    st.markdown(f"""
    <div class="progress-bg">
        <div class="progress-fill"
             style="background: linear-gradient(90deg, {accent}cc, {accent}); width:{pct}%; box-shadow: 0 0 12px {accent}40;"></div>
    </div>
    <div style="text-align:right; font-size:0.75rem; color:#64748b; margin-top:3px;
                font-family:'JetBrains Mono',monospace; font-weight:600;">{pct}%</div>
    """, unsafe_allow_html=True)


def verdict_badge(verdict, color):
    """Renders a glowing verdict badge pill."""
    accent = ACCENT_MAP.get(color, "#94a3b8")
    st.markdown(f"""
    <div style="text-align:center;">
        <div class="verdict-box"
             style="background: {accent}15; border: 1px solid {accent}40; color: {accent};
                    box-shadow: 0 0 25px {accent}15;">
            {verdict}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_engine_card(title: str, weight: str, score: float, color: str, label: str) -> None:
    """Renders a single engine diagnostic card."""
    accent = ACCENT_MAP.get(color, "#94a3b8")
    pct = int(score * 100)
    st.markdown(f"""
    <div class="engine-card" style="border-top: 2px solid {accent};">
        <div class="engine-title">{title}</div>
        <div class="engine-weight">{weight}</div>
        <div class="engine-score" style="color: {accent};">{pct}%</div>
        <div class="progress-bg" style="height:5px;">
            <div class="progress-fill"
                 style="background: linear-gradient(90deg, {accent}99, {accent}); width:{pct}%;
                        box-shadow: 0 0 8px {accent}30;"></div>
        </div>
        <div class="engine-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_card(data: dict) -> None:
    """Renders a section card with gradient accent."""
    label     = data.get("label", "")
    verdict   = data.get("verdict", "")
    score     = data.get("composite_score", 0.0)
    color     = data.get("color", "grey")
    accent    = ACCENT_MAP.get(color, "#94a3b8")
    pct       = int(score * 100)
    st.markdown(f"""
    <div class="section-card" style="border-left: 3px solid {accent};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-weight:600; color:var(--text-primary); font-size:1rem;">{label}</span>
            <span style="color:{accent}; font-weight:700; font-size:1.3rem;
                         font-family:'JetBrains Mono',monospace;">{pct}%</span>
        </div>
        <div style="color:var(--text-muted); font-size:0.85rem; margin-bottom:10px; font-weight:500;">
            {verdict}
        </div>
        <div class="progress-bg" style="height:5px;">
            <div class="progress-fill"
                 style="background: linear-gradient(90deg, {accent}99, {accent}); width:{pct}%;
                        box-shadow: 0 0 8px {accent}25;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ner_panel(ner_result: dict) -> None:
    if not ner_result or ner_result.get("total_unique", 0) == 0:
        st.info("No named entities detected in this text.")
        return

    st.markdown(f"**{ner_result['summary']}**")
    st.markdown("")

    if ner_result.get("most_mentioned"):
        for ent in ner_result["most_mentioned"]:
            frame      = ent["framing"].get("frame_label", "Neutral Framing")
            type_code  = ent.get("type_code", "")
            chip_color = FRAMING_COLORS.get(frame, "#94a3b8")
            type_color = ENTITY_COLORS.get(type_code, "#94a3b8")

            frame_indicator = (
                "▼ " if frame == "Negative Framing" else
                "▲ " if frame == "Positive Framing" else "● "
            )

            st.markdown(f"""
            <span class="entity-chip"
                  style="background:{chip_color}15; border:1px solid {chip_color}35; color:{chip_color};">
                {frame_indicator}{ent['text']}
            </span>
            <span class="entity-chip"
                  style="background:{type_color}12; border:1px solid {type_color}30; color:{type_color}; font-size:0.72rem;">
                {ent['type']} &middot; {ent['count']}×
            </span>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Negatively Framed Entities**")
        neg = ner_result.get("negatively_framed", [])
        if neg:
            for e in neg:
                st.markdown(f"- {e}")
        else:
            st.markdown("*None detected*")
    with col2:
        st.markdown("**Positively Framed Entities**")
        pos = ner_result.get("positively_framed", [])
        if pos:
            for e in pos:
                st.markdown(f"- {e}")
        else:
            st.markdown("*None detected*")


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
    st.caption(f"**{label}** &mdash; {signal}")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sentences", total)
    with col2:
        st.metric("Passive Voice Count", p_count)
    with col3:
        st.metric("Agentless Count", len(gaps))

    if gaps:
        st.markdown("**Critical Agentless Passives (Responsibility Hidden):**")
        for s in gaps[:5]:
            st.markdown(
                f'<div class="gap-sentence">{s}</div>',
                unsafe_allow_html=True
            )

    other_passives = [
        s["full_sent"] for s in passive_s
        if s["full_sent"] not in gaps
    ]
    if other_passives:
        st.markdown("**Other Passive Constructions:**")
        for s in other_passives[:5]:
            st.markdown(
                f'<div class="passive-sentence">{s}</div>',
                unsafe_allow_html=True
            )


def render_linguistic_panel(ling_dict: dict) -> None:
    if not ling_dict:
        return

    FEATURE_LABELS = {
        "passive_rate":           ("Passive Voice Rate",     "orange"),
        "lexicon_score":          ("Loaded Word Density",    "red"),
        "subjectivity":           ("Subjectivity",           "blue"),
        "emotion_intensity":      ("Emotional Intensity",    "orange"),
        "sent_length_variance":   ("Sentence Variance",      "blue"),
        "punctuation_density":    ("Exclamation/? Density",  "red"),
        "first_person_rate":      ("First-Person Voice",     "orange"),
        "quote_density_inverted": ("Opinion vs Attribution", "blue"),
    }

    for key, (label, color) in FEATURE_LABELS.items():
        val  = ling_dict.get(key, 0.0)
        accent = ACCENT_MAP.get(color, "#94a3b8")
        pct = int(min(val, 1.0) * 100)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
            <div style="min-width:180px; font-size:0.85rem; font-weight:500; color:var(--text-secondary);">
                {label}
            </div>
            <div style="flex:1;">
                <div class="progress-bg" style="height:6px;">
                    <div class="progress-fill"
                         style="background:linear-gradient(90deg,{accent}99,{accent}); width:{pct}%;
                                box-shadow:0 0 8px {accent}25;"></div>
                </div>
            </div>
            <div style="min-width:42px; text-align:right; font-size:0.78rem; font-weight:600;
                        color:{accent}; font-family:'JetBrains Mono',monospace;">
                {pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_loaded_words(loaded_words, category_counts):
    if not loaded_words:
        st.info("No loaded or emotionally charged words detected.")
        return
    unique_words = list(set(w for w, _ in loaded_words))
    chips = "".join(
        f'<span class="loaded-word">{w}</span>'
        for w in sorted(unique_words)
    )
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("---")
    if category_counts:
        st.markdown("**By Category:**")
        for cat, count in sorted(
            category_counts.items(), key=lambda x: -x[1]
        ):
            st.markdown(
                f"- **{cat.replace('_',' ').title()}**: {count} hit(s)"
            )


def render_hedge_panel(hedge_result: dict) -> None:
    """Renders the hedging language analysis panel."""
    if not hedge_result:
        st.info("No hedging analysis data available.")
        return

    score  = hedge_result.get("hedge_score", 0.0)
    label  = hedge_result.get("hedge_label", "N/A")
    e_count = hedge_result.get("epistemic_count", 0)
    i_count = hedge_result.get("inflation_count", 0)
    e_rate  = hedge_result.get("epistemic_rate", 0.0)
    i_rate  = hedge_result.get("inflation_rate", 0.0)
    flagged = hedge_result.get("flagged_sentences", [])

    color = (
        "green"  if score < 0.10 else
        "blue"   if score < 0.25 else
        "orange" if score < 0.45 else
        "red"
    )
    score_bar(score, color)
    st.caption(f"**{label}**")
    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Hedge Score", f"{int(score * 100)}%")
    with mc2:
        st.metric("Epistemic Hedges", e_count)
    with mc3:
        st.metric("Certainty Inflators", i_count)
    with mc4:
        st.metric("Total Flagged", len(flagged))

    if not flagged:
        st.info("No significant hedging language detected.")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # Breakdown by type
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Epistemic Hedges** — *vague attribution, unverified claims*")
        epistemic_sents = [f for f in flagged if f["type"] in ("epistemic", "both")]
        if epistemic_sents:
            for fs in epistemic_sents[:5]:
                css_class = "hedge-both" if fs["type"] == "both" else "hedge-epistemic"
                phrases_html = "".join(
                    f'<span class="hedge-phrase-tag" '
                    f'style="background:rgba(6,182,212,0.15); color:#22d3ee;">'
                    f'{p}</span>'
                    for p in fs.get("epistemic_phrases", [])
                )
                st.markdown(
                    f'<div class="{css_class}">{fs["sentence"][:200]}'
                    f'<br>{phrases_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No epistemic hedges detected.")

    with col2:
        st.markdown("**Certainty Inflation** — *opinion presented as fact*")
        inflation_sents = [f for f in flagged if f["type"] in ("inflation", "both")]
        if inflation_sents:
            for fs in inflation_sents[:5]:
                css_class = "hedge-both" if fs["type"] == "both" else "hedge-inflation"
                phrases_html = "".join(
                    f'<span class="hedge-phrase-tag" '
                    f'style="background:rgba(139,92,246,0.15); color:#a78bfa;">'
                    f'{p}</span>'
                    for p in fs.get("inflation_phrases", [])
                )
                st.markdown(
                    f'<div class="{css_class}">{fs["sentence"][:200]}'
                    f'<br>{phrases_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No certainty inflation detected.")

# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.title("News Bias Analysis Engine")
    st.markdown(
        '<p class="hero-subtitle">'
        'Detect linguistic bias, emotional framing, passive voice, '
        'named entity targeting, loaded language, and hedging patterns using a '
        '6-engine NLP pipeline.'
        '</p>',
        unsafe_allow_html=True
    )

    with st.expander("Domain Constraints & Interpretation Guide"):
        st.markdown('''
        **This model is mathematically optimized for English-language political, economic, and social journalism.**
        
        **Supported Inputs (Factual vs. Manipulative):** Articles about elections, government policy, supreme court rulings, and international relations (e.g., Reuters, AP News, Fox News, CNN).
        
        **Unsupported Domains (May Cause False Positives):**
        * **Sports Reporting:** The system will misinterpret aggressive sports verbs ("destroyed", "crushed") as extreme political bias.
        * **Product/Movie Reviews:** Reviews are designed to be subjective. The system will flag them as highly opinionated.
        * **Academic Papers:** The system will flag the heavy use of scientific passive voice as manipulative framing.
        ''')

    st.markdown("---")

    with st.spinner("Initializing AI engines..."):
        pipeline, sia, nlp = get_engines()

    # ── Input Mode ────────────────────────────────────────────────────────
    st.subheader("Data Input")
    input_mode = st.radio(
        "Select input method:",
        ["Paste URL", "Paste Raw Text"],
        horizontal=True,
    )

    headline   = ""
    body_text  = ""
    body_paras = []
    ready      = False

    # ── URL Mode ──────────────────────────────────────────────────────────
    if input_mode == "Paste URL":
        url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article (English political/social news only)"
        )
        if st.button("Analyze Article", type="primary"):
            if not url.strip():
                st.warning("Please enter a valid URL.")
            else:
                with st.spinner("Extracting article content..."):
                    scraped = scrape_article(url)

                if not scraped["success"]:
                    st.error(
                        f"Failed to extract content from URL.\n\n"
                        f"Reason: {scraped['error']}\n\n"
                        f"Consider pasting the article text directly."
                    )
                elif scraped["word_count"] < 50:
                    st.warning(
                        "Insufficient text extracted. "
                        "The site may use client-side rendering or paywalls. "
                        "Consider pasting the text directly."
                    )
                else:
                    headline   = scraped["headline"]
                    body_text  = scraped["body_text"]
                    body_paras = scraped["body_paras"]
                    ready      = True

                    is_news = any(
                        domain in url.lower()
                        for domain in NEWS_DOMAINS
                    )
                    if not is_news:
                        st.info(
                            "Domain Notice: This URL does not match known major political news domains. "
                            "Results for technical, educational, or non-news content may not be meaningful."
                        )
                    else:
                        st.success(
                            "Recognized news domain. Proceeding with standard analysis."
                        )

    # ── Raw Text Mode ─────────────────────────────────────────────────────
    else:
        headline = st.text_input(
            "Headline (Optional)",
            placeholder="Enter the article headline..."
        )
        body_text_input = st.text_area(
            "Article Body",
            placeholder="Paste the complete article text here...",
            height=250,
        )
        if st.button("Analyze Text", type="primary"):
            if not body_text_input.strip():
                st.warning("Please input text for analysis.")
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
        with st.spinner("Executing 6-engine analytical pipeline..."):
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
        st.subheader("Executive Summary")

        # ── Hero Gauge ────────────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            render_radial_gauge(overall["composite_score"])
            verdict_badge(
                overall["verdict"],
                overall["color"],
            )

        st.markdown("---")

        # ── 6 Engine Score Row ────────────────────────────────────────────
        st.subheader("Engine Diagnostics")
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            render_engine_card(
                "ML Model", "Weight: 60%",
                overall["ml_probability"],
                overall["color"],
                overall["ml_label"]
            )

        with c2:
            render_engine_card(
                "Emotion", "Weight: 10%",
                overall["emotion_intensity"],
                "orange",
                overall["emotion_label"]
            )

        with c3:
            render_engine_card(
                "Subjectivity", "Weight: 10%",
                overall["subjectivity_score"],
                "blue",
                overall["subjectivity_label"]
            )

        with c4:
            passive_score = overall.get("passive_score", 0.0)
            render_engine_card(
                "Passive Voice", "Weight: 10%",
                passive_score,
                "orange",
                overall.get("passive", {}).get("label", "N/A")
            )

        with c5:
            lex_score = min(
                overall.get("lexicon_score", 0.0) * 10, 1.0
            )
            render_engine_card(
                "Loaded Words", "Weight: 5%",
                lex_score,
                "red",
                f"{overall.get('unique_loaded_words', 0)} unique instances"
            )

        with c6:
            render_engine_card(
                "Hedging", "Weight: 5%",
                overall.get("hedge_score", 0.0),
                "blue",
                overall.get("hedge_label", "N/A")
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Interpretation Guide Expander ─────────────────────────────────
        with st.expander("Score & Engine Interpretation Methodology"):
            st.markdown("""
            ### The 5-Tier Composite Score
            The overall score quantifies **Linguistic Bias** (the degree of narrative manipulation), rather than political alignment.
            * **0% – 29% (Appears Neutral):** Objective, highly informational reporting. Heavy reliance on active voice and direct quotes.
            * **30% – 46% (Slightly Opinionated):** Standard journalism with editorial narrative. Employs framing techniques while maintaining factual baseline.
            * **47% – 62% (Moderate Bias):** Leans into commentary. The author systematically guides reader interpretation of the facts.
            * **63% – 77% (Highly Opinionated):** Heavily partisan writing structurally designed to persuade rather than inform.
            * **78% – 100% (Extreme Bias):** Maximum emotional intensity, extreme subjectivity, and high density of manipulative vocabulary.

            ### The 6 Analytical Engines
            * **ML Model (60% Weight):** A Logistic Regression algorithm trained on 3,000+ expert-labeled political sentences. Detects vocabulary patterns endemic to partisan publishing.
            * **Emotion (10% Weight):** Measures the intensity of emotional language utilizing VADER sentiment heuristics.
            * **Subjectivity (10% Weight):** Calculates the ratio of authorial opinion versus verifiable factual statements.
            * **Passive Voice (10% Weight):** Penalizes structural grammar that obscures responsibility (e.g., stating *"Protesters were shot"* instead of *"Police shot protesters"*).
            * **Loaded Words (5% Weight):** Cross-references text against a custom taxonomy of manipulative adjectives and political descriptors.
            * **Hedging (5% Weight):** Detects epistemic hedges ("reportedly", "sources say") and certainty inflation ("clearly", "obviously") — language that presents unverified claims or disguises opinion as fact.
            """)

        st.markdown("---")

        # ── Tabs ──────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Headline vs Body",
            "Narrative Arc",
            "Named Entities",
            "Passive Voice",
            "Loaded Language",
            "Hedging Language",
            "Linguistic Features",
            "Quote vs Opinion",
        ])

        # Tab 1 — Headline vs Body
        with tab1:
            if headline:
                hc1, hc2 = st.columns(2)
                with hc1:
                    st.markdown("**Headline Analysis**")
                    render_section_card(results["headline"])
                    st.caption(
                        f'*"{headline[:100]}..."*'
                        if len(headline) > 100
                        else f'*"{headline}"*'
                    )
                with hc2:
                    st.markdown("**Body Analysis**")
                    render_section_card(results["overall"])
            else:
                st.info(
                    "Headline data omitted. Input a headline to activate structural comparison."
                )

        # Tab 2 — Narrative Arc
        with tab2:
            st.markdown(
                "Evaluates structural tone shifts across the document. "
                "Escalating bias scores frequently indicate editorialized conclusions."
            )
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                render_section_card(results["beginning"])
            with ac2:
                render_section_card(results["middle"])
            with ac3:
                render_section_card(results["end"])

            st.markdown("<br>", unsafe_allow_html=True)
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
                "Identifies proper nouns and structural framing surrounding them. "
                "Disproportionate negative framing around specific entities denotes targeted bias."
            )
            render_ner_panel(overall.get("ner", {}))

        # Tab 4 — Passive Voice
        with tab4:
            st.markdown(
                "Passive voice inherently obscures responsibility. "
                "Agentless passives actively remove the actor from the sentence, representing a significant narrative manipulation risk."
            )
            render_passive_panel(overall.get("passive", {}))

        # Tab 5 — Loaded Language
        with tab5:
            st.markdown(
                "Highlights vocabulary carrying emotional, political, or ideological connotations beyond literal definitions."
            )
            render_loaded_words(
                overall["loaded_words"],
                overall["category_counts"],
            )

        # Tab 6 — Hedging Language
        with tab6:
            st.markdown(
                "Detects epistemic hedges (vague attribution like *'sources say'*) and certainty inflation "
                "(opinion stated as fact via *'clearly'*, *'obviously'*). These patterns often evade other engines."
            )
            render_hedge_panel(overall.get("hedge_result", {}))

        # Tab 7 — Linguistic Features
        with tab7:
            st.markdown(
                "Quantifies 8 distinct structural patterns that reveal grammatical framing strategies."
            )
            render_linguistic_panel(overall.get("linguistic", {}))

        # Tab 8 — Quote vs Opinion
        with tab8:
            st.markdown(
                "Analyzes the ratio of authorial statements versus attributed reporting. "
                "Elevated opinion ratios correlate with editorialized, low-source content."
            )
            qo = overall.get("quote_opinion", {})
            if qo.get("total_sentences", 0) == 0:
                st.info("Insufficient text volume for structural quote analysis.")
            else:
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Total Sentences",    qo["total_sentences"])
                with mc2:
                    st.metric("Quoted Content",     qo["quoted_sentences"])
                with mc3:
                    st.metric("Attributed",         qo["attributed_sentences"])
                with mc4:
                    st.metric("Authorial Opinion",  qo["opinion_sentences"])

                st.markdown("<br>", unsafe_allow_html=True)
                score_bar(qo["opinion_ratio"], "orange")
                st.caption(
                    f"**Calculated Ratio: {int(qo['opinion_ratio']*100)}%** "
                    f"&mdash; {qo['opinion_label']}"
                )

                st.markdown("---")
                ex = qo.get("examples", {})
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Sample Quoted Content**")
                    if ex.get("quoted"):
                        for s in ex["quoted"]:
                            st.markdown(
                                f'<div class="passive-sentence">{s}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No instances found.")
                with c2:
                    st.markdown("**Sample Attributed Content**")
                    if ex.get("attributed"):
                        for s in ex["attributed"]:
                            st.markdown(
                                f'<div class="passive-sentence">{s}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No instances found.")
                with c3:
                    st.markdown("**Sample Opinion Content**")
                    if ex.get("opinion"):
                        for s in ex["opinion"]:
                            st.markdown(
                                f'<div class="gap-sentence">{s}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No instances found.")

        st.markdown("---")

        # ── Full Comparison Table ─────────────────────────────────────────
        st.subheader("Data Matrix Summary")
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
                "Passive":      f"{int(d.get('passive_score', 0.0)*100)}%",
                "Hedge":        f"{int(d.get('hedge_score', 0.0)*100)}%",
                "Entities":     d.get("ner", {}).get("total_unique", 0),
                "Opinion %":    f"{int(d.get('quote_opinion',{}).get('opinion_ratio',0)*100)}%",
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── Footer ────────────────────────────────────────────────────────
        st.markdown('<div class="footer-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="footer-text">'
            'TF-IDF + Logistic Regression (60%) · '
            'VADER Emotion Heuristics (10%) · '
            'TextBlob Subjectivity Engine (10%) · '
            'spaCy Syntactic Passive Detection (10%) · '
            'Custom Bias Lexicon Mapping (5%) · '
            'Hedge Detection Engine (5%)<br>'
            'spaCy NER Engine · 8 Linguistic Feature Matrix · '
            'Quote/Opinion Ratio Engine<br>'
            'Model Basis: BABE Dataset (3,121 expert-labelled points) · '
            'Target: English-language political journalism'
            '</div>',
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()