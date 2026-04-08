"""
Final app.py — Complete Streamlit Dashboard
Streamlined for Single Article Analysis ONLY.
UI Update: Industry-level professional UI, emojis removed, modern CSS.
"""

import streamlit as st
import pandas as pd
from utils.scorer import load_engines, analyze_chunk, analyze_article
from utils.scraper import scrape_article

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "News Bias Analysis Engine",
    layout     = "wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Base typography and clean dashboard look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Score Typography */
    .big-score {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        color: #0f172a;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    
    /* Verdict Badge */
    .verdict-box {
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        text-align: center;
        font-size: 1.05rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Section Cards */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: box-shadow 0.2s ease;
    }
    .section-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Chips and Tags */
    .loaded-word {
        display: inline-block;
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 4px;
        padding: 4px 12px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        color: #b91c1c;
        font-weight: 600;
    }
    .entity-chip {
        display: inline-block;
        border-radius: 4px;
        padding: 4px 10px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Sentence Highlights */
    .passive-sentence {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.95rem;
        color: #334155 !important;
        line-height: 1.5;
    }
    .gap-sentence {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.95rem;
        color: #334155 !important;
        line-height: 1.5;
    }
    
    /* Progress Bars */
    .progress-bg {
        background: #f1f5f9;
        border-radius: 6px;
        height: 12px;
        width: 100%;
        overflow: hidden;
        margin-top: 4px;
    }
    .progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

# Professional Corporate Palette
COLOR_MAP = {
    "green":  "#10b981", # Emerald
    "blue":   "#3b82f6", # Royal Blue
    "orange": "#f59e0b", # Amber
    "red":    "#ef4444", # Crimson
    "grey":   "#64748b", # Slate
}

ENTITY_COLORS = {
    "PERSON": "#3b82f6", # Blue
    "ORG":    "#8b5cf6", # Purple
    "GPE":    "#10b981", # Emerald
    "NORP":   "#f59e0b", # Amber
    "LAW":    "#14b8a6", # Teal
    "EVENT":  "#ef4444", # Red
}

FRAMING_COLORS = {
    "Negative Framing": "#ef4444",
    "Positive Framing": "#10b981",
    "Neutral Framing":  "#64748b",
}

NEWS_DOMAINS = [
    # US outlets
    "reuters.com", "cnn.com", "foxnews.com",
    "theguardian.com", "apnews.com", "nytimes.com",
    "washingtonpost.com", "politico.com", "thehill.com",
    "nbcnews.com", "npr.org", "bloomberg.com",
    "aljazeera.com", "usatoday.com", "breitbart.com",
    "huffpost.com", "nypost.com",
    # Indian outlets
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

def score_bar(score: float, color: str) -> None:
    hex_color = COLOR_MAP.get(color, "#64748b")
    pct = int(score * 100)
    st.markdown(f"""
    <div class="progress-bg">
        <div class="progress-fill" style="background:{hex_color}; width:{pct}%;"></div>
    </div>
    <div style="text-align:right; font-size:0.8rem; color:#64748b; margin-top:2px; font-weight:600;">{pct}%</div>
    """, unsafe_allow_html=True)


def verdict_badge(verdict, color):
    hex_color = COLOR_MAP.get(color, "#64748b")
    st.markdown(f"""
    <div class="verdict-box"
         style="background:{hex_color}15; border:1px solid {hex_color}40; color:{hex_color};">
        {verdict}
    </div>
    """, unsafe_allow_html=True)


def render_section_card(data: dict) -> None:
    label     = data.get("label", "")
    verdict   = data.get("verdict", "")
    score     = data.get("composite_score", 0.0)
    color     = data.get("color", "grey")
    hex_color = COLOR_MAP.get(color, "#64748b")
    pct       = int(score * 100)
    st.markdown(f"""
    <div class="section-card" style="border-left: 4px solid {hex_color};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-weight:600; color:#0f172a; font-size:1.05rem;">{label}</span>
            <span style="color:{hex_color}; font-weight:800; font-size:1.15rem;">{pct}%</span>
        </div>
        <div style="color:#64748b; font-size:0.9rem; margin-bottom:10px; font-weight:500;">
            {verdict}
        </div>
        <div class="progress-bg" style="height:6px;">
            <div class="progress-fill" style="background:{hex_color}; width:{pct}%;"></div>
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
            chip_color = FRAMING_COLORS.get(frame, "#64748b")
            type_color = ENTITY_COLORS.get(type_code, "#64748b")
            
            frame_indicator = (
                "[Negative]" if frame == "Negative Framing" else
                "[Positive]" if frame == "Positive Framing" else ""
            )
            
            st.markdown(f"""
            <span class="entity-chip"
                  style="background:{chip_color}15; border:1px solid {chip_color}40; color:{chip_color};">
                {frame_indicator} {ent['text']}
            </span>
            <span class="entity-chip"
                  style="background:{type_color}15; border:1px solid {type_color}40; color:{type_color}; font-size:0.75rem;">
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

    score   = passive_result.get("score", 0.0)
    label   = passive_result.get("label", "")
    signal  = passive_result.get("bias_signal", "")
    passive_s = passive_result.get("passive_sentences", [])
    gaps    = passive_result.get("responsibility_gaps", [])
    total   = passive_result.get("total_sentences", 0)
    p_count = passive_result.get("passive_count", 0)

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
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(
                f"<div style='font-size:0.9rem; padding-top:4px; font-weight:500; color:#475569;'>{label}</div>",
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

# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.title("News Bias Analysis Engine")
    st.markdown(
        "Detect linguistic bias, emotional framing, passive voice, "
        "named entity targeting, and loaded language using a "
        "5-engine NLP pipeline."
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

                    # ── Domain Detection ──────────────────────────────
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
        with st.spinner("Executing 5-engine analytical pipeline..."):
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

        # ── Hero Score ────────────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f'<div class="big-score">'
                f'{int(overall["composite_score"]*100)}%'
                f'</div>',
                unsafe_allow_html=True,
            )
            verdict_badge(
                overall["verdict"],
                overall["color"],
            )

        st.markdown("---")

        # ── 5 Engine Score Row ────────────────────────────────────────────
        st.subheader("Engine Diagnostics")
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown("**ML Model**")
            score_bar(overall["ml_probability"], overall["color"])
            st.caption(overall["ml_label"])

        with c2:
            st.markdown("**Emotion**")
            score_bar(overall["emotion_intensity"], "orange")
            st.caption(overall["emotion_label"])

        with c3:
            st.markdown("**Subjectivity**")
            score_bar(overall["subjectivity_score"], "blue")
            st.caption(overall["subjectivity_label"])

        with c4:
            st.markdown("**Passive Voice**")
            passive_score = overall.get(
                "passive", {}
            ).get("score", 0.0)
            score_bar(passive_score, "orange")
            st.caption(
                overall.get("passive", {}).get("label", "N/A")
            )

        with c5:
            st.markdown("**Loaded Words**")
            lex_score = min(
                overall.get("lexicon_score", 0.0) * 10, 1.0
            )
            score_bar(lex_score, "red")
            st.caption(
                f"{overall.get('unique_loaded_words', 0)} "
                f"unique instances"
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

            ### The Analytical Engines
            * **ML Model (65% Weight):** A Logistic Regression algorithm trained on 3,000+ expert-labeled political sentences. Detects vocabulary patterns endemic to partisan publishing.
            * **Emotion (10% Weight):** Measures the intensity of emotional language utilizing VADER sentiment heuristics.
            * **Subjectivity (10% Weight):** Calculates the ratio of authorial opinion versus verifiable factual statements.
            * **Passive Voice (10% Weight):** Penalizes structural grammar that obscures responsibility (e.g., stating *"Protesters were shot"* instead of *"Police shot protesters"*).
            * **Loaded Words (5% Weight):** Cross-references text against a custom taxonomy of manipulative adjectives and political descriptors.
            """)

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

        # Tab 6 — Linguistic Features
        with tab6:
            st.markdown(
                "Quantifies 8 distinct structural patterns that reveal grammatical framing strategies."
            )
            render_linguistic_panel(overall.get("linguistic", {}))

        # Tab 7 — Quote vs Opinion 
        with tab7:
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
                "Verdict":      d['verdict'],
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
        st.caption(
            "Analysis Architecture: TF-IDF + Logistic Regression | "
            "spaCy NER Engine | Syntactic Passive Detection | "
            "VADER Sentiment Heuristics | TextBlob Subjectivity Engine | "
            "Custom Bias Lexicon Mapping | 10 Linguistic Feature Matrix | "
            "Quote/Opinion Ratio Engine | "
            "Model Basis: BABE Dataset (3,121 expert-labelled points) | "
            "Target Specification: English-language political journalism."
        )

if __name__ == "__main__":
    main()