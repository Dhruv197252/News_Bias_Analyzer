"""
Step 6: Composite Scoring Algorithm
--------------------------------------
The master orchestrator. Takes a block of text and runs it through
ALL six engines, then combines results into one final score.

Weighting:
  ML Model Probability      → 60%  (trained on real journalism patterns)
  VADER Emotional Intensity → 10%  (raw emotional charge)
  TextBlob Subjectivity     → 10%  (opinion vs fact language)
  Passive Voice Score       → 10%  (agency-obscuring grammar)
  Lexicon Score             →  5%  (loaded/manipulative vocabulary)
  Hedge Score               →  5%  (epistemic hedges + certainty inflation)

Why these weights?
  The ML model has the most signal — it learned from 3000+ expert-labelled
  real news sentences. The auxiliary engines catch what ML misses
  (emotional framing, opinionated language, passive constructions,
  loaded vocabulary, hedging language) but are noisier signals, so they
  get smaller weights.
"""

import re
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from utils.bias_lexicon  import scan_text
from utils.ml_engine     import load_model, predict_bias
from utils.nlp_engines   import run_auxiliary_engines, ensure_vader_ready
from utils.ner_engine    import load_nlp, extract_entities
from utils.passive_voice   import analyze_passive_voice
from utils.enhanced_ml     import LinguisticFeatureExtractor
from utils.hedge_detector  import analyze_hedging

# ── 1. Weights ────────────────────────────────────────────────────────────────

WEIGHTS = {
    "ml":          0.60,
    "emotion":     0.10,
    "subjectivity":0.10,
    "passive":     0.10,
    "lexicon":     0.05,
    "hedge":       0.05,
}

# ── 2. Label Thresholds ───────────────────────────────────────────────────────

def composite_label(score: float) -> dict:
    """
    Maps a 0.0–1.0 composite score to a plain-English verdict.
    Thresholds match the UI Interpretation Guide exactly.

    Returns
    -------
    dict:
        label : str   — human-readable category
        color : str   — for Streamlit UI colouring (Step 7)
        emoji : str   — visual indicator
    """
    if score < 0.30:
        return {"label": "Appears Neutral",        "color": "green",  "emoji": "🟢"}
    elif score < 0.47:
        return {"label": "Slightly Opinionated",   "color": "blue",   "emoji": "🔵"}
    elif score < 0.63:
        return {"label": "Moderate Bias",          "color": "orange", "emoji": "🟠"}
    elif score < 0.78:
        return {"label": "Highly Opinionated",     "color": "red",    "emoji": "🔴"}
    else:
        return {"label": "Extreme Bias Detected",  "color": "red",    "emoji": "🚨"}


# ── 3. Core Chunk Analyser ────────────────────────────────────────────────────

def analyze_chunk(
    text:     str,
    pipeline,                        # fitted sklearn pipeline
    sia:      SentimentIntensityAnalyzer,
    label:    str = "chunk",
    nlp=None,                        # spaCy model for passive voice
) -> dict:
    """
    Master analysis function. Runs all five engines on a text block
    and returns a single unified result dictionary.

    Parameters
    ----------
    text     : str  — any block of text (headline, paragraph, full article)
    pipeline : fitted sklearn Pipeline from ml_engine.py
    sia      : VADER SentimentIntensityAnalyzer instance
    label    : str  — name for this chunk (e.g. "Headline", "Body", "Beginning")
    nlp      : spaCy Language model (optional — needed for passive voice engine)

    Returns
    -------
    Unified dict with all scores, labels, and detected loaded words.
    """
    if not text or not text.strip():
        return _empty_result(label)

    # ── Engine 1: Lexicon Scanner ─────────────────────────────────────────────
    lexicon_result = scan_text(text)
    # Normalise raw lexicon score (0.0–1.0); mirrors the *10 cap used in app.py
    lexicon_score  = min(lexicon_result["naive_score"] * 10, 1.0)

    # ── Engine 2: ML Model ────────────────────────────────────────────────────
    ml_result = predict_bias(text, pipeline)
    ml_prob   = ml_result["probability"]          # 0.0 – 1.0

    # ── Engine 3 & 4: VADER + TextBlob ───────────────────────────────────────
    aux       = run_auxiliary_engines(text, sia)
    emotion   = aux["emotional_intensity"]["intensity"]   # 0.0 – 1.0
    subj      = aux["subjectivity"]["score"]              # 0.0 – 1.0

    # ── Engine 5: Passive Voice ───────────────────────────────────────────────
    if nlp:
        passive_result = analyze_passive_voice(text, nlp)
        passive_score  = passive_result.get("score", 0.0)
    else:
        passive_result = {}
        passive_score  = 0.0

    # ── Engine 6: Hedge Detection ────────────────────────────────────────────
    hedge_result = analyze_hedging(text)
    hedge_score  = hedge_result.hedge_score

    # ── Composite Score ───────────────────────────────────────────────────────
    composite = (
        WEIGHTS["ml"]           * ml_prob       +
        WEIGHTS["emotion"]      * emotion       +
        WEIGHTS["subjectivity"] * subj          +
        WEIGHTS["passive"]      * passive_score +
        WEIGHTS["lexicon"]      * lexicon_score +
        WEIGHTS["hedge"]        * hedge_score
    )
    composite = round(composite, 4)

    verdict = composite_label(composite)

    return {
        # Identity
        "label":           label,
        "text_preview":    text[:120] + "..." if len(text) > 120 else text,

        # Final composite
        "composite_score": composite,
        "verdict":         verdict["label"],
        "color":           verdict["color"],
        "emoji":           verdict["emoji"],

        # Individual engine scores (for dashboard breakdown)
        "ml_probability":  round(ml_prob, 4),
        "ml_label":        ml_result["label"],

        "emotion_intensity": round(emotion, 4),
        "emotion_label":     aux["emotional_intensity"]["label"],
        "sentiment":         aux["emotional_intensity"]["sentiment"],
        "vader_compound":    aux["emotional_intensity"]["raw_compound"],

        "subjectivity_score": round(subj, 4),
        "subjectivity_label": aux["subjectivity"]["label"],

        # Passive voice engine (embedded dict for app.py compatibility)
        "passive":         passive_result,
        "passive_score":   round(passive_score, 4),

        # Lexicon engine
        "loaded_words":       lexicon_result["matched_words"],
        "loaded_word_count":  len(lexicon_result["matched_words"]),
        "unique_loaded_words":lexicon_result["unique_matches"],
        "lexicon_score":      lexicon_result["naive_score"],
        "category_counts":    dict(lexicon_result["category_counts"]),

        # Hedge engine
        "hedge_score":        round(hedge_score, 4),
        "hedge_label":        hedge_result.hedge_label,
        "hedge_result":       {
            "hedge_score":        round(hedge_score, 4),
            "hedge_label":        hedge_result.hedge_label,
            "epistemic_count":    hedge_result.epistemic_count,
            "inflation_count":    hedge_result.inflation_count,
            "epistemic_rate":     round(hedge_result.epistemic_rate, 4),
            "inflation_rate":     round(hedge_result.inflation_rate, 4),
            "flagged_sentences":  hedge_result.flagged_sentences,
        },
    }


# ── 4. Article-Level Analyser ─────────────────────────────────────────────────

def analyze_article(
    headline:   str,
    body_text:  str,
    body_paras: list[str],
    pipeline,
    sia:        SentimentIntensityAnalyzer,
    nlp=None,
) -> dict:
    """
    Full article analysis:
      • Headline vs Body comparison
      • Beginning / Middle / End breakdown
      • Overall composite score

    Parameters
    ----------
    headline   : str        — article headline
    body_text  : str        — full body as one string
    body_paras : list[str]  — body split into paragraphs
    pipeline   : sklearn Pipeline
    sia        : VADER SIA
    nlp        : spaCy model (optional)

    Returns
    -------
    dict with all chunk results + overall verdict
    """
    print("🔍 Analysing article...\n")

    # ── Overall body score ────────────────────────────────────────────────────
    overall   = analyze_chunk(body_text, pipeline, sia, label="Full Body", nlp=nlp)
    headline_ = analyze_chunk(headline,  pipeline, sia, label="Headline",  nlp=nlp)

    if nlp:
        overall["ner"] = extract_entities(body_text, nlp)
        # passive already computed inside analyze_chunk when nlp is provided
    else:
        overall["ner"]     = {}

    # ── Quote vs Opinion ─────────────────────────────────────────────────────
    overall["quote_opinion"] = analyze_quote_opinion(body_text)

    # ── Linguistic Features ──────────────────────────────────────────────────
    overall["linguistic"] = extract_linguistic_features(body_text)

    # ── Chronological thirds ──────────────────────────────────────────────────
    thirds    = _split_into_thirds(body_paras)
    beginning = analyze_chunk(thirds[0], pipeline, sia, label="Beginning", nlp=nlp)
    middle    = analyze_chunk(thirds[1], pipeline, sia, label="Middle",    nlp=nlp)
    end       = analyze_chunk(thirds[2], pipeline, sia, label="End",       nlp=nlp)

    return {
        "overall":    overall,
        "headline":   headline_,
        "beginning":  beginning,
        "middle":     middle,
        "end":        end,
    }


# ── 5. Helpers ────────────────────────────────────────────────────────────────

def _split_into_thirds(paragraphs: list[str]) -> tuple[str, str, str]:
    """
    Splits a list of paragraphs into Beginning / Middle / End thirds.
    Joins each third into a single string for analysis.
    Handles edge cases (very short articles gracefully).
    """
    n = len(paragraphs)

    if n == 0:
        return ("", "", "")
    elif n < 3:
        text = " ".join(paragraphs)
        return (text, text, text)

    third     = max(1, n // 3)
    beginning = " ".join(paragraphs[:third])
    middle    = " ".join(paragraphs[third : third * 2])
    end       = " ".join(paragraphs[third * 2:])

    return (beginning, middle, end)


def _empty_result(label: str) -> dict:
    """Returns a zeroed-out result dict for empty/missing text."""
    return {
        "label":              label,
        "text_preview":       "(no text)",
        "composite_score":    0.0,
        "verdict":            "No text provided",
        "color":              "grey",
        "emoji":              "⚪",
        "ml_probability":     0.0,
        "ml_label":           "N/A",
        "emotion_intensity":  0.0,
        "emotion_label":      "N/A",
        "sentiment":          "N/A",
        "vader_compound":     0.0,
        "subjectivity_score": 0.0,
        "subjectivity_label": "N/A",
        "passive":            {},
        "passive_score":      0.0,
        "loaded_words":       [],
        "loaded_word_count":  0,
        "unique_loaded_words":0,
        "lexicon_score":      0.0,
        "category_counts":    {},
        "hedge_score":        0.0,
        "hedge_label":        "N/A",
        "hedge_result":       {},
    }


# ── 6. Quote vs Opinion Analysis ─────────────────────────────────────────────

def analyze_quote_opinion(text: str) -> dict:
    """
    Classifies each sentence as Quoted, Attributed, or Opinion
    to measure how much of the text is the journalist's own voice
    versus sourced / attributed reporting.

    Categories
    ----------
    Quoted     : contains direct quotation marks
    Attributed : uses attribution verbs (said, told, claimed…)
    Opinion    : journalist's own unattributed voice
    """
    if not text or not text.strip():
        return _empty_quote_opinion()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    total = len(sentences)

    if total == 0:
        return _empty_quote_opinion()

    ATTRIBUTION_VERBS = {
        "said", "says", "told", "stated", "claimed", "argued",
        "noted", "explained", "added", "warned", "insisted",
        "suggested", "announced", "declared", "reported",
        "acknowledged", "confirmed", "denied", "testified",
        "remarked", "contended", "asserted", "maintained",
    }
    ATTRIBUTION_PHRASES = [
        "according to", "in a statement", "in an interview",
        "told reporters", "in a press conference", "speaking to",
    ]

    quoted     = []
    attributed = []
    opinion    = []

    for sent in sentences:
        sent_lower = sent.lower()

        has_quotes      = bool(re.search(r'["\u201c\u201d]', sent))
        words           = set(sent_lower.split())
        has_attribution = (
            bool(words & ATTRIBUTION_VERBS) or
            any(phrase in sent_lower for phrase in ATTRIBUTION_PHRASES)
        )

        if has_quotes:
            quoted.append(sent)
        elif has_attribution:
            attributed.append(sent)
        else:
            opinion.append(sent)

    opinion_ratio     = len(opinion)    / total
    quote_ratio       = len(quoted)     / total
    attribution_ratio = len(attributed) / total

    if opinion_ratio < 0.30:
        opinion_label = "Heavily Attributed"
    elif opinion_ratio < 0.50:
        opinion_label = "Balanced Attribution"
    elif opinion_ratio < 0.70:
        opinion_label = "Mostly Editorial"
    else:
        opinion_label = "Pure Opinion"

    return {
        "total_sentences":      total,
        "quoted_sentences":     len(quoted),
        "attributed_sentences": len(attributed),
        "opinion_sentences":    len(opinion),
        "opinion_ratio":        round(opinion_ratio, 4),
        "quote_ratio":          round(quote_ratio, 4),
        "attribution_ratio":    round(attribution_ratio, 4),
        "opinion_label":        opinion_label,
        "examples": {
            "quoted":     [s[:200] for s in quoted[:3]],
            "attributed": [s[:200] for s in attributed[:3]],
            "opinion":    [s[:200] for s in opinion[:3]],
        },
    }


def _empty_quote_opinion() -> dict:
    """Returns a zeroed-out quote_opinion result dict."""
    return {
        "total_sentences":      0,
        "quoted_sentences":     0,
        "attributed_sentences": 0,
        "opinion_sentences":    0,
        "opinion_ratio":        0.0,
        "quote_ratio":          0.0,
        "attribution_ratio":    0.0,
        "opinion_label":        "N/A",
        "examples":             {"quoted": [], "attributed": [], "opinion": []},
    }


# ── 7. Linguistic Feature Extraction ─────────────────────────────────────────

def extract_linguistic_features(text: str) -> dict:
    """
    Extract the 8+ linguistic features used by the dashboard display.
    Uses LinguisticFeatureExtractor from enhanced_ml.py.

    Returns
    -------
    dict : feature_name → float value (rounded to 4 d.p.)
    """
    if not text or not text.strip():
        return {}
    extractor  = LinguisticFeatureExtractor()
    features   = extractor._extract(text)
    feat_names = extractor.get_feature_names_out()
    return dict(zip(feat_names, [round(f, 4) for f in features]))


# ── 8. Engine Loader ──────────────────────────────────────────────────────────

def load_engines() -> tuple:
    """Load all engines once at startup."""
    ensure_vader_ready()
    pipeline = load_model()
    sia      = SentimentIntensityAnalyzer()
    nlp      = load_nlp()
    print("✅ All engines loaded.\n")
    return pipeline, sia, nlp


# ── 9. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline, sia, nlp = load_engines()

    HEADLINE = "Radical Policy Threatens to Devastate Working Families"

    PARAGRAPHS = [
        "The government's draconian new policy has triggered outrage among "
        "experts who warn of catastrophic consequences for ordinary citizens.",

        "Critics say the regime has recklessly ignored warnings from "
        "economists, pushing forward with what many call a shameful agenda.",

        "The bill passed with 52 votes in the Senate after three weeks "
        "of debate. Officials say implementation will begin next quarter.",

        "Several amendments were proposed during the review process, "
        "though most were rejected by the committee in a close vote.",

        "Protesters gathered outside parliament calling the move "
        "outrageous and unconscionable, vowing to challenge it in court.",

        "Analysts warn that unless the corrupt administration reverses "
        "course, the consequences could be both dire and irreversible.",
    ]

    BODY_TEXT = " ".join(PARAGRAPHS)

    result = analyze_article(
        headline   = HEADLINE,
        body_text  = BODY_TEXT,
        body_paras = PARAGRAPHS,
        pipeline   = pipeline,
        sia        = sia,
        nlp        = nlp,
    )

    print(f"\n{'═'*65}")
    print("  COMPOSITE ANALYSIS REPORT")
    print(f"{'═'*65}")

    for section, data in result.items():
        print(f"\n  {data['emoji']}  {data['label'].upper()}")
        print(f"     Verdict          : {data['verdict']}")
        print(f"     Composite Score  : {data['composite_score']:.0%}")
        print(f"     ML Probability   : {data['ml_probability']:.0%}")
        print(f"     Emotion Intensity: {data['emotion_intensity']:.0%}")
        print(f"     Subjectivity     : {data['subjectivity_score']:.0%}")
        print(f"     Passive Score    : {data['passive_score']:.0%}")
        print(f"     Lexicon Score    : {min(data['lexicon_score']*10, 1.0):.0%}")
        if data["loaded_words"]:
            unique = list(set(w for w, _ in data["loaded_words"]))
            print(f"     Loaded Words     : {', '.join(unique[:6])}")

    print(f"\n{'═'*65}\n")