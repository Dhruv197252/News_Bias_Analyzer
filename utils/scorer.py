"""
Step 6: Composite Scoring Algorithm
--------------------------------------
The master orchestrator. Takes a block of text and runs it through
ALL four engines, then combines results into one final score.

Weighting:
  ML Model Probability   → 60%  (trained on real journalism patterns)
  VADER Emotional Intensity → 20%  (raw emotional charge)
  TextBlob Subjectivity  → 20%  (opinion vs fact language)

Why these weights?
  The ML model has the most signal — it learned from 3000+ expert-labelled
  real news sentences. The auxiliary engines catch what ML misses
  (emotional framing, opinionated language) but are noisier signals,
  so they get smaller weights.
"""

import re
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from utils.bias_lexicon  import scan_text
from utils.ml_engine     import load_model, predict_bias
from utils.nlp_engines   import run_auxiliary_engines, ensure_vader_ready
from utils.ner_engine    import load_nlp, extract_entities
from utils.passive_voice import analyze_passive_voice
from utils.enhanced_ml   import LinguisticFeatureExtractor

# ── 1. Weights ────────────────────────────────────────────────────────────────

WEIGHTS = {
    "ml":          0.60,
    "emotion":     0.20,
    "subjectivity":0.20,
}

# ── 2. Label Thresholds ───────────────────────────────────────────────────────

def composite_label(score: float) -> dict:
    """
    Maps a 0.0–1.0 composite score to a plain-English verdict.

    Returns
    -------
    dict:
        label : str   — human-readable category
        color : str   — for Streamlit UI colouring (Step 7)
        emoji : str   — visual indicator
    """
    if score < 0.20:
        return {"label": "Appears Neutral",        "color": "green",  "emoji": "🟢"}
    elif score < 0.40:
        return {"label": "Slightly Opinionated",   "color": "blue",   "emoji": "🔵"}
    elif score < 0.60:
        return {"label": "Moderate Bias",          "color": "orange", "emoji": "🟠"}
    elif score < 0.80:
        return {"label": "Highly Opinionated",     "color": "red",    "emoji": "🔴"}
    else:
        return {"label": "Extreme Bias Detected",  "color": "red",    "emoji": "🚨"}


# ── 3. Core Chunk Analyser ────────────────────────────────────────────────────

def analyze_chunk(
    text:     str,
    pipeline,                        # fitted sklearn pipeline
    sia:      SentimentIntensityAnalyzer,
    label:    str = "chunk",
) -> dict:
    """
    Master analysis function. Runs all four engines on a text block
    and returns a single unified result dictionary.

    Parameters
    ----------
    text     : str  — any block of text (headline, paragraph, full article)
    pipeline : fitted sklearn Pipeline from ml_engine.py
    sia      : VADER SentimentIntensityAnalyzer instance
    label    : str  — name for this chunk (e.g. "Headline", "Body", "Beginning")

    Returns
    -------
    Unified dict with all scores, labels, and detected loaded words.
    """
    if not text or not text.strip():
        return _empty_result(label)

    # ── Engine 1: Lexicon Scanner ─────────────────────────────────────────────
    lexicon_result = scan_text(text)

    # ── Engine 2: ML Model ────────────────────────────────────────────────────
    ml_result = predict_bias(text, pipeline)
    ml_prob   = ml_result["probability"]          # 0.0 – 1.0

    # ── Engine 3 & 4: VADER + TextBlob ───────────────────────────────────────
    aux       = run_auxiliary_engines(text, sia)
    emotion   = aux["emotional_intensity"]["intensity"]   # 0.0 – 1.0
    subj      = aux["subjectivity"]["score"]              # 0.0 – 1.0

    # ── Composite Score ───────────────────────────────────────────────────────
    composite = (
        WEIGHTS["ml"]           * ml_prob  +
        WEIGHTS["emotion"]      * emotion  +
        WEIGHTS["subjectivity"] * subj
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

        # Lexicon engine
        "loaded_words":       lexicon_result["matched_words"],
        "loaded_word_count":  len(lexicon_result["matched_words"]),
        "unique_loaded_words":lexicon_result["unique_matches"],
        "lexicon_score":      lexicon_result["naive_score"],
        "category_counts":    dict(lexicon_result["category_counts"]),
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

    Returns
    -------
    dict with all chunk results + overall verdict
    """
    print("🔍 Analysing article...\n")

    # ── Overall body score ────────────────────────────────────────────────────
    overall   = analyze_chunk(body_text,  pipeline, sia, label="Full Body")
    headline_ = analyze_chunk(headline,   pipeline, sia, label="Headline")
    
    if nlp:
        overall["ner"]     = extract_entities(body_text, nlp)
        overall["passive"] = analyze_passive_voice(body_text, nlp)
    else:
        overall["ner"]     = {}
        overall["passive"] = {}

    # ── Chronological thirds ──────────────────────────────────────────────────
    thirds    = _split_into_thirds(body_paras)
    beginning = analyze_chunk(thirds[0], pipeline, sia, label="Beginning")
    middle    = analyze_chunk(thirds[1], pipeline, sia, label="Middle")
    end       = analyze_chunk(thirds[2], pipeline, sia, label="End")

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
        # Treat whole article as all three sections
        text = " ".join(paragraphs)
        return (text, text, text)

    third = max(1, n // 3)
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
        "loaded_words":       [],
        "loaded_word_count":  0,
        "unique_loaded_words":0,
        "lexicon_score":      0.0,
        "category_counts":    {},
    }


# ── 6. Engine Loader ──────────────────────────────────────────────────────────

def load_engines() -> tuple:
    """Load all engines once at startup."""
    ensure_vader_ready()
    pipeline = load_model()
    sia      = SentimentIntensityAnalyzer()
    
    # THIS IS THE MISSING LINE! We have to load the AI before we return it.
    nlp      = load_nlp()  
    
    print("✅ All engines loaded.\n")
    return pipeline, sia, nlp

# ── 7. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline, sia, nlp = load_engines()

    # Simulate a full article
    HEADLINE = "Radical Policy Threatens to Devastate Working Families"

    PARAGRAPHS = [
        # Beginning — alarmist framing
        "The government's draconian new policy has triggered outrage among "
        "experts who warn of catastrophic consequences for ordinary citizens.",

        "Critics say the regime has recklessly ignored warnings from "
        "economists, pushing forward with what many call a shameful agenda.",

        # Middle — slightly more neutral
        "The bill passed with 52 votes in the Senate after three weeks "
        "of debate. Officials say implementation will begin next quarter.",

        "Several amendments were proposed during the review process, "
        "though most were rejected by the committee in a close vote.",

        # End — charged again
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
    # ── Print Report ──────────────────────────────────────────────────────────
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
        if data["loaded_words"]:
            unique = list(set(w for w, _ in data["loaded_words"]))
            print(f"     Loaded Words     : {', '.join(unique[:6])}")

    print(f"\n{'═'*65}\n")