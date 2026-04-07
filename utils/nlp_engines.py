"""
Step 4: Auxiliary NLP Engines
-------------------------------
Two independent scoring engines that run ALONGSIDE the ML model:

  1. TextBlob  → Subjectivity Score (0.0 = objective, 1.0 = subjective)
  2. VADER     → Emotional Intensity (compound score -1.0 to +1.0)

Why both?
  • TextBlob catches opinionated language even when it's not "negative"
    e.g. "The heroic leader delivered a landmark speech" scores high
    subjectivity but neutral/positive sentiment.
  • VADER catches emotional charge — anger, fear, outrage — regardless
    of whether the text is technically "subjective".
  Together they cover the full spectrum of biased writing styles.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# ── 1. Download VADER lexicon (one-time, cached after first run) ──────────────

def ensure_vader_ready() -> None:
    """Download VADER lexicon if not already present."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("📥 Downloading VADER lexicon...")
        nltk.download("vader_lexicon", quiet=True)
        print("✅ VADER lexicon ready.\n")


# ── 2. TextBlob Subjectivity Engine ──────────────────────────────────────────

def get_subjectivity(text: str) -> dict:
    """
    Uses TextBlob's pattern-based analyser to score subjectivity.

    TextBlob.sentiment returns (polarity, subjectivity):
      polarity    : -1.0 (negative) → +1.0 (positive)  [we don't use this]
      subjectivity:  0.0 (factual)  →  1.0 (opinionated)

    Returns
    -------
    dict:
        score : float 0.0–1.0
        label : plain-English category
    """
    blob  = TextBlob(text)
    score = round(blob.sentiment.subjectivity, 4)

    if score < 0.2:
        label = "Very Objective"
    elif score < 0.4:
        label = "Mostly Objective"
    elif score < 0.6:
        label = "Mixed"
    elif score < 0.8:
        label = "Mostly Subjective"
    else:
        label = "Highly Subjective"

    return {"score": score, "label": label}


# ── 3. VADER Emotional Intensity Engine ───────────────────────────────────────

def get_emotional_intensity(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    """
    Uses NLTK's VADER to measure emotional charge in the text.

    VADER returns four scores:
      neg, neu, pos : proportion of text in each category
      compound      : normalised aggregate score (-1.0 to +1.0)

    We use the ABSOLUTE value of compound as "emotional intensity"
    because both strong positive AND strong negative language
    signal emotional (potentially biased) writing.

    Returns
    -------
    dict:
        raw_compound  : float -1.0 to +1.0  (original VADER output)
        intensity     : float  0.0 to  1.0  (abs value — our metric)
        sentiment     : "Positive" | "Negative" | "Neutral"
        label         : plain-English intensity category
    """
    scores   = sia.polarity_scores(text)
    compound = scores["compound"]
    intensity = round(abs(compound), 4)

    # Sentiment direction
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Intensity label
    if intensity < 0.2:
        label = "Calm / Measured"
    elif intensity < 0.4:
        label = "Mildly Charged"
    elif intensity < 0.6:
        label = "Moderately Charged"
    elif intensity < 0.8:
        label = "Emotionally Charged"
    else:
        label = "Highly Emotionally Charged"

    return {
        "raw_compound": round(compound, 4),
        "intensity":    intensity,
        "sentiment":    sentiment,
        "label":        label,
    }


# ── 4. Combined Runner ────────────────────────────────────────────────────────

def run_auxiliary_engines(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    """
    Runs both engines and returns a single merged result dict.
    This is what the composite scorer in Step 6 will call.

    Returns
    -------
    dict:
        subjectivity       : dict (score, label)
        emotional_intensity: dict (raw_compound, intensity, sentiment, label)
    """
    return {
        "subjectivity":        get_subjectivity(text),
        "emotional_intensity": get_emotional_intensity(text, sia),
    }


# ── 5. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_vader_ready()
    sia = SentimentIntensityAnalyzer()

    TEST_CASES = {
        "Outrage / High Bias": (
            "The regime recklessly imposed draconian laws, triggering a "
            "catastrophic collapse. Radical extremists have shamefully "
            "undermined everything we stand for."
        ),
        "Neutral / Factual": (
            "The government announced new infrastructure spending on Tuesday. "
            "Officials said the bill would be reviewed by a committee "
            "before a final vote next month."
        ),
        "Positive Spin / Hidden Bias": (
            "In a historic and heroic move, the visionary leader delivered "
            "a landmark speech that patriots across the nation celebrated "
            "as a triumphant moment for freedom."
        ),
        "Negative Emotional / Alarming": (
            "The catastrophic and devastating failure of leadership has "
            "plunged the country into chaos and despair, alarming experts "
            "who warn of an apocalyptic outcome."
        ),
    }

    print(f"\n{'═'*65}")
    print("  AUXILIARY NLP ENGINE REPORT")
    print(f"{'═'*65}")

    for name, text in TEST_CASES.items():
        result = run_auxiliary_engines(text, sia)
        sub    = result["subjectivity"]
        emo    = result["emotional_intensity"]

        print(f"\n  ── {name}")
        print(f"     TextBlob Subjectivity  : {sub['score']:.2f}  →  {sub['label']}")
        print(f"     VADER Compound         : {emo['raw_compound']:+.2f}  "
              f"({emo['sentiment']})")
        print(f"     Emotional Intensity    : {emo['intensity']:.2f}  →  {emo['label']}")

    print(f"\n{'═'*65}\n")