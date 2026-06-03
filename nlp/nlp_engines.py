"""
Auxiliary NLP Engines — VADER + TextBlob
-----------------------------------------
  1. TextBlob  → Subjectivity Score (0.0 = objective, 1.0 = subjective)
  2. VADER     → Emotional Intensity (compound score -1.0 to +1.0)
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob


def ensure_vader_ready() -> None:
    """Download VADER lexicon if not already present."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def get_subjectivity(text: str) -> dict:
    blob  = TextBlob(text)
    score = round(blob.sentiment.subjectivity, 4)
    if score < 0.2:    label = "Very Objective"
    elif score < 0.4:  label = "Mostly Objective"
    elif score < 0.6:  label = "Mixed"
    elif score < 0.8:  label = "Mostly Subjective"
    else:              label = "Highly Subjective"
    return {"score": score, "label": label}


def get_emotional_intensity(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    scores    = sia.polarity_scores(text)
    compound  = scores["compound"]
    intensity = round(abs(compound), 4)
    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    if intensity < 0.2:    label = "Calm / Measured"
    elif intensity < 0.4:  label = "Mildly Charged"
    elif intensity < 0.6:  label = "Moderately Charged"
    elif intensity < 0.8:  label = "Emotionally Charged"
    else:                  label = "Highly Emotionally Charged"
    return {"raw_compound": round(compound, 4), "intensity": intensity,
            "sentiment": sentiment, "label": label}


def run_auxiliary_engines(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    return {
        "subjectivity":        get_subjectivity(text),
        "emotional_intensity": get_emotional_intensity(text, sia),
    }
