"""
Composite Scoring Algorithm — orchestrates all 6 NLP engines.
Weights: ML=60%, Emotion=10%, Subjectivity=10%, Passive=10%, Lexicon=5%, Hedge=5%
"""

import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Updated imports: utils.* → nlp.*
from nlp.bias_lexicon   import scan_text
from nlp.ml_engine      import load_model, predict_bias
from nlp.nlp_engines    import run_auxiliary_engines, ensure_vader_ready
from nlp.ner_engine     import load_nlp, extract_entities
from nlp.passive_voice  import analyze_passive_voice
from nlp.enhanced_ml    import LinguisticFeatureExtractor
from nlp.hedge_detector import analyze_hedging

WEIGHTS = {
    "ml": 0.60, "emotion": 0.10, "subjectivity": 0.10,
    "passive": 0.10, "lexicon": 0.05, "hedge": 0.05,
}


def composite_label(score: float) -> dict:
    if score < 0.30:   return {"label": "Appears Neutral",        "color": "green",  "emoji": "🟢"}
    elif score < 0.47: return {"label": "Slightly Opinionated",   "color": "blue",   "emoji": "🔵"}
    elif score < 0.63: return {"label": "Moderate Bias",          "color": "orange", "emoji": "🟠"}
    elif score < 0.78: return {"label": "Highly Opinionated",     "color": "red",    "emoji": "🔴"}
    else:              return {"label": "Extreme Bias Detected",  "color": "red",    "emoji": "🚨"}


def analyze_chunk(text, pipeline, sia, label="chunk", nlp=None) -> dict:
    if not text or not text.strip():
        return _empty_result(label)

    lexicon_result = scan_text(text)
    lexicon_score  = min(lexicon_result["naive_score"] * 10, 1.0)
    ml_result      = predict_bias(text, pipeline)
    ml_prob        = ml_result["probability"]
    aux            = run_auxiliary_engines(text, sia)
    emotion        = aux["emotional_intensity"]["intensity"]
    subj           = aux["subjectivity"]["score"]

    if nlp:
        passive_result = analyze_passive_voice(text, nlp)
        passive_score  = passive_result.get("score", 0.0)
    else:
        passive_result = {}
        passive_score  = 0.0

    hedge_result = analyze_hedging(text)
    hedge_score  = hedge_result.hedge_score

    composite = round(
        WEIGHTS["ml"]           * ml_prob       +
        WEIGHTS["emotion"]      * emotion       +
        WEIGHTS["subjectivity"] * subj          +
        WEIGHTS["passive"]      * passive_score +
        WEIGHTS["lexicon"]      * lexicon_score +
        WEIGHTS["hedge"]        * hedge_score,
        4
    )
    verdict = composite_label(composite)

    return {
        "label": label, "text_preview": text[:120] + "..." if len(text) > 120 else text,
        "composite_score": composite, "verdict": verdict["label"],
        "color": verdict["color"], "emoji": verdict["emoji"],
        "ml_probability": round(ml_prob, 4), "ml_label": ml_result["label"],
        "emotion_intensity": round(emotion, 4), "emotion_label": aux["emotional_intensity"]["label"],
        "sentiment": aux["emotional_intensity"]["sentiment"],
        "vader_compound": aux["emotional_intensity"]["raw_compound"],
        "subjectivity_score": round(subj, 4), "subjectivity_label": aux["subjectivity"]["label"],
        "passive": passive_result, "passive_score": round(passive_score, 4),
        "loaded_words": lexicon_result["matched_words"],
        "loaded_word_count": len(lexicon_result["matched_words"]),
        "unique_loaded_words": lexicon_result["unique_matches"],
        "lexicon_score": lexicon_result["naive_score"],
        "category_counts": dict(lexicon_result["category_counts"]),
        "hedge_score": round(hedge_score, 4), "hedge_label": hedge_result.hedge_label,
        "hedge_result": {
            "hedge_score": round(hedge_score, 4), "hedge_label": hedge_result.hedge_label,
            "epistemic_count": hedge_result.epistemic_count,
            "inflation_count": hedge_result.inflation_count,
            "epistemic_rate": round(hedge_result.epistemic_rate, 4),
            "inflation_rate": round(hedge_result.inflation_rate, 4),
            "flagged_sentences": hedge_result.flagged_sentences,
        },
    }


def analyze_article(headline, body_text, body_paras, pipeline, sia, nlp=None) -> dict:
    overall   = analyze_chunk(body_text, pipeline, sia, label="Full Body", nlp=nlp)
    headline_ = analyze_chunk(headline,  pipeline, sia, label="Headline",  nlp=nlp)

    overall["ner"]           = extract_entities(body_text, nlp) if nlp else {}
    overall["quote_opinion"] = analyze_quote_opinion(body_text)
    overall["linguistic"]    = extract_linguistic_features(body_text)

    thirds    = _split_into_thirds(body_paras)
    beginning = analyze_chunk(thirds[0], pipeline, sia, label="Beginning", nlp=nlp)
    middle    = analyze_chunk(thirds[1], pipeline, sia, label="Middle",    nlp=nlp)
    end       = analyze_chunk(thirds[2], pipeline, sia, label="End",       nlp=nlp)

    return {"overall": overall, "headline": headline_,
            "beginning": beginning, "middle": middle, "end": end}


def _split_into_thirds(paragraphs):
    n = len(paragraphs)
    if n == 0:  return ("", "", "")
    if n < 3:
        text = " ".join(paragraphs)
        return (text, text, text)
    third = max(1, n // 3)
    return (" ".join(paragraphs[:third]),
            " ".join(paragraphs[third:third*2]),
            " ".join(paragraphs[third*2:]))


def _empty_result(label):
    return {
        "label": label, "text_preview": "(no text)", "composite_score": 0.0,
        "verdict": "No text provided", "color": "grey", "emoji": "⚪",
        "ml_probability": 0.0, "ml_label": "N/A", "emotion_intensity": 0.0,
        "emotion_label": "N/A", "sentiment": "N/A", "vader_compound": 0.0,
        "subjectivity_score": 0.0, "subjectivity_label": "N/A",
        "passive": {}, "passive_score": 0.0, "loaded_words": [],
        "loaded_word_count": 0, "unique_loaded_words": 0, "lexicon_score": 0.0,
        "category_counts": {}, "hedge_score": 0.0, "hedge_label": "N/A", "hedge_result": {},
    }


def analyze_quote_opinion(text: str) -> dict:
    if not text or not text.strip():
        return {"total_sentences": 0, "quoted_sentences": 0, "attributed_sentences": 0,
                "opinion_sentences": 0, "opinion_ratio": 0.0, "quote_ratio": 0.0,
                "attribution_ratio": 0.0, "opinion_label": "N/A",
                "examples": {"quoted": [], "attributed": [], "opinion": []}}

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    total = len(sentences)
    ATTR_VERBS = {"said","says","told","stated","claimed","argued","noted","explained",
                  "added","warned","insisted","suggested","announced","declared","reported"}
    ATTR_PHRASES = ["according to","in a statement","in an interview","told reporters"]

    quoted, attributed, opinion = [], [], []
    for sent in sentences:
        sl = sent.lower()
        if re.search(r'["\u201c\u201d]', sent):
            quoted.append(sent)
        elif set(sl.split()) & ATTR_VERBS or any(p in sl for p in ATTR_PHRASES):
            attributed.append(sent)
        else:
            opinion.append(sent)

    opr = len(opinion) / max(total, 1)
    label = ("Heavily Attributed" if opr < 0.30 else "Balanced Attribution"
             if opr < 0.50 else "Mostly Editorial" if opr < 0.70 else "Pure Opinion")
    return {
        "total_sentences": total, "quoted_sentences": len(quoted),
        "attributed_sentences": len(attributed), "opinion_sentences": len(opinion),
        "opinion_ratio": round(opr, 4),
        "quote_ratio": round(len(quoted)/max(total,1), 4),
        "attribution_ratio": round(len(attributed)/max(total,1), 4),
        "opinion_label": label,
        "examples": {"quoted": [s[:200] for s in quoted[:3]],
                     "attributed": [s[:200] for s in attributed[:3]],
                     "opinion": [s[:200] for s in opinion[:3]]},
    }


def extract_linguistic_features(text: str) -> dict:
    if not text or not text.strip():
        return {}
    ex = LinguisticFeatureExtractor()
    return dict(zip(ex.get_feature_names_out(), [round(f, 4) for f in ex._extract(text)]))


def load_engines() -> tuple:
    ensure_vader_ready()
    pipeline = load_model()
    sia      = SentimentIntensityAnalyzer()
    nlp      = load_nlp()
    return pipeline, sia, nlp
