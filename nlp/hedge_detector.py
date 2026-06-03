"""
utils/hedge_detector.py
------------------------
Detects hedging language in news articles — words and phrases that present
unverified claims with false certainty, or that soften attribution to avoid
accountability.
"""

import re
from dataclasses import dataclass, field

EPISTEMIC_HEDGES: list[str] = [
    "reportedly", "allegedly", "purportedly", "supposedly",
    "according to sources", "sources say", "sources claim",
    "sources familiar with", "people familiar with",
    "officials say", "officials claim", "insiders say",
    "it is understood", "it is believed", "it is claimed",
    "it has been reported", "it is alleged",
    "appears to", "seems to", "is said to", "is thought to",
    "is believed to", "is understood to", "is reported to",
    "may have", "might have", "could have", "would appear",
    "rumours suggest", "rumours indicate", "speculation",
    "unconfirmed reports", "unverified claims",
]

CERTAINTY_INFLATORS: list[str] = [
    "clearly", "obviously", "undeniably", "undoubtedly",
    "without question", "without doubt", "beyond question",
    "beyond doubt", "it is clear", "it is obvious",
    "it is evident", "it is apparent", "it is plain",
    "needless to say", "of course", "naturally",
    "goes without saying", "everyone knows",
    "certainly", "definitely", "absolutely", "unquestionably",
    "categorically", "demonstrably", "manifestly", "patently",
    "indisputably", "incontrovertibly", "incontestably",
    "as we all know", "as everyone knows", "as is well known",
    "as is widely known", "widely acknowledged", "widely accepted",
    "widely recognized", "it is well established", "no one disputes",
    "few would argue", "most people agree",
]


@dataclass
class HedgeMatch:
    phrase:       str
    category:     str
    sentence:     str
    sentence_idx: int


@dataclass
class HedgeResult:
    hedge_score:        float
    hedge_label:        str
    epistemic_count:    int
    inflation_count:    int
    epistemic_rate:     float
    inflation_rate:     float
    epistemic_matches:  list[HedgeMatch]
    inflation_matches:  list[HedgeMatch]
    flagged_sentences:  list[dict]


def _split_sentences(text: str) -> list[str]:
    protected = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|U\.S|U\.K|vs)\.',
                       lambda m: m.group().replace('.', '<!DOT!>'), text)
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    sentences = [s.replace('<!DOT!>', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def _find_matches(sentence: str, sent_idx: int, phrases: list[str], category: str) -> list[HedgeMatch]:
    matches = []
    sentence_lower = sentence.lower()
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        if re.search(pattern, sentence_lower):
            matches.append(HedgeMatch(
                phrase       = phrase,
                category     = category,
                sentence     = sentence,
                sentence_idx = sent_idx,
            ))
    return matches


def analyze_hedging(text: str) -> HedgeResult:
    if not text or not text.strip():
        return HedgeResult(
            hedge_score       = 0.0,
            hedge_label       = "No hedging detected",
            epistemic_count   = 0,
            inflation_count   = 0,
            epistemic_rate    = 0.0,
            inflation_rate    = 0.0,
            epistemic_matches = [],
            inflation_matches = [],
            flagged_sentences = [],
        )

    sentences = _split_sentences(text)
    n_sents   = max(len(sentences), 1)

    epistemic_matches: list[HedgeMatch] = []
    inflation_matches: list[HedgeMatch] = []

    for idx, sentence in enumerate(sentences):
        epistemic_matches.extend(_find_matches(sentence, idx, EPISTEMIC_HEDGES, "epistemic"))
        inflation_matches.extend(_find_matches(sentence, idx, CERTAINTY_INFLATORS, "inflation"))

    epistemic_sents = len({m.sentence_idx for m in epistemic_matches})
    inflation_sents = len({m.sentence_idx for m in inflation_matches})

    epistemic_rate = min(epistemic_sents / n_sents, 1.0)
    inflation_rate = min(inflation_sents / n_sents, 1.0)

    hedge_score = min((0.4 * epistemic_rate) + (0.6 * inflation_rate), 1.0)

    flagged_idxs = {m.sentence_idx for m in epistemic_matches} | \
                   {m.sentence_idx for m in inflation_matches}

    flagged_sentences = []
    for idx in sorted(flagged_idxs):
        sent = sentences[idx]
        e_phrases = [m.phrase for m in epistemic_matches if m.sentence_idx == idx]
        i_phrases = [m.phrase for m in inflation_matches if m.sentence_idx == idx]
        flagged_sentences.append({
            "sentence"         : sent,
            "epistemic_phrases": e_phrases,
            "inflation_phrases": i_phrases,
            "type"             : (
                "both"      if e_phrases and i_phrases else
                "epistemic" if e_phrases               else
                "inflation"
            ),
        })

    hedge_label = _score_to_label(hedge_score)

    return HedgeResult(
        hedge_score       = round(hedge_score, 4),
        hedge_label       = hedge_label,
        epistemic_count   = len(epistemic_matches),
        inflation_count   = len(inflation_matches),
        epistemic_rate    = round(epistemic_rate, 4),
        inflation_rate    = round(inflation_rate, 4),
        epistemic_matches = epistemic_matches,
        inflation_matches = inflation_matches,
        flagged_sentences = flagged_sentences,
    )


def _score_to_label(score: float) -> str:
    if score < 0.05:   return "Minimal hedging"
    elif score < 0.15: return "Occasional hedging"
    elif score < 0.30: return "Moderate hedging"
    elif score < 0.50: return "Heavy hedging"
    else:              return "Pervasive hedging"


def hedge_summary(result: HedgeResult) -> str:
    if result.hedge_score < 0.05:
        return "No significant hedging language detected."
    parts = []
    if result.epistemic_count > 0:
        parts.append(f"{result.epistemic_count} epistemic hedge(s) found — claims presented without verified attribution.")
    if result.inflation_count > 0:
        parts.append(f"{result.inflation_count} certainty inflator(s) found — editorial opinion presented as fact.")
    return f"{result.hedge_label.capitalize()} detected (score {result.hedge_score:.2f}). " + " ".join(parts)
