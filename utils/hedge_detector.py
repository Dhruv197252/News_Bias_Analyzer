"""
utils/hedge_detector.py
------------------------
Detects hedging language in news articles — words and phrases that present
unverified claims with false certainty, or that soften attribution to avoid
accountability.

Two distinct hedge patterns matter for bias analysis:

  1. EPISTEMIC HEDGES — "reportedly", "allegedly", "sources say"
     The writer distances themselves from a claim's truth.
     Used legitimately in breaking news; used manipulatively to spread
     unverified accusations without legal liability.

  2. CERTAINTY INFLATION — "clearly", "obviously", "undeniably"
     The writer presents a contested opinion AS IF it were fact.
     This is the more insidious bias pattern — it smuggles editorial
     judgement into what looks like straight reporting.

Why this matters for bias detection:
  A sentence like "The minister clearly lied to Parliament" uses zero
  loaded words from our lexicon, scores low on VADER emotion, and passes
  TextBlob subjectivity — yet it is deeply biased. "Clearly" is doing the
  work of turning an accusation into a fact. The hedge detector catches this.

Scoring logic:
  hedge_score = (epistemic_rate * 0.4) + (inflation_rate * 0.6)

  Certainty inflation weighted higher (0.6) because it is the more
  deceptive pattern — it hides opinion inside factual-sounding language.
  Epistemic hedges (0.4) are often legitimate journalistic attribution.
"""

import re
from dataclasses import dataclass, field


# ── 1. Hedge Lexicons ─────────────────────────────────────────────────────────

# Epistemic hedges — writer distances from truth of claim
# Pattern: claim is attributed to unnamed/vague sources, or flagged as
# unverified. Legitimate in breaking news; manipulative when used to
# spread accusations without accountability.
EPISTEMIC_HEDGES: list[str] = [
    # Source attribution hedges
    "reportedly", "allegedly", "purportedly", "supposedly",
    "according to sources", "sources say", "sources claim",
    "sources familiar with", "people familiar with",
    "officials say", "officials claim", "insiders say",
    "it is understood", "it is believed", "it is claimed",
    "it has been reported", "it is alleged",

    # Epistemic distance markers
    "appears to", "seems to", "is said to", "is thought to",
    "is believed to", "is understood to", "is reported to",
    "may have", "might have", "could have", "would appear",

    # Rumour/unverified flags
    "rumours suggest", "rumours indicate", "speculation",
    "unconfirmed reports", "unverified claims",
]

# Certainty inflation — writer presents opinion as established fact
# Pattern: contested claim stated without qualification, using adverbs
# that manufacture consensus. This is the more dangerous bias pattern.
CERTAINTY_INFLATORS: list[str] = [
    # Factuality adverbs (presenting opinion as fact)
    "clearly", "obviously", "undeniably", "undoubtedly",
    "without question", "without doubt", "beyond question",
    "beyond doubt", "it is clear", "it is obvious",
    "it is evident", "it is apparent", "it is plain",
    "needless to say", "of course", "naturally",
    "goes without saying", "everyone knows",

    # Certainty intensifiers
    "certainly", "definitely", "absolutely", "unquestionably",
    "categorically", "demonstrably", "manifestly", "patently",
    "indisputably", "incontrovertibly", "incontestably",

    # Consensus-manufacturing phrases
    "as we all know", "as everyone knows", "as is well known",
    "as is widely known", "widely acknowledged", "widely accepted",
    "widely recognized", "it is well established", "no one disputes",
    "few would argue", "most people agree",
]


# ── 2. Data Classes ───────────────────────────────────────────────────────────

@dataclass
class HedgeMatch:
    """A single hedge detected in the text."""
    phrase:       str          # the matched hedge phrase
    category:     str          # "epistemic" or "inflation"
    sentence:     str          # the full sentence it was found in
    sentence_idx: int          # which sentence (0-indexed)


@dataclass
class HedgeResult:
    """Full result from analyze_hedging()."""
    hedge_score:        float               # composite 0.0–1.0
    hedge_label:        str                 # human-readable verdict
    epistemic_count:    int                 # raw count of epistemic hedges
    inflation_count:    int                 # raw count of certainty inflators
    epistemic_rate:     float               # epistemic hits / n_sentences
    inflation_rate:     float               # inflation hits / n_sentences
    epistemic_matches:  list[HedgeMatch]    # detailed match list
    inflation_matches:  list[HedgeMatch]    # detailed match list
    flagged_sentences:  list[dict]          # sentences with both types for UI


# ── 3. Core Detection Logic ───────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Uses regex rather than NLTK to avoid an extra dependency.
    Handles common abbreviations (Mr., Dr., U.S.) to reduce false splits.
    """
    # Temporarily protect common abbreviations
    protected = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|U\.S|U\.K|vs)\.',
                       lambda m: m.group().replace('.', '<!DOT!>'), text)
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    # Restore dots
    sentences = [s.replace('<!DOT!>', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def _find_matches(sentence: str, sent_idx: int,
                  phrases: list[str], category: str) -> list[HedgeMatch]:
    """
    Scan one sentence for all phrases in the given list.
    Uses regex word boundaries (\\b) to prevent partial substring matches.
    Returns a HedgeMatch for each hit found.
    """
    matches = []
    sentence_lower = sentence.lower()
    
    for phrase in phrases:
        # \\b ensures we only match whole words
        # re.escape makes sure any weird characters in the phrase don't break the regex
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
    """
    Main entry point. Analyse a body of text for hedging language.

    Parameters
    ----------
    text : str
        Article body or paragraph to analyse.

    Returns
    -------
    HedgeResult dataclass with all counts, rates, matches, and score.

    Scoring formula
    ---------------
    epistemic_rate = epistemic_count / n_sentences
    inflation_rate = inflation_count / n_sentences

    Both rates are capped at 1.0 before weighting (a sentence can only
    count once per category even if multiple phrases match).

    hedge_score = (0.4 * epistemic_rate) + (0.6 * inflation_rate)
    hedge_score is capped at 1.0.

    Certainty inflation is weighted 0.6 because it is the more deceptive
    pattern — it disguises opinion as fact rather than merely flagging
    uncertainty. Epistemic hedges are often legitimate journalism.
    """
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

    # Scan sentence by sentence so we can track which sentence each match is in
    for idx, sentence in enumerate(sentences):
        epistemic_matches.extend(
            _find_matches(sentence, idx, EPISTEMIC_HEDGES, "epistemic")
        )
        inflation_matches.extend(
            _find_matches(sentence, idx, CERTAINTY_INFLATORS, "inflation")
        )

    # Sentence-level deduplication — count each sentence once per category
    # (one sentence with 3 hedges shouldn't score 3x a sentence with 1)
    epistemic_sents = len({m.sentence_idx for m in epistemic_matches})
    inflation_sents = len({m.sentence_idx for m in inflation_matches})

    epistemic_rate = min(epistemic_sents / n_sents, 1.0)
    inflation_rate = min(inflation_sents / n_sents, 1.0)

    hedge_score = min(
        (0.4 * epistemic_rate) + (0.6 * inflation_rate),
        1.0
    )

    # Build flagged sentence list for the dashboard
    # A sentence is "flagged" if it contains either type of hedge
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


# ── 4. Score → Label ──────────────────────────────────────────────────────────

def _score_to_label(score: float) -> str:
    """
    Map numeric hedge score to a human-readable verdict.

    Thresholds chosen to match the scorer.py verdict band system
    so the hedge tab feels consistent with the rest of the dashboard.
    """
    if score < 0.05:
        return "Minimal hedging"
    elif score < 0.15:
        return "Occasional hedging"
    elif score < 0.30:
        return "Moderate hedging"
    elif score < 0.50:
        return "Heavy hedging"
    else:
        return "Pervasive hedging"


# ── 5. Convenience Summary ────────────────────────────────────────────────────

def hedge_summary(result: HedgeResult) -> str:
    """
    Return a one-paragraph plain-English summary of the hedge analysis.
    Used by the dashboard tooltip and the history log.
    """
    if result.hedge_score < 0.05:
        return "No significant hedging language detected."

    parts = []
    if result.epistemic_count > 0:
        parts.append(
            f"{result.epistemic_count} epistemic hedge(s) found "
            f"(e.g. 'reportedly', 'sources say') — claims presented "
            f"without verified attribution."
        )
    if result.inflation_count > 0:
        parts.append(
            f"{result.inflation_count} certainty inflator(s) found "
            f"(e.g. 'clearly', 'obviously') — editorial opinion "
            f"presented as established fact."
        )

    label = result.hedge_label.lower()
    return (
        f"{label.capitalize()} detected (score {result.hedge_score:.2f}). "
        + " ".join(parts)
    )


# ── 6. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    SAMPLES = [
        (
            "Wire report (should be clean)",
            "The government announced a new infrastructure bill on Tuesday. "
            "The legislation will be reviewed by a parliamentary committee. "
            "Officials confirmed the vote is scheduled for next week."
        ),
        (
            "Breaking news with legitimate epistemic hedges",
            "The prime minister reportedly met with the opposition leader "
            "on Monday, according to sources familiar with the matter. "
            "It is understood that the talks focused on the budget. "
            "The meeting has not been officially confirmed."
        ),
        (
            "Editorial with certainty inflation",
            "It is clearly the case that the government has failed its citizens. "
            "Obviously, the minister knew about the scandal all along. "
            "Needless to say, this administration has lost all credibility. "
            "Everyone knows that the policy was designed to benefit the wealthy."
        ),
        (
            "Mixed: both patterns",
            "The minister allegedly approved the contract, sources say. "
            "Clearly this represents a serious conflict of interest. "
            "It is reportedly the largest corruption case in a decade. "
            "Obviously, the public deserves answers."
        ),
    ]

    for title, text in SAMPLES:
        print(f"\n{'═'*60}")
        print(f"  {title}")
        print(f"{'═'*60}")

        result = analyze_hedging(text)

        print(f"  Score        : {result.hedge_score:.4f}  →  {result.hedge_label}")
        print(f"  Epistemic    : {result.epistemic_count} hit(s)  "
              f"(rate {result.epistemic_rate:.2f})")
        print(f"  Inflation    : {result.inflation_count} hit(s)  "
              f"(rate {result.inflation_rate:.2f})")

        if result.flagged_sentences:
            print(f"\n  Flagged sentences:")
            for fs in result.flagged_sentences:
                tag = f"[{fs['type'].upper()}]"
                print(f"    {tag:<12} {fs['sentence'][:80]}")
                if fs["epistemic_phrases"]:
                    print(f"               Epistemic : {fs['epistemic_phrases']}")
                if fs["inflation_phrases"]:
                    print(f"               Inflation : {fs['inflation_phrases']}")

        print(f"\n  Summary: {hedge_summary(result)}")