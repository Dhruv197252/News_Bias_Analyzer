"""
Step 9: Passive Voice Detector
--------------------------------
Uses spaCy's dependency parser to detect passive voice constructions
in news text.

Why passive voice matters for bias detection:
  Passive voice is one of the most common tools journalists use
  to obscure responsibility and agency in news writing.

  ACTIVE  : "Police shot the protesters"     → clear responsibility
  PASSIVE : "Protesters were shot by police" → responsibility softened
  PASSIVE : "Protesters were shot"           → responsibility removed entirely

  Biased articles systematically use passive voice to:
    1. Protect favoured actors from blame
    2. Make negative actions seem like they "just happened"
    3. Create distance between perpetrators and their actions

Passive voice patterns we detect:
  Pattern 1: auxpass + past participle
    e.g. "was arrested", "were killed", "is being investigated"
  Pattern 2: nsubjpass (passive nominal subject)
    e.g. "The bill was passed", "Citizens were ignored"
"""

import spacy
from collections import Counter


# ── 1. Passive Voice Patterns ─────────────────────────────────────────────────

def is_passive_sentence(sent) -> dict:
    """
    Checks a single spaCy sentence span for passive voice.

    Detection strategy:
      - Look for tokens with dep_ == 'auxpass' (passive auxiliary)
        e.g. "was", "were", "is being", "has been"
      - Look for tokens with dep_ == 'nsubjpass' (passive subject)
      - Both patterns reliably signal passive constructions

    Returns
    -------
    dict:
        is_passive  : bool
        pattern     : str  — which pattern triggered
        aux_verb    : str  — the auxiliary verb found
        main_verb   : str  — the main verb of the construction
        full_sent   : str  — the full sentence text
    """
    for token in sent:
        # Pattern 1 — auxpass dependency
        if token.dep_ == "auxpass":
            main_verb = token.head.text
            return {
                "is_passive": True,
                "pattern":    "auxpass",
                "aux_verb":   token.text,
                "main_verb":  main_verb,
                "full_sent":  sent.text.strip(),
            }

        # Pattern 2 — passive nominal subject
        if token.dep_ == "nsubjpass":
            main_verb = token.head.text
            return {
                "is_passive": True,
                "pattern":    "nsubjpass",
                "aux_verb":   "",
                "main_verb":  main_verb,
                "full_sent":  sent.text.strip(),
            }

    return {
        "is_passive": False,
        "pattern":    "",
        "aux_verb":   "",
        "main_verb":  "",
        "full_sent":  sent.text.strip(),
    }


# ── 2. Responsibility Gap Detector ────────────────────────────────────────────

def has_responsibility_gap(sent_text: str) -> bool:
    """
    Detects 'agentless passives' — passive constructions where
    the agent (the 'by X' phrase) is completely omitted.

    e.g.
      "Protesters were shot"         → agentless (gap = True)
      "Protesters were shot by police" → agent present (gap = False)

    This is the most concerning form of passive bias because it
    completely erases who performed the action.
    """
    sent_lower = sent_text.lower()

    # If passive but no 'by' phrase → agentless
    passive_indicators = ["was ", "were ", "is being ", "are being ",
                          "has been ", "have been ", "had been "]
    has_passive = any(ind in sent_lower for ind in passive_indicators)
    has_agent   = " by " in sent_lower

    return has_passive and not has_agent


# ── 3. Core Analysis Function ─────────────────────────────────────────────────

def analyze_passive_voice(text: str, nlp) -> dict:
    """
    Full passive voice analysis on a block of text.

    Parameters
    ----------
    text : str  — article text
    nlp  : spaCy Language model

    Returns
    -------
    dict:
        passive_sentences    : list of passive sentence dicts
        active_sentences     : list of active sentence texts
        total_sentences      : int
        passive_count        : int
        passive_rate         : float  0.0–1.0
        responsibility_gaps  : list of agentless passive sentences
        gap_count            : int
        bias_signal          : str   plain-English assessment
        score                : float 0.0–1.0 (normalized bias signal)
        label                : str   severity label
    """
    if not text or not text.strip():
        return _empty_passive_result()

    doc = nlp(text)
    sentences = list(doc.sents)
    total     = len(sentences)

    if total == 0:
        return _empty_passive_result()

    passive_sentences   = []
    active_sentences    = []
    responsibility_gaps = []

    for sent in sentences:
        result = is_passive_sentence(sent)

        if result["is_passive"]:
            passive_sentences.append(result)

            # Check for responsibility gap
            if has_responsibility_gap(sent.text):
                responsibility_gaps.append(sent.text.strip())
        else:
            active_sentences.append(sent.text.strip())

    passive_count = len(passive_sentences)
    passive_rate  = round(passive_count / total, 4) if total > 0 else 0.0
    gap_count     = len(responsibility_gaps)

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Score combines passive rate + responsibility gap penalty
    # Gap sentences are weighted more heavily because they
    # completely erase agency — the most biased form of passive

    gap_penalty = min(gap_count * 0.1, 0.3)   # max 0.3 penalty
    score       = round(min(passive_rate + gap_penalty, 1.0), 4)

    # ── Labels ────────────────────────────────────────────────────────────────
    if score < 0.15:
        label = "Minimal Passive Voice"
    elif score < 0.30:
        label = "Moderate Passive Voice"
    elif score < 0.50:
        label = "High Passive Voice Usage"
    else:
        label = "Excessive Passive Voice — Strong Bias Signal"

    # ── Plain-English bias signal ─────────────────────────────────────────────
    signal_parts = []

    if passive_count == 0:
        bias_signal = "No passive voice detected. Article uses clear, direct language."
    else:
        signal_parts.append(
            f"{passive_count} of {total} sentences use passive voice "
            f"({passive_rate*100:.0f}%)."
        )
        if gap_count > 0:
            signal_parts.append(
                f"{gap_count} sentence(s) use agentless passive — "
                f"responsibility is completely obscured."
            )
        bias_signal = " ".join(signal_parts)

    return {
        "passive_sentences":   passive_sentences,
        "active_sentences":    active_sentences,
        "total_sentences":     total,
        "passive_count":       passive_count,
        "passive_rate":        passive_rate,
        "responsibility_gaps": responsibility_gaps,
        "gap_count":           gap_count,
        "bias_signal":         bias_signal,
        "score":               score,
        "label":               label,
    }


# ── 4. Helpers ────────────────────────────────────────────────────────────────

def _empty_passive_result() -> dict:
    return {
        "passive_sentences":   [],
        "active_sentences":    [],
        "total_sentences":     0,
        "passive_count":       0,
        "passive_rate":        0.0,
        "responsibility_gaps": [],
        "gap_count":           0,
        "bias_signal":         "No text provided.",
        "score":               0.0,
        "label":               "N/A",
    }


# ── 5. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    TEST_CASES = {
        "High Passive / Responsibility Hidden": (
            "Three protesters were killed during the demonstration. "
            "Several activists were arrested and detained overnight. "
            "The building was set on fire. "
            "Tear gas was deployed against the crowd. "
            "Dozens of people were injured in the clashes."
        ),
        "Active Voice / Clear Responsibility": (
            "Police officers shot three protesters during the demonstration. "
            "Officers arrested several activists and detained them overnight. "
            "Rioters set the building on fire. "
            "Officers deployed tear gas against the crowd. "
            "The clashes injured dozens of people."
        ),
        "Mixed — Typical News Article": (
            "The president signed the new bill into law on Monday. "
            "The legislation was passed by Congress last week. "
            "Critics say the policy will hurt working families. "
            "The measure was opposed by several senators. "
            "Officials announced the changes would take effect immediately."
        ),
        "Agentless Passive — Worst Case": (
            "Civilians were killed in the overnight strikes. "
            "Homes were destroyed and infrastructure was damaged. "
            "Aid workers were prevented from entering the area. "
            "Food and medicine were seized at the border checkpoint. "
            "No explanation was given for the delays."
        ),
    }

    for name, text in TEST_CASES.items():
        result = analyze_passive_voice(text, nlp)

        print(f"\n{'═'*65}")
        print(f"  PASSIVE VOICE REPORT — {name}")
        print(f"{'═'*65}")
        print(f"  Score       : {result['score']:.0%}")
        print(f"  Label       : {result['label']}")
        print(f"  Bias Signal : {result['bias_signal']}")

        if result["passive_sentences"]:
            print(f"\n  ── Passive Sentences ────────────────────────────")
            for s in result["passive_sentences"]:
                gap = " ⚠️ [NO AGENT]" if has_responsibility_gap(
                    s["full_sent"]) else ""
                print(f"    • {s['full_sent']}{gap}")

        if result["responsibility_gaps"]:
            print(f"\n  ── Agentless Passives (Most Concerning) ─────────")
            for gap in result["responsibility_gaps"]:
                print(f"    🚨 {gap}")

    print(f"\n{'═'*65}\n")
