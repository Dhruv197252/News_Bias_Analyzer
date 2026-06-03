"""
Step 9: Passive Voice Detector
--------------------------------
Uses spaCy's dependency parser to detect passive voice constructions in news text.
"""

import spacy
from collections import Counter


def is_passive_sentence(sent) -> dict:
    for token in sent:
        if token.dep_ == "auxpass":
            main_verb = token.head.text
            return {
                "is_passive": True,
                "pattern":    "auxpass",
                "aux_verb":   token.text,
                "main_verb":  main_verb,
                "full_sent":  sent.text.strip(),
            }
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


def has_responsibility_gap(sent_text: str) -> bool:
    sent_lower = sent_text.lower()
    passive_indicators = ["was ", "were ", "is being ", "are being ",
                          "has been ", "have been ", "had been "]
    has_passive = any(ind in sent_lower for ind in passive_indicators)
    has_agent   = " by " in sent_lower
    return has_passive and not has_agent


def analyze_passive_voice(text: str, nlp) -> dict:
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
            if has_responsibility_gap(sent.text):
                responsibility_gaps.append(sent.text.strip())
        else:
            active_sentences.append(sent.text.strip())

    passive_count = len(passive_sentences)
    passive_rate  = round(passive_count / total, 4) if total > 0 else 0.0
    gap_count     = len(responsibility_gaps)

    gap_penalty = min(gap_count * 0.1, 0.3)
    score       = round(min(passive_rate + gap_penalty, 1.0), 4)

    if score < 0.15:
        label = "Minimal Passive Voice"
    elif score < 0.30:
        label = "Moderate Passive Voice"
    elif score < 0.50:
        label = "High Passive Voice Usage"
    else:
        label = "Excessive Passive Voice — Strong Bias Signal"

    signal_parts = []
    if passive_count == 0:
        bias_signal = "No passive voice detected. Article uses clear, direct language."
    else:
        signal_parts.append(f"{passive_count} of {total} sentences use passive voice ({passive_rate*100:.0f}%).")
        if gap_count > 0:
            signal_parts.append(f"{gap_count} sentence(s) use agentless passive — responsibility is completely obscured.")
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
