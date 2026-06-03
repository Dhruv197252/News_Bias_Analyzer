"""
Step 8: Named Entity Recognition (NER) Engine
------------------------------------------------
Uses spaCy to detect named entities and how they are framed.
"""

import spacy
from collections import defaultdict, Counter

def load_nlp():
    return spacy.load("en_core_web_sm")

RELEVANT_ENTITY_TYPES = {
    "PERSON": "Person / Public Figure",
    "ORG":    "Organization / Institution",
    "GPE":    "Country / City / State",
    "NORP":   "Political / National Group",
    "LAW":    "Law / Policy / Bill",
    "EVENT":  "Named Event",
}

NEGATIVE_FRAME_WORDS = {
    "corrupt", "criminal", "radical", "extreme", "dangerous",
    "reckless", "draconian", "shameful", "disgraceful", "tyrannical",
    "incompetent", "dishonest", "failed", "weak", "disastrous",
    "catastrophic", "terrible", "horrible", "awful", "evil",
    "threat", "threatening", "destructive", "harmful", "toxic",
    "illegitimate", "authoritarian", "regime", "puppet", "stooge",
}

POSITIVE_FRAME_WORDS = {
    "heroic", "visionary", "landmark", "historic", "patriotic",
    "bold", "courageous", "triumphant", "excellent", "outstanding",
    "brilliant", "successful", "strong", "effective", "honest",
    "transparent", "democratic", "legitimate", "respected", "trusted",
}


def extract_entities(text: str, nlp) -> dict:
    if not text or not text.strip():
        return _empty_ner_result()

    doc = nlp(text)
    entity_counts  = Counter()
    by_type        = defaultdict(list)
    entity_labels  = {}

    for ent in doc.ents:
        if ent.label_ not in RELEVANT_ENTITY_TYPES:
            continue
        clean = ent.text.strip()
        if len(clean) < 2:
            continue
        entity_counts[clean] += 1
        entity_labels[clean]  = ent.label_
        if clean not in by_type[ent.label_]:
            by_type[ent.label_].append(clean)

    framing = defaultdict(lambda: {"positive": 0, "negative": 0, "net": 0})
    for sent in doc.sents:
        sent_text  = sent.text.lower()
        sent_words = set(sent_text.split())
        for ent in sent.ents:
            if ent.label_ not in RELEVANT_ENTITY_TYPES:
                continue
            clean = ent.text.strip()
            pos_hits = len(sent_words & POSITIVE_FRAME_WORDS)
            neg_hits = len(sent_words & NEGATIVE_FRAME_WORDS)
            framing[clean]["positive"] += pos_hits
            framing[clean]["negative"] += neg_hits
            framing[clean]["net"]      += (pos_hits - neg_hits)

    framing_final = {}
    for entity, scores in framing.items():
        net = scores["net"]
        if net > 0:
            frame_label = "Positive Framing"
        elif net < 0:
            frame_label = "Negative Framing"
        else:
            frame_label = "Neutral Framing"

        framing_final[entity] = {
            "positive":    scores["positive"],
            "negative":    scores["negative"],
            "net":         net,
            "frame_label": frame_label,
        }

    most_mentioned = [
        {
            "text":       ent,
            "count":      count,
            "type":       RELEVANT_ENTITY_TYPES.get(entity_labels.get(ent, ""), "Unknown"),
            "type_code":  entity_labels.get(ent, ""),
            "framing":    framing_final.get(ent, {"frame_label": "Neutral Framing"}),
        }
        for ent, count in entity_counts.most_common(5)
    ]

    total_entities = len(entity_counts)
    negatively_framed = [e for e, f in framing_final.items() if f["frame_label"] == "Negative Framing"]
    positively_framed = [e for e, f in framing_final.items() if f["frame_label"] == "Positive Framing"]

    summary_parts = []
    if total_entities == 0:
        summary = "No named entities detected."
    else:
        summary_parts.append(f"{total_entities} named entities detected across {len(by_type)} categories.")
        if negatively_framed:
            summary_parts.append(f"Negative framing detected around: {', '.join(negatively_framed[:3])}.")
        if positively_framed:
            summary_parts.append(f"Positive framing detected around: {', '.join(positively_framed[:3])}.")
        summary = " ".join(summary_parts)

    return {
        "entities":       most_mentioned,
        "entity_counts":  dict(entity_counts),
        "by_type":        dict(by_type),
        "framing":        framing_final,
        "most_mentioned": most_mentioned,
        "total_unique":   total_entities,
        "summary":        summary,
        "negatively_framed": negatively_framed,
        "positively_framed": positively_framed,
    }


def _empty_ner_result() -> dict:
    return {
        "entities":          [],
        "entity_counts":     {},
        "by_type":           {},
        "framing":           {},
        "most_mentioned":    [],
        "total_unique":      0,
        "summary":           "No text provided.",
        "negatively_framed": [],
        "positively_framed": [],
    }
