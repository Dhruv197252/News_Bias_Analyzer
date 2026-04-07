"""
Step 8: Named Entity Recognition (NER) Engine
------------------------------------------------
Uses spaCy to detect WHICH real-world entities are being
talked about in a news article, and HOW they are framed.

Why this matters for bias detection:
  - Biased articles often use loaded language DIRECTED AT
    specific named entities (politicians, countries, orgs)
  - Detecting entities lets us say "this article uses 4 negative
    loaded words specifically targeting [PERSON: Trump]"
  - This is far more informative than just a raw bias score

Entity types we care about:
  PERSON   → Politicians, public figures
  ORG      → Governments, corporations, parties
  GPE      → Countries, cities, states
  NORP     → Nationalities, political groups
  LAW      → Bills, policies, regulations
  EVENT    → Named events (elections, wars)
"""

import spacy
from collections import defaultdict, Counter

# ── 1. Load spaCy Model ───────────────────────────────────────────────────────
def load_nlp():
    return spacy.load("en_core_web_sm")

# ── 2. Entity Types We Care About ────────────────────────────────────────────

RELEVANT_ENTITY_TYPES = {
    "PERSON": "Person / Public Figure",
    "ORG":    "Organization / Institution",
    "GPE":    "Country / City / State",
    "NORP":   "Political / National Group",
    "LAW":    "Law / Policy / Bill",
    "EVENT":  "Named Event",
}

# ── 3. Sentiment Word Lists for Entity-Level Framing ─────────────────────────
# These are used to detect whether an entity is being framed
# positively or negatively in the sentences that mention it.

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


# ── 4. Core NER Function ──────────────────────────────────────────────────────

def extract_entities(text: str, nlp) -> dict:
    """
    Runs spaCy NER on text and returns structured entity data.

    Parameters
    ----------
    text : str  — any block of text
    nlp  : spaCy Language model

    Returns
    -------
    dict:
        entities        : list of dicts (text, label, count)
        entity_counts   : Counter  {entity_text: count}
        by_type         : dict     {entity_type: [entity_texts]}
        framing         : dict     {entity_text: {positive, negative, net}}
        most_mentioned  : list     top 5 most mentioned entities
        summary         : str      plain-English summary
    """
    if not text or not text.strip():
        return _empty_ner_result()

    doc = nlp(text)

    # ── Collect all relevant entities ────────────────────────────────────────
    entity_counts  = Counter()
    by_type        = defaultdict(list)
    entity_labels  = {}   # entity_text → entity_type

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

    # ── Entity-level framing analysis ────────────────────────────────────────
    # For each sentence, check if it mentions an entity AND
    # contains positive/negative framing words
    framing = defaultdict(lambda: {"positive": 0, "negative": 0, "net": 0})

    for sent in doc.sents:
        sent_text  = sent.text.lower()
        sent_words = set(sent_text.split())

        # Which entities appear in this sentence?
        for ent in sent.ents:
            if ent.label_ not in RELEVANT_ENTITY_TYPES:
                continue
            clean = ent.text.strip()

            pos_hits = len(sent_words & POSITIVE_FRAME_WORDS)
            neg_hits = len(sent_words & NEGATIVE_FRAME_WORDS)

            framing[clean]["positive"] += pos_hits
            framing[clean]["negative"] += neg_hits
            framing[clean]["net"]      += (pos_hits - neg_hits)

    # ── Build final framing dict ──────────────────────────────────────────────
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

    # ── Top entities ──────────────────────────────────────────────────────────
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

    # ── Plain-English summary ─────────────────────────────────────────────────
    total_entities = len(entity_counts)
    negatively_framed = [
        e for e, f in framing_final.items()
        if f["frame_label"] == "Negative Framing"
    ]
    positively_framed = [
        e for e, f in framing_final.items()
        if f["frame_label"] == "Positive Framing"
    ]

    summary_parts = []
    if total_entities == 0:
        summary = "No named entities detected."
    else:
        summary_parts.append(
            f"{total_entities} named entities detected across "
            f"{len(by_type)} categories."
        )
        if negatively_framed:
            summary_parts.append(
                f"Negative framing detected around: "
                f"{', '.join(negatively_framed[:3])}."
            )
        if positively_framed:
            summary_parts.append(
                f"Positive framing detected around: "
                f"{', '.join(positively_framed[:3])}."
            )
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


# ── 5. Helpers ────────────────────────────────────────────────────────────────

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


# ── 6. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nlp = load_nlp()

    TEST_ARTICLES = {
        "High Bias": (
            "The corrupt Biden administration recklessly pushed a draconian "
            "bill through Congress, threatening the democratic values that "
            "American citizens hold dear. Republican senators called the move "
            "shameful and dangerous. The White House refused to comment on "
            "the catastrophic implications for the United States economy."
        ),
        "Balanced": (
            "President Biden signed the infrastructure bill into law on "
            "Monday. Republican and Democratic senators both praised aspects "
            "of the legislation. The White House said the bill represents "
            "a historic investment in American infrastructure. Critics in "
            "Congress raised concerns about the overall cost."
        ),
        "Negative Foreign Framing": (
            "The Chinese regime has recklessly militarized the South China Sea "
            "threatening neighboring countries including Taiwan and Japan. "
            "NATO allies warned that Beijing's authoritarian expansion poses "
            "a catastrophic threat to global stability and democracy."
        ),
    }

    for name, text in TEST_ARTICLES.items():
        result = extract_entities(text, nlp)

        print(f"\n{'═'*65}")
        print(f"  NER REPORT — {name}")
        print(f"{'═'*65}")
        print(f"  Summary : {result['summary']}")

        if result["most_mentioned"]:
            print(f"\n  ── Top Entities ─────────────────────────────────")
            for ent in result["most_mentioned"]:
                frame = ent["framing"].get("frame_label", "Neutral")
                print(f"    • '{ent['text']}'")
                print(f"      Type    : {ent['type']}")
                print(f"      Mentions: {ent['count']}")
                print(f"      Framing : {frame}")

        if result["by_type"]:
            print(f"\n  ── Entities by Category ─────────────────────────")
            for etype, entities in result["by_type"].items():
                label = RELEVANT_ENTITY_TYPES.get(etype, etype)
                print(f"    [{label}]: {', '.join(entities)}")

    print(f"\n{'═'*65}\n")
    
    
    
    
    
    
    
    
    
    