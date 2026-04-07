import re
from collections import Counter

LOADED_LEXICON: dict[str, list[str]] = {
    "authoritarian_framing": [
        "regime", "dictatorship", "tyranny", "puppet", "stooge",
        "junta", "autocrat", "despot", "overlord", "iron-fisted",
    ],
    "alarmist_language": [
        "catastrophic", "devastating", "apocalyptic", "crisis",
        "collapse", "chaos", "meltdown", "explosive", "alarming",
        "dire", "doomsday",
    ],
    "moral_condemnation": [
        "draconian", "recklessly", "shameful", "outrageous",
        "disgraceful", "unconscionable", "vile", "corrupt",
        "criminal", "immoral", "scandalous",
    ],
    "loaded_positive_spin": [
        "heroic", "visionary", "landmark", "historic",
        "unprecedented", "patriotic", "bold", "courageous",
        "triumphant", "savior",
    ],
    "tribal_othering": [
        "radical", "extremist", "socialist", "elitist", "globalist",
        "nationalist", "far-left", "far-right", "woke", "cult",
    ],
}

WORD_TO_CATEGORY: dict[str, str] = {
    word: category
    for category, words in LOADED_LEXICON.items()
    for word in words
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def score_label(score: float) -> str:
    if score == 0.0:
        return "Clean / No loaded language detected"
    elif score < 0.02:
        return "Minimal Bias Signal"
    elif score < 0.05:
        return "Moderate Bias Signal"
    elif score < 0.10:
        return "Strong Bias Signal"
    else:
        return "Highly Loaded Language"


def scan_text(text: str) -> dict:
    tokens = tokenize(text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return {
            "total_tokens": 0,
            "matched_words": [],
            "unique_matches": 0,
            "naive_score": 0.0,
            "label": "Empty input",
            "category_counts": Counter(),
        }

    matched_words: list[tuple[str, str]] = [
        (token, WORD_TO_CATEGORY[token])
        for token in tokens
        if token in WORD_TO_CATEGORY
    ]

    category_counts = Counter(category for _, category in matched_words)
    naive_score = len(matched_words) / total_tokens

    return {
        "total_tokens": total_tokens,
        "matched_words": matched_words,
        "unique_matches": len(set(w for w, _ in matched_words)),
        "naive_score": round(naive_score, 4),
        "label": score_label(naive_score),
        "category_counts": category_counts,
    }


def print_report(result: dict, source_label: str = "Input Text") -> None:
    print(f"\n{'═'*60}")
    print(f"  LEXICON SCAN REPORT — {source_label}")
    print(f"{'═'*60}")
    print(f"  Total tokens analysed : {result['total_tokens']}")
    print(f"  Loaded words found    : {len(result['matched_words'])} "
          f"({result['unique_matches']} unique)")
    print(f"  Naive bias score      : {result['naive_score']*100:.2f}%")
    print(f"  Severity label        : {result['label']}")
    if result["matched_words"]:
        print(f"\n  ── Detected Words ──────────────────────────────")
        for word, category in sorted(set(result["matched_words"])):
            print(f"    • '{word}'  →  [{category}]")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    SAMPLE_TEXTS = {
        "High Bias": (
            "The regime recklessly imposed draconian measures, "
            "triggering a catastrophic collapse of civil liberties. "
            "The radical extremist faction has shamefully undermined "
            "the heroic efforts of patriotic citizens."
        ),
        "Low Bias": (
            "The government announced new tax legislation on Tuesday. "
            "Officials said the bill would be reviewed by a parliamentary "
            "committee before a final vote next month."
        ),
    }
    for label, text in SAMPLE_TEXTS.items():
        result = scan_text(text)
        print_report(result, source_label=label)