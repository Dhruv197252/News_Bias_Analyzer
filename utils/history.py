"""
utils/history.py
-----------------
Saves every article analysis to a local CSV log so the user can
track bias patterns across sources over time.

Each row in the log represents one full article analysis and captures:
  - Timestamp + source URL / label
  - Composite score + verdict
  - Per-engine breakdown (ML, emotion, subjectivity, passive, lexicon)
  - Quote vs opinion ratio
  - Hedge score (if hedge_detector was run)
  - Top loaded words (comma-separated)
  - NER summary (entity count + any negatively framed entities)

Design decisions
----------------
  • CSV not SQLite — opens directly in Excel, no DB dependency
  • Append-only — never overwrites existing rows, safe to run concurrently
  • One row per article — headline + body treated as one analysis unit
  • Section breakdown (beginning/middle/end) stored as JSON in one column
    so the CSV stays flat and readable without losing granularity
  • All float columns rounded to 4 decimal places for readability
  • history.py has zero imports from scorer.py to avoid circular imports —
    it receives a plain dict and serialises it

Public API
----------
  save_analysis(result_dict, url, source_label)  → row dict written
  load_history(path)                             → pd.DataFrame
  clear_history(path)                            → deletes the file
  summary_stats(path)                            → dict of aggregate stats
"""

import csv
import json
import os
import re
from datetime import datetime

import pandas as pd


# ── 1. Config ─────────────────────────────────────────────────────────────────

DEFAULT_LOG_PATH = "data/analysis_history.csv"

# Columns written to CSV — order matches the header row
CSV_COLUMNS = [
    "timestamp",
    "source_label",
    "url",
    "headline",
    "composite_score",
    "verdict",
    "ml_probability",
    "emotion_intensity",
    "subjectivity_score",
    "passive_score",
    "lexicon_score",
    "loaded_word_count",
    "top_loaded_words",
    "opinion_ratio",
    "opinion_label",
    "quote_ratio",
    "attribution_ratio",
    "hedge_score",
    "hedge_label",
    "ner_entity_count",
    "negatively_framed_entities",
    "word_count",
    "section_breakdown",     # JSON blob: {beginning, middle, end} scores
]


# ── 2. Save ───────────────────────────────────────────────────────────────────

def save_analysis(
    results:      dict,
    url:          str  = "",
    source_label: str  = "",
    headline:     str  = "",
    hedge_result        = None,   # HedgeResult dataclass or None
    log_path:     str  = DEFAULT_LOG_PATH,
) -> dict:
    """
    Serialise a scorer.py result dict into a single CSV row and append it
    to the history log.

    Parameters
    ----------
    results      : dict returned by analyze_article()
                   Must contain keys: overall, headline, beginning, middle, end
    url          : source URL (empty string if raw text mode)
    source_label : human-readable label e.g. "Reuters", "Raw Text"
    headline     : article headline string
    hedge_result : HedgeResult dataclass from hedge_detector.py (optional)
    log_path     : path to the CSV log file

    Returns
    -------
    dict — the row that was written (useful for display/testing)
    """
    overall = results.get("overall", {})
    if not overall:
        raise ValueError("results dict must contain an 'overall' key.")

    # ── Extract fields ────────────────────────────────────────────────────────
    passive_score = overall.get("passive", {}).get("score", 0.0)
    ner_data      = overall.get("ner", {})
    qo            = overall.get("quote_opinion", {})

    # Top loaded words — up to 6 unique words, comma-separated
    loaded_words = overall.get("loaded_words", [])
    unique_loaded = list({w for w, _ in loaded_words}) if loaded_words else []
    top_loaded_str = ", ".join(unique_loaded[:6])

    # Negatively framed entities — pipe-separated list
    neg_framed = ner_data.get("negatively_framed", [])
    neg_framed_str = " | ".join(neg_framed[:5]) if neg_framed else ""

    # Section breakdown — compact JSON blob
    section_breakdown = json.dumps({
        "beginning": results.get("beginning", {}).get("composite_score", 0.0),
        "middle":    results.get("middle",    {}).get("composite_score", 0.0),
        "end":       results.get("end",       {}).get("composite_score", 0.0),
    })

    # Hedge fields (optional — defaults to 0 if not run)
    hedge_score = 0.0
    hedge_label = "Not analysed"
    if hedge_result is not None:
        # Accept both dataclass and dict
        if hasattr(hedge_result, "hedge_score"):
            hedge_score = hedge_result.hedge_score
            hedge_label = hedge_result.hedge_label
        elif isinstance(hedge_result, dict):
            hedge_score = hedge_result.get("hedge_score", 0.0)
            hedge_label = hedge_result.get("hedge_label", "Not analysed")

    # Approximate word count from body text preview
    body_preview = overall.get("text_preview", "")
    word_count   = len(body_preview.split())   # rough; full text not stored

    # ── Build row ─────────────────────────────────────────────────────────────
    row = {
        "timestamp":                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_label":              source_label or _infer_source(url),
        "url":                       url,
        "headline":                  headline[:200],   # cap at 200 chars
        "composite_score":           round(overall.get("composite_score", 0.0), 4),
        "verdict":                   overall.get("verdict", ""),
        "ml_probability":            round(overall.get("ml_probability", 0.0), 4),
        "emotion_intensity":         round(overall.get("emotion_intensity", 0.0), 4),
        "subjectivity_score":        round(overall.get("subjectivity_score", 0.0), 4),
        "passive_score":             round(passive_score, 4),
        "lexicon_score":             round(overall.get("lexicon_score", 0.0), 4),
        "loaded_word_count":         overall.get("loaded_word_count", 0),
        "top_loaded_words":          top_loaded_str,
        "opinion_ratio":             round(qo.get("opinion_ratio", 0.0), 4),
        "opinion_label":             qo.get("opinion_label", ""),
        "quote_ratio":               round(qo.get("quote_ratio", 0.0), 4),
        "attribution_ratio":         round(qo.get("attribution_ratio", 0.0), 4),
        "hedge_score":               round(hedge_score, 4),
        "hedge_label":               hedge_label,
        "ner_entity_count":          ner_data.get("total_unique", 0),
        "negatively_framed_entities": neg_framed_str,
        "word_count":                word_count,
        "section_breakdown":         section_breakdown,
    }

    # ── Write to CSV ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()   # write column names on first run only
        writer.writerow(row)

    print(f"📝 Analysis saved to '{log_path}' "
          f"[{row['source_label']} | {row['verdict']} | "
          f"{row['composite_score']:.0%}]")

    return row


# ── 3. Load ───────────────────────────────────────────────────────────────────

def load_history(log_path: str = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """
    Load the full history log as a DataFrame.

    Returns an empty DataFrame with correct columns if the file
    does not exist yet (safe to call on first run).
    """
    if not os.path.isfile(log_path):
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.read_csv(log_path, encoding="utf-8")

    # Parse timestamp column if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


# ── 4. Clear ──────────────────────────────────────────────────────────────────

def clear_history(log_path: str = DEFAULT_LOG_PATH) -> None:
    """
    Delete the history log entirely.
    Useful for testing or resetting from the dashboard.
    """
    if os.path.isfile(log_path):
        os.remove(log_path)
        print(f"🗑️  History log '{log_path}' cleared.")
    else:
        print(f"ℹ️  No history log found at '{log_path}' — nothing to clear.")


# ── 5. Summary Stats ──────────────────────────────────────────────────────────

def summary_stats(log_path: str = DEFAULT_LOG_PATH) -> dict:
    """
    Compute aggregate statistics over all saved analyses.

    Returns
    -------
    dict:
        total_analyses    : int
        avg_composite     : float
        avg_ml_prob       : float
        avg_opinion_ratio : float
        most_biased_source: str   (source_label with highest mean composite)
        verdict_counts    : dict  {verdict: count}
        date_range        : dict  {first: str, last: str}
    """
    df = load_history(log_path)

    if df.empty:
        return {
            "total_analyses":     0,
            "avg_composite":      0.0,
            "avg_ml_prob":        0.0,
            "avg_opinion_ratio":  0.0,
            "most_biased_source": "N/A",
            "verdict_counts":     {},
            "date_range":         {"first": "N/A", "last": "N/A"},
        }

    # Most biased source — source with highest mean composite score
    source_means = (
        df.groupby("source_label")["composite_score"]
        .mean()
        .sort_values(ascending=False)
    )
    most_biased = source_means.index[0] if not source_means.empty else "N/A"

    verdict_counts = (
        df["verdict"].value_counts().to_dict()
        if "verdict" in df.columns else {}
    )

    ts = df["timestamp"].dropna()
    date_range = {
        "first": str(ts.min())[:19] if not ts.empty else "N/A",
        "last":  str(ts.max())[:19] if not ts.empty else "N/A",
    }

    return {
        "total_analyses":     len(df),
        "avg_composite":      round(df["composite_score"].mean(), 4),
        "avg_ml_prob":        round(df["ml_probability"].mean(), 4),
        "avg_opinion_ratio":  round(df["opinion_ratio"].mean(), 4),
        "most_biased_source": most_biased,
        "verdict_counts":     verdict_counts,
        "date_range":         date_range,
    }


# ── 6. Helpers ────────────────────────────────────────────────────────────────

def _infer_source(url: str) -> str:
    """
    Extract a clean source label from a URL.
    Falls back to "Raw Text" if url is empty.

    Examples
    --------
    "https://www.reuters.com/world/..."  →  "reuters.com"
    "https://theprint.in/politics/..."  →  "theprint.in"
    ""                                  →  "Raw Text"
    """
    if not url or not url.strip():
        return "Raw Text"
    # Strip scheme and www.
    domain = re.sub(r"^https?://(www\.)?", "", url)
    # Take only the domain part (up to first /)
    domain = domain.split("/")[0]
    return domain or "Unknown"




# ── 7. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.scorer import load_engines, analyze_article

    TEST_LOG = "data/test_history.csv"

    # Clear any leftover test log
    clear_history(TEST_LOG)

    pipeline, sia, nlp = load_engines()

    # ── Two sample articles ───────────────────────────────────────────────────
    ARTICLES = [
        {
            "url":      "https://reuters.com/sample",
            "headline": "Government Announces Infrastructure Spending Plan",
            "paras": [
                "The government announced a $2 billion infrastructure plan on Tuesday.",
                "Officials said the funds would be allocated over five years.",
                "The bill passed with bipartisan support in the Senate.",
                "Several amendments were proposed during the review process.",
            ],
        },
        {
            "url":      "https://example-opinion.com/article",
            "headline": "Corrupt Regime's Draconian Policy Threatens Democracy",
            "paras": [
                "The regime's shameful new policy has outraged civil society groups.",
                "Critics warn the draconian measures will devastate ordinary citizens.",
                "Radical officials have recklessly ignored expert warnings.",
                "This tyrannical administration must be held accountable immediately.",
            ],
        },
    ]

    print("\n── Running analyses and saving to history log ───────────\n")

    for article in ARTICLES:
        body_text = " ".join(article["paras"])
        result = analyze_article(
            headline   = article["headline"],
            body_text  = body_text,
            body_paras = article["paras"],
            pipeline   = pipeline,
            sia        = sia,
            nlp        = nlp,
        )
        save_analysis(
            results      = result,
            url          = article["url"],
            headline     = article["headline"],
            log_path     = TEST_LOG,
        )

    # ── Load and display ──────────────────────────────────────────────────────
    print("\n── History Log ──────────────────────────────────────────\n")
    df = load_history(TEST_LOG)
    display_cols = [
        "timestamp", "source_label", "verdict",
        "composite_score", "opinion_ratio", "opinion_label"
    ]
    print(df[display_cols].to_string(index=False))

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n── Summary Stats ────────────────────────────────────────\n")
    stats = summary_stats(TEST_LOG)
    for k, v in stats.items():
        print(f"   {k:<25} {v}")

    # Clean up test file
    clear_history(TEST_LOG)
    print("\n✅ Demo complete.\n")