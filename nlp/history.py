"""
utils/history.py
-----------------
Saves every article analysis to a local CSV log.
"""

import csv
import json
import os
import re
from datetime import datetime
import pandas as pd

DEFAULT_LOG_PATH = "data/analysis_history.csv"

CSV_COLUMNS = [
    "timestamp", "source_label", "url", "headline", "composite_score", "verdict",
    "ml_probability", "emotion_intensity", "subjectivity_score", "passive_score",
    "lexicon_score", "loaded_word_count", "top_loaded_words", "opinion_ratio",
    "opinion_label", "quote_ratio", "attribution_ratio", "hedge_score",
    "hedge_label", "ner_entity_count", "negatively_framed_entities", "word_count",
    "section_breakdown",
]


def save_analysis(
    results:      dict,
    url:          str  = "",
    source_label: str  = "",
    headline:     str  = "",
    hedge_result        = None,
    log_path:     str  = DEFAULT_LOG_PATH,
) -> dict:
    overall = results.get("overall", {})
    if not overall:
        raise ValueError("results dict must contain an 'overall' key.")

    passive_score = overall.get("passive", {}).get("score", 0.0)
    ner_data      = overall.get("ner", {})
    qo            = overall.get("quote_opinion", {})

    loaded_words = overall.get("loaded_words", [])
    unique_loaded = list({w for w, _ in loaded_words}) if loaded_words else []
    top_loaded_str = ", ".join(unique_loaded[:6])

    neg_framed = ner_data.get("negatively_framed", [])
    neg_framed_str = " | ".join(neg_framed[:5]) if neg_framed else ""

    section_breakdown = json.dumps({
        "beginning": results.get("beginning", {}).get("composite_score", 0.0),
        "middle":    results.get("middle",    {}).get("composite_score", 0.0),
        "end":       results.get("end",       {}).get("composite_score", 0.0),
    })

    hedge_score = 0.0
    hedge_label = "Not analysed"
    if hedge_result is not None:
        if hasattr(hedge_result, "hedge_score"):
            hedge_score = hedge_result.hedge_score
            hedge_label = hedge_result.hedge_label
        elif isinstance(hedge_result, dict):
            hedge_score = hedge_result.get("hedge_score", 0.0)
            hedge_label = hedge_result.get("hedge_label", "Not analysed")

    body_preview = overall.get("text_preview", "")
    word_count   = len(body_preview.split())

    row = {
        "timestamp":                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_label":              source_label or _infer_source(url),
        "url":                       url,
        "headline":                  headline[:200],
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

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def load_history(log_path: str = DEFAULT_LOG_PATH) -> pd.DataFrame:
    if not os.path.isfile(log_path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(log_path, encoding="utf-8")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def clear_history(log_path: str = DEFAULT_LOG_PATH) -> None:
    if os.path.isfile(log_path):
        os.remove(log_path)


def summary_stats(log_path: str = DEFAULT_LOG_PATH) -> dict:
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

    source_means = df.groupby("source_label")["composite_score"].mean().sort_values(ascending=False)
    most_biased = source_means.index[0] if not source_means.empty else "N/A"
    verdict_counts = df["verdict"].value_counts().to_dict() if "verdict" in df.columns else {}
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


def _infer_source(url: str) -> str:
    if not url or not url.strip():
        return "Raw Text"
    domain = re.sub(r"^https?://(www\.)?", "", url)
    domain = domain.split("/")[0]
    return domain or "Unknown"
