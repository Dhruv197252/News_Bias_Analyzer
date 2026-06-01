"""
MBFC Domain Loader
-------------------
Loads Media Bias / Fact Check domain ratings.
Used for domain-level enrichment (NOT ML training).
Outputs: data/mbfc_domains.json
"""

import os
import json
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT, "data", "mbfc_domains.json")


# ── Curated MBFC domains (top global + Indian outlets) ────────────────────────
# Source: Media Bias / Fact Check (mbfc.com) — manually curated for reliability

MBFC_DOMAINS = {
    # ── Global ────────────────────────────────────────────────────────────────
    "cnn.com":          {"leaning": "Left",         "factual": "High",      "country": "US"},
    "foxnews.com":      {"leaning": "Right",        "factual": "Mixed",     "country": "US"},
    "bbc.com":          {"leaning": "Center",       "factual": "Very High", "country": "UK"},
    "bbc.co.uk":        {"leaning": "Center",       "factual": "Very High", "country": "UK"},
    "reuters.com":      {"leaning": "Center",       "factual": "Very High", "country": "UK"},
    "apnews.com":       {"leaning": "Center",       "factual": "Very High", "country": "US"},
    "nytimes.com":      {"leaning": "Left-Center",  "factual": "High",      "country": "US"},
    "washingtonpost.com":{"leaning":"Left-Center",  "factual": "High",      "country": "US"},
    "wsj.com":          {"leaning": "Right-Center", "factual": "High",      "country": "US"},
    "theguardian.com":  {"leaning": "Left-Center",  "factual": "High",      "country": "UK"},
    "breitbart.com":    {"leaning": "Extreme Right","factual": "Low",       "country": "US"},
    "infowars.com":     {"leaning": "Extreme Right","factual": "Very Low",  "country": "US"},
    "huffpost.com":     {"leaning": "Left",         "factual": "Mixed",     "country": "US"},
    "aljazeera.com":    {"leaning": "Left-Center",  "factual": "High",      "country": "QA"},
    "bloomberg.com":    {"leaning": "Center",       "factual": "High",      "country": "US"},

    # ── Indian Outlets ────────────────────────────────────────────────────────
    "ndtv.com":         {"leaning": "Left-Center",  "factual": "High",      "country": "IN"},
    "thehindu.com":     {"leaning": "Left-Center",  "factual": "High",      "country": "IN"},
    "hindustantimes.com":{"leaning":"Center",       "factual": "Mostly",    "country": "IN"},
    "timesofindia.com": {"leaning": "Center",       "factual": "Mostly",    "country": "IN"},
    "republicworld.com":{"leaning": "Right",        "factual": "Mixed",     "country": "IN"},
    "republic.tv":      {"leaning": "Right",        "factual": "Mixed",     "country": "IN"},
    "thewire.in":       {"leaning": "Left",         "factual": "High",      "country": "IN"},
    "scroll.in":        {"leaning": "Left-Center",  "factual": "High",      "country": "IN"},
    "theprint.in":      {"leaning": "Center",       "factual": "High",      "country": "IN"},
    "firstpost.com":    {"leaning": "Right-Center", "factual": "Mostly",    "country": "IN"},
    "news18.com":       {"leaning": "Right-Center", "factual": "Mostly",    "country": "IN"},
    "indiatoday.in":    {"leaning": "Center",       "factual": "High",      "country": "IN"},
    "zeenews.india.com":{"leaning": "Right",        "factual": "Mixed",     "country": "IN"},
    "outlookindia.com": {"leaning": "Left-Center",  "factual": "High",      "country": "IN"},
    "deccanherald.com": {"leaning": "Center",       "factual": "High",      "country": "IN"},
    "indianexpress.com":{"leaning": "Left-Center",  "factual": "High",      "country": "IN"},
    "livemint.com":     {"leaning": "Right-Center", "factual": "High",      "country": "IN"},
    "businessline.com": {"leaning": "Center",       "factual": "High",      "country": "IN"},
    "economictimes.indiatimes.com": {"leaning": "Center", "factual": "High","country": "IN"},
    "opindia.com":      {"leaning": "Extreme Right","factual": "Low",       "country": "IN"},
    "swarajyamag.com":  {"leaning": "Right",        "factual": "Mixed",     "country": "IN"},
    "thequint.com":     {"leaning": "Left-Center",  "factual": "High",      "country": "IN"},
    "newslaundry.com":  {"leaning": "Left",         "factual": "High",      "country": "IN"},
    "newsminute.com":   {"leaning": "Left-Center",  "factual": "High",      "country": "IN"},
    "aninews.in":       {"leaning": "Right",        "factual": "Mostly",    "country": "IN"},
    "ptinews.com":      {"leaning": "Center",       "factual": "High",      "country": "IN"},
    "wionews.com":      {"leaning": "Right-Center", "factual": "Mostly",    "country": "IN"},
}


def save_mbfc(data: dict, path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Saved {len(data)} MBFC domain entries to: {path}")


def load_mbfc(path: str = OUTPUT_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return MBFC_DOMAINS  # return built-in fallback


def get_domain_info(domain: str, path: str = OUTPUT_PATH) -> dict:
    """Lookup a domain in the MBFC database. Returns {} if not found."""
    data = load_mbfc(path)
    # Try exact match, then try stripping 'www.'
    return data.get(domain, data.get(domain.replace("www.", ""), {}))


if __name__ == "__main__":
    save_mbfc(MBFC_DOMAINS)
    print(f"\nSample lookup — ndtv.com:")
    print(get_domain_info("ndtv.com"))
