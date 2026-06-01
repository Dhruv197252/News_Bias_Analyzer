"""
Analysis Service
-----------------
Orchestrates all NLP engines and produces the unified analysis result.

Render free-tier RAM strategy:
  • All heavy objects (ML pipeline, spaCy, VADER) loaded ONCE as singletons
  • No transformer models loaded at startup
  • spaCy uses 'en_core_web_sm' (~50 MB, not en_core_web_trf)
"""

import logging
import uuid
from datetime import datetime, timezone

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Existing engine imports (reuse everything from the original codebase)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.ml_engine      import get_pipeline, predict_bias
from utils.scorer         import analyze_article, load_engines
from utils.scraper        import scrape_article, extract_domain
from backend.services.domain_service import get_domain_info_for

logger = logging.getLogger(__name__)

# ── Singletons (loaded once at first use) ─────────────────────────────────────

_pipeline = None
_sia      = None
_nlp      = None


def _get_engines():
    """Lazy-load all engines. Thread-safe enough for --workers 1."""
    global _pipeline, _sia, _nlp
    if _pipeline is None:
        logger.info("Loading ML engines (first request)...")
        _pipeline, _sia, _nlp = load_engines()
        # Ensure we use the best available model
        _pipeline = get_pipeline()
        logger.info("✅ All engines ready.")
    return _pipeline, _sia, _nlp


# ── Core Analysis ─────────────────────────────────────────────────────────────

def analyze_text(text: str, headline: str = "") -> dict:
    """
    Full analysis of plain text.
    Returns unified result dict ready for API response.
    """
    pipeline, sia, nlp = _get_engines()

    # If headline not provided, use first sentence as headline
    if not headline:
        sentences = text.split(".")
        headline = sentences[0].strip() if sentences else ""

    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Run full analysis
    result = analyze_article(
        headline   = headline,
        body_text  = text,
        body_paras = paragraphs,
        pipeline   = pipeline,
        sia        = sia,
        nlp        = nlp,
    )

    return _build_response(
        analysis_result = result,
        url             = None,
        domain          = None,
        headline        = headline,
        body_text       = text,
        word_count      = len(text.split()),
        scrape_engine   = "text_input",
    )


def analyze_url(url: str) -> dict:
    """
    Scrape URL and run full analysis.
    Uses multi-engine scraper (trafilatura → newspaper3k → bs4).
    """
    pipeline, sia, nlp = _get_engines()

    # Scrape
    scraped = scrape_article(url)
    if not scraped["success"]:
        return {
            "success": False,
            "error":   scraped["error"],
            "url":     url,
        }

    headline   = scraped["headline"]
    body_text  = scraped["body_text"]
    body_paras = scraped["body_paras"]
    domain     = extract_domain(url)

    # Run full analysis
    result = analyze_article(
        headline   = headline,
        body_text  = body_text,
        body_paras = body_paras,
        pipeline   = pipeline,
        sia        = sia,
        nlp        = nlp,
    )

    return _build_response(
        analysis_result = result,
        url             = url,
        domain          = domain,
        headline        = headline,
        body_text       = body_text,
        word_count      = scraped["word_count"],
        scrape_engine   = scraped.get("engine", "unknown"),
    )


# ── Response Builder ──────────────────────────────────────────────────────────

def _build_response(
    analysis_result: dict,
    url:             str | None,
    domain:          str | None,
    headline:        str,
    body_text:       str,
    word_count:      int,
    scrape_engine:   str,
) -> dict:
    """Flatten the nested analyze_article() result into a flat API response dict."""
    overall  = analysis_result.get("overall", {})
    headline_r = analysis_result.get("headline", {})

    composite   = overall.get("composite_score", 0.0)
    verdict     = overall.get("verdict", "Unknown")
    ml_prob     = overall.get("ml_probability", 0.0)
    ml_label    = overall.get("ml_label", 0)   # int: 0 or 1

    # If ml_label came back as string from old scorer, convert
    if isinstance(ml_label, str):
        ml_label = 1 if ml_label == "Biased" else 0

    engine_scores = {
        "ml_probability":    ml_prob,
        "ml_label":          ml_label,
        "ml_verdict":        "Biased" if ml_label == 1 else "Neutral",
        "emotion_intensity": overall.get("emotion_intensity", 0.0),
        "subjectivity_score": overall.get("subjectivity_score", 0.0),
        "passive_score":     overall.get("passive_score", 0.0),
        "lexicon_score":     min(overall.get("lexicon_score", 0.0) * 10, 1.0),
        "hedge_score":       overall.get("hedge_score", 0.0),
        # Detailed breakdowns
        "headline_score":    headline_r.get("composite_score", 0.0),
        "beginning_score":   analysis_result.get("beginning", {}).get("composite_score", 0.0),
        "middle_score":      analysis_result.get("middle", {}).get("composite_score", 0.0),
        "end_score":         analysis_result.get("end", {}).get("composite_score", 0.0),
        # Extra detail
        "loaded_words":      overall.get("loaded_words", []),
        "unique_loaded_words": overall.get("unique_loaded_words", 0),
        "hedge_label":       overall.get("hedge_label", ""),
        "opinion_ratio":     overall.get("quote_opinion", {}).get("opinion_ratio", 0.0),
        "opinion_label":     overall.get("quote_opinion", {}).get("opinion_label", ""),
        "ner":               overall.get("ner", {}),
    }

    # Domain enrichment
    domain_info = get_domain_info_for(domain) if domain else {}

    return {
        "success":         True,
        "id":              str(uuid.uuid4()),
        "url":             url,
        "domain":          domain,
        "headline":        headline,
        "body_preview":    body_text[:500] if body_text else "",
        "composite_score": composite,
        "verdict":         verdict,
        "ml_probability":  ml_prob,
        "ml_label":        ml_label,       # 0 = Neutral, 1 = Biased
        "engine_scores":   engine_scores,
        "domain_info":     domain_info,
        "scrape_engine":   scrape_engine,
        "word_count":      word_count,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "analysis_version": "v2.0",
    }
