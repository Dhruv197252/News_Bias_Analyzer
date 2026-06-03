"""
Unit tests for the NLP engine pipeline.
Run with: pytest tests/ -v
"""
import pytest


def test_bias_lexicon_scan_empty():
    from nlp.bias_lexicon import scan_text
    result = scan_text("")
    assert result["total_tokens"] == 0
    assert result["naive_score"] == 0.0


def test_bias_lexicon_scan_loaded_words():
    from nlp.bias_lexicon import scan_text
    result = scan_text("The regime is catastrophic and draconian.")
    assert result["unique_matches"] > 0
    assert result["naive_score"] > 0


def test_nlp_engines_subjectivity():
    from nlp.nlp_engines import get_subjectivity
    result = get_subjectivity("The government announced new policy.")
    assert 0.0 <= result["score"] <= 1.0
    assert "label" in result


def test_scraper_domain_extraction():
    from nlp.scraper import extract_domain
    assert extract_domain("https://www.ndtv.com/india/article") == "ndtv.com"
    assert extract_domain("https://thehindu.com/news") == "thehindu.com"
    assert extract_domain("not-a-url") == ""


def test_composite_label():
    from nlp.scorer import composite_label
    assert composite_label(0.10)["label"] == "Appears Neutral"
    assert composite_label(0.35)["label"] == "Slightly Opinionated"
    assert composite_label(0.55)["label"] == "Moderate Bias"
    assert composite_label(0.70)["label"] == "Highly Opinionated"
    assert composite_label(0.85)["label"] == "Extreme Bias Detected"
