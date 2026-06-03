"""
URL Scraper — Multi-Engine Fallback Chain
-----------------------------------------
Extraction pipeline:
  1. trafilatura
  2. newspaper3k
  3. BeautifulSoup
"""

import re
import logging
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False

try:
    from newspaper import Article as NewspaperArticle
    _NEWSPAPER_AVAILABLE = True
except ImportError:
    _NEWSPAPER_AVAILABLE = False

logger = logging.getLogger(__name__)

STEALTH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.google.com/",
    "Connection":      "keep-alive",
}


def fetch_html(url: str, timeout: int = 20) -> tuple[str | None, int]:
    try:
        response = requests.get(url, headers=STEALTH_HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.text, response.status_code
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None, 0


def _extract_via_trafilatura(url: str) -> dict | None:
    if not _TRAFILATURA_AVAILABLE:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            html, _ = fetch_html(url)
            if not html: return None
            downloaded = html
        result = trafilatura.extract(downloaded, include_comments=False, url=url, output_format="txt")
        metadata = trafilatura.extract_metadata(downloaded, default_url=url)
        title = metadata.title if metadata else ""
        if result and len(result.split()) > 30:
            paragraphs = [p.strip() for p in result.split("\n") if len(p.strip().split()) >= 5]
            body_text = " ".join(paragraphs)
            return {
                "headline":   title or "",
                "body_paras": paragraphs,
                "body_text":  body_text,
                "word_count": len(body_text.split()),
                "engine":     "trafilatura",
            }
    except Exception as e:
        logger.warning(f"trafilatura failed: {e}")
    return None


def _extract_via_newspaper(url: str) -> dict | None:
    if not _NEWSPAPER_AVAILABLE:
        return None
    try:
        article = NewspaperArticle(url, language="en", request_timeout=20)
        article.download()
        article.parse()
        title = article.title or ""
        body = article.text or ""
        if body and len(body.split()) > 30:
            paragraphs = [p.strip() for p in body.split("\n") if len(p.strip().split()) >= 5]
            body_text = " ".join(paragraphs)
            return {
                "headline":   title,
                "body_paras": paragraphs,
                "body_text":  body_text,
                "word_count": len(body_text.split()),
                "engine":     "newspaper3k",
            }
    except Exception as e:
        logger.warning(f"newspaper3k failed: {e}")
    return None


SKIP_PHRASES = ["subscribe", "sign up", "newsletter", "cookie", "privacy policy", "terms of service"]


def _extract_via_bs4(url: str) -> dict | None:
    html, _ = fetch_html(url)
    if not html: return None
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "button"]):
        tag.decompose()
    h1 = soup.find("h1")
    headline = h1.get_text(strip=True) if h1 else ""
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if len(text.split()) < 5: continue
        if any(phrase in text.lower() for phrase in SKIP_PHRASES): continue
        paragraphs.append(text)
    body_text = " ".join(paragraphs)
    if len(body_text.split()) > 20:
        return {
            "headline":   headline,
            "body_paras": paragraphs,
            "body_text":  body_text,
            "word_count": len(body_text.split()),
            "engine":     "beautifulsoup",
        }
    return None


def scrape_article(url: str) -> dict:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError()
    except Exception:
        return _failed_result(url, "Invalid URL format. Please include https://")

    engines = [
        ("trafilatura",  _extract_via_trafilatura),
        ("newspaper3k",  _extract_via_newspaper),
        ("beautifulsoup", _extract_via_bs4),
    ]

    for name, fn in engines:
        result = fn(url)
        if result and result.get("word_count", 0) > 30:
            return {
                "url":        url,
                "headline":   result.get("headline", "").strip(),
                "body_paras": result.get("body_paras", []),
                "body_text":  result.get("body_text", ""),
                "word_count": result.get("word_count", 0),
                "success":    True,
                "error":      "",
                "engine":     result.get("engine", name),
            }

    return _failed_result(
        url,
        "Could not extract article text. The site may require login, use heavy JavaScript, or block scrapers. Please paste the article text manually."
    )


def _failed_result(url: str, error: str) -> dict:
    return {
        "url":        url,
        "headline":   "",
        "body_paras": [],
        "body_text":  "",
        "word_count": 0,
        "success":    False,
        "error":      error,
        "engine":     "none",
    }


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""
