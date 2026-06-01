"""
URL Scraper — Multi-Engine Fallback Chain
-----------------------------------------
Extraction pipeline (tries each in order until one succeeds):

  1. trafilatura      — Best-in-class news extraction (handles 90%+ of sites,
                        JS-rendered pages, paywalls, anti-bot, Indian sites)
  2. newspaper3k      — Article-specific NLP extraction (good fallback)
  3. BeautifulSoup    — Raw <p> tag extraction (last resort)

Why trafilatura first?
  • Designed specifically for web article extraction
  • Handles gzip, brotli, redirects automatically
  • Has built-in boilerplate removal (nav, ads, footers)
  • Works on The Hindu, NDTV, Republic TV, Times of India, etc.
"""

import re
import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Optional imports — graceful degradation if not installed
try:
    import trafilatura
    from trafilatura.settings import use_config
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False

try:
    from newspaper import Article as NewspaperArticle
    _NEWSPAPER_AVAILABLE = True
except ImportError:
    _NEWSPAPER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── 1. Stealth Headers ────────────────────────────────────────────────────────

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
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control":   "max-age=0",
}


# ── 2. Raw HTML Fetcher ───────────────────────────────────────────────────────

def fetch_html(url: str, timeout: int = 20) -> tuple[str | None, int]:
    """
    Downloads raw HTML. Returns (html_string, status_code).
    Returns (None, 0) on failure.
    """
    try:
        response = requests.get(
            url,
            headers=STEALTH_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response.text, response.status_code

    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP Error {e.response.status_code} for {url}")
        return None, getattr(e.response, "status_code", 0)
    except requests.exceptions.ConnectionError:
        logger.warning(f"Connection Error: {url}")
        return None, 0
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching: {url}")
        return None, 0
    except Exception as e:
        logger.warning(f"Unexpected error fetching {url}: {e}")
        return None, 0


# ── 3. Engine 1: trafilatura (Best) ──────────────────────────────────────────

def _extract_via_trafilatura(url: str) -> dict | None:
    """
    Uses trafilatura for extraction. Returns dict or None if failed.
    trafilatura handles: JS-heavy sites, Indian news, anti-bot, paywalls.
    """
    if not _TRAFILATURA_AVAILABLE:
        return None

    try:
        # Let trafilatura handle the download (has its own retry logic)
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            # Fall back to our own download
            html, _ = fetch_html(url)
            if not html:
                return None
            downloaded = html

        # Extract with metadata
        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=False,    # favor_recall for more text
            url=url,
            output_format="txt",
        )

        # Also get metadata (title, author, date)
        metadata = trafilatura.extract_metadata(downloaded, default_url=url)
        title = ""
        if metadata:
            title = metadata.title or metadata.sitename or ""

        if result and len(result.split()) > 30:
            paragraphs = [p.strip() for p in result.split("\n") if len(p.strip().split()) >= 5]
            body_text = " ".join(paragraphs)
            return {
                "headline":   title,
                "body_paras": paragraphs,
                "body_text":  body_text,
                "word_count": len(body_text.split()),
                "engine":     "trafilatura",
            }

    except Exception as e:
        logger.warning(f"trafilatura failed for {url}: {e}")

    return None


# ── 4. Engine 2: newspaper3k ──────────────────────────────────────────────────

def _extract_via_newspaper(url: str) -> dict | None:
    """
    Uses newspaper3k for extraction. Good at news-specific parsing.
    """
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
        logger.warning(f"newspaper3k failed for {url}: {e}")

    return None


# ── 5. Engine 3: BeautifulSoup (Fallback) ────────────────────────────────────

SKIP_PHRASES = [
    "subscribe", "sign up", "newsletter", "cookie", "privacy policy",
    "terms of service", "all rights reserved", "click here",
    "advertisement", "read more", "follow us", "share this",
    "download app", "install app", "breaking news alerts",
]


def _extract_via_bs4(url: str) -> dict | None:
    """
    Raw BeautifulSoup <p> extraction. Last resort fallback.
    """
    html, status = fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "noscript", "header",
                     "footer", "nav", "aside", "form", "button"]):
        tag.decompose()

    # Headline
    headline = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        headline = h1.get_text(strip=True)
    elif soup.find("title"):
        raw = soup.find("title").get_text(strip=True)
        headline = re.split(r"\s[\|–\-]\s", raw)[0].strip()

    # Body paragraphs
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)
        if len(text.split()) < 5:
            continue
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in SKIP_PHRASES):
            continue
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


# ── 6. Master Scrape Function ─────────────────────────────────────────────────

def scrape_article(url: str) -> dict:
    """
    Full pipeline: tries trafilatura → newspaper3k → BeautifulSoup.
    Returns the first successful extraction.

    Returns
    -------
    dict:
        url        : str   — original URL
        headline   : str   — extracted headline
        body_paras : list  — list of clean paragraph strings
        body_text  : str   — full body joined as one string
        word_count : int   — total word count of body
        success    : bool  — False if all engines failed
        error      : str   — error message if success=False
        engine     : str   — which engine succeeded
    """
    logger.info(f"Scraping: {url}")

    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")
    except Exception:
        return _failed_result(url, "Invalid URL format. Please include https://")

    # Try engines in order
    engines = [
        ("trafilatura",  _extract_via_trafilatura),
        ("newspaper3k",  _extract_via_newspaper),
        ("beautifulsoup", _extract_via_bs4),
    ]

    for engine_name, engine_fn in engines:
        logger.info(f"  Trying {engine_name}...")
        result = engine_fn(url)
        if result and result.get("word_count", 0) > 30:
            logger.info(f"  ✅ {engine_name} succeeded ({result['word_count']} words)")
            return {
                "url":        url,
                "headline":   result.get("headline", "").strip(),
                "body_paras": result.get("body_paras", []),
                "body_text":  result.get("body_text", ""),
                "word_count": result.get("word_count", 0),
                "success":    True,
                "error":      "",
                "engine":     result.get("engine", engine_name),
            }
        logger.info(f"  ❌ {engine_name} insufficient content")

    # All engines failed
    return _failed_result(
        url,
        "Could not extract article text. The site may require login, "
        "use heavy JavaScript, or block scrapers. Please paste the article text manually."
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


# ── 7. Domain Extractor ───────────────────────────────────────────────────────

def extract_domain(url: str) -> str:
    """Extract clean domain name from URL. e.g. 'https://www.ndtv.com/...' → 'ndtv.com'"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


# ── 8. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_URLS = [
        "https://www.ndtv.com/india-news",
        "https://www.thehindu.com/news/national/",
        "https://en.wikipedia.org/wiki/Media_bias",
    ]

    for url in TEST_URLS:
        result = scrape_article(url)
        print(f"\n{'═'*65}")
        print(f"  URL    : {result['url']}")
        print(f"  Engine : {result.get('engine', 'N/A')}")
        print(f"  Status : {'✅ Success' if result['success'] else '❌ Failed'}")
        if result["success"]:
            print(f"  Title  : {result['headline'][:80]}")
            print(f"  Words  : {result['word_count']}")
        else:
            print(f"  Error  : {result['error']}")
        print(f"{'═'*65}")