"""
Step 5: Web Scraper
--------------------
Fetches a news article from a URL using requests + BeautifulSoup.

Key features:
  • Stealth headers  — mimics a real browser to bypass basic anti-bot blocks
  • Headline extraction  — grabs the <h1> tag
  • Smart body extraction — filters <p> tags, ignores nav/footer/ad text
  • Paragraph length filter — drops any <p> under 5 words (menus, captions)
  • Fallback handling — graceful errors, never crashes the dashboard
"""

import requests
from bs4 import BeautifulSoup
import re


# ── 1. Stealth Headers ────────────────────────────────────────────────────────
# Many news sites block default Python/requests User-Agents.
# We mimic a real Chrome browser on Windows to bypass basic firewalls.

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
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.google.com/",
    "DNT":             "1",                    # Do Not Track (looks human)
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ── 2. Raw HTML Fetcher ───────────────────────────────────────────────────────

def fetch_html(url: str, timeout: int = 15) -> str | None:
    """
    Downloads raw HTML from a URL using stealth headers.

    Returns
    -------
    str : raw HTML content
    None : if the request failed (prints reason)
    """
    try:
        response = requests.get(
            url,
            headers = STEALTH_HEADERS,
            timeout = timeout,
        )
        response.raise_for_status()   # Raises on 4xx / 5xx
        return response.text

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not reach '{url}'")
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: The request took longer than {timeout}s")
    except requests.exceptions.RequestException as e:
        print(f"❌ Unexpected error: {e}")

    return None


# ── 3. Headline Extractor ─────────────────────────────────────────────────────

def extract_headline(soup: BeautifulSoup) -> str:
    """
    Tries multiple strategies to extract the article headline.

    Priority:
      1. <h1> tag (most reliable)
      2. <title> tag (fallback)
      3. "No headline found" (last resort)
    """
    # Strategy 1 — <h1>
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    # Strategy 2 — <title>
    title = soup.find("title")
    if title and title.get_text(strip=True):
        # Remove site name suffix e.g. "Article Title | CNN"
        raw = title.get_text(strip=True)
        return re.split(r"\s[\|–\-]\s", raw)[0].strip()

    return "No headline found"


# ── 4. Body Text Extractor ────────────────────────────────────────────────────

def extract_body(soup: BeautifulSoup, min_words: int = 5) -> list[str]:
    """
    Extracts clean article paragraphs from <p> tags.

    Filters OUT:
      • Paragraphs under `min_words` words (nav links, captions, labels)
      • Paragraphs that are purely whitespace
      • Common boilerplate phrases (cookie notices, subscription prompts)

    Parameters
    ----------
    soup      : parsed BeautifulSoup object
    min_words : int — minimum word count to keep a paragraph (default 5)

    Returns
    -------
    list[str] : clean paragraph strings
    """
    # Boilerplate phrases to skip
    SKIP_PHRASES = [
        "subscribe", "sign up", "newsletter", "cookie", "privacy policy",
        "terms of service", "all rights reserved", "click here",
        "advertisement", "read more", "follow us", "share this",
    ]

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ", strip=True)

        # Filter 1 — too short
        if len(text.split()) < min_words:
            continue

        # Filter 2 — boilerplate
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in SKIP_PHRASES):
            continue

        paragraphs.append(text)

    return paragraphs


# ── 5. Master Scrape Function ─────────────────────────────────────────────────

def scrape_article(url: str) -> dict:
    """
    Full pipeline: Fetch → Parse → Extract Headline + Body

    Returns
    -------
    dict:
        url        : str   — original URL
        headline   : str   — extracted headline
        body_paras : list  — list of clean paragraph strings
        body_text  : str   — full body joined as one string
        word_count : int   — total word count of body
        success    : bool  — False if scraping failed
        error      : str   — error message if success=False
    """
    print(f"🌐 Scraping: {url}")

    # Step 1 — Fetch HTML
    html = fetch_html(url)
    if html is None:
        return {
            "url":        url,
            "headline":   "",
            "body_paras": [],
            "body_text":  "",
            "word_count": 0,
            "success":    False,
            "error":      "Failed to fetch page. Site may block scrapers.",
        }

    # Step 2 — Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Step 3 — Remove script/style tags (they pollute text extraction)
    for tag in soup(["script", "style", "noscript", "header",
                     "footer", "nav", "aside"]):
        tag.decompose()

    # Step 4 — Extract
    headline   = extract_headline(soup)
    body_paras = extract_body(soup)
    body_text  = " ".join(body_paras)

    print(f"✅ Headline    : {headline[:80]}...")
    print(f"   Paragraphs : {len(body_paras)}")
    print(f"   Word count : {len(body_text.split())}\n")

    return {
        "url":        url,
        "headline":   headline,
        "body_paras": body_paras,
        "body_text":  body_text,
        "word_count": len(body_text.split()),
        "success":    True,
        "error":      "",
    }


# ── 6. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test with a reliable, always-accessible news article
    TEST_URLS = [
       "https://en.wikipedia.org/wiki/Media_bias",
    ]

    for url in TEST_URLS:
        result = scrape_article(url)

        if result["success"]:
            print(f"{'═'*60}")
            print(f"  URL      : {result['url']}")
            print(f"  Headline : {result['headline']}")
            print(f"  Words    : {result['word_count']}")
            print(f"\n  ── First 3 Paragraphs ──────────────────────")
            for i, para in enumerate(result["body_paras"][:3], 1):
                print(f"  [{i}] {para[:120]}...")
            print(f"{'═'*60}\n")
        else:
            print(f"⚠️  Scrape failed for {url}: {result['error']}\n")