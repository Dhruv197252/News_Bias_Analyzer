"""
Domain Service
---------------
Looks up domain bias metadata from:
  1. data/mbfc_domains.json  (MBFC ratings)
  2. data/indian_domains.json  (Indian outlets)
Returns merged context dict for each domain.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MBFC_PATH   = os.path.join(ROOT, "data", "mbfc_domains.json")
INDIAN_PATH = os.path.join(ROOT, "data", "indian_domains.json")

# Inline fallback from mbfc_loader.py
from training.mbfc_loader import MBFC_DOMAINS

_MBFC_DATA:   dict | None = None
_INDIAN_DATA: dict | None = None


def _load_mbfc() -> dict:
    global _MBFC_DATA
    if _MBFC_DATA is None:
        if os.path.exists(MBFC_PATH):
            with open(MBFC_PATH, "r", encoding="utf-8") as f:
                _MBFC_DATA = json.load(f)
        else:
            _MBFC_DATA = MBFC_DOMAINS
    return _MBFC_DATA


def _load_indian() -> dict:
    global _INDIAN_DATA
    if _INDIAN_DATA is None:
        if os.path.exists(INDIAN_PATH):
            with open(INDIAN_PATH, "r", encoding="utf-8") as f:
                _INDIAN_DATA = json.load(f)
        else:
            _INDIAN_DATA = {}
    return _INDIAN_DATA


def get_domain_info_for(domain: str | None) -> dict:
    """
    Return domain metadata dict for a given domain string.
    Searches MBFC first, then Indian domains DB.
    """
    if not domain:
        return {}

    # Normalize
    d = domain.lower().strip()
    if d.startswith("www."):
        d = d[4:]

    mbfc   = _load_mbfc()
    indian = _load_indian()

    # Try MBFC
    info = mbfc.get(d, mbfc.get("www." + d, {}))

    # Try Indian domains (may have more detail for IN outlets)
    indian_info = indian.get(d, indian.get("www." + d, {}))

    # Merge (Indian data wins on conflict for IN outlets)
    merged = {**info, **indian_info}
    if merged:
        merged["domain"] = d

    return merged


def list_all_domains() -> list[dict]:
    """Return all known domains with metadata."""
    mbfc   = _load_mbfc()
    indian = _load_indian()

    seen = set()
    result = []

    for domain, info in {**mbfc, **indian}.items():
        if domain not in seen:
            seen.add(domain)
            result.append({"domain": domain, **info})

    return sorted(result, key=lambda x: x.get("domain", ""))
