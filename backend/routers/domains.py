"""
Domains Router — GET /api/domains
"""

from fastapi import APIRouter
from backend.services.domain_service import list_all_domains

router = APIRouter(prefix="/api/domains", tags=["domains"])


@router.get("")
def get_domains():
    """Return all known domains with MBFC bias ratings."""
    return {"domains": list_all_domains()}
