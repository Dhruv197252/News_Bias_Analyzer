"""
Analyze Router — POST /api/analyze/text and POST /api/analyze/url
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.schemas   import TextAnalyzeRequest, UrlAnalyzeRequest, AnalysisResponse
from backend.database  import get_db, Analysis
from backend.services  import analysis_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analyze", tags=["analyze"])


def _save_to_db(result: dict, db: Session) -> Analysis:
    """Persist analysis result to database."""
    record = Analysis(
        id               = result["id"],
        url              = result.get("url"),
        domain           = result.get("domain"),
        headline         = result.get("headline", "")[:2000],
        body_preview     = result.get("body_preview", "")[:1000],
        composite_score  = result.get("composite_score", 0.0),
        verdict          = result.get("verdict", ""),
        ml_probability   = result.get("ml_probability", 0.0),
        ml_label         = result.get("ml_label", 0),
        engine_scores    = result.get("engine_scores", {}),
        analysis_version = "v2.0",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("/text", response_model=AnalysisResponse)
def analyze_text(req: TextAnalyzeRequest, db: Session = Depends(get_db)):
    """Analyze raw text for media bias."""
    try:
        result = analysis_service.analyze_text(req.text, req.headline or "")
        _save_to_db(result, db)
        return result
    except Exception as e:
        logger.error(f"Text analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/url", response_model=AnalysisResponse)
def analyze_url(req: UrlAnalyzeRequest, db: Session = Depends(get_db)):
    """Scrape a URL and analyze for media bias."""
    try:
        result = analysis_service.analyze_url(req.url)
        if not result.get("success", True):
            raise HTTPException(status_code=422, detail=result.get("error", "Scraping failed"))
        _save_to_db(result, db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
