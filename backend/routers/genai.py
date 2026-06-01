"""
GenAI Router — POST /api/genai/explain, /rewrite, /summarize
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.schemas  import GenAIRequest, GenAIResponse
from backend.database import get_db, Analysis
from backend.services import genai_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/genai", tags=["genai"])


def _get_text_for_request(req: GenAIRequest, db: Session) -> tuple[str, str, dict, float]:
    """Fetch text + headline + scores from DB or directly from request."""
    text      = req.text or ""
    headline  = req.headline or ""
    scores    = {}
    composite = 0.0

    if req.analysis_id:
        record = db.query(Analysis).filter(Analysis.id == req.analysis_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        text      = record.body_preview or ""
        headline  = record.headline or ""
        scores    = record.engine_scores or {}
        composite = record.composite_score or 0.0

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text provided for GenAI analysis.")

    return text, headline, scores, composite


@router.post("/explain", response_model=GenAIResponse)
def explain(req: GenAIRequest, db: Session = Depends(get_db)):
    """Generate a Gemini-powered bias explanation."""
    text, headline, scores, composite = _get_text_for_request(req, db)
    try:
        explanation = genai_service.explain_bias(text, headline, composite, scores)

        # Persist to DB if we have an analysis_id
        if req.analysis_id:
            db.query(Analysis).filter(Analysis.id == req.analysis_id).update(
                {"genai_explanation": explanation}
            )
            db.commit()

        return GenAIResponse(analysis_id=req.analysis_id, explanation=explanation)
    except Exception as e:
        logger.error(f"GenAI explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rewrite", response_model=GenAIResponse)
def rewrite(req: GenAIRequest, db: Session = Depends(get_db)):
    """Generate a neutral rewrite of the article."""
    text, headline, _, _ = _get_text_for_request(req, db)
    try:
        neutral = genai_service.rewrite_neutral(text, headline)

        if req.analysis_id:
            db.query(Analysis).filter(Analysis.id == req.analysis_id).update(
                {"genai_rewrite": neutral}
            )
            db.commit()

        return GenAIResponse(analysis_id=req.analysis_id, rewrite=neutral)
    except Exception as e:
        logger.error(f"GenAI rewrite error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=GenAIResponse)
def summarize(req: GenAIRequest, db: Session = Depends(get_db)):
    """Generate a 3-bullet factual summary."""
    text, headline, _, _ = _get_text_for_request(req, db)
    try:
        summary = genai_service.summarize_facts(text, headline)

        if req.analysis_id:
            db.query(Analysis).filter(Analysis.id == req.analysis_id).update(
                {"genai_summary": summary}
            )
            db.commit()

        return GenAIResponse(analysis_id=req.analysis_id, summary=summary)
    except Exception as e:
        logger.error(f"GenAI summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
