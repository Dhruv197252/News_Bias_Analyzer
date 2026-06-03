from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database import get_db, Analysis
from backend.schemas  import HistoryResponse, HistoryItem, AnalysisResponse

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
def get_history(
    page:     int   = Query(1, ge=1),
    per_page: int   = Query(20, ge=1, le=100),
    domain:   str   = Query(None),
    verdict:  str   = Query(None),
    db: Session = Depends(get_db),
):
    """Return paginated analysis history."""
    query = db.query(Analysis).order_by(desc(Analysis.created_at))

    if domain:
        query = query.filter(Analysis.domain == domain)
    if verdict:
        query = query.filter(Analysis.verdict == verdict)

    total  = query.count()
    items  = query.offset((page - 1) * per_page).limit(per_page).all()

    return HistoryResponse(
        items=[
            HistoryItem(
                id              = r.id,
                url             = r.url,
                domain          = r.domain,
                headline        = r.headline,
                composite_score = r.composite_score or 0.0,
                verdict         = r.verdict or "",
                ml_label        = r.ml_label or 0,
                created_at      = r.created_at,
            )
            for r in items
        ],
        total    = total,
        page     = page,
        per_page = per_page,
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Return a single analysis by ID."""
    record = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return AnalysisResponse(
        id               = record.id,
        url              = record.url,
        domain           = record.domain,
        headline         = record.headline,
        body_preview     = record.body_preview,
        composite_score  = record.composite_score or 0.0,
        verdict          = record.verdict or "",
        ml_probability   = record.ml_probability or 0.0,
        ml_label         = record.ml_label or 0,
        engine_scores    = record.engine_scores,
        genai_explanation = record.genai_explanation,
        created_at       = record.created_at,
        analysis_version = record.analysis_version or "v2.0",
    )
