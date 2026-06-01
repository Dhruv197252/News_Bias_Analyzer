"""
Pydantic Schemas — Request / Response Models
"""

from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, Any
from datetime import datetime


# ── Requests ──────────────────────────────────────────────────────────────────

class TextAnalyzeRequest(BaseModel):
    text: str
    headline: Optional[str] = ""

    @validator("text")
    def text_min_length(cls, v):
        if len(v.strip().split()) < 10:
            raise ValueError("Text must be at least 10 words.")
        return v.strip()


class UrlAnalyzeRequest(BaseModel):
    url: str

    @validator("url")
    def url_not_empty(cls, v):
        if not v.strip():
            raise ValueError("URL cannot be empty.")
        return v.strip()


class CompareRequest(BaseModel):
    url1: Optional[str] = None
    url2: Optional[str] = None
    text1: Optional[str] = None
    text2: Optional[str] = None
    label1: Optional[str] = "Article A"
    label2: Optional[str] = "Article B"


class GenAIRequest(BaseModel):
    analysis_id: Optional[str] = None
    text: Optional[str] = None
    headline: Optional[str] = ""


# ── Engine Scores ─────────────────────────────────────────────────────────────

class EngineScores(BaseModel):
    ml_probability:    float
    ml_label:          int       # 0 = Neutral, 1 = Biased
    emotion_intensity: float
    subjectivity_score: float
    passive_score:     float
    lexicon_score:     float
    hedge_score:       float


# ── Analysis Response ─────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    id:               str
    url:              Optional[str]     = None
    domain:           Optional[str]     = None
    headline:         Optional[str]     = None
    body_preview:     Optional[str]     = None
    composite_score:  float
    verdict:          str
    ml_probability:   float
    ml_label:         int               # 0 = Neutral, 1 = Biased
    engine_scores:    Optional[dict]    = None
    domain_info:      Optional[dict]    = None
    scrape_engine:    Optional[str]     = None
    word_count:       Optional[int]     = None
    created_at:       Optional[datetime] = None
    analysis_version: str = "v2.0"


# ── Gemini Response ───────────────────────────────────────────────────────────

class GenAIResponse(BaseModel):
    analysis_id: Optional[str] = None
    explanation: Optional[str] = None
    rewrite:     Optional[str] = None
    summary:     Optional[str] = None
    error:       Optional[str] = None


# ── History ───────────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id:              str
    url:             Optional[str]
    domain:          Optional[str]
    headline:        Optional[str]
    composite_score: float
    verdict:         str
    ml_label:        int
    created_at:      Optional[datetime]


class HistoryResponse(BaseModel):
    items:   list[HistoryItem]
    total:   int
    page:    int
    per_page: int


# ── Domain ────────────────────────────────────────────────────────────────────

class DomainInfo(BaseModel):
    domain:                str
    total_articles:        int
    avg_bias_score:        float
    mbfc_leaning:          Optional[str]
    mbfc_factual_reporting: Optional[str]
    country:               Optional[str]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str
    version: str
    model:   str
    db:      str
