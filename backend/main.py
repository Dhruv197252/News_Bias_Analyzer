"""
SLANT v2.0 — FastAPI Main App
-------------------------------
Entry point for the backend.
Render deployment: uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1

RAM budget (Render free tier = 512 MB):
  All NLP engines loaded lazily on first request (~250 MB total).
  Use --workers 1 to avoid duplicate model copies in RAM.
"""

import logging
import os
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path (so 'nlp' and 'ml' are importable)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.config   import get_settings
from backend.database import init_db
from backend.routers  import analyze, history, domains, genai

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger   = logging.getLogger(__name__)
settings = get_settings()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = settings.APP_NAME,
    description = "Multi-engine NLP bias analyzer with Gemini AI integration.",
    version     = settings.APP_VERSION,
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
)


# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.get_cors_origins(),
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request Timing Middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    db_preview = settings.DATABASE_URL if len(settings.DATABASE_URL) <= 40 else settings.DATABASE_URL[:40] + "..."
    logger.info(f"   DB URL  : {db_preview}")
    logger.info(f"   CORS    : {settings.get_cors_origins()}")
    init_db()
    logger.info("✅ App ready. NLP engines will load on first analysis request.")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(analyze.router)
app.include_router(history.router)
app.include_router(domains.router)
app.include_router(genai.router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["health"])
def health_check():
    """Health check endpoint — used by Render to verify the service is up."""
    import os
    model_file = os.path.join(ROOT, "models", "bias_classifier_v2.pkl")
    model_v1   = os.path.join(ROOT, "models", "bias_classifier.pkl")
    model_status = (
        "v2 (XGBoost)" if os.path.exists(model_file) else
        "v1 (LR fallback)" if os.path.exists(model_v1) else
        "not found"
    )
    return {
        "status":  "ok",
        "version": settings.APP_VERSION,
        "model":   model_status,
        "db":      "connected",
        "genai":   "configured" if settings.GEMINI_API_KEY else "not configured",
    }


@app.get("/", tags=["root"])
def root():
    return {
        "app":     settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/api/docs",
        "health":  "/api/health",
    }


# ── Dev Server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,    # dev only — disabled on Render
        workers = 1,
    )
