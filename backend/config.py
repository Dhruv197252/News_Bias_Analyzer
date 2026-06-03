"""
Config — Environment Variables
--------------------------------
All config loaded from environment. Falls back to defaults for local dev.
Set these on Render dashboard (never commit secrets to git).
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./slant_dev.db"   # local dev default

    # Gemini AI
    GEMINI_API_KEY: str = "YOUR_API_KEY_HERE"

    # CORS — comma-separated list of allowed origins
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # App
    APP_NAME: str    = "SLANT v2.0 - News Bias Analyzer"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool      = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
