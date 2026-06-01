"""
Gemini AI Service
------------------
Wraps Google Gemini 1.5 Flash API for:
  1. Bias explanation
  2. Neutral rewrite
  3. Factual summary

Free tier: 15 requests/minute. Uses gemini-1.5-flash (fast + cheap).
"""

import logging
import os
from backend.config import get_settings

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed. GenAI features disabled.")

settings = get_settings()
_MODEL   = None


def _get_model():
    global _MODEL
    if _MODEL is None and _GENAI_AVAILABLE:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables.")
        genai.configure(api_key=api_key)
        _MODEL = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("✅ Gemini 1.5 Flash model initialized.")
    return _MODEL


def _call_gemini(prompt: str, max_tokens: int = 1024) -> str:
    """Call Gemini API and return response text."""
    model = _get_model()
    if not model:
        return ""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens = max_tokens,
                temperature       = 0.3,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise


def explain_bias(text: str, headline: str, composite_score: float, engine_scores: dict) -> str:
    """
    Generate a bias explanation using Gemini.
    References specific score values for credibility.
    """
    if not _GENAI_AVAILABLE or not settings.GEMINI_API_KEY:
        return "Gemini AI not configured. Set GEMINI_API_KEY environment variable."

    score_pct = f"{composite_score:.0%}"
    ml_verdict = "Biased" if engine_scores.get("ml_label", 0) == 1 else "Neutral"
    emotion = f"{engine_scores.get('emotion_intensity', 0):.0%}"
    subj    = f"{engine_scores.get('subjectivity_score', 0):.0%}"
    loaded  = engine_scores.get("unique_loaded_words", 0)
    opinion = engine_scores.get("opinion_label", "")

    prompt = f"""You are an expert media bias analyst. Analyze the following news article for bias.

HEADLINE: {headline or 'N/A'}

ARTICLE TEXT (first 1500 chars):
{text[:1500]}

BIAS SCORES (from our NLP pipeline):
- Overall Bias Score: {score_pct}
- ML Classifier: {ml_verdict} (trained on thousands of labeled news articles)
- Emotional Intensity: {emotion}
- Subjectivity Score: {subj}
- Loaded/Charged Words: {loaded} unique instances
- Writing Style: {opinion}

In 3-4 concise paragraphs:
1. State the overall bias verdict and what it means
2. Point to SPECIFIC phrases or sentences in the text that demonstrate bias (quote them)
3. Explain what perspective or agenda the bias appears to serve
4. Give one sentence on what a neutral version would look like

Be specific, evidence-based, and educational. Avoid political opinions yourself."""

    return _call_gemini(prompt, max_tokens=800)


def rewrite_neutral(text: str, headline: str) -> str:
    """Rewrite article in neutral, objective journalistic style."""
    if not _GENAI_AVAILABLE or not settings.GEMINI_API_KEY:
        return "Gemini AI not configured."

    prompt = f"""You are a professional editor at a neutral wire service (like Reuters or AP).

Rewrite the following news article to be completely neutral and objective:
- Remove all loaded, emotional, or charged language
- Replace opinion statements with factual attributions
- Maintain all factual information
- Use passive constructions where appropriate
- Keep the same approximate length

ORIGINAL HEADLINE: {headline or ''}

ORIGINAL TEXT:
{text[:2000]}

Write the NEUTRAL VERSION below. Start directly with the rewritten headline:"""

    return _call_gemini(prompt, max_tokens=1000)


def summarize_facts(text: str, headline: str) -> str:
    """Extract 3 bullet-point factual summary stripped of loaded language."""
    if not _GENAI_AVAILABLE or not settings.GEMINI_API_KEY:
        return "Gemini AI not configured."

    prompt = f"""Extract exactly 3 bullet points of VERIFIABLE FACTS from this article.
Rules:
- Only include claims that can be fact-checked
- Remove all emotional language and opinion
- Each bullet should be one sentence
- If a claim is attributed to someone, include the attribution

ARTICLE: {text[:2000]}

Respond with exactly 3 bullet points starting with •:"""

    return _call_gemini(prompt, max_tokens=300)
