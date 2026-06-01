"""
ML Engine — Bias Classifier
-----------------------------
Binary classifier: Biased=1 / Neutral=0

Label mapping (aligned with training data):
  Left, Right leaning articles  → biased = 1
  Center / Neutral articles     → biased = 0

Model priority:
  1. bias_classifier_v2.pkl  (XGBoost — trained on BABE + AllSides)
  2. bias_classifier.pkl     (LR fallback — original BABE-only model)

Render free-tier safe:
  • Model loaded ONCE at startup as module-level singleton
  • ~60-80 MB RAM for model + TF-IDF vocabulary
"""

import os
import logging
import joblib
import pandas as pd
from sklearn.pipeline         import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (classification_report,
                                      confusion_matrix,
                                      roc_auc_score)

logger = logging.getLogger(__name__)


# ── 1. Config ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(_PROJECT_ROOT, "data", "babe_clean.csv")
CORPUS_PATH    = os.path.join(_PROJECT_ROOT, "data", "merged_corpus.csv")
MODEL_PATH_V2  = os.path.join(_PROJECT_ROOT, "models", "bias_classifier_v2.pkl")
MODEL_PATH_V1  = os.path.join(_PROJECT_ROOT, "models", "bias_classifier.pkl")

# Singleton — loaded once at module import time when used in FastAPI
_PIPELINE_SINGLETON: Pipeline | None = None


# ── 2. Model Loader ───────────────────────────────────────────────────────────

def load_model(path: str | None = None) -> Pipeline:
    """
    Load the saved model from disk.

    Priority:
      1. Explicit path (if given)
      2. bias_classifier_v2.pkl  (XGBoost + AllSides)
      3. bias_classifier.pkl     (LR fallback)

    Returns the fitted sklearn Pipeline.
    """
    if path:
        logger.info(f"Loading model from explicit path: {path}")
        return joblib.load(path)

    if os.path.exists(MODEL_PATH_V2):
        logger.info(f"Loading v2 model: {MODEL_PATH_V2}")
        return joblib.load(MODEL_PATH_V2)

    if os.path.exists(MODEL_PATH_V1):
        logger.warning(f"v2 model not found. Loading v1 fallback: {MODEL_PATH_V1}")
        return joblib.load(MODEL_PATH_V1)

    raise FileNotFoundError(
        "No model file found. Run `python training/train_model.py` to train."
    )


def get_pipeline() -> Pipeline:
    """
    Return the singleton pipeline (load once, reuse).
    Used by FastAPI to avoid loading the model on every request.
    """
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = load_model()
    return _PIPELINE_SINGLETON


# ── 3. Build Legacy Pipeline (for retrain from scratch) ──────────────────────

def build_pipeline() -> Pipeline:
    """
    TF-IDF + Logistic Regression pipeline.
    Used for quick retraining without XGBoost.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = 25_000,
            ngram_range   = (1, 3),
            sublinear_tf  = True,
            min_df        = 2,
            strip_accents = "unicode",
        )),
        ("clf", LogisticRegression(
            C             = 1.5,
            max_iter      = 1500,
            class_weight  = "balanced",
            random_state  = 42,
        )),
    ])


# ── 4. Load Data ──────────────────────────────────────────────────────────────

def load_data(path: str | None = None) -> tuple[list[str], list[int]]:
    """
    Load training data.
    Prefers merged_corpus.csv (BABE + AllSides), falls back to babe_clean.csv.

    Label convention:
        1 → Biased   (Left or Right leaning)
        0 → Neutral  (Center)
    """
    if path is None:
        path = CORPUS_PATH if os.path.exists(CORPUS_PATH) else DATA_PATH

    logger.info(f"Loading data from '{path}'...")
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    logger.info(f"   {len(df)} rows loaded.")
    return df["text"].tolist(), df["label"].tolist()


# ── 5. Train ──────────────────────────────────────────────────────────────────

def train(save: bool = True) -> Pipeline:
    """
    Quick train using LR (for standalone use or CI).
    For full XGBoost training, use training/train_model.py.
    """
    texts, labels = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size    = 0.2,
        random_state = 42,
        stratify     = labels,
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Neutral (0)", "Biased (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    print("── Confusion Matrix ─────────────────────────────────")
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}\n")

    auc = roc_auc_score(y_test, y_proba)
    print(f"── ROC-AUC Score ────────────────────────────────────")
    print(f"   AUC = {auc:.4f}\n")

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH_V1), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH_V1)
        print(f"💾 Model saved to '{MODEL_PATH_V1}'")

    return pipeline


# ── 6. Inference ──────────────────────────────────────────────────────────────

def predict_bias(text: str, pipeline: Pipeline) -> dict:
    """
    Run inference on a single string.

    Returns
    -------
    dict:
        label       : "Biased" (1) or "Neutral" (0)
        probability : float 0.0–1.0  (P(biased))
        ml_label    : int 0 or 1
    """
    if not text or not text.strip():
        return {"label": "Neutral", "probability": 0.0, "ml_label": 0}

    proba     = pipeline.predict_proba([text])[0]   # [P(neutral), P(biased)]
    bias_prob = round(float(proba[1]), 4)
    ml_label  = 1 if bias_prob >= 0.5 else 0

    return {
        "label":       "Biased" if ml_label == 1 else "Neutral",
        "probability": bias_prob,
        "ml_label":    ml_label,
    }


# ── 7. Batch Inference ────────────────────────────────────────────────────────

def predict_bias_batch(texts: list[str], pipeline: Pipeline) -> list[dict]:
    """
    Batch inference for multiple texts. More efficient than calling predict_bias() in a loop.
    """
    if not texts:
        return []

    probas = pipeline.predict_proba(texts)[:, 1]
    results = []
    for prob in probas:
        bias_prob = round(float(prob), 4)
        ml_label  = 1 if bias_prob >= 0.5 else 0
        results.append({
            "label":       "Biased" if ml_label == 1 else "Neutral",
            "probability": bias_prob,
            "ml_label":    ml_label,
        })
    return results


# ── 8. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = train(save=True)

    TEST_SENTENCES = [
        # Neutral (label=0 expected)
        "The government announced new infrastructure spending on Tuesday.",
        "Officials confirmed the bill passed with bipartisan support in the Senate.",
        "NDTV reports that the Prime Minister will visit flood-affected areas tomorrow.",
        # Biased (label=1 expected)
        "Radical extremists have shamefully undermined our democratic institutions.",
        "The heroic whistleblower exposed the corrupt and tyrannical administration.",
        "Republic TV's vile propaganda machine continues its assault on minorities.",
    ]

    print("\n── Live Inference Test ───────────────────────────────")
    print(f"  {'Label':<8} {'Prob':>6}   Text")
    print(f"  {'─'*8} {'─'*6}   {'─'*60}")
    for sentence in TEST_SENTENCES:
        result = predict_bias(sentence, pipeline)
        bar    = "🔴" if result["ml_label"] == 1 else "🟢"
        print(f"  {bar} {result['label']:<7} {result['probability']:>5.0%}   {sentence[:70]}")