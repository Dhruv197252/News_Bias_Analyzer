"""
ML Engine — Bias Classifier
-----------------------------
Binary classifier: Biased=1 / Neutral=0

Priority:
  1. bias_classifier_v2.pkl (XGBoost)
  2. bias_classifier.pkl (LR fallback)
"""

import os
import logging
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)

# babe_clean.csv lives directly in data/ (not data/raw/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(_PROJECT_ROOT, "data", "babe_clean.csv")
CORPUS_PATH = os.path.join(_PROJECT_ROOT, "data", "raw", "merged_corpus.csv")
MODEL_PATH_V2 = os.path.join(_PROJECT_ROOT, "models", "bias_classifier_v2.pkl")
MODEL_PATH_V1 = os.path.join(_PROJECT_ROOT, "models", "bias_classifier.pkl")

_PIPELINE_SINGLETON: dict | None = None


def load_model(path: str | None = None) -> dict:
    models = {}
    if path:
        logger.info(f"Loading model from explicit path: {path}")
        models['custom'] = joblib.load(path)
        return models

    if os.path.exists(MODEL_PATH_V2):
        logger.info(f"Loading v2 model (XGBoost): {MODEL_PATH_V2}")
        models['v2'] = joblib.load(MODEL_PATH_V2)

    if os.path.exists(MODEL_PATH_V1):
        logger.info(f"Loading v1 model (LR): {MODEL_PATH_V1}")
        models['v1'] = joblib.load(MODEL_PATH_V1)

    if not models:
        # No pre-trained model found — auto-train from BABE dataset on first run
        logger.warning("No pre-trained model found. Auto-training from BABE dataset...")
        logger.warning(f"   Data:  {DATA_PATH}")
        logger.warning(f"   Model: {MODEL_PATH_V1}")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Cannot auto-train: dataset not found at '{DATA_PATH}'. "
                "Ensure data/babe_clean.csv exists."
            )
        models['v1'] = train(save=True)
        
    return models


def get_pipeline() -> dict:
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = load_model()
    return _PIPELINE_SINGLETON


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=25000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.5,
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def load_data(path: str | None = None) -> tuple[list[str], list[int]]:
    if path is None:
        path = CORPUS_PATH if os.path.exists(CORPUS_PATH) else DATA_PATH

    logger.info(f"Loading data from '{path}'...")
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    logger.info(f"   {len(df)} rows loaded.")
    return df["text"].tolist(), df["label"].tolist()


def train(save: bool = True) -> Pipeline:
    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Neutral (0)", "Biased (1)"]))

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH_V1), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH_V1)
        print(f"💾 Model saved to '{MODEL_PATH_V1}'")

    return pipeline


def predict_bias(text: str, pipeline: dict) -> dict:
    if not text or not text.strip():
        return {"label": "Neutral", "probability": 0.0, "ml_label": 0}

    probs = []
    for name, model in pipeline.items():
        proba = model.predict_proba([text])[0]
        probs.append(float(proba[1]))
        
    bias_prob = round(sum(probs) / len(probs), 4) if probs else 0.0
    ml_label = 1 if bias_prob >= 0.5 else 0

    return {
        "label": "Biased" if ml_label == 1 else "Neutral",
        "probability": bias_prob,
        "ml_label": ml_label,
    }


def predict_bias_batch(texts: list[str], pipeline: dict) -> list[dict]:
    if not texts:
        return []
        
    all_probs = []
    for name, model in pipeline.items():
        probas = model.predict_proba(texts)[:, 1]
        all_probs.append(probas)
        
    results = []
    num_texts = len(texts)
    for i in range(num_texts):
        # Average probability across all loaded models for this specific text
        probs_for_text = [model_probs[i] for model_probs in all_probs]
        bias_prob = round(float(sum(probs_for_text) / len(probs_for_text)), 4) if probs_for_text else 0.0
        ml_label = 1 if bias_prob >= 0.5 else 0
        
        results.append({
            "label": "Biased" if ml_label == 1 else "Neutral",
            "probability": bias_prob,
            "ml_label": ml_label,
        })
    return results
