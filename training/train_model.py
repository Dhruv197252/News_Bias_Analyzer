"""
Model Trainer — XGBoost + TF-IDF Pipeline
------------------------------------------
Trains on merged_corpus.csv (BABE + AllSides).
Labels: Biased=1, Neutral=0

Render free tier safe:
  • XGBoost is CPU-only, no GPU required
  • Model file size ~2-5 MB (vs. BERT's 400 MB+)
  • Inference time < 50ms per article
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.pipeline          import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       roc_auc_score,
                                       f1_score)
from sklearn.utils             import class_weight as sklearn_class_weight

# Optional XGBoost — falls back to LR if not installed
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH   = os.path.join(ROOT, "data", "merged_corpus.csv")
BABE_PATH     = os.path.join(ROOT, "data", "babe_clean.csv")
MODEL_PATH_V2 = os.path.join(ROOT, "models", "bias_classifier_v2.pkl")
MODEL_PATH_V1 = os.path.join(ROOT, "models", "bias_classifier.pkl")   # legacy


# ── 1. Load Data ──────────────────────────────────────────────────────────────

def load_training_data() -> tuple[list[str], list[int]]:
    """Load merged corpus if available, else fall back to BABE."""
    path = CORPUS_PATH if os.path.exists(CORPUS_PATH) else BABE_PATH

    logger.info(f"Loading training data from: {path}")
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["text"]  = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len() >= 5]

    logger.info(f"  Total rows  : {len(df)}")
    logger.info(f"  Biased  (1) : {(df['label']==1).sum()}")
    logger.info(f"  Neutral (0) : {(df['label']==0).sum()}")

    return df["text"].tolist(), df["label"].tolist()


# ── 2. Build Pipeline ─────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    """
    TF-IDF + Logistic Regression.
    RAM usage: ~60 MB with 20k features. Safe for Render free tier.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = 25_000,
            ngram_range   = (1, 3),    # unigrams + bigrams + trigrams
            sublinear_tf  = True,
            min_df        = 2,
            strip_accents = "unicode",
            analyzer      = "word",
        )),
        ("clf", LogisticRegression(
            C             = 1.5,
            max_iter      = 1500,
            class_weight  = "balanced",
            random_state  = 42,
            solver        = "lbfgs",
            n_jobs        = -1,
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    """
    TF-IDF + XGBoost.
    Better accuracy than LR on this type of text data.
    RAM usage: ~80 MB. Safe for Render free tier.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = 25_000,
            ngram_range   = (1, 2),
            sublinear_tf  = True,
            min_df        = 2,
            strip_accents = "unicode",
        )),
        ("clf", XGBClassifier(
            n_estimators       = 300,
            max_depth          = 6,
            learning_rate      = 0.1,
            subsample          = 0.8,
            colsample_bytree   = 0.8,
            use_label_encoder  = False,
            eval_metric        = "logloss",
            random_state       = 42,
            n_jobs             = -1,
            tree_method        = "hist",   # CPU-only (no GPU needed)
        )),
    ])


# ── 3. Train + Evaluate ───────────────────────────────────────────────────────

def train(save: bool = True) -> Pipeline:
    """
    Train best available model, evaluate with 5-fold CV, save to disk.
    Returns fitted pipeline.
    """
    texts, labels = load_training_data()

    # Choose model
    if _XGBOOST_AVAILABLE:
        logger.info("\n🤖 Training TF-IDF + XGBoost pipeline...")
        pipeline = build_xgb_pipeline()
        model_name = "XGBoost"
    else:
        logger.info("\n🤖 XGBoost not installed. Training TF-IDF + LogisticRegression...")
        pipeline = build_lr_pipeline()
        model_name = "LogisticRegression"

    # 5-fold cross-validation
    logger.info(f"   Running 5-fold cross-validation ({model_name})...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro", n_jobs=-1)
    logger.info(f"   CV F1 (macro): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Final fit on all data
    logger.info(f"\n   Fitting on full dataset ({len(texts)} samples)...")
    pipeline.fit(texts, labels)
    logger.info("   ✅ Training complete.\n")

    # Evaluate on training set (sanity check)
    y_pred  = pipeline.predict(texts)
    y_proba = pipeline.predict_proba(texts)[:, 1]
    auc     = roc_auc_score(labels, y_proba)

    logger.info("── Training Set Performance (sanity check) ──────────────")
    logger.info(classification_report(labels, y_pred, target_names=["Neutral", "Biased"]))
    logger.info(f"   ROC-AUC  : {auc:.4f}")
    logger.info(f"   CV F1    : {cv_f1.mean():.4f} (5-fold, macro)")

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH_V2), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH_V2)
        logger.info(f"\n💾 Model saved to: {MODEL_PATH_V2}")
        logger.info(f"   Model size: {os.path.getsize(MODEL_PATH_V2) / 1024:.1f} KB")

    return pipeline


# ── 4. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = train(save=True)

    # Live inference test
    TEST_SENTENCES = [
        "The government announced new infrastructure spending on Tuesday.",
        "Radical extremists have shamefully undermined our democratic institutions.",
        "Officials confirmed the bill passed with bipartisan support in the Senate.",
        "The heroic whistleblower exposed the corrupt and tyrannical administration.",
        "Republic TV's propaganda machine continues its vile attacks on minorities.",
        "NDTV reports that the Prime Minister will visit flood-affected areas tomorrow.",
    ]

    print("\n── Live Inference Test ───────────────────────────────────")
    for sentence in TEST_SENTENCES:
        proba = pipeline.predict_proba([sentence])[0]
        bias_prob = round(float(proba[1]), 4)
        label = "Biased" if bias_prob >= 0.5 else "Neutral"
        bar = "🔴" if label == "Biased" else "🟢"
        print(f"{bar} [{label:<7} | {bias_prob:.0%}]  {sentence[:80]}")
