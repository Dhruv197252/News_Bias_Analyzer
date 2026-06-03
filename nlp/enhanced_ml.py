"""
Enhanced ML Engine — TF-IDF + 10 Linguistic Features
------------------------------------------------------
Combines sparse TF-IDF with handcrafted features validated by Khan et al. (2025).
"""

import re
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline                import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.preprocessing           import StandardScaler
from sklearn.base                    import BaseEstimator, TransformerMixin
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import classification_report, roc_auc_score, confusion_matrix
from nltk.sentiment.vader            import SentimentIntensityAnalyzer
from textblob                        import TextBlob
import nltk

nltk.download("vader_lexicon", quiet=True)

# ── Shared constants ──────────────────────────────────────────────────────────
from nlp.bias_lexicon import WORD_TO_CATEGORY   # updated import

PASSIVE_INDICATORS = ["was ", "were ", "is being ", "are being ",
                      "has been ", "have been ", "had been "]
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after",
}


# ── 1. Custom Feature Extractor ───────────────────────────────────────────────

class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer: text → 10 handcrafted linguistic features."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([self._extract(text) for text in X])

    def _extract(self, text: str) -> list:
        if not text or not text.strip():
            return [0.0] * 10

        words     = text.lower().split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        n_words   = max(len(words), 1)
        n_sents   = max(len(sentences), 1)

        passive_count    = sum(1 for s in sentences if any(ind in s.lower() for ind in PASSIVE_INDICATORS))
        loaded_hits      = sum(1 for w in words if w in WORD_TO_CATEGORY)
        vader_scores     = self.sia.polarity_scores(text)
        sent_lengths     = [len(s.split()) for s in sentences]
        sent_var         = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
        first_person     = sum(1 for w in words if w in FIRST_PERSON)
        quote_count      = text.count('"') // 2
        stop_hits        = sum(1 for w in words if w in STOP_WORDS)

        return [
            passive_count / n_sents,                             # f1 passive rate
            loaded_hits / n_words,                               # f2 lexicon score
            TextBlob(text).sentiment.subjectivity,               # f3 subjectivity
            abs(vader_scores["compound"]),                       # f4 emotion
            min(sent_var / 100, 1.0),                            # f5 sent variance
            (text.count("!") + text.count("?")) / n_words,      # f6 punctuation
            first_person / n_words,                              # f7 first-person
            1.0 - min(quote_count / max(n_sents, 1), 1.0),      # f8 quote density (inverted)
            len(set(words)) / n_words,                           # f9 unique word ratio
            stop_hits / n_words,                                 # f10 stop word ratio
        ]

    def get_feature_names_out(self):
        return [
            "passive_rate", "lexicon_score", "subjectivity", "emotion_intensity",
            "sent_length_variance", "punctuation_density", "first_person_rate",
            "quote_density_inverted", "unique_word_ratio", "stop_word_ratio",
        ]


# ── 2. Build Pipeline ─────────────────────────────────────────────────────────

def build_enhanced_pipeline() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(
                max_features=8_000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=2, strip_accents="unicode",
            )),
            ("linguistic", Pipeline([
                ("extract", LinguisticFeatureExtractor()),
                ("scale",   StandardScaler()),
            ])),
        ])),
        ("clf", LogisticRegression(
            C=2.0, max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])


# ── 3. Train & Evaluate ───────────────────────────────────────────────────────

def train_enhanced(save: bool = True) -> Pipeline:
    df     = pd.read_csv("data/raw/babe_clean.csv")
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = build_enhanced_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=["Neutral", "Biased"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    if save:
        joblib.dump(pipeline, "models/enhanced_bias_classifier.pkl")
        print("Model saved to models/enhanced_bias_classifier.pkl")
    return pipeline


# ── 4. Inference ──────────────────────────────────────────────────────────────

def predict_enhanced(text: str, pipeline) -> dict:
    extractor  = LinguisticFeatureExtractor()
    features   = extractor._extract(text)
    feat_names = extractor.get_feature_names_out()
    proba      = pipeline.predict_proba([text])[0]
    bias_prob  = round(float(proba[1]), 4)
    return {
        "label":       "Biased" if bias_prob >= 0.5 else "Neutral",
        "probability": bias_prob,
        "features":    dict(zip(feat_names, [round(f, 4) for f in features])),
    }
