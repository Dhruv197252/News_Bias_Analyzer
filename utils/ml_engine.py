import os
import joblib
import pandas as pd
from sklearn.pipeline         import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (classification_report,
                                      confusion_matrix,
                                      roc_auc_score)


# ── 1. Config ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(_PROJECT_ROOT, "data", "babe_clean.csv")
MODEL_PATH    = os.path.join(_PROJECT_ROOT, "models", "bias_classifier.pkl")


# ── 2. Load Data ──────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> tuple[list[str], list[int]]:
    """Load the cleaned BABE CSV and return texts + labels."""
    print(f"📂 Loading data from '{path}'...")
    df = pd.read_csv(path)
    print(f"   {len(df)} rows loaded.\n")
    return df["text"].tolist(), df["label"].tolist()


# ── 3. Build the Sklearn Pipeline ─────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Two-stage sklearn Pipeline:

    Stage 1 — TfidfVectorizer
      • max_features=20000  : vocabulary cap (keeps memory sane)
      • ngram_range=(1,2)   : unigrams + bigrams
        e.g. "far left" is more informative than "far" and "left" alone
      • sublinear_tf=True   : log-scale TF to dampen very frequent words
      • min_df=2            : ignore terms that appear in only 1 document

    Stage 2 — LogisticRegression
      • C=1.0               : default regularisation (tune if needed)
      • max_iter=1000       : enough iterations to converge
      • class_weight='balanced' : compensates for the 1740/1381 imbalance
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = 20_000,
            ngram_range   = (1, 2),
            sublinear_tf  = True,
            min_df        = 2,
            strip_accents = "unicode",
        )),
        ("clf", LogisticRegression(
            C             = 1.0,
            max_iter      = 1000,
            class_weight  = "balanced",
            random_state  = 42,
        )),
    ])


# ── 4. Train & Evaluate ───────────────────────────────────────────────────────

def train(save: bool = True) -> Pipeline:
    """
    Full train → evaluate → save flow.

    Returns the fitted pipeline so it can be imported directly
    by the composite scorer without loading from disk.
    """
    texts, labels = load_data()

    # 80/20 stratified split (keeps class ratio equal in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size    = 0.2,
        random_state = 42,
        stratify     = labels,
    )
    print(f"   Train size : {len(X_train)}")
    print(f"   Test  size : {len(X_test)}\n")

    # Train
    print("🤖 Training TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("✅ Training complete.\n")

    # Evaluate
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]   # P(biased)

    print("── Classification Report ────────────────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=["Neutral", "Biased"]))

    print("── Confusion Matrix ─────────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}\n")

    print(f"── ROC-AUC Score ────────────────────────────────────")
    auc = roc_auc_score(y_test, y_proba)
    print(f"   AUC = {auc:.4f}\n")

    # Save
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"💾 Model saved to '{MODEL_PATH}'")

    return pipeline


# ── 5. Inference Helpers ──────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> Pipeline:
    """Load the saved model from disk."""
    return joblib.load(path)


def predict_bias(text: str, pipeline: Pipeline) -> dict:
    """
    Run inference on a single string.

    Returns
    -------
    dict:
        label       : "Biased" or "Neutral"
        probability : float 0.0–1.0  (confidence of being Biased)
    """
    proba = pipeline.predict_proba([text])[0]   # [P(neutral), P(biased)]
    bias_prob = round(float(proba[1]), 4)
    return {
        "label":       "Biased" if bias_prob >= 0.5 else "Neutral",
        "probability": bias_prob,
    }


# ── 6. Quick Demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Train and save the model
    pipeline = train(save=True)

    # Test on a few hand-crafted examples
    TEST_SENTENCES = [
        "The regime recklessly imposed draconian laws that crushed civil liberties.",
        "The government announced new infrastructure spending on Tuesday.",
        "Radical extremists have shamefully undermined our democratic institutions.",
        "Officials confirmed the bill passed with bipartisan support in the Senate.",
        "The heroic whistleblower exposed the corrupt and tyrannical administration.",
    ]

    print("\n── Live Inference Test ───────────────────────────────")
    for sentence in TEST_SENTENCES:
        result = predict_bias(sentence, pipeline)
        bar = "🔴" if result["label"] == "Biased" else "🟢"
        print(f"{bar} [{result['label']:<7} | {result['probability']:.0%}]  "
              f"{sentence[:80]}...")