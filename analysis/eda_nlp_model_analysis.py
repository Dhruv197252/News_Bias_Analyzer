"""
News Bias Analyzer - EDA + NLP + Model Comparison Analysis
Generates all visualizations and saves them to analysis/plots/
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "babe_clean.csv")
PLOTS_DIR  = os.path.join(BASE_DIR, "analysis", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)

# ── Global style ───────────────────────────────────────────────────────────────
PALETTE    = {"Biased": "#E74C3C", "Neutral": "#2ECC71"}
BG_COLOR   = "#0F1117"
CARD_COLOR = "#1A1D2E"
TEXT_COLOR = "#EAEAEA"
GRID_COLOR = "#2A2D3E"
ACCENT     = "#7C5CBF"
BLUE_ACCENT = "#3498DB"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    CARD_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  CARD_COLOR,
    "legend.edgecolor":  GRID_COLOR,
})

def save_fig(fig, name, dpi=150):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [SAVED] {name}")
    plt.close(fig)

print("=" * 65)
print("  NEWS BIAS ANALYZER — Full Analysis Pipeline")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 0. Load Data
# ══════════════════════════════════════════════════════════════════════════════
print("\n[0] Loading data...")
df = pd.read_csv(DATA_PATH)
df["label_name"] = df["label"].map({1: "Biased", 0: "Neutral"})
print(f"    Dataset shape: {df.shape}")
print(f"    Biased: {(df.label==1).sum()} | Neutral: {(df.label==0).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. EDA SECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] EDA Visualizations...")

# ── 1a. Class Distribution  ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Class Distribution — BABE Dataset", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02)

counts  = df["label_name"].value_counts()
colors  = [PALETTE[l] for l in counts.index]

# Bar
bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor=GRID_COLOR, linewidth=1.2, width=0.5)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f"{val:,}\n({val/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold", color=TEXT_COLOR)
axes[0].set_title("Absolute Count", fontsize=13)
axes[0].set_ylabel("Number of Samples", fontsize=11)
axes[0].set_ylim(0, counts.max() * 1.2)
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_axisbelow(True)

# Pie
wedge_props = {"edgecolor": BG_COLOR, "linewidth": 2.5}
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, wedgeprops=wedge_props,
            textprops={"color": TEXT_COLOR, "fontsize": 12, "fontweight": "bold"},
            pctdistance=0.75)
axes[1].set_title("Proportion", fontsize=13)

plt.tight_layout()
save_fig(fig, "01_class_distribution.png")

# ── 1b. Text Length Analysis ─────────────────────────────────────────────────
df["text_length"]   = df["text"].str.len()
df["word_count"]    = df["text"].str.split().str.len()
df["avg_word_len"]  = df["text"].apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0)
df["sentence_count"]= df["text"].apply(lambda t: max(len(re.split(r'[.!?]+', t)), 1))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Text Length Analysis by Class", fontsize=16, fontweight="bold", color=TEXT_COLOR)

metrics = [
    ("text_length",  "Character Count",      "chars"),
    ("word_count",   "Word Count",           "words"),
    ("avg_word_len", "Avg Word Length",      "chars/word"),
    ("sentence_count","Sentence Count",      "sentences"),
]

for ax, (col, title, unit) in zip(axes.flat, metrics):
    for cls in ["Biased", "Neutral"]:
        data = df[df["label_name"] == cls][col]
        ax.hist(data, bins=40, alpha=0.65, color=PALETTE[cls], label=cls,
                edgecolor="none", density=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(unit, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, "02_text_length_analysis.png")

# ── 1c. Box plots for length metrics ─────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 6))
fig.suptitle("Text Length Distribution — Box Plots by Class", fontsize=15, fontweight="bold")

for ax, (col, title, _unit) in zip(axes, metrics):
    data_b = df[df["label_name"]=="Biased"][col]
    data_n = df[df["label_name"]=="Neutral"][col]
    bp = ax.boxplot([data_b, data_n],
                    patch_artist=True, widths=0.5,
                    medianprops={"color": "white", "linewidth": 2},
                    flierprops={"marker":"o","color":GRID_COLOR,"alpha":0.4,"markersize":3})
    for patch, color in zip(bp["boxes"], [PALETTE["Biased"], PALETTE["Neutral"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Biased", "Neutral"], fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save_fig(fig, "03_text_length_boxplots.png")

# ── 1d. Correlation heat map of numeric features ──────────────────────────────
numeric_cols = ["text_length", "word_count", "avg_word_len", "sentence_count", "label"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("Feature Correlation Heat Map", fontsize=14, fontweight="bold")
cmap = LinearSegmentedColormap.from_list("bias_cmap", ["#2ECC71", CARD_COLOR, "#E74C3C"])
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax, cmap=cmap, annot=True, fmt=".2f",
            linewidths=0.5, linecolor=BG_COLOR,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 11, "weight": "bold"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
save_fig(fig, "04_correlation_heatmap.png")

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# 2. NLP Feature EDA
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
print("\n[2] NLP Feature Engineering & Visualizations...")

# Install / import VADER & TextBlob
import nltk
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords",     quiet=True)
from nltk.sentiment.vader     import SentimentIntensityAnalyzer
from nltk.corpus              import stopwords
from textblob                 import TextBlob

sia        = SentimentIntensityAnalyzer()
STOP_WORDS = set(stopwords.words("english"))

print("   Computing NLP features (may take ~30s)...")

def compute_nlp_features(text):
    if not text or not text.strip():
        return pd.Series([0.0]*7)
    blob    = TextBlob(text)
    vader   = sia.polarity_scores(text)
    words   = text.lower().split()
    n_words = max(len(words), 1)
    passive_count = sum(1 for s in re.split(r'[.!?]+', text)
                        if any(p in s.lower() for p in ["was ", "were ", "is being ", "are being ", "has been ", "have been "]))
    n_sents = max(len(re.split(r'[.!?]+', text)), 1)
    return pd.Series([
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
        abs(vader["compound"]),
        passive_count / n_sents,
        sum(1 for w in words if w not in STOP_WORDS) / n_words,
        len(set(words)) / n_words,
        (text.count("!") + text.count("?")) / n_words,
    ])

feature_names = [
    "polarity", "subjectivity", "emotion_intensity",
    "passive_rate", "content_word_ratio",
    "unique_word_ratio", "punct_density"
]

nlp_features = df["text"].apply(compute_nlp_features)
nlp_features.columns = feature_names
df_nlp = pd.concat([df, nlp_features], axis=1)

# ── 2a. NLP Feature Distributions ────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("NLP Feature Distributions by Class", fontsize=16, fontweight="bold")

for ax, feat in zip(axes.flat, feature_names):
    for cls in ["Biased", "Neutral"]:
        data = df_nlp[df_nlp["label_name"] == cls][feat]
        ax.hist(data, bins=35, alpha=0.65, color=PALETTE[cls], label=cls,
                edgecolor="none", density=True)
    ax.set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

axes.flat[-1].set_visible(False)  # hide 8th subplot
plt.tight_layout()
save_fig(fig, "05_nlp_feature_distributions.png")

# ── 2b. NLP Feature Violin Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("NLP Feature Violin Plots by Class", fontsize=16, fontweight="bold")

for ax, feat in zip(axes.flat, feature_names):
    data_b = df_nlp[df_nlp["label_name"]=="Biased"][feat].values
    data_n = df_nlp[df_nlp["label_name"]=="Neutral"][feat].values
    vp = ax.violinplot([data_b, data_n], positions=[1, 2], showmedians=True, showextrema=False)
    for body, color in zip(vp["bodies"], [PALETTE["Biased"], PALETTE["Neutral"]]):
        body.set_facecolor(color)
        body.set_alpha(0.65)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Biased", "Neutral"], fontsize=9)
    ax.set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

axes.flat[-1].set_visible(False)
plt.tight_layout()
save_fig(fig, "06_nlp_feature_violins.png")

# ── 2c. NLP Feature Pair correlations ────────────────────────────────────────
corr_nlp = df_nlp[feature_names + ["label"]].corr()["label"].drop("label").sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = [PALETTE["Biased"] if v > 0 else PALETTE["Neutral"] for v in corr_nlp.values]
bars = ax.barh(corr_nlp.index, corr_nlp.values, color=colors_bar, edgecolor=GRID_COLOR, height=0.6)
for bar, v in zip(bars, corr_nlp.values):
    ax.text(v + (0.002 if v >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
            f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=10, fontweight="bold")
ax.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
ax.set_title("NLP Feature Correlation with Bias Label", fontsize=14, fontweight="bold")
ax.set_xlabel("Pearson Correlation Coefficient", fontsize=11)
ax.grid(axis="x", alpha=0.3)
fig.suptitle("Positive = More Biased  |  Negative = More Neutral", fontsize=10, y=1.01, color="gray")
plt.tight_layout()
save_fig(fig, "07_nlp_feature_correlations.png")

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# 3. WORD CLOUD & TOP N-GRAMS
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
print("\n[3] Word Cloud & N-gram Visualizations...")

try:
    from wordcloud import WordCloud
    WC_AVAILABLE = True
except ImportError:
    WC_AVAILABLE = False
    print("   ⚠  wordcloud not installed — skipping word-cloud plots.")

def get_tokens(texts, stop_words=STOP_WORDS, min_len=3):
    tokens = []
    for t in texts:
        for w in re.findall(r"[a-zA-Z]+", t.lower()):
            if w not in stop_words and len(w) >= min_len:
                tokens.append(w)
    return tokens

biased_texts  = df[df["label"]==1]["text"].tolist()
neutral_texts = df[df["label"]==0]["text"].tolist()

biased_tokens  = get_tokens(biased_texts)
neutral_tokens = get_tokens(neutral_texts)

if WC_AVAILABLE:
    # ── 3a. Word Cloud comparison ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Word Clouds — Biased vs Neutral", fontsize=16, fontweight="bold")

    for ax, tokens, cls, bg, fg in [
        (axes[0], biased_tokens,  "Biased",  "#1a0a0a", "Reds"),
        (axes[1], neutral_tokens, "Neutral", "#0a1a0a", "Greens"),
    ]:
        text_blob = " ".join(tokens)
        wc = WordCloud(
            width=700, height=400,
            background_color=bg,
            colormap=fg,
            max_words=150,
            min_font_size=10,
            prefer_horizontal=0.85,
        ).generate(text_blob)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{cls} Articles", fontsize=14, fontweight="bold", color=PALETTE[cls])
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, "08_wordclouds.png")

    # ── 3b. Frequency-contrast WordCloud ─────────────────────────────────────
    biased_freq  = Counter(biased_tokens)
    neutral_freq = Counter(neutral_tokens)
    # Words unique to biased
    exclusive_biased = {w: f for w, f in biased_freq.items()
                        if f > 3 and neutral_freq.get(w, 0) < f * 0.3}
    exclusive_neutral = {w: f for w, f in neutral_freq.items()
                         if f > 3 and biased_freq.get(w, 0) < f * 0.3}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Exclusive Vocabulary — Biased vs Neutral", fontsize=16, fontweight="bold")

    for ax, freq_dict, cls, bg, cm in [
        (axes[0], exclusive_biased,  "Biased",  "#1a0a0a", "Reds"),
        (axes[1], exclusive_neutral, "Neutral", "#0a1a0a", "Greens"),
    ]:
        if freq_dict:
            wc = WordCloud(width=700, height=400, background_color=bg,
                           colormap=cm, max_words=100, prefer_horizontal=0.9
                           ).generate_from_frequencies(freq_dict)
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Exclusive to {cls}", fontsize=13, fontweight="bold", color=PALETTE[cls])
        ax.axis("off")

    plt.tight_layout()
    save_fig(fig, "09_exclusive_wordclouds.png")

# ── 3c. Top Unigrams Side-by-Side ────────────────────────────────────────────
def top_n(tokens, n=20):
    return Counter(tokens).most_common(n)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Top 20 Unigrams by Class", fontsize=16, fontweight="bold")

for ax, tokens, cls in [
    (axes[0], biased_tokens,  "Biased"),
    (axes[1], neutral_tokens, "Neutral"),
]:
    top = top_n(tokens, 20)
    words_top, counts_top = zip(*top)
    bars = ax.barh(list(reversed(words_top)), list(reversed(counts_top)),
                   color=PALETTE[cls], edgecolor=GRID_COLOR, alpha=0.85)
    for bar, cnt in zip(bars, reversed(counts_top)):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                str(cnt), va="center", fontsize=9)
    ax.set_title(f"{cls} — Top 20 Words", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
save_fig(fig, "10_top_unigrams.png")

# ── 3d. Top Bigrams ───────────────────────────────────────────────────────────
from nltk import bigrams as nltk_bigrams, trigrams as nltk_trigrams

def get_ngrams(texts, n=2, stop_words=STOP_WORDS, min_len=3, top_k=15):
    all_ngrams = []
    for text in texts:
        tokens = [w for w in re.findall(r"[a-zA-Z]+", text.lower())
                  if w not in stop_words and len(w) >= min_len]
        if n == 2:
            all_ngrams.extend([" ".join(ng) for ng in nltk_bigrams(tokens)])
        elif n == 3:
            all_ngrams.extend([" ".join(ng) for ng in nltk_trigrams(tokens)])
    return Counter(all_ngrams).most_common(top_k)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Top Bigrams & Trigrams by Class", fontsize=16, fontweight="bold")

for row, (n, label) in enumerate([(2, "Bigrams"), (3, "Trigrams")]):
    for col, (texts, cls) in enumerate([(biased_texts, "Biased"), (neutral_texts, "Neutral")]):
        ax     = axes[row][col]
        ngrams = get_ngrams(texts, n=n, top_k=15)
        if not ngrams:
            ax.set_visible(False)
            continue
        phrases, cnts = zip(*ngrams)
        bars = ax.barh(list(reversed(phrases)), list(reversed(cnts)),
                       color=PALETTE[cls], edgecolor=GRID_COLOR, alpha=0.85)
        for bar, cnt in zip(bars, reversed(cnts)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(cnt), va="center", fontsize=9)
        ax.set_title(f"{cls} — Top 15 {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
save_fig(fig, "11_bigrams_trigrams.png")

# ── 3e. Sentiment Distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Sentiment & Subjectivity Analysis by Class", fontsize=15, fontweight="bold")

sentiment_cols = ["polarity", "subjectivity", "emotion_intensity"]
sentiment_labels = ["TextBlob Polarity", "TextBlob Subjectivity", "VADER Emotion Intensity"]

for ax, col, lbl in zip(axes, sentiment_cols, sentiment_labels):
    for cls in ["Biased", "Neutral"]:
        data = df_nlp[df_nlp["label_name"]==cls][col]
        ax.hist(data, bins=40, alpha=0.65, color=PALETTE[cls], label=cls,
                edgecolor="none", density=True)
    ax.set_title(lbl, fontsize=12, fontweight="bold")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, "12_sentiment_analysis.png")

# ── 3f. VADER Quadrant Plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
for cls in ["Biased", "Neutral"]:
    sub = df_nlp[df_nlp["label_name"]==cls]
    ax.scatter(sub["polarity"], sub["subjectivity"],
               alpha=0.25, s=18, color=PALETTE[cls], label=cls)

ax.axhline(0.5, color="white", linewidth=1, linestyle="--", alpha=0.4)
ax.axvline(0.0, color="white", linewidth=1, linestyle="--", alpha=0.4)
ax.set_xlabel("Polarity (Negative ← → Positive)", fontsize=12)
ax.set_ylabel("Subjectivity (Objective ← → Subjective)", fontsize=12)
ax.set_title("Polarity vs Subjectivity Scatter", fontsize=14, fontweight="bold")
ax.legend(markerscale=3, fontsize=11)
ax.text(-0.95, 0.98, "Neg. Subjective", fontsize=9, color="gray", ha="left", va="top")
ax.text( 0.55, 0.98, "Pos. Subjective", fontsize=9, color="gray", ha="left", va="top")
ax.text(-0.95, 0.01, "Neg. Objective",  fontsize=9, color="gray", ha="left", va="bottom")
ax.text( 0.55, 0.01, "Pos. Objective",  fontsize=9, color="gray", ha="left", va="bottom")
ax.grid(alpha=0.2)
plt.tight_layout()
save_fig(fig, "13_polarity_subjectivity_scatter.png")

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# 4. MODEL COMPARISON
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
print("\n[4] Model Comparison (LR vs NB vs SVM vs RF)...")

from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.naive_bayes       import MultinomialNB
from sklearn.svm               import LinearSVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, roc_curve,
    precision_recall_curve
)
from sklearn.calibration       import CalibratedClassifierCV

texts  = df["text"].tolist()
labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

TFIDF_BASE = dict(max_features=20_000, ngram_range=(1,2),
                  sublinear_tf=True, min_df=2, strip_accents="unicode")

# ── Define models ─────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_BASE)),
        ("clf",   LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced",
                                     solver="lbfgs", random_state=42)),
    ]),
    "Multinomial NB": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1,2),
                                  min_df=2, strip_accents="unicode")),
        ("clf",   MultinomialNB(alpha=0.5)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_BASE)),
        ("clf",   CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=3000, class_weight="balanced", random_state=42)
        )),
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5_000, ngram_range=(1,2),
                                  sublinear_tf=True, min_df=2)),
        ("clf",   RandomForestClassifier(n_estimators=200, max_depth=30,
                                         class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
}

results = {}
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in MODELS.items():
    print(f"   Training {name}...")
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv_acc = cross_val_score(pipe, texts, labels, cv=cv, scoring="accuracy",  n_jobs=-1)
    cv_f1  = cross_val_score(pipe, texts, labels, cv=cv, scoring="f1_weighted", n_jobs=-1)
    cv_auc = cross_val_score(pipe, texts, labels, cv=cv, scoring="roc_auc",   n_jobs=-1)

    results[name] = {
        "accuracy":   accuracy_score(y_test, y_pred),
        "f1":         f1_score(y_test, y_pred, average="weighted"),
        "precision":  precision_score(y_test, y_pred, average="weighted"),
        "recall":     recall_score(y_test, y_pred, average="weighted"),
        "roc_auc":    roc_auc_score(y_test, y_proba),
        "cv_acc":     cv_acc.mean(),
        "cv_acc_std": cv_acc.std(),
        "cv_f1":      cv_f1.mean(),
        "cv_f1_std":  cv_f1.std(),
        "cv_auc":     cv_auc.mean(),
        "cv_auc_std": cv_auc.std(),
        "y_pred":     y_pred,
        "y_proba":    y_proba,
        "cm":         confusion_matrix(y_test, y_pred),
    }
    print(f"       Accuracy={results[name]['accuracy']:.4f}  AUC={results[name]['roc_auc']:.4f}")

# ── 4a. Model Comparison Bar Chart (Main Figure) ──────────────────────────────
metrics_to_plot = ["accuracy", "f1", "precision", "recall", "roc_auc"]
metric_labels   = ["Accuracy", "F1-Score", "Precision", "Recall", "ROC-AUC"]
model_names     = list(results.keys())
model_colors    = [ACCENT, "#3498DB", "#F39C12", "#1ABC9C"]

n_metrics = len(metrics_to_plot)
n_models  = len(model_names)
x         = np.arange(n_metrics)
width     = 0.18

fig, ax = plt.subplots(figsize=(16, 8))
fig.suptitle("Model Performance Comparison — Test Set", fontsize=17, fontweight="bold")

for i, (name, color) in enumerate(zip(model_names, model_colors)):
    values = [results[name][m] for m in metrics_to_plot]
    offset = (i - n_models/2 + 0.5) * width
    bars   = ax.bar(x + offset, values, width, label=name, color=color,
                    alpha=0.88, edgecolor=GRID_COLOR, linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0.55, 1.05)
ax.legend(fontsize=11, loc="upper right")
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
# Highlight LR bar region
ax.axhspan(0.55, 1.05, xmin=0.0, xmax=0.0, alpha=0)  # dummy for layout

plt.tight_layout()
save_fig(fig, "14_model_comparison_bar.png")

# ── 4b. Cross-Validation Comparison ──────────────────────────────────────────
cv_metrics  = ["cv_acc", "cv_f1", "cv_auc"]
cv_stds     = ["cv_acc_std", "cv_f1_std", "cv_auc_std"]
cv_labels   = ["CV Accuracy", "CV F1-Score", "CV ROC-AUC"]

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("5-Fold Cross-Validation Comparison", fontsize=16, fontweight="bold")

x    = np.arange(len(cv_metrics))
for i, (name, color) in enumerate(zip(model_names, model_colors)):
    vals = [results[name][m] for m in cv_metrics]
    stds = [results[name][s] for s in cv_stds]
    offset = (i - n_models/2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=name, color=color,
                  alpha=0.88, edgecolor=GRID_COLOR, linewidth=0.8,
                  yerr=stds, capsize=4, error_kw={"ecolor":"white","linewidth":1.5})
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(cv_labels, fontsize=13)
ax.set_ylabel("Score (with ±1 std)", fontsize=12)
ax.set_ylim(0.55, 1.05)
ax.legend(fontsize=11)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, "15_cross_validation_comparison.png")

# ── 4c. Radar Chart ───────────────────────────────────────────────────────────
from matplotlib.patches import FancyArrowPatch

radar_metrics = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
N = len(radar_metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
ax.set_facecolor(CARD_COLOR)
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Model Performance Radar Chart", fontsize=16, fontweight="bold", color=TEXT_COLOR)

for (name, color) in zip(model_names, model_colors):
    values = [
        results[name]["accuracy"],
        results[name]["f1"],
        results[name]["precision"],
        results[name]["recall"],
        results[name]["roc_auc"],
    ]
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=name)
    ax.fill(angles, values, color=color, alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=12, color=TEXT_COLOR)
ax.set_ylim(0.60, 1.0)
ax.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
ax.grid(color=GRID_COLOR, alpha=0.5)
ax.spines["polar"].set_color(GRID_COLOR)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11)
plt.tight_layout()
save_fig(fig, "16_radar_chart.png")

# ── 4d. Confusion Matrices (2x2 grid) ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold")

for ax, (name, color) in zip(axes.flat, zip(model_names, model_colors)):
    cm = results[name]["cm"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("mono", [CARD_COLOR, color])
    sns.heatmap(cm_pct, ax=ax, annot=False, fmt="", cmap=cmap,
                linewidths=1.5, linecolor=BG_COLOR,
                cbar_kws={"shrink": 0.7})
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.5, f"{cm[i,j]}\n({cm_pct[i,j]:.1%})",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white")
    ax.set_xticklabels(["Neutral", "Biased"], fontsize=11)
    ax.set_yticklabels(["Neutral", "Biased"], rotation=0, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    ax.set_title(name, fontsize=13, fontweight="bold", color=color)

plt.tight_layout()
save_fig(fig, "17_confusion_matrices.png")

# ── 4e. ROC Curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
fig.suptitle("ROC Curves — All Models", fontsize=15, fontweight="bold")

for (name, color) in zip(model_names, model_colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    auc_val      = results[name]["roc_auc"]
    lw = 3 if name == "Logistic Regression" else 1.8
    ax.plot(fpr, tpr, color=color, linewidth=lw,
            label=f"{name} (AUC={auc_val:.4f})")

ax.plot([0,1],[0,1], "w--", linewidth=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "18_roc_curves.png")

# ── 4f. Precision-Recall Curves ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
fig.suptitle("Precision-Recall Curves — All Models", fontsize=15, fontweight="bold")

for (name, color) in zip(model_names, model_colors):
    prec, rec, _ = precision_recall_curve(y_test, results[name]["y_proba"])
    lw = 3 if name == "Logistic Regression" else 1.8
    ax.plot(rec, prec, color=color, linewidth=lw, label=name)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "19_precision_recall_curves.png")

# ── 4g. Summary Table Plot ────────────────────────────────────────────────────
summary_data = []
for name in model_names:
    r = results[name]
    summary_data.append([
        name,
        f"{r['accuracy']:.4f}",
        f"{r['f1']:.4f}",
        f"{r['precision']:.4f}",
        f"{r['recall']:.4f}",
        f"{r['roc_auc']:.4f}",
        f"{r['cv_acc']:.4f}±{r['cv_acc_std']:.4f}",
        f"{r['cv_auc']:.4f}±{r['cv_auc_std']:.4f}",
    ])

cols = ["Model", "Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV Acc", "CV AUC"]

fig, ax = plt.subplots(figsize=(16, 4))
ax.axis("off")
fig.patch.set_facecolor(BG_COLOR)
table = ax.table(cellText=summary_data, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(cols))))

# Style header
for j in range(len(cols)):
    table[(0, j)].set_facecolor(ACCENT)
    table[(0, j)].set_text_props(color="white", fontweight="bold")

# Style LR row (highlight)
for j in range(len(cols)):
    table[(1, j)].set_facecolor("#291a40")
    table[(1, j)].set_text_props(fontweight="bold", color=TEXT_COLOR)

# Other rows
for i in range(2, len(model_names)+1):
    for j in range(len(cols)):
        table[(i, j)].set_facecolor(CARD_COLOR)
        table[(i, j)].set_text_props(color=TEXT_COLOR)

fig.suptitle("Model Comparison Summary Table", fontsize=14, fontweight="bold", color=TEXT_COLOR)
plt.tight_layout()
save_fig(fig, "20_summary_table.png")

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# 5. LOGISTIC REGRESSION DEEP DIVE
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
print("\n[5] Logistic Regression Deep Dive...")

lr_pipe   = MODELS["Logistic Regression"]
lr_clf    = lr_pipe.named_steps["clf"]
lr_tfidf  = lr_pipe.named_steps["tfidf"]
vocab     = lr_tfidf.get_feature_names_out()
coefs     = lr_clf.coef_[0]

# Sort by coefficient
idx_sorted   = np.argsort(coefs)
top_biased   = idx_sorted[-25:][::-1]
top_neutral  = idx_sorted[:25]

# ── 5a. Top Coefficients ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle("Logistic Regression — Top Feature Coefficients", fontsize=16, fontweight="bold")

for ax, idxs, cls in [(axes[0], top_biased, "Biased"), (axes[1], top_neutral, "Neutral")]:
    feats = [vocab[i] for i in idxs]
    coef_vals = [coefs[i] for i in idxs]
    colors_c  = [PALETTE[cls]] * len(feats)

    bars = ax.barh(list(reversed(feats)), list(reversed(coef_vals)),
                   color=colors_c, edgecolor=GRID_COLOR, alpha=0.85)
    for bar, val in zip(bars, reversed(coef_vals)):
        ax.text(val * 1.02 if val > 0 else val * 1.02,
                bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", fontsize=8)
    ax.set_title(f"Top 25 Features → {cls}", fontsize=13, fontweight="bold", color=PALETTE[cls])
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=11)
    ax.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.4)
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
save_fig(fig, "21_lr_top_coefficients.png")

# ── 5b. Coefficient Distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(coefs, bins=80, color=ACCENT, edgecolor=GRID_COLOR, alpha=0.8)
ax.axvline(0, color="white", linewidth=2, linestyle="--")
ax.axvline(coefs.mean(), color=PALETTE["Biased"], linewidth=2, linestyle="-", label=f"Mean: {coefs.mean():.3f}")
ax.set_title("Logistic Regression — Coefficient Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Coefficient Value", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "22_lr_coef_distribution.png")

# ── 5c. Probability Histogram ─────────────────────────────────────────────────
lr_proba = results["Logistic Regression"]["y_proba"]
lr_pred  = results["Logistic Regression"]["y_pred"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Logistic Regression — Prediction Probability Analysis", fontsize=15, fontweight="bold")

# Histogram by actual class
for ax_i, (ax, cls_val, cls_name) in enumerate(
    [(axes[0], 1, "Biased"), (axes[1], 0, "Neutral")]
):
    mask  = np.array(y_test) == cls_val
    probs = lr_proba[mask]
    ax.hist(probs, bins=30, color=PALETTE[cls_name], edgecolor=GRID_COLOR, alpha=0.8)
    ax.axvline(0.5, color="white", linewidth=2, linestyle="--", label="Decision Boundary")
    ax.set_title(f"P(Biased) distribution for actually {cls_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Probability of Bias", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig(fig, "23_lr_probability_histogram.png")

# ── 5d. Calibration Curve ────────────────────────────────────────────────────
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("Logistic Regression — Calibration Curve", fontsize=14, fontweight="bold")

frac_pos, mean_pred = calibration_curve(y_test, lr_proba, n_bins=10)
ax.plot(mean_pred, frac_pos, "o-", color=ACCENT, linewidth=2.5, markersize=8, label="LR Calibration")
ax.plot([0, 1], [0, 1], "w--", linewidth=1.5, alpha=0.6, label="Perfect Calibration")
ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color=ACCENT)
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "24_lr_calibration_curve.png")

# ── 5e. Learning Curve ───────────────────────────────────────────────────────
from sklearn.model_selection import learning_curve

print("   Computing learning curves (may take ~60s)...")

lr_simple = Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_BASE)),
    ("clf",   LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced",
                                 solver="lbfgs", random_state=42)),
])

train_sizes, train_scores, val_scores = learning_curve(
    lr_simple, texts, labels,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Logistic Regression — Learning Curve (ROC-AUC)", fontsize=14, fontweight="bold")

ax.plot(train_sizes, train_mean, "o-", color=PALETTE["Biased"], linewidth=2.5, label="Training Score")
ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color=PALETTE["Biased"])
ax.plot(train_sizes, val_mean, "s-", color=PALETTE["Neutral"], linewidth=2.5, label="Validation Score")
ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color=PALETTE["Neutral"])

ax.set_xlabel("Training Set Size", fontsize=12)
ax.set_ylabel("ROC-AUC Score",     fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "25_lr_learning_curve.png")

# ── 5f. Regularization Sensitivity ───────────────────────────────────────────
print("   Computing regularization sensitivity...")
C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
C_acc, C_auc = [], []

for C_val in C_values:
    pipe_c = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_BASE)),
        ("clf",   LogisticRegression(C=C_val, max_iter=2000, class_weight="balanced",
                                     solver="lbfgs", random_state=42)),
    ])
    pipe_c.fit(X_train, y_train)
    y_p = pipe_c.predict(X_test)
    y_pr = pipe_c.predict_proba(X_test)[:,1]
    C_acc.append(accuracy_score(y_test, y_p))
    C_auc.append(roc_auc_score(y_test, y_pr))

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Logistic Regression — Regularization Sensitivity (C hyperparameter)", fontsize=14, fontweight="bold")
x_c = np.arange(len(C_values))
ax.plot(x_c, C_acc, "o-", color=PALETTE["Biased"], linewidth=2.5, label="Test Accuracy", markersize=8)
ax.plot(x_c, C_auc, "s-", color=PALETTE["Neutral"], linewidth=2.5, label="ROC-AUC", markersize=8)
for i, (acc, auc) in enumerate(zip(C_acc, C_auc)):
    ax.text(i, acc + 0.003, f"{acc:.3f}", ha="center", fontsize=8, color=PALETTE["Biased"])
    ax.text(i, auc - 0.010, f"{auc:.3f}", ha="center", fontsize=8, color=PALETTE["Neutral"])
ax.set_xticks(x_c)
ax.set_xticklabels([str(c) for c in C_values], fontsize=10)
ax.set_xlabel("C (Inverse Regularization Strength)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, "26_lr_regularization_sensitivity.png")

# ── 5g. Per-class F1 detailed ────────────────────────────────────────────────
from sklearn.metrics import classification_report
cr = classification_report(y_test, results["Logistic Regression"]["y_pred"],
                            target_names=["Neutral","Biased"], output_dict=True)

classes  = ["Neutral", "Biased"]
metrics_c = ["precision", "recall", "f1-score", "support"]
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Logistic Regression — Per-Class Classification Metrics", fontsize=15, fontweight="bold")

for ax, metric in zip(axes, ["precision", "recall", "f1-score"]):
    vals   = [cr[cls][metric] for cls in classes]
    colors = [PALETTE["Neutral"], PALETTE["Biased"]]
    bars   = ax.bar(classes, vals, color=colors, edgecolor=GRID_COLOR, width=0.45, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_title(metric.title(), fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

plt.tight_layout()
save_fig(fig, "27_lr_per_class_metrics.png")

# ── 5h. Error Analysis — Misclassified examples ───────────────────────────────
y_test_arr = np.array(y_test)
X_test_arr = np.array(X_test)

fp_mask = (y_test_arr==0) & (results["Logistic Regression"]["y_pred"]==1)
fn_mask = (y_test_arr==1) & (results["Logistic Regression"]["y_pred"]==0)

fp_texts = X_test_arr[fp_mask][:5]  # False Positives (Neutral → predicted Biased)
fn_texts = X_test_arr[fn_mask][:5]  # False Negatives (Biased → predicted Neutral)

print(f"   FP (Neutral→Biased): {fp_mask.sum()}")
print(f"   FN (Biased→Neutral): {fn_mask.sum()}")

# ── 5i. Probability scored word-level analysis via TFIDF ──────────────────────
# Top biased words by coefficient × idf weight
idf_weights = lr_tfidf.idf_
positive_importance = coefs * idf_weights  # how much each feature actually matters
top20_pos_idx = np.argsort(positive_importance)[-20:][::-1]
top20_neg_idx = np.argsort(positive_importance)[:20]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("LR Feature Importance: Coefficient × IDF Weight", fontsize=15, fontweight="bold")

for ax, idxs, title, color in [
    (axes[0], top20_pos_idx, "Top 20 → BIASED",  PALETTE["Biased"]),
    (axes[1], top20_neg_idx, "Top 20 → NEUTRAL",  PALETTE["Neutral"]),
]:
    feats = [vocab[i] for i in idxs]
    vals  = [abs(positive_importance[i]) for i in idxs]
    bars  = ax.barh(list(reversed(feats)), list(reversed(vals)),
                    color=color, edgecolor=GRID_COLOR, alpha=0.85)
    for bar, val in zip(bars, reversed(vals)):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", color=color)
    ax.set_xlabel("|Coefficient × IDF|", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
save_fig(fig, "28_lr_coef_idf_importance.png")

# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# 6. FINAL SUMMARY DASHBOARD
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
print("\n[6] Generating Final Summary Dashboard...")

fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)
gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.35, hspace=0.45)

fig.suptitle("News Bias Analyzer — Analysis Summary Dashboard",
             fontsize=20, fontweight="bold", color=TEXT_COLOR, y=0.99)

# ── A. Class Distribution (top-left) ─────────────────────────────────────────
ax_pie = fig.add_subplot(gs[0, 0])
counts = df["label_name"].value_counts()
ax_pie.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=[PALETTE[l] for l in counts.index],
           wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
           textprops={"color": TEXT_COLOR, "fontsize": 10, "fontweight": "bold"},
           pctdistance=0.75)
ax_pie.set_title("Class Balance", fontsize=12, fontweight="bold")

# ── B. Word Count Distribution ────────────────────────────────────────────────
ax_wc = fig.add_subplot(gs[0, 1])
for cls in ["Biased", "Neutral"]:
    ax_wc.hist(df[df["label_name"]==cls]["word_count"], bins=30,
               alpha=0.65, color=PALETTE[cls], label=cls, density=True)
ax_wc.set_title("Word Count Distribution", fontsize=12, fontweight="bold")
ax_wc.set_xlabel("Words per sample")
ax_wc.legend(fontsize=9)
ax_wc.grid(alpha=0.25)

# ── C. Model Comparison (top-right, spans 2 cols) ────────────────────────────
ax_cmp = fig.add_subplot(gs[0, 2:])
bar_width = 0.18
x = np.arange(5)
flat_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
for i, (name, color) in enumerate(zip(model_names, model_colors)):
    vals   = [results[name][m] for m in flat_metrics]
    offset = (i - n_models/2 + 0.5) * bar_width
    ax_cmp.bar(x + offset, vals, bar_width, label=name, color=color,
               alpha=0.88, edgecolor=GRID_COLOR, linewidth=0.5)
ax_cmp.set_xticks(x)
ax_cmp.set_xticklabels(["Accuracy","F1","Precision","Recall","AUC"], fontsize=10)
ax_cmp.set_ylim(0.6, 1.0)
ax_cmp.legend(fontsize=9, loc="upper right")
ax_cmp.set_title("Model Performance (Test Set)", fontsize=12, fontweight="bold")
ax_cmp.grid(axis="y", alpha=0.25)

# ── D. ROC Curve ─────────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 0:2])
for name, color in zip(model_names, model_colors):
    fpr_r, tpr_r, _ = roc_curve(y_test, results[name]["y_proba"])
    lw = 3 if name == "Logistic Regression" else 1.5
    ax_roc.plot(fpr_r, tpr_r, color=color, linewidth=lw,
                label=f"{name} ({results[name]['roc_auc']:.3f})")
ax_roc.plot([0,1],[0,1],"w--",alpha=0.4)
ax_roc.set_title("ROC Curves", fontsize=12, fontweight="bold")
ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
ax_roc.legend(fontsize=9, loc="lower right")
ax_roc.grid(alpha=0.25)

# ── E. LR Top Biased Features ────────────────────────────────────────────────
ax_feats = fig.add_subplot(gs[1, 2:])
top10_b = idx_sorted[-10:][::-1]
feats10 = [vocab[i] for i in top10_b]
vals10  = [coefs[i] for i in top10_b]
bars = ax_feats.barh(list(reversed(feats10)), list(reversed(vals10)),
                     color=PALETTE["Biased"], edgecolor=GRID_COLOR, alpha=0.85)
ax_feats.set_title("Top 10 Biased Features (LR Coefs)", fontsize=12, fontweight="bold")
ax_feats.set_xlabel("Coefficient")
ax_feats.grid(axis="x", alpha=0.25)

# ── F. Calibration ───────────────────────────────────────────────────────────
ax_cal = fig.add_subplot(gs[2, 0:2])
ax_cal.plot(mean_pred, frac_pos, "o-", color=ACCENT, linewidth=2.5, markersize=6, label="LR Calibration")
ax_cal.plot([0,1],[0,1],"w--",alpha=0.5,label="Perfect")
ax_cal.set_title("LR Calibration Curve", fontsize=12, fontweight="bold")
ax_cal.set_xlabel("Mean Predicted Probability"); ax_cal.set_ylabel("Fraction of Positives")
ax_cal.legend(fontsize=9); ax_cal.grid(alpha=0.25)

# ── G. Confusion Matrix (LR) ──────────────────────────────────────────────────
ax_conf = fig.add_subplot(gs[2, 2])
cm_lr   = results["Logistic Regression"]["cm"]
cm_pct  = cm_lr.astype(float) / cm_lr.sum(axis=1, keepdims=True)
cmap_lr = LinearSegmentedColormap.from_list("lr_cmap", [CARD_COLOR, ACCENT])
sns.heatmap(cm_pct, ax=ax_conf, cmap=cmap_lr, annot=False,
            linewidths=1.5, linecolor=BG_COLOR, cbar=False)
for i in range(2):
    for j in range(2):
        ax_conf.text(j+0.5, i+0.5, f"{cm_lr[i,j]}\n({cm_pct[i,j]:.1%})",
                     ha="center", va="center", fontsize=11, fontweight="bold", color="white")
ax_conf.set_xticklabels(["Neutral","Biased"], fontsize=10)
ax_conf.set_yticklabels(["Neutral","Biased"], rotation=0, fontsize=10)
ax_conf.set_title("LR Confusion Matrix", fontsize=12, fontweight="bold")

# ── H. Key Stats Text ────────────────────────────────────────────────────────
ax_stats = fig.add_subplot(gs[2, 3])
ax_stats.axis("off")
lr_res = results["Logistic Regression"]
stats_text = (
    f"  LOGISTIC REGRESSION\n"
    f"  ─────────────────────\n"
    f"  Accuracy  :  {lr_res['accuracy']:.4f}\n"
    f"  F1-Score  :  {lr_res['f1']:.4f}\n"
    f"  Precision :  {lr_res['precision']:.4f}\n"
    f"  Recall    :  {lr_res['recall']:.4f}\n"
    f"  ROC-AUC   :  {lr_res['roc_auc']:.4f}\n"
    f"  CV Acc    :  {lr_res['cv_acc']:.4f}\n"
    f"  CV AUC    :  {lr_res['cv_auc']:.4f}\n"
    f"  ─────────────────────\n"
    f"  Dataset   :  {len(df):,} samples\n"
    f"  Biased    :  {(df.label==1).sum():,}\n"
    f"  Neutral   :  {(df.label==0).sum():,}\n"
)
ax_stats.text(0.05, 0.95, stats_text,
              transform=ax_stats.transAxes,
              fontsize=10, va="top", ha="left",
              fontfamily="monospace",
              color=TEXT_COLOR,
              bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=ACCENT, linewidth=1.5))

save_fig(fig, "00_summary_dashboard.png", dpi=120)

print("\n" + "="*65)
print("  ALL PLOTS SAVED TO: analysis/plots/")
print("="*65)

# Print final results table
print("\n  Final Model Comparison:")
print(f"  {'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}  {'CV_Acc':>8}  {'CV_AUC':>8}")
print("  " + "─"*65)
for name in model_names:
    r = results[name]
    star = " ★" if name == "Logistic Regression" else ""
    print(f"  {name:<25} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f}"
          f"  {r['cv_acc']:>8.4f}  {r['cv_auc']:>8.4f}{star}")
