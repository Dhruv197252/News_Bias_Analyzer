"""
Training Data Collector
-----------------------
Builds the merged training corpus from:
  1. BABE dataset (already in data/babe_clean.csv) — sentence-level labels
  2. AllSides headlines (HuggingFace dataset) — Left/Right=1, Center=0
  3. Outputs: data/merged_corpus.csv
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BABE_PATH   = os.path.join(ROOT, "data", "babe_clean.csv")
OUTPUT_PATH = os.path.join(ROOT, "data", "merged_corpus.csv")


# ── Label Mapping ─────────────────────────────────────────────────────────────

# AllSides uses: Left, Lean Left, Center, Lean Right, Right
# Binary mapping: Left/Right leaning → biased=1, Center → biased=0
ALLSIDES_LABEL_MAP = {
    "left":        1,
    "lean left":   1,
    "center":      0,
    "lean right":  1,
    "right":       1,
}


# ── 1. Load BABE ─────────────────────────────────────────────────────────────

def load_babe(path: str = BABE_PATH) -> pd.DataFrame:
    """Load BABE dataset. Expected columns: text, label (0 or 1)."""
    logger.info(f"Loading BABE from {path}...")
    df = pd.read_csv(path)

    # Ensure correct columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("BABE CSV must have 'text' and 'label' columns")

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["source"] = "babe"
    logger.info(f"  BABE: {len(df)} rows (biased={df['label'].sum()}, neutral={len(df)-df['label'].sum()})")
    return df


# ── 2. Load AllSides from HuggingFace ────────────────────────────────────────

def load_allsides() -> pd.DataFrame:
    """
    Load AllSides dataset from HuggingFace.
    Maps Left/Right → 1 (biased), Center → 0 (neutral).
    Falls back to empty DataFrame if network unavailable.
    """
    logger.info("Loading AllSides dataset from HuggingFace...")
    try:
        from datasets import load_dataset  # type: ignore

        # Try multiple known AllSides HuggingFace datasets
        candidates = [
            ("valurank/PoliticalBias_AllSides_Txt", "train"),
        ]

        for dataset_name, split in candidates:
            try:
                logger.info(f"  Trying {dataset_name}...")
                ds = load_dataset(dataset_name, split=split)
                df = ds.to_pandas()
                logger.info(f"  Loaded {len(df)} rows from {dataset_name}")
                logger.info(f"  Columns: {list(df.columns)}")

                # Find text and label columns (different datasets use different names)
                text_col  = _find_column(df, ["text", "content", "headline", "title", "article"])
                label_col = _find_column(df, ["bias_label", "label", "bias", "leaning", "political_leaning"])

                if text_col and label_col:
                    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
                    df = df.dropna()

                    # Map labels to binary
                    df["label"] = df["raw_label"].apply(_map_allsides_label)
                    df = df[df["label"] != -1]  # drop unknown labels
                    df = df[["text", "label"]]
                    df["source"] = "allsides"

                    # Filter to reasonable text length
                    df = df[df["text"].str.split().str.len().between(5, 500)]

                    logger.info(f"  AllSides: {len(df)} rows after mapping")
                    logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
                    return df

            except Exception as e:
                logger.warning(f"  {dataset_name} failed: {e}")
                continue

        logger.warning("AllSides load failed from all candidates. Skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    except ImportError:
        logger.warning("'datasets' package not installed. Run: pip install datasets")
        return pd.DataFrame(columns=["text", "label", "source"])

    except Exception as e:
        logger.warning(f"AllSides load failed: {e}")
        return pd.DataFrame(columns=["text", "label", "source"])


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


def _map_allsides_label(raw_label) -> int:
    """Map AllSides label string to binary int. Returns -1 for unknown."""
    if pd.isna(raw_label):
        return -1
    label_str = str(raw_label).lower().strip()
    return ALLSIDES_LABEL_MAP.get(label_str, -1)


# ── 3. Merge + Clean ─────────────────────────────────────────────────────────

def build_corpus(babe: pd.DataFrame, allsides: pd.DataFrame) -> pd.DataFrame:
    """
    Merge datasets, deduplicate, and balance classes.
    """
    frames = [babe]
    if len(allsides) > 0:
        frames.append(allsides)

    merged = pd.concat(frames, ignore_index=True)

    # Clean text
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged[merged["text"].str.split().str.len() >= 5]

    # Deduplicate on text (case-insensitive)
    merged["text_lower"] = merged["text"].str.lower()
    merged = merged.drop_duplicates(subset="text_lower")
    merged = merged.drop(columns=["text_lower"])

    # Ensure label is int
    merged["label"] = merged["label"].astype(int)

    logger.info(f"\n── Merged Corpus Summary ─────────────────────")
    logger.info(f"  Total rows  : {len(merged)}")
    logger.info(f"  Biased  (1) : {(merged['label']==1).sum()}")
    logger.info(f"  Neutral (0) : {(merged['label']==0).sum()}")
    if "source" in merged.columns:
        logger.info(f"  By source   : {merged['source'].value_counts().to_dict()}")

    return merged[["text", "label"]]


# ── 4. Save ───────────────────────────────────────────────────────────────────

def save_corpus(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"\n✅ Saved merged corpus to: {path}")
    logger.info(f"   {len(df)} total rows")


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    babe     = load_babe()
    allsides = load_allsides()
    corpus   = build_corpus(babe, allsides)
    save_corpus(corpus)
    return corpus


if __name__ == "__main__":
    main()
