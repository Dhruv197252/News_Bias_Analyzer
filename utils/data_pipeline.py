import pandas as pd
from datasets import load_dataset


# ── 1. Load from Hugging Face ─────────────────────────────────────────────────

def load_babe_dataset() -> pd.DataFrame:
    """
    Downloads BABE from HuggingFace Hub and returns a clean DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        text  : str   — the news sentence
        label : int   — 1 (Biased) or 0 (Neutral)
    """
    print("📥 Downloading BABE dataset from Hugging Face...")
    dataset = load_dataset("mediabiasgroup/BABE", trust_remote_code=True)
    print("✅ Download complete.\n")

    # BABE only has a 'train' split — convert to Pandas
    df = dataset["train"].to_pandas()

    print(f"Raw dataset shape  : {df.shape}")
    print(f"Columns available  : {list(df.columns)}\n")

    return df


# ── 2. Inspect & Understand the Schema ───────────────────────────────────────

def inspect_dataset(df: pd.DataFrame) -> None:
    """Print a quick summary so we understand what we're working with."""
    print("── Label Distribution (raw) ─────────────────────────")
    print(df["label"].value_counts())
    print()
    print("── Sample Rows ──────────────────────────────────────")
    print(df[["text", "label"]].head(5).to_string(index=False))
    print()


# ── 3. Clean & Binarize ───────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updated for BABE's actual schema:
      - Labels are already integers: 1 (Biased), 0 (Neutral)
      - Just drop missing text and reset index
    """
    print("🧹 Cleaning dataset...")

    # Step 1 — Drop missing/empty text
    before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    print(f"  Dropped {before - len(df)} rows with missing text")

    # Step 2 — Keep only valid binary labels (0 or 1)
    before = len(df)
    df = df[df["label"].isin([0, 1])].copy()
    print(f"  Dropped {before - len(df)} rows with ambiguous labels")

    # Step 3 — Ensure label is integer type
    df["label"] = df["label"].astype(int)

    # Step 4 — Keep only the two columns we need
    df = df[["text", "label"]].reset_index(drop=True)

    print(f"\n✅ Clean dataset shape : {df.shape}")
    print(f"   Biased   (1) : {(df['label'] == 1).sum()}")
    print(f"   Neutral  (0) : {(df['label'] == 0).sum()}")
    print()

    return df

# ── 4. Save to Disk ───────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, path: str = "data/babe_clean.csv") -> None:
    """Save the cleaned DataFrame as CSV so we don't re-download every run."""
    df.to_csv(path, index=False)
    print(f"💾 Saved clean dataset to '{path}'")


# ── 5. Master Function ────────────────────────────────────────────────────────

def build_dataset(save: bool = True) -> pd.DataFrame:
    """
    Full pipeline:  Download → Inspect → Clean → (Save) → Return

    Parameters
    ----------
    save : bool
        If True, writes 'data/babe_clean.csv' to disk.

    Returns
    -------
    Clean pd.DataFrame ready for ML training.
    """
    df_raw   = load_babe_dataset()
    inspect_dataset(df_raw)
    df_clean = clean_dataset(df_raw)

    if save:
        save_dataset(df_clean)

    return df_clean


# ── 6. Quick Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_dataset(save=True)
    print("\n── Final Preview ─────────────────────────────────────")
    print(df.sample(5, random_state=42).to_string(index=False))