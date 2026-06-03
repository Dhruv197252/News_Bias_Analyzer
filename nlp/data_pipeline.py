import pandas as pd
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def load_babe_dataset() -> pd.DataFrame:
    """Downloads BABE from HuggingFace and returns a clean DataFrame."""
    if load_dataset is None:
        raise ImportError("Install datasets: pip install datasets")
    dataset = load_dataset("mediabiasgroup/BABE", trust_remote_code=True)
    return dataset["train"].to_pandas()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing text, keep binary labels 0/1, return [text, label]."""
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].reset_index(drop=True)


def save_dataset(df: pd.DataFrame, path: str = "data/raw/babe_clean.csv") -> None:
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to '{path}'")


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Full pipeline: Download → Clean → (Save) → Return."""
    df_raw   = load_babe_dataset()
    df_clean = clean_dataset(df_raw)
    if save:
        save_dataset(df_clean)
    return df_clean


if __name__ == "__main__":
    df = build_dataset(save=True)
    print(df.sample(5, random_state=42).to_string(index=False))
