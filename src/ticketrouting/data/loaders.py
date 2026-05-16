"""Dataset loaders. Bitext is the primary clean source; Twitter is the noisy realistic one."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ticketrouting.data.intent_mapping import map_intent

BITEXT_HF_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def load_bitext(cache_dir: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load the Bitext dataset and map intents to our Category taxonomy.

    Returns a DataFrame with columns: text, intent, category.
    """
    from datasets import load_dataset

    ds = load_dataset(
        BITEXT_HF_ID,
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    df = ds.to_pandas()

    text_col = "instruction" if "instruction" in df.columns else "utterance"
    intent_col = "intent"

    df = df[[text_col, intent_col]].rename(columns={text_col: "text"})
    df["category"] = df[intent_col].map(lambda i: map_intent(i).value)

    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)

    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
