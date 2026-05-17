"""Loader for the Customer Support on Twitter dataset.

Dataset: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
~3M tweets between customers and major brands (Apple, Amazon, Uber, airlines, etc.).

Why Twitter on top of Bitext:
- Bitext is templated and only honestly covers BILLING / ACCOUNT / OTHER.
- Twitter is real customer text and contains the BUG / TECHNICAL / FEATURE_REQUEST
  examples the routing system needs to learn.
- It is also messier: emoji, mentions, URLs, sarcasm, multilingual. That noise is
  the whole point — the production system will see exactly this.

Filtering rules applied here:
- `inbound == True` so we only keep customer-sent tweets, not brand replies.
- `in_response_to_tweet_id` is null so we get the customer's ORIGINAL message rather
  than a mid-thread "thanks" or "still broken". This is what arrives in a real
  ticketing system.
- Minimum text length so we drop one-word replies that carry no routing signal.
- Anonymise `@handle` and URLs so the classifier learns from the complaint text,
  not from which brand was mentioned (which would be a trivial shortcut).
- Deduplicate by anonymised text so identical template complaints don't dominate
  the sample.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
TWCS_FILENAME = "twcs.csv"

_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def find_twcs_csv(root: Path) -> Path | None:
    """Locate twcs.csv anywhere under `root`. Returns None if not found.

    Kaggle's zip puts the file in a `twcs/` subdir; older mirrors put it at the
    root. Scan rather than hardcode.
    """
    matches = list(root.rglob(TWCS_FILENAME))
    # Prefer the largest file — there's also a tiny sample.csv but a real
    # `twcs.csv` should be hundreds of MB.
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_size)


def download_twitter_dataset(target_dir: Path) -> Path:
    """Download + unzip the Twitter customer-support dataset via the Kaggle CLI.

    Returns the path to twcs.csv (wherever in the extracted tree it landed).
    If the file already exists, this is a no-op.
    Requires Kaggle API credentials at ~/.kaggle/kaggle.json or ~/.kaggle/access_token.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = find_twcs_csv(target_dir)
    if existing is not None:
        return existing

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(target_dir),
            "--unzip",
        ],
        check=True,
    )

    found = find_twcs_csv(target_dir)
    if found is None:
        raise FileNotFoundError(
            f"Kaggle download finished but no {TWCS_FILENAME} under {target_dir}. "
            f"Contents: {[p.name for p in target_dir.iterdir()]}"
        )
    return found


def _anonymise(text: str) -> str:
    text = _MENTION_RE.sub("@brand", text)
    text = _URL_RE.sub("<url>", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_twitter(
    csv_path: Path,
    limit: int | None = None,
    min_chars: int = 20,
    brands: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load Twitter customer-support tweets into a clean DataFrame.

    Returns columns: tweet_id, text, raw_text, created_at, brand.

    `brands` (optional) restricts to tweets where the original company handle
    (the first @mention in the raw text) is in the given list — useful for
    sanity-checking on a single brand before running the full sample.
    """
    df = pd.read_csv(csv_path)

    df = df[df["inbound"] == True]  # noqa: E712 — pandas mask, `is True` would not vectorise
    df = df[df["in_response_to_tweet_id"].isna()]

    df = df.dropna(subset=["text"])
    df["raw_text"] = df["text"].astype(str)
    df["brand"] = df["raw_text"].str.extract(r"@([A-Za-z0-9_]+)", expand=False)
    df["text"] = df["raw_text"].map(_anonymise)

    df = df[df["text"].str.len() >= min_chars]

    if brands:
        df = df[df["brand"].isin(brands)]

    df = df.drop_duplicates(subset=["text"])

    df = df[["tweet_id", "text", "raw_text", "created_at", "brand"]].reset_index(drop=True)

    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    return df
