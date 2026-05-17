"""Tests for the Twitter loader. No network — we synthesize twcs.csv on disk."""
from __future__ import annotations

import pandas as pd
import pytest

from ticketrouting.data.twitter import _anonymise, load_twitter


@pytest.fixture
def twcs_csv(tmp_path):
    """A tiny synthetic twcs.csv covering the filter rules we care about."""
    rows = [
        # 1: clean inbound original — keep, anonymise.
        (1, "user_a", True, "2023-01-01", "@AppleSupport my iPhone screen is cracked", "", None),
        # 2: outbound brand reply — drop (inbound=False).
        (2, "AppleSupport", False, "2023-01-01", "@user_a we're sorry, please DM us", "", 1.0),
        # 3: mid-thread reply from a user — drop (in_response_to_tweet_id set).
        (3, "user_a", True, "2023-01-01", "@AppleSupport still broken", "", 2.0),
        # 4: too short — drop (under min_chars).
        (4, "user_b", True, "2023-01-01", "@AmazonHelp wtf", "", None),
        # 5: URL + mention; long enough — keep, anonymise both.
        (
            5,
            "user_c",
            True,
            "2023-01-02",
            "@SpotifyCares my playlists vanished see https://t.co/abc help please",
            "",
            None,
        ),
        # 6: exact dupe of 5 after anonymisation — drop on dedupe.
        (
            6,
            "user_d",
            True,
            "2023-01-02",
            "@SpotifyCares my playlists vanished see https://t.co/xyz help please",
            "",
            None,
        ),
        # 7: clean inbound to Amazon — keep.
        (7, "user_e", True, "2023-01-03", "@AmazonHelp my package never arrived this is awful", "", None),
        # 8: missing text — drop.
        (8, "user_f", True, "2023-01-03", None, "", None),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )
    path = tmp_path / "twcs.csv"
    df.to_csv(path, index=False)
    return path


def test_anonymise_replaces_handles_and_urls():
    raw = "@AppleSupport my screen broke see https://t.co/abc thanks"
    assert _anonymise(raw) == "@brand my screen broke see <url> thanks"


def test_anonymise_collapses_whitespace():
    raw = "@AppleSupport   help   me\n\nplease"
    assert _anonymise(raw) == "@brand help me please"


def test_load_twitter_applies_all_filters(twcs_csv):
    df = load_twitter(twcs_csv, min_chars=20)

    # Surviving rows: 1, 5, 7. Row 6 is a post-anonymisation dupe of 5.
    assert sorted(df["tweet_id"].tolist()) == [1, 5, 7]


def test_load_twitter_anonymises_text(twcs_csv):
    df = load_twitter(twcs_csv, min_chars=20)

    for text in df["text"]:
        assert "https://" not in text, "URLs should be replaced"
        # Mentions are anonymised to literal "@brand" — original handles must be gone.
        assert "@AppleSupport" not in text
        assert "@AmazonHelp" not in text
        assert "@SpotifyCares" not in text


def test_load_twitter_preserves_raw_text_for_provenance(twcs_csv):
    df = load_twitter(twcs_csv, min_chars=20)

    # raw_text is what we'd show a human reviewer; text is what the model sees.
    row1 = df[df["tweet_id"] == 1].iloc[0]
    assert "@AppleSupport" in row1["raw_text"]
    assert "@brand" in row1["text"]


def test_load_twitter_extracts_brand_from_first_mention(twcs_csv):
    df = load_twitter(twcs_csv, min_chars=20)

    brand_by_id = dict(zip(df["tweet_id"], df["brand"]))
    assert brand_by_id[1] == "AppleSupport"
    assert brand_by_id[5] == "SpotifyCares"
    assert brand_by_id[7] == "AmazonHelp"


def test_load_twitter_filters_by_brand(twcs_csv):
    df = load_twitter(twcs_csv, min_chars=20, brands=["AmazonHelp"])

    assert df["tweet_id"].tolist() == [7]


def test_load_twitter_limit_caps_rows(twcs_csv):
    df = load_twitter(twcs_csv, min_chars=20, limit=2, seed=0)

    assert len(df) == 2


def test_load_twitter_min_chars_is_respected(twcs_csv):
    # min_chars=200 should reject everything in the fixture.
    df = load_twitter(twcs_csv, min_chars=200)

    assert len(df) == 0
