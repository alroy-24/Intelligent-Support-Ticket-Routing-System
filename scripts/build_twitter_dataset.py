"""Build the labeled Twitter training dataset.

This is the script that fills the BUG / TECHNICAL / FEATURE_REQUEST gap that
Bitext can't cover (see src/ticketrouting/data/intent_mapping.py for why).

Pipeline:
  1. Download Customer Support on Twitter from Kaggle (skips if already on disk).
  2. Filter to inbound customer-originated tweets, anonymise mentions/URLs,
     dedupe, sample.
  3. Call an LLM to assign one of our 6 Categories per ticket. Disk-cached by
     sha256(model + rubric + text), so reruns are free.
  4. Save as parquet for downstream training.

Usage:
  # smoke test (100 rows, free on Groq):
  python scripts/build_twitter_dataset.py --limit 100 --out data/processed/twitter_smoke.parquet

  # full run (~20k, takes hours, will burn tokens):
  python scripts/build_twitter_dataset.py --limit 20000 --out data/processed/twitter_full.parquet

  # restrict to one brand for sanity-checking:
  python scripts/build_twitter_dataset.py --limit 500 --brand AppleSupport

The labeler is cached, so partial runs that crash mid-way resume for free.
"""
from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from ticketrouting.data.label import CategoryLabeler, make_client
from ticketrouting.data.loaders import save_processed
from ticketrouting.data.twitter import (
    download_twitter_dataset,
    find_twcs_csv,
    load_twitter,
)

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    limit: int = typer.Option(100, help="Number of tickets to sample and label."),
    out: Path = typer.Option(
        Path("data/processed/twitter_smoke.parquet"), help="Output parquet path."
    ),
    raw_dir: Path = typer.Option(
        Path("data/raw/twitter"), help="Where twcs.csv lives (or will be downloaded to)."
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/llm"), help="Where the LLM response cache lives."
    ),
    provider: str = typer.Option(
        "auto", help="LLM provider: auto | groq | anthropic."
    ),
    model: str | None = typer.Option(
        None, help="Override the provider's default model."
    ),
    brand: str | None = typer.Option(
        None, help="Restrict to one brand's mentions (e.g. AppleSupport)."
    ),
    download: bool = typer.Option(
        True, help="Auto-download via Kaggle CLI if twcs.csv isn't present."
    ),
    min_chars: int = typer.Option(
        20, help="Drop tweets shorter than this many characters."
    ),
    seed: int = typer.Option(42, help="Random seed for sampling."),
    skip_labeling: bool = typer.Option(
        False, help="Skip the LLM step (useful for sanity-checking the loader)."
    ),
):
    csv_path = find_twcs_csv(raw_dir)
    if csv_path is None:
        if not download:
            raise typer.Exit(
                f"No twcs.csv under {raw_dir} and --no-download set. "
                "Either drop the CSV there or rerun without --no-download."
            )
        console.rule("[bold]Step 0: Download from Kaggle")
        csv_path = download_twitter_dataset(raw_dir)
        console.print(f"[green]Downloaded to {csv_path}[/green]")
    else:
        console.print(f"Using existing {csv_path}")

    console.rule("[bold]Step 1: Load + filter Twitter")
    df = load_twitter(
        csv_path,
        limit=limit,
        min_chars=min_chars,
        brands=[brand] if brand else None,
        seed=seed,
    )
    console.print(f"Loaded {len(df):,} tickets.")
    if "brand" in df.columns:
        top_brands = df["brand"].value_counts().head(10).to_dict()
        console.print(f"Top brands in sample: {top_brands}")

    if skip_labeling:
        console.print("[yellow]Skipping LLM labeling (--skip-labeling).[/yellow]")
        save_processed(df, out)
        console.print(f"[green]Wrote {out}[/green]")
        return

    client = make_client(provider=provider, model=model)
    console.rule(f"[bold]Step 2: Label category via {type(client).__name__} ({client.model})")
    labeler = CategoryLabeler(client=client, cache_dir=cache_dir)

    categories: list[str] = []
    reasonings: list[str] = []
    failures = 0

    for text in tqdm(df["text"].tolist(), desc="labeling"):
        try:
            label = labeler.label(text)
            categories.append(label.category.value)
            reasonings.append(label.reasoning)
        except Exception as e:  # noqa: BLE001 — keep going on individual failures
            failures += 1
            categories.append("other")
            reasonings.append(f"FAILED: {e}")

    df["category"] = categories
    df["category_reasoning"] = reasonings

    console.rule("[bold]Step 3: Save")
    save_processed(df, out)
    console.print(f"[green]Wrote {out}[/green] ({len(df):,} rows, {failures} label failures)")
    console.print("Category distribution:")
    console.print(df["category"].value_counts().to_dict())


if __name__ == "__main__":
    app()
