"""Build the labeled training dataset.

Pipeline:
  1. Load Bitext (already has intent labels, no urgency).
  2. Map intents -> our Category taxonomy.
  3. Call Claude to assign urgency labels (cached on disk by sha256).
  4. Save as parquet for downstream training.

Usage:
  python scripts/build_dataset.py --limit 200 --out data/processed/labeled.parquet
  python scripts/build_dataset.py --limit 20000 --out data/processed/labeled_full.parquet

The labeler is cached, so re-running the same sample after a failure is free.
"""
from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from ticketrouting.data.label import UrgencyLabeler, make_client
from ticketrouting.data.loaders import load_bitext, save_processed

load_dotenv()  # Pull ANTHROPIC_API_KEY etc. from .env if present.

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    limit: int = typer.Option(200, help="Number of tickets to sample and label."),
    out: Path = typer.Option(
        Path("data/processed/labeled.parquet"),
        help="Output parquet path.",
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/llm"), help="Where the LLM response cache lives."
    ),
    provider: str = typer.Option(
        "auto", help="LLM provider: auto | groq | anthropic. 'auto' picks based on which key is set."
    ),
    model: str | None = typer.Option(
        None, help="Override the provider's default model (e.g. llama-3.3-70b-versatile)."
    ),
    skip_labeling: bool = typer.Option(
        False, help="Skip the LLM step (useful for sanity-checking the loader)."
    ),
):
    console.rule("[bold]Step 1: Load Bitext + map intents")
    df = load_bitext(limit=limit)
    console.print(f"Loaded {len(df):,} rows. Category distribution:")
    console.print(df["category"].value_counts().to_dict())

    if skip_labeling:
        console.print("[yellow]Skipping LLM labeling (--skip-labeling).[/yellow]")
        save_processed(df, out)
        console.print(f"[green]Wrote {out}[/green]")
        return

    client = make_client(provider=provider, model=model)
    console.rule(f"[bold]Step 2: Label urgency via {type(client).__name__} ({client.model})")
    labeler = UrgencyLabeler(client=client, cache_dir=cache_dir)

    urgencies: list[str] = []
    reasonings: list[str] = []
    failures = 0

    for text in tqdm(df["text"].tolist(), desc="labeling"):
        try:
            label = labeler.label(text)
            urgencies.append(label.urgency.value)
            reasonings.append(label.reasoning)
        except Exception as e:  # noqa: BLE001 — we want to keep going on individual failures
            failures += 1
            urgencies.append("low")
            reasonings.append(f"FAILED: {e}")

    df["urgency"] = urgencies
    df["urgency_reasoning"] = reasonings

    console.rule("[bold]Step 3: Save")
    save_processed(df, out)
    console.print(f"[green]Wrote {out}[/green] ({len(df):,} rows, {failures} label failures)")
    console.print("Urgency distribution:")
    console.print(df["urgency"].value_counts().to_dict())


if __name__ == "__main__":
    app()
