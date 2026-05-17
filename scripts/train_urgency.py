"""Train the TF-IDF + Ordinal LogReg urgency baseline.

Input: a parquet with `text` and `urgency` columns. Produce one by running
`scripts/build_dataset.py --limit 1000 --out data/processed/labeled_urgency.parquet`
first (Bitext text + LLM-assigned urgency labels).

Usage:
  python scripts/train_urgency.py --data data/processed/labeled_urgency.parquet
  python scripts/train_urgency.py --data ... --out artifacts/urgency_baseline.joblib

What it prints:
  - MAE on integer ranks (the metric that actually matters here).
  - Off-by->=2 rate (share of predictions catastrophically wrong on the ordinal axis).
  - Macro / weighted F1 for cross-comparison with the category baseline.
  - Per-class F1 and the confusion matrix as a rich table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ticketrouting.models.urgency.baseline import (
    URGENCY_ORDER,
    predict_one,
    save,
    train_test_eval,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    data: Path = typer.Option(
        Path("data/processed/labeled_urgency.parquet"),
        help="Labeled parquet with `text` and `urgency` columns.",
    ),
    out: Path = typer.Option(
        Path("artifacts/urgency_baseline.joblib"),
        help="Where to save the fitted pipeline.",
    ),
    test_size: float = typer.Option(0.2, help="Test split fraction."),
    seed: int = typer.Option(42, help="Random seed for the split."),
    show_samples: int = typer.Option(
        5, help="Print this many sample predictions after training."
    ),
):
    if not data.exists():
        raise typer.Exit(
            f"{data} not found. Run scripts/build_dataset.py first to produce an "
            "urgency-labeled parquet."
        )

    console.rule("[bold]Step 1: Load")
    df = pd.read_parquet(data)
    if "urgency" not in df.columns:
        raise typer.Exit(
            f"{data} is missing an `urgency` column — was build_dataset.py run "
            "with --skip-labeling?"
        )
    console.print(f"Loaded {len(df):,} rows. Urgency distribution:")
    console.print(df["urgency"].value_counts().to_dict())

    console.rule("[bold]Step 2: Train + evaluate")
    pipe, report = train_test_eval(df, test_size=test_size, random_state=seed)

    console.rule("[bold]Step 3: Report")
    console.print(report.pretty())

    table = Table(title="Confusion matrix (rows = true, cols = predicted)")
    table.add_column("true \\ pred")
    for label in URGENCY_ORDER:
        table.add_column(label)
    for i, true_label in enumerate(report.labels):
        row = [true_label] + [str(int(report.confusion[i, j])) for j in range(len(report.labels))]
        table.add_row(*row)
    console.print(table)

    console.rule("[bold]Step 4: Save")
    save(pipe, out)
    console.print(f"[green]Saved pipeline to {out}[/green]")

    if show_samples:
        console.rule("[bold]Sanity-check predictions")
        examples = [
            "The whole site is down, none of our team can log in",
            "I was charged twice for my subscription this month, please refund",
            "When does my free trial end?",
            "Could you add a dark mode to the dashboard?",
            "Security alert: someone tried to log into my account from another country",
        ]
        for text in examples[:show_samples]:
            urg, conf = predict_one(pipe, text)
            console.print(f"[cyan]{conf:.2f}[/cyan] [bold]{urg.value:9s}[/bold]  {text}")


if __name__ == "__main__":
    app()
