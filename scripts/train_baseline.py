"""Train the TF-IDF + LogReg baseline category classifier on Bitext.

Usage:
  python scripts/train_baseline.py
  python scripts/train_baseline.py --limit 5000 --out artifacts/baseline.joblib

The script will:
  1. Load Bitext + map intents to our Category taxonomy.
  2. Stratified split (80/20).
  3. Train TF-IDF + LogReg with balanced class weights.
  4. Print macro/weighted F1, per-class F1, classification report, confusion matrix.
  5. Save the fitted pipeline as a joblib file.
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ticketrouting.data.loaders import load_bitext
from ticketrouting.models.category.baseline import predict_one, save, train_test_eval

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    limit: int | None = typer.Option(
        None, help="Cap the number of rows (None = full dataset, ~27k)."
    ),
    out: Path = typer.Option(
        Path("artifacts/baseline_category.joblib"),
        help="Where to save the fitted pipeline.",
    ),
    test_size: float = typer.Option(0.2, help="Test split fraction."),
    seed: int = typer.Option(42, help="Random seed for the split."),
    show_samples: int = typer.Option(
        5, help="Print this many sample predictions after training."
    ),
):
    console.rule("[bold]Step 1: Load data")
    df = load_bitext(limit=limit)
    console.print(f"Loaded {len(df):,} rows. Categories: {df['category'].value_counts().to_dict()}")

    console.rule("[bold]Step 2: Train + evaluate")
    pipe, report = train_test_eval(df, test_size=test_size, random_state=seed)

    console.rule("[bold]Step 3: Report")
    console.print(report.pretty())

    # Confusion matrix as a rich table
    table = Table(title="Confusion matrix (rows = true, cols = predicted)")
    table.add_column("true \\ pred")
    for label in report.labels:
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
            "My credit card was charged twice this month",
            "I can't log into my account, password reset isn't working",
            "How do I update my shipping address?",
            "The app crashes when I try to open the dashboard",
            "Could you add dark mode to the settings page?",
        ]
        for text in examples[:show_samples]:
            cat, conf = predict_one(pipe, text)
            console.print(f"[cyan]{conf:.2f}[/cyan] [bold]{cat.value:12s}[/bold]  {text}")


if __name__ == "__main__":
    app()
