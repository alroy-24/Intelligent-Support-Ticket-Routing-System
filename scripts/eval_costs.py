"""Run the business-weighted cost analysis on a trained category model.

Compares four strategies on the same labeled dataset:
  1. Argmax — the model's raw highest-probability prediction.
  2. Cost-tuned — same model, but the decision rule is argmin(expected cost)
     using the cost matrix. Should never be WORSE than argmax on cost.
  3. Majority class — always predict the most-common label. Strong baseline
     because Bitext is skewed billing-heavy.
  4. Random routing — uniform random across all categories. The "do nothing"
     floor.

Usage:
  python scripts/eval_costs.py \
      --model artifacts/baseline_category.joblib \
      --data data/processed/twitter_smoke.parquet

The output table reports cost per 1000 tickets, accuracy, and percentage cost
reduction vs random routing. That's the headline number for the README:
"the model saves X% of triage-minutes vs naive routing."
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ticketrouting.eval.cost import (
    compare_strategies,
    default_category_costs,
    majority_predictions,
    predict_min_cost,
    random_predictions,
)
from ticketrouting.models.category.baseline import load as load_pipe

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model: Path = typer.Option(
        Path("artifacts/baseline_category.joblib"),
        help="Joblib of the fitted category pipeline.",
    ),
    data: Path = typer.Option(
        ...,
        help="Parquet with `text` and `category` columns.",
    ),
    text_col: str = typer.Option("text"),
    label_col: str = typer.Option("category"),
    seed: int = typer.Option(42, help="Random seed for the random-routing baseline."),
):
    if not model.exists():
        raise typer.Exit(f"Model {model} not found.")
    if not data.exists():
        raise typer.Exit(f"Data {data} not found.")

    console.rule("[bold]Step 1: Load model + data")
    pipe = load_pipe(model)
    df = pd.read_parquet(data)
    if label_col not in df.columns:
        raise typer.Exit(f"{data} is missing label column {label_col!r}.")

    X = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()
    console.print(f"Loaded {len(df):,} rows. Label distribution: {dict(pd.Series(y_true).value_counts())}")

    console.rule("[bold]Step 2: Generate predictions")
    proba = pipe.predict_proba(X)
    classes = list(pipe.classes_)

    argmax_preds = [classes[int(i)] for i in proba.argmax(axis=1)]

    costs = default_category_costs()
    min_cost_preds = predict_min_cost(proba, classes, costs).tolist()

    majority_preds = majority_predictions(y_true)
    random_preds = random_predictions(y_true, labels=costs.labels, seed=seed)

    console.rule("[bold]Step 3: Score each strategy")
    reports = compare_strategies(
        y_true=y_true,
        predictions={
            "Random routing": random_preds,
            "Majority class": majority_preds,
            "Model (argmax)": argmax_preds,
            "Model (cost-tuned)": min_cost_preds,
        },
        cost_matrix=costs,
    )

    baseline = next(r for r in reports if r.name == "Random routing").cost_per_1000

    table = Table(title=f"Cost analysis ({costs.unit} per 1000 tickets)")
    table.add_column("Strategy")
    table.add_column("Cost / 1000", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("vs Random", justify="right")
    for r in reports:
        reduction = 1 - (r.cost_per_1000 / baseline) if baseline > 0 else 0.0
        table.add_row(
            r.name,
            f"{r.cost_per_1000:,.1f}",
            f"{r.accuracy:.2%}",
            f"{reduction:+.1%}",
        )
    console.print(table)

    # Highlight headline
    model_argmax = next(r for r in reports if r.name == "Model (argmax)")
    saved_pct = 1 - (model_argmax.cost_per_1000 / baseline) if baseline > 0 else 0.0
    saved_abs = baseline - model_argmax.cost_per_1000
    console.rule()
    console.print(
        f"[bold green]Headline:[/bold green] model saves "
        f"[bold]{saved_pct:.1%}[/bold] of triage time vs random routing "
        f"([bold]{saved_abs:,.0f}[/bold] {costs.unit} per 1000 tickets)."
    )


if __name__ == "__main__":
    app()
