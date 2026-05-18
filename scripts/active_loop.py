"""Active-learning CLI: surface low-confidence predictions, merge corrections.

Two subcommands:

  surface  — read the prediction log, dump the N most uncertain predictions to
             a JSONL review queue the human edits in place.
  merge    — read the (now-edited) review queue and append accepted corrections
             to data/processed/reviewed.parquet, which the next training run
             can load alongside Bitext / Twitter.

Workflow:

  # 1. Run the API with active logging enabled
  $env:ACTIVE_LOG_PATH = "data/processed/predictions.jsonl"
  uvicorn ticketrouting.api.app:app

  # 2. After some traffic, surface the uncertain ones
  python scripts/active_loop.py surface \
      --log data/processed/predictions.jsonl \
      --out data/processed/review_queue.jsonl \
      --threshold 0.6 --max 50

  # 3. Open review_queue.jsonl, fill in "human_label" on each row.
  #    Use one of: billing, technical, account, bug, feature_request, other, or "discard".

  # 4. Merge corrections back into the training corpus.
  python scripts/active_loop.py merge \
      --reviews data/processed/review_queue.jsonl \
      --out data/processed/reviewed.parquet
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ticketrouting.active.loop import (
    PredictionLog,
    apply_corrections,
    surface_low_confidence,
    write_review_queue,
)
from ticketrouting.schemas import Category

app = typer.Typer(add_completion=False)
console = Console()

VALID_CATEGORIES = {c.value for c in Category}


@app.command()
def surface(
    log: Path = typer.Option(
        Path("data/processed/predictions.jsonl"), help="Prediction log (JSONL)."
    ),
    out: Path = typer.Option(
        Path("data/processed/review_queue.jsonl"),
        help="Where to write the review queue.",
    ),
    threshold: float = typer.Option(0.6, help="Surface rows with route_confidence below this."),
    max_rows: int = typer.Option(50, "--max", help="Cap the queue at N rows."),
):
    """Pull the most uncertain predictions out of the log for human review."""
    if not log.exists():
        raise typer.Exit(f"Log {log} does not exist — nothing to surface.")

    pred_log = PredictionLog(log)
    total = len(pred_log)
    rows = surface_low_confidence(pred_log, threshold=threshold, max_n=max_rows)

    console.rule("[bold]Surface low-confidence predictions")
    console.print(f"Log {log}: {total:,} predictions total.")
    console.print(f"Below threshold ({threshold}): up to {len(rows)} surfaced.")

    if not rows:
        console.print("[yellow]Nothing below the threshold. No queue written.[/yellow]")
        return

    write_review_queue(rows, out)
    console.print(f"[green]Wrote {len(rows)} rows to {out}[/green]")

    # Show a preview
    preview = Table(title=f"First {min(10, len(rows))} (lowest confidence)")
    preview.add_column("conf", justify="right")
    preview.add_column("predicted")
    preview.add_column("text", overflow="fold")
    for r in rows[:10]:
        preview.add_row(f"{r.confidence:.2f}", r.predicted_category, r.text[:120])
    console.print(preview)
    console.print(
        f"\n[bold]Next:[/bold] edit {out} — fill in [cyan]human_label[/cyan] on each row "
        f"with one of: {sorted(VALID_CATEGORIES)} or [yellow]\"discard\"[/yellow]."
    )


@app.command()
def merge(
    reviews: Path = typer.Option(
        Path("data/processed/review_queue.jsonl"),
        help="The (now-edited) review JSONL.",
    ),
    out: Path = typer.Option(
        Path("data/processed/reviewed.parquet"),
        help="Where to append accepted corrections.",
    ),
):
    """Merge human-edited corrections into the reviewed training set."""
    if not reviews.exists():
        raise typer.Exit(f"Reviews file {reviews} not found.")

    console.rule("[bold]Merge human corrections")
    n = apply_corrections(reviews, out, valid_categories=VALID_CATEGORIES)

    if n == 0:
        console.print(
            "[yellow]No new rows accepted. Either no human_label was filled in, "
            "all were 'discard', or all were already in the parquet.[/yellow]"
        )
    else:
        console.print(f"[green]Appended {n} corrected rows to {out}[/green]")
        console.print(
            "These rows will be picked up by the next training run when it reads "
            f"{out} alongside the Bitext / Twitter parquets."
        )


if __name__ == "__main__":
    app()
