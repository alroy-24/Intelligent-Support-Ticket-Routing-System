"""Cron-friendly drift check on the PredictionLog.

What it does:
  1. Reads the PredictionLog (default: data/processed/predictions.jsonl).
  2. Splits into a reference window (e.g. 7 days ago vs the prior 7 days)
     and a recent window. Falls back to a 50/50 row-split if the log doesn't
     span the requested time windows yet.
  3. Computes a DriftReport: PSI on category distribution, per-category
     shifts, mean-confidence delta.
  4. Prints the report.
  5. Optionally appends the report to a JSONL history file (`--history`).
  6. Exits 0 (no drift), 1 (moderate), or 2 (significant), so a cron job
     can branch on the exit code: alert at 2, log at 1, ignore at 0.

Run weekly:
  python scripts/check_drift.py \
      --log data/processed/predictions.jsonl \
      --ref-days 7 --recent-days 7 \
      --history data/processed/drift_history.jsonl
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from ticketrouting.active.loop import PredictionLog
from ticketrouting.drift.distribution import (
    compute_drift_report,
    split_log_by_time,
    split_log_in_half,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    log: Path = typer.Option(
        Path("data/processed/predictions.jsonl"),
        help="Prediction log (JSONL) produced by the API.",
    ),
    ref_days: int = typer.Option(7, help="Reference window length, in days."),
    recent_days: int = typer.Option(7, help="Recent window length, in days."),
    force_split: bool = typer.Option(
        False,
        help="Skip time-based windowing and split the log 50/50 by row order. "
             "Useful for fresh logs or tests.",
    ),
    history: Path | None = typer.Option(
        None,
        help="Append the report (as JSON) to this file for trend tracking.",
    ),
):
    if not log.exists():
        raise typer.Exit(f"Log {log} not found.")

    pred_log = PredictionLog(log)
    rows = list(pred_log)
    if not rows:
        raise typer.Exit("Log is empty — nothing to check.")

    if force_split:
        reference, recent = split_log_in_half(rows)
        windowing = "row-50/50"
    else:
        reference, recent = split_log_by_time(rows, ref_days, recent_days)
        windowing = f"time ({ref_days}d ref / {recent_days}d recent)"
        if not reference or not recent:
            console.print(
                f"[yellow]Time windowing produced ref={len(reference)} "
                f"recent={len(recent)}; falling back to 50/50 row split.[/yellow]"
            )
            reference, recent = split_log_in_half(rows)
            windowing = "row-50/50 (fallback)"

    if not reference or not recent:
        raise typer.Exit("Both windows are empty — log too small to compute drift.")

    report = compute_drift_report(reference, recent)

    console.rule(f"[bold]Drift report  (windowing: {windowing})")
    console.print(report.pretty())

    if history is not None:
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "checked_at": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "windowing": windowing,
                        **report.to_dict(),
                    }
                )
                + "\n"
            )
        console.print(f"\nAppended to history: {history}")

    if report.severity == "significant":
        console.print(
            "\n[bold red]Significant drift detected.[/bold red] Investigate before "
            "the next training run."
        )
        raise typer.Exit(code=2)
    if report.severity == "moderate":
        console.print(
            "\n[bold yellow]Moderate drift.[/bold yellow] Keep an eye on the next "
            "few cycles."
        )
        raise typer.Exit(code=1)
    console.print("\n[bold green]No significant drift.[/bold green]")


if __name__ == "__main__":
    app()
