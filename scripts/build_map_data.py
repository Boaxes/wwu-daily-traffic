"""Build the map's dataset: every term's meetings, geocoded, with predictions.

Turns each raw Banner term file into main-campus meeting rows (the same logic
as ``scrape_schedule.py``) and attaches the model's prediction for each
section where one exists: walk-forward backtest predictions for past terms and
the forecast for the upcoming term. Run ``train_model.py`` and ``forecast.py``
first.

Usage:
    python scripts/build_map_data.py
"""

from __future__ import annotations

import gzip
import json
import sys

import pandas as pd

from common import FORECAST_TERM, PROCESSED_DIR, RAW_DIR, ROOT
from scrape_schedule import build_dataset, fetch_buildings

# Constants
OUTPUT = ROOT / "data" / "map_terms.parquet"
MAP_COLUMNS = [
    "term",
    "term_desc",
    "class",
    "crn",
    "title",
    "official_map_name",
    "room",
    "schedule_days",
    "start_time",
    "end_time",
    "enrolled_students",
    "latitude",
    "longitude",
]
PREDICTION_COLUMNS = ["term", "crn", "capacity", "predicted_students", "predicted_low", "predicted_high"]


def main() -> None:
    buildings = fetch_buildings()
    terms = []
    for path in sorted(RAW_DIR.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as file:
            terms.append(build_dataset(json.load(file), buildings)[MAP_COLUMNS])
    meetings = pd.concat(terms, ignore_index=True)

    predictions = pd.concat(
        [
            pd.read_parquet(PROCESSED_DIR / "backtest_predictions.parquet"),
            pd.read_parquet(PROCESSED_DIR / "forecast_predictions.parquet"),
        ],
        ignore_index=True,
    ).reindex(columns=PREDICTION_COLUMNS)
    mapped = meetings.merge(predictions, on=["term", "crn"], how="left")
    # Registration for the forecast term may open before the next update; its
    # partial enrollment should not be shown as actual.
    mapped["forecast_only"] = mapped["term"] == FORECAST_TERM
    mapped.to_parquet(OUTPUT, index=False)

    summary = mapped.groupby("term").agg(
        meetings=("crn", "size"),
        with_prediction=("predicted_students", lambda values: values.notna().mean()),
    )
    print(summary.to_string(float_format="{:.0%}".format), file=sys.stderr)
    print(f"Wrote {len(mapped):,} meetings across {len(summary)} terms to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
