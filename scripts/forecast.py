"""Forecast enrollment for the upcoming term, section by section.

The forecast uses the same procedure the backtest in ``train_model.py``
validates: settings chosen on the most recent year, then a model
retrained on every earlier term. Each section's 80% interval comes from the
quantile model, trained on terms before the test year and conformally
calibrated on the test-year terms. Run ``train_model.py`` first.

Usage:
    python scripts/forecast.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from common import FORECAST_TERM, PROCESSED_DIR, TEST_YEAR
from train_model import (
    FEATURES,
    TARGET,
    TUNING_PATH,
    apply_offset,
    conformal_offset,
    fit_for_year,
    load_data,
    make_interval_model,
    predict_fill,
    predict_quantiles,
)

# Constants
PREDICTIONS_PATH = PROCESSED_DIR / "forecast_predictions.parquet"


def main() -> None:
    data = load_data()
    target = data[data["term"] == FORECAST_TERM].copy()
    history = data[data["term"] != FORECAST_TERM]

    model, params, n_trees = fit_for_year(history, int(FORECAST_TERM[:4]))
    point = predict_fill(model, target)

    interval = json.loads(TUNING_PATH.read_text())["interval"]
    interval_model = make_interval_model(
        n_estimators=interval["n_trees"], early_stopping_rounds=None, **interval["params"]
    )
    train = history[history["year"] < TEST_YEAR]
    calibration = history[history["year"] == TEST_YEAR]
    interval_model.fit(train[FEATURES], train[TARGET], verbose=False)
    offset = conformal_offset(predict_quantiles(interval_model, calibration), calibration[TARGET].to_numpy())
    quantiles = apply_offset(predict_quantiles(interval_model, target), offset)

    # The point forecast comes from a different model, so stretch the few intervals that miss it.
    low = np.minimum(quantiles[:, 0], point)
    high = np.maximum(quantiles[:, -1], point)
    stretched = int(((point < quantiles[:, 0]) | (point > quantiles[:, -1])).sum())

    capacity = target["capacity"].to_numpy()
    target["predicted_fill"] = point
    target["predicted_students"] = np.rint(point * capacity).astype(int)
    target["predicted_low"] = np.floor(low * capacity).astype(int)
    target["predicted_high"] = np.ceil(high * capacity).astype(int)
    columns = [
        "term", "crn", "capacity", "predicted_fill", "predicted_students", "predicted_low", "predicted_high"
    ]
    target[columns].to_parquet(PREDICTIONS_PATH, index=False)
    print(
        f"Forecast {len(target):,} sections for {FORECAST_TERM} with {params}, {n_trees} trees: "
        f"{int(target['predicted_students'].sum()):,} predicted enrollments; "
        f"conformal offset {offset:.3f}; {stretched} intervals stretched to contain the forecast",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
