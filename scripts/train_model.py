"""Train and evaluate the XGBoost fill-rate model, year by year.

For every test year from 2019 to 2026, the model's settings are chosen on the
year before it, the model is retrained on all earlier terms, and it is scored
on the test year against a simple baseline. No test year influences its own
model. The 2026 model is saved for ``explain_model.py``, and every year's
predictions feed the map's forecast layer.

A second, quantile model gives each section an 80% prediction interval, which
is conformally calibrated on a term it never trained or stopped on.

Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from common import FORECAST_TERM, MAX_FILL_RATE, MODEL_DIR, PROCESSED_DIR, REPORT_DIR, TEST_YEAR

# Constants
DATA_PATH = PROCESSED_DIR / "sections.parquet"
MODEL_PATH = MODEL_DIR / "enrollment_xgb.json"
TUNING_PATH = MODEL_DIR / "tuning.json"
PREDICTIONS_PATH = PROCESSED_DIR / "backtest_predictions.parquet"
FIRST_TEST_YEAR = 2019
RANDOM_STATE = 42
QUANTILES = [0.1, 0.5, 0.9]
INTERVAL_COVERAGE = QUANTILES[-1] - QUANTILES[0]

CATEGORICAL_FEATURES = [
    "subject",
    "schedule_type",
    "instructional_method",
    "campus",
    "part_of_term",
    "building",
    "season",
    "sequence_prefix",
]
NUMERIC_FEATURES = [
    "year",
    "course_level",
    "credits",
    "capacity",
    "n_meetings",
    "days_per_week",
    "start_minutes",
    "weekly_hours",
    "course_capacity",
    "course_sections",
    "prev_fill",
    "prev_enrollment",
    "terms_since_offered",
    "times_offered_before",
    "hist_mean_fill",
    "prev_season_fill",
    "prev_season_enrollment",
    "demand_ratio",
    "subject_prev_fill",
]
BOOLEAN_FEATURES = [
    "is_crosslisted",
    "is_linked",
    "has_number_suffix",
    "meets_friday",
    "meets_weekend",
    "is_online",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES
TARGET = "fill_rate"
# Recorded at the end of the term, so they may carry some information from registration.
CAPACITY_FEATURES = {"capacity", "course_capacity", "course_sections", "demand_ratio"}

PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 10],
    "subsample": [0.8],
    "colsample_bytree": [0.6, 0.9],
}

# Data
def load_data() -> pd.DataFrame:
    data = pd.read_parquet(DATA_PATH)
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna("NONE").astype("category")
    for column in BOOLEAN_FEATURES:
        data[column] = data[column].astype(int)
    return data

# Models
def persistence_baseline(frame: pd.DataFrame, fallback: float) -> np.ndarray:
    """Last same-season fill for the course, then last fill, then subject, then mean."""
    return (
        frame["prev_season_fill"]
        .fillna(frame["prev_fill"])
        .fillna(frame["subject_prev_fill"])
        .fillna(fallback)
        .clip(upper=MAX_FILL_RATE)
        .to_numpy()
    )


def make_model(**params) -> xgb.XGBRegressor:
    settings = {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "tree_method": "hist",
        "enable_categorical": True,
        "max_cat_to_onehot": 1,
        "early_stopping_rounds": 100,
        "eval_metric": "mae",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return xgb.XGBRegressor(**(settings | params))


def make_interval_model(**params) -> xgb.XGBRegressor:
    return make_model(
        objective="reg:quantileerror",
        quantile_alpha=np.array(QUANTILES),
        eval_metric="quantile",
        **params,
    )


def predict_fill(model: xgb.XGBRegressor, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(frame[FEATURES]).clip(0, MAX_FILL_RATE)


def predict_quantiles(model: xgb.XGBRegressor, frame: pd.DataFrame) -> np.ndarray:
    # Sorting each row fixes the occasional crossed quantile.
    return np.sort(model.predict(frame[FEATURES]), axis=1).clip(0, MAX_FILL_RATE)


def conformal_offset(quantiles: np.ndarray, actual: np.ndarray) -> float:
    """Conformalized quantile regression (Romano et al., 2019) widening term."""
    scores = np.maximum(quantiles[:, 0] - actual, actual - quantiles[:, -1])
    level = min(1.0, INTERVAL_COVERAGE * (1 + 1 / len(scores)))
    return float(np.quantile(scores, level, method="higher"))


def apply_offset(quantiles: np.ndarray, offset: float) -> np.ndarray:
    adjusted = quantiles.copy()
    adjusted[:, 0] = (adjusted[:, 0] - offset).clip(0, MAX_FILL_RATE)
    adjusted[:, -1] = (adjusted[:, -1] + offset).clip(0, MAX_FILL_RATE)
    return adjusted


def tune(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[dict, int]:
    """Grid search with early stopping; returns the best settings and tree count."""
    trials = []
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID, values))
        model = make_model(**params)
        model.fit(
            train[FEATURES],
            train[TARGET],
            eval_set=[(valid[FEATURES], valid[TARGET])],
            verbose=False,
        )
        mae = mean_absolute_error(valid[TARGET], predict_fill(model, valid))
        trials.append({"params": params, "n_trees": model.best_iteration + 1, "valid_mae": mae})
    best = min(trials, key=lambda trial: trial["valid_mae"])
    return best["params"], best["n_trees"]


def fit(train: pd.DataFrame, params: dict, n_trees: int, features: list[str] = FEATURES) -> xgb.XGBRegressor:
    model = make_model(n_estimators=n_trees, early_stopping_rounds=None, **params)
    model.fit(train[features], train[TARGET], verbose=False)
    return model


def fit_for_year(data: pd.DataFrame, year: int) -> tuple[xgb.XGBRegressor, dict, int]:
    """Choose settings on the year before `year`, then retrain on every earlier term."""
    params, n_trees = tune(data[data["year"] < year - 1], data[data["year"] == year - 1])
    return fit(data[data["year"] < year], params, n_trees), params, n_trees

# Evaluation
def score(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    actual = frame[TARGET].to_numpy()
    return {
        "mae_fill_rate": mean_absolute_error(actual, predicted),
        "rmse_fill_rate": root_mean_squared_error(actual, predicted),
        "r2": r2_score(actual, predicted),
        "mae_seats": mean_absolute_error(frame["enrollment"], predicted * frame["capacity"]),
        "n": len(frame),
    }


def score_segments(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    has_history = frame["prev_fill"].notna().to_numpy()
    return {
        "all": score(frame, predicted),
        "returning_courses": score(frame[has_history], predicted[has_history]),
        "new_courses": score(frame[~has_history], predicted[~has_history]),
    }


def interval_scores(frame: pd.DataFrame, quantiles: np.ndarray) -> dict:
    actual = frame[TARGET].to_numpy()
    inside = (quantiles[:, 0] <= actual) & (actual <= quantiles[:, -1])
    width = quantiles[:, -1] - quantiles[:, 0]
    return {
        "coverage": float(inside.mean()),
        "mean_width_fill_rate": float(width.mean()),
        "mean_width_seats": float((width * frame["capacity"].to_numpy()).mean()),
        "n": len(frame),
    }


def evaluate_intervals(data: pd.DataFrame, params: dict) -> dict:
    """Train on years before the last one, stop on its Winter and Spring, calibrate on its Fall."""
    last = TEST_YEAR - 1
    train = data[data["year"] < last]
    stop = data[data["term"].isin([f"{last}10", f"{last}20"])]
    calibration = data[data["term"] == f"{last}40"]
    test = data[data["year"] == TEST_YEAR]

    model = make_interval_model(**params)
    model.fit(train[FEATURES], train[TARGET], eval_set=[(stop[FEATURES], stop[TARGET])], verbose=False)
    offset = conformal_offset(predict_quantiles(model, calibration), calibration[TARGET].to_numpy())
    raw = predict_quantiles(model, test)
    return {
        "quantiles": QUANTILES,
        "target_coverage": INTERVAL_COVERAGE,
        "params": params,
        "n_trees": model.best_iteration + 1,
        "calibration_term": f"{last}40",
        "conformal_offset": offset,
        "raw": interval_scores(test, raw),
        "conformal": interval_scores(test, apply_offset(raw, offset)),
    }


# Main
def main() -> None:
    data = load_data()
    data = data[data["term"] != FORECAST_TERM]

    years, predictions, tuning = [], [], {}
    for year in range(FIRST_TEST_YEAR, TEST_YEAR + 1):
        model, params, n_trees = fit_for_year(data, year)
        test = data[data["year"] == year]
        train_mean = float(data.loc[data["year"] < year, TARGET].mean())
        predicted = predict_fill(model, test)
        baseline = persistence_baseline(test, train_mean)
        tuning[year] = {"params": params, "n_trees": n_trees}
        predictions.append(
            test[["term", "crn", "capacity", "enrollment", TARGET]].assign(
                predicted_fill=predicted,
                baseline_fill=baseline,
                predicted_students=np.rint(predicted * test["capacity"]).astype(int),
            )
        )

        xgb_mae = mean_absolute_error(test[TARGET], predicted)
        baseline_mae = mean_absolute_error(test[TARGET], baseline)
        years.append(
            {
                "year": year,
                "train_sections": int((data["year"] < year).sum()),
                "test_sections": len(test),
                "xgboost_mae": xgb_mae,
                "baseline_mae": baseline_mae,
                "xgboost_r2": r2_score(test[TARGET], predicted),
                "baseline_r2": r2_score(test[TARGET], baseline),
                "improvement": 1 - xgb_mae / baseline_mae,
                "params": params,
                "n_trees": n_trees,
            }
        )
        print(
            f"  {year}: XGBoost MAE {xgb_mae:.4f} vs baseline {baseline_mae:.4f} "
            f"({years[-1]['improvement']:+.1%}), {params}, {n_trees} trees",
            file=sys.stderr,
        )

    # The loop ends on TEST_YEAR, so model, test, and the predictions below belong to it.
    results = {
        "global_mean": score_segments(test, np.full(len(test), train_mean)),
        "last_fill_persistence": score_segments(test, baseline),
        "xgboost": score_segments(test, predicted),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    # How much the end-of-term capacity features help: retrain without them.
    without_capacity = [name for name in FEATURES if name not in CAPACITY_FEATURES]
    capacity_check = fit(data[data["year"] < TEST_YEAR], params, n_trees, without_capacity)
    mae_without_capacity = mean_absolute_error(
        test[TARGET], capacity_check.predict(test[without_capacity]).clip(0, MAX_FILL_RATE)
    )

    print("Evaluating prediction intervals...", file=sys.stderr)
    intervals = evaluate_intervals(data, tuning[TEST_YEAR - 1]["params"])

    TUNING_PATH.write_text(
        json.dumps(
            {
                "by_test_year": {str(year): settings for year, settings in tuning.items()},
                "interval": {"params": intervals["params"], "n_trees": intervals["n_trees"]},
            },
            indent=2,
        )
    )
    pd.concat(predictions, ignore_index=True).to_parquet(PREDICTIONS_PATH, index=False)
    report = {
        "test_year": TEST_YEAR,
        "train_terms": sorted(data.loc[data["year"] < TEST_YEAR, "term"].unique()),
        "validation_terms": sorted(data.loc[data["year"] == TEST_YEAR - 1, "term"].unique()),
        "test_terms": sorted(data.loc[data["year"] == TEST_YEAR, "term"].unique()),
        "params": tuning[TEST_YEAR]["params"],
        "n_trees": tuning[TEST_YEAR]["n_trees"],
        "n_features": len(FEATURES),
        "test_results": results,
        "mae_without_capacity_features": mae_without_capacity,
        "intervals": intervals,
        "backtest": {
            "years": years,
            "mean_improvement": float(np.mean([row["improvement"] for row in years])),
            "years_beating_baseline": sum(row["improvement"] > 0 for row in years),
        },
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    overall = results["xgboost"]["all"]
    print(
        f"\n{TEST_YEAR}: MAE {overall['mae_fill_rate']:.4f}, R2 {overall['r2']:.3f}, "
        f"seats MAE {overall['mae_seats']:.2f}; beat baseline in "
        f"{report['backtest']['years_beating_baseline']}/{len(years)} years\n"
        f"80% intervals: raw coverage {intervals['raw']['coverage']:.1%}, "
        f"conformal {intervals['conformal']['coverage']:.1%}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
