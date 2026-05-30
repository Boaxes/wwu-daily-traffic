"""Explain the saved 2026 model on the 2026 test terms.

Computes SHAP values with XGBoost's built-in TreeSHAP, saves a SHAP summary
plot, and writes mean absolute SHAP per feature plus a calibration table of
predicted vs actual fill rate to ``reports/explain.json``.
Run ``train_model.py`` first.

Usage:
    python scripts/explain_model.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from common import REPORT_DIR, TEST_YEAR
from train_model import CATEGORICAL_FEATURES, FEATURES, MODEL_PATH, TARGET, load_data, predict_fill

# Constants
OUTPUT = REPORT_DIR / "explain.json"
FIGURE_PATH = REPORT_DIR / "figures" / "shap_summary.png"
CALIBRATION_BINS = 10
FEATURE_LABELS = {
    "hist_mean_fill": "Average past fill rate",
    "demand_ratio": "Demand ratio",
    "prev_fill": "Fill rate last time offered",
    "prev_season_fill": "Fill rate, last time in same quarter",
    "prev_enrollment": "Enrollment last time offered",
    "prev_season_enrollment": "Enrollment, last time in same quarter",
    "subject_prev_fill": "Department fill rate last quarter",
    "times_offered_before": "Times offered before",
    "terms_since_offered": "Quarters since last offered",
    "subject": "Department",
    "building": "Building",
    "schedule_type": "Type of class",
    "instructional_method": "Delivery method",
    "course_level": "Course level",
    "start_minutes": "Start time",
    "capacity": "Section capacity",
    "course_capacity": "Total seats for the course",
    "course_sections": "Sections of the course",
    "credits": "Credits",
    "weekly_hours": "Hours per week",
    "days_per_week": "Days per week",
    "year": "Year",
    "season": "Quarter",
}


def shap_values(model: xgb.XGBRegressor, frame: pd.DataFrame) -> np.ndarray:
    matrix = xgb.DMatrix(frame[FEATURES], enable_categorical=True)
    return model.get_booster().predict(matrix, pred_contribs=True)[:, :-1]


def plot_shap_summary(contributions: np.ndarray, frame: pd.DataFrame) -> None:
    display = frame[FEATURES].copy()
    # Category codes and single-valued columns (the test year) have no high or low value;
    # blanking them makes SHAP draw those rows in gray.
    for column in FEATURES:
        if column in CATEGORICAL_FEATURES or frame[column].nunique() == 1:
            display[column] = np.nan

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
    shap.summary_plot(
        contributions * 100,
        display,
        feature_names=[FEATURE_LABELS.get(name, name) for name in FEATURES],
        max_display=15,
        color_bar_label="Feature value (blue = low, red = high)",
        show=False,
    )
    plt.xlabel("Change in predicted fill rate (percentage points)")
    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=200)
    plt.close()


def calibration(frame: pd.DataFrame, predicted: np.ndarray) -> list[dict]:
    """Mean and interquartile range of actual fill rate within predicted-fill deciles."""
    table = pd.DataFrame(
        {
            "predicted": predicted,
            "actual": frame[TARGET].to_numpy(),
            "bin": pd.qcut(predicted, CALIBRATION_BINS, labels=False, duplicates="drop"),
        }
    )
    grouped = table.groupby("bin").agg(
        predicted=("predicted", "mean"),
        actual=("actual", "mean"),
        actual_low=("actual", lambda values: values.quantile(0.25)),
        actual_high=("actual", lambda values: values.quantile(0.75)),
        sections=("actual", "size"),
    )
    return grouped.round(4).to_dict("records")


def main() -> None:
    data = load_data()
    test = data[data["year"] == TEST_YEAR]
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    predicted = predict_fill(model, test)
    contributions = shap_values(model, test)
    mean_abs = pd.Series(np.abs(contributions).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    plot_shap_summary(contributions, test)

    direction = {}
    for index, name in enumerate(FEATURES):
        if name in CATEGORICAL_FEATURES or test[name].nunique() == 1:
            continue
        # Rank correlation: +1 means higher values always raise the prediction, -1 always lower it.
        effects = pd.Series(contributions[:, index], index=test.index)
        direction[name] = round(float(test[name].corr(effects, method="spearman")), 3)
    explain = {
        "test_mae": mean_absolute_error(test[TARGET], predicted),
        "shap_mean_abs": mean_abs.round(5).to_dict(),
        "shap_direction": direction,
        "feature_labels": FEATURE_LABELS,
        "calibration": calibration(test, predicted),
    }
    OUTPUT.write_text(json.dumps(explain, indent=2))
    print(
        f"Wrote {OUTPUT} (test MAE {explain['test_mae']:.4f}; "
        f"top features: {', '.join(mean_abs.index[:5])})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
