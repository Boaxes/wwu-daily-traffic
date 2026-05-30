"""Paths, term settings, and small helpers shared by the pipeline scripts."""

from __future__ import annotations

from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "banner"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

# Terms. Banner codes are the year plus 10 (Winter), 20 (Spring), 30 (Summer), or 40 (Fall).
SEASONS = {"10": "Winter", "20": "Spring", "30": "Summer", "40": "Fall"}
FIRST_YEAR = 2016
TEST_YEAR = 2026
FORECAST_TERM = "202710"

# Banner meeting records flag each day with these keys.
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
ALL_DAYS = WEEKDAYS + ["saturday", "sunday"]

# Fill rates above this are capped so a few heavily overenrolled sections don't dominate.
MAX_FILL_RATE = 1.5


def to_minutes(time: str) -> int:
    """Minutes after midnight for "0930" or "09:30"."""
    digits = time.replace(":", "")
    return int(digits[:2]) * 60 + int(digits[2:])
