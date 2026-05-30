"""Turn raw Banner term files into a section-level modeling table.

Reads ``data/raw/banner/*.json.gz`` and writes ``data/processed/sections.parquet``
with one row per section: schedule and capacity features from the term file,
course history features from earlier terms only, and the fill-rate target.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import gzip
import json
import sys

import numpy as np
import pandas as pd

from common import ALL_DAYS, MAX_FILL_RATE, PROCESSED_DIR, RAW_DIR, SEASONS, to_minutes

# Constants
OUTPUT = PROCESSED_DIR / "sections.parquet"

# Individualized sections whose caps are placeholders, not real seat limits.
EXCLUDED_SCHEDULE_TYPES = {
    "Independent Study",
    "Internship",
    "Research",
    "Thesis",
    "Senior Thesis,Project,Capstone",
    "Field Experience",
}
MIN_CAPACITY = 5
MAX_CAPACITY = 500

# Loading
def load_raw_sections() -> list[dict]:
    sections = []
    for path in sorted(RAW_DIR.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as file:
            sections.extend(json.load(file))
    return sections


def section_row(section: dict) -> dict:
    meetings = [meeting["meetingTime"] for meeting in section["meetingsFaculty"]]
    timed = [meeting for meeting in meetings if meeting["beginTime"]]
    days = {day for meeting in timed for day in ALL_DAYS if meeting[day]}
    starts = [to_minutes(meeting["beginTime"]) for meeting in timed]
    buildings = [meeting["building"] for meeting in meetings if meeting["building"]]
    sequence = section["sequenceNumber"] or ""
    number = section["courseNumber"] or ""

    return {
        "term": section["term"],
        "crn": section["courseReferenceNumber"],
        "subject": section["subject"],
        "course_number": number,
        "course": f"{section['subject']} {number}",
        "sequence": sequence,
        "title": section["courseTitle"],
        "schedule_type": section["scheduleTypeDescription"],
        "instructional_method": section["instructionalMethod"],
        "campus": section["campusDescription"],
        "part_of_term": section["partOfTerm"],
        "credits": section["creditHourLow"] or section["creditHourHigh"],
        "capacity": section["maximumEnrollment"],
        "is_crosslisted": section["crossList"] is not None,
        "is_linked": bool(section["isSectionLinked"]),
        "course_level": int(number[0]) if number[:1].isdigit() else np.nan,
        "has_number_suffix": number[-1:].isalpha(),
        "sequence_prefix": sequence[:1] if sequence[:1].isalpha() else "",
        "n_meetings": len(meetings),
        "days_per_week": len(days),
        "meets_friday": "friday" in days,
        "meets_weekend": bool(days & {"saturday", "sunday"}),
        "start_minutes": min(starts) if starts else np.nan,
        "weekly_hours": sum(meeting["hoursWeek"] or 0 for meeting in timed),
        "building": buildings[0] if buildings else "NONE",
        # COVID-era remote sections kept method FTF and used a REMOTE building code.
        "is_online": section["instructionalMethod"] in {"DIOL", "DICO"}
        or bool({"ONLINE", "REMOTE"} & set(buildings)),
        "enrollment": section["enrollment"],
    }

# History features
def add_history_features(data: pd.DataFrame) -> pd.DataFrame:
    """Course and subject history, using only terms before each row's term."""
    terms = sorted(data["term"].unique())
    term_index = {term: index for index, term in enumerate(terms)}

    courses = (
        data.groupby(["course", "term"], as_index=False)
        .agg(
            course_enrollment=("enrollment", "sum"),
            course_capacity=("capacity", "sum"),
            course_sections=("crn", "nunique"),
            season=("season", "first"),
        )
        .sort_values(["course", "term"])
    )
    courses["course_fill"] = courses["course_enrollment"] / courses["course_capacity"]
    courses["term_index"] = courses["term"].map(term_index)

    by_course = courses.groupby("course")
    courses["prev_fill"] = by_course["course_fill"].shift()
    courses["prev_enrollment"] = by_course["course_enrollment"].shift()
    courses["terms_since_offered"] = courses["term_index"] - by_course["term_index"].shift()
    courses["times_offered_before"] = by_course.cumcount()
    courses["hist_mean_fill"] = by_course["course_fill"].transform(
        lambda fill: fill.shift().expanding().mean()
    )

    by_season = courses.groupby(["course", "season"])
    courses["prev_season_fill"] = by_season["course_fill"].shift()
    courses["prev_season_enrollment"] = by_season["course_enrollment"].shift()

    # Last same-season demand against this term's total seats for the course.
    courses["demand_ratio"] = courses["prev_season_enrollment"] / courses["course_capacity"]

    subjects = (
        data.groupby(["subject", "term"], as_index=False)
        .agg(subject_enrollment=("enrollment", "sum"), subject_capacity=("capacity", "sum"))
        .sort_values(["subject", "term"])
    )
    subjects["subject_prev_fill"] = (
        (subjects["subject_enrollment"] / subjects["subject_capacity"])
        .groupby(subjects["subject"])
        .shift()
    )

    history_columns = [
        "course",
        "term",
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
    ]
    return data.merge(courses[history_columns], on=["course", "term"]).merge(
        subjects[["subject", "term", "subject_prev_fill"]], on=["subject", "term"]
    )

# Build
def build() -> pd.DataFrame:
    data = pd.DataFrame([section_row(section) for section in load_raw_sections()])
    data["season"] = data["term"].str[-2:].map(SEASONS)
    data["year"] = data["term"].str[:4].astype(int)

    modeled = data[
        ~data["schedule_type"].isin(EXCLUDED_SCHEDULE_TYPES)
        & data["capacity"].between(MIN_CAPACITY, MAX_CAPACITY)
    ].copy()
    modeled = add_history_features(modeled)
    modeled["fill_rate"] = (modeled["enrollment"] / modeled["capacity"]).clip(upper=MAX_FILL_RATE)
    return modeled.sort_values(["term", "crn"]).reset_index(drop=True)


def main() -> None:
    data = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT, index=False)
    print(
        f"Wrote {len(data):,} sections across {data['term'].nunique()} terms to {OUTPUT}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
